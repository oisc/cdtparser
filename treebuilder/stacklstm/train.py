# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/7
@Description: 训练篇章上的 Transition-based dependency parsing with stack long short-term memory
Ref: Dyer C et al. Transition-based dependency parsing with stack long short-term memory. 2015.
"""
import config
import logging
import gensim
import dataset
import numpy as np
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from structure.tree import Discourse, EDU, RelationNode, Relation
from transition.shiftreduce import SRTransition, SRConfiguration
from transition import Session
from treebuilder.stacklstm.annotator import StackLSTMTreeBuilder
from util.metrics import CDTBMetrics
from .model import StackLSTMTransModel


config_section = "treebuilder.stacklstm"


def sr_oracle(discorse: Discourse):
    for node in discorse.binarize().traverse():
        if isinstance(node, EDU):
            yield SRTransition.SHIFT, None, None
        elif isinstance(node, RelationNode):
            yield SRTransition.REDUCE, node.nuclear, node.relation.fine


def load_pretrained_w2v(embedding_file):
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    word_vocab = list(w2v_model.vocab)
    word_embedding = np.zeros((len(word_vocab), w2v_model.vector_size), dtype=np.float64)
    for i, word in enumerate(word_vocab):
        word_embedding[i] = w2v_model[word]
    return word_vocab, word_embedding


def build_model(discourses):
    # reproducible
    seed = config.get(config_section, "seed", rtype=int, defult=21)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # word embedding
    word_embedding_file = config.get(config_section, "word_embedding")
    word_vocab, word_embedding = load_pretrained_w2v(word_embedding_file)

    # build part of speech vocab
    pos_vocab = set()
    for discourse in discourses:
        for sentence in discourse.sentences:
            for pos, word in sentence.tags():
                pos_vocab.add(pos)

    labels = set()
    action_vocab = set()
    relation_vocab = set()
    for discourse in discourses:
        for label in sr_oracle(discourse):
            action, nuclear, relation = label
            labels.add(label)
            action_vocab.add("%s_%s" % (action, nuclear))
            relation_vocab.add(relation or "NOREL")

    hidden_size = config.get(config_section, "hidden_size", rtype=int)
    lstm_layers = config.get(config_section, "lstm_layers", rtype=int)
    lstm_dropout = config.get(config_section, "lstm_dropout", rtype=float)
    pos_embedding_size = config.get(config_section, "pos_embedding_size", rtype=int)
    relation_embedding_size = config.get(config_section, "relation_embedding_size", rtype=int)
    action_embedding_size = config.get(config_section, "action_embedding_size", rtype=int)
    sentence_position_cutoff = config.get(config_section, "sentence_position_cutoff", rtype=int)
    sentence_position_embedding_size = config.get(config_section, "sentence_position_embedding_size", rtype=int)
    edu_position_cutoff = config.get(config_section, "edu_position_cutoff", rtype=int)
    edu_position_embedding_size = config.get(config_section, "edu_position_embedding_size", rtype=int)
    edu_length_cutoff = config.get(config_section, "edu_length_cutoff", rtype=int)
    edu_length_embedding_size = config.get(config_section, "edu_length_embedding_size", rtype=int)

    model = StackLSTMTransModel(hidden_size=hidden_size, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout,
                                word_vocab=word_vocab, word_embedding=word_embedding,
                                pos_vocab=pos_vocab, pos_embedding_size=pos_embedding_size,
                                action_vocab=action_vocab, action_embedding_size=action_embedding_size,
                                relation_vocab=relation_vocab, relation_embedding_size=relation_embedding_size,
                                sentence_position_cutoff=sentence_position_cutoff,
                                sentence_position_embedding_size=sentence_position_embedding_size,
                                edu_position_cutoff=edu_position_cutoff,
                                edu_position_embedding_size=edu_position_embedding_size,
                                edu_length_cutoff=edu_length_cutoff,
                                edu_length_embedding_size=edu_length_embedding_size,
                                labels=labels)
    return model


def new_session(model, discourse):
    transition = SRTransition()
    conf = SRConfiguration(discourse.strip())
    state = model.new_state(conf)
    session = Session(conf, transition, state=state)
    return session


def evaluate(model, discourses):
    treebuilder = StackLSTMTreeBuilder(model)
    parses = []
    for discourse in discourses:
        parse = treebuilder.annotate(discourse.strip())
        parses.append(parse)
    metrics = CDTBMetrics(discourses, parses)
    print(metrics.parser_report())
    return metrics.nuclear_fine_score.f1()


def train(cdtb):
    model = build_model(cdtb.train)
    model.train()
    print(model)

    num_epoch = config.get(config_section, "num_epoch", rtype=int)
    batch_size = config.get(config_section, "batch_size", rtype=int)
    eval_every = config.get(config_section, "eval_every", rtype=int)
    lr = config.get(config_section, "lr", rtype=float)
    l2 = config.get(config_section, "l2_penalty", rtype=float)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = MultiStepLR(optimizer, milestones=[3, 6, 12], gamma=0.5)
    optimizer.zero_grad()
    step = 0
    batch = 0
    batch_loss = 0.
    best_model_score = 0.
    model_dir = config.get(config_section, "model_dir")
    for epoch in range(num_epoch):
        epoch += 1
        scheduler.step()
        print("leraning rate: %s" % scheduler.get_lr()[0])
        for discourse in np.random.permutation(cdtb.train):
            step += 1
            session = new_session(model, discourse.strip())
            scores = []
            grounds = []
            for label in sr_oracle(discourse):
                action, nuclear, fine = label
                scores.append(model(session.state))
                grounds.append(model.label2idx[label])
                if action == SRTransition.SHIFT:
                    session(action)
                else:
                    session(action, nuclear=nuclear, relation=Relation(fine=fine))
                session.state = model.update(session.state, action, nuclear, fine)
            loss = criterion(torch.stack(scores), torch.Tensor(grounds).long())
            loss.backward()
            batch_loss += loss.item()

            if step % batch_size == 0:
                batch += 1
                optimizer.step()
                optimizer.zero_grad()
                print("step %d, epoch: %d, batch: %d, batch loss: %.3f" % (step, epoch, batch, batch_loss / batch_size))
                batch_loss = 0.
                if batch % eval_every == 0:
                    model_score = evaluate(model, cdtb.test)
                    if model_score > best_model_score:
                        best_model_score = model_score
                        with open("%s.%.3f" % (model_dir, model_score), "wb+") as best_model_fd:
                            print("save new best model to %s.%.3f" % (model_dir, model_score))
                            torch.save(model, best_model_fd)
                    model.train()
    # copy best model to model dir
    shutil.copy2("%s.%.3f" % (model_dir, best_model_score), model_dir)


def main():
    cdtb = dataset.load_cdtb_by_config()
    train(cdtb)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
