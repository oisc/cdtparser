# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 训练 SPINN 篇章结构生成模型
"""
import dataset
import config
import gensim
import torch
import torch.nn as nn
import shutil
import numpy as np
import logging
from structure.tree import Discourse, EDU, RelationNode
from transition import Session
from transition.shiftreduce import SRConfiguration, SRTransition
from util.metrics import CDTBMetrics
from .model import SPINN
from .annotator import SPINNTreeBuilder


config_section = "treebuilder.spinn_bow"


def sr_oracle(discorse: Discourse):
    for node in discorse.binarize().traverse():
        if isinstance(node, EDU):
            yield SRTransition.SHIFT, None
        elif isinstance(node, RelationNode):
            yield SRTransition.REDUCE, node.nuclear


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

    # build label
    labels = set()
    for discourse in discourses:
        for action in sr_oracle(discourse):
            labels.add(action)
    pos_embedding_size = config.get(config_section, "pos_embedding_size", rtype=int)

    hidden_size = config.get(config_section, "hidden_size", rtype=int)
    proj_dropout = config.get(config_section, "proj_dropout", rtype=float)
    mlp_layers = config.get(config_section, "mlp_layers", rtype=int)
    mlp_dropout = config.get(config_section, "mlp_dropout", rtype=float)
    model = SPINN(hidden_size=hidden_size,
                  proj_dropout=proj_dropout,
                  mlp_layers=mlp_layers, mlp_dropout=mlp_dropout,
                  pos_vocab=pos_vocab, pos_embedding_size=pos_embedding_size,
                  word_vocab=word_vocab, word_embedding=word_embedding,
                  labels=labels)
    return model


def evaluate(model, discourses):
    treebuilder = SPINNTreeBuilder(model)
    parses = []
    for discourse in discourses:
        parse = treebuilder.annotate(discourse.strip())
        parses.append(parse)
    metrics = CDTBMetrics(discourses, parses)
    print(metrics.parser_report())
    return metrics.span_score.f1() + metrics.nuclear_score.f1()


def train(cdtb):
    model = build_model(cdtb.train)
    transition = SRTransition()

    num_epoch = config.get(config_section, "num_epoch", rtype=int)
    batch_size = config.get(config_section, "batch_size", rtype=int)
    eval_every = config.get(config_section, "eval_every", rtype=int)
    lr = config.get(config_section, "lr", rtype=float)
    l2 = config.get(config_section, "l2_penalty", rtype=float)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6, 12], gamma=0.5)
    model.train()
    optimizer.zero_grad()

    step = 0
    batch = 0
    batch_loss = 0.
    best_model_score = 0.
    model_dir = config.get(config_section, "model_dir")
    for epoch in range(num_epoch):
        epoch += 1
        scheduler.step()
        print("learning rate: %f" % scheduler.get_lr()[0])
        for discourse in np.random.permutation(cdtb.train):
            step += 1
            conf = SRConfiguration(discourse.strip())
            state = model.new_state(conf)
            session = Session(conf, transition, state=state)
            scores = []
            grounds = []
            for label in sr_oracle(discourse):
                action, nuclear = label
                scores.append(model(session.state))
                grounds.append(model.label2idx[label])
                if action == SRTransition.SHIFT:
                    session(action)
                    session.state = model.shift(session.state)
                else:
                    session(action, nuclear=nuclear)
                    session.state = model.reduce(session.state, nuclear)
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
