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
    seed = config.get("treebuilder.spinn", "seed", rtype=int, defult=21)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # word embedding
    word_embedding_file = config.get("treebuilder.spinn", "word_embedding")
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
    pos_embedding_size = config.get("treebuilder.spinn", "pos_embedding_size", rtype=int)

    hidden_size = config.get("treebuilder.spinn", "hidden_size", rtype=int)
    proj_dropout = config.get("treebuilder.spinn", "proj_dropout", rtype=float)
    mlp_layers = config.get("treebuilder.spinn", "mlp_layers", rtype=int)
    mlp_dropout = config.get("treebuilder.spinn", "mlp_dropout", rtype=float)
    model = SPINN(hidden_size=hidden_size, proj_dropout=proj_dropout, mlp_layers=mlp_layers, mlp_dropout=mlp_dropout,
                  pos_vocab=pos_vocab, pos_embedding_size=pos_embedding_size,
                  word_vocab=word_vocab, word_embedding=word_embedding,
                  labels=labels)
    return model


def new_session(model, discourse):
    transition = SRTransition()
    conf = SRConfiguration(discourse.strip())
    state = model.new_state(conf)
    session = Session(conf, transition, history=True, state=state)
    return session


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
    model.train()
    num_epoch = config.get("treebuilder.spinn", "num_epoch", rtype=int)
    batch_size = config.get("treebuilder.spinn", "batch_size", rtype=int)
    eval_every = config.get("treebuilder.spinn", "eval_every", rtype=int)
    lr = config.get("treebuilder.spinn", "lr", rtype=float)
    l2 = config.get("treebuilder.spinn", "l2_penalty", rtype=float)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2)
    optim.zero_grad()
    step = 0
    batch = 0
    batch_loss = 0.

    best_model_score = 0.
    model_dir = config.get("treebuilder.spinn", "model_dir")
    for epoch in range(num_epoch):
        epoch += 1
        for discourse in np.random.permutation(cdtb.train):
            step += 1
            session = new_session(model, discourse.strip())
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
                    session.state = model.reduce(session.state)
            loss = criterion(torch.stack(scores), torch.Tensor(grounds).long())
            loss.backward()
            batch_loss += loss.item()

            if step % batch_size == 0:
                batch += 1
                optim.step()
                optim.zero_grad()
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
