# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/6/25
@Description: 
"""
import gensim
import numpy as np
import dataset
import config
from structure.tree import EDU, RelationNode
from .model import DistanceModel
import torch
import torch.optim as optim
import torch.nn.functional as nnfunc


config_section = "treebuilder.distance"
UNK = "-UNK-"
PAD = "-PAD-"
START = "-START-"
END = "-END-"


def bitree2distance(root):
    if isinstance(root, EDU):
        return [], [], [], 0  # (distance, nuclear, relation, height)
    else:
        assert isinstance(root, RelationNode) and len(root) == 2
        ldists, lnucs, lrels, lheight = bitree2distance(root[0])
        rdists, rnucs, rrels, rheight = bitree2distance(root[1])
        height = max(lheight, rheight) + 1
        dists = ldists + [height] + rdists
        nucs = lnucs + [root.nuclear] + rnucs
        rels = lrels + [root.relation] + rrels
        return dists, nucs, rels, height


def load_pretrained_wordemb(word_emb_file):
    wv = gensim.models.KeyedVectors.load_word2vec_format(word_emb_file, binary=True)
    vocab = {PAD: 0, UNK: 1, START: 2, END: 3}
    for word in wv.vocab:
        vocab[word] = len(vocab)
    weights = np.random.normal(wv.vectors.mean(), wv.vectors.var(), size=(len(vocab), wv.vector_size))
    return vocab, weights


def build_model():
    dropout_p = config.get(config_section, "dropout", rtype=float)
    hidden_size = config.get(config_section, "hidden_size", rtype=int)
    vocab, weights = load_pretrained_wordemb(config.get(config_section, "word_embedding"))
    model = DistanceModel(vocab=vocab, pretrained=weights,
                          dropout_p=dropout_p, hidden_size=hidden_size)
    return model


def rankloss(gold, pred):
    gold_diff = gold[:, None] - gold[None, :]
    pred_diff = pred[:, None] - pred[None, :]
    loss = nnfunc.relu(1 - gold_diff.sign() * pred_diff)
    loss = loss.sum() / (2 * gold.size(0) * gold.size(0))
    return loss


def train(cdtb):
    distances = []
    for discourse in cdtb.train:
        ...

    model = build_model()
    optimizer = optim.Adam(model.parameters())
    for discourse in cdtb.train:
        pred = model(discourse)
        distance, _, _, _ = bitree2distance(discourse.binarize().tree())
        gold = torch.FloatTensor(distance)
        loss = rankloss(gold, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(gold)
        print(pred)
        print(loss)
        print()


def main():
    cdtb = dataset.load_cdtb_by_config()
    train(cdtb)


if __name__ == '__main__':
    main()
