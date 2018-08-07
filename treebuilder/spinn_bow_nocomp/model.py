# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: SPINN shift reduce model
Ref: Bowman S R, Gauthier J, Rastogi A, et al. A fast unified model for parsing and sentence understanding[J]. 2016.
"""
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
import numpy as np
from structure.tree import Discourse


UNK = "<UNK>"
PAD = "<PAD>"


class SPINNState:
    def __init__(self, stack, buffer, tracking):
        self.stack = stack
        self.buffer = buffer
        self.tracking = tracking

    def __copy__(self):
        stack = [(hs.clone(), cs.clone()) for hs, cs in self.stack]
        buffer = deque([(hb.clone(), cb.clone()) for hb, cb in self.buffer])
        h, c = self.tracking
        tracking = h.clone(), c.clone()
        return SPINNState(stack, buffer, tracking)


class Tracker(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(3 * self.hidden_size, hidden_size)

    def init_state(self):
        return tuple(torch.zeros(self.hidden_size).view(1, -1) for _ in range(2))

    def forward(self, state):
        stack, buffer, tracking = state.stack, state.buffer, state.tracking
        (s2h, _), (s1h, _) = stack[-2], stack[-1]
        b1h, _ = buffer[0]
        cell_input = torch.cat([s2h, s1h, b1h]).view(1, -1)
        tracking = self.rnn(cell_input, tracking)
        return tracking


class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_p, num_classes):
        nn.Module.__init__(self)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_p) for _ in range(num_layers - 1)])
        self.logits = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden):
        for linear, dropout, activation in zip(self.linears, self.dropouts, self.activations):
            hidden = linear(hidden)
            hidden = activation(hidden)
            hidden = dropout(hidden)
        return self.logits(hidden)


class SPINN(nn.Module):
    def __init__(self, hidden_size, proj_dropout, mlp_layers, mlp_dropout,
                 pos_vocab, pos_embedding_size, word_vocab, word_embedding, labels):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        # build word vocab and initialize embedding
        word2idx = {UNK: 0, PAD: 1}
        for word in word_vocab:
            word2idx[word] = len(word2idx)
        self.word2idx = word2idx
        self.wordemb_size = word_embedding.shape[1]
        word_embedding = np.vstack([
            np.random.normal(word_embedding.std(), word_embedding.var(), (2, self.wordemb_size)),
            word_embedding])
        self.wordemb = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=False)

        # build pos vocab and initialize embedding
        pos2idx = {UNK: 0, PAD: 1}
        for pos in pos_vocab:
            pos2idx[pos] = len(pos2idx)
        self.pos2idx = pos2idx
        self.posemb = nn.Embedding(len(self.pos2idx), pos_embedding_size)

        # build label index
        self.label_size = len(labels)
        self.labels = list(labels)
        self.label2idx = {label: i for i, label in enumerate(self.labels)}

        self.dumb = nn.Parameter(torch.randn(self.hidden_size * 2))
        self.dumb.requires_grad = True
        self.tracker = Tracker(self.hidden_size)
        self.proj = nn.Linear(self.wordemb_size * 3 + pos_embedding_size, self.hidden_size * 2)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.mlp = MLP(hidden_size, mlp_layers, mlp_dropout, self.label_size)

    def new_state(self, conf):
        stack = [self.dumb.chunk(2), self.dumb.chunk(2)]
        buffer = deque()
        for idx in conf.buffer:
            buffer.append(self.node_encode(conf.discourse, idx))
        buffer.append(self.dumb.chunk(2))
        tracker_init_state = self.tracker.init_state()
        state = SPINNState(stack, buffer, tracker_init_state)
        state = self.update_tracking(state)
        return state

    def forward(self, state):
        tracking_h = state.tracking[0].view(-1)
        return self.mlp(tracking_h)

    def score(self, state):
        label_score = nnfunc.softmax(self(state), dim=0).tolist()
        return dict(zip(self.labels, label_score))

    def shift(self, state):
        state.stack.append(state.buffer.popleft())
        return self.update_tracking(state)

    def reduce(self, state, nuclear):
        right = state.stack.pop()
        left = state.stack.pop()
        if nuclear == Discourse.SN:
            state.stack.append(right)
        else:
            state.stack.append(left)
        return self.update_tracking(state)

    def update_tracking(self, state):
        tracking = self.tracker(state)
        state.tracking = tracking
        return state

    def node_encode(self, discourse, node_index):
        node = discourse[node_index]
        words = [word if word in self.word2idx else UNK for tag, word in discourse.tags(node.span)] or [PAD]
        tags = [tag if tag in self.pos2idx else UNK for tag, word in discourse.tags(node.span)] or [PAD]
        word_emb = self.wordemb(torch.Tensor([self.word2idx[word] for word in words]).long())
        tags_emb = self.posemb(torch.Tensor([self.pos2idx[tag] for tag in tags]).long())

        w1 = word_emb[0]
        w2 = word_emb[-1]
        t1 = tags_emb[0]
        bow = word_emb.mean(dim=0)

        edu_emb = torch.cat([w1, w2, t1, bow])
        proj = self.proj(edu_emb)
        proj_dropout = self.proj_dropout(proj)
        return proj_dropout.chunk(2)
