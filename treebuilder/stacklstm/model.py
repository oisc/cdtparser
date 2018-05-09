# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/7
@Description: Stack LSTM shift-reduce model
"""
from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
from transition.shiftreduce import SRTransition
import numpy as np


UNK = "<UNK>"
PAD = "<PAD>"


class StackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnns = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_p)
        self.empty = nn.Parameter(torch.randn(self.hidden_size))

    def init_state(self):
        return [[(torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size))
                for _ in range(self.num_layers)]]

    def push(self, stack, x):
        if x.dim() == 1:
            x = x.view(1, -1)
        new_state = []
        for i, cell in enumerate(self.rnns):
            x = self.dropout(x)
            h, c = cell(x, stack[-1][i])
            new_state.append((h, c))
            x = h
        stack.append(new_state)
        return stack

    def pop(self, stack):
        if len(stack) <= 1:
            raise ValueError()
        stack.pop()
        return stack

    def forward(self, stack):
        return stack[-1][-1][0].view(-1) if len(stack) > 1 else self.empty


class Reducer(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.comp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )

    def forward(self, left, right):
        comp = self.comp(torch.cat((left, right), dim=0))
        return comp


class StackLSTMState:
    class StackState:
        def __init__(self, lstm):
            self.lstm = lstm
            self.input = []
            self.states = lstm.init_state()

        def push(self, x):
            self.input.append(x)
            self.states = self.lstm.push(self.states, x)

        def pop(self):
            o = self.input.pop()
            self.states = self.lstm.pop(self.states)
            return o

        def output(self):
            return self.lstm(self.states)

        def __copy__(self):
            copy_state = self.__new__(type(self))
            copy_state.lstm = self.lstm
            copy_state.input = [x.clone() for x in self.input]
            copy_state.states = [[(h.clone(), c.clone()) for h, c in s] for s in self.states]

    def __init__(self, stack_lstm, buffer_lstm, action_lstm):
        self.stack = StackLSTMState.StackState(stack_lstm)
        self.buffer = StackLSTMState.StackState(buffer_lstm)
        self.action = StackLSTMState.StackState(action_lstm)

    def __copy__(self):
        copy_state = self.__new__(type(self))
        copy_state.stack = copy(self.stack)
        copy_state.buffer = copy(self.buffer)
        copy_state.action = copy(self.action)
        return copy_state


class StackLSTMTransModel(nn.Module):
    def __init__(self, hidden_size, lstm_layers, lstm_dropout,
                 word_vocab, word_embedding,
                 pos_vocab, pos_embedding_size,
                 action_vocab, action_embedding_size,
                 relation_vocab, relation_embedding_size,
                 sentence_position_cutoff, sentence_position_embedding_size,
                 edu_position_cutoff, edu_position_embedding_size,
                 edu_length_cutoff, edu_length_embedding_size,
                 labels):
        nn.Module.__init__(self)
        # word embedding
        word_embedding_size = word_embedding.shape[1]
        self.word_embedding_size = word_embedding_size
        self.word2idx = {PAD: 0, UNK: 1}
        for word in word_vocab:
            self.word2idx[word] = len(self.word2idx)
        word_embedding = np.vstack([
            np.random.normal(word_embedding.std(), word_embedding.var(), (2, word_embedding_size)),
            word_embedding
        ])
        self.wordemb = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=False)
        # part of speech embedding
        self.pos_embedding_size = pos_embedding_size
        self.pos2idx = {PAD: 0, UNK: 1}
        for pos in pos_vocab:
            self.pos2idx[pos] = len(self.pos2idx)
        self.posemb = nn.Embedding(len(self.pos2idx), self.pos_embedding_size)
        # action embedding
        self.action_embedding_size = action_embedding_size
        self.action2idx = {action: i for i, action in enumerate(action_vocab)}
        self.actionemb = nn.Embedding(len(self.action2idx), self.action_embedding_size)

        # sentence position embedding
        self.sentence_cutoff = sentence_position_cutoff
        self.sentposemb = nn.Embedding(sentence_position_cutoff, sentence_position_embedding_size)
        # edu position embedding
        self.edu_position_cutoff = edu_position_cutoff
        self.eduposemb = nn.Embedding(edu_position_cutoff, edu_position_embedding_size)
        # edu length embedding
        self.edu_length_cutoff = edu_length_cutoff
        self.edulenemb = nn.Embedding(edu_length_cutoff, edu_length_embedding_size)
        # relation embedding
        self.relation_embedding_size = relation_embedding_size
        self.relation2idx = {UNK: 0}
        for relation in relation_vocab:
            self.relation2idx[relation] = len(self.relation2idx)
        self.relationemb = nn.Embedding(len(self.relation2idx), relation_embedding_size)

        self.stack_lstm = StackLSTM(hidden_size, hidden_size, lstm_layers, lstm_dropout)
        self.buffer_lstm = StackLSTM(hidden_size, hidden_size, lstm_layers, lstm_dropout)
        self.action_lstm = StackLSTM(hidden_size, hidden_size, lstm_layers, lstm_dropout)
        self.edu_proj = nn.Sequential(
            nn.Linear(word_embedding_size * 2 + pos_embedding_size + edu_length_embedding_size +
                      sentence_position_embedding_size + edu_position_embedding_size, hidden_size),
            nn.ReLU()
        )
        self.reducer = Reducer(hidden_size)
        self.action_proj = nn.Sequential(
            nn.Linear(action_embedding_size + relation_embedding_size, hidden_size),
            nn.ReLU()
        )
        self.labels = list(labels)
        self.label2idx = {label: i for i, label in enumerate(self.labels)}
        self.logits = nn.Linear(hidden_size * 3, len(labels))

    def new_state(self, conf):
        state = StackLSTMState(self.stack_lstm, self.buffer_lstm, self.action_lstm)
        for node_idx in reversed(conf.buffer):
            state.buffer.push(self.edu_encode(conf.discourse, node_idx))
        return state

    def update(self, state, action, nuclear=None, relation=None):
        action_emb = self.action_encode(action, nuclear, relation)
        state.action.push(action_emb)
        if action == SRTransition.SHIFT:
            x = state.buffer.pop()
            state.stack.push(x)
        else:
            right = state.stack.pop()
            left = state.stack.pop()
            comp = self.reducer(left, right)
            state.stack.push(comp)
        return state

    def score(self, state):
        label_score = nnfunc.softmax(self(state), dim=0).tolist()
        return dict(zip(self.labels, label_score))

    def forward(self, state):
        summury = torch.cat([state.stack.output(), state.buffer.output(), state.action.output()], dim=0)
        return self.logits(summury)

    def edu_encode(self, discourse, node_index):
        node = discourse[node_index]
        words = list(discourse.words(node.span))
        w1 = words[0] if len(words) > 1 else PAD
        w2 = words[-1] if len(words) > 2 else PAD
        w1_id = self.word2idx[w1] if w1 in self.word2idx else self.word2idx[UNK]
        w2_id = self.word2idx[w2] if w2 in self.word2idx else self.word2idx[UNK]
        word_emb = self.wordemb(torch.Tensor([w1_id, w2_id]).long()).view(-1)
        tags = [tag for tag, word in discourse.tags(node.span)]
        t1 = tags[0] if len(tags) else PAD
        t1_id = self.pos2idx[t1] if t1 in self.pos2idx else self.pos2idx[UNK]
        pos_emb = self.posemb(torch.Tensor([t1_id]).long()).view(-1)

        sentence_of_edu = discourse.nearest_sentence(node.span)
        sentpos = discourse.sentences.index(sentence_of_edu)
        edus_same_sent = list(discourse.neareset_edus(sentence_of_edu.span))
        edupos = edus_same_sent.index(node)
        edulen = len(node.text)
        sentpos = sentpos if sentpos < self.sentence_cutoff else self.sentence_cutoff - 1
        edupos = edupos if edupos < self.edu_position_cutoff else self.edu_position_cutoff - 1
        edulen = edulen if edulen < self.edu_length_cutoff else self.edu_length_cutoff - 1

        sentpos_emb = self.sentposemb(torch.Tensor([sentpos]).long()).view(-1)
        edupos_emb = self.eduposemb(torch.Tensor([edupos]).long()).view(-1)
        edulen_emb = self.edulenemb(torch.Tensor([edulen]).long()).view(-1)

        proj = self.edu_proj(torch.cat((word_emb, pos_emb, sentpos_emb, edupos_emb, edulen_emb)))
        return proj

    def action_encode(self, action, nuclear, relation):
        action = "%s_%s" % (action, nuclear)
        relation = relation or "NOREL"
        action_id = self.action2idx[action]
        relation = relation if relation in self.relation2idx else UNK
        relation_id = self.relation2idx[relation]
        action_emb = self.actionemb(torch.Tensor([action_id]).long()).view(-1)
        relation_emb = self.relationemb(torch.Tensor([relation_id]).long()).view(-1)
        proj = self.action_proj(torch.cat((action_emb, relation_emb)))
        return proj
