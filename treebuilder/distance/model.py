# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/6/25
@Description: Syntactic Distance Model
Ref: Shen Y, Lin Z, Jacob A P, et al. Straight to the Tree: Constituency Parsing with Neural Syntactic Distance. 2018.
"""
import torch
import torch.nn as nn


UNK = "-UNK-"
PAD = "-PAD-"
START = "-START-"
END = "-END-"


class DistanceModel(nn.Module):
    def __init__(self, vocab, dropout_p, hidden_size, wordemb_size=300, pretrained=None):
        nn.Module.__init__(self)
        self.vocab = vocab
        if pretrained is not None:
            self.wordemb = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained), freeze=False)
            self.wordemb_size = self.wordemb.weight.size(1)
        else:
            self.wordemb = nn.Embedding(len(vocab), wordemb_size)
            self.wordemb_size = wordemb_size

        self.word_rnn = nn.LSTM(self.wordemb_size, hidden_size, num_layers=2, bidirectional=True,
                                batch_first=True, dropout=dropout_p)
        self.word_cnn = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.edu_rnn = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                               bidirectional=True, batch_first=True, dropout=dropout_p)
        self.edu_cnn = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=2),
            nn.ReLU()
        )
        self.arc_rnn = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                               bidirectional=True, batch_first=True, dropout=dropout_p)

        self.distance = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, discourse):
        edu_encoded = self.encode_edus(discourse)
        arc_encoded = self.encode_arcs(edu_encoded)
        distance = self.distance(arc_encoded).view(-1)
        return distance

    def encode_arcs(self, edu_encoded):
        arcs_inputs, _ = self.edu_rnn(edu_encoded.unsqueeze(0))
        arcs_encoded = self.edu_cnn(arcs_inputs.permute(0, 2, 1))
        arcs_encoded = arcs_encoded.permute(0, 2, 1)
        arcs_encoded, _ = self.arc_rnn(arcs_encoded)
        arcs_encoded = arcs_encoded.squeeze(-1)
        return arcs_encoded

    def encode_edus(self, discourse):
        word_ids = []
        mask = []
        for i, edu in enumerate(discourse.edus):
            words = list(discourse.words(edu.span))
            if i == 0:
                words = [START] + words
            if i + 1 == len(discourse.edus):
                words = words + [END]
            if len(words) < 2:
                words = words + [PAD] * (2 - len(words))
            word_ids.extend([self.vocab[word] if word in self.vocab else self.vocab[UNK] for word in words])
            mask.extend([1] + [0] * (len(words) - 2) + [1])

        word_ids = torch.LongTensor(word_ids)
        word_emb = self.wordemb(word_ids).unsqueeze(0)
        word_encoded, _ = self.word_rnn(word_emb)
        word_encoded = word_encoded.squeeze(0)
        edu_inputs = word_encoded[torch.ByteTensor(mask)].unsqueeze(0).permute(0, 2, 1)
        edu_encoded = self.word_cnn(edu_inputs).permute(0, 2, 1).squeeze(0)
        return edu_encoded
