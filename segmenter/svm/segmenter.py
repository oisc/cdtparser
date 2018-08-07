# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/3
@Description: Segmenter as comma classification
Ref: 李艳翠, 冯文贺, 周国栋, 等. 基于逗号的汉语子句识别研究[J]. 北京大学学报 (自然科学版), 2013, 49(1): 7-14.
"""
from interface import Segmenter, SentenceParseError, SegmentError
from util import ZhDefaultParser
from structure.tree import Sentence, EDU, Discourse
import logging

logger = logging.getLogger(__name__)


class SVMCommaSegmenter(Segmenter):
    def __init__(self, model):
        self._eos = ['!', '.', '?', '！', '。', '？']
        self._edu_candidate = ',，;；'
        self._edu_terminate = '.。!！?？'
        self._pairs = {'“': "”", "「": "」"}
        self.parser = ZhDefaultParser()
        self.model = model

    def cut_sent(self, text, start=0, end=-1):
        if end < 0:
            end = len(text)
        last_cut = start
        for i in range(start, end):
            if text[i] in self._eos:
                sent_span = last_cut, i + 1
                yield Sentence(sent_span, text[slice(*sent_span)])
                last_cut = i + 1
        if last_cut < end:
            sent_span = last_cut, end
            yield Sentence(sent_span, text[slice(*sent_span)])

    def cut_edu(self, sentence):
        text = sentence.text
        try:
            parse = sentence.parse()
        except ValueError:
            parse = self.parser.parse(text)
            sentence.set(parse=parse)
        offset = sentence.span[0]
        last_cut = 0
        for i in range(len(text)):
            if text[i] in self._edu_terminate:
                edu_span = last_cut + offset, i + offset + 1
                edu_text = text[last_cut: i + 1]
                last_cut = i + 1
                yield EDU(edu_span, edu_text)
            elif text[i] in self._edu_candidate and self.model.predict(i, parse):
                edu_span = last_cut + offset, i + offset + 1
                edu_text = text[last_cut: i + 1]
                last_cut = i + 1
                yield EDU(edu_span, edu_text)
        if last_cut + offset < sentence.span[1]:
            edu_span = last_cut + offset, sentence.span[1]
            edu_text = text[last_cut:]
            yield EDU(edu_span, edu_text)

    def cut(self, label, text, start=0, end=-1, info=None):
        if end < 0:
            end = len(text)
        sentences = []
        edus = []
        try:
            for sentence in self.cut_sent(text, start, end):
                sentences.append(sentence)
                edus.extend(self.cut_edu(sentence))
        except SentenceParseError as e:
            logger.error("during segmenting %s, %s" % (label, e))
            raise SegmentError("error segmenting %s" % label, label, text, start, end, info)
        return Discourse(label, text, (start, end), edus, sentences)
