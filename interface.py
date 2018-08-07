# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/2
@Description: 公用接口
"""
from abc import abstractmethod


class SentenceParser:
    # 词法解析器名称
    name = "auto"

    @abstractmethod
    def cut(self, text):
        """
        分词接口
        :param text: 输入句子
        :return 词可迭代对象
        """
        raise NotImplementedError()

    @abstractmethod
    def tag(self, text):
        """
        词性标注接口
        :param text: 输入句子
        :return: (word, tag) 可迭代对象
        """
        raise NotImplementedError()

    @abstractmethod
    def parse(self, text):
        """
        成分句法解析接口
        :param text: 输入句子
        :return: nltk.tree.Tree 对象
        """
        raise NotImplementedError()

    @abstractmethod
    def dependency(self, text):
        """
        依存句法解析接口
        :param text: 输入句子
        :return: nltk.parse.DependencyGraph 对象
        """
        raise NotImplementedError()


class SentenceParseError(Exception):
    """ 句法解析错误 """


class Segmenter:
    """ 篇章子句分割器接口 """

    @abstractmethod
    def cut(self, label, text, start=0, end=-1, info=None):
        """
        篇章分割方法
        :param label: 篇章编号
        :param text: 篇章文本
        :param start: 子句切分的text起始偏移量
        :param end: 子句切分的text终止偏移量
        :param info: 调试信息
        :return: 不包含任何关系的 Discourse
        """
        raise NotImplementedError()

    @abstractmethod
    def cut_sent(self, text, start=0, end=-1):
        """
        将文本切割成句子
        :return: List[Sentence]
        """
        raise NotImplementedError()

    @abstractmethod
    def cut_edu(self, sentence):
        """
        将句子切割成 EDU
        :param sentence:
        :return:
        """


class SegmentError(Exception):
    def __init__(self, message, label, text, start, end, info):
        self.message = message
        self.label = label
        self.text = text
        self.start = start
        self.end = end
        self.info = info


class Annotator:
    @abstractmethod
    def annotate(self, discourse):
        raise NotImplementedError()


class ParseError(Exception):
    def __init__(self, message, last_discourse):
        self.message = message
        self.last_discourse = last_discourse
