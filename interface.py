# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/2
@Description: 公用接口
"""
from abc import abstractmethod


class SentenceParser:
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
