# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/4/28
@Description: 篇章树结构
"""
from collections import deque
from copy import copy
from itertools import chain
from typing import List
from nltk.tree import ParentedTree


class CDTNode(ParentedTree):
    """ 篇章树节点基类 """
    def __init__(self, nodename):
        self._nodename = nodename
        ParentedTree.__init__(self, nodename, [])

    def _get_node(self):
        raise DeprecationWarning()

    def _set_node(self, value):
        raise DeprecationWarning()

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "<%s@%d>" % (self._nodename, hash(self))


class EDU(CDTNode):
    """ 基本篇章单元 """
    def __init__(self, span, text):
        CDTNode.__init__(self, "EDU")
        self.span = span
        self.text = text
        self.append(text)

    def label(self):
        return "EDU %s" % str(self.span)


class Sentence:
    """ 句子 """
    def __init__(self, span, text, words=None, tags=None, parse=None):
        self.span = span
        self.offset = span[0]
        self.text = text
        self._words = words
        self._tags = tags
        self._parse = parse

    def set(self, words=None, tags=None, parse=None):
        if not (words or tags or parse):
            raise ValueError("at least one of words, tags or parse should be given.")
        if words:
            self._words = words
        if tags:
            self._tags = tags
        if parse:
            self._parse = parse

    def words(self, span=None):
        if self._words:
            _words = self._words
        else:
            _words = [word for tag, word in self.tags()]
        if span is None:
            yield from _words
        else:
            offset = self.offset
            for word in _words:
                if span[0] <= offset <= offset + len(word) <= span[1]:
                    yield word
                offset += len(word)

    def tags(self, span=None):
        if self._tags:
            _tags = self._tags
        else:
            _tags = [(node.label(), node[0])
                     for node in self.parse().subtrees(lambda t: t.height() == 2 and t.label() != '-NONE-')]
        if span is None:
            yield from _tags
        else:
            offset = self.offset
            for tag, word in _tags:
                if span[0] <= offset <= offset + len(word) <= span[1]:
                    yield tag, word
                offset += len(word)

    def parse(self):
        if not self._parse:
            raise ValueError("No syntactic information given")
        else:
            return self._parse

    def __copy__(self):
        _copy = Sentence(self.span, self.text, self._words, self._tags, self._parse)
        return _copy


class Relation(object):
    """ 篇章关系 """
    def __init__(self, explicit=None, fine=None, coarse=None, connective=None, connective_span=None):
        """
        :param explicit: 显式关系还是隐式关系
        :param fine: 细粒度关系
        :param coarse: 粗粒度关系
        :param connective: 连接词
        :param connective_span: 连接词起止位置
        """
        object.__setattr__(self, 'explicit', explicit)
        object.__setattr__(self, 'fine', fine)
        object.__setattr__(self, 'coarse', coarse)
        object.__setattr__(self, 'connective', connective)
        object.__setattr__(self, 'connective_span', connective_span)

    def __setattr__(self, key, value):
        # 让 Relation 变成不可变类型
        raise TypeError("Relation is immutable")


class RelationNode(CDTNode):
    """ 关系节点 NS SN NN 为篇章关系中心位置 """
    NS = "NS"
    SN = "SN"
    NN = "NN"

    def __init__(self, span, nuclear, children, relation=None):
        CDTNode.__init__(self, "RelationNode")
        self.span = span
        self.nuclear = nuclear
        self.extend(children)
        if relation is None:
            relation = Relation()
        self.relation = relation

    def label(self):
        if self.relation.explicit is None:
            explicit = "未指定"
        else:
            explicit = "显式关系" if self.relation.explicit else "隐式关系"
        return 'Relation %s \n%s %s\n%s %s %s' % (str(self.span), self.nuclear, explicit,
                                                  self.relation.fine, self.relation.coarse, self.relation.connective)


class Discourse:
    NS = RelationNode.NS
    SN = RelationNode.SN
    NN = RelationNode.NN

    """ 篇章结构 """
    def __init__(self, label, text, edus, sentences, info=None):
        self.label = label
        self.text = text
        self.edus = edus
        self.sentences = sentences  # type: List[Sentence]
        self.forest = edus[:]
        self.relations = []  # type: List[RelationNode]
        self.info = info

    def add_relation(self, span, nuclear, parent=None, children=None, relation=None):
        """
        添加篇章关系，可以自底向上，也可以自顶向下
        :param span: 关系管辖范围
        :param nuclear: 关系核心位置
        :param parent: 关系节点的父节点，如果不指定，则认为自底向上添加关系
        :param children: 关系节点合并的子节点，如果不指定，从森林中包括所有管辖范围内节点
        :param relation: 关系类型
        :return: 添加的关系节点
        """
        if parent is None:
            parent = self.forest
        if children is None:
            children = []
            for child in parent:
                if span[0] <= child.span[0] <= child.span[1] <= span[1]:
                    children.append(child)

        child_indices = [parent.index(child) for child in children]
        for i in sorted(child_indices, reverse=True):
            del parent[i]
        node = RelationNode(span, nuclear, children, relation)
        parent.insert(min(child_indices), node)
        self.relations.append(node)
        return node

    def complete(self):
        """ 篇章树是否完整树结构 """
        return len(self.forest) == 1

    def tree(self):
        """ 返回篇章树，如果不完整，抛出异常 """
        if self.complete():
            return self.forest[0]
        else:
            raise ValueError("Discourse tree is not complete")

    def words(self, span):
        """ 返回范围内所有词 """
        yield from chain(*[sent.words(span) for sent in self.sentences])

    def tags(self, span):
        """ 返回管辖范围内词性标注 (tag, word) """
        yield from chain(*[sent.tags(span) for sent in self.sentences])

    def nearest_sentence(self, span):
        """ 返回覆盖范围的最近句子 """
        for sent in self.sentences:
            if sent.span[0] <= span[0] <= span[1] <= sent.span[1]:
                return sent
        return None

    def traverse(self, root=None, prior=False):
        """ 遍历森林，prior=True则为先序遍历 """
        if root is None:
            for root in self.forest:
                yield from self.traverse(root, prior)
        else:
            if prior:
                yield root
            for node in root:
                if isinstance(node, EDU):
                    yield node
                else:
                    yield from self.traverse(node, prior=prior)
            if not prior:
                yield root

    def __copy__(self):
        """ 覆盖拷贝函数 """
        copy_edus = [EDU(edu.span, edu.text) for edu in self.edus]
        copy_sentence = [copy(sentence) for sentence in self.sentences]
        self2copy = dict(zip(self.edus, copy_edus))

        copy_discourse = Discourse(self.label, self.text, copy_edus, copy_sentence)
        # copy relation node bottom up
        for node in self.traverse():
            if isinstance(node, RelationNode):
                children = []
                for child in node:
                    children.append(self2copy[child])
                copy_relation = copy_discourse.add_relation(node.span, node.nuclear,
                                                            children=children, relation=node.relation)
                self2copy[node] = copy_relation
        return copy_discourse

    def binarize(self, prior=False):
        """ 二元化篇章树（森林），prior=True 则先合并前面的子节点
         prior=True: (A B C) -> ((A B) C)
         prior=False: (A B C) -> (A (B C))
         """
        copy_edus = [EDU(edu.span, edu.text) for edu in self.edus]
        copy_sentence = [copy(sentence) for sentence in self.sentences]
        self2copy = dict(zip(self.edus, copy_edus))

        bin_discoursre = Discourse(self.label, self.text, copy_edus, copy_sentence)
        # copy relation node bottom up
        for node in self.traverse():
            if isinstance(node, RelationNode):
                children = deque()
                for child in node:
                    children.append(self2copy[child])
                while len(children) > 2:
                    if prior:
                        left = children.popleft()
                        right = children.popleft()
                    else:
                        right = children.pop()
                        left = children.pop()
                    span = left.span[0], right.span[1]
                    bin_relation = bin_discoursre.add_relation(span, node.nuclear,
                                                               children=[left, right], relation=node.relation)
                    if prior:
                        children.appendleft(bin_relation)
                    else:
                        children.append(bin_relation)
                copy_relation = bin_discoursre.add_relation(node.span, node.nuclear,
                                                            children=children, relation=node.relation)
                self2copy[node] = copy_relation
        return bin_discoursre
