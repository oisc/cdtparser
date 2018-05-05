# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: Shift-Reduce 转移系统
"""
from copy import copy
from collections import deque
from .base import Configuration, Transition


class SRConfiguration(Configuration):
    def __init__(self, discourse, inputs=None):
        self.discourse = discourse
        self.stack = []
        if inputs is None:
            inputs = [discourse.index(node) for node in discourse.forest]
        self.buffer = deque(inputs)

    def buffer_empty(self):
        return len(self.buffer) == 0

    def __copy__(self):
        _copy = self.__new__(type(self))
        _copy.discourse = copy(self.discourse)
        _copy.stack = self.stack[:]
        _copy.buffer = self.buffer.copy()
        return _copy


class SRTransition(Transition):
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def __init__(self):
        Transition.__init__(self)

    @Transition.sign_action(SHIFT)
    def shift(self, conf: SRConfiguration):
        conf.stack.append(conf.buffer.popleft())
        return conf

    @Transition.sign_action(REDUCE)
    def reduce(self, conf: SRConfiguration, nuclear=None, relation=None):
        s1 = conf.stack.pop()
        s2 = conf.stack.pop()
        s1_node = conf.discourse[s1]
        s2_node = conf.discourse[s2]
        parent_span = s2_node.span[0], s1_node.span[1]
        parent = conf.discourse.add_relation(parent_span, nuclear, children=[s2_node, s1_node], relation=relation)
        conf.stack.append(conf.discourse.index(parent))
        return conf

    def valid(self, conf):
        if self.terminate(conf):
            return set()
        if self.determine(conf):
            return {self.determine(conf)}
        return {self.SHIFT, self.REDUCE}

    def terminate(self, conf: SRConfiguration):
        return conf.buffer_empty() and len(conf.stack) == 1

    def determine(self, conf: SRConfiguration):
        if conf.buffer_empty() and (not self.terminate(conf)):
            return self.REDUCE
        elif len(conf.stack) < 2 and (not conf.buffer_empty()):
            return self.SHIFT
        else:
            return None
