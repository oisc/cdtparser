# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/4/28
@Description: Basic Compoent of Transition System
"""
from collections import namedtuple
from copy import copy
from functools import partial
from abc import abstractmethod


class Configuration:
    """ Configuration 实例用于存储转移系统状态"""
    def __hash__(self):
        return id(self)


class ActionDescriptor:
    """ 转移系统动作的注解类 """
    def __init__(self, name, func):
        """
        :param name: 转移动作名称
        :param func: 绑定的动作执行函数
        """
        self.name = name
        self.func = func

    def __get__(self, instance, owner):
        return partial(self.func, instance)


class Transition:
    """ 转移系统 """
    def __init__(self):
        self.action_names = set()
        self.action_funcs = {}
        for k, v in self.__class__.__dict__.items():
            if isinstance(v, ActionDescriptor):
                self.action_names.add(v.name)
                self.action_funcs[v.name] = getattr(self, k)

    @staticmethod
    def sign_action(name):
        """ 注册转移动作注解方法 """
        def wrapper(func):
            return ActionDescriptor(name, func)
        return wrapper

    def __call__(self, action, conf, *args, **kwargs):
        assert action in self.valid(conf)
        return self.action_funcs[action](conf, *args, **kwargs)

    @abstractmethod
    def valid(self, conf):
        """ 返回转移状态可行的转移动作集合 """
        raise NotImplementedError()

    @abstractmethod
    def terminate(self, conf):
        """ 转移状态是否终止 """
        return NotImplementedError()

    @abstractmethod
    def determine(self, conf):
        """ 是否由确定的转移动作 """
        return NotImplementedError()


class Session:
    Memory = namedtuple("Memory", "confs actions params")

    def __init__(self, init_conf, trans, history=False, state=None):
        """
        Session 表示一个转移过程
        :param init_conf: 初始状态
        :param trans: 转移系统
        :param history: 是否保留转移历史
        """
        self.current = init_conf
        self.trans = trans
        self.history = history
        self.state = state
        if history:
            self.memory = Session.Memory([], [], [])

    def __call__(self, action, *args, **kwargs):
        if self.history:
            conf = copy(self.current)
            self.memory.confs.append(self.current)
            self.memory.actions.append(action)
            self.memory.params.append((args, kwargs))
        else:
            conf = self.current
        self.current = self.trans(action, conf, *args, **kwargs)
        return self

    def valid(self):
        return self.trans.valid(self.current)

    def terminate(self):
        return self.trans.terminate(self.current)

    def determine(self):
        return self.trans.determine(self.current)

    def __copy__(self):
        _copy = self.__new__(type(self))
        _copy.current = copy(self.current)
        _copy.trans = self.trans
        _copy.history = self.history
        _copy.state = copy(self.state)
        if self.history:
            _copy.memory = Session.Memory(self.memory.confs[:], self.memory.actions[:], self.memory.params[:])
        return _copy
