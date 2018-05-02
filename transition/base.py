# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/4/28
@Description: Basic Compoent of Transition System
"""
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
    def __init__(self, init_conf, trans, ishistory=False):
        """
        Session 表示一个转移过程
        :param init_conf: 初始状态
        :param trans: 转移系统
        :param ishistory: 是否保留转移历史
        """
        self.conf = init_conf
        self.trans = trans
        self.ishistory = ishistory
        if ishistory:
            self.history_confs = []
            self.history_actions = []
            self.history_params = []

    def __call__(self, action, *args, **kwargs):
        if self.ishistory:
            conf = copy(self.conf)
            self.history_confs.append(self.conf)
            self.history_actions.append(action)
            self.history_params.append((args, kwargs))
        else:
            conf = self.conf
        self.conf = self.trans(action, conf, *args, **kwargs)
        return self.conf

    def history(self):
        if not self.ishistory:
            return None
        else:
            return zip(self.history_confs, self.history_actions, self.history_params)

    def hook(self, when, func):
        pass

    def valid(self):
        return self.trans.valid(self.conf)

    def terminate(self):
        return self.trans.terminate(self.conf)

    def determine(self):
        return self.trans.determine(self.conf)

    def __copy__(self):
        _copy = self.__new__(type(self))
        _copy.conf = copy(self.conf)
        _copy.trans = self.trans
        _copy.ishistory = self.ishistory
        if self.ishistory:
            _copy.history_confs = self.history_confs[:]
            _copy.history_actions = self.history_actions[:]
            _copy.history_params = self.history_params[:]
        return _copy
