# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 解析流水线
"""
import os
import pickle
from abc import abstractmethod
from typing import List
import torch
from interface import Segmenter, Annotator
from structure.tree import Discourse
from segmenter.svm import SVMCommaSegmenter
from treebuilder.spinn_bow import SPINNTreeBuilder
import config


class Pipeline:
    def __init__(self, segmenter, annotators):
        self.segmenter = segmenter  # type: Segmenter
        self.annotators = annotators  # type: List[Annotator]

    def segment(self, label, text, start=0, end=-1, info=None):
        return self.segmenter.cut(label, text, start, end, info)

    def annotate(self, discourse):
        for annotator in self.annotators:
            discourse = annotator.annotate(discourse)
        return discourse

    def __call__(self, label, text, start=0, end=-1, info=None):
        """
        按流程进行篇章解析
        :param label: 篇章编号
        :param text: 篇章文本
        :param start: 需要解析的篇章起始偏移量
        :param end: 需要解析的篇章终止偏移量
        :param info: 额外调试信息
        :return: 解析后篇章
        :rtype: Discourse
        """
        discourse = self.segment(label, text, start, end, info)
        return self.annotate(discourse)


class Schema:
    @staticmethod
    @abstractmethod
    def build_pipeline():
        """
        :rtype: Pipeline
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def name():
        raise NotImplementedError()


class Baseline(Schema):
    @staticmethod
    def name():
        return "baseline"

    @staticmethod
    def build_pipeline():
        __prefix = os.path.dirname(__file__)
        segmenter_model_dir = config.get("segmenter.svm", "model_dir")
        with open(os.path.join(__prefix, segmenter_model_dir), "rb") as model_fd:
            segmenter_model = pickle.load(model_fd)
        segmenter = SVMCommaSegmenter(segmenter_model)
        treebuilder_model_dir = config.get("treebuilder.spinn_cnn", "model_dir")
        beam_size = config.get("treebuilder.spinn_bow", "beam_size", rtype=int)
        with open(os.path.join(__prefix, treebuilder_model_dir), "rb") as model_fd:
            treebuilder_model = torch.load(model_fd)
        treebuilder = SPINNTreeBuilder(treebuilder_model, beam_size=beam_size)
        return Pipeline(segmenter, [treebuilder])


def create_pipeline(schema_name):
    """
    Pipeline 工厂函数
    :param schema_name: 解析策略名称
    :return:
    :rtype: Pipeline
    """
    for schema in Schema.__subclasses__():
        if schema.name() == schema_name:
            return schema.build_pipeline()
    raise ValueError("No schema named \"%s\"" % schema_name)
