# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 解析流水线
"""
import pickle
from abc import abstractmethod
from typing import List
from interface import Segmenter, Annotator


class Pipeline:
    def __init__(self, segmenter, annotators):
        self._segmenter = segmenter  # type: Segmenter
        self._annotators = annotators  # type: List[Annotator]

    def segment(self, label, text, start=0, end=-1, info=None):
        return self._segmenter.cut(label, text, start, end, info)

    def annotate(self, discourse):
        for annotator in self._annotators:
            discourse = annotator.annotate(discourse)
        return discourse

    def __call__(self, label, text, start=0, end=-1, info=None):
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
        import config
        from segmenter.svm import SVMCommaSegmenter
        segmenter_model_dir = config.get("segmenter.svm", "model_dir")
        with open(segmenter_model_dir, "rb") as model_fd:
            model = pickle.load(model_fd)
        segmenter = SVMCommaSegmenter(model)
        return Pipeline(segmenter, [])


def create(schema_name):
    for schema in Schema.__subclasses__():
        if schema.name() == schema_name:
            return schema.build_pipeline()
