# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/9
@Description: 接口函数
"""
from nltk import ParentedTree
import config
import schemas
from dataset.cdtb import CDTB
from structure.tree import Sentence


_pipeline = None


def get_pipeline():
    """ 返回 Pipeline 单例 """
    global _pipeline
    if _pipeline is None:
        default_schema = config.get("global", "default_schema")
        _pipeline = schemas.create_pipeline(default_schema)
    return _pipeline


def load(path, encoding="utf-8"):
    """
    从文件夹加载 CDTB 格式文档
    :param path: 文档文件夹路径
    :param encoding: 文档编码
    :return structure.tree.Discourse 生成器
    """
    return CDTB.load(path, encoding=encoding)


def load_xml(file, encoding="utf-8"):
    """
    从文件加载 CDTB 格式文档
    :param file: 文档路径
    :param encoding: 文档编码
    :return: structure.tree.Discourse
    """
    return CDTB.load_xml(file, encoding=encoding)


def save(discourses, file, encoding="utf-8"):
    """
    将一组篇章结构序列化存储到XML格式文档
    :param discourses: List[structure.tree.Discourse]
    :param file: 保存路径
    :param encoding: 文档编码
    :return:
    """
    CDTB.save_xml(discourses, file, encoding=encoding)


def raw2sentences(text):
    """
    将文本切分成句子
    :param text: 切割文本
    :return: structure.tree.Sentence 生成器
    """
    pipeline = get_pipeline()
    segmenter = pipeline.segmenter
    return segmenter.cut_sent(text)


def raw2edus(text):
    """
    将文本切分成 EDU
    :param text: 切分文本
    :return: List[structure.tree.EDU]
    """
    pipeline = get_pipeline()
    bare = pipeline.segment("INSTANCE", text)
    return bare.edus


def sentence2edus(sentence):
    """
    将句子切割为 EDU
    :param sentence: 句子文本
    :return: structure.tree.EDU 生成器
    """
    pipeline = get_pipeline()
    segmenter = pipeline.segmenter
    sentence = Sentence((0, len(sentence)), sentence)
    return segmenter.cut_edu(sentence)


def parse2edus(parse):
    """
    将成分句法树切割为 EDU
    :param parse: 成分句法树 Bracket 格式文本， e.g. '( (IP (NP (PN 我)) (VP (VV 爱) (NP (NR 北京) (NR 天安门)))))'
    :return: structure.tree.EDU 生成器
    """
    pipeline = get_pipeline()
    segmenter = pipeline.segmenter
    parse = ParentedTree.fromstring(parse)
    childs = list(parse.subtrees(lambda t: t.height() == 2 and t.label() != '-NONE-'))
    text = ''.join([child[0] for child in childs])
    sentence = Sentence((0, len(text)), text, parse=parse)
    return segmenter.cut_edu(sentence)


def raw2discourse(label, text, start=0, end=-1, info=None):
    """
    将文本解析为篇章树结构
    :param label: 篇章树编号
    :param text: 篇章文本
    :param start: 解析文本起始偏移量
    :param end: 解析文本结束偏移量
    :param info: 额外信息
    :return: structure.tree.Discourse
    """
    pipeline = get_pipeline()
    return pipeline(label, text, start, end, info)
