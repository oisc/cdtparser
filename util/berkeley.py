# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/2
@Description: Default syntactic and dependency parser
"""

from interface import SentenceParser
from nltk.tree import ParentedTree
from nltk.parse.stanford import StanfordDependencyParser
from structure.dependency import DependencyGraph
import subprocess
import jieba
from jieba import posseg
import os
import logging


jieba.setLogLevel(logging.ERROR)
LRB = '-LRB-'
RRB = '-RRB-'
PREFIX = os.path.dirname(__file__)
BERKELEY_JAR = os.path.join(PREFIX, "../berkeleyparser/BerkeleyParser-1.7.jar")
BERKELEY_GRAMMAR = os.path.join(PREFIX, "../berkeleyparser/chn_sm5.gr")
STANFORD_JAR = os.path.join(PREFIX, "../stanford/stanford-parser.jar")
STANFORD_MODEL = os.path.join(PREFIX, "../stanford/stanford-chinese-corenlp-2018-02-27-models.jar")
STANFORD_GRAMMAR = "edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz"


class BerkeleyWarpper(object):
    def __init__(self, path_to_jar: str, path_to_grammar: str, binarize=False):
        self.env = dict(os.environ)
        self.java_opt = ['-Xmx1024m']
        self.jar = path_to_jar
        self.gr = path_to_grammar

        # check java
        # subprocess.check_output(['java', '-version'])

        # start berkeley parser process
        cmd = ['java']
        cmd.extend(self.java_opt)
        cmd.extend(['-jar', self.jar, '-gr', self.gr])
        if binarize:
            cmd.append('-binarize')
        self.process = subprocess.Popen(cmd, env=self.env, universal_newlines=True, shell=False, bufsize=0,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def parse(self, text: str):
        self.process.stdin.write(text + '\n')
        self.process.stdin.flush()
        return self.process.stdout.readline().strip()

    def __del__(self):
        try:
            self.process.terminate()
        except KeyboardInterrupt:
            pass


class StanfordWrapper(StanfordDependencyParser):
    def _execute(self, cmd, input_, verbose=False):
        # command hack
        cmd.extend(['-outputFormatOptions', 'includePunctuationDependencies'])
        return StanfordDependencyParser._execute(self, cmd, input_, verbose)

    def _make_tree(self, result):
        # pickleable hack
        return DependencyGraph(result, top_relation_label='root')

    def grammar(self):
        raise NotImplementedError()


class ZhBerkeleyParser(SentenceParser):
    name = "berkeley"

    def __init__(self, binarize=False):
        self.berkeley = BerkeleyWarpper(path_to_jar=BERKELEY_JAR, path_to_grammar=BERKELEY_GRAMMAR, binarize=binarize)
        self.stanford = StanfordWrapper(path_to_jar=STANFORD_JAR, path_to_models_jar=STANFORD_MODEL,
                                        model_path=STANFORD_GRAMMAR)
        self.jieba = jieba
        self.posseg = posseg

    def cut(self, text: str):
        yield from self.jieba.cut(text)

    def tag(self, text: str):
        for pair in self.posseg.cut(text):
            yield pair.word, pair.flag

    def parse(self, text: str):
        text = ' '.join(self.cut(text))
        text = text.replace("(", LRB)
        text = text.replace(")", RRB)
        parse_text = self.berkeley.parse(text)
        _parse = ParentedTree.fromstring(parse_text)
        for child in list(_parse.subtrees(lambda t: t.height() == 2 and t.label() != '-NONE-')):
            if child[0] == LRB:
                child[0] = '('
            if child[0] == RRB:
                child[0] = ')'
        return _parse

    def dependency(self, text):
        cuted = list(self.cut(text))
        return next(self.stanford.parse(cuted))
