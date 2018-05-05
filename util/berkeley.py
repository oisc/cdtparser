# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/2
@Description: 
"""

from interface import SentenceParser
from nltk.tree import ParentedTree
import subprocess
import jieba
from jieba import posseg
import os
import logging


jieba.setLogLevel(logging.ERROR)


class BerkeleyWarpper(object):
    def __init__(self, path_to_jar: str, path_to_grammar: str):
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


class ZhBerkeleyParser(SentenceParser):
    name = "berkeley"

    def __init__(self):
        _prefix = os.path.dirname(__file__)
        self.berkeley = BerkeleyWarpper(path_to_jar=os.path.join(_prefix, '../berkeleyparser/BerkeleyParser-1.7.jar'),
                                        path_to_grammar=os.path.join(_prefix, '../berkeleyparser/chn_sm5.gr'))
        self.jieba = jieba
        self.posseg = posseg

    def cut(self, text: str):
        yield from self.jieba.cut(text)

    def tag(self, text: str):
        for pair in self.posseg.cut(text):
            yield pair.word, pair.flag

    def parse(self, text: str):
        text = ' '.join(self.cut(text))
        parse_text = self.berkeley.parse(text)
        return ParentedTree.fromstring(parse_text)

    def dependency(self, text):
        raise NotImplementedError()
