# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/2
@Description: CDTB 数据集工具
"""
import os
import pickle
import re

import tqdm
from xml.etree import ElementTree
from nltk.tree import ParentedTree
from structure.tree import EDU, Sentence, Relation, Discourse
from util import DefaultParser
import logging


finemap = {'因果类': ['因果关系', '推断关系', '假设关系', '目的关系', '条件关系', '背景关系'],
           '并列类': ['并列关系', '顺承关系', '递进关系', '选择关系', '对比关系'],
           '转折类': ['转折关系', '让步关系'],
           '解说类': ['解说关系', '总分关系', '例证关系', '评价关系']}
coarsemap = {}
for coarse_class, fine_classes in finemap.items():
    coarsemap.update((sub_class, coarse_class) for sub_class in fine_classes)
logger = logging.getLogger(__name__)
ctb_pat = re.compile(r'<S ID=(?P<sid>\d*?\w?)>(?P<stext>.*?)</S>', re.DOTALL | re.M)
nuclear_map = {"1": Discourse.NS, "2": Discourse.SN, "3": Discourse.NN}


class CDTB:
    """
    CDTB 数据集工具类
    """
    def __init__(self, train, test, ctb=None, cache_dir=None, encoding="utf-8", ctb_encoding="utf-8"):
        """
        :param train: 训练集路径
        :param test: 测试集路径
        :param ctb: CTB 路径，如果不给出，使用自动句法解析器补充句法信息
        :param cache_dir: 缓存路径，如果不给出，则不缓存
        :param encoding: CDTB 编码
        :param ctb_encoding: CTB 编码
        """
        self.train_path = train
        self.test_path = test
        self.ctb_encoding = ctb_encoding
        self.ctb = self.load_ctb(ctb) if ctb else None
        self.parser = DefaultParser()
        self.train = []
        self.test = []
        self.cache_dir = cache_dir
        self.cdtb_encoding = encoding
        self.build()

    def build(self):
        # cache key
        cache_key = "CDTB_cache_%s.pickle" % ("ctb" if self.ctb else self.parser.name)
        if self.cache_dir:
            # if cache exists load from cache
            cache_file = os.path.join(self.cache_dir, cache_key)
            if os.path.exists(cache_file):
                logging.info("load cached CDTB dataset from %s" % cache_key)
                with open(cache_file, "rb") as cache_fd:
                    self.train = pickle.load(cache_fd)
                    self.test = pickle.load(cache_fd)
                return

        if not self.ctb:
            logger.info("Notice! CTB is not given, %s parser will be used to generate extra syntacitcal information. "
                        "This may take a while." % self.parser.name)
        # load train set
        logger.info("loading train set")
        for file in tqdm.tqdm(os.listdir(self.train_path)):
            for tree in self.load_xml(os.path.join(self.train_path, file)):
                self.train.append(tree)

        # load test set
        logger.info("loading test set")
        for file in tqdm.tqdm(os.listdir(self.test_path)):
            for tree in self.load_xml(os.path.join(self.test_path, file)):
                self.test.append(tree)

        # save as cache file
        if self.cache_dir:
            logger.info("save cached CDTB dataset to %s" % cache_key)
            with open(os.path.join(self.cache_dir, cache_key), "wb+") as cache_fd:
                pickle.dump(self.train, cache_fd)
                pickle.dump(self.test, cache_fd)

    def load_ctb(self, ctb_path):
        ctb = {}
        for file in os.listdir(ctb_path):
            with open(os.path.join(ctb_path, file), "r", encoding=self.ctb_encoding) as fd:
                for sid, parse in ctb_pat.findall(fd.read()):
                    ctb[sid] = ParentedTree.fromstring(parse)
        return ctb

    def load_xml(self, file):
        with open(file, "r", encoding=self.cdtb_encoding) as fd:
            dom = ElementTree.fromstring(fd.read())  # type: ElementTree.Element
            for p in dom.iterfind("P"):  # type: ElementTree.Element
                pid = int(p.get("ID"))
                # 忽略没有关系的段落
                if pid > 0:
                    yield self._dom2discourse(p, pid, os.path.dirname(file))

    def _dom2discourse(self, p, label, info):
        raw = p.find("RAW")  # type: ElementTree.Element
        _text = raw.get("Sentence")
        sentspans = self._xml2pos(raw.get("SentencePosition"))
        _sentences = []
        _offset = 0
        for s, e in sentspans:
            _sentences.append(_text[s+_offset:e+_offset])
            _offset += 1
        text = ''.join(_sentences)
        sids = raw.get("SID").split("|") if raw.get("SID") else []
        sentences = []
        for i, span in enumerate(sentspans):
            if self.ctb and sids:
                parse = self.ctb[sids[i]]
            else:
                parse = self.parser.parse(text[slice(*span)])
            sentences.append(Sentence(span, text[slice(*span)], parse=parse))
        eduspans = self._xml2pos(raw.get("EduPosition"))
        edus = [EDU(span, text[slice(*span)]) for span in eduspans]
        del _text, _sentences, sids, sentspans, eduspans

        discourse = Discourse(label, text, edus, sentences, info)
        relations = {}
        for r in p.find("Relations"):  # type: ElementTree.Element
            rid = r.get("ID")
            prid = r.get("ParentId")
            parent = relations[prid] if prid in relations else None
            rspans = self._xml2pos(r.get("SentencePosition"))
            rspan = rspans[0][0], rspans[-1][1]
            nuclear = nuclear_map[r.get("Center")]

            relation_explicit = r.get("ConnectiveType") == "显式关系"
            if relation_explicit:
                relation_connective = r.get("Connective").split("…")
                relation_connectivespan = self._xml2pos(r.get("ConnectivePosition"))
            else:
                relation_connective = None
                relation_connectivespan = None
            relation_finetype = r.get("RelationType")
            relation_coarsetype = coarsemap[relation_finetype]
            relation_type = Relation(explicit=relation_explicit,
                                     connective=relation_connective, connective_span=relation_connectivespan,
                                     fine=relation_finetype, coarse=relation_coarsetype)
            relation = discourse.add_relation(rspan, nuclear, relation=relation_type, parent=parent)
            relations[rid] = relation
        return discourse

    def _xml2pos(self, s):
        """
        1…26|27…89 -> [(0, 26), (26, 89)]
        """
        s = s.strip()
        if not s:
            return []
        s = s.split("|")
        spans = [span.split("…") for span in s]
        spans = [(int(s) - 1, int(e)) for s, e in spans]
        return spans
