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
from itertools import chain
from typing import List
from xml.etree import ElementTree
from nltk.tree import ParentedTree
from structure.tree import EDU, Sentence, Relation, Discourse
from util import ZhDefaultParser
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
        self.ctb = self.load_ctb(ctb, ctb_encoding) if ctb else None
        self.parser = ZhDefaultParser()
        self.train = []  # type: List[Discourse]
        self.test = []  # type: List[Discourse]
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
                logger.info("load cached CDTB dataset from %s" % cache_key)
                with open(cache_file, "rb") as cache_fd:
                    self.train = pickle.load(cache_fd)
                    self.test = pickle.load(cache_fd)
                return

        # load train set
        logger.info("loading train set")
        self.train.extend(self.load(self.train_path, self.cdtb_encoding))

        # load test set
        logger.info("loading test set")
        self.test.extend(self.load(self.test_path, self.cdtb_encoding))

        # add syntactic information
        logger.info("add syntactic parse to sentence")
        if not self.ctb:
            logger.info("Notice! CTB is not given, %s parser will be used to generate extra syntacitcal information. "
                        "This may take a while." % self.parser.name)
        for discourse in tqdm.tqdm(chain(self.train, self.test)):
            for sentence in discourse.sentences:
                sentence.set(parse=self.lookup_parse(sentence))

        # save as cache file
        if self.cache_dir:
            logger.info("save cached CDTB dataset to %s" % cache_key)
            with open(os.path.join(self.cache_dir, cache_key), "wb+") as cache_fd:
                pickle.dump(self.train, cache_fd)
                pickle.dump(self.test, cache_fd)

    @staticmethod
    def load_ctb(ctb_path, encoding="utf-8"):
        ctb = {}
        for file in os.listdir(ctb_path):
            with open(os.path.join(ctb_path, file), "r", encoding=encoding) as fd:
                for sid, parse in ctb_pat.findall(fd.read()):
                    ctb[sid] = ParentedTree.fromstring(parse)
        return ctb

    def lookup_parse(self, sentence):
        if sentence.sid and self.ctb and sentence.sid in self.ctb:
            return self.ctb[sentence.sid]
        else:
            return self.parser.parse(sentence.text)

    @staticmethod
    def load(path, encoding="utf-8"):
        for file in tqdm.tqdm(os.listdir(path)):
            yield from CDTB.load_xml(os.path.join(path, file), encoding)

    @staticmethod
    def load_xml(file, encoding="utf-8"):
        with open(file, "r", encoding=encoding) as fd:
            dom = ElementTree.fromstring(fd.read())  # type: ElementTree.Element
            for p in dom.iterfind("P"):  # type: ElementTree.Element
                pid = int(p.get("ID"))
                # 忽略没有关系的段落
                if pid > 0:
                    yield CDTB._dom2discourse(p, pid, os.path.dirname(file))

    @staticmethod
    def _dom2discourse(p, label, info):
        raw = p.find("RAW")  # type: ElementTree.Element
        _text = raw.get("Sentence")
        sentspans = CDTB._xml2pos(raw.get("SentencePosition"))
        _sentences = []
        _offset = 0
        for s, e in sentspans:
            _sentences.append(_text[s+_offset:e+_offset])
            _offset += 1
        text = ''.join(_sentences)
        sids = raw.get("SID").split("|") if raw.get("SID") else []
        sentences = []
        for i, span in enumerate(sentspans):
            sid = sids[i] if sids else None
            sentences.append(Sentence(span, text[slice(*span)], sid=sid))
        eduspans = CDTB._xml2pos(raw.get("EduPosition"))
        edus = [EDU(span, text[slice(*span)]) for span in eduspans]
        del _text, _sentences, sids, sentspans, eduspans

        discourse_span = CDTB._xml2pos(raw.get("AnnotatedPosition"))[0]
        discourse = Discourse(label, text, discourse_span, edus, sentences, info)
        relations = {}
        for r in p.find("Relations"):  # type: ElementTree.Element
            rid = r.get("ID")
            prid = r.get("ParentId")
            parent = relations[prid] if prid in relations else None
            rspans = CDTB._xml2pos(r.get("SentencePosition"))
            rspan = rspans[0][0], rspans[-1][1]
            nuclear = nuclear_map[r.get("Center")]

            relation_explicit = r.get("ConnectiveType") == "显式关系"
            if relation_explicit:
                relation_connective = r.get("Connective").split("…")
                relation_connectivespan = CDTB._xml2pos(r.get("ConnectivePosition"))
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

    @staticmethod
    def _xml2pos(s):
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
