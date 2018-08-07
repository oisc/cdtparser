# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/2
@Description: CDTB 数据集工具
"""
import os
import pickle
import re
from tqdm import tqdm
from typing import List
from xml.etree import ElementTree
from lxml import etree as et
from nltk.tree import ParentedTree
from structure.tree import EDU, Sentence, Relation, Discourse, RelationNode
import multiprocessing
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
nuclear_map_rev = {v: k for k, v in nuclear_map.items()}


worker_ctb = None
worker_parser = None


def worker_initializer(ctb=None):
    global worker_ctb, worker_parser
    worker_ctb = ctb
    worker_parser = ZhDefaultParser()


def worker_fn(discourse):
    global worker_ctb, worker_parser
    for sentence in discourse.sentences:
        if sentence.sid and worker_ctb and sentence.sid in worker_ctb:
            sentence.set(parse=worker_ctb[sentence.sid])
        else:
            sentence.set(parse=worker_parser.parse(sentence.text))
    return discourse


class ErrorEmptyDiscourse:
    def __init__(self, label, text, start, end, info):
        self.label = label
        self.text = text
        self.span = start, end
        self.info = info


class CDTB:
    """
    CDTB 数据集工具类
    """
    def __init__(self, train, validate, test, ctb=None, cache_dir=None, encoding="utf-8", ctb_encoding="utf-8",
                 threads=2):
        """
        :param train: 训练集路径
        :param test: 测试集路径
        :param ctb: CTB 路径，如果不给出，使用自动句法解析器补充句法信息
        :param cache_dir: 缓存路径，如果不给出，则不缓存
        :param encoding: CDTB 编码
        :param ctb_encoding: CTB 编码
        """
        self.train_path = train
        self.validate_path = validate
        self.test_path = test
        self.ctb_encoding = ctb_encoding
        self.ctb = self.load_ctb(ctb, ctb_encoding) if ctb else None
        self.parser = ZhDefaultParser()
        self.train = []  # type: List[Discourse]
        self.validate = []  # type: List[Discourse]
        self.test = []  # type: List[Discourse]
        self.cache_dir = cache_dir
        self.cdtb_encoding = encoding
        self.threads = threads
        self.build()

    def build(self):
        # cache key
        cache_key = "CDTB_cache_%s.pickle" % "ctb" if self.ctb else self.parser.name
        if self.cache_dir:
            # load from cache if exists
            cache_file = os.path.join(self.cache_dir, cache_key)
            if os.path.exists(cache_file):
                logger.info("load cached CDTB dataset from %s" % cache_key)
                with open(cache_file, "rb") as cache_fd:
                    self.train = pickle.load(cache_fd)
                    self.validate = pickle.load(cache_fd)
                    self.test = pickle.load(cache_fd)
                return

        # load train set
        logger.info("loading train set")
        train = list(self.load(self.train_path, self.cdtb_encoding))
        # load validate set
        logger.info("loading validate set")
        validate = list(self.load(self.validate_path, self.cdtb_encoding))
        # load test set
        logger.info("loading test set")
        test = list(self.load(self.test_path, self.cdtb_encoding))

        # add syntactic information
        logger.info("add syntactic information to discourse for further training")
        if not self.ctb:
            logger.info("Notice! CTB is not given, %s parser will be used to generate extra syntacitcal information. "
                        % self.parser.name)
        logger.info("This may take a while")
        process_pool = multiprocessing.Pool(self.threads, initializer=worker_initializer, initargs={"ctb": self.ctb})
        self.train = list(tqdm(
            process_pool.imap(worker_fn, train),
            desc="processing train set", total=len(train)))
        self.validate = list(tqdm(
            process_pool.imap(worker_fn, validate),
            desc="processing validate set", total=len(validate)
        ))
        self.test = list(tqdm(
            process_pool.imap(worker_fn, test),
            desc="processing test set", total=len(test)
        ))

        # save as cache file
        if self.cache_dir:
            logger.info("save cached CDTB dataset to %s" % cache_key)
            with open(os.path.join(self.cache_dir, cache_key), "wb+") as cache_fd:
                pickle.dump(self.train, cache_fd)
                pickle.dump(self.validate, cache_fd)
                pickle.dump(self.test, cache_fd)

    @staticmethod
    def load_ctb(ctb_path, encoding="utf-8"):
        ctb = {}
        for file in os.listdir(ctb_path):
            with open(os.path.join(ctb_path, file), "r", encoding=encoding) as fd:
                for sid, parse in ctb_pat.findall(fd.read()):
                    ctb[sid] = ParentedTree.fromstring(parse)
        return ctb

    @staticmethod
    def load(path, encoding="utf-8"):
        for file in tqdm(os.listdir(path)):
            yield from CDTB.load_xml(os.path.join(path, file), encoding)

    @staticmethod
    def save_xml(discourses, path, encoding="utf-8"):
        doc = et.Element("DOC")  # type: et.Element
        for i, discourse in enumerate(discourses):
            if isinstance(discourse, Discourse):
                p = CDTB._discourse2dom(discourse)
            else:
                p = CDTB._error2dom(discourse)
            p.attrib["Order"] = str(i + 1)
            doc.append(p)
        with open(path, "w+", encoding=encoding) as dom_df:
            dom_df.write(et.tostring(doc, pretty_print=True, encoding=encoding).decode(encoding))

    @staticmethod
    def _error2dom(discourse: ErrorEmptyDiscourse):
        p = et.Element("P")
        p.attrib["ID"] = "-" + str(discourse.label)
        raw = et.Element("RAW")
        raw.attrib["Sentence"] = discourse.text
        raw.attrib["AnnotatedPosition"] = CDTB._pos2xml([discourse.span])
        raw.attrib["SentencePosition"] = ""
        raw.attrib["EduPosition"] = ""
        raw.attrib["SID"] = ""
        raw.attrib["ROOT"] = ""
        p.append(raw)
        return p

    @staticmethod
    def _discourse2dom(discourse: Discourse):
        p = et.Element("P")
        p.attrib["ID"] = str(discourse.label)
        raw = et.Element("RAW")
        sent_spans = [sentence.span for sentence in discourse.sentences]
        if sent_spans[0][0] != 0:
            sent_spans = [(0, sent_spans[0][0])] + sent_spans
        if sent_spans[-1][1] != len(discourse.text):
            sent_spans = sent_spans + [(sent_spans[-1][1], len(discourse.text))]
        raw.attrib["AnnotatedPosition"] = CDTB._pos2xml([discourse.span])
        raw.attrib["SentencePosition"] = CDTB._pos2xml(sent_spans)
        raw.attrib["SID"] = ""
        edu_spans = [edu.span for edu in discourse.edus]
        raw.attrib["EduPosition"] = CDTB._pos2xml(edu_spans)
        raw.attrib["Sentence"] = '|'.join([discourse.text[slice(*span)] for span in sent_spans])
        if discourse.complete():
            root = discourse.tree()
            root_id = discourse.index(root)
            raw.attrib["ROOT"] = str(root_id)
        else:
            raw.attrib["ROOT"] = ""
        relations = et.Element("Relations")
        relations.attrib["Nodes"] = str(len(discourse.relations))
        if discourse.complete():
            relations.attrib["Height"] = str(discourse.tree().height() - 2)
        else:
            relations.attrib["Height"] = ""
        for relation in discourse.relations:
            r = et.Element("R")
            r.attrib["ID"] = str(discourse.index(relation))
            if relation.nuclear:
                r.attrib["Center"] = nuclear_map_rev[relation.nuclear]
            else:
                r.attrib["Center"] = ""
            relations.append(r)
            child_ids = [discourse.index(child) for child in relation if isinstance(child, RelationNode)]
            child_spans = [child.span for child in relation]
            r.attrib["ChildList"] = "|".join(map(str, child_ids))
            r.attrib["SentencePosition"] = CDTB._pos2xml(child_spans)
            if relation.parent():
                r.attrib["ParentId"] = str(discourse.index(relation.parent()))
            else:
                r.attrib["ParentId"] = "-1"
            r.attrib["Sentence"] = "|".join([discourse.text[slice(*span)] for span in child_spans])

            # relation type
            relation_type = relation.relation
            if relation_type:
                explicit, fine, coarse, connective, connective_span = relation_type.explicit, \
                                                                      relation_type.fine, \
                                                                      relation_type.coarse, \
                                                                      relation_type.connective, \
                                                                      relation_type.connective_span
            else:
                explicit, fine, coarse, connective, connective_span = None, None, None, None, None
            if explicit is not None:
                r.attrib["ConnectiveType"] = "显式关系" if explicit else "隐式关系"
            else:
                r.attrib["ConnectiveType"] = ""
            r.attrib["RelationType"] = fine or ""
            if coarse:
                r.attrib["CoarseRelationType"] = coarse
            elif fine:
                r.attrib["CoarseRelationType"] = coarsemap[fine]
            else:
                r.attrib["CoarseRelationType"] = ""
            r.attrib["Connective"] = "…".join(connective) if connective and explicit else ""
            r.attrib["ConnectivePosition"] = CDTB._pos2xml(connective_span) if explicit and connective_span else ""

        p.append(raw)
        p.append(relations)
        return p

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
            nuclear = nuclear_map[r.get("Center")] if r.get("Center") else None

            if not r.get("ConnectiveType"):
                relation_explicit = None
            elif r.get("ConnectiveType") == "显式关系":
                relation_explicit = True
            else:
                relation_explicit = False

            if relation_explicit:
                relation_connective = r.get("Connective").split("…")
                relation_connectivespan = CDTB._xml2pos(r.get("ConnectivePosition"))
            else:
                relation_connective = None
                relation_connectivespan = None
            relation_finetype = r.get("RelationType") or None
            if "CoarseRelationType" in r.attrib and r.get("CoarseRelationType"):
                relation_coarsetype = r.get("CoarseRelationType")
            elif relation_finetype:
                relation_coarsetype = coarsemap[relation_finetype]
            else:
                relation_coarsetype = None
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

    @staticmethod
    def _pos2xml(spans):
        _s = ""
        _s += "|".join(["%d…%d" % (s+1, e) for s, e in spans])
        return _s
