# -*- coding: utf-8 -*-

from .cdtb import CDTB, ErrorEmptyDiscourse


def load_cdtb_by_config():
    import config
    cdtb_train = config.get("CDTB", "train")
    cdtb_validate = config.get("CDTB", "validate")
    cdtb_test = config.get("CDTB", "test")
    cdtb_encoding = config.get("CDTB", "encoding", defult="utf-8")
    ctb = config.get("CDTB", "ctb", defult=None)
    ctb_encoding = config.get("CDTB", "ctb_encoding", defult="utf-8")
    edu_lexicalize = config.get("CDTB", "edu_lexicalize", defult=False, rtype=bool)
    cache_dir = config.get("CDTB", "cache", defult=None)
    return CDTB(cdtb_train, cdtb_validate, cdtb_test, ctb,
                cache_dir=cache_dir, encoding=cdtb_encoding, ctb_encoding=ctb_encoding)
