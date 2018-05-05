# -*- coding: utf-8 -*-

from .cdtb import CDTB


def load_cdtb_by_config():
    import config
    cdtb_train = config.get("CDTB", "train")
    cdtb_test = config.get("CDTB", "test")
    cdtb_encoding = config.get("CDTB", "encoding", defult="utf-8")
    ctb = config.get("CDTB", "ctb", defult=None)
    ctb_encoding = config.get("CDTB", "ctb_encoding", defult="utf-8")
    cache_dir = config.get("CDTB", "cache", defult=None)
    return CDTB(cdtb_train, cdtb_test, ctb, cache_dir=cache_dir, encoding=cdtb_encoding, ctb_encoding=ctb_encoding)
