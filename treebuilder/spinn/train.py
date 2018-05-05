# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 训练 SPINN 篇章结构生成模型
"""
from structure.tree import Discourse, EDU, RelationNode
from transition import Session
from transition.shiftreduce import SRConfiguration, SRTransition
import dataset
import logging


def oracle(discorse: Discourse):
    for node in discorse.binarize().traverse():
        if isinstance(node, EDU):
            yield SRTransition.SHIFT
        elif isinstance(node, RelationNode):
            yield SRTransition.REDUCE


def main():
    cdtb = dataset.load_cdtb_by_config()
    transition = SRTransition()
    for discourse in cdtb.train:
        conf = SRConfiguration(discourse.strip())
        session = Session(conf, transition, history=True)
        for action in oracle(discourse):
            session(action)
        print(session.memory.actions)
        session.current.discourse.tree().draw()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
