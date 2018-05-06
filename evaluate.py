# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/3
@Description: 评测脚本
"""
from argparse import ArgumentParser
import logging
from dataset import CDTB
from util.metrics import CDTBMetrics


def main(args):
    golds = list(CDTB.load(args.gold))
    parses = {d.label: d for d in CDTB.load(args.parse)}
    parses = [parses[g.label] for g in golds]
    metrics = CDTBMetrics(golds, parses)
    print(metrics.segmenter_report())
    print(metrics.parser_report())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    argparser = ArgumentParser()
    argparser.add_argument('-parse', required=True)
    argparser.add_argument('-gold', required=True)
    argparser.add_argument('--encoding', default="utf-8")
    main(argparser.parse_args())
