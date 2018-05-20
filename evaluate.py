# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/3
@Description: 评测脚本
"""
from argparse import ArgumentParser
import logging
from tqdm import tqdm
import dataset
import schemas
from util.metrics import CDTBMetrics


def evaluate(schema_name, use_gold_edu=False):
    pipeline = schemas.create_pipeline(schema_name)
    cdtb = dataset.load_cdtb_by_config()
    parses = []
    if use_gold_edu:
        print("parsing with gold EDU")
    else:
        print("parsing with auto discourse segmenter")
    for gold in tqdm(cdtb.test, desc="parsing for evaluation"):
        if use_gold_edu:
            discourse = pipeline.annotate(gold.strip())
        else:
            discourse = pipeline(gold.label, gold.text, gold.span[0], gold.span[1], gold.info)
        parses.append(discourse)
    metrics = CDTBMetrics(golds=cdtb.test, parses=parses)
    if not use_gold_edu:
        print(metrics.segmenter_report())
    print(metrics.parser_report())
    print(metrics.nuclear_report())
    print(metrics.relation_report())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    argparser = ArgumentParser()
    argparser.add_argument('-schema_name', required=True)
    argparser.add_argument('--use_gold_edu', dest='use_gold_edu', action='store_true')
    argparser.set_defaults(use_gold_edu=False)
    args = argparser.parse_args()
    evaluate(args.schema_name, args.use_gold_edu)
