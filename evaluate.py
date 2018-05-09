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


def evaluate(args):
    golds = list(dataset.CDTB.load(args.gold))
    parses = {d.label: d for d in dataset.CDTB.load(args.parse)}
    parses = [parses[g.label] for g in golds]
    metrics = CDTBMetrics(golds, parses)
    print(metrics.segmenter_report())
    print(metrics.parser_report())
    print(metrics.nuclear_report())
    print(metrics.relation_report())


def evaluate_gold_edu():
    schema_name = "baseline"
    pipeline = schemas.create_pipeline(schema_name)
    cdtb = dataset.load_cdtb_by_config()
    golds = cdtb.test
    parses = []
    for discourse in tqdm(cdtb.test):
        parse = pipeline.annotate(discourse.strip())
        parses.append(parse)
    metrics = CDTBMetrics(golds, parses)
    print(metrics.segmenter_report())
    print(metrics.parser_report())
    print(metrics.nuclear_report())
    print(metrics.relation_report())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    argparser = ArgumentParser()
    argparser.add_argument('-parse', required=True)
    argparser.add_argument('-gold', required=True)
    argparser.add_argument('--encoding', default="utf-8")
    evaluate(argparser.parse_args())
    # evaluate_gold_edu()
