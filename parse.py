# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 篇章解析
"""
from argparse import ArgumentParser
import os
from dataset import CDTB
from util.metrics import CDTBMetrics
from tqdm import tqdm
import dataset
import schemas


def main(args):
    pipeline = schemas.create(args.schema)
    for file in tqdm(os.listdir(args.source)):
        discourses = []
        with open(os.path.join(args.source, file), "r", encoding=args.encoding) as source_fd:
            for line in source_fd:
                line = line.strip()
                if line:
                    label, start, end, text = line.split("\t")
                    parse = pipeline(label, text, int(start), int(end))
                    discourses.append(parse)
        CDTB.save_xml(discourses, os.path.join(args.save, file + '.xml'), encoding=args.encoding)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-schema', required=True)
    argparser.add_argument('-source', required=True)
    argparser.add_argument('-save', required=True)
    argparser.add_argument('--encoding', default="utf-8")
    main(argparser.parse_args())
