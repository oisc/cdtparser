# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 篇章解析
"""
from argparse import ArgumentParser
import dataset
import schemas


def main(args):
    pipeline = schemas.create(args.schema)
    cdtb = dataset.load_cdtb_by_config()
    for discourse in cdtb.test:
        print(discourse.span)
        new_discourse = pipeline(discourse.label, discourse.text, discourse.span[0], discourse.span[1])
        print(new_discourse.span)
        print(new_discourse.text)
        for edu in new_discourse.edus:
            print(edu.span, edu.text)
        print()


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-schema', required=True)
    main(argparser.parse_args())
