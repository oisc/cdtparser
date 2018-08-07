# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 篇章解析
"""
from argparse import ArgumentParser
import os
from dataset import CDTB, ErrorEmptyDiscourse
import schemas
import config
import multiprocessing
from interface import SegmentError, ParseError
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discourse parsing")


pipeline = None


class ParseWorker:
    def __init__(self, schema_name, work_dir, encoding):
        self.schema_name = schema_name
        self.work_dir = work_dir
        self.encoding = encoding
        self.pipeline = None

    def init(self):
        global pipeline
        pipeline = schemas.create_pipeline(self.schema_name)

    def __call__(self, file):
        global pipeline

        discourses = []
        with open(os.path.join(self.work_dir, file), "r", encoding=self.encoding) as source_fd:
            for line in source_fd:
                line = line.strip()
                if line:
                    label, start, end, text = line.split("\t")
                    start, end = int(start), int(end)
                    try:
                        if end - start > 3000:
                            logger.error("%s has more than 3000 characters, ignored" % label)
                            raise SegmentError("refuse segmenting %s" % label, label, text, start, end, file)
                        discourse = pipeline(label, text, start, end)
                    except SegmentError as e:
                        logger.error("error segmenting %s, discourse will be replace with ErrorEmptyDiscourse"
                                     % str(label))
                        discourse = ErrorEmptyDiscourse(e.label, e.text, e.start, e.end, e.info)
                    except ParseError as e:
                        logger.error("error parsing %s, imcomplete discourse may be saved" % label)
                        discourse = e.last_discourse
                    logger.info("parsing file: \"%s\" label: \"%s\" finished" % (file, str(discourse.label)))
                    discourses.append(discourse)
        return file, discourses


def main(args):
    worker = ParseWorker(args.schema, args.source, args.encoding)
    pool = multiprocessing.Pool(args.jobs, initializer=worker.init)
    all_files = os.listdir(args.source)

    for i, (file, discourses) in enumerate(pool.imap_unordered(worker, all_files), start=1):
        save_file = os.path.join(args.save, file + '.xml')

        logger.info("[%d/%d] parsing %s finished, saved to %s" % file, save_file)
        CDTB.save_xml(discourses, save_file, encoding=args.encoding)


if __name__ == '__main__':
    argparser = ArgumentParser()
    default_schema = config.get("global", "default_schema")
    argparser.add_argument('-source', required=True)
    argparser.add_argument('-save', required=True)
    argparser.add_argument('--schema', default=default_schema)
    argparser.add_argument('--encoding', default="utf-8")
    argparser.add_argument('--jobs', type=int, default=4)
    main(argparser.parse_args())
