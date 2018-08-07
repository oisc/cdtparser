#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/3
@Description: 训练 SVM Segmenter 模型
"""
import pickle
import config
import dataset
import logging
from itertools import chain
from .model import SVMCommaClassifier
from sklearn.metrics import classification_report


logger = logging.getLogger(__name__)


def golden2sample(discourse):
    comma_candidate = config.get("segmenter.svm", "comma_candidate")
    # 分割两侧的位置偏移量
    seg_offsets = set()
    for edu in discourse.edus:
        seg_offsets.add(edu.span[0])
        seg_offsets.add(edu.span[1] - 1)

    for sentence in discourse.sentences:
        parse = sentence.parse()
        offset = sentence.offset
        for word in sentence.words():
            if word in comma_candidate:
                if offset in seg_offsets:
                    yield offset-sentence.offset, parse, True
                else:
                    yield offset-sentence.offset, parse, False
            offset += len(word)


def train(discourses):
    # load connective clues
    clues_file = config.get("segmenter.svm", "clues")
    clues_encoding = config.get("segmenter.svm", "clues_encoding")

    logger.info("load connective clues from %s" % clues_file)
    with open(clues_file, "r", encoding=clues_encoding) as clues_fd:
        connectives = [line.strip() for line in clues_fd]
    # model
    random_seed = config.get("segmenter.svm", "seed", defult=21, rtype=int)
    model = SVMCommaClassifier(connectives, random_seed)

    # get train samples from golden discourse
    # [(comma_pos, parse), (comma_pos, parse), ...]
    samples = chain(*[golden2sample(discourse) for discourse in discourses])
    feats = []
    labels = []
    pos_count = neg_count = 0
    for comma_pos, parse, label in samples:
        feats.append(model.extract_features(comma_pos, parse))
        labels.append(label)
        if label:
            pos_count += 1
        else:
            neg_count += 1
    logger.info("get %d samples, %d positive, %d negtive" % (len(feats), pos_count, neg_count))
    x = model.fet_vector.fit_transform(feats)
    logger.info("sample's matrix shape %s" % str(x.shape))
    logger.info("start trainig svm segmenter model")
    model.clf.fit(x, labels)
    logger.info("training finished!")
    return model


def save(model, model_dir):
    logger.info("save model to %s" % model_dir)
    with open(model_dir, "wb+") as model_fd:
        pickle.dump(model, model_fd)


def evaluate(discourses, model):
    samples = chain(*[golden2sample(discourse) for discourse in discourses])
    x = []
    y = []
    for comma_pos, parse, label in samples:
        x.append((comma_pos, parse))
        y.append(label)
    predict = model.predict_many(x)
    print("evaluation score:")
    print(classification_report(y, predict))


def main():
    cdtb = dataset.load_cdtb_by_config()
    model = train(cdtb.train + cdtb.validate)
    model_dir = config.get("segmenter.svm", "model_dir")
    save(model, model_dir)
    evaluate(cdtb.test, model)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
