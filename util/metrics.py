# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/6
@Description: 评测工具类
"""
import numpy
from structure.tree import Discourse


class F1Score:
    def __init__(self, average=True):
        self.average = average
        self.corr = []
        self.gold = []
        self.parse = []

    def precision(self):
        if not self.average:
            total = sum(self.parse)
            return float(sum(self.corr)) / float(total) if total else 0
        else:
            precisions = [float(corr) / float(parse) if parse else 0 for corr, parse in zip(self.corr, self.parse)]
            return numpy.average(precisions).item()

    def recall(self):
        if not self.average:
            total = sum(self.gold)
            return float(sum(self.corr)) / float(total) if total else 0
        else:
            recalls = [float(corr) / float(gold) if gold else 0 for corr, gold in zip(self.corr, self.gold)]
            return numpy.average(recalls).item()

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        if precision + recall:
            return 2 * precision * recall / (precision + recall)
        else:
            return 0

    def __add__(self, other):
        corr, gold, parse = other
        self.corr.append(corr)
        self.gold.append(gold)
        self.parse.append(parse)
        return self


class CDTBMetrics:
    def __init__(self, golds, parses, tree_average=True, binarize=True, left_heavy=False):
        self.sentence_score = F1Score(tree_average)
        self.edu_score = F1Score(tree_average)
        self.span_score = F1Score(tree_average)
        self.nuclear_score = F1Score(tree_average)
        self.fine_score = F1Score(tree_average)
        self.coarse_score = F1Score(tree_average)
        self.nuclear_fine_score = F1Score(tree_average)
        self.nuclear_coarse_score = F1Score(tree_average)
        self.binarize = binarize
        self.left_heavy = left_heavy
        for gold, parse in zip(golds, parses):
            self.eval(gold, parse)

    def eval(self, gold: Discourse, parse: Discourse):
        if self.binarize:
            gold = gold.binarize(left_heavy=self.left_heavy)
            parse = parse.binarize(left_heavy=self.left_heavy)

        gold_sent_spans = [sentence.span for sentence in gold.sentences
                           if gold.span[0] <= sentence.span[0] <= sentence.span[1] <= gold.span[1]]
        parse_sent_spans = [sentence.span for sentence in parse.sentences
                            if parse.span[0] <= sentence.span[0] <= sentence.span[1] <= parse.span[1]]
        corr_sent_spans = set(gold_sent_spans) & set(parse_sent_spans)
        self.sentence_score += len(corr_sent_spans), len(gold_sent_spans), len(parse_sent_spans)

        gold_edu_spans = [edu.span for edu in gold.edus
                          if gold.span[0] <= edu.span[0] <= edu.span[1] <= gold.span[1]]
        parse_edu_spans = [edu.span for edu in parse.edus
                           if parse.span[0] <= edu.span[0] <= edu.span[1] <= parse.span[1]]
        corr_edu_spans = set(gold_edu_spans) & set(parse_edu_spans)
        self.edu_score += len(corr_edu_spans), len(gold_edu_spans), len(parse_edu_spans)

        gold_quads = self.factorize(gold)
        parse_quads = self.factorize(parse)
        correct_span = set(gold_quads.keys()) & set(parse_quads.keys())
        correct_nuclear = []
        correct_fine = []
        correct_coarse = []
        correct_nuclear_fine = []
        correct_nuclear_coarse = []
        for span in correct_span:
            gold_quad = gold_quads[span]
            parse_quad = parse_quads[span]
            nuclear_flag = False
            if gold_quad[1] == parse_quad[1]:
                nuclear_flag = True
                correct_nuclear.append(parse_quad)
            if gold_quad[2] == parse_quad[2]:
                correct_fine.append(parse_quad)
                if nuclear_flag:
                    correct_nuclear_fine.append(parse_quad)
            if gold_quad[3] == parse_quad[3]:
                correct_coarse.append(parse_quad)
                if nuclear_flag:
                    correct_nuclear_coarse.append(parse_quad)
        self.span_score += len(correct_span), len(gold_quads), len(parse_quads)
        self.nuclear_score += len(correct_nuclear), len(gold_quads), len(parse_quads)
        self.fine_score += len(correct_fine), len(gold_quads), len(parse_quads)
        self.coarse_score += len(correct_coarse), len(gold_quads), len(parse_quads)
        self.nuclear_fine_score += len(correct_nuclear_fine), len(gold_quads), len(parse_quads)
        self.nuclear_coarse_score += len(correct_nuclear_coarse), len(gold_quads), len(parse_quads)

    def factorize(self, discourse: Discourse):
        quad = {}
        for relation in discourse.relations:
            if len(relation):
                span = tuple(child.span for child in relation)
                nuclear = relation.nuclear
                fine = relation.relation.fine if relation.relation else None
                coarse = relation.relation.coarse if relation.relation else None
                quad[span] = (span, nuclear, fine, coarse)
        return quad

    def segmenter_report(self):
        _report = '\n'
        _report += '          precision    recall    f1\n'
        _report += '--------------------------------------\n'
        _report += 'sentence  %5.3f        %5.3f     %5.3f\n' % (self.sentence_score.precision(),
                                                                 self.sentence_score.recall(),
                                                                 self.sentence_score.f1())
        _report += 'edu       %5.3f        %5.3f     %5.3f\n' % (self.edu_score.precision(),
                                                                 self.edu_score.recall(),
                                                                 self.edu_score.f1())
        return _report

    def parser_report(self):
        _report = '\n'
        _report += '                 precision    recall    f1\n'
        _report += '---------------------------------------------\n'
        _report += 'span             %5.3f        %5.3f     %5.3f\n' % (
                                                                self.span_score.precision(),
                                                                self.span_score.recall(),
                                                                self.span_score.f1())
        _report += 'nuclear          %5.3f        %5.3f     %5.3f\n' % (
                                                                self.nuclear_score.precision(),
                                                                self.nuclear_score.recall(),
                                                                self.nuclear_score.f1())
        _report += 'fine             %5.3f        %5.3f     %5.3f\n' % (
                                                                self.fine_score.precision(),
                                                                self.fine_score.recall(),
                                                                self.fine_score.f1())
        _report += 'coarse           %5.3f        %5.3f     %5.3f\n' % (
                                                                self.coarse_score.precision(),
                                                                self.coarse_score.recall(),
                                                                self.coarse_score.f1())
        _report += 'nuclear+fine     %5.3f        %5.3f     %5.3f\n' % (
                                                                self.nuclear_fine_score.precision(),
                                                                self.nuclear_fine_score.recall(),
                                                                self.nuclear_fine_score.f1())
        _report += 'nuclear+coarse   %5.3f        %5.3f     %5.3f\n' % (
                                                                self.nuclear_coarse_score.precision(),
                                                                self.nuclear_coarse_score.recall(),
                                                                self.nuclear_coarse_score.f1())
        return _report
