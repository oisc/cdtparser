# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/6
@Description: 评测工具类
"""
from collections import defaultdict, Counter

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
            if not precisions:
                return 0
            else:
                return numpy.average(precisions).item()

    def recall(self):
        if not self.average:
            total = sum(self.gold)
            return float(sum(self.corr)) / float(total) if total else 0
        else:
            recalls = [float(corr) / float(gold) if gold else 0 for corr, gold in zip(self.corr, self.gold)]
            if not recalls:
                return 0
            else:
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
        self.nn_score = F1Score(average=False)
        self.ns_score = F1Score(average=False)
        self.sn_score = F1Score(average=False)
        self.fine_score = F1Score(tree_average)
        self.coarse_score = F1Score(tree_average)
        self.nuclear_fine_score = F1Score(tree_average)
        self.nuclear_coarse_score = F1Score(tree_average)
        self.fine_class_scores = defaultdict(lambda: F1Score(average=False))
        self.coarse_class_scores = defaultdict(lambda: F1Score(average=False))
        self.binarize = binarize
        self.left_heavy = left_heavy
        for gold, parse in zip(golds, parses):
            self.eval(gold, parse)

    def eval(self, gold: Discourse, parse: Discourse):
        if self.binarize:
            gold = gold.binarize(left_heavy=self.left_heavy)
            parse = parse.binarize(left_heavy=self.left_heavy)

        self._eval_sent(gold, parse)
        self._eval_edu(gold, parse)
        self._eval_span(gold, parse)
        self._eval_nuclear(gold, parse)
        self._eval_relation(gold, parse)
        self._eval_nuclear_classes(gold, parse)
        self._eval_relation_classes(gold, parse)

    def _eval_sent(self, gold, parse):
        gold_sent_spans = [sentence.span for sentence in gold.sentences
                           if gold.span[0] <= sentence.span[0] <= sentence.span[1] <= gold.span[1]]
        parse_sent_spans = [sentence.span for sentence in parse.sentences
                            if parse.span[0] <= sentence.span[0] <= sentence.span[1] <= parse.span[1]]
        corr_sent_spans = set(gold_sent_spans) & set(parse_sent_spans)
        self.sentence_score += len(corr_sent_spans), len(gold_sent_spans), len(parse_sent_spans)

    def _eval_edu(self, gold, parse):
        gold_edu_spans = [edu.span for edu in gold.edus
                          if gold.span[0] <= edu.span[0] <= edu.span[1] <= gold.span[1]]
        parse_edu_spans = [edu.span for edu in parse.edus
                           if parse.span[0] <= edu.span[0] <= edu.span[1] <= parse.span[1]]
        corr_edu_spans = set(gold_edu_spans) & set(parse_edu_spans)
        self.edu_score += len(corr_edu_spans), len(gold_edu_spans), len(parse_edu_spans)

    def _eval_span(self, gold, parse):
        gold_quads = self.factorize(gold)
        parse_quads = self.factorize(parse)
        correct_span = set(gold_quads.keys()) & set(parse_quads.keys())
        self.span_score += len(correct_span), len(gold_quads), len(parse_quads)

    def _eval_nuclear(self, gold, parse):
        gold_quads = self.factorize(gold)
        parse_quads = self.factorize(parse)
        correct_span = set(gold_quads.keys()) & set(parse_quads.keys())
        correct_nuclear = []
        for span in correct_span:
            gold_quad = gold_quads[span]
            parse_quad = parse_quads[span]
            if gold_quad[1] == parse_quad[1]:
                correct_nuclear.append(parse_quad)
        self.nuclear_score += len(correct_nuclear), len(gold_quads), len(parse_quads)

    def _eval_relation(self, gold, parse):
        gold_quads = self.factorize(gold)
        parse_quads = self.factorize(parse)
        correct_span = set(gold_quads.keys()) & set(parse_quads.keys())
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
            if gold_quad[2] == parse_quad[2]:
                correct_fine.append(parse_quad)
                if nuclear_flag:
                    correct_nuclear_fine.append(parse_quad)
            if gold_quad[3] == parse_quad[3]:
                correct_coarse.append(parse_quad)
                if nuclear_flag:
                    correct_nuclear_coarse.append(parse_quad)
        self.fine_score += len(correct_fine), len(gold_quads), len(parse_quads)
        self.coarse_score += len(correct_coarse), len(gold_quads), len(parse_quads)
        self.nuclear_fine_score += len(correct_nuclear_fine), len(gold_quads), len(parse_quads)
        self.nuclear_coarse_score += len(correct_nuclear_coarse), len(gold_quads), len(parse_quads)

    def _eval_nuclear_classes(self, gold, parse):
        gold_quads = self.factorize(gold)
        parse_quads = self.factorize(parse)
        correct_span = set(gold_quads.keys()) & set(parse_quads.keys())
        gold_ns = [quad for quad in gold_quads.values() if quad[1] == Discourse.NS]
        gold_sn = [quad for quad in gold_quads.values() if quad[1] == Discourse.SN]
        gold_nn = [quad for quad in gold_quads.values() if quad[1] == Discourse.NN]
        parse_ns = [quad for quad in parse_quads.values() if quad[1] == Discourse.NS]
        parse_sn = [quad for quad in parse_quads.values() if quad[1] == Discourse.SN]
        parse_nn = [quad for quad in parse_quads.values() if quad[1] == Discourse.NN]
        correct_ns = []
        correct_sn = []
        correct_nn = []
        for span in correct_span:
            gold_quad = gold_quads[span]
            parse_quad = parse_quads[span]
            if gold_quad[1] == parse_quad[1]:
                if gold_quad[1] == Discourse.NS:
                    correct_ns.append(gold_quad)
                elif gold_quad[1] == Discourse.SN:
                    correct_sn.append(gold_quad)
                else:
                    correct_nn.append(gold_quad)
        if len(gold_ns) + len(parse_ns):
            self.ns_score += len(correct_ns), len(gold_ns), len(parse_ns)
        if len(gold_sn) + len(parse_sn):
            self.sn_score += len(correct_sn), len(gold_sn), len(parse_sn)
        if len(gold_nn) + len(parse_nn):
            self.nn_score += len(correct_nn), len(gold_nn), len(parse_nn)

    def _eval_relation_classes(self, gold, parse):
        gold_quads = self.factorize(gold)
        parse_quads = self.factorize(parse)
        correct_span = set(gold_quads.keys()) & set(parse_quads.keys())
        gold_fine_counter = Counter([quad[2] for quad in gold_quads.values() if quad[2]])
        gold_coarse_counter = Counter([quad[3] for quad in gold_quads.values() if quad[3]])
        parse_fine_counter = Counter([quad[2] for quad in parse_quads.values() if quad[2]])
        parse_coarse_counter = Counter([quad[3] for quad in parse_quads.values() if quad[3]])
        correct_fine_counter = Counter()
        correct_coarse_counter = Counter()
        for span in correct_span:
            gold_quad = gold_quads[span]
            parse_quad = parse_quads[span]
            if gold_quad[2] == parse_quad[2]:
                correct_fine_counter.update([parse_quad[2]])
            if gold_quad[3] == parse_quad[3]:
                correct_coarse_counter.update([parse_quad[3]])
        for fine in set(gold_fine_counter.keys()) | set(parse_fine_counter.keys()):
            self.fine_class_scores[fine] += correct_fine_counter[fine], \
                                            gold_fine_counter[fine], \
                                            parse_fine_counter[fine]

        for coarse in set(gold_coarse_counter.keys()) | set(parse_coarse_counter.keys()):
            self.coarse_class_scores[coarse] += correct_coarse_counter[coarse],\
                                                gold_coarse_counter[coarse],\
                                                parse_coarse_counter[coarse]

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

    def nuclear_report(self):
        _report = '\n'
        _report += '          precision    recall    f1\n'
        _report += '--------------------------------------\n'
        _report += 'NS        %5.3f        %5.3f     %5.3f\n' % (self.ns_score.precision(),
                                                                 self.ns_score.recall(),
                                                                 self.ns_score.f1())
        _report += 'SN        %5.3f        %5.3f     %5.3f\n' % (self.sn_score.precision(),
                                                                 self.sn_score.recall(),
                                                                 self.sn_score.f1())
        _report += 'NN        %5.3f        %5.3f     %5.3f\n' % (self.nn_score.precision(),
                                                                 self.nn_score.recall(),
                                                                 self.nn_score.f1())
        return _report

    def relation_report(self):
        _report = '\n'
        _report += '                 precision    recall    f1\n'
        _report += '--------------------------------------------\n'
        for coarse, score in self.coarse_class_scores.items():
            _report += '%-5s         %5.3f        %5.3f     %5.3f\n' % (coarse,
                                                                        self.coarse_class_scores[coarse].precision(),
                                                                        self.coarse_class_scores[coarse].recall(),
                                                                        self.coarse_class_scores[coarse].f1())
        _report += '--------------------------------------------\n'
        for fine, score in self.fine_class_scores.items():
            _report += '%-5s         %5.3f        %5.3f     %5.3f\n' % (fine,
                                                                        self.fine_class_scores[fine].precision(),
                                                                        self.fine_class_scores[fine].recall(),
                                                                        self.fine_class_scores[fine].f1())

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
