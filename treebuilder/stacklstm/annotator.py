# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/8
@Description: 
"""
from interface import Annotator
from transition import Session
from transition.shiftreduce import SRTransition, SRConfiguration
from structure.tree import Relation


class StackLSTMTreeBuilder(Annotator):
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.transition = SRTransition()

    def annotate(self, discourse):
        conf = SRConfiguration(discourse)
        state = self.model.new_state(conf)
        session = Session(conf, self.transition, history=True, state=state)
        while not session.terminate():
            valid = session.valid()
            score = self.model.score(session.state)
            score_valid = {k: v for k, v in score.items() if k[0] in valid}
            action, nuclear, fine = max(score_valid, key=score_valid.get)
            if action == SRTransition.SHIFT:
                session(action)
            else:
                relation = Relation(fine=fine)
                session(action, nuclear=nuclear, relation=relation)
            session.state = self.model.update(session.state, action, nuclear, fine)
        return session.current.discourse
