# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 
"""
from interface import Annotator
from transition import Session
from transition.shiftreduce import SRTransition, SRConfiguration


class SPINNTreeBuilder(Annotator):
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
            action, nuclear = max(score_valid, key=score_valid.get)
            if action == SRTransition.SHIFT:
                session(action)
                session.state = self.model.shift(session.state)
            else:
                session(action, nuclear=nuclear)
                session.state = self.model.reduce(session.state)
        return session.current.discourse
