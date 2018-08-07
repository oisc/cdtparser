# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/4
@Description: 
"""
import math
from collections import namedtuple
from copy import copy
from interface import Annotator
from transition import Session
from transition.shiftreduce import SRTransition, SRConfiguration


class SPINNTreeBuilder(Annotator):
    def __init__(self, model, beam_size=1):
        self.model = model
        self.model.eval()
        self.transition = SRTransition()
        self.beam_size = beam_size

    def annotate(self, discourse):
        conf = SRConfiguration(discourse)
        state = self.model.new_state(conf)
        BeamNode = namedtuple("BeamNode", "session cost")
        fringe = [BeamNode(Session(conf, self.transition, state=state), cost=0)]
        hypotheses = []

        next_fringe = []
        while fringe:
            for node in fringe:
                if node.session.terminate():
                    hypotheses.append(node)
                else:
                    valid_action = node.session.valid()
                    for (action, nuclear), prob in self.model.score(node.session.state).items():
                        if action in valid_action:
                            session = copy(node.session)
                            if action == SRTransition.SHIFT:
                                session(action)
                                session.state = self.model.shift(session.state)
                            else:
                                session(action, nuclear=nuclear)
                                session.state = self.model.reduce(session.state, nuclear)
                            cost = -math.log(prob)
                            next_fringe.append(BeamNode(session=session, cost=node.cost + cost))
            fringe = sorted(next_fringe, key=lambda n: n.cost)[:self.beam_size]
            next_fringe = []
        hypotheses.sort(key=lambda n: n.cost)
        high_rank_discourse = hypotheses[0].session.current.discourse
        return high_rank_discourse
