from abc import ABC
from typing import List, Any
import numpy as np


class BeamState(object):
    def __init__(self):
        self.score_vector: np.ndarray = None


class BeamNode(object):

    def __init__(self, current_choice, previous_choices: List, state: BeamState, depth = 0):
        self.choice = current_choice
        self.previous_choices = previous_choices
        self.all_choices = set(previous_choices)
        self.all_choices.add(current_choice)

        self.state = state
        self.depth = depth


class BeamChoiceFunction(ABC):
    def choose(self, node: BeamNode, max_candidates=3) -> List[BeamNode]:
        return []


class BeamSearcher(object):
    nodes: List[BeamNode]

    def __init__(self, choice_fun, state_fun, beam_size=20):
        self.choice_fun = choice_fun
        self.state_fun = state_fun
        self.beam_size = beam_size
        self.nodes = []
