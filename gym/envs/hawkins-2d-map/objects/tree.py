import sys
from enum import Enum


class LeavesColor(Enum):
    NO_LEAVES = 0
    GREEN = 5
    WHITE = 6
    ORANGE = 7
    YELLOW = 4


class LookScore(Enum):
    NOT_GOOD = 0
    GOOD = 1
    WOW = 3


class Tree:
    def __init__(self, location, height, width_span, has_leaves, leaves_color, look_score):
        """
        :param location:
        :param height:
        :param width_span:
        :param has_leaves:
        :param leaves_color:
        :param look_score:
        """
        self.location = location
        self.height = height
        self.width_span = width_span
        self.has_leaves = has_leaves
        self.leaves_color = leaves_color
        self.look_score = look_score

        self.aesthetic_score = self.get_aesthetic_score()

    def get_aesthetic_score(self):
        # Determine if the tree is aesthetically pleasing based on its features
        # checks if the tree is tall enough and has leaves
        return self.look_score
