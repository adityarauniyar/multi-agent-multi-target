from enum import Enum


class Color(Enum):
    NO_COLOR = 0
    GREEN = 1
    WHITE = 2
    ORANGE = 3
    YELLOW = 4


class LookScore(Enum):
    NOT_GOOD = 0
    GOOD = 1
    WOW = 3
