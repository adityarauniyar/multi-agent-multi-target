from typing import \
    Tuple, \
    List
from enum import Enum

ThreeIntTuple = Tuple[int, int, int]

TwoIntTupleList = List[Tuple[int, int]]

TwoDArray = List[List[float]]


class AgentType(Enum):
    DRONE = 0
    ACTOR = 1
