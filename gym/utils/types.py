from typing import \
    Tuple, \
    List
from enum import Enum

ThreeIntTuple = Tuple[int, int, int]
TwoIntTuple = Tuple[int, int]
TwoIntTupleList = List[TwoIntTuple]

TwoDArray = List[List[float]]


class AgentType(Enum):
    DRONE = 0
    ACTOR = 1
