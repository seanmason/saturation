from dataclasses import dataclass
from typing import Tuple

from sortedcontainers import SortedKeyList


# Type definitions
# Arc in radians
Arc = Tuple[float, float]

# (x, y) location
Location = Tuple[float, float]


@dataclass(frozen=True, kw_only=True)
class Crater:
    id: int
    x: float
    y: float
    radius: float


class SortedArcList(SortedKeyList[Arc]):
    def __init__(self, iterable=None):
        super(SortedArcList, self).__init__(iterable=iterable, key=lambda x: x[0])

    def reverse(self):
        super().reverse()

    def append(self, value):
        super().append(value)

    def extend(self, values):
        super().extend(values)

    def insert(self, index, value):
        super().insert(index, value)
