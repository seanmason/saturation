from typing import Tuple, Dict, NamedTuple
from collections import OrderedDict

import numba as nb
from numba.experimental import jitclass

# Type definitions
# Arc in radians
Arc = Tuple[float, float]

# (x, y) location
Location = Tuple[float, float]

spec = OrderedDict({
    "id": nb.types.int64,
    "x": nb.types.float32,
    "y": nb.types.float32,
    "radius": nb.types.float32,
})


class Crater(NamedTuple):
    id: int
    x: float
    y: float
    radius: float

    def __hash__(self) -> int:
        return int(self.id)

    def __eq__(self, other) -> bool:
        return other.id == self.id

    def to_dict(self) -> Dict:
        return dict(
            [
                ("id", self.id),
                ("x", self.x),
                ("y", self.y),
                ("radius", self.radius)
            ]
        )
