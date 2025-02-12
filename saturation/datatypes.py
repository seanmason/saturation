import numpy as np
import numba as nb

from typing import Tuple, Dict, NamedTuple

# Type definitions
# Arc in radians
Arc = Tuple[float, float]

# (x, y) location
Location = Tuple[float, float]


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


CraterType = nb.typeof(Crater(np.int64(1), np.float32(1.0), np.float32(1.0), np.float32(1.0)))
