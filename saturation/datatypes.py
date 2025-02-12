import numpy as np
import numba as nb
from numba.experimental import jitclass

from typing import Tuple, Dict, NamedTuple

# Type definitions
# Arc in radians
Arc = Tuple[float, float]

# (x, y) location
Location = Tuple[float, float]


spec = [
    ('id', nb.int64),
    ('x', nb.float32),
    ('y', nb.float32),
    ('radius', nb.float32),
]
@jitclass(spec)
class Crater(object):
    def __init__(self, id, x, y, radius):
        self.id = id
        self.x = x
        self.y = y
        self.radius = radius

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
        }

CraterType = nb.typeof(Crater(np.int64(1), np.float32(1.0), np.float32(1.0), np.float32(1.0)))
