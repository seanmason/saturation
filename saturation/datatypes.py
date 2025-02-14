import numpy as np

# We explicitly do not use the numba utils here.
import numba as nb

from typing import Tuple

# Type definitions
# Arc in radians
Arc = Tuple[float, float]

# (x, y) location
Location = Tuple[float, float]


spec = [
    ('id', nb.int64),
    ('x', nb.float64),
    ('y', nb.float64),
    ('radius', nb.float64),
]
@nb.experimental.jitclass(spec)
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

CraterType = nb.typeof(Crater(np.int64(1), np.float64(1.0), np.float64(1.0), np.float64(1.0)))
