from typing import Tuple, NamedTuple

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
        return self.id

    def __eq__(self, other) -> bool:
        return other.id == self.id
