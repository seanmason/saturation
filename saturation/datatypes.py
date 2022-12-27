from dataclasses import dataclass
from typing import Tuple


# Type definitions
# Arc in radians
Arc = Tuple[float, float]

# (x, y) location
Location = Tuple[float, float]


@dataclass(frozen=True, kw_only=True, slots=True)
class Crater:
    id: int
    x: float
    y: float
    radius: float

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other) -> bool:
        return other.id == self.id
