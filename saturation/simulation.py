from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


# Type definitions
Location = Tuple[float, float]

# Represents an arc of a crater rim, represented as a tuple of radians
RimSegment = Tuple[float, float]
RimSegments = List[RimSegment]


@dataclass
class Crater:
    """
    Attributes representing a 2D crater for use in simulations.
    Rim segments are represented in tuples of radians.
    """
    id: int
    location: Location
    radius: float
    rim_segments: RimSegments


def get_crater_locations(n_craters: int) -> np.array:
    return np.zeros(n_craters)
