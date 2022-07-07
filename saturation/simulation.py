from dataclasses import dataclass
from typing import Tuple, Iterable, Iterator, Callable

import numpy as np

# Type definitions
from saturation.distributions import ProbabilityDistribution

Location = Tuple[float, float]
LocationFunc = Callable[[], Location]

# Represents an arc of a crater rim, represented as a tuple of radians
RimSegment = Tuple[float, float]
RimSegments = Iterable[RimSegment]


@dataclass(frozen=True, kw_only=True)
class Crater:
    """
    Attributes representing a 2D crater for use in simulations.
    Rim segments are represented in tuples of radians.
    """
    id: int
    location: Location
    radius: float
    rim_segments: RimSegments


def get_crater_location() -> Location:
    """
    Returns a uniform random crater location with (x, y) in [0, 1]
    """
    r = np.random.rand(2)
    return r[0], r[1]


def get_craters(size_distribution: ProbabilityDistribution,
                location_func: LocationFunc = get_crater_location) -> Iterator[Crater]:
    """
    Returns an infinite iterator of craters. Uses the supplied distribution to generate crater radii.
    """
    crater_id = 1
    while True:
        yield Crater(id=crater_id,
                     location=location_func(),
                     radius=size_distribution.uniform_to_value(np.random.random()),
                     rim_segments=[(0, 2 * np.pi)])
        crater_id += 1
