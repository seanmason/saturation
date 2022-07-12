from typing import Tuple, Iterable, Callable

import pandas as pd
import numpy as np

from saturation.distributions import ProbabilityDistribution

# Type definitions
LocationFunc = Callable[[int], np.array]


def get_crater_locations(n_craters: int) -> np.array:
    """
    Returns n_craters crater locations, uniformly distributed on [0, 1]
    """
    return np.random.rand(n_craters, 2)


def get_craters(n_craters: int,
                size_distribution: ProbabilityDistribution,
                scale: float,
                location_func: LocationFunc = get_crater_locations) -> pd.DataFrame:
    """
    Returns a dataframe of n_craters, including (x, y) center locations and radii.
    Scale defines the maximum size of the terrain.
    """
    ids = np.arange(1, n_craters + 1)
    locations = location_func(n_craters) * scale
    radii = [size_distribution.uniform_to_value(x) for x in np.random.rand(n_craters)]
    data_dict = {
        'id': ids,
        'x': locations[:, 0],
        'y': locations[:, 1],
        'radius': radii
    }
    data = pd.DataFrame(data_dict).set_index(['id'])

    return data
