from typing import Tuple, Iterable, Callable, Set

import pandas as pd
import numpy as np

from saturation.distributions import ProbabilityDistribution

# Type definitions
from saturation.geometry import calculate_rim_percentage_remaining

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
    radii = [size_distribution.inverse_cdf(x) for x in np.random.rand(n_craters)]
    data_dict = {
        'id': ids,
        'x': locations[:, 0],
        'y': locations[:, 1],
        'radius': radii
    }
    data = pd.DataFrame(data_dict).set_index(['id'])

    return data


def get_craters_to_remove(erased_rim_arcs: pd.DataFrame, minimum_rim_percentage: float) -> Set[int]:
    """
    Given a set of craters, the corresponding rim arcs that have been erased, and a minimum rim percentage,
    returns the set of craters that should be removed from the record.
    """
    result = set()

    groups = erased_rim_arcs.groupby(['old_id'])
    for group in groups:
        erased_arcs_list = [(x.theta1, x.theta2) for x in group[1].itertuples()]
        percent_remaining = calculate_rim_percentage_remaining(erased_arcs_list)
        if percent_remaining < minimum_rim_percentage:
            result.add(group[0])

    return result


def run_simulation(n_craters: int,
                   size_distribution: ProbabilityDistribution,
                   max_crater_size: float,
                   terrain_size: int,
                   output_filename: str):
    """
    Runs a simulation.

    :param n_craters: Number of craters to generate.
    :param size_distribution: Probability distribution for crater sizes.
    :param max_crater_size: Maximum crater size allowed.
    :param terrain_size: Size of the terrain.
    :param output_filename: Output filename.
    """
    pass