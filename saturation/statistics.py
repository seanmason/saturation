from typing import List

import numpy as np


def calculate_z_statistic(nearest_neighbor_distances: List[float],
                          area: float) -> float:
    """
    Calculates the Z statistic defined by Clark and Evans (1954).
    :param nearest_neighbor_distances: The nearest neighbor distances for all craters in the record
                                       that fall in the study region.
    :param area: The area of the study region.
    """
    n_craters = len(nearest_neighbor_distances)
    if n_craters == 0:
        return np.nan

    sigma = 0.26136 / np.sqrt(n_craters**2 / area)
    nn_exp = 0.5 / np.sqrt(n_craters / area)
    nn_obs = np.mean(nearest_neighbor_distances)

    return (nn_obs - nn_exp) / sigma


def calculate_za_statistic(nearest_neighbor_distances: List[float],
                           area_covered: float,
                           area: float) -> float:
    """
    Calculates the Za statistic defined by Squyres et al. (1998)
    :param nearest_neighbor_distances: The nearest neighbor distances for all craters in the record
                                       that fall in the study region.
    :param area_covered: The area of the study region covered.
    :param area: The size of the study region.
    """
    n_craters = len(nearest_neighbor_distances)
    if n_craters == 0:
        return np.nan

    sigma = 0.26136 / np.sqrt(n_craters**2 / (area - area_covered))
    nn_exp = 0.5 / np.sqrt(n_craters / (area - area_covered))
    nn_obs = np.mean(nearest_neighbor_distances)

    return (nn_obs - nn_exp) / sigma
