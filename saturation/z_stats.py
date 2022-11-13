import numpy as np


def calculate_z_statistic(mean_nearest_neighbor_distance: float,
                          n_craters: int,
                          area: float) -> float:
    """
    Calculates the Z statistic defined by Clark and Evans (1954).
    :param mean_nearest_neighbor_distance: The mean nearest neighbor distance for all craters in the record
                                       that fall in the study region.
    :param n_craters: Number of craters in the record that fall in the study region.
    :param area: The area of the study region.
    """
    if n_craters == 0:
        return np.nan

    sigma = 0.26136 / np.sqrt(n_craters**2 / area)
    nn_exp = 0.5 / np.sqrt(n_craters / area)

    return (mean_nearest_neighbor_distance - nn_exp) / sigma


def calculate_za_statistic(mean_nearest_neighbor_distance: float,
                           n_craters: int,
                           area_covered: float,
                           area: float) -> float:
    """
    Calculates the Za statistic defined by Squyres et al. (1998)
    :param mean_nearest_neighbor_distance: The mean nearest neighbor distance for all craters in the record
                                       that fall in the study region.
    :param n_craters: Number of craters in the record that fall in the study region.
    :param area_covered: The area of the study region covered.
    :param area: The size of the study region.
    """
    if n_craters == 0:
        return np.nan

    sigma = 0.26136 / np.sqrt(n_craters**2 / (area - area_covered))
    nn_exp = 0.5 / np.sqrt(n_craters / (area - area_covered))

    return (mean_nearest_neighbor_distance - nn_exp) / sigma
