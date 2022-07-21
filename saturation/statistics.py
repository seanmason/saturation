import numpy as np


def calculate_z_statistic(nearest_neighbor_distances: np.array, terrain_size: float) -> float:
    """
    Calculates the Z statistic defined by Clark and Evans (1954).
    :param nearest_neighbor_distances: The nearest neighbor distances for all craters in the record
                                       that fall in the terrain limits.
    :param terrain_size: The size of the terrain.
    """
    n_craters = nearest_neighbor_distances.shape[0]
    if n_craters == 0:
        return np.nan

    area = terrain_size**2
    sigma = 0.26136 / np.sqrt(n_craters**2 / area)
    nn_exp = 0.5 / np.sqrt(n_craters / area)
    nn_obs = nearest_neighbor_distances.mean()

    return (nn_obs - nn_exp) / sigma


def calculate_za_statistic(nearest_neighbor_distances: np.array,
                           terrain_area_covered: float,
                           terrain_size: float) -> float:
    """
    Calculates the Za statistic defined by Squyres et al. (1998)
    :param nearest_neighbor_distances: The nearest neighbor distances for all craters in the record
                                       that fall in the terrain limits.
    :param terrain_area_covered: The area of the terrain covered.
    :param terrain_size: The size of the terrain.
    """
    n_craters = nearest_neighbor_distances.shape[0]
    if n_craters == 0:
        return np.nan

    area = terrain_size**2
    sigma = 0.26136 / np.sqrt(n_craters**2 / (area - terrain_area_covered))
    nn_exp = 0.5 / np.sqrt(n_craters / (area - terrain_area_covered))
    nn_obs = nearest_neighbor_distances.mean()

    return (nn_obs - nn_exp) / sigma
