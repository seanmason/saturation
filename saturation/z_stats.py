import numpy as np


def calculate_z_statistic(mean_nnd: float,
                          nobs: int,
                          area: float) -> float:
    """
    Calculates the Z statistic defined by Clark and Evans (1954).
    :param mean_nnd: The mean nearest neighbor distance for all craters in the record that fall in the study region.
    :param nobs: Number of craters in the record that fall in the study region.
    :param area: The area of the study region.
    """
    if nobs == 0:
        return np.nan

    sigma = 0.26136 / np.sqrt(nobs**2 / area)
    nn_exp = 0.5 / np.sqrt(nobs / area)

    return (mean_nnd - nn_exp) / sigma


def calculate_za_statistic(mean_nnd: float,
                           nobs: int,
                           area_covered: float,
                           area: float) -> float:
    """
    Calculates the Za statistic defined by Squyres et al. (1998)
    :param mean_nnd: The mean nearest neighbor distance for all craters in the record that fall in the study region.
    :param nobs: Number of craters in the record that fall in the study region.
    :param area_covered: The area of the study region covered.
    :param area: The size of the study region.
    """
    if nobs == 0:
        return np.nan

    try:
        sigma = 0.26136 / np.sqrt(nobs**2 / (area - area_covered))
        nn_exp = 0.5 / np.sqrt(nobs / (area - area_covered))

        return (mean_nnd - nn_exp) / sigma
    except:
        return np.nan
