from typing import Tuple

import pandas as pd
import numpy as np
from numba import njit


@njit()
def _get_mins_and_maxes(x: float,
                        y: float,
                        radius: float,
                        margin: int,
                        limited_terrain_size: int) -> Tuple[int, int, int, int]:
    """
    Calculates min and max x and y square bounding a circle within a terrain bounded by
    [0, limited_terrain_size] with a specified margin.
    """
    x_min = int(max(x - radius, margin))
    x_max = int(min(x + radius, limited_terrain_size - 1))
    y_min = int(max(y - radius, margin))
    y_max = int(min(y + radius, limited_terrain_size - 1))

    return x_min, x_max, y_min, y_max


@njit()
def _increment_terrain(x: float,
                       y: float,
                       radius: float,
                       increment: int,
                       terrain: np.array,
                       margin: int,
                       limited_terrain_size: int):
    """
    Increments points in the terrain for the placement of a specified circle.
    """
    x_min, x_max, y_min, y_max = _get_mins_and_maxes(x, y, radius, margin, limited_terrain_size)
    limit = int(radius ** 2)

    for test_x in range(x_min, x_max + 1):
        for test_y in range(y_min, y_max + 1):
            if (test_x - x) ** 2 + (test_y - y) ** 2 <= limit:
                terrain[test_x - margin, test_y - margin] += increment


@njit()
def _get_cratered_area(x: float,
                       y: float,
                       radius: float,
                       terrain: np.array,
                       margin: int,
                       limited_terrain_size: int) -> int:
    """
    Gets the total cratered area of the bounding rectangle for a specified circle.
    """
    x_min, x_max, y_min, y_max = _get_mins_and_maxes(x, y, radius, margin, limited_terrain_size)
    return np.count_nonzero(terrain[x_min - margin:x_max - margin + 1,
                            y_min - margin:y_max - margin + 1])


@njit()
def _update(new_craters: np.array,
            new_erased_craters: np.array,
            terrain: np.array,
            margin: int,
            limited_terrain_size: int) -> int:
    """
    Updates the terrain, adding and removing specified craters.
    """
    difference = 0

    for x, y, radius in new_craters:
        # Calculate the difference in the cratered area before and after crater addition.
        before = _get_cratered_area(x, y, radius, terrain, margin, limited_terrain_size)
        _increment_terrain(x, y, radius, 1, terrain, margin, limited_terrain_size)
        after = _get_cratered_area(x, y, radius, terrain, margin, limited_terrain_size)
        difference += after - before

    for x, y, radius in new_erased_craters:
        # Calculate the difference in the cratered area before and after crater removal.
        before = _get_cratered_area(x, y, radius, terrain, margin, limited_terrain_size)
        _increment_terrain(x, y, radius, -1, terrain, margin, limited_terrain_size)
        after = _get_cratered_area(x, y, radius, terrain, margin, limited_terrain_size)
        difference += after - before

    return difference


class ArealDensityCalculator(object):
    def __init__(self, terrain_size: int, margin: int):
        self._terrain_size = terrain_size
        self._margin = margin

        self._terrain = np.zeros((terrain_size - 2 * margin, terrain_size - 2 * margin), dtype='uint16')
        self._limited_terrain_size = self._terrain.shape[0]

        self._total_area = self._limited_terrain_size * self._limited_terrain_size
        self._cratered_area = 0

    def update(self, new_craters: pd.DataFrame, new_erased_craters: pd.DataFrame):
        difference = _update(new_craters[['x', 'y', 'radius']].values.astype('float64'),
                             new_erased_craters[['x', 'y', 'radius']].values.astype('float64'),
                             self._terrain,
                             self._margin,
                             self._limited_terrain_size)
        self._cratered_area += difference

    def get_areal_density(self):
        return self._cratered_area / self._total_area
