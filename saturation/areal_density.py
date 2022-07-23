from typing import Tuple, List

import numpy as np
from numba import njit

from saturation.datatypes import Crater


@njit()
def _get_mins_and_maxes(x: float,
                        y: float,
                        radius: float,
                        observed_terrain_size: int,
                        terrain_padding: int) -> Tuple[int, int, int, int]:
    """
    Calculates min and max x and y square bounding a circle within a terrain bounded by
    [0, observed_terrain_size] with a specified margin.
    """
    x_min = int(max(x - radius - 1, terrain_padding))
    x_max = int(min(x + radius + 1, observed_terrain_size + terrain_padding - 1))
    y_min = int(max(y - radius - 1, terrain_padding))
    y_max = int(min(y + radius + 1, observed_terrain_size + terrain_padding - 1))

    return x_min, x_max, y_min, y_max


@njit()
def _increment_terrain(x: float,
                       y: float,
                       radius: float,
                       increment: int,
                       terrain: np.array,
                       terrain_padding: int):
    """
    Increments points in the terrain for the placement of a specified circle.
    """
    x_min, x_max, y_min, y_max = _get_mins_and_maxes(x, y, radius, terrain.shape[0], terrain_padding)
    limit = radius ** 2

    for test_x in range(x_min, x_max + 1):
        for test_y in range(y_min, y_max + 1):
            if (test_x - x) ** 2 + (test_y - y) ** 2 <= limit:
                terrain[test_x - terrain_padding, test_y - terrain_padding] += increment


@njit()
def _get_cratered_area(x: float,
                       y: float,
                       radius: float,
                       terrain: np.array,
                       terrain_padding: int) -> int:
    """
    Gets the total cratered area of the bounding rectangle for a specified circle.
    """
    x_min, x_max, y_min, y_max = _get_mins_and_maxes(x, y, radius, terrain.shape[0], terrain_padding)
    return np.count_nonzero(terrain[x_min - terrain_padding:x_max - terrain_padding + 1,
                            y_min - terrain_padding:y_max - terrain_padding + 1])


class ArealDensityCalculator(object):
    def __init__(self, observed_terrain_size: int, terrain_padding: int):
        self._observed_terrain_size = observed_terrain_size
        self._terrain_padding = terrain_padding

        self._terrain = np.zeros((observed_terrain_size, observed_terrain_size), dtype='uint8')

        self._total_area = self._observed_terrain_size * self._observed_terrain_size
        self._cratered_area = 0

    def add_crater(self, new_crater: Crater):
        # Calculate the difference in the cratered area before and after crater addition.
        before = _get_cratered_area(new_crater.x, new_crater.y, new_crater.radius, self._terrain, self._terrain_padding)
        _increment_terrain(new_crater.x, new_crater.y, new_crater.radius, 1, self._terrain, self._terrain_padding)
        after = _get_cratered_area(new_crater.x, new_crater.y, new_crater.radius, self._terrain, self._terrain_padding)

        self._cratered_area += after - before

    def remove_craters(self, new_erased_craters: List[Crater]):
        difference = 0

        for erased in new_erased_craters:
            # Calculate the difference in the cratered area before and after crater removal.
            before = _get_cratered_area(erased.x, erased.y, erased.radius, self._terrain, self._terrain_padding)
            _increment_terrain(erased.x, erased.y, erased.radius, -1, self._terrain, self._terrain_padding)
            after = _get_cratered_area(erased.x, erased.y, erased.radius, self._terrain, self._terrain_padding)
            difference += after - before

        self._cratered_area += difference

    @property
    def areal_density(self) -> float:
        return self._cratered_area / self._total_area

    @property
    def area_covered(self) -> float:
        return self._cratered_area
