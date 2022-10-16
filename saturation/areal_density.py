from typing import Tuple, List

import numpy as np
from numba import njit

from saturation.datatypes import Crater


@njit()
def _get_mins_and_maxes(x: float,
                        y: float,
                        radius: float,
                        study_region_size: int,
                        study_region_padding: int) -> Tuple[int, int, int, int]:
    """
    Calculates min and max x and y square bounding a circle within a region bounded by
    [0, study_region_size] with a specified margin.
    """
    x_min = int(max(x - radius - 1, study_region_padding))
    x_max = int(min(x + radius + 1, study_region_size + study_region_padding - 1))
    y_min = int(max(y - radius - 1, study_region_padding))
    y_max = int(min(y + radius + 1, study_region_size + study_region_padding - 1))

    return x_min, x_max, y_min, y_max


@njit()
def _increment_study_region(x: float,
                            y: float,
                            radius: float,
                            increment: int,
                            study_region: np.array,
                            study_region_padding: int):
    """
    Increments points in the study region for the placement of a specified circle.
    """
    x_min, x_max, y_min, y_max = _get_mins_and_maxes(x, y, radius, study_region.shape[0], study_region_padding)

    limit = radius ** 2

    for test_x in range(x_min, x_max + 1):
        for test_y in range(y_min, y_max + 1):
            if (test_x - x) ** 2 + (test_y - y) ** 2 <= limit:
                study_region[test_x - study_region_padding, test_y - study_region_padding] += increment


@njit()
def _get_cratered_area(x: float,
                       y: float,
                       radius: float,
                       study_region: np.array,
                       study_region_padding: int) -> int:
    """
    Gets the total cratered area of the bounding rectangle for a specified circle.
    """
    x_min, x_max, y_min, y_max = _get_mins_and_maxes(x, y, radius, study_region.shape[0], study_region_padding)
    return np.count_nonzero(study_region[x_min - study_region_padding:x_max - study_region_padding + 1,
                            y_min - study_region_padding:y_max - study_region_padding + 1])


class ArealDensityCalculator(object):
    def __init__(self, study_region_size: int, study_region_padding: int, r_stat: float):
        self._study_region_size = study_region_size
        self._study_region_padding = study_region_padding
        self._r_stat = r_stat

        self._study_region = np.zeros((study_region_size, study_region_size), dtype='uint8')

        self._total_study_region_area = self._study_region_size ** 2
        self._cratered_area = 0

    def add_crater(self, new_crater: Crater):
        if new_crater.radius >= self._r_stat \
                and self._study_region_padding <= new_crater.x <= self._study_region_size + self._study_region_padding \
                and self._study_region_padding <= new_crater.y <= self._study_region_size + self._study_region_padding:
            # Calculate the difference in the cratered area before and after crater addition.
            before = _get_cratered_area(new_crater.x,
                                        new_crater.y,
                                        new_crater.radius,
                                        self._study_region,
                                        self._study_region_padding)
            _increment_study_region(new_crater.x,
                                    new_crater.y,
                                    new_crater.radius,
                                    1,
                                    self._study_region,
                                    self._study_region_padding)
            after = _get_cratered_area(new_crater.x, new_crater.y, new_crater.radius, self._study_region,
                                       self._study_region_padding)

            self._cratered_area += after - before

    def remove_craters(self, new_erased_craters: List[Crater]):
        difference = 0

        for erased in new_erased_craters:
            if erased.radius >= self._r_stat \
                    and self._study_region_padding <= erased.x <= self._study_region_size + self._study_region_padding \
                    and self._study_region_padding <= erased.y <= self._study_region_size + self._study_region_padding:
                # Calculate the difference in the cratered area before and after crater removal.
                before = _get_cratered_area(erased.x,
                                            erased.y,
                                            erased.radius,
                                            self._study_region,
                                            self._study_region_padding)
                _increment_study_region(erased.x,
                                        erased.y,
                                        erased.radius,
                                        -1,
                                        self._study_region,
                                        self._study_region_padding)
                after = _get_cratered_area(erased.x,
                                           erased.y,
                                           erased.radius,
                                           self._study_region,
                                           self._study_region_padding)
                difference += after - before

        self._cratered_area += difference

    @property
    def areal_density(self) -> float:
        return self._cratered_area / self._total_study_region_area

    @property
    def area_covered(self) -> float:
        return self._cratered_area
