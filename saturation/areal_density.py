from typing import Tuple, List
from collections import OrderedDict

import numpy as np
import numba as nb
from numba.experimental import jitclass

from saturation.datatypes import Crater

spec = OrderedDict({
    "_study_region_size": nb.types.UniTuple(nb.types.int64, 2),
    "_study_region_padding": nb.types.UniTuple(nb.types.int64, 2),
    "_r_stat": nb.types.float32,
    "_study_region": nb.types.uint8[:, :],
    "_total_study_region_area": nb.types.int64,
    "_cratered_area": nb.types.int64[:],
})


@jitclass(spec=spec)
class ArealDensityCalculator(object):
    def __init__(self,
                 study_region_size: Tuple[int, int],
                 study_region_padding: Tuple[int, int],
                 r_stat: float):
        self._study_region_size = study_region_size
        self._study_region_padding = study_region_padding
        self._r_stat = r_stat

        self._study_region = np.zeros((self._study_region_size[0], self._study_region_size[1]), dtype='uint8')

        self._total_study_region_area = self._study_region_size[0] * self._study_region_size[1]
        self._cratered_area = np.zeros(3, dtype="int64")

    def add_crater(self, new_crater: Crater):
        if new_crater.radius >= self._r_stat \
                and self._study_region_padding[0] <= new_crater.x <= self._study_region_size[0] + \
                self._study_region_padding[0] \
                and self._study_region_padding[1] <= new_crater.y <= self._study_region_size[1] + \
                self._study_region_padding[1]:
            # Calculate the difference in the cratered area before and after crater addition.
            before = self._get_cratered_area(new_crater)
            self._increment_study_region(new_crater, 1)
            after = self._get_cratered_area(new_crater)

            self._cratered_area += after - before

    def remove_craters(self, new_erased_craters: List[Crater]):
        difference = np.zeros(3, dtype="int64")

        for erased in new_erased_craters:
            if erased.radius >= self._r_stat \
                    and self._study_region_padding[0] <= erased.x <= self._study_region_size[0] + \
                    self._study_region_padding[0] \
                    and self._study_region_padding[1] <= erased.y <= self._study_region_size[1] + \
                    self._study_region_padding[1]:
                # Calculate the difference in the cratered area before and after crater removal.
                before = self._get_cratered_area(erased)
                self._increment_study_region(erased, -1)
                after = self._get_cratered_area(erased)
                difference += after - before

        self._cratered_area += difference

    @property
    def areal_density(self) -> float:
        return self._cratered_area[0] / self._total_study_region_area

    @property
    def areal_density_overlap_2(self) -> float:
        return self._cratered_area[1] / self._total_study_region_area

    @property
    def areal_density_overlap_3(self) -> float:
        return self._cratered_area[2] / self._total_study_region_area

    @property
    def area_covered(self) -> float:
        return self._cratered_area[0]

    def _get_mins_and_maxes(self, crater: Crater) -> Tuple[int, int, int, int]:
        """
        Calculates min and max x and y square bounding a circle within a region bounded by
        [0, study_region_size] with a specified margin.
        """
        x_min = int(max(crater.x - crater.radius - 1, self._study_region_padding[0]))
        x_max = int(min(crater.x + crater.radius + 1, self._study_region_size[0] + self._study_region_padding[0] - 1))
        y_min = int(max(crater.y - crater.radius - 1, self._study_region_padding[1]))
        y_max = int(min(crater.y + crater.radius + 1, self._study_region_size[1] + self._study_region_padding[1] - 1))

        return x_min, x_max, y_min, y_max

    def _increment_study_region(self, crater: Crater, increment: int):
        """
        Increments points in the study region for the placement of a specified circle.
        """
        x_min, x_max, y_min, y_max = self._get_mins_and_maxes(crater)

        limit = crater.radius ** 2

        for test_x in range(x_min, x_max + 1):
            for test_y in range(y_min, y_max + 1):
                if (test_x - crater.x) ** 2 + (test_y - crater.y) ** 2 <= limit:
                    x = test_x - self._study_region_padding[0]
                    y = test_y - self._study_region_padding[1]
                    self._study_region[x, y] += increment

    def _get_cratered_area(self, crater: Crater) -> np.array:
        """
        Gets the total cratered area of the bounding rectangle for a specified circle.
        Returns an array with three entries for the areas covered by 0, 1, and 2 overlapping craters.
        """
        result = np.zeros(3, dtype="int64")

        x_min, x_max, y_min, y_max = self._get_mins_and_maxes(crater)
        for x in range(x_min - self._study_region_padding[0], x_max - self._study_region_padding[0] + 1):
            for y in range(y_min - self._study_region_padding[1], y_max - self._study_region_padding[1] + 1):
                cell = min(self._study_region[x, y], 3)
                for c in range(1, cell + 1):
                    result[c - 1] += 1

        return result
