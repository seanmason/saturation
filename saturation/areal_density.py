from typing import Tuple, List
from collections import OrderedDict

import numpy as np
import numba as nb
from numba.experimental import jitclass

from saturation.datatypes import Crater

spec = OrderedDict({
    "_study_region_size": nb.types.int64,
    "_study_region_padding": nb.types.int64,
    "_rstat": nb.types.float64,
    "_study_region": nb.types.uint8[:, :],
    "_total_study_region_area": nb.types.int64,
    "_cratered_area": nb.types.int64,
})


@jitclass(spec=spec)
class ArealDensityCalculator(object):
    def __init__(
        self,
        study_region_size: int,
        study_region_padding: int,
        rstat: float
    ):
        self._study_region_size = study_region_size
        self._study_region_padding = study_region_padding
        self._rstat = rstat

        self._study_region = np.zeros((self._study_region_size, self._study_region_size), dtype='uint8')

        self._total_study_region_area = self._study_region_size ** 2
        self._cratered_area = 0

    def add_crater(self, new_crater: Crater):
        if new_crater.radius >= self._rstat and self._crater_is_in_study_region(new_crater):
            # Calculate the difference in the cratered area before and after crater addition.
            before = self._get_cratered_area(new_crater)
            self._increment_study_region(new_crater, 1)
            after = self._get_cratered_area(new_crater)

            self._cratered_area += after - before

    def _crater_is_in_study_region(self, new_crater: Crater):
        return (
            self._study_region_padding <= new_crater.x <= self._study_region_size + self._study_region_padding
            and self._study_region_padding <= new_crater.y <= self._study_region_size + self._study_region_padding
        )

    def remove_craters(self, new_erased_craters: List[Crater]):
        difference = 0

        for erased in new_erased_craters:
            if erased.radius >= self._rstat and self._crater_is_in_study_region(erased):
                # Calculate the difference in the cratered area before and after crater removal.
                before = self._get_cratered_area(erased)
                self._increment_study_region(erased, -1)
                after = self._get_cratered_area(erased)
                difference += after - before

        self._cratered_area += difference

    @property
    def areal_density(self) -> float:
        return self._cratered_area / self._total_study_region_area

    @property
    def area_covered(self) -> int:
        return self._cratered_area

    def _get_mins_and_maxes(self, crater: Crater) -> Tuple[int, int, int, int]:
        """
        Calculates min and max x and y square bounding a circle within a region bounded by
        [0, study_region_size] with a specified margin.
        """
        x_min = int(max(crater.x - crater.radius - 1, self._study_region_padding))
        x_max = int(min(crater.x + crater.radius + 1, self._study_region_size + self._study_region_padding - 1))
        y_min = int(max(crater.y - crater.radius - 1, self._study_region_padding))
        y_max = int(min(crater.y + crater.radius + 1, self._study_region_size + self._study_region_padding - 1))

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
                    x = test_x - self._study_region_padding
                    y = test_y - self._study_region_padding
                    self._study_region[x, y] += increment

    def _get_cratered_area(self, crater: Crater) -> int:
        """
        Gets the total cratered area of the bounding rectangle for a specified circle.
        """
        x_min, x_max, y_min, y_max = self._get_mins_and_maxes(crater)
        return np.count_nonzero(self._study_region[x_min - self._study_region_padding:
                                                   x_max - self._study_region_padding + 1,
                                y_min - self._study_region_padding:
                                y_max - self._study_region_padding + 1])
