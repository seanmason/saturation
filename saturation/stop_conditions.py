from abc import ABC, abstractmethod
from typing import Dict

from saturation.writers import StatisticsRow


class StopCondition(ABC):
    @abstractmethod
    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        """
        Returns True if the simulation should stop.
        """
        pass


class NCratersStopCondition(StopCondition):
    """
    Stops the simulation when a specified number of craters have been added to the study region.
    """

    def __init__(self, n_craters: int):
        self._n_craters = n_craters
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1
        return self._counter == self._n_craters


class CraterCountAndArealDensityStopCondition(StopCondition):
    """
    Stops the simulation when no new maximum crater count and areal density have been reached in one third of
    the simulation iterations.
    """
    MIN_CRATERS = 15000

    def __init__(self):
        self._areal_density_high_points: Dict[int, float] = {0: 0.0}
        self._craters_in_study_region_high_points: Dict[int, int] = {0: 0}
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1

        self._areal_density_high_points[self._counter] = max(
            self._areal_density_high_points[self._counter - 1],
            statistics_row.areal_density
        )
        self._craters_in_study_region_high_points[self._counter] = max(
            self._craters_in_study_region_high_points[self._counter - 1],
            statistics_row.n_craters_in_study_region
        )

        if self._counter < self.MIN_CRATERS:
            return False

        checkpoint = self._counter // 3 * 2
        max_areal_density_before_checkpoint = self._areal_density_high_points[checkpoint]
        max_areal_density_after_checkpoint = self._areal_density_high_points[self._counter]
        max_n_craters_before_checkpoint = self._craters_in_study_region_high_points[checkpoint]
        max_n_craters_after_checkpoint = self._craters_in_study_region_high_points[self._counter]

        return max_areal_density_before_checkpoint >= max_areal_density_after_checkpoint \
               and max_n_craters_before_checkpoint >= max_n_craters_after_checkpoint
