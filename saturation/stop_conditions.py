from abc import ABC, abstractmethod
from typing import List

from saturation.writers import StatisticsRow


class StopCondition(ABC):
    @abstractmethod
    def should_stop(self, statistics_rows: List[StatisticsRow]) -> bool:
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

    def should_stop(self, statistics_rows: List[StatisticsRow]) -> bool:
        return statistics_rows and statistics_rows[-1].n_craters_added_in_study_region == self._n_craters


class CraterCountAndArealDensityStopCondition(StopCondition):
    """
    Stops the simulation when no new maximum crater count and areal density have been reached in one third of
    the simulation iterations.
    """
    MIN_CRATERS = 5000

    def should_stop(self, statistics_rows: List[StatisticsRow]) -> bool:
        n_rows = len(statistics_rows)
        if n_rows < self.MIN_CRATERS:
            return False

        checkpoint = n_rows // 3 * 2
        max_areal_density_before_checkpoint = max((x.areal_density for x in statistics_rows[:checkpoint]))
        max_areal_density_after_checkpoint = max((x.areal_density for x in statistics_rows[checkpoint:]))
        max_n_craters_before_checkpoint = max((x.n_craters_in_study_region for x in statistics_rows[:checkpoint]))
        max_n_craters_after_checkpoint = max((x.n_craters_in_study_region for x in statistics_rows[checkpoint:]))
        return max_areal_density_before_checkpoint >= max_areal_density_after_checkpoint and \
               max_n_craters_before_checkpoint >= max_n_craters_after_checkpoint
