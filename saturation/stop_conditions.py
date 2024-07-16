from abc import ABC, abstractmethod
from typing import Dict

from saturation.writers import StatisticsRow


def get_stop_condition(stop_condition_config: Dict):
    name = stop_condition_config["name"]
    if name == "crater_count_and_areal_density":
        return CraterCountAndArealDensityStopCondition()
    elif name == "areal_density":
        return ArealDensityStopCondition(stop_condition_config["percentage_increase"],
                                         stop_condition_config["min_craters"])
    elif name == "n_craters_max":
        return NCratersMaxStopCondition(stop_condition_config["percentage_increase"],
                                        stop_condition_config["min_craters"])
    elif name == "n_craters":
        return NCratersStopCondition(stop_condition_config["n_craters"])
    elif name == "information_remaining":
        return CraterRecordInformationRemainingStopCondition(stop_condition_config["information_remaining_threshold"])


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


class NCratersMaxStopCondition(StopCondition):
    """
    Stops the simulation when no new maximum crater count has been reached in one third of the simulation iterations.
    """
    def __init__(self, percentage_increase: float, min_craters: int):
        self._percentage_increase = percentage_increase
        self._min_craters = min_craters

        self._n_craters_high_points: Dict[int, float] = {0: 0.0}
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1

        self._n_craters_high_points[self._counter] = max(
            self._n_craters_high_points[self._counter - 1],
            statistics_row.nobs
        )

        if self._counter < self._min_craters:
            return False

        checkpoint = self._counter // 3 * 2
        max_n_before_checkpoint = self._n_craters_high_points[checkpoint]
        max_n_after_checkpoint = self._n_craters_high_points[self._counter]

        return (max_n_after_checkpoint - max_n_before_checkpoint) \
            / max_n_after_checkpoint < self._percentage_increase


class CraterCountAndArealDensityStopCondition(StopCondition):
    """
    Stops the simulation when no new maximum crater count and areal density have been reached in one third of
    the simulation iterations.
    """
    MIN_CRATERS = 250000

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
            statistics_row.nobs
        )

        if self._counter < self.MIN_CRATERS:
            return False

        checkpoint = self._counter // 2
        max_areal_density_before_checkpoint = self._areal_density_high_points[checkpoint]
        max_areal_density_after_checkpoint = self._areal_density_high_points[self._counter]
        max_n_craters_before_checkpoint = self._craters_in_study_region_high_points[checkpoint]
        max_n_craters_after_checkpoint = self._craters_in_study_region_high_points[self._counter]

        return max_areal_density_before_checkpoint >= max_areal_density_after_checkpoint \
               and max_n_craters_before_checkpoint >= max_n_craters_after_checkpoint


class ArealDensityStopCondition(StopCondition):
    """
    Stops the simulation when the maximum areal density has not increased by more than a given percentage in half the
    total simulation time
    """
    def __init__(self, percentage_increase: float, min_craters: int):
        self._percentage_increase = percentage_increase
        self._min_craters = min_craters

        self._areal_density_high_points: Dict[int, float] = {0: 0.0}
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1

        self._areal_density_high_points[self._counter] = max(
            self._areal_density_high_points[self._counter - 1],
            statistics_row.areal_density
        )

        if self._counter < self._min_craters:
            return False

        checkpoint = self._counter // 3 * 2
        max_areal_density_before_checkpoint = self._areal_density_high_points[checkpoint]
        max_areal_density_after_checkpoint = self._areal_density_high_points[self._counter]

        return (max_areal_density_after_checkpoint - max_areal_density_before_checkpoint) \
            / max_areal_density_after_checkpoint < self._percentage_increase


class CraterRecordInformationRemainingStopCondition(StopCondition):
    """
    Stops the simulation when crater record's information remaining, calculated as n_craters_current / n_craters_total,
    reaches a given threshold.
    """
    def __init__(self, information_remaining_threshold: float):
        self._information_remaining_threshold = information_remaining_threshold

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        information_remaining = statistics_row.nobs / statistics_row.ntot
        return information_remaining < self._information_remaining_threshold
