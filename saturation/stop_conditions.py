from abc import ABC, abstractmethod
from typing import Dict

from saturation.writers import StatisticsRow


def get_stop_condition(stop_condition_config: Dict):
    name = stop_condition_config["name"]
    if name == "crater_count_and_areal_density":
        return CraterCountAndArealDensityStopCondition()
    elif name == "areal_density":
        return ArealDensityStopCondition(stop_condition_config["percentage_increase"],
                                         stop_condition_config["min_ntot"])
    elif name == "ntot_max":
        return NCratersMaxStopCondition(stop_condition_config["percentage_increase"],
                                        stop_condition_config["min_ntot"])
    elif name == "ntot":
        return NTotStopCondition(stop_condition_config["ntot"])
    elif name == "information_remaining":
        return CraterRecordInformationRemainingStopCondition(stop_condition_config["information_remaining_threshold"])


class StopCondition(ABC):
    @abstractmethod
    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        """
        Returns True if the simulation should stop.
        """
        pass


class NTotStopCondition(StopCondition):
    """
    Stops the simulation when a specified number of craters have been added to the study region.
    """

    def __init__(self, ntot: int):
        self._ntot = ntot
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1
        return self._counter == self._ntot


class NCratersMaxStopCondition(StopCondition):
    """
    Stops the simulation when no new maximum crater count has been reached in one third of the simulation iterations.
    """
    def __init__(self, percentage_increase: float, min_ntot: int):
        self._percentage_increase = percentage_increase
        self._min_ntot = min_ntot

        self._nobs_high_points: Dict[int, float] = {0: 0.0}
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1

        self._nobs_high_points[self._counter] = max(
            self._nobs_high_points[self._counter - 1],
            statistics_row.nobs
        )

        if self._counter < self._min_ntot:
            return False

        checkpoint = self._counter // 3 * 2
        max_n_before_checkpoint = self._nobs_high_points[checkpoint]
        max_n_after_checkpoint = self._nobs_high_points[self._counter]

        return (max_n_after_checkpoint - max_n_before_checkpoint) \
            / max_n_after_checkpoint < self._percentage_increase


class CraterCountAndArealDensityStopCondition(StopCondition):
    """
    Stops the simulation when no new maximum crater count and areal density have been reached in one third of
    the simulation iterations.
    """
    MIN_NTOT = 250000

    def __init__(self):
        self._areal_density_high_points: Dict[int, float] = {0: 0.0}
        self._nobs_high_points: Dict[int, int] = {0: 0}
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1

        self._areal_density_high_points[self._counter] = max(
            self._areal_density_high_points[self._counter - 1],
            statistics_row.areal_density
        )
        self._nobs_high_points[self._counter] = max(
            self._nobs_high_points[self._counter - 1],
            statistics_row.nobs
        )

        if self._counter < self.MIN_NTOT:
            return False

        checkpoint = self._counter // 2
        max_areal_density_before_checkpoint = self._areal_density_high_points[checkpoint]
        max_areal_density_after_checkpoint = self._areal_density_high_points[self._counter]
        max_nobs_before_checkpoint = self._nobs_high_points[checkpoint]
        max_nobs_after_checkpoint = self._nobs_high_points[self._counter]

        return (
            max_areal_density_before_checkpoint >= max_areal_density_after_checkpoint
            and max_nobs_before_checkpoint >= max_nobs_after_checkpoint
        )


class ArealDensityStopCondition(StopCondition):
    """
    Stops the simulation when the maximum areal density has not increased by more than a given percentage in half the
    total simulation time
    """
    def __init__(self, percentage_increase: float, min_ntot: int):
        self._percentage_increase = percentage_increase
        self._min_ntot = min_ntot

        self._areal_density_high_points: Dict[int, float] = {0: 0.0}
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1

        self._areal_density_high_points[self._counter] = max(
            self._areal_density_high_points[self._counter - 1],
            statistics_row.areal_density
        )

        if self._counter < self._min_ntot:
            return False

        checkpoint = self._counter // 3 * 2
        max_areal_density_before_checkpoint = self._areal_density_high_points[checkpoint]
        max_areal_density_after_checkpoint = self._areal_density_high_points[self._counter]

        return (max_areal_density_after_checkpoint - max_areal_density_before_checkpoint) \
            / max_areal_density_after_checkpoint < self._percentage_increase


class CraterRecordInformationRemainingStopCondition(StopCondition):
    """
    Stops the simulation when crater record's information remaining, calculated as nobs / ntot,
    reaches a given threshold.
    """
    MIN_NTOT = 50000

    def __init__(self, information_remaining_threshold: float):
        self._information_remaining_threshold = information_remaining_threshold

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        if statistics_row.ntot < CraterRecordInformationRemainingStopCondition.MIN_NTOT:
            return False

        information_remaining = statistics_row.nobs / statistics_row.ntot
        return information_remaining < self._information_remaining_threshold
