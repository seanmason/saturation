from abc import ABC, abstractmethod
from typing import Dict

from saturation.writers import StatisticsRow


def get_stop_condition(stop_condition_config: Dict):
    name = stop_condition_config["name"]
    if name == "nstat":
        return NStatStopCondition(stop_condition_config["nstat"])

    raise Exception(f"Unknown stop condition {name}")


class StopCondition(ABC):
    @abstractmethod
    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        """
        Returns True if the simulation should stop.
        """
        pass


class NStatStopCondition(StopCondition):
    """
    Stops the simulation when a specified number of craters have been added to the study region.
    """
    def __init__(self, nstat: int):
        self._nstat = nstat
        self._counter = 0

    def should_stop(self, statistics_row: StatisticsRow) -> bool:
        self._counter += 1
        return self._counter == self._nstat
