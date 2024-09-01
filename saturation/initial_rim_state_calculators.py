from typing import Any, Dict

import numpy as np

from saturation.datatypes import Crater


class InitialRimStateCalculator(object):
    """
    Base class. Calculates the initial rim state of a crater.
    """
    def calculate(self, crater: Crater) -> float:
        raise NotImplementedError("Not implemented")


class CircumferenceInitialRimStateCalculator(InitialRimStateCalculator):
    """
    Uses the crater's circumference as the initial state.
    """
    def calculate(self, crater: Crater) -> float:
        return crater.radius * 2 * np.pi


class ExponentInitialRimStateCalculator(InitialRimStateCalculator):
    """
    Uses the crater's radius raised to an exponent as the initial state.
    """
    def __init__(self, exponent: float):
        self._exponent = exponent

    def calculate(self, crater: Crater) -> float:
        return crater.radius ** self._exponent


def get_initial_rim_state_calculator(config: Dict[str, Any]) -> InitialRimStateCalculator:
    name = config["name"]

    if name == "circumference":
        return CircumferenceInitialRimStateCalculator()
    if name == "exponent":
        return ExponentInitialRimStateCalculator(config["exponent"])

    raise Exception("Unexpected initial rim state calculator name.")