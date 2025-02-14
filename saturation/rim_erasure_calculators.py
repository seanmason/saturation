from typing import Dict

from saturation.numba_utils import *
import numpy as np

from saturation.datatypes import Crater
from saturation.geometry import get_intersection_arc


class RimErasureCalculator(object):
    """
    Calculates the amount of rim erasure resulting from a new crater forming.
    Base class.
    """
    def get_min_radius_threshold(self) -> float:
        """
        Returns the minimum crater radius that is able to affect another crater.
        """
        raise NotImplementedError("Not implemented.")

    def calculate_new_rim_state(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ) -> float:
        raise NotImplementedError("Not implemented")


@nb.experimental.jitclass(spec={
    "_exponent": nb.types.float64,
    "_ratio": nb.types.float64,
    "_rmult": nb.types.float64,
    "_min_radius_threshold": nb.types.float64
})
class ExponentRadiusConditionalRimOverlapRimErasureCalculator(RimErasureCalculator):
    """
    A portion of the rim is erased if r_n > r_e**exponent.
    """
    def __init__(
        self,
        exponent: float,
        ratio: float,
        rmult: float,
        rstat: float
    ):
        self._exponent = exponent
        self._ratio = ratio
        self._rmult = rmult
        self._min_radius_threshold = rstat ** exponent / self._ratio

    def get_min_radius_threshold(self) -> float:
        return self._min_radius_threshold

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        return new.radius > existing.radius**self._exponent / self._ratio

    def calculate_new_rim_state(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ) -> float:
        if not self._crater_can_be_affected(existing, existing_rim_state, new):
            return existing_rim_state

        theta1, theta2 = get_intersection_arc(
            (existing.x, existing.y),
            existing.radius,
            (new.x, new.y),
            new.radius * self._rmult
        )
        if theta1 < theta2:
            arc_length = np.abs(theta2 - theta1)
        else:
            arc_length = np.abs(theta1 - theta2 - 2 * np.pi)
        return existing_rim_state * (1 - arc_length / (2 * np.pi))


def get_rim_erasure_calculator(
    *,
    config: Dict[str, any],
    rmult: float,
    rstat: float
) -> RimErasureCalculator:
    name = config["name"]

    if name == "exponent_radius_ratio":
        return ExponentRadiusConditionalRimOverlapRimErasureCalculator(
            exponent=config["exponent"],
            ratio=config["ratio"],
            rmult=rmult,
            rstat=rstat
        )
    else:
        raise Exception(f"Unknown rim erasure calculator {name}")
