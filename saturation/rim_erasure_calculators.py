from typing import Dict

import numba as nb
from numba.experimental import jitclass
import numpy as np

from saturation.datatypes import Crater
from saturation.geometry import get_intersection_arc


class RimErasureCalculator(object):
    """
    Calculates the amount of rim erasure resulting from a new crater forming.
    Base class.
    """
    def can_affect_rims(self, new_radius: float) -> bool:
        """
        Returns True if a crater with the supplied radius has the potential to affect ANY crater's rim.
        """
        raise NotImplementedError("Not implemented")

    def calculate_new_rim_state(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ) -> float:
        raise NotImplementedError("Not implemented")


class ConditionalRimOverlapRimErasureCalculator(RimErasureCalculator):
    """
    Base class for rim erasure calculators in which the erasure amount is a percentage of the existing rim, determined
    by the amount of rim overlapped. Erasure only occurs if a condition `_crater_can_be_affected` is met.
    """
    _rmult: float

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        raise NotImplementedError("Not implemented")

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


@jitclass(spec={"_ratio": nb.types.float32, "_rmult": nb.types.float32, "_min_radius_threshold": nb.types.float32})
class RadiusRatioConditionalRimOverlapRimErasureCalculator(ConditionalRimOverlapRimErasureCalculator):
    """
    A portion of the rim is erased if r_e / r_n < ratio.
    """
    def __init__(
        self,
        ratio: float,
        rmult: float,
        r_stat: float
    ):
        self._ratio = ratio
        self._rmult = rmult
        self._min_radius_threshold = r_stat / ratio

    def can_affect_rims(self, new_radius: float) -> bool:
        return new_radius >= self._min_radius_threshold

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        return existing.radius / new.radius < self._ratio


@jitclass(spec={
    "_exponent": nb.types.float32,
    "_ratio": nb.types.float32,
    "_rmult": nb.types.float32,
    "_min_radius_threshold": nb.types.float32
})
class ExponentRadiusConditionalRimOverlapRimErasureCalculator(ConditionalRimOverlapRimErasureCalculator):
    """
    A portion of the rim is erased if r_n > r_e**exponent.
    """
    def __init__(
        self,
        exponent: float,
        ratio: float,
        rmult: float,
        r_stat: float
    ):
        self._exponent = exponent
        self._ratio = ratio
        self._rmult = rmult
        self._min_radius_threshold = r_stat ** exponent / self._ratio

    def can_affect_rims(self, new_radius: float) -> bool:
        return new_radius >= self._min_radius_threshold

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        return new.radius > existing.radius**self._exponent / self._ratio


def get_rim_erasure_calculator(
    *,
    config: Dict[str, any],
    rmult: float,
    r_stat: float
) -> RimErasureCalculator:
    name = config["name"]

    result = None
    if name == "radius_ratio":
        result = RadiusRatioConditionalRimOverlapRimErasureCalculator(
            ratio=config["ratio"],
            rmult=rmult,
            r_stat=r_stat
        )
    elif name == "exponent":
        result = ExponentRadiusConditionalRimOverlapRimErasureCalculator(
            exponent=config["exponent"],
            ratio=config["ratio"],
            rmult=rmult,
            r_stat=r_stat
        )

    return result

