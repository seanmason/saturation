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


@jitclass(spec={"_ratio": nb.types.float32, "_rmult": nb.types.float32})
class RadiusRatioConditionalRimOverlapRimErasureCalculator(ConditionalRimOverlapRimErasureCalculator):
    """
    A portion of the rim is erased if r_e / r_n < ratio.
    """
    def __init__(self, ratio: float, rmult: float):
        self._ratio = ratio
        self._rmult = rmult

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        return existing.radius / new.radius < self._ratio


@jitclass(spec={"_exponent": nb.types.float32, "_rmult": nb.types.float32})
class ExponentRadiusConditionalRimOverlapRimErasureCalculator(ConditionalRimOverlapRimErasureCalculator):
    """
    A portion of the rim is erased if r_n > r_e**exponent.
    """
    def __init__(self, exponent: float, rmult: float):
        self._exponent = exponent
        self._rmult = rmult

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        return new.radius > existing.radius**self._exponent


@jitclass(spec={"_rmult": nb.types.float32})
class LogRadiusConditionalRimOverlapRimErasureCalculator(ConditionalRimOverlapRimErasureCalculator):
    """
    A portion of the rim is erased if r_n > ln(r_e).
    """
    def __init__(self, rmult: float):
        self._rmult = rmult

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        return new.radius > np.log(existing.radius)


@jitclass(spec={"_rmult": nb.types.float32})
class SqrtRadiusConditionalRimOverlapRimErasureCalculator(ConditionalRimOverlapRimErasureCalculator):
    """
    A portion of the rim is erased if r_n > sqrt(r_e).
    """
    def __init__(self, rmult: float):
        self._rmult = rmult

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        return new.radius > np.sqrt(existing.radius)


@jitclass(spec={"_scale": nb.types.float32, "_rmult": nb.types.float32})
class SinLogRadiusConditionalRimOverlapRimErasureCalculator(ConditionalRimOverlapRimErasureCalculator):
    """
    A portion of the rim is erased if r_n > scaled function of sin(log(r_e)).
    """
    def __init__(
        self,
        min_r_period: float,
        max_r_period: float,
        n_periods: float,
        rmult: float
    ):
        self._scale = 2 * np.pi / np.log(max_r_period / min_r_period) * n_periods
        self._rmult = rmult

    def _crater_can_be_affected(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ):
        return new.radius > (np.sin(np.log(existing.radius) * self._scale) + 2) * existing.radius / 6


@jitclass(spec={"_constant": nb.types.float32})
class EnergyRimErasureCalculator(RimErasureCalculator):
    """
    Calculates the amount of rim erasure with energy scaling.
    """
    def __init__(self, constant: float):
        self._constant = constant

    def calculate_new_rim_state(
        self,
        existing: Crater,
        existing_rim_state: float,
        new: Crater
    ) -> float:
        x_diff = existing.x - new.x
        y_diff = existing.y - new.y
        distance = np.sqrt(x_diff * x_diff + y_diff * y_diff)

        erasure = self._constant * new.radius ** 3
        if distance > new.radius + existing.radius:
            erasure /= np.sqrt(np.abs(existing.radius**2 - distance**2))

        return existing_rim_state - erasure


def get_rim_erasure_calculator(config: Dict[str, any], rmult: float) -> RimErasureCalculator:
    name = config["name"]

    result = None
    if name == "radius_ratio":
        result = RadiusRatioConditionalRimOverlapRimErasureCalculator(config["ratio"], rmult)
    elif name == "log":
        result = LogRadiusConditionalRimOverlapRimErasureCalculator(rmult)
    elif name == "exponent":
        result = ExponentRadiusConditionalRimOverlapRimErasureCalculator(config["exponent"], rmult)
    elif name == "sin_log":
        result = SinLogRadiusConditionalRimOverlapRimErasureCalculator(
            config["n_periods"],
            config["min_r_period"],
            config["max_r_period"],
            rmult
        )
    elif name == "energy":
        result = EnergyRimErasureCalculator(config["constant"])

    return result

