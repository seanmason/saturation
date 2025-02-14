from abc import abstractmethod
from typing import Union

import numpy as np
from saturation.numba_utils import *


class ProbabilityDistribution(object):
    """
    Abstract base class for a probability distribution.
    """

    @abstractmethod
    def pullback(self, u: Union[float, np.array]) -> Union[float, np.array]:
        """
        Converts a random uniform value on [0, 1] to a value from the distribution
        """
        pass

    @abstractmethod
    def cdf(self, x: Union[float, np.array]) -> Union[float, np.array]:
        """
        Returns the CDF at x
        """
        pass

    @abstractmethod
    def pdf(self, x: Union[float, np.array]) -> Union[float, np.array]:
        """
        Returns the PDF at x
        """
        pass


@nb.experimental.jitclass(spec={
    "_alpha": nb.types.float64,
    "_x_min": nb.types.float64,
    "_x_max": nb.types.float64,
    "_u_max": nb.types.float64
})
class ParetoProbabilityDistribution(ProbabilityDistribution):
    """
    Represents a truncated Pareto distribution.
    """
    def __init__(self, alpha: float, x_min: float, x_max: float):
        self._alpha = alpha
        self._x_min = x_min
        self._x_max = x_max
        self._u_max = 1 - (x_min / x_max) ** alpha

    def pullback(self, u: Union[float, np.array]) -> Union[float, np.array]:
        return (1 - (u * self._u_max)) ** (-1. / self._alpha) * self._x_min

    def cdf(self, x: Union[float, np.array]) -> Union[float, np.array]:
        if x < self._x_min:
            return 0.0
        elif x > self._x_max:
            return 1.0

        return 1 - (self._x_min / x) ** self._alpha

    def pdf(self, x: Union[float, np.array]) -> Union[float, np.array]:
        if x < self._x_min:
            return 0.0

        return self._alpha * self._x_min**self._alpha / x ** (self._alpha + 1)
