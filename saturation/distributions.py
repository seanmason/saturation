from abc import ABC, abstractmethod

import numpy as np


class ProbabilityDistribution(ABC):
    """
    Abstract base class for a probability distribution.
    """

    @abstractmethod
    def inverse_cdf(self, p: float) -> float:
        """
        Converts a random uniform value on [0, 1] to a value in the distribution
        """
        pass

    @abstractmethod
    def cdf(self, x: float) -> float:
        """
        Returns the CDF at x
        """
        pass

    @abstractmethod
    def pdf(self, x: float) -> float:
        """
        Returns the PDF at x
        """
        pass


class PowerLawProbabilityDistribution(ProbabilityDistribution):
    """
    Represents a power law distribution given the slope of the PDF and the min value.
    PDF has the form of p(x) = constant * x^slope
    """

    def __init__(self, *, slope: float, min_value: float, max_value: float = None):
        assert slope < -1, "PDF slope must be < -1"

        self._slope = slope
        self._min_value = min_value

        # Calculate the constant
        self._constant = (-self._slope - 1) / min_value ** (self._slope + 1)

        # Calculate the max allowed value of a uniform random input
        if max_value:
            self._max_uniform = self.cdf(max_value)
        else:
            self._max_uniform = 1

    def inverse_cdf(self, p: float) -> float:
        return self._min_value * (1 - p * self._max_uniform) ** (1 / (1 + self._slope))

    def cdf(self, x: float) -> float:
        return 1 - self._constant / (-self._slope - 1) * x ** (self._slope + 1)

    def pdf(self, x: float) -> float:
        return self._constant * x ** self._slope / self._max_uniform
