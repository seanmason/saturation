from abc import ABC, abstractmethod

import numpy as np


class ProbabilityDistribution(ABC):
    """
    Abstract base class for a probability distribution.
    """

    @abstractmethod
    def uniform_to_value(self, uniform: float) -> float:
        """
        Converts a random uniform value on [0, 1] to a value in the distribution
        """
        pass

    @abstractmethod
    def value_to_probability(self, x: float) -> float:
        """
        Returns the probability at x
        """
        pass

    @abstractmethod
    def probability_to_value(self, p: float) -> float:
        """
        Returns the value of the distribution at probability p.
        """
        pass


class PowerLawProbabilityDistribution(ProbabilityDistribution):
    """
    Represents a power law distribution given the slope of the PDF and the min value.
    PDF has the form of p(x) = constant * x^slope
    """
    def __init__(self, *, slope: float, min_value: float):
        assert slope < -1, "PDF slope must be < -1"

        self._slope = slope
        self._min_value = min_value

        # Calculate the constant
        self._constant = (-self._slope - 1) / min_value**(self._slope + 1)

    def uniform_to_value(self, uniform: float) -> float:
        return self._min_value * (1 - uniform)**(1/(1 + self._slope))

    def value_to_probability(self, x: float) -> float:
        return self._constant * x**self._slope

    def probability_to_value(self, p: float) -> float:
        return np.exp(np.log(p/self._constant) / self._slope)

