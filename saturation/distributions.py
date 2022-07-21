from abc import ABC, abstractmethod


class ProbabilityDistribution(ABC):
    """
    Abstract base class for a probability distribution.
    """

    @abstractmethod
    def pullback(self, u: float) -> float:
        """
        Converts a random uniform value on [0, 1] to a value from the distribution
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


class ParetoProbabilityDistribution(ProbabilityDistribution):
    """
    Represents a Pareto distribution given the slope of the CDF and the min/max values.
    PDF has the form of p(x) = constant * x^slope
    """
    def __init__(self, *, cdf_slope: float, x_min: float, x_max: float):
        self._cdf_slope = cdf_slope
        self._x_min = x_min
        self._x_max = x_max
        self._u_max = 1 - (x_min / x_max) ** cdf_slope

    def pullback(self, u: float) -> float:
        return (1 - (u * self._u_max)) ** (-1. / self._cdf_slope) * self._x_min

    def cdf(self, x: float) -> float:
        if x < self._x_min:
            return 0.0
        elif x > self._x_max:
            return 1.0

        return 1 - (self._x_min / x) ** self._cdf_slope

    def pdf(self, x: float) -> float:
        if x < self._x_min:
            return 0.0

        return self._cdf_slope * self._x_min**self._cdf_slope / x ** (self._cdf_slope + 1)
