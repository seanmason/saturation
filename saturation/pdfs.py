from abc import ABC, abstractmethod

import numpy as np


class PDF(ABC):
    """
    Abstract base class for a probability density function (PDF).
    """
    @abstractmethod
    def get_value(self, x: float) -> float:
        """
        Returns the PDF's value at x
        """
        pass

    @abstractmethod
    def get_inverse(self, x: float) -> float:
        """
        Returns the inverse of the PDF at the supplied value.
        """
        pass


class PowerLawPDF(PDF):
    """
    Represents a power law PDF given the slope of the PDF and the min and max values.
    PDF has the form of p(x) = constant * x^slope
    """
    def __init__(self, *, slope: float, min_value: float, max_value: float):
        assert slope < -2, "PDF slope must be < -2"

        self._slope = slope
        self._min_value = min_value
        self._max_value = max_value

        # Calculate the constant
        eval_max = max_value**(slope + 1)
        eval_min = min_value**(slope + 1)
        self._constant = 1 / (eval_max - eval_min) / (self._slope + 1)

    def get_value(self, x: float) -> float:
        return self._constant * x**self._slope

    def get_inverse(self, p: float) -> float:
        return np.exp(np.log(p/self._constant) / self._slope)
