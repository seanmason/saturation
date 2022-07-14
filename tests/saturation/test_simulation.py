import numpy as np

from saturation.distributions import ProbabilityDistribution
from saturation.simulation import get_crater_locations, get_craters


class DummyProbabilityDistribution(ProbabilityDistribution):
    def __init__(self):
        self.inverse_cdf_result = 0
        self.pdf_result = 0
        self.cdf_result = 0

    def inverse_cdf(self, uniform: float) -> float:
        return self.inverse_cdf_result

    def pdf(self, x: float) -> float:
        return self.pdf_result

    def cdf(self, p: float) -> float:
        return self.cdf_result


def test_get_crater_locations():
    # Act
    result = get_crater_locations(10)

    # Assert
    assert np.shape(result) == (10, 2)


def test_get_craters_uses_probability_distribution_and_location_func():
    # Arrange
    def location_func(n_craters):
        return np.array([7, 13] * n_craters).reshape(n_craters, 2)

    distribution = DummyProbabilityDistribution()
    distribution.inverse_cdf_result = 37

    # Act
    result = get_craters(1, distribution, 2, location_func=location_func)

    # Assert
    first = result.iloc[0]
    assert first.x == 14
    assert first.y == 26
    assert first.radius == 37


def test_get_craters_ids_increase():
    # Arrange
    distribution = DummyProbabilityDistribution()

    # Act
    result = get_craters(10, distribution, 2)

    # Assert
    assert list(result.index) == list(range(1, 11))
