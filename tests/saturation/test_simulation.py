import itertools

import numpy as np

from saturation.distributions import ProbabilityDistribution
from saturation.simulation import get_crater_location, get_craters


class DummyProbabilityDistribution(ProbabilityDistribution):
    def __init__(self):
        self.inverse_cdf_result = 0
        self.pdf_result = 0
        self.cdf_result = 0

    def pullback(self, uniform: float) -> float:
        return self.inverse_cdf_result

    def pdf(self, x: float) -> float:
        return self.pdf_result

    def cdf(self, p: float) -> float:
        return self.cdf_result


def test_get_crater_locations():
    # Act
    result = get_crater_location()

    # Assert
    assert np.shape(result) == (2,)


def test_get_craters_uses_probability_distribution_and_location_func():
    # Arrange
    def location_func():
        return np.array([7, 13])

    distribution = DummyProbabilityDistribution()
    distribution.inverse_cdf_result = 37

    # Act
    result = list(itertools.islice(get_craters(distribution, 2, location_func=location_func), 2))

    # Assert
    assert len(result) == 2

    first = result[0]
    assert first.x == 14
    assert first.y == 26
    assert first.radius == 37


def test_get_craters_ids_increase():
    # Arrange
    distribution = DummyProbabilityDistribution()

    # Act
    result = list(itertools.islice(get_craters(distribution, 2), 10))

    # Assert
    ids = [x.id for x in result]
    assert ids == list(range(1, 11))
