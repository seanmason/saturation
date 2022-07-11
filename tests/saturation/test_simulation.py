import numpy as np

from saturation.distributions import ProbabilityDistribution
from saturation.simulation import get_crater_locations, get_craters


class DummyProbabilityDistribution(ProbabilityDistribution):
    def __init__(self):
        self.uniform_to_value_result = 0
        self.value_to_probability_result = 0
        self.probability_to_value_result = 0

    def uniform_to_value(self, uniform: float) -> float:
        return self.uniform_to_value_result

    def value_to_probability(self, x: float) -> float:
        return self.value_to_probability_result

    def probability_to_value(self, p: float) -> float:
        return self.probability_to_value_result


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
    distribution.uniform_to_value_result = 37

    # Act
    result = get_craters(1, distribution, location_func=location_func)

    # Assert
    first = result.loc[1]
    assert first.x == 7
    assert first.y == 13
    assert first.radius == 37


def test_get_craters_ids_increase():
    # Arrange
    distribution = DummyProbabilityDistribution()

    # Act
    result = get_craters(10, distribution)

    # Assert
    assert list(result.index) == list(range(1, 11))
