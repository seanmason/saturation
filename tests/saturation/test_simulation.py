import numpy as np

from saturation.distributions import ProbabilityDistribution
from saturation.simulation import get_crater_location, get_craters


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


def test_get_crater_location():
    # Act
    result = get_crater_location()

    # Assert
    assert np.shape(result) == (2, 2)


def test_get_craters_yields_infinitely():
    # Arrange
    distribution = DummyProbabilityDistribution()

    # Act
    result = [next(get_craters(distribution)) for _ in range(5)]

    # Assert
    assert len(result) == 5


def test_get_craters_uses_probability_distribution_and_location_func():
    # Arrange
    def location_func():
        return 7, 13
    distribution = DummyProbabilityDistribution()
    distribution.uniform_to_value_result = 37

    # Act
    result = next(get_craters(distribution, location_func=location_func))

    # Assert
    assert result.id == 1
    assert result.location == (7, 13)
    assert result.radius == 37
    assert result.rim_segments == [(0, 2 * np.pi)]


def test_get_craters_ids_increase():
    # Arrange
    distribution = DummyProbabilityDistribution()

    # Act
    iterator = get_craters(distribution)
    result = [next(iterator).id for _ in range(5)]

    # Assert
    assert result == list(range(1, 6))
