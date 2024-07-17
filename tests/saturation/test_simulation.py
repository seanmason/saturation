import itertools

import numpy as np

from saturation.distributions import ProbabilityDistribution
from saturation.simulation import get_craters


class DummyProbabilityDistribution(ProbabilityDistribution):
    def __init__(self):
        self.inverse_cdf_result = np.array([0] * 100)
        self.pdf_result = 0
        self.cdf_result = 0

    def pullback(self, uniform: float) -> float:
        return self.inverse_cdf_result

    def pdf(self, x: float) -> float:
        return self.pdf_result

    def cdf(self, p: float) -> float:
        return self.cdf_result


def test_get_craters_ids_increase():
    # Arrange
    distribution = DummyProbabilityDistribution()

    # Act
    result = list(itertools.islice(get_craters(distribution, 2), 10))

    # Assert
    ids = [x.id for x in result]
    assert ids == list(range(1, 11))
