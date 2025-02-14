import itertools

import numpy as np

from saturation.distributions import ParetoProbabilityDistribution
from saturation.simulation import get_grouped_craters


def test_get_craters_ids_increase():
    # Act
    craters = get_grouped_craters(
        size_distribution=ParetoProbabilityDistribution(1.0, 1.0, 10.0),
        rstat=3.0,
        region_size=2,
        min_radius_threshold=1,
        random_seed=123
    )
    result = list(itertools.chain(*[x[0] for x in itertools.islice(craters, 10)]))[:10]

    # Assert
    ids = [x.id for x in result]
    assert ids == list(range(1, 11))
