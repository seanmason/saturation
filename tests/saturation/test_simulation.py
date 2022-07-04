import numpy as np
from saturation.simulation import get_crater_locations


def test_get_crater_locations():
    # Act
    result = get_crater_locations(2)

    # Assert
    assert len(result) == 2
    assert np.all(result == 0)
