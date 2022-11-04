import numpy as np
from numpy.testing import assert_almost_equal

from saturation.z_stats import calculate_z_statistic, calculate_za_statistic


def test_calculate_z_statistic_one_value():
    # Act
    result = calculate_z_statistic(1, 1, 100)

    # Assert
    assert_almost_equal(-1.5304560, result)


def test_calculate_z_statistic_multiple_values():
    # Act
    result = calculate_z_statistic(2, 6, 100)

    # Assert
    assert_almost_equal(result, -0.0946773469222112)


def test_calculate_za_statistic_one_value():
    # Act
    result = calculate_za_statistic(1, 1, 100, 2500)

    # Assert
    assert_almost_equal(result, -1.8349693353)


def test_calculate_za_statistic_multiple_values():
    # Act
    result = calculate_za_statistic(2, 6, 100, 2500)

    # Assert
    assert_almost_equal(result, -3.74883645972)
