import numpy as np
from numpy.testing import assert_almost_equal

from saturation.statistics import calculate_z_statistic, calculate_za_statistic


def test_calculate_z_statistic_one_value():
    # Arrange
    nn_distances = [1.]

    # Act
    result = calculate_z_statistic(nn_distances, 100)

    # Assert
    assert_almost_equal(-1.5304560, result)


def test_calculate_z_statistic_multiple_values():
    # Arrange
    nn_distances = [1., 1., 2., 2., 3., 3.]

    # Act
    result = calculate_z_statistic(nn_distances, 100)

    # Assert
    assert_almost_equal(result, -0.0946773469222112)


def test_calculate_za_statistic_one_value():
    # Arrange
    nn_distances = [1.]

    # Act
    result = calculate_za_statistic(nn_distances, 100, 2500)

    # Assert
    assert_almost_equal(result, -1.8349693353)


def test_calculate_za_statistic_multiple_values():
    # Arrange
    nn_distances = [1., 1., 2., 2., 3., 3.]

    # Act
    result = calculate_za_statistic(nn_distances, 100, 2500)

    # Assert
    assert_almost_equal(result, -3.74883645972)
