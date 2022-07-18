import numpy as np
from numpy.testing import assert_almost_equal

from saturation.statistics import calculate_z_statistic, calculate_za_statistic


def test_calculate_z_statistic_one_value():
    # Arrange
    nn_distances = np.array([1.])

    # Act
    result = calculate_z_statistic(nn_distances, 10)

    # Assert
    assert_almost_equal(-1.5304560, result)


def test_calculate_z_statistic_multiple_values():
    # Arrange
    nn_distances = np.array([1., 1., 2., 2., 3., 3.])

    # Act
    result = calculate_z_statistic(nn_distances, 10)

    # Assert
    assert_almost_equal(result, -0.0946773469222112)


def test_calculate_za_statistic_one_value():
    # Arrange
    nn_distances = np.array([1.])

    # Act
    result = calculate_za_statistic(nn_distances, 10, 50)

    # Assert
    assert_almost_equal(-1.83639378, result)


def test_calculate_za_statistic_multiple_values():
    # Arrange
    nn_distances = np.array([1., 1., 2., 2., 3., 3.])

    # Act
    result = calculate_za_statistic(nn_distances, 10, 50)

    # Assert
    assert_almost_equal(result, -3.765929853)
