import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from saturation.areal_density import ArealDensityCalculator


def test_calculate_areal_density_no_edges():
    # Arrange
    # A single crater that does not hit the edges
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 100},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    terrain_size = 1000
    margin = 0
    calculator = ArealDensityCalculator(terrain_size, margin)

    # Act
    calculator.update(craters, pd.DataFrame(columns=['x', 'y', 'radius']))
    result = calculator.get_areal_density()

    # Assert
    # It won't be exact, because of discretization, but it should be close.
    expected = craters.iloc[0].radius ** 2 * np.pi / terrain_size ** 2
    assert_almost_equal(result, expected, decimal=3)


def test_calculate_areal_density_uses_margin():
    # Arrange
    # A single crater that has a quarter of its area within the margin
    terrain_size = 5000
    margin = 100
    data = [
        {'id': 1, 'x': margin, 'y': margin, 'radius': 200},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    calculator = ArealDensityCalculator(terrain_size, margin)

    # Act
    calculator.update(craters, pd.DataFrame(columns=['x', 'y', 'radius']))
    result = calculator.get_areal_density()

    # Assert
    expected = craters.iloc[0].radius ** 2 * np.pi / 4 / (terrain_size - 2 * margin) ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_uses_both_margins():
    # Arrange
    # Two craters that have a quarter of each's area within the margin
    terrain_size = 5000
    margin = 100
    data = [
        {'id': 1, 'x': margin, 'y': margin, 'radius': 200},
        {'id': 2, 'x': terrain_size - margin - 1, 'y': terrain_size - margin - 1, 'radius': 500},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    calculator = ArealDensityCalculator(terrain_size, margin)

    # Act
    calculator.update(craters, pd.DataFrame(columns=['x', 'y', 'radius']))
    result = calculator.get_areal_density()

    # Assert
    crater1_area = craters.iloc[0].radius ** 2 * np.pi / 4
    crater2_area = craters.iloc[1].radius ** 2 * np.pi / 4
    expected = (crater1_area + crater2_area) / (terrain_size - 2 * margin) ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_overlapping_craters():
    # Arrange
    # Completely overlapping craters
    terrain_size = 5000
    margin = 100
    data = [
        {'id': 1, 'x': margin, 'y': margin, 'radius': 200},
        {'id': 2, 'x': margin, 'y': margin, 'radius': 500},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    calculator = ArealDensityCalculator(terrain_size, margin)

    # Act
    calculator.update(craters, pd.DataFrame(columns=['x', 'y', 'radius']))
    result = calculator.get_areal_density()

    # Assert
    crater2_area = craters.iloc[1].radius ** 2 * np.pi / 4
    expected = crater2_area / (terrain_size - 2 * margin) ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_disjoint_add_and_remove():
    # Arrange
    # Two craters that have a quarter of each's area within the margin
    terrain_size = 5000
    margin = 100
    data = [
        {'id': 1, 'x': margin, 'y': margin, 'radius': 200},
        {'id': 2, 'x': terrain_size - 2 * margin - 1, 'y': terrain_size - 2 * margin - 1, 'radius': 500},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    calculator = ArealDensityCalculator(terrain_size, margin)
    new_erased_craters = craters.loc[[2]]

    # Act
    calculator.update(craters, new_erased_craters)
    result = calculator.get_areal_density()

    # Assert
    crater1_area = craters.iloc[0].radius ** 2 * np.pi / 4
    expected = crater1_area / (terrain_size - 2 * margin) ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_overlapping_add_and_remove():
    # Arrange
    # Two overlapping craters
    terrain_size = 5000
    margin = 100
    data = [
        {'id': 1, 'x': margin, 'y': margin, 'radius': 200},
        {'id': 2, 'x': margin + 100, 'y': margin + 100, 'radius': 50},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    calculator = ArealDensityCalculator(terrain_size, margin)
    new_erased_craters = craters.loc[[1]]

    # Act
    calculator.update(craters, new_erased_craters)
    result = calculator.get_areal_density()

    # Assert
    crater2_area = craters.iloc[1].radius ** 2 * np.pi
    expected = crater2_area / (terrain_size - 2 * margin) ** 2
    assert_almost_equal(result, expected, decimal=4)

