import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from saturation.areal_density import ArealDensityCalculator
from saturation.datatypes import Crater


def test_calculate_areal_density_no_edges():
    # Arrange
    # A single crater that does not hit the edges
    crater = Crater(id=1, x=100, y=100, radius=100)
    observed_terrain_size = 1000
    terrain_padding = 0
    calculator = ArealDensityCalculator(observed_terrain_size, terrain_padding)

    # Act
    calculator.add_crater(crater)
    result = calculator.areal_density

    # Assert
    # It won't be exact, because of discretization, but it should be close.
    expected = crater.radius ** 2 * np.pi / observed_terrain_size ** 2
    assert_almost_equal(result, expected, decimal=3)


def test_calculate_areal_density_uses_margin():
    # Arrange
    # A single crater that has a quarter of its area within the observable area
    observed_terrain_size = 5000
    terrain_padding = 100
    calculator = ArealDensityCalculator(observed_terrain_size, terrain_padding)
    crater = Crater(id=1, x=terrain_padding, y=terrain_padding, radius=200)

    # Act
    calculator.add_crater(crater)
    result = calculator.areal_density

    # Assert
    expected = crater.radius ** 2 * np.pi / 4 / observed_terrain_size ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_uses_both_margins():
    # Arrange
    # Two craters that have a quarter of each's area within the observable area
    observed_terrain_size = 5000
    terrain_padding = 100
    calculator = ArealDensityCalculator(observed_terrain_size, terrain_padding)
    crater1 = Crater(id=1, x=terrain_padding, y=terrain_padding, radius=200)
    crater2 = Crater(id=2, x=observed_terrain_size + terrain_padding - 1, y=observed_terrain_size + terrain_padding - 1, radius=500)

    # Act
    calculator.add_crater(crater1)
    calculator.add_crater(crater2)
    result = calculator.areal_density

    # Assert
    crater1_area = crater1.radius ** 2 * np.pi / 4
    crater2_area = crater2.radius ** 2 * np.pi / 4
    expected = (crater1_area + crater2_area) / observed_terrain_size ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_overlapping_craters():
    # Arrange
    # Completely overlapping craters
    observed_terrain_size = 5000
    terrain_padding = 100
    calculator = ArealDensityCalculator(observed_terrain_size, terrain_padding)
    crater1 = Crater(id=1, x=terrain_padding, y=terrain_padding, radius=200)
    crater2 = Crater(id=2, x=terrain_padding, y=terrain_padding, radius=500)

    # Act
    calculator.add_crater(crater1)
    calculator.add_crater(crater2)
    result = calculator.areal_density

    # Assert
    crater2_area = crater2.radius ** 2 * np.pi / 4
    expected = crater2_area / observed_terrain_size ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_disjoint_add_and_remove():
    # Arrange
    # Two craters that have a quarter of each's area within the margin
    observed_terrain_size = 5000
    terrain_padding = 100
    calculator = ArealDensityCalculator(observed_terrain_size, terrain_padding)
    crater1 = Crater(id=1, x=terrain_padding, y=terrain_padding, radius=200)
    crater2 = Crater(id=2, x=observed_terrain_size + terrain_padding, y=observed_terrain_size + terrain_padding, radius=500)

    # Act
    calculator.add_crater(crater1)
    calculator.add_crater(crater2)
    calculator.remove_craters([crater1])
    result = calculator.areal_density

    # Assert
    crater2_area = crater2.radius ** 2 * np.pi / 4
    expected = crater2_area / observed_terrain_size ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_overlapping_add_and_remove():
    # Arrange
    # Two overlapping craters
    observed_terrain_size = 5000
    terrain_padding = 100
    calculator = ArealDensityCalculator(observed_terrain_size, terrain_padding)
    crater1 = Crater(id=1, x=terrain_padding + 100, y=terrain_padding + 100, radius=500)
    crater2 = Crater(id=2, x=terrain_padding, y=terrain_padding, radius=200)

    # Act
    calculator.add_crater(crater1)
    calculator.add_crater(crater2)
    calculator.remove_craters([crater1])
    result = calculator.areal_density

    # Assert
    crater2_area = crater2.radius ** 2 * np.pi / 4
    expected = crater2_area / observed_terrain_size ** 2
    assert_almost_equal(result, expected, decimal=4)
