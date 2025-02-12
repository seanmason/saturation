import numpy as np
from numpy.testing import assert_almost_equal

from saturation.areal_density import ArealDensityCalculator
from saturation.datatypes import Crater


def test_calculate_areal_density_no_edges():
    # Arrange
    # A single crater that does not hit the edges
    crater = Crater(id=1, x=100, y=100, radius=100)
    study_region_size = 1000
    study_region_padding = 0
    calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=3
    )

    # Act
    calculator.add_crater(crater)
    result = calculator.areal_density

    # Assert
    # It won't be exact, because of discretization, but it should be close.
    expected = crater.radius ** 2 * np.pi / study_region_size ** 2
    assert_almost_equal(result, expected, decimal=3)


def test_calculate_areal_density_uses_margin():
    # Arrange
    # A single crater that has a quarter of its area within the study region
    study_region_size = 5000
    study_region_padding = 100
    calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=3
    )
    crater = Crater(id=1, x=study_region_padding, y=study_region_padding, radius=200)

    # Act
    calculator.add_crater(crater)
    result = calculator.areal_density

    # Assert
    expected = crater.radius ** 2 * np.pi / 4 / study_region_size ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_uses_both_margins():
    # Arrange
    # Two craters that have a quarter of each's area within the study region
    study_region_size = 5000
    study_region_padding = 100
    calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=3
    )
    crater1 = Crater(id=1, x=study_region_padding, y=study_region_padding, radius=200)
    crater2 = Crater(id=2, x=study_region_size + study_region_padding - 1, y=study_region_size + study_region_padding - 1, radius=500)

    # Act
    calculator.add_crater(crater1)
    calculator.add_crater(crater2)
    result = calculator.areal_density

    # Assert
    crater1_area = crater1.radius ** 2 * np.pi / 4
    crater2_area = crater2.radius ** 2 * np.pi / 4
    expected = (crater1_area + crater2_area) / study_region_size ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_overlapping_craters():
    # Arrange
    # Completely overlapping craters
    study_region_size = 5000
    study_region_padding = 100
    calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=3
    )
    crater1 = Crater(id=1, x=study_region_padding, y=study_region_padding, radius=200)
    crater2 = Crater(id=2, x=study_region_padding, y=study_region_padding, radius=500)

    # Act
    calculator.add_crater(crater1)
    calculator.add_crater(crater2)
    result = calculator.areal_density

    # Assert
    crater2_area = crater2.radius ** 2 * np.pi / 4
    expected = crater2_area / study_region_size ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_disjoint_add_and_remove():
    # Arrange
    # Two craters that have a quarter of each's area within the margin
    study_region_size = 5000
    study_region_padding = 100
    calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=3
    )
    crater1 = Crater(id=1, x=study_region_padding, y=study_region_padding, radius=200)
    crater2 = Crater(id=2, x=study_region_size + study_region_padding - 1, y=study_region_size + study_region_padding - 1, radius=500)

    # Act
    calculator.add_crater(crater1)
    calculator.add_crater(crater2)
    calculator.remove_craters([crater1])
    result = calculator.areal_density

    # Assert
    crater2_area = crater2.radius ** 2 * np.pi / 4
    expected = crater2_area / study_region_size ** 2
    assert_almost_equal(result, expected, decimal=4)


def test_calculate_areal_density_overlapping_add_and_remove():
    # Arrange
    # Two craters, the first completely overlapping by the second.
    # The first crater gets removed, exposing the second as the only cratered area.
    study_region_size = 5000
    study_region_padding = 100
    calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=3
    )
    crater1 = Crater(id=1, x=study_region_padding + 100, y=study_region_padding + 100, radius=500)
    crater2 = Crater(id=2, x=study_region_padding, y=study_region_padding, radius=200)

    # Act
    calculator.add_crater(crater1)
    calculator.add_crater(crater2)
    calculator.remove_craters([crater1])
    result = calculator.areal_density

    # Assert
    expected = crater2.radius**2 * np.pi / 4 / study_region_size**2
    assert_almost_equal(result, expected, decimal=4)


def test_craters_outside_study_region_do_not_affect_areal_density():
    # Arrange
    # Single crater outside the study region, but with a radius that is inside the study region.
    study_region_size = 5000
    study_region_padding = 100
    calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=3
    )
    crater1 = Crater(id=1, x=0, y=0, radius=500)

    # Act
    calculator.add_crater(crater1)
    result = calculator.areal_density

    # Assert
    assert_almost_equal(result, 0, decimal=4)


def test_craters_smaller_than_r_stat_do_not_affect_areal_density():
    # Arrange
    # Single crater that is too small
    study_region_size = 5000
    study_region_padding = 100
    calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=10
    )
    crater1 = Crater(id=1, x=study_region_padding, y=study_region_padding, radius=9)

    # Act
    calculator.add_crater(crater1)
    result = calculator.areal_density

    # Assert
    assert_almost_equal(result, 0, decimal=4)
