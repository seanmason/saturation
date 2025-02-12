import numpy as np
from numpy.testing import assert_almost_equal

from saturation.datatypes import Crater
from saturation.rim_erasure_calculators import ExponentRadiusConditionalRimOverlapRimErasureCalculator


def test_exponent_radius_conditional_overlap_too_small():
    # Arrange
    calculator = ExponentRadiusConditionalRimOverlapRimErasureCalculator(
        exponent=0.5,
        ratio=1.0,
        rmult=1.0,
        rstat=10.0
    )
    existing = Crater(id=1, x=0, y=0, radius=5**2)
    new = Crater(id=2, x=0, y=existing.radius, radius=4)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert result == 50.0


def test_exponent_radius_conditional_overlap_large_enough():
    # Arrange
    calculator = ExponentRadiusConditionalRimOverlapRimErasureCalculator(
        exponent=0.5,
        ratio=1.0,
        rmult=1.0,
        rstat=10.0
    )
    existing = Crater(id=1, x=0, y=0, radius=5**2)
    new = Crater(id=2, x=0, y=existing.radius, radius=5.1)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert_almost_equal(result, 50.0 * 0.9349516551049367)


def test_exponent_radius_conditional_overlap_large_enough_with_ratio():
    # Arrange
    ratio = 3.0
    calculator = ExponentRadiusConditionalRimOverlapRimErasureCalculator(
        exponent=0.5,
        ratio=ratio,
        rmult=1.0,
        rstat=10.0
    )
    existing = Crater(id=1, x=0, y=0, radius=5**2 / ratio)
    new = Crater(id=2, x=0, y=existing.radius, radius=5.1)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert_almost_equal(result, 40.100924215514105)


def test_get_min_radius_threshold():
    # Arrange
    ratio = 3.0
    calculator = ExponentRadiusConditionalRimOverlapRimErasureCalculator(
        exponent=0.5,
        ratio=ratio,
        rmult=1.0,
        rstat=10.0
    )

    # Assert
    assert_almost_equal(calculator.get_min_radius_threshold(), 10.0**0.5 / ratio, decimal=5)
