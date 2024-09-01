import numpy as np
from numpy.testing import assert_almost_equal

from saturation.datatypes import Crater
from saturation.rim_erasure_calculators import (
    RadiusRatioConditionalRimOverlapRimErasureCalculator,
    SqrtRadiusConditionalRimOverlapRimErasureCalculator,
    LogRadiusConditionalRimOverlapRimErasureCalculator,
    SinLogRadiusConditionalRimOverlapRimErasureCalculator,
)


def test_radius_ratio_conditional_no_overlap():
    # Arrange
    calculator = RadiusRatioConditionalRimOverlapRimErasureCalculator(7.0, 1.0)
    existing = Crater(id=1, x=0, y=0, radius=50)
    new = Crater(id=2, x=0, y=0, radius=5)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert result == 50.0


def test_radius_ratio_conditional_totally_inside():
    # Arrange
    calculator = RadiusRatioConditionalRimOverlapRimErasureCalculator(7.0, 1.0)
    existing = Crater(id=1, x=0, y=0, radius=50)
    new = Crater(id=2, x=0, y=0, radius=5)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert result == 50.0


def test_radius_ratio_conditional_overlap():
    # Arrange
    calculator = RadiusRatioConditionalRimOverlapRimErasureCalculator(10.0, 1.0)
    existing = Crater(id=1, x=0, y=0, radius=5)
    new = Crater(id=2, x=0, y=existing.radius, radius=5)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert_almost_equal(result, 50 * (2 / 3))


def test_radius_ratio_conditional_overlap_too_small():
    # Arrange
    calculator = RadiusRatioConditionalRimOverlapRimErasureCalculator(1.0, 1.0)
    existing = Crater(id=1, x=0, y=0, radius=50)
    new = Crater(id=2, x=0, y=existing.radius, radius=5)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert result == 50.0


def test_log_radius_conditional_overlap():
    # Arrange
    calculator = LogRadiusConditionalRimOverlapRimErasureCalculator(1.0)
    existing = Crater(id=1, x=0, y=0, radius=5)
    new = Crater(id=2, x=0, y=existing.radius, radius=5)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert_almost_equal(result, 50 * (2 / 3))


def test_log_radius_conditional_overlap_too_small():
    # Arrange
    calculator = LogRadiusConditionalRimOverlapRimErasureCalculator(1.0)
    existing = Crater(id=1, x=0, y=0, radius=np.exp(5))
    new = Crater(id=2, x=0, y=existing.radius, radius=4)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert result == 50.0


def test_log_radius_conditional_overlap_large_enough():
    # Arrange
    calculator = LogRadiusConditionalRimOverlapRimErasureCalculator(1.0)
    existing = Crater(id=1, x=0, y=0, radius=np.exp(5))
    new = Crater(id=2, x=0, y=existing.radius, radius=5.1)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert_almost_equal(result, 50 * 0.9890612105165573)


def test_sqrt_radius_conditional_overlap_too_small():
    # Arrange
    calculator = SqrtRadiusConditionalRimOverlapRimErasureCalculator(1.0)
    existing = Crater(id=1, x=0, y=0, radius=5**2)
    new = Crater(id=2, x=0, y=existing.radius, radius=4)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert result == 50.0


def test_sqrt_radius_conditional_overlap_large_enough():
    # Arrange
    calculator = SqrtRadiusConditionalRimOverlapRimErasureCalculator(1.0)
    existing = Crater(id=1, x=0, y=0, radius=5**2)
    new = Crater(id=2, x=0, y=existing.radius, radius=5.1)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert_almost_equal(result, 50.0 * 0.9349516551049367)


def test_sin_log_radius_conditional_overlap_too_small():
    # Arrange
    calculator = SinLogRadiusConditionalRimOverlapRimErasureCalculator(
        n_periods=3.0,
        min_r_period=5.0,
        max_r_period=50.0,
        rmult=1.0
    )
    existing = Crater(id=1, x=0, y=0, radius=5)
    new = Crater(id=2, x=0, y=existing.radius, radius=1.)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert result == 50.0


def test_sin_log_radius_conditional_overlap_large_enough():
    # Arrange
    calculator = SinLogRadiusConditionalRimOverlapRimErasureCalculator(
        n_periods=3.0,
        min_r_period=5.0,
        max_r_period=50.0,
        rmult=1.0
    )
    existing = Crater(id=1, x=0, y=0, radius=5)
    new = Crater(id=2, x=0, y=existing.radius, radius=3.)

    # Act
    result = calculator.calculate_new_rim_state(existing, 50, new)

    # Assert
    assert_almost_equal(result, 50.0 * 0.8060266319586433)