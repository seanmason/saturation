import numpy as np
from typing import Tuple

import pandas as pd
from numpy.testing import assert_almost_equal
from saturation.geometry import get_xy_intersection, get_intersection_arc, get_erased_rim_arcs, calculate_areal_density


def assert_tuples_equal(t1: Tuple[float, float], t2: Tuple[float, float]):
    assert_almost_equal(t1[0], t2[0])
    assert_almost_equal(t1[1], t2[1])


def test_get_xy_intersection():
    # Act
    # Circle centered at (10, 10) with radius 10
    # Circle centered at (20, 10) with radius 10
    result = get_xy_intersection(
        (10, 10),
        (20, 10),
        10,
        10
    )

    # Assert
    # The math is a little ugly, this is just for regression purposes
    assert_tuples_equal(result[0], (15.0, 1.3397459621556145))
    assert_tuples_equal(result[1], (15.0, 18.660254037844386))


def test_get_intersection_arc():
    # Act
    # Circle centered at (10, 10) with radius 10
    # Circle centered at (20, 10) with radius 10
    result = get_intersection_arc(
        (10, 10),
        10,
        (20, 10),
        10
    )

    # Assert
    # The math is again ugly, this is just for regression purposes.
    expected = 5.235987755982989, 1.0471975511965976
    assert_tuples_equal(result, expected)


def test_get_intersection_arc_reversed():
    # Act
    # The circles are reversed; the result should still be the same arc.
    result = get_intersection_arc(
        (20, 10),
        10,
        (10, 10),
        10
    )

    # Assert
    # The math is again ugly, this is just for regression purposes.
    expected = 2.0943951023931957, 4.1887902047863905
    assert_tuples_equal(result, expected)


def test_get_intersection_arc_overlap():
    # Act
    # Circle 2 completely encompasses circle 1
    result = get_intersection_arc(
        (10, 10),
        1,
        (10, 10),
        10
    )

    # Assert
    # The math is again ugly, this is just for regression purposes.
    expected = 0, 2 * np.pi
    assert_tuples_equal(result, expected)


def test_get_erased_rim_arcs_complete_overlap():
    # Arrange
    # The second circle completely encompasses the first
    data = [
        {'id': 1, 'x': 10, 'y': 10, 'radius': 1},
        {'id': 2, 'x': 10, 'y': 10, 'radius': 10},
    ]
    craters = pd.DataFrame(data).set_index(['id'])

    # Act
    result = get_erased_rim_arcs(craters, 1.0)

    # Assert
    assert result.shape[0] == 1

    first_result = result.iloc[0]
    assert first_result.impacting_id == 2
    assert first_result.impacted_id == 1
    assert first_result.theta1 == 0
    assert first_result.theta2 == 2 * np.pi


def test_get_erased_rim_arcs_overlap():
    # Arrange
    # The second circle intersects the rim of the first
    data = [
        {'id': 1, 'x': 10, 'y': 10, 'radius': 10},
        {'id': 2, 'x': 20, 'y': 10, 'radius': 10},
    ]
    craters = pd.DataFrame(data).set_index(['id'])

    # Act
    result = get_erased_rim_arcs(craters, 1.0)

    # Assert
    assert result.shape[0] == 1

    first_result = result.iloc[0]
    assert first_result.impacting_id == 2
    assert first_result.impacted_id == 1
    assert first_result.theta1 == 5.235987755982989
    assert first_result.theta2 == 1.0471975511965976


def test_get_erased_rim_arcs_larger_overlap_with_effective_size():
    # Arrange
    # The second circle intersects the rim of the first
    data = [
        {'id': 1, 'x': 10, 'y': 10, 'radius': 10},
        {'id': 2, 'x': 20, 'y': 10, 'radius': 10},
    ]
    craters = pd.DataFrame(data).set_index(['id'])

    # Act
    result = get_erased_rim_arcs(craters, 1.3)

    # Assert
    assert result.shape[0] == 1

    first_result = result.iloc[0]
    assert first_result.impacting_id == 2
    assert first_result.impacted_id == 1
    assert first_result.theta1 == 4.868016433728875
    assert first_result.theta2 == 1.415168873450711


def test_calculate_areal_density_no_edges():
    # Arrange
    # A single crater that does not hit the edges
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 100},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    terrain_size = 1000

    # Act
    result = calculate_areal_density(craters, terrain_size, 0)

    # Assert
    # It won't be exact, because of discretization, but it should be close.
    expected = craters.iloc[0].radius ** 2 * np.pi / terrain_size**2
    assert abs(1 - result / expected) < 1e-3


def test_calculate_areal_density_uses_margin():
    # Arrange
    # A single crater that has a quarter of its area within the margin
    terrain_size = 5000
    margin = 100
    data = [
        {'id': 1, 'x': margin, 'y': margin, 'radius': 200},
    ]
    craters = pd.DataFrame(data).set_index(['id'])

    # Act
    result = calculate_areal_density(craters, terrain_size, margin)

    # Assert
    # It won't be exact, because of discretization, but it should be close.
    expected = craters.iloc[0].radius ** 2 * np.pi / 4 / (terrain_size - 2 * margin - 1) ** 2
    assert abs(1 - result / expected) < 1e-2


def test_calculate_areal_density_uses_both_margins():
    # Arrange
    # Two craters that have a quarter of each's area within the margin
    terrain_size = 5000
    margin = 100
    data = [
        {'id': 1, 'x': margin, 'y': margin, 'radius': 200},
        {'id': 2, 'x': terrain_size - 2 * margin - 1, 'y': terrain_size - 2 * margin - 1, 'radius': 500},
    ]
    craters = pd.DataFrame(data).set_index(['id'])

    # Act
    result = calculate_areal_density(craters, terrain_size, margin)

    # Assert
    # It won't be exact, because of discretization, but it should be close.
    crater1_area = craters.iloc[0].radius ** 2 * np.pi / 4
    crater2_area = craters.iloc[1].radius ** 2 * np.pi / 4
    expected = (crater1_area + crater2_area) / (terrain_size - 2 * margin)**2
    assert abs(1 - result / expected) < 1e-2