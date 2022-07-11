import numpy as np
from typing import Tuple
from numpy.testing import assert_almost_equal
from saturation.geometry import get_xy_intersection, get_intersection_arc


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
        (20, 10),
        10,
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
        (10, 10),
        10,
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
        (10, 10),
        1,
        10
    )

    # Assert
    # The math is again ugly, this is just for regression purposes.
    expected = 0, 2 * np.pi
    assert_tuples_equal(result, expected)
