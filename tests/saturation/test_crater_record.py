import numpy
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from saturation.crater_record import CraterRecord


def test_update_adds_to_craters():
    # Arrange
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 100},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    record = CraterRecord(
        craters,
        min_crater_radius_for_stats=10,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        terrain_size=1000,
        margin=100
    )

    # Act
    record.update(1)
    result = record.get_craters()

    # Assert
    pd.testing.assert_frame_equal(result, craters)


def test_update_does_not_add_small_craters():
    # Arrange
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 100},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    record = CraterRecord(
        craters,
        min_crater_radius_for_stats=1000,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        terrain_size=1000,
        margin=100
    )

    # Act
    record.update(1)
    result = record.get_craters()

    # Assert
    assert result.shape[0] == 0


def test_update_removes_obliterated_craters():
    # Arrange
    # Second crater completely obliterates the first.
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 10},
        {'id': 2, 'x': 100, 'y': 100, 'radius': 100},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    record = CraterRecord(
        craters,
        min_crater_radius_for_stats=10,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        terrain_size=1000,
        margin=100
    )

    # Act
    record.update(1)
    record.update(2)
    result = record.get_craters()

    # Assert
    expected = craters.loc[[2]]
    pd.testing.assert_frame_equal(expected, result)


def test_update_leaves_partially_removed_craters():
    # Arrange
    # Second crater partially destroys the first's rim.
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 10},
        {'id': 2, 'x': 100, 'y': 110, 'radius': 10},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    record = CraterRecord(
        craters,
        min_crater_radius_for_stats=10,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        terrain_size=1000,
        margin=100
    )

    # Act
    record.update(1)
    record.update(2)
    result = record.get_craters()

    # Assert
    pd.testing.assert_frame_equal(craters, result)


def test_get_nearest_neighbor_distances_two_craters():
    # Arrange
    # Two non-overlapping craters
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 10},
        {'id': 2, 'x': 100, 'y': 200, 'radius': 10},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    record = CraterRecord(
        craters,
        min_crater_radius_for_stats=10,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        terrain_size=1000,
        margin=100
    )

    # Act
    record.update(1)
    record.update(2)
    result = record.get_nearest_neighbor_distances()

    # Assert
    assert_array_almost_equal(result, [100., 100.])


def test_get_nearest_neighbor_distances_three_craters_one_destroyed():
    # Arrange
    # Three craters
    # Third crater destroys the first.
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 10},
        {'id': 2, 'x': 100, 'y': 200, 'radius': 10},
        {'id': 3, 'x': 100, 'y': 110, 'radius': 50},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    record = CraterRecord(
        craters,
        min_crater_radius_for_stats=10,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        terrain_size=1000,
        margin=100
    )

    # Act
    record.update(1)
    record.update(2)
    record.update(3)
    result = record.get_nearest_neighbor_distances()

    # Assert
    assert_array_almost_equal(result, [90., 90.])


def test_get_nearest_neighbor_distances_three_non_overlapping_craters():
    # Arrange
    # Three non-overlapping craters all along x=100
    data = [
        {'id': 1, 'x': 100, 'y': 100, 'radius': 10},
        {'id': 2, 'x': 100, 'y': 150, 'radius': 10},
        {'id': 3, 'x': 100, 'y': 180, 'radius': 20},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    record = CraterRecord(
        craters,
        min_crater_radius_for_stats=10,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        terrain_size=1000,
        margin=100
    )

    # Act
    record.update(1)
    record.update(2)
    record.update(3)
    result = np.sort(record.get_nearest_neighbor_distances())

    # Assert
    assert_array_almost_equal(result, [30., 30., 50.])


def test_get_nearest_neighbor_distances_respects_terrain_limits():
    # Arrange
    # First and fourth craters are outside the terrain limits.
    data = [
        {'id': 1, 'x': 100, 'y': 50, 'radius': 10},
        {'id': 2, 'x': 100, 'y': 100, 'radius': 10},
        {'id': 3, 'x': 100, 'y': 700, 'radius': 10},
        {'id': 4, 'x': 100, 'y': 1100, 'radius': 10},
    ]
    craters = pd.DataFrame(data).set_index(['id'])
    record = CraterRecord(
        craters,
        min_crater_radius_for_stats=10,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        terrain_size=1000,
        margin=100
    )

    # Act
    record.update(1)
    record.update(2)
    record.update(3)
    record.update(4)
    result = record.get_nearest_neighbor_distances()

    # Assert
    assert_array_almost_equal(result, [50., 400.])
