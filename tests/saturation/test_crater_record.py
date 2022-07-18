import pandas as pd

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
        effective_radius_multiplier=1.0
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
        effective_radius_multiplier=1.0
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
        effective_radius_multiplier=1.0
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
        effective_radius_multiplier=1.0
    )

    # Act
    record.update(1)
    record.update(2)
    result = record.get_craters()

    # Assert
    pd.testing.assert_frame_equal(craters, result)
