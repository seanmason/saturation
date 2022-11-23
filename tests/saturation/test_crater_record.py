import numpy as np

from saturation.crater_record import CraterRecord
from saturation.datatypes import Crater


def test_add_in_study_region():
    # Arrange
    crater = Crater(id=1, x=100, y=100, radius=100)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=1000,
        study_region_padding=100,
    )

    # Act
    removed = record.add(crater)
    all_craters = record.all_craters_in_record
    craters_in_study_region = record.craters_in_study_region
    n_craters_added = record.n_craters_added_in_study_region

    # Assert
    assert not removed
    assert [crater] == all_craters
    assert [crater] == craters_in_study_region
    assert n_craters_added == 1


def test_add_outside_study_region():
    # Arrange
    crater = Crater(id=1, x=100, y=100, radius=100)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=1000,
        study_region_padding=200,
    )

    # Act
    record.add(crater)
    all_craters = record.all_craters_in_record
    craters_in_study_region = record.craters_in_study_region
    n_craters_added = record.n_craters_added_in_study_region

    # Assert
    assert [crater] == all_craters
    assert not craters_in_study_region
    assert n_craters_added == 0


def test_add_does_not_add_small_craters():
    # Arrange
    crater = Crater(id=1, x=100, y=100, radius=9)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=1000,
        study_region_padding=100,
    )

    # Act
    record.add(crater)
    all_craters = record.all_craters_in_record
    n_craters_added = record.n_craters_added_in_study_region

    # Assert
    assert not all_craters
    assert n_craters_added == 0


def test_add_removes_obliterated_craters():
    # Arrange
    # Second crater completely obliterates the first.
    crater1 = Crater(id=1, x=100, y=100, radius=10)
    crater2 = Crater(id=2, x=110, y=110, radius=50)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=1000,
        study_region_padding=100,
    )

    # Act
    record.add(crater1)
    removed = record.add(crater2)
    all_craters = record.all_craters_in_record
    n_craters_added = record.n_craters_added_in_study_region

    # Assert
    assert [crater2] == all_craters
    assert [crater1] == removed
    assert n_craters_added == 2


def test_add_leaves_partially_removed_craters():
    # Arrange
    # Second crater partially destroys the first's rim.
    crater1 = Crater(id=1, x=100, y=100, radius=10)
    crater2 = Crater(id=2, x=110, y=110, radius=10)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=1000,
        study_region_padding=100,
    )

    # Act
    record.add(crater1)
    removed = record.add(crater2)
    all_craters = record.all_craters_in_record
    n_craters_added = record.n_craters_added_in_study_region

    # Assert
    assert {crater1, crater2} == set(all_craters)
    assert not removed
    assert n_craters_added == 2


def test_crater_rims_truncated_by_study_region_edges():
    # Arrange
    crater = Crater(id=1, x=0, y=0, radius=10)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=1000,
        study_region_padding=0,
    )

    # Act
    record.add(crater)

    # Assert
    assert record._erased_arcs[crater.id]


def test_crater_radius_ratio_respected():
    # Arrange
    # For a new crater to affect an old crater, (new crater radius) > (old crater radius) / r_stat_multiplier
    # Here we pepper crater1 with a bunch of craters below this ratio. crater1 should not be removed.
    crater1 = Crater(id=1, x=100, y=100, radius=10)
    crater2 = Crater(id=2, x=100, y=110, radius=4)
    crater3 = Crater(id=3, x=90, y=100, radius=4)
    crater4 = Crater(id=4, x=100, y=110, radius=4)
    crater5 = Crater(id=5, x=100, y=90, radius=4)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=2,
        min_rim_percentage=0.9,
        effective_radius_multiplier=1.0,
        study_region_size=1000,
        study_region_padding=0,
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    record.add(crater4)
    record.add(crater5)
    all_craters = record.all_craters_in_record

    # Assert
    assert all_craters == [crater1]


def test_get_mean_nearest_neighbor_distance_empty_from():
    # Arrange
    crater1 = Crater(id=1, x=200, y=200, radius=10)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=100,
        study_region_padding=100
    )

    # Act
    record.add(crater1)
    result = record.get_mean_nearest_neighbor_distance()

    # Assert
    assert result == 0


def test_nearest_neighbor_single_from_and_to():
    # Arrange
    crater1 = Crater(id=1, x=50, y=150, radius=10)
    crater2 = Crater(id=2, x=150, y=150, radius=10)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=100,
        study_region_padding=100,
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    result = record.get_mean_nearest_neighbor_distance()

    # Assert
    assert result == 100


def test_nearest_neighbor_gets_shortest_distance_from_to_craters():
    # Arrange
    crater1 = Crater(id=1, x=100, y=100, radius=10)
    crater2 = Crater(id=2, x=100, y=210, radius=10)
    crater3 = Crater(id=3, x=100, y=230, radius=10)
    crater4 = Crater(id=4, x=99, y=100, radius=10)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.0,
        effective_radius_multiplier=1.0,
        study_region_size=100,
        study_region_padding=100,
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    record.add(crater4)
    result = record.get_mean_nearest_neighbor_distance()

    # Assert
    assert result == 1


def test_get_mean_nearest_neighbor_distance_is_shortest_distance_for_all_from_craters():
    # Arrange
    crater1 = Crater(id=1, x=100, y=100, radius=10)
    crater2 = Crater(id=2, x=100, y=120, radius=10)
    crater3 = Crater(id=3, x=110, y=100, radius=10)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.0,
        effective_radius_multiplier=1.0,
        study_region_size=100,
        study_region_padding=100,
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    result = record.get_mean_nearest_neighbor_distance()

    # Assert
    assert result == np.mean([10, 20, 10])


def test_get_mean_nearest_neighbor_distance_ignores_smaller_than_r_stat():
    # Arrange
    crater1 = Crater(id=1, x=100, y=100, radius=10)
    crater2 = Crater(id=2, x=100, y=120, radius=10)
    crater3 = Crater(id=3, x=110, y=100, radius=5)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=100,
        study_region_padding=100,
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    result = record.get_mean_nearest_neighbor_distance()

    # Assert
    assert result == np.mean([20, 20])


def test_get_mean_nearest_neighbor_distance_ignores_removed_craters():
    # Arrange
    crater1 = Crater(id=1, x=100, y=100, radius=10)
    crater2 = Crater(id=2, x=100, y=150, radius=10)
    crater3 = Crater(id=3, x=100, y=160, radius=50)
    record = CraterRecord(
        r_stat=10,
        r_stat_multiplier=3,
        min_rim_percentage=0.5,
        effective_radius_multiplier=1.0,
        study_region_size=100,
        study_region_padding=100,
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    result = record.get_mean_nearest_neighbor_distance()

    # Assert
    assert result == np.mean([60, 60])

