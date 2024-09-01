import numpy as np

from saturation.crater_record import CraterRecord
from saturation.datatypes import Crater
from saturation.rim_erasure_calculators import get_rim_erasure_calculator


def _create_crater(id: int, x: float, y: float, radius: float) -> Crater:
    return Crater(id=np.int64(id), x=np.float32(x), y=np.float32(y), radius=np.float32(radius))


def effectiveness_function(existing: Crater, new: Crater) -> float:
    if existing.radius < new.radius / 3.0:
        return existing.radius * np.pi * 2

    return 0.0


def test_add_in_study_region():
    # Arrange
    crater = _create_crater(id=1, x=100.0, y=100.0, radius=100.0)
    record = CraterRecord(
        r_stat=10.0,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=1000,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    removed = record.add(crater)
    all_craters = record.all_craters_in_record
    craters_in_study_region = record.craters_in_study_region
    ntot = record.ntot

    # Assert
    assert not removed
    assert [crater] == list(all_craters)
    assert [crater] == list(craters_in_study_region)
    assert ntot == 1


def test_add_outside_study_region():
    # Arrange
    crater = _create_crater(id=1, x=100.0, y=100.0, radius=100.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=1000,
        study_region_padding=200,
        cell_size=50
    )

    # Act
    record.add(crater)
    all_craters = record.all_craters_in_record
    craters_in_study_region = record.craters_in_study_region
    ntot = record.ntot

    # Assert
    assert [crater] == list(all_craters)
    assert not craters_in_study_region
    assert ntot == 0


def test_add_does_not_add_small_craters():
    # Arrange
    crater = _create_crater(id=1, x=100.0, y=100.0, radius=9.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=1000,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater)
    all_craters = record.all_craters_in_record
    ntot = record.ntot

    # Assert
    assert not all_craters
    assert ntot == 0


def test_add_removes_obliterated_craters():
    # Arrange
    # Second crater completely obliterates the first.
    crater1 = _create_crater(id=1, x=100.0, y=100.0, radius=10.0)
    crater2 = _create_crater(id=2, x=110.0, y=110.0, radius=50.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=1000,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater1)
    removed = record.add(crater2)
    all_craters = record.all_craters_in_record
    ntot = record.ntot

    # Assert
    assert [crater2] == list(all_craters)
    assert [crater1] == list(removed)
    assert ntot == 2


def test_add_leaves_partially_removed_craters():
    # Arrange
    # Second crater partially destroys the first's rim.
    crater1 = _create_crater(id=1, x=100.0, y=100.0, radius=10.0)
    crater2 = _create_crater(id=2, x=110.0, y=110.0, radius=10.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=1000,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater1)
    removed = record.add(crater2)
    all_craters = record.all_craters_in_record
    ntot = record.ntot

    # Assert
    assert {crater1, crater2} == set(all_craters)
    assert not removed
    assert ntot == 2


def test_crater_radius_ratio_respected():
    # Arrange
    # For a new crater to affect an old crater, (new crater radius) > (old crater radius) / erat
    # Here we pepper crater1 with a bunch of craters below this ratio. crater1 should not be removed.
    crater1 = _create_crater(id=1, x=100.0, y=100.0, radius=10.0)
    crater2 = _create_crater(id=2, x=100.0, y=110.0, radius=4.0)
    crater3 = _create_crater(id=3, x=90.0, y=100.0, radius=4.0)
    crater4 = _create_crater(id=4, x=100.0, y=110.0, radius=4.0)
    crater5 = _create_crater(id=5, x=100.0, y=90.0, radius=4.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.9,
        rmult=1.0,
        study_region_size=1000,
        study_region_padding=0,
        cell_size=50
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    record.add(crater4)
    record.add(crater5)
    all_craters = record.all_craters_in_record

    # Assert
    assert list(all_craters) == [crater1]


def test_get_mnnd_empty_from():
    # Arrange
    crater1 = _create_crater(id=1, x=200.0, y=200.0, radius=10.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=100,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater1)
    result = record.get_mnnd()

    # Assert
    assert result == 0


def test_nn_single_from_and_to():
    # Arrange
    crater1 = _create_crater(id=1, x=50.0, y=150.0, radius=10.0)
    crater2 = _create_crater(id=2, x=150.0, y=150.0, radius=10.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=100,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    result = record.get_mnnd()

    # Assert
    assert result == 100


def test_nn_gets_shortest_distance_from_to_craters():
    # Arrange
    crater1 = _create_crater(id=1, x=100.0, y=100.0, radius=10.0)
    crater2 = _create_crater(id=2, x=100.0, y=210.0, radius=10.0)
    crater3 = _create_crater(id=3, x=100.0, y=230.0, radius=10.0)
    crater4 = _create_crater(id=4, x=99.0, y=100.0, radius=10.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.0,
        rmult=1.0,
        study_region_size=100,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    record.add(crater4)
    result = record.get_mnnd()

    # Assert
    assert result == 1


def test_get_mean_nnd_is_shortest_distance_for_all_from_craters():
    # Arrange
    crater1 = _create_crater(id=1, x=100.0, y=100.0, radius=10.0)
    crater2 = _create_crater(id=2, x=100.0, y=120.0, radius=10.0)
    crater3 = _create_crater(id=3, x=110.0, y=100.0, radius=10.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.0,
        rmult=1.0,
        study_region_size=100,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    result = record.get_mnnd()

    # Assert
    assert result == np.mean([10, 20, 10])


def test_get_mean_nnd_ignores_smaller_than_r_stat():
    # Arrange
    crater1 = _create_crater(id=1, x=100.0, y=100.0, radius=10.0)
    crater2 = _create_crater(id=2, x=100.0, y=120.0, radius=10.0)
    crater3 = _create_crater(id=3, x=110.0, y=100.0, radius=5.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=100,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    result = record.get_mnnd()

    # Assert
    assert result == np.mean([20, 20])


def test_get_mean_nnd_ignores_removed_craters():
    # Arrange
    crater1 = _create_crater(id=1, x=100.0, y=100.0, radius=10.0)
    crater2 = _create_crater(id=2, x=100.0, y=150.0, radius=10.0)
    crater3 = _create_crater(id=3, x=100.0, y=160.0, radius=50.0)
    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.5,
        rmult=1.0,
        study_region_size=100,
        study_region_padding=100,
        cell_size=50
    )

    # Act
    record.add(crater1)
    record.add(crater2)
    record.add(crater3)
    result = record.get_mnnd()

    # Assert
    assert result == np.mean([60, 60])


def test_removal_percentage_too_low():
    # Arrange
    center = 500.0
    large_radius = 100.0
    percent_each_crater = 0.05
    n_craters = 9

    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=effectiveness_function,
        mrp=0.51,
        rmult=1.0,
        study_region_size=500,
        study_region_padding=0,
        cell_size=5
    )

    # Act
    crater1 = _create_crater(id=1, x=center, y=center, radius=large_radius)
    record.add(crater1)

    small_theta = 2 * np.pi * percent_each_crater / 2
    small_radius = np.sqrt(
        (large_radius * np.sin(small_theta))**2
        + (large_radius - large_radius * np.cos(small_theta))**2
    )
    theta_delta = 2 * np.pi / n_craters
    for offset in range(n_craters):
        x = center + large_radius * np.cos(theta_delta * offset)
        y = center + large_radius * np.sin(theta_delta * offset)
        crater = _create_crater(id=offset + 2, x=x, y=y, radius=small_radius)
        removed = record.add(crater)

    assert crater1 in record.all_craters_in_record


def test_removal_percentage_high_enough():
    # Arrange
    center = 500.0
    large_radius = 100.0
    percent_each_crater = 0.05
    n_craters = 10

    func = get_rim_erasure_calculator({
        "name": "multiplier",
        "multiplier": 10.0
    }, 1.0)

    record = CraterRecord(
        r_stat=10,
        rim_erasure_calculator=func,
        mrp=0.51,
        rmult=1.0,
        study_region_size=500,
        study_region_padding=0,
        cell_size=5
    )

    # Act
    crater1 = _create_crater(id=1, x=center, y=center, radius=large_radius)
    record.add(crater1)

    small_theta = 2 * np.pi * percent_each_crater / 2
    small_radius = np.sqrt(
        (large_radius * np.sin(small_theta))**2
        + (large_radius - large_radius * np.cos(small_theta))**2
    )
    theta_delta = 2 * np.pi / n_craters
    for offset in range(n_craters):
        x = center + large_radius * np.cos(theta_delta * offset)
        y = center + large_radius * np.sin(theta_delta * offset)
        crater = _create_crater(id=offset + 2, x=x, y=y, radius=small_radius)
        removed = record.add(crater)

    assert crater1 not in record.all_craters_in_record
