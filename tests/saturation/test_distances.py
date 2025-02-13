import numpy as np
import math

from saturation.data_structures.spatial_hash import _get_distance
from saturation.datatypes import Crater
from saturation.distances import Distances


def _create_crater(id: int, x: float, y: float, radius: float) -> Crater:
    return Crater(id=id, x=np.float64(x), y=np.float64(y), radius=np.float64(radius))


def assert_float_equal(first: float, second: float, *, percentage: float = None, absolute: float = None):
    if percentage:
        assert abs(first - second) < percentage * second
    if absolute:
        assert abs(first - second) < absolute


def test_get_mnnd_no_craters():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=True)

    # Act
    dist = nn.get_mnnd()

    # Assert
    assert dist == 0.0


def test_get_mnnd_single_pair_tracked():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=True)
    craters = [
        (_create_crater(id=1, x=10.0, y=10.0, radius=10.0), True),
        (_create_crater(id=2, x=20.0, y=10.0, radius=10.0), True)
    ]
    for crater, tracked in craters:
        nn.add(crater, tracked)

    # Act
    dist = nn.get_mnnd()

    # Assert
    assert dist == 10.0


def test_add_and_get_craters_with_overlapping_rims():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=False)
    crater1 = _create_crater(id=1, x=10.0, y=10.0, radius=10.0)

    # Act
    nn.add(crater1, True)

    # Assert
    found = nn.get_craters_with_overlapping_rims(
        crater1.x + 5,
        crater1.y + 5,
        10
    )
    assert len(found) == 1
    assert crater1.id in found
    assert len(nn.get_craters_with_overlapping_rims(crater1.x + 50, crater1.y + 50, 10)) == 0


def test_add_and_remove_and_get_craters_with_overlapping_rims():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=False)
    crater1 = _create_crater(id=1, x=10.0, y=10.0, radius=10.0)

    # Act
    nn.add(crater1, True)
    nn.remove([crater1])

    # Assert
    found = nn.get_craters_with_overlapping_rims(
        crater1.x + 5,
        crater1.y + 5,
        10
    )
    assert len(found) == 0


def test_get_mnnd_single_pair_untracked():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=True)
    craters = [
        (_create_crater(id=1, x=10.0, y=10.0, radius=10.0), True),
        (_create_crater(id=2, x=20.0, y=10.0, radius=10.0), False),
        (_create_crater(id=3, x=10.0, y=11.0, radius=10.0), False),
    ]
    for crater, tracked in craters:
        nn.add(crater, tracked)

    # Act
    dist = nn.get_mnnd()

    # Assert
    assert dist == 1.0


def test_get_craters_with_overlapping_rims_no_overlaps():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=True)
    craters = [
        _create_crater(id=1, x=100.0, y=100.0, radius=50.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    overlaps = nn.get_craters_with_overlapping_rims(100, 100, 40)

    # Assert
    assert overlaps == set()


def test_get_craters_with_overlapping_rims_contained():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=True)
    craters = [
        _create_crater(id=1, x=100.0, y=100.0, radius=50.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    overlaps = nn.get_craters_with_overlapping_rims(100, 100, 60)

    # Assert
    assert overlaps == {1}


def test_get_craters_with_overlapping_rims_simple_case():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=True)
    craters = [
        _create_crater(id=1, x=100.0, y=100.0, radius=50.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    overlaps = nn.get_craters_with_overlapping_rims(150, 125, 50)

    # Assert
    assert overlaps == {1}


def test_get_craters_with_overlapping_rims_tiny_overlap():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500, calculate_nearest_neighbor_stats=True)
    craters = [
        _create_crater(id=1, x=100.0, y=100.0, radius=50.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    overlaps = nn.get_craters_with_overlapping_rims(200, 200, 92)

    # Assert
    assert overlaps == {1}


def test_get_mean_nnd_random_adds_and_deletes():
    N_REPEATS = 10
    N_POINTS = 300
    STUDY_REGION_SIZE = 500
    P_TRACKED = 0.8

    for n in range(N_REPEATS):
        # Arrange
        np.random.seed(n)

        nn = Distances(cell_size=5, boundary_min=0, boundary_max=STUDY_REGION_SIZE, calculate_nearest_neighbor_stats=True)
        points = np.random.rand(N_POINTS, 2) * STUDY_REGION_SIZE
        craters = [
            (
                _create_crater(id=x + 1, x=point[0], y=point[1], radius=np.random.rand() * 25 + 1),
                np.random.rand() < P_TRACKED)
            for x, point in enumerate(points)
        ]
        for crater, tracked in craters:
            nn.add(crater, tracked)

        # Randomly remove some points
        to_remove = [craters[x][0] for x in np.random.choice(N_POINTS, N_POINTS // 2, replace=False)]
        nn.remove(to_remove)

        # Act
        mnnd = nn.get_mnnd()
        nnd_min = nn.get_min_nnd()
        nnd_max = nn.get_max_nnd()
        nnd_stdev = nn.get_nnd_stdev()

        # Assert
        # Manually calculate the nnds
        tracked_after_removal = [x[0] for x in craters if x[1] and x[0] not in to_remove]
        nns = [
            sorted(((crater, x, _get_distance(x.x, x.y, crater.x, crater.y))
                    for x, _ in craters if x not in to_remove and x != crater), key=lambda x: x[2])[0]
            for crater in tracked_after_removal
        ]
        nnds = [x[2] for x in nns]

        expected_mnnd = np.mean(nnds)
        expected_nnd_min = np.min(nnds)
        expected_nnd_max = np.max(nnds)
        expected_nnd_stdev = np.std(nnds, ddof=1)

        math.isclose(mnnd, expected_mnnd)
        math.isclose(nnd_min, expected_nnd_min)
        math.isclose(nnd_max, expected_nnd_max)
        math.isclose(nnd_stdev, expected_nnd_stdev)
