import numpy as np
import math

from data_structures.spatial_hash import _get_distance
from saturation.datatypes import Crater
from saturation.distances import Distances


def assert_float_equal(first: float, second: float, *, percentage: float = None, absolute: float = None):
    if percentage:
        assert abs(first - second) < percentage * second
    if absolute:
        assert abs(first - second) < absolute


def test_get_mean_center_to_center_nearest_neighbor_distance_no_craters():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)

    # Act
    dist = nn.get_center_to_center_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 0.0


def test_get_mean_rim_to_rim_nearest_neighbor_distance_no_craters():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)

    # Act
    dist = nn.get_rim_to_rim_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 0.0


def test_get_center_to_center_nearest_neighbor_distance_mean_single_pair_tracked():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    craters = [
        (Crater(id=1, x=10.0, y=10.0, radius=10.0), True),
        (Crater(id=2, x=20.0, y=10.0, radius=10.0), True)
    ]
    for crater, tracked in craters:
        nn.add(crater, tracked)

    # Act
    dist = nn.get_center_to_center_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 10.0


def test_get_rim_to_rim_nearest_neighbor_distance_mean_single_pair_tracked():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    craters = [
        (Crater(id=1, x=10.0, y=10.0, radius=10.0), True),
        (Crater(id=2, x=50.0, y=10.0, radius=10.0), True)
    ]
    for crater, tracked in craters:
        nn.add(crater, tracked)

    # Act
    dist = nn.get_rim_to_rim_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 20.0


def test_get_center_to_center_nearest_neighbor_distance_mean_single_pair_untracked():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    craters = [
        (Crater(id=1, x=10.0, y=10.0, radius=10.0), True),
        (Crater(id=2, x=20.0, y=10.0, radius=10.0), False),
        (Crater(id=3, x=10.0, y=11.0, radius=10.0), False),
    ]
    for crater, tracked in craters:
        nn.add(crater, tracked)

    # Act
    dist = nn.get_center_to_center_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 1.0


def test_get_rim_to_rim_nearest_neighbor_distance_mean_single_pair_untracked():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    craters = [
        (Crater(id=1, x=10.0, y=10.0, radius=10.0), True),
        (Crater(id=2, x=120.0, y=10.0, radius=10.0), False),
        (Crater(id=3, x=10.0, y=110.0, radius=10.0), False),
    ]
    for crater, tracked in craters:
        nn.add(crater, tracked)

    # Act
    dist = nn.get_rim_to_rim_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 100.0


def test_get_craters_with_overlapping_rims_no_overlaps():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    craters = [
        Crater(id=1, x=100.0, y=100.0, radius=50.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    overlaps = nn.get_craters_with_overlapping_rims(100, 100, 40)

    # Assert
    assert overlaps == set()


def test_get_craters_with_overlapping_rims_contained():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    craters = [
        Crater(id=1, x=100.0, y=100.0, radius=50.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    overlaps = nn.get_craters_with_overlapping_rims(100, 100, 60)

    # Assert
    assert overlaps == set(craters)


def test_get_craters_with_overlapping_rims_simple_case():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    craters = [
        Crater(id=1, x=100.0, y=100.0, radius=50.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    overlaps = nn.get_craters_with_overlapping_rims(150, 125, 50)

    # Assert
    assert overlaps == {craters[0]}


def test_get_craters_with_overlapping_rims_tiny_overlap():
    # Arrange
    nn = Distances(cell_size=50, boundary_min=0, boundary_max=500)
    craters = [
        Crater(id=1, x=100.0, y=100.0, radius=50.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    overlaps = nn.get_craters_with_overlapping_rims(200, 200, 92)

    # Assert
    assert overlaps == {craters[0]}


def test_get_mean_nearest_neighbor_distance_random_adds_and_deletes():
    N_REPEATS = 100
    N_POINTS = 300
    STUDY_REGION_SIZE = 500
    P_TRACKED = 0.8

    for n in range(N_REPEATS):
        print(n)

        # Arrange
        np.random.seed(n)

        nn = Distances(cell_size=5, boundary_min=0, boundary_max=STUDY_REGION_SIZE)
        points = np.random.rand(N_POINTS, 2) * STUDY_REGION_SIZE
        craters = [
            (Crater(id=x + 1, x=point[0], y=point[1], radius=np.random.rand() * 25 + 1), np.random.rand() < P_TRACKED)
            for x, point in enumerate(points)
        ]
        for crater, tracked in craters:
            nn.add(crater, tracked)

        # Randomly remove some points
        to_remove = [craters[x][0] for x in np.random.choice(N_POINTS, N_POINTS // 2, replace=False)]
        nn.remove(to_remove)

        # Act
        c2c_mean = nn.get_center_to_center_nearest_neighbor_distance_mean()
        c2c_min = nn.get_center_to_center_nearest_neighbor_distance_min()
        c2c_max = nn.get_center_to_center_nearest_neighbor_distance_max()
        c2c_stdev = nn.get_center_to_center_nearest_neighbor_distance_stdev()

        r2r_mean = nn.get_rim_to_rim_nearest_neighbor_distance_mean()
        r2r_max = nn.get_rim_to_rim_nearest_neighbor_distance_max()
        r2r_stdev = nn.get_rim_to_rim_nearest_neighbor_distance_stdev()

        r2r_non_zero_count = nn.get_n_non_zero_rim_to_rim_nearest_neighbor_distances()

        # Assert
        # Manually calculate the c2c distances
        tracked_after_removal = [x[0] for x in craters if x[1] and x[0] not in to_remove]
        c2c_nns = [
            sorted(((crater, x, _get_distance(x.x, x.y, crater.x, crater.y))
                    for x, _ in craters if x not in to_remove and x != crater), key=lambda x: x[2])[0]
            for crater in tracked_after_removal
        ]
        c2c_dists = [x[2] for x in c2c_nns]

        expected_c2c_mean = np.mean(c2c_dists)
        expected_c2c_min = np.min(c2c_dists)
        expected_c2c_max = np.max(c2c_dists)
        expected_c2c_stdev = np.std(c2c_dists, ddof=1)

        math.isclose(c2c_mean, expected_c2c_mean)
        math.isclose(c2c_min, expected_c2c_min)
        math.isclose(c2c_max, expected_c2c_max)
        math.isclose(c2c_stdev, expected_c2c_stdev)

        # Manually calculate the r2r distances
        r2r_nns = [
            sorted([(crater, x, max(_get_distance(x.x, x.y, crater.x, crater.y) - x.radius - crater.radius, 0.0))
                    for x, _ in craters if x not in to_remove and x != crater], key=lambda x: x[2])[0]
            for crater in tracked_after_removal
        ]
        r2r_dists = [x[2] for x in r2r_nns]

        expected_r2r_non_zero_count = len([x for x in r2r_dists if x != 0.0])
        expected_r2r_mean = np.mean(r2r_dists)
        expected_r2r_max = np.max(r2r_dists)
        expected_r2r_stdev = np.std(r2r_dists, ddof=1)

        assert expected_r2r_non_zero_count == r2r_non_zero_count
        math.isclose(r2r_mean, expected_r2r_mean)
        math.isclose(r2r_max, expected_r2r_max)
        math.isclose(r2r_stdev, expected_r2r_stdev)
