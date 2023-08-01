from datetime import datetime

import numpy as np

from data_structures.spatial_hash import _get_distance
from saturation.datatypes import Crater
from saturation.distances import Distances


def test_get_mean_center_to_center_nearest_neighbor_distance_no_craters():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    crater = Crater(id=1, x=10.0, y=10.0, radius=10.0)

    # Act
    dist = nn.get_center_to_center_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 0.0


def test_get_mean_rim_to_rim_nearest_neighbor_distance_no_craters():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    crater = Crater(id=1, x=10.0, y=10.0, radius=10.0)

    # Act
    dist = nn.get_rim_to_rim_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 0.0


def test_get_mean_nearest_neighbor_distance_single_pair():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    craters = [
        Crater(id=1, x=10.0, y=10.0, radius=10.0),
        Crater(id=2, x=20.0, y=10.0, radius=10.0)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    dist = nn.get_center_to_center_nearest_neighbor_distance_mean()

    # Assert
    assert dist == 10.0


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
    # Arrange
    np.random.seed(123456)
    start = datetime.now()
    N_POINTS = 1000
    STUDY_REGION_SIZE = 1000

    nn = Distances(cell_size=5, boundary_min=0, boundary_max=STUDY_REGION_SIZE)
    points = np.random.rand(N_POINTS, 2) * STUDY_REGION_SIZE
    craters = [
        Crater(id=x, x=point[0], y=point[1], radius=10.0)
        for x, point in enumerate(points)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Randomly remove some points
    to_remove = [craters[x] for x in np.random.choice(N_POINTS, N_POINTS // 2, replace=False)]
    nn.remove(to_remove)

    # Act
    reduced_set = [x for x in craters if x not in to_remove]

    # Print distances' calculations
    # print()
    # for crater in reduced_set:
    #     print(f"{crater} nn = {nn.get_center_to_center_nearest_neighbor(crater)}")

    c2c_mean = nn.get_center_to_center_nearest_neighbor_distance_mean()
    c2c_min = nn.get_center_to_center_nearest_neighbor_distance_min()
    c2c_max = nn.get_center_to_center_nearest_neighbor_distance_max()
    c2c_stdev = nn.get_center_to_center_nearest_neighbor_distance_stdev()
    
    r2r_mean = nn.get_rim_to_rim_nearest_neighbor_distance_mean()
    r2r_max = nn.get_rim_to_rim_nearest_neighbor_distance_max()
    r2r_stdev = nn.get_rim_to_rim_nearest_neighbor_distance_stdev()

    r2r_non_zero_mean = nn.get_rim_to_rim_non_zero_nearest_neighbor_distance_mean()
    r2r_non_zero_count = nn.get_rim_to_rim_non_zero_nearest_neighbor_distance_count()
    r2r_non_zero_min = nn.get_rim_to_rim_non_zero_nearest_neighbor_distance_min()
    r2r_non_zero_stdev = nn.get_rim_to_rim_non_zero_nearest_neighbor_distance_stdev()
    print(f"Took {datetime.now() - start} seconds")

    # Assert
    # Manually calculate the mean c2c distance
    # print("Expected:")
    # results_dists = [
    #     (crater,
    #      min((np.sqrt((x.x - crater.x) ** 2 + (x.y - crater.y) ** 2) for x in reduced_set if x != crater)),
    #      [x for x in reduced_set if x != crater][np.argmin([np.sqrt((x.x - crater.x) ** 2 + (x.y - crater.y) ** 2) for x in reduced_set if x != crater])]
    #      )
    #     for crater in reduced_set
    # ]
    # for crater, distance, dest_crater in results_dists:
    #     print(f"{crater} nn at {distance}, {dest_crater}")

    c2c_dists = [
        min((_get_distance(x.x, x.y, crater.x, crater.y) for x in reduced_set if x != crater))
        for crater in reduced_set
    ]
    expected_c2c_mean = np.mean(c2c_dists)
    expected_c2c_min = np.min(c2c_dists)
    expected_c2c_max = np.max(c2c_dists)
    expected_c2c_stdev = np.std(c2c_dists, ddof=1)

    assert abs(c2c_mean - expected_c2c_mean) / expected_c2c_mean < 1e-4
    assert abs(c2c_min - expected_c2c_min) / expected_c2c_min < 1e-4
    assert abs(c2c_max - expected_c2c_max) / expected_c2c_max < 1e-4
    assert abs(c2c_stdev - expected_c2c_stdev) / expected_c2c_stdev < 1e-4

    # Manually calculate the mean r2r distance
    # Print distances' calculations
    # print()
    # for crater in reduced_set:
    #     print(f"{crater} nn = {nn.get_rim_to_rim_nearest_neighbor(crater)}")
    #
    # print("Expected:")
    # results_dists = [
    #     (crater,
    #      min((max(0, np.sqrt((x.x - crater.x) ** 2 + (x.y - crater.y) ** 2) - x.radius - crater.radius)
    #           for x in reduced_set if x != crater)),
    #      [x for x in reduced_set if x != crater][
    #          np.argmin([max(0, np.sqrt((x.x - crater.x) ** 2 + (x.y - crater.y) ** 2) - x.radius - crater.radius)
    #                     for x in reduced_set if x != crater])]
    #      )
    #     for crater in reduced_set
    # ]
    # for crater, distance, dest_crater in results_dists:
    #     print(f"{crater} nn at {distance}, {dest_crater}")

    r2r_dists = [
        min((max(0.0, _get_distance(x.x, x.y, crater.x, crater.y) - x.radius - crater.radius)
             for x in reduced_set if x != crater))
        for crater in reduced_set
    ]
    expected_r2r_mean = np.mean(r2r_dists)
    expected_r2r_max = np.max(r2r_dists)
    expected_r2r_stdev = np.std(r2r_dists, ddof=1)

    assert abs(r2r_mean - expected_r2r_mean) / expected_r2r_mean < 1e-4
    assert abs(r2r_max - expected_r2r_max) / expected_r2r_max < 1e-4
    assert abs(r2r_stdev - expected_r2r_stdev) / expected_r2r_stdev < 1e-4

    r2r_non_zero_dists = [x for x in r2r_dists if x > 0.0]
    expected_r2r_non_zero_count = len(r2r_non_zero_dists)
    expected_r2r_non_zero_mean = np.mean(r2r_non_zero_dists)
    expected_r2r_non_zero_min = np.min(r2r_non_zero_dists)
    expected_r2r_non_zero_stdev = np.std(r2r_non_zero_dists, ddof=1)

    assert expected_r2r_non_zero_count == r2r_non_zero_count
    assert abs(r2r_non_zero_mean - expected_r2r_non_zero_mean) / expected_r2r_non_zero_mean < 1e-4
    assert abs(r2r_non_zero_min - expected_r2r_non_zero_min) / expected_r2r_non_zero_min < 1e-4
    assert abs(r2r_non_zero_stdev - expected_r2r_non_zero_stdev) / expected_r2r_non_zero_stdev < 1e-4

