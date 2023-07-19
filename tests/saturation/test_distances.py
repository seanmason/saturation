from datetime import datetime

import numpy as np

from saturation.datatypes import Crater
from saturation.distances import Distances


def test_get_all_nearest_neighbor_distances_no_craters():
    # Arrange
    nn = Distances(cell_size=50,  boundary_min=0, boundary_max=500)
    crater = Crater(id=1, x=10.0, y=10.0, radius=10.0)

    # Act
    dist = nn.get_mean_nearest_neighbor_distance()

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
    dist = nn.get_mean_nearest_neighbor_distance()

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
    print()
    for crater in reduced_set:
        print(f"{crater} nn = {nn.get_nearest_neighbor(crater)}")

    dist = nn.get_mean_nearest_neighbor_distance()
    print(f"Took {datetime.now() - start} seconds")

    # Assert
    # Manually calculate the mean
    print("Expected:")
    results_dists = [
        (crater,
         min((np.sqrt((x.x - crater.x) ** 2 + (x.y - crater.y) ** 2) for x in reduced_set if x != crater)),
         reduced_set[np.argmin([np.sqrt((x.x - crater.x) ** 2 + (x.y - crater.y) ** 2) for x in reduced_set if x != crater])]
         )
        for crater in reduced_set
    ]
    for crater, distance, dest_crater in results_dists:
        print(f"{crater} nn at {distance}, {dest_crater}")

    results = [
        min((np.sqrt((x.x - crater.x) ** 2 + (x.y - crater.y) ** 2) for x in reduced_set if x != crater))
        for crater in reduced_set
    ]
    expected = np.mean(results)

    assert abs(dist - expected) < 0.0001
