from datetime import datetime

import numpy as np

from saturation.datatypes import Crater
from saturation.distances import Distances


def test_get_all_nearest_neighbor_distances_no_craters():
    # Arrange
    nn = Distances(1000)
    crater = Crater(id=1, x=10, y=10, radius=10)

    # Act
    dist = nn.get_mean_nearest_neighbor_distance([crater])

    # Assert
    assert dist == 0.0


def test_get_mean_nearest_neighbor_distance_single_pair():
    # Arrange
    nn = Distances(1000)
    craters = [
        Crater(id=1, x=10, y=10, radius=10),
        Crater(id=2, x=20, y=10, radius=10)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Act
    dist = nn.get_mean_nearest_neighbor_distance(craters)

    # Assert
    assert dist == 10.0


def test_get_mean_nearest_neighbor_distance_random_adds_and_deletes():
    # Arrange
    start = datetime.now()
    N_POINTS = 1000
    STUDY_REGION_SIZE = 12500

    nn = Distances(STUDY_REGION_SIZE * 1.5)
    points = np.random.rand(N_POINTS, 2) * STUDY_REGION_SIZE
    craters = [
        Crater(id=x, x=point[0], y=point[1], radius=10)
        for x, point in enumerate(points)
    ]
    for crater in craters:
        nn.add(crater, True)

    # Randomly remove some points
    to_remove = [craters[x] for x in np.random.choice(N_POINTS, N_POINTS // 2, replace=False)]
    nn.remove(to_remove)

    # Act
    reduced_set = [x for x in craters if x not in to_remove]
    dist = nn.get_mean_nearest_neighbor_distance(reduced_set)
    print(f"Took {datetime.now() - start} seconds")

    # Assert
    # Manually calculate the mean
    results = [
        min((np.sqrt((x.x - crater.x) ** 2 + (x.y - crater.y) ** 2) for x in reduced_set if x != crater))
        for crater in reduced_set
    ]
    expected = np.mean(results)

    assert abs(dist - expected) < 0.0001
