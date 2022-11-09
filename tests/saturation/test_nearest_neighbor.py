import numpy as np
from numpy.testing import assert_almost_equal

from saturation.datatypes import Crater
from saturation.nearest_neighbor import NearestNeighbor


def test_get_all_nearest_neighbor_distances_no_craters():
    # Arrange
    nn = NearestNeighbor()
    crater = Crater(id=1, x=10, y=10, radius=10)

    # Act
    dist = nn.get_mean_nearest_neighbor_distance([crater])

    # Assert
    assert dist == 0.0


def test_get_mean_nearest_neighbor_distance_single_pair():
    # Arrange
    nn = NearestNeighbor()
    craters = [
        Crater(id=1, x=10, y=10, radius=10),
        Crater(id=2, x=20, y=10, radius=10)
    ]
    for crater in craters:
        nn.add(crater)

    # Act
    dist = nn.get_mean_nearest_neighbor_distance(craters)

    # Assert
    assert dist == 10.0


def test_get_mean_nearest_neighbor_distance_random_adds_and_deletes():
    # Arrange
    N_POINTS = 1000

    nn = NearestNeighbor()
    points = np.random.rand(N_POINTS, 2) * 10000
    craters = [
        Crater(id=x, x=point[0], y=point[1], radius=10)
        for x, point in enumerate(points)
    ]
    for crater in craters:
        nn.add(crater)

    # Randomly remove some points
    to_remove = [craters[x] for x in np.random.choice(N_POINTS, N_POINTS // 2, replace=False)]
    nn.remove(to_remove)

    # Act
    reduced_set = [x for x in craters if x not in to_remove]
    dist = nn.get_mean_nearest_neighbor_distance(reduced_set)

    # Assert
    results = []

    # Manually calculate the mean
    for crater in reduced_set:
        closest = min((np.sqrt((x.x - crater.x)**2 + (x.y - crater.y)**2) for x in reduced_set if x != crater))
        results.append(closest)

    expected = np.mean(results)
    assert dist == expected
