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


def test_get_all_nearest_neighbor_distances_single_pair():
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
