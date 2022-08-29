from saturation.nearest_neighbor_calculator import NearestNeighborCalculator
from saturation.datatypes import Crater


def test_nearest_neighbor_empty_from():
    # Arrange
    from_craters = []
    to_craters = []
    calculator = NearestNeighborCalculator()

    # Act
    result = calculator.get_nearest_neighbor_distances(from_craters, to_craters)

    # Assert
    assert not result


def test_nearest_neighbor_single_from_and_to():
    # Arrange
    from_craters = [
        Crater(id=1, x=10, y=10, radius=10)
    ]
    to_craters = [
        Crater(id=2, x=10, y=20, radius=10)
    ]
    calculator = NearestNeighborCalculator()

    # Act
    result = calculator.get_nearest_neighbor_distances(from_craters, to_craters)

    # Assert
    assert result == [10]


def test_nearest_neighbor_gets_shortest_distance_from_to_craters():
    # Arrange
    from_craters = [
        Crater(id=1, x=10, y=10, radius=10)
    ]
    to_craters = [
        Crater(id=2, x=10, y=20, radius=10),
        Crater(id=3, x=10, y=30, radius=10),
        Crater(id=4, x=11, y=10, radius=10),
    ]
    calculator = NearestNeighborCalculator()

    # Act
    result = calculator.get_nearest_neighbor_distances(from_craters, to_craters)

    # Assert
    assert result == [1]


def test_nearest_neighbor_gets_shortest_distance_for_all_from_craters():
    # Arrange
    crater1 = Crater(id=1, x=10, y=10, radius=10)
    crater2 = Crater(id=2, x=10, y=20, radius=10)
    from_craters = [crater1, crater2]
    to_craters = [
        crater1,
        crater2,
        Crater(id=3, x=9, y=10, radius=10),
    ]
    calculator = NearestNeighborCalculator()

    # Act
    result = calculator.get_nearest_neighbor_distances(from_craters, to_craters)

    # Assert
    assert result == [1, 10]
