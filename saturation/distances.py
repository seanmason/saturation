from collections import defaultdict
from typing import Iterable, Dict, Set

from sortedcontainers import SortedList

from data_structures.spatial_hash import SpatialHash
from saturation.datatypes import Crater


class Distances:
    MAX_SEARCH_DISTANCE = 15000

    def __init__(self, max_search_distance: float):
        self._spatial_hash: SpatialHash = SpatialHash(50, max_search_distance)

        self._all_craters: Dict[int, Crater] = dict()

        # Mapping from crater ids to nearest distance
        self._nearest_neighbor_distances: Dict[int, float] = dict()
        self._nearest_neighbor_distances_list: SortedList[float] = SortedList()

        self._tracked_nearest_neighbors: Set[Crater] = set()
        self._total_tracked_nn_distances: float = 0.0
        self._total_tracked_nn_count: int = 0

        # Reverse lookup for nearest neighbors
        self._nearest_neighbors_reverse_lookup: Dict[int, Set[int]] = defaultdict(lambda: set())

        self._rebalance_counter = 0

    def add(self, crater: Crater, tracked: bool):
        self._all_craters[crater.id] = crater
        self._spatial_hash.add(crater)

        if tracked:
            self._tracked_nearest_neighbors.add(crater)

        nearest_neighbor, nearest_neighbor_dist = self._spatial_hash.get_nearest_neighbor(crater)

        if nearest_neighbor:
            self._nearest_neighbor_distances[crater.id] = nearest_neighbor_dist
            self._nearest_neighbor_distances_list.add(nearest_neighbor_dist)
            self._nearest_neighbors_reverse_lookup[nearest_neighbor.id].add(crater.id)
            if tracked:
                self._total_tracked_nn_distances += nearest_neighbor_dist
                self._total_tracked_nn_count += 1

        max_distance = self._nearest_neighbor_distances_list[-1] if self._nearest_neighbor_distances \
            else self.MAX_SEARCH_DISTANCE
        candidates_and_distances = self._spatial_hash.get_craters_with_centers_within_radius(crater.x,
                                                                                             crater.y,
                                                                                             max_distance)

        for existing_crater, new_distance in candidates_and_distances:
            if existing_crater == crater:
                continue

            old_distance = self._nearest_neighbor_distances.get(existing_crater.id, self.MAX_SEARCH_DISTANCE)
            if new_distance < old_distance:
                self._nearest_neighbor_distances[existing_crater.id] = new_distance
                try:
                    self._nearest_neighbor_distances_list.remove(old_distance)
                except ValueError:
                    pass

                self._nearest_neighbor_distances_list.add(new_distance)
                self._nearest_neighbors_reverse_lookup[crater.id].add(existing_crater.id)
                if existing_crater in self._tracked_nearest_neighbors:
                    if old_distance != self.MAX_SEARCH_DISTANCE:
                        self._total_tracked_nn_distances -= old_distance
                    else:
                        self._total_tracked_nn_count += 1
                    self._total_tracked_nn_distances += new_distance

    def remove(self, craters: Iterable[Crater]):
        for crater in craters:
            if crater in self._tracked_nearest_neighbors:
                self._total_tracked_nn_distances -= self._nearest_neighbor_distances[crater.id]
                self._total_tracked_nn_count -= 1
                self._tracked_nearest_neighbors.remove(crater)

            self._spatial_hash.remove(crater)
            if crater.id in self._nearest_neighbor_distances:
                self._nearest_neighbor_distances_list.remove(self._nearest_neighbor_distances[crater.id])
                del self._nearest_neighbor_distances[crater.id]
            del self._all_craters[crater.id]

        # Fix up affected neighbors' nearest neighbors
        for removed_crater in craters:
            for neighbor_id in self._nearest_neighbors_reverse_lookup[removed_crater.id]:
                if neighbor_id in self._all_craters:
                    neighbor = self._all_craters[neighbor_id]
                    old_distance = self._nearest_neighbor_distances[neighbor_id]
                    node, distance = self._spatial_hash.get_nearest_neighbor(neighbor)
                    self._nearest_neighbor_distances_list.remove(self._nearest_neighbor_distances[neighbor_id])
                    self._nearest_neighbor_distances[neighbor_id] = distance
                    self._nearest_neighbor_distances_list.add(distance)

                    if neighbor in self._tracked_nearest_neighbors:
                        self._total_tracked_nn_distances -= old_distance
                        self._total_tracked_nn_distances += distance

            del self._nearest_neighbors_reverse_lookup[removed_crater.id]

    def get_craters_with_overlapping_rims(self,
                                          x: float,
                                          y: float,
                                          radius: float) -> Iterable[Crater]:
        return self._spatial_hash.get_craters_with_intersecting_rims(x, y, radius)

    def get_mean_nearest_neighbor_distance(self) -> float:
        if not self._nearest_neighbor_distances:
            return 0.0

        return self._total_tracked_nn_distances / self._total_tracked_nn_count
