from collections import defaultdict
from typing import Iterable, Dict, Set

import numpy as np

from data_structures.spatial_hash import SpatialHash
from saturation.datatypes import Crater


class Distances:
    MAX_SEARCH_DISTANCE = 15000

    def __init__(self, max_search_distance: float):
        self._spatial_hash: SpatialHash = SpatialHash(100, max_search_distance)

        self._all_craters: Dict[int, Crater] = dict()

        # Mapping from crater ids to nearest distance
        self._nearest_neighbor_distances: Dict[int, float] = dict()

        # Reverse lookup for nearest neighbors
        self._nearest_neighbors_reverse_lookup: Dict[int, Set[int]] = defaultdict(lambda: set())

        self._rebalance_counter = 0

    def add(self, crater: Crater):
        self._all_craters[crater.id] = crater
        self._spatial_hash.add(crater)

        nearest_neighbor, nearest_neighbor_dist = self._spatial_hash.get_nearest_neighbor(crater)

        if nearest_neighbor:
            self._nearest_neighbor_distances[crater.id] = nearest_neighbor_dist
            self._nearest_neighbors_reverse_lookup[nearest_neighbor.id].add(crater.id)

        max_distance = max(self._nearest_neighbor_distances.values()) if self._nearest_neighbor_distances else self.MAX_SEARCH_DISTANCE
        candidates_and_distances = self._spatial_hash.get_craters_with_centers_within_radius(crater.x,
                                                                                             crater.y,
                                                                                             max_distance)

        for existing_crater, new_distance in candidates_and_distances:
            if existing_crater == crater:
                continue

            old_distance = self._nearest_neighbor_distances.get(existing_crater.id, self.MAX_SEARCH_DISTANCE)
            if new_distance < old_distance:
                self._nearest_neighbor_distances[existing_crater.id] = new_distance
                self._nearest_neighbors_reverse_lookup[crater.id].add(existing_crater.id)

    def remove(self, craters: Iterable[Crater]):
        for crater in craters:
            self._spatial_hash.remove(crater)
            if crater.id in self._nearest_neighbor_distances:
                del self._nearest_neighbor_distances[crater.id]
            del self._all_craters[crater.id]

        # Fix up affected neighbors' nearest neighbors
        for removed_crater in craters:
            for neighbor_id in self._nearest_neighbors_reverse_lookup[removed_crater.id]:
                if neighbor_id in self._all_craters:
                    neighbor = self._all_craters[neighbor_id]
                    node, distance = self._spatial_hash.get_nearest_neighbor(neighbor)
                    self._nearest_neighbor_distances[neighbor_id] = distance

            del self._nearest_neighbors_reverse_lookup[removed_crater.id]

    def get_craters_with_overlapping_rims(self,
                                          x: float,
                                          y: float,
                                          radius: float) -> Iterable[Crater]:
        return self._spatial_hash.get_craters_with_intersecting_rims(x, y, radius)

    def get_mean_nearest_neighbor_distance(self, craters: Iterable[Crater]) -> float:
        if not self._nearest_neighbor_distances:
            return 0.0

        return np.mean([self._nearest_neighbor_distances[x.id] for x in craters])
