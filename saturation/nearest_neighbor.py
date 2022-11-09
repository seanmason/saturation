import math
from collections import defaultdict
from typing import Optional, Iterable, Dict, Set

import numpy as np

from data_structures.kdtree import KDNode
from saturation.datatypes import Crater


class NearestNeighbor:
    REBALANCE_CADENCE = 250

    def __init__(self):
        self._tree: Optional[KDNode] = None

        self._all_craters: Dict[int, Crater] = dict()

        # Mapping from crater ids to nearest distance
        self._nearest_neighbor_distances: Dict[int, float] = dict()

        # Reverse lookup for nearest neighbors
        self._nearest_neighbors_reverse_lookup: Dict[int, Set[int]] = defaultdict(lambda: set())

        self._rebalance_counter = 0

    def add(self, crater: Crater):
        point = (crater.x, crater.y)

        if self._tree is None:
            self._tree = KDNode(
                data=point,
                sel_axis=lambda prev_axis: (prev_axis + 1) % 2,
                axis=0,
                dimensions=2
            )
        else:
            self._tree.add(point)

        min_distance = np.inf
        closest_crater = None
        for existing_crater in self._all_craters.values():
            x = existing_crater.x - crater.x
            y = existing_crater.y - crater.y
            new_distance = math.sqrt(x * x + y * y)

            old_distance = self._nearest_neighbor_distances.get(existing_crater.id, np.inf)
            if new_distance < old_distance:
                self._nearest_neighbor_distances[existing_crater.id] = new_distance
                self._nearest_neighbors_reverse_lookup[crater.id].add(existing_crater.id)

            if new_distance < min_distance:
                min_distance = new_distance
                closest_crater = existing_crater

        if closest_crater:
            self._nearest_neighbor_distances[crater.id] = min_distance
            self._nearest_neighbors_reverse_lookup[closest_crater.id].add(crater.id)
        self._all_craters[crater.id] = crater

        # Rebalance on occasion
        self._rebalance_counter += 1
        if self._rebalance_counter % self.REBALANCE_CADENCE == 0 and not self._tree.is_balanced:
            self._tree = self._tree.rebalance()

    def remove(self, craters: Iterable[Crater]):
        # Fix up the tree
        for crater in craters:
            point = (crater.x, crater.y)
            self._tree = self._tree.remove(point)

            del self._nearest_neighbor_distances[crater.id]
            del self._all_craters[crater.id]

        # Fix up affected neighbors' nearest neighbors
        for removed_crater in craters:
            for neighbor_id in self._nearest_neighbors_reverse_lookup[removed_crater.id]:
                if neighbor_id in self._all_craters:
                    neighbor = self._all_craters[neighbor_id]
                    self._nearest_neighbor_distances[neighbor_id] = self._tree.get_nn_dist((neighbor.x, neighbor.y))

            del self._nearest_neighbors_reverse_lookup[removed_crater.id]

    def get_mean_nearest_neighbor_distance(self, craters: Iterable[Crater]) -> float:
        if self._tree is None:
            return 0.0

        return np.mean([self._nearest_neighbor_distances[x.id] for x in craters])
