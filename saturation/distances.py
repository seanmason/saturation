from typing import Dict, Tuple, List, Set
from collections import OrderedDict

import numba as nb
from numba.experimental import jitclass

from data_structures.spatial_hash import SpatialHash
from saturation.datatypes import Crater

null_crater = Crater(-1, 0.0, 0.0, 0.0)
crater_type = nb.typeof(null_crater)
crater_set_type = nb.types.DictType(
    keyty=crater_type,
    valty=nb.boolean
)
int_set_type = nb.types.DictType(
    keyty=nb.int64,
    valty=nb.boolean
)

spec = OrderedDict({
    "_max_search_distance": nb.int64,
    "_spatial_hash": SpatialHash.class_type.instance_type,
    "_all_craters": nb.types.DictType(
        keyty=nb.int64,
        valty=crater_type
    ),
    "_nearest_neighbor_distances": nb.types.DictType(
        keyty=nb.int64,
        valty=nb.float64
    ),
    "_tracked_nearest_neighbors": crater_set_type,
    "_total_tracked_nn_distances": nb.float64,
    "_total_tracked_nn_count": nb.int64,
    "_nearest_neighbors_reverse_lookup": nb.types.DictType(
        keyty=nb.int64,
        valty=int_set_type
    ),
    "_max_nn_distance": nb.float64,
})


@jitclass(spec=spec)
class Distances:
    def __init__(self, cell_size: int, boundary_min: int, boundary_max: int):
        self._max_search_distance = int(1.5 * (boundary_max - boundary_min))
        self._spatial_hash: SpatialHash = SpatialHash(cell_size, boundary_min, boundary_max)

        self._all_craters: Dict[int, Crater] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=crater_type
        )

        # Mapping from crater ids to nearest distance
        self._nearest_neighbor_distances: Dict[int, float] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.float64
        )

        self._tracked_nearest_neighbors: Dict[Crater, bool] = nb.typed.Dict.empty(
            key_type=crater_type,
            value_type=nb.boolean
        )
        self._total_tracked_nn_distances: float = 0.0
        self._total_tracked_nn_count: int = 0

        # Reverse lookup for nearest neighbors
        self._nearest_neighbors_reverse_lookup: Dict[int, Dict[int, bool]] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=int_set_type
        )

        self._max_nn_distance: float = 0.0

    def _add_crater_to_reverse_lookup(self, from_crater: Crater, to_crater: Crater):
        values = self._nearest_neighbors_reverse_lookup.setdefault(from_crater.id,
                                                                   nb.typed.Dict.empty(
                                                                       key_type=nb.int64,
                                                                       value_type=nb.boolean
                                                                   ))
        values[to_crater.id] = True

    def add(self, crater: Crater, tracked: bool):
        self._all_craters[crater.id] = crater
        self._spatial_hash.add(crater)

        if tracked:
            self._tracked_nearest_neighbors[crater] = True

        nearest_neighbor, nearest_neighbor_dist = self._spatial_hash.get_nearest_neighbor(crater)

        if nearest_neighbor.id >= 0:
            self._nearest_neighbor_distances[crater.id] = nearest_neighbor_dist
            self._add_crater_to_reverse_lookup(nearest_neighbor, crater)
            if tracked:
                self._total_tracked_nn_distances += nearest_neighbor_dist
                self._total_tracked_nn_count += 1

        max_distance = self._max_nn_distance if self._max_nn_distance > 0.0 else self._max_search_distance
        candidates_and_distances = self._spatial_hash.get_craters_with_centers_within_radius(crater.x,
                                                                                             crater.y,
                                                                                             max_distance)

        for existing_crater, new_distance in candidates_and_distances.items():
            if existing_crater == crater:
                continue

            old_distance = self._nearest_neighbor_distances.get(existing_crater.id, self._max_search_distance)
            if new_distance < old_distance:
                self._nearest_neighbor_distances[existing_crater.id] = new_distance

                if self._max_nn_distance == old_distance:
                    self._max_nn_distance = max(self._nearest_neighbor_distances.values())
                elif new_distance > self._max_nn_distance:
                    self._max_nn_distance = new_distance

                self._add_crater_to_reverse_lookup(crater, existing_crater)
                if existing_crater in self._tracked_nearest_neighbors:
                    if old_distance != self._max_search_distance:
                        self._total_tracked_nn_distances -= old_distance
                    else:
                        self._total_tracked_nn_count += 1
                    self._total_tracked_nn_distances += new_distance

    def remove(self, craters: List[Crater]):
        recalculate_max_nn_distance = False

        for crater in craters:
            if crater in self._tracked_nearest_neighbors:
                self._total_tracked_nn_distances -= self._nearest_neighbor_distances[crater.id]
                self._total_tracked_nn_count -= 1
                del self._tracked_nearest_neighbors[crater]

            self._spatial_hash.remove(crater)
            if crater.id in self._nearest_neighbor_distances:
                distance = self._nearest_neighbor_distances[crater.id]
                if self._max_nn_distance == distance:
                    recalculate_max_nn_distance = True
                del self._nearest_neighbor_distances[crater.id]

            del self._all_craters[crater.id]

        # Fix up affected neighbors' nearest neighbors
        for removed_crater in craters:
            if removed_crater.id not in self._nearest_neighbors_reverse_lookup:
                continue

            for neighbor_id in self._nearest_neighbors_reverse_lookup[removed_crater.id]:
                if neighbor_id in self._all_craters:
                    neighbor = self._all_craters[neighbor_id]
                    old_distance = self._nearest_neighbor_distances[neighbor_id]
                    node, distance = self._spatial_hash.get_nearest_neighbor(neighbor)

                    if self._max_nn_distance == old_distance:
                        recalculate_max_nn_distance = True

                    self._nearest_neighbor_distances[neighbor_id] = distance

                    if neighbor in self._tracked_nearest_neighbors:
                        self._total_tracked_nn_distances -= old_distance
                        self._total_tracked_nn_distances += distance

            del self._nearest_neighbors_reverse_lookup[removed_crater.id]

        if recalculate_max_nn_distance:
            self._max_nn_distance = max(self._nearest_neighbor_distances.values())

    def get_craters_with_overlapping_rims(self,
                                          x: float,
                                          y: float,
                                          radius: float) -> Set[Crater]:
        return self._spatial_hash.get_craters_with_intersecting_rims(x, y, radius)

    def get_nearest_neighbor(self, crater: Crater) -> Tuple[Crater, float]:
        return self._spatial_hash.get_nearest_neighbor(crater)

    def get_mean_nearest_neighbor_distance(self) -> float:
        if not self._nearest_neighbor_distances:
            return 0.0

        return self._total_tracked_nn_distances / self._total_tracked_nn_count
