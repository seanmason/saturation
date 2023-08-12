from typing import Dict, Tuple, List, Set
from collections import OrderedDict

import numba as nb
import numpy as np
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
    "_c2c_nn_distances": nb.types.DictType(
        keyty=nb.int64,
        valty=nb.float64
    ),
    "_c2c_nns": nb.types.DictType(
        keyty=nb.int64,
        valty=nb.int64
    ),
    "_r2r_nn_distances": nb.types.DictType(
        keyty=nb.int64,
        valty=nb.float64
    ),
    "_r2r_nns": nb.types.DictType(
        keyty=nb.int64,
        valty=nb.int64
    ),
    "_tracked_nns": int_set_type,
    "_sum_tracked_c2c_nn_distances": nb.float64,
    "_sum_tracked_squared_c2c_nn_distances": nb.float64,
    "_sum_tracked_r2r_nn_distances": nb.float64,
    "_sum_tracked_squared_r2r_nn_distances": nb.float64,
    "_tracked_nn_count": nb.int64,
    "_tracked_r2r_non_zero_count": nb.int64,
    "_c2c_nn_reverse_lookup": nb.types.DictType(
        keyty=nb.int64,
        valty=int_set_type
    ),
    "_r2r_nn_reverse_lookup": nb.types.DictType(
        keyty=nb.int64,
        valty=int_set_type
    ),
    "_min_c2c_nn_distance": nb.float64,
    "_max_c2c_nn_distance": nb.float64,
    "_max_c2c_search_distance": nb.float64,
    "_max_r2r_nn_distance": nb.float64,
    "_recalculate_min_c2c_nn_distance": nb.boolean,
    "_recalculate_max_c2c_nn_distance": nb.boolean,
    "_recalculate_max_r2r_nn_distance": nb.boolean,
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

        # Mapping from crater ids to nearest center-to-center neighbor
        self._c2c_nns: Dict[int, int] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.int64
        )

        # Mapping from crater ids to nearest center-to-center distance
        self._c2c_nn_distances: Dict[int, float] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.float64
        )

        # Mapping from crater ids to nearest edge-to-edge neighbor
        self._r2r_nns: Dict[int, int] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.int64
        )

        # Mapping from crater ids to nearest edge-to-edge distance
        self._r2r_nn_distances: Dict[int, float] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.float64
        )

        self._tracked_nns: Dict[int, bool] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.boolean
        )
        self._sum_tracked_c2c_nn_distances: float = 0.0
        self._sum_tracked_squared_c2c_nn_distances: float = 0.0

        self._sum_tracked_r2r_nn_distances: float = 0.0
        self._sum_tracked_squared_r2r_nn_distances: float = 0.0

        self._tracked_nn_count: int = 0
        self._tracked_r2r_non_zero_count: int = 0

        # Reverse lookup for center-to-center nearest neighbors
        self._c2c_nn_reverse_lookup: Dict[int, Dict[int, bool]] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=int_set_type
        )

        # Reverse lookup for rim-to-rim nearest neighbors
        self._r2r_nn_reverse_lookup: Dict[int, Dict[int, bool]] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=int_set_type
        )

        self._min_c2c_nn_distance: float = self._max_search_distance
        self._max_c2c_nn_distance: float = 0.0
        self._max_c2c_search_distance: float = 0.0

        self._max_r2r_nn_distance: float = 0.0

        self._recalculate_min_c2c_nn_distance: bool = False
        self._recalculate_max_c2c_nn_distance: bool = False
        self._recalculate_max_r2r_nn_distance: bool = False

    def _add_crater_to_c2c_reverse_lookup(self, from_crater: Crater, to_crater: Crater):
        values = self._c2c_nn_reverse_lookup.setdefault(from_crater.id,
                                                        nb.typed.Dict.empty(
                                                                       key_type=nb.int64,
                                                                       value_type=nb.boolean
                                                                   ))
        values[to_crater.id] = True

    def _add_crater_to_r2r_reverse_lookup(self, from_crater: Crater, to_crater: Crater):
        values = self._r2r_nn_reverse_lookup.setdefault(from_crater.id,
                                                        nb.typed.Dict.empty(
                                                                       key_type=nb.int64,
                                                                       value_type=nb.boolean
                                                                   ))
        values[to_crater.id] = True

    def add(self, crater: Crater, tracked: bool):
        self._all_craters[crater.id] = crater
        self._spatial_hash.add(crater)

        if tracked:
            self._tracked_nns[crater.id] = True

        self._update_c2c_distances(crater, tracked)
        self._update_r2r_distances(crater, tracked)

    def _update_c2c_nn(self,
                       crater: Crater,
                       new_nearest_neighbor: Crater,
                       new_distance: float,
                       force: bool,
                       tracked: bool):
        old_distance = self._c2c_nn_distances.get(crater.id, 0.0)
        if force or old_distance == 0.0 or new_distance < old_distance:
            if tracked:
                if old_distance == 0.0:
                    self._tracked_nn_count += 1

                self._sum_tracked_c2c_nn_distances += new_distance - old_distance
                self._sum_tracked_squared_c2c_nn_distances += new_distance ** 2 - old_distance ** 2

                if self._max_c2c_nn_distance == old_distance or self._max_c2c_search_distance == old_distance:
                    self._recalculate_max_c2c_nn_distance = True

                if new_distance > self._max_c2c_nn_distance:
                    self._max_c2c_nn_distance = new_distance
                if new_distance < self._min_c2c_nn_distance:
                    self._min_c2c_nn_distance = new_distance

            # Update reverse lookup if necessary
            nn_id = self._c2c_nns.get(crater.id, 0)
            if nn_id != 0 and crater.id in self._c2c_nn_reverse_lookup[nn_id]:
                del self._c2c_nn_reverse_lookup[nn_id][crater.id]

            if new_distance > self._max_c2c_search_distance:
                self._max_c2c_search_distance = new_distance

            self._c2c_nn_distances[crater.id] = new_distance
            self._c2c_nns[crater.id] = new_nearest_neighbor.id
            self._add_crater_to_c2c_reverse_lookup(new_nearest_neighbor, crater)

    def _update_r2r_nn(self,
                       crater: Crater,
                       new_nearest_neighbor: Crater,
                       new_distance: float,
                       force: bool,
                       tracked: bool):
        old_distance = self._r2r_nn_distances.get(crater.id, -1.0)
        if force or old_distance == -1.0 or new_distance < old_distance:
            if tracked:
                if old_distance == -1.0:
                    if new_distance > 0.0:
                        self._tracked_r2r_non_zero_count += 1

                    # Ugly, but we needed -1.0 as a sentinel value
                    old_distance = 0.0
                elif old_distance > 0.0 and new_distance == 0.0:
                    self._tracked_r2r_non_zero_count -= 1
                elif old_distance == 0.0 and new_distance > 0.0:
                    self._tracked_r2r_non_zero_count += 1

                self._sum_tracked_r2r_nn_distances += new_distance - old_distance
                self._sum_tracked_squared_r2r_nn_distances += new_distance ** 2 - old_distance ** 2

                if self._max_r2r_nn_distance == old_distance:
                    self._recalculate_max_r2r_nn_distance = True

                if new_distance > self._max_r2r_nn_distance:
                    self._max_r2r_nn_distance = new_distance

            # Update reverse lookup if necessary
            nn_id = self._r2r_nns.get(crater.id, 0)
            if nn_id != 0 and crater.id in self._r2r_nn_reverse_lookup[nn_id]:
                del self._r2r_nn_reverse_lookup[nn_id][crater.id]

            self._r2r_nn_distances[crater.id] = new_distance
            self._r2r_nns[crater.id] = new_nearest_neighbor.id
            self._add_crater_to_r2r_reverse_lookup(new_nearest_neighbor, crater)

    def _recalculate_max_c2c_distance_if_necessary(self):
        if self._recalculate_max_c2c_nn_distance:
            self._max_c2c_nn_distance = max([x[1] for x in self._c2c_nn_distances.items() if x[0] in self._tracked_nns])
            self._max_c2c_search_distance = max(self._c2c_nn_distances.values())
            self._recalculate_max_c2c_nn_distance = False

    def _update_c2c_distances(self, crater: Crater, tracked: bool):
        nearest_neighbor, nearest_neighbor_dist = self._spatial_hash.get_nearest_neighbor_center_to_center(crater)
        if nearest_neighbor.id >= 0:
            self._update_c2c_nn(crater, nearest_neighbor, nearest_neighbor_dist, force=False, tracked=tracked)

        max_distance = (self._max_c2c_search_distance
                        if self._max_c2c_search_distance > 0.0
                        else self._max_search_distance)
        candidates_and_distances = self._spatial_hash.get_craters_with_centers_within_radius(crater.x,
                                                                                             crater.y,
                                                                                             max_distance)
        for existing_crater, new_distance in candidates_and_distances.items():
            if existing_crater == crater:
                continue

            tracked = existing_crater.id in self._tracked_nns
            self._update_c2c_nn(existing_crater, crater, new_distance, force=False, tracked=tracked)

        self._recalculate_max_c2c_distance_if_necessary()

    def _recalculate_max_r2r_distance_if_necessary(self):
        if self._recalculate_max_r2r_nn_distance:
            self._max_r2r_nn_distance = max([x[1] for x in self._r2r_nn_distances.items() if x[0] in self._tracked_nns])
            self._recalculate_max_r2r_nn_distance = False

    def _update_r2r_distances(self, crater: Crater, tracked: bool):
        nearest_neighbor, nearest_neighbor_dist = self._spatial_hash.get_nearest_neighbor_rim_to_rim(crater)
        if nearest_neighbor.id >= 0:
            self._update_r2r_nn(crater, nearest_neighbor, nearest_neighbor_dist, force=False, tracked=tracked)

        max_distance = (self._max_c2c_search_distance
                        if self._max_c2c_search_distance > 0.0
                        else self._max_search_distance)
        candidates_and_distances = self._spatial_hash.get_craters_with_rims_within_radius(crater, max_distance)

        for existing_crater, new_distance in candidates_and_distances.items():
            if existing_crater == crater:
                continue

            tracked = existing_crater.id in self._tracked_nns
            self._update_r2r_nn(existing_crater, crater, new_distance, force=False, tracked=tracked)

        self._recalculate_max_r2r_distance_if_necessary()

    def remove(self, craters: List[Crater]):
        for crater in craters:
            self._spatial_hash.remove(crater)

            if crater.id in self._tracked_nns:
                c2c_dist = self._c2c_nn_distances[crater.id]
                self._sum_tracked_c2c_nn_distances -= c2c_dist
                self._sum_tracked_squared_c2c_nn_distances -= c2c_dist ** 2

                r2r_dist = self._r2r_nn_distances[crater.id]
                self._sum_tracked_r2r_nn_distances -= r2r_dist
                self._sum_tracked_squared_r2r_nn_distances -= r2r_dist ** 2

                self._tracked_nn_count -= 1
                del self._tracked_nns[crater.id]

                if r2r_dist > 0.0:
                    self._tracked_r2r_non_zero_count -= 1

            if crater.id in self._c2c_nn_distances:
                c2c_distance = self._c2c_nn_distances[crater.id]

                if self._max_c2c_search_distance == c2c_distance or self._max_c2c_nn_distance == c2c_distance:
                    self._recalculate_max_c2c_nn_distance = True
                elif self._min_c2c_nn_distance == c2c_distance:
                    self._recalculate_min_c2c_nn_distance = True

                r2r_distance = self._r2r_nn_distances[crater.id]
                if self._max_r2r_nn_distance == r2r_distance:
                    self._recalculate_max_r2r_nn_distance = True

        # Fix up affected neighbors' nearest neighbors
        removed_set = set([x.id for x in craters])
        for removed_crater in craters:
            if removed_crater.id in self._c2c_nn_reverse_lookup:
                items = list(self._c2c_nn_reverse_lookup[removed_crater.id].keys())
                for neighbor_id in items:
                    if neighbor_id in removed_set:
                        continue

                    neighbor = self._all_craters[neighbor_id]
                    new_nn, c2c_distance = self._spatial_hash.get_nearest_neighbor_center_to_center(neighbor)

                    tracked = neighbor_id in self._tracked_nns
                    self._update_c2c_nn(neighbor, new_nn, c2c_distance, force=True, tracked=tracked)

                del self._c2c_nn_reverse_lookup[removed_crater.id]

            if removed_crater.id in self._r2r_nn_reverse_lookup:
                items = list(self._r2r_nn_reverse_lookup[removed_crater.id].keys())
                for neighbor_id in items:
                    if neighbor_id in removed_set:
                        continue

                    neighbor = self._all_craters[neighbor_id]
                    new_nn, r2r_distance = self._spatial_hash.get_nearest_neighbor_rim_to_rim(neighbor)

                    tracked = neighbor_id in self._tracked_nns
                    self._update_r2r_nn(neighbor, new_nn, r2r_distance, force=True, tracked=tracked)

                del self._r2r_nn_reverse_lookup[removed_crater.id]

        for crater in craters:
            del self._all_craters[crater.id]
            del self._c2c_nn_distances[crater.id]
            del self._c2c_nns[crater.id]
            del self._r2r_nn_distances[crater.id]
            del self._r2r_nns[crater.id]

        if self._recalculate_min_c2c_nn_distance:
            self._min_c2c_nn_distance = min([x[1] for x in self._c2c_nn_distances.items() if x[0] in self._tracked_nns])
            self._recalculate_min_c2c_nn_distance = False

        self._recalculate_max_c2c_distance_if_necessary()
        self._recalculate_max_r2r_distance_if_necessary()

    def get_craters_with_overlapping_rims(self,
                                          x: float,
                                          y: float,
                                          radius: float) -> Set[Crater]:
        return self._spatial_hash.get_craters_with_intersecting_rims(x, y, radius)

    def get_center_to_center_nearest_neighbor(self, crater: Crater) -> Tuple[Crater, float]:
        return self._spatial_hash.get_nearest_neighbor_center_to_center(crater)

    def get_rim_to_rim_nearest_neighbor(self, crater: Crater) -> Tuple[Crater, float]:
        return self._spatial_hash.get_nearest_neighbor_rim_to_rim(crater)

    def get_center_to_center_nearest_neighbor_distance_mean(self) -> float:
        if not self._c2c_nn_distances:
            return 0.0

        return self._sum_tracked_c2c_nn_distances / self._tracked_nn_count

    def get_center_to_center_nearest_neighbor_distance_min(self) -> float:
        return self._min_c2c_nn_distance

    def get_center_to_center_nearest_neighbor_distance_max(self) -> float:
        return self._max_c2c_nn_distance

    def get_center_to_center_nearest_neighbor_distance_stdev(self) -> float:
        if not self._c2c_nn_distances:
            return 0.0

        n = self._tracked_nn_count
        sq_dists = self._sum_tracked_squared_c2c_nn_distances
        dists = self._sum_tracked_c2c_nn_distances
        return np.sqrt((n * sq_dists - dists ** 2) / (n * (n - 1)))

    def get_rim_to_rim_nearest_neighbor_distance_mean(self) -> float:
        if not self._r2r_nn_distances:
            return 0.0

        return self._sum_tracked_r2r_nn_distances / self._tracked_nn_count

    def get_rim_to_rim_nearest_neighbor_distance_max(self) -> float:
        return self._max_r2r_nn_distance

    def get_rim_to_rim_nearest_neighbor_distance_stdev(self) -> float:
        if not self._r2r_nn_distances:
            return 0.0

        n = self._tracked_nn_count
        sq_dists = self._sum_tracked_squared_r2r_nn_distances
        dists = self._sum_tracked_r2r_nn_distances
        return np.sqrt((n * sq_dists - dists ** 2) / (n * (n - 1)))

    def get_rim_to_rim_non_zero_nearest_neighbor_distance_count(self) -> int:
        return self._tracked_r2r_non_zero_count
