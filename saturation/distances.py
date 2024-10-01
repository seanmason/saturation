from typing import Dict, Tuple, List, Set
from collections import OrderedDict

import numba as nb
import numpy as np
from numba.experimental import jitclass

from saturation.data_structures.spatial_hash import SpatialHash
from saturation.datatypes import Crater

crater_type = nb.typeof(Crater(np.int64(1), np.float32(1.0), np.float32(1.0), np.float32(1.0)))
crater_set_type = nb.types.DictType(
    keyty=crater_type,
    valty=nb.boolean
)
int_set_type = nb.types.DictType(
    keyty=nb.int64,
    valty=nb.boolean
)

spec = OrderedDict({
    "_absolute_max_search_distance": nb.int64,
    "_spatial_hash": SpatialHash.class_type.instance_type,
    "_calculate_nearest_neighbor_stats": nb.boolean,
    "_all_craters": nb.types.DictType(
        keyty=nb.int64,
        valty=crater_type
    ),
    "_nnds": nb.types.DictType(
        keyty=nb.int64,
        valty=nb.float64
    ),
    "_nns": nb.types.DictType(
        keyty=nb.int64,
        valty=nb.int64
    ),
    "_tracked_nns": int_set_type,
    "_sum_tracked_nnds": nb.float64,
    "_sum_tracked_squared_nnds": nb.float64,
    "_tracked_nn_count": nb.int64,
    "_nn_reverse_lookup": nb.types.DictType(
        keyty=nb.int64,
        valty=int_set_type
    ),
    "_min_nnd": nb.float64,
    "_max_nnd": nb.float64,
    "_max_current_search_distance": nb.float64,
    "_recalculate_min_nnd": nb.boolean,
    "_recalculate_max_nnd": nb.boolean,
})


@jitclass(spec=spec)
class Distances:
    def __init__(
        self,
        cell_size: int,
        boundary_min: int,
        boundary_max: int,
        calculate_nearest_neighbor_stats: bool
    ):
        self._absolute_max_search_distance = int(1.5 * (boundary_max - boundary_min))
        self._spatial_hash: SpatialHash = SpatialHash(cell_size, boundary_min, boundary_max)
        self._calculate_nearest_neighbor_stats = calculate_nearest_neighbor_stats

        self._all_craters: Dict[int, Crater] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=crater_type
        )

        # Mapping from crater ids to nearest neighbor
        self._nns: Dict[int, int] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.int64
        )

        # Mapping from crater ids to nearest distance
        self._nnds: Dict[int, float] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.float64
        )

        self._tracked_nns: Dict[int, bool] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.boolean
        )
        self._sum_tracked_nnds: float = 0.0
        self._sum_tracked_squared_nnds: float = 0.0

        self._tracked_nn_count: int = 0

        # Reverse lookup for center-to-center nearest neighbors
        self._nn_reverse_lookup: Dict[int, Dict[int, bool]] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=int_set_type
        )

        self._min_nnd: float = self._absolute_max_search_distance
        self._max_nnd: float = 0.0
        self._max_current_search_distance: float = 0.0

        self._recalculate_min_nnd: bool = False
        self._recalculate_max_nnd: bool = False

    def _add_crater_to_reverse_lookup(self, from_crater_id: int, to_crater_id: int):
        values = self._nn_reverse_lookup.setdefault(from_crater_id,
                                                    nb.typed.Dict.empty(
                                                                       key_type=nb.int64,
                                                                       value_type=nb.boolean
                                                                   ))
        values[to_crater_id] = True

    def add(self, crater: Crater, tracked: bool):
        self._all_craters[crater.id] = crater
        self._spatial_hash.add(crater)

        if tracked:
            self._tracked_nns[crater.id] = True

        if self._calculate_nearest_neighbor_stats:
            self._update_nnds(crater, tracked)

    def _update_nn(self,
                   crater_id: int,
                   new_nn_id: int,
                   new_distance: float,
                   force: bool,
                   tracked: bool):
        old_distance = self._nnds.get(crater_id, 0.0)
        if force or old_distance == 0.0 or new_distance < old_distance:
            if tracked:
                if old_distance == 0.0:
                    self._tracked_nn_count += 1

                self._sum_tracked_nnds += new_distance - old_distance
                self._sum_tracked_squared_nnds += new_distance ** 2 - old_distance ** 2

                if self._max_nnd == old_distance or self._max_current_search_distance == old_distance:
                    self._recalculate_max_nnd = True

                if new_distance > self._max_nnd:
                    self._max_nnd = new_distance
                if new_distance < self._min_nnd:
                    self._min_nnd = new_distance

            # Update reverse lookup if necessary
            nn_id = self._nns.get(crater_id, 0)
            if nn_id != 0 and crater_id in self._nn_reverse_lookup[nn_id]:
                del self._nn_reverse_lookup[nn_id][crater_id]

            if new_distance > self._max_current_search_distance:
                self._max_current_search_distance = new_distance

            self._nnds[crater_id] = new_distance

            if new_nn_id != 0:
                self._nns[crater_id] = new_nn_id
                self._add_crater_to_reverse_lookup(new_nn_id, crater_id)
            else:
                del self._nns[crater_id]

    def _recalculate_max_nnd_if_necessary(self):
        if self._recalculate_max_nnd:
            if not self._tracked_nns:
                self._max_nnd = 0.0
            else:
                self._max_nnd = (
                    max([x[1] for x in self._nnds.items() if x[0] in self._tracked_nns])
                    if len(self._nnds) > 0
                    else 0.0
                )
            self._max_current_search_distance = (
                max(self._nnds.values())
                if len(self._nnds) > 0
                else 0.0
            )
            self._recalculate_max_nnd = False

    def _update_nnds(self, crater: Crater, tracked: bool):
        nn_id, nnd = self._spatial_hash.get_nnd(crater)
        if nn_id != 0:
            self._update_nn(crater.id, nn_id, nnd, force=False, tracked=tracked)

        max_distance = (self._max_current_search_distance
                        if self._max_current_search_distance > 0.0
                        else self._absolute_max_search_distance)
        candidates_and_distances = self._spatial_hash.get_craters_with_centers_within_radius(
            crater.x,
            crater.y,
            max_distance
        )
        for existing_crater_id, new_distance in candidates_and_distances.items():
            if existing_crater_id == crater.id:
                continue

            tracked = existing_crater_id in self._tracked_nns
            self._update_nn(existing_crater_id, crater.id, new_distance, force=False, tracked=tracked)

        self._recalculate_max_nnd_if_necessary()

    def remove(self, craters: List[Crater]):
        for crater in craters:
            self._spatial_hash.remove(crater)

        if self._calculate_nearest_neighbor_stats:
            for crater in craters:
                if crater.id in self._tracked_nns:
                    if crater.id in self._nnds:
                        nn_dist = self._nnds[crater.id]
                        self._sum_tracked_nnds -= nn_dist
                        self._sum_tracked_squared_nnds -= nn_dist ** 2

                    self._tracked_nn_count -= 1

                if crater.id in self._nnds:
                    nn_dist = self._nnds[crater.id]

                    if self._max_current_search_distance == nn_dist or self._max_nnd == nn_dist:
                        self._recalculate_max_nnd = True
                    elif self._min_nnd == nn_dist:
                        self._recalculate_min_nnd = True

            # Fix up affected neighbors' nearest neighbors
            removed_set = set([x.id for x in craters])
            for removed_crater in craters:
                if removed_crater.id in self._nn_reverse_lookup:
                    items = list(self._nn_reverse_lookup[removed_crater.id].keys())
                    for neighbor_id in items:
                        if neighbor_id in removed_set:
                            continue

                        neighbor = self._all_craters[neighbor_id]
                        new_nn_id, nn_dist = self._spatial_hash.get_nnd(neighbor)

                        tracked = neighbor_id in self._tracked_nns
                        self._update_nn(neighbor_id, new_nn_id, nn_dist, force=True, tracked=tracked)

                    del self._nn_reverse_lookup[removed_crater.id]

            for crater in craters:
                nn_id = self._nns.get(crater.id, 0)
                if nn_id != 0 and nn_id in self._nn_reverse_lookup and crater.id in self._nn_reverse_lookup[nn_id]:
                    del self._nn_reverse_lookup[nn_id][crater.id]

                if crater.id in self._nns:
                    del self._nns[crater.id]
                    del self._nnds[crater.id]

                if crater.id in self._tracked_nns:
                    del self._tracked_nns[crater.id]

            if self._recalculate_min_nnd:
                self._min_nnd = (
                    min([x[1] for x in self._nnds.items() if x[0] in self._tracked_nns])
                    if len(self._nnds) > 0
                    else 0.0
                )
                self._recalculate_min_nnd = False

            self._recalculate_max_nnd_if_necessary()

        for crater in craters:
            del self._all_craters[crater.id]

    def get_craters_with_overlapping_rims(self,
                                          x: float,
                                          y: float,
                                          radius: float) -> Set[int]:
        return self._spatial_hash.get_craters_with_intersecting_rims(x, y, radius)

    def get_nn(self, crater: Crater) -> Tuple[int, float]:
        return self._spatial_hash.get_nnd(crater)

    def get_mnnd(self) -> float:
        if not self._nnds:
            return 0.0

        return self._sum_tracked_nnds / self._tracked_nn_count

    def get_min_nnd(self) -> float:
        return self._min_nnd

    def get_max_nnd(self) -> float:
        return self._max_nnd

    def get_nnd_stdev(self) -> float:
        n = self._tracked_nn_count

        sq_dists = self._sum_tracked_squared_nnds
        dists = self._sum_tracked_nnds
        numerator = n * sq_dists - dists ** 2
        if n < 2 or numerator < 0.0:
            return 0.0

        return np.sqrt(numerator / (n * (n - 1)))
