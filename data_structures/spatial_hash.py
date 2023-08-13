from typing import Set, Tuple, Dict, List, Optional, Iterable
from collections import OrderedDict

import numpy as np
import numba as nb
from numba.experimental import jitclass

from saturation.datatypes import Crater


null_crater = Crater(-1, 0.0, 0.0, 0.0)
tuple_type = nb.types.UniTuple(nb.types.int64, 2)
crater_type = nb.typeof(null_crater)
crater_set_type = nb.types.DictType(
    keyty=crater_type,
    valty=nb.types.boolean
)
list_crater_set_type = nb.types.ListType(crater_set_type)


spatial_hash_spec = OrderedDict({
    "_cell_size": nb.types.int64,
    "_max_search_distance": nb.types.int64,
    "_max_search_radius_cells": nb.types.int64,
    "_boundary_min_cell": nb.types.int64,
    "_boundary_max_cell": nb.types.int64,
    "_rim_contents": nb.types.DictType(
        keyty=tuple_type,
        valty=crater_set_type
    ),
    "_buckets_for_crater_rims": nb.types.DictType(
        keyty=crater_type,
        valty=list_crater_set_type
    ),
    "_center_contents": nb.types.DictType(
        keyty=tuple_type,
        valty=crater_set_type
    ),
    "_buckets_for_crater_centers": nb.types.DictType(
        keyty=crater_type,
        valty=crater_set_type
    ),
})


@nb.njit(fastmath=True)
def _get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    x_diff = x1 - x2
    y_diff = y1 - y2
    return np.sqrt(x_diff * x_diff + y_diff * y_diff)


@jitclass(spec=spatial_hash_spec)
class SpatialHash:
    """
    Structure for fast collision checking.
    """

    def __init__(self,
                 cell_size: int,
                 boundary_min: float,
                 boundary_max: float):
        self._cell_size: int = cell_size
        self._max_search_distance = int(1.5 * (boundary_max - boundary_min))
        self._max_search_radius_cells = int(self._max_search_distance / self._cell_size + 1)

        self._boundary_min_cell = int(np.floor(boundary_min / self._cell_size))
        self._boundary_max_cell = int(np.ceil(boundary_max / self._cell_size))

        # Tracking of crater rims
        self._rim_contents: Dict[Tuple[int, int], Dict[Crater, bool]] = nb.typed.Dict.empty(
            key_type=tuple_type,
            value_type=crater_set_type
        )
        self._buckets_for_crater_rims: Dict[Crater, List[Dict[Crater, bool]]] = nb.typed.Dict.empty(
            key_type=crater_type,
            value_type=list_crater_set_type
        )

        # Tracking of crater centers
        self._center_contents: Dict[Tuple[int, int], Dict[Crater, bool]] = nb.typed.Dict.empty(
            key_type=tuple_type,
            value_type=crater_set_type
        )
        self._buckets_for_crater_centers: Dict[Crater, Dict[Crater, bool]] = nb.typed.Dict.empty(
            key_type=crater_type,
            value_type=crater_set_type
        )

    def _get_rim_contents(self, location: Tuple[int, int]) -> Dict[Crater, bool]:
        return self._rim_contents.setdefault(
            location,
            nb.typed.Dict.empty(
                key_type=crater_type,
                value_type=nb.types.boolean
            )
        )

    def _get_center_contents(self, location: Tuple[int, int]) -> Dict[Crater, bool]:
        return self._center_contents.setdefault(
            location,
            nb.typed.Dict.empty(
                key_type=crater_type,
                value_type=nb.types.boolean
            )
        )

    def _hash(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / self._cell_size), int(y / self._cell_size)

    def _get_within_boundary(self, value: int) -> int:
        return max(min(value, self._boundary_max_cell), self._boundary_min_cell)

    def _get_point_within_boundary(self, point: Tuple[int, int]) -> Tuple[int, int]:
        return self._get_within_boundary(point[0]), self._get_within_boundary(point[1])

    def _get_hash_min_and_max(self,
                              x: float,
                              y: float,
                              radius: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return self._get_point_within_boundary(self._hash(x - radius, y - radius)), \
            self._get_point_within_boundary(self._hash(x + radius, y + radius))

    def _add_to_crater_rims(self, crater: Crater):
        min_point, max_point = self._get_hash_min_and_max(crater.x, crater.y, crater.radius)

        buckets = nb.typed.List.empty_list(crater_set_type)

        # iterate over the rectangular region
        for i in range(min_point[0], max_point[0] + 1):
            for j in range(min_point[1], max_point[1] + 1):
                # append to each intersecting cell
                bucket = self._get_rim_contents((i, j))
                buckets.append(bucket)
                bucket[crater] = True

        self._buckets_for_crater_rims[crater] = buckets

    def _add_to_crater_centers(self, crater: Crater):
        point = self._hash(crater.x, crater.y)
        bucket = self._get_center_contents(point)

        bucket[crater] = True
        self._buckets_for_crater_centers[crater] = bucket

    @staticmethod
    def _get_rim_to_rim_distance(crater: Crater, x: float, y: float, radius: float) -> float:
        r2r_dist = _get_distance(crater.x, crater.y, x, y) - crater.radius - radius
        return r2r_dist if r2r_dist > 0.0 else 0.0

    def add(self, crater: Crater):
        """
        Insert a crater.
        """
        self._add_to_crater_centers(crater)
        self._add_to_crater_rims(crater)

    def remove(self, crater: Crater):
        """
        Remove a crater.
        """
        if crater in self._buckets_for_crater_rims:
            for bucket in self._buckets_for_crater_rims[crater]:
                del bucket[crater]
            del self._buckets_for_crater_rims[crater]

        if crater in self._buckets_for_crater_centers:
            del self._buckets_for_crater_centers[crater][crater]
            del self._buckets_for_crater_centers[crater]

    def get_craters_with_intersecting_rims(self, x: float, y: float, radius: float) -> Set[Crater]:
        """
        Returns overlapping craters.
        """
        min_point, max_point = self._get_hash_min_and_max(x, y, radius)

        results: Set[Crater] = set()

        # iterate over the rectangular region
        for i in range(min_point[0], max_point[0] + 1):
            for j in range(min_point[1], max_point[1] + 1):
                if (i, j) not in self._rim_contents:
                    continue

                candidates = self._rim_contents[(i, j)]

                for crater in candidates:
                    if crater in results:
                        continue

                    distance = _get_distance(crater.x, crater.y, x, y)
                    if distance < crater.radius + radius:
                        # Craters that may overlap
                        if distance > crater.radius - radius:
                            # The new crater's rim is not entirely within the existing crater's bowl
                            results.add(crater)

        return results

    def get_craters_with_centers_within_radius(self,
                                               x: float,
                                               y: float,
                                               radius: float) -> Dict[Crater, float]:
        min_point, max_point = self._get_hash_min_and_max(x, y, radius)

        results: Dict[Crater, float] = {}

        # iterate over the rectangular region
        for i in range(min_point[0], max_point[0] + 1):
            for j in range(min_point[1], max_point[1] + 1):
                if (i, j) not in self._center_contents:
                    continue

                candidates = self._center_contents[(i, j)]
                for crater in candidates.keys():
                    if crater in results:
                        continue

                    distance = _get_distance(crater.x, crater.y, x, y)
                    if distance <= crater.radius + radius:
                        results[crater] = distance

        return results

    def get_craters_with_rims_within_radius(self,
                                            crater: Crater,
                                            radius: float) -> Dict[Crater, float]:
        min_point, max_point = self._get_hash_min_and_max(crater.x, crater.y, radius)

        results: Dict[Crater, float] = {}

        # iterate over the rectangular region
        for i in range(min_point[0], max_point[0] + 1):
            for j in range(min_point[1], max_point[1] + 1):
                if (i, j) not in self._rim_contents:
                    continue

                candidates = self._rim_contents[(i, j)]
                for candidate in candidates.keys():
                    if candidate in results:
                        continue

                    distance = _get_distance(candidate.x, candidate.y, crater.x, crater.y) - candidate.radius
                    if distance <= radius:
                        results[candidate] = self._get_rim_to_rim_distance(candidate, crater.x, crater.y, crater.radius)

        return results

    def _get_perimeter_cells(self, center_x: float, center_y: float, radius_cells: int) -> Iterable[Tuple[int, int]]:
        point = self._hash(center_x, center_y)

        if radius_cells == 0:
            yield point
        else:
            # Top side
            y = point[1] + radius_cells
            if y <= self._boundary_max_cell:
                min_x = max(self._boundary_min_cell, point[0] - radius_cells)
                max_x = min(self._boundary_max_cell, point[0] + radius_cells)
                for x in range(min_x, max_x + 1):
                    yield x, y

            # Left side
            x = point[0] - radius_cells
            if x >= self._boundary_min_cell:
                min_y = max(self._boundary_min_cell, point[1] - radius_cells + 1)
                max_y = min(self._boundary_max_cell, point[1] + radius_cells)
                for y in range(min_y, max_y):
                    yield x, y

            # Right side
            x = point[0] + radius_cells
            if x <= self._boundary_max_cell:
                min_y = max(self._boundary_min_cell, point[1] - radius_cells + 1)
                max_y = min(self._boundary_max_cell, point[1] + radius_cells)
                for y in range(min_y, max_y):
                    yield x, y

            # Bottom
            y = point[1] - radius_cells
            if y >= self._boundary_min_cell:
                min_x = max(self._boundary_min_cell, point[0] - radius_cells)
                max_x = min(self._boundary_max_cell, point[0] + radius_cells)
                for x in range(min_x, max_x + 1):
                    yield x, y

    def get_nearest_neighbor_center_to_center(self, crater: Crater) -> Tuple[Optional[Crater], float]:
        """
        Finds the nearest neighbor (center-to-center) using an expanding radial search.
        """
        nearest_neighbor_found_radius = self._max_search_radius_cells + 1
        nearest_neighbor = null_crater
        closest_distance = self._max_search_distance
        for radius in range(0, self._max_search_radius_cells + 1):
            for x, y in self._get_perimeter_cells(crater.x, crater.y, radius):
                if (x, y) in self._center_contents:
                    for candidate in self._center_contents[(x, y)]:
                        distance = _get_distance(candidate.x, candidate.y, crater.x, crater.y)
                        if distance != 0 and distance < closest_distance:
                            nearest_neighbor_found_radius = radius
                            nearest_neighbor = candidate
                            closest_distance = distance

            # Once we find a neighbor, we need to keep scanning out another factor of sqrt(2)
            # In the worst case, the first neighbor found could be at a 45 degree angle, while the true closest may
            # be located at a multiple of 90 degrees.
            if nearest_neighbor != null_crater and (nearest_neighbor_found_radius + 1) * 1.5 < radius:
                break

        return nearest_neighbor, closest_distance

    def get_nearest_neighbor_rim_to_rim(self, crater: Crater) -> Tuple[Optional[Crater], float]:
        """
        Finds the nearest neighbor (rim to rim) using an expanding radial search.
        """
        nearest_neighbor_found_radius = self._max_search_radius_cells + 1
        nearest_neighbor = null_crater
        closest_distance = self._max_search_distance
        for radius in range(0, self._max_search_radius_cells + 1):
            if closest_distance == 0.0:
                break

            for x, y in self._get_perimeter_cells(crater.x, crater.y, radius):
                if closest_distance == 0.0:
                    break

                if (x, y) in self._rim_contents:
                    for candidate in self._rim_contents[(x, y)]:
                        if candidate.id == crater.id:
                            continue

                        distance = self._get_rim_to_rim_distance(candidate, crater.x, crater.y, crater.radius)
                        if distance < closest_distance:
                            nearest_neighbor_found_radius = radius
                            nearest_neighbor = candidate
                            closest_distance = distance

            # Once we find a neighbor, we need to keep scanning out another factor of sqrt(2)
            # In the worst case, the first neighbor found could be at a 45 degree angle, while the true closest may
            # be located at a multiple of 90 degrees.
            if nearest_neighbor != null_crater and (nearest_neighbor_found_radius + 1) * 1.5 < radius:
                break

        return nearest_neighbor, closest_distance
