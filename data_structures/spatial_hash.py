import math
from collections import defaultdict
from typing import Set, Tuple, Dict, List, Optional, Iterable

from saturation.datatypes import Crater


class SpatialHash:
    """
    Structure for fast collision checking.
    """
    def __init__(self,
                 cell_size: int,
                 boundary_min: float,
                 boundary_max: float):
        self._cell_size: int = cell_size
        self._max_search_distance = 1.5 * (boundary_max - boundary_min)
        self._max_search_radius_cells = int(self._max_search_distance / self._cell_size + 1)

        self._boundary_min_cell = math.floor(boundary_min / self._cell_size)
        self._boundary_max_cell = math.ceil(boundary_max / self._cell_size)

        # Tracking of crater rims
        self._rim_contents: Dict[Tuple[int, int], Set[Crater]] = defaultdict(lambda: set())
        self._buckets_for_crater_rims: Dict[Crater, List[Set[Crater]]] = defaultdict(lambda: [])

        # Tracking of crater centers
        self._center_contents: Dict[Tuple[int, int], Set[Crater]] = defaultdict(lambda: set())
        self._buckets_for_crater_centers: Dict[Crater, Set[Crater]] = dict()

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
        # return self._hash(x - radius, y - radius), self._hash(x + radius, y + radius)
        return self._get_point_within_boundary(self._hash(x - radius, y - radius)), \
            self._get_point_within_boundary(self._hash(x + radius, y + radius))

    def _add_to_crater_rims(self, crater: Crater):
        min_point, max_point = self._get_hash_min_and_max(crater.x, crater.y, crater.radius)

        buckets = []

        # iterate over the rectangular region
        for i in range(min_point[0], max_point[0] + 1):
            for j in range(min_point[1], max_point[1] + 1):
                # append to each intersecting cell
                bucket = self._rim_contents[(i, j)]
                buckets.append(bucket)
                bucket.add(crater)

        self._buckets_for_crater_rims[crater] = buckets

    def _add_to_crater_centers(self, crater: Crater):
        point = self._hash(crater.x, crater.y)
        bucket = self._center_contents[point]
        bucket.add(crater)
        self._buckets_for_crater_centers[crater] = bucket

    @staticmethod
    def _get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        x_diff = x1 - x2
        y_diff = y1 - y2
        return math.sqrt(x_diff * x_diff + y_diff * y_diff)

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
        for bucket in self._buckets_for_crater_rims[crater]:
            bucket.remove(crater)
        del self._buckets_for_crater_rims[crater]

        self._buckets_for_crater_centers[crater].remove(crater)
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
                candidates = self._rim_contents.get((i, j), None)
                if not candidates:
                    continue

                for crater in candidates:
                    if crater in results:
                        continue

                    distance = self._get_distance(crater.x, crater.y, x, y)
                    if distance < crater.radius + radius:
                        # Craters that may overlap
                        if distance > crater.radius - radius:
                            # The new crater's rim is not entirely within the existing crater's bowl
                            results.add(crater)

        return results

    def get_craters_with_centers_within_radius(self,
                                               x: float,
                                               y: float,
                                               radius: float) -> Set[Tuple[Crater, float]]:
        min_point, max_point = self._get_hash_min_and_max(x, y, radius)

        results: Set[Tuple[Crater, float]] = set()

        # iterate over the rectangular region
        for i in range(min_point[0], max_point[0] + 1):
            for j in range(min_point[1], max_point[1] + 1):
                candidates = self._center_contents.get((i, j), None)
                if not candidates:
                    continue

                for crater in candidates:
                    if crater in results:
                        continue

                    distance = self._get_distance(crater.x, crater.y, x, y)
                    if distance < crater.radius + radius:
                        results.add((crater, distance))

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

    def get_nearest_neighbor(self, crater: Crater) -> Tuple[Optional[Crater], float]:
        """
        Finds the nearest neighbor using an expanding radial search.
        """
        nearest_neighbor_found_radius = self._max_search_radius_cells + 1
        nearest_neighbor = None
        closest_distance = self._max_search_distance
        for radius in range(0, self._max_search_radius_cells + 1):
            for x, y in self._get_perimeter_cells(crater.x, crater.y, radius):
                for candidate in self._center_contents.get((x, y), set()):
                    distance = self._get_distance(candidate.x, candidate.y, crater.x, crater.y)
                    if distance != 0 and distance < closest_distance:
                        nearest_neighbor_found_radius = radius
                        nearest_neighbor = candidate
                        closest_distance = distance

            # Once we find a neighbor, we need to keep scanning out another factor of sqrt(2)
            # In the worst case, the first neighbor found could be at a 45 degree angle, while the true closest may
            # be located at a multiple of 90 degrees.
            if nearest_neighbor and nearest_neighbor_found_radius * 1.5 < radius:
                break

        return nearest_neighbor, closest_distance

    # def get_nearest_neighbor(self, crater: Crater) -> Tuple[Optional[Crater], float]:
    #     search_multiplier = 1.5
    #     search_distance = self._cell_size / 4
    #     nearest_crater = None
    #     closest_distance = self._max_search_distance
    #
    #     last_min_point = None
    #     while True:
    #         min_point, max_point = self._get_hash_min_and_max(crater.x, crater.y, search_distance)
    #         if last_min_point != min_point:
    #             for candidate, distance in self.get_craters_with_centers_within_radius(crater.x,
    #                                                                                    crater.y,
    #                                                                                    search_distance):
    #                 if distance != 0 and distance < closest_distance:
    #                     nearest_crater = candidate
    #                     closest_distance = distance
    #
    #             if nearest_crater:
    #                 return nearest_crater, closest_distance
    #
    #         if search_distance == self._max_search_distance:
    #             break
    #
    #         last_min_point = min_point
    #         search_distance *= search_multiplier
    #         if search_distance > self._max_search_distance:
    #             search_distance = self._max_search_distance
    #
    #     return None, self._max_search_distance
