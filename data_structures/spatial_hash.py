import math
from collections import defaultdict
from typing import Set, Tuple, Dict, List, Optional

from saturation.datatypes import Crater


class SpatialHash:
    """
    Structure for fast collision checking.
    """
    def __init__(self, cell_size: int, max_search_distance: float):
        self._cell_size: int = cell_size
        self._max_search_distance = max_search_distance

        # Tracking of crater rims
        self._rim_contents: Dict[Tuple[int, int], Set[Crater]] = defaultdict(lambda: set())
        self._buckets_for_crater_rims: Dict[Crater, List[Set[Crater]]] = defaultdict(lambda: [])

        # Tracking of crater centers
        self._center_contents: Dict[Tuple[int, int], Set[Crater]] = defaultdict(lambda: set())
        self._buckets_for_crater_centers: Dict[Crater, Set[Crater]] = dict()

    def _hash(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / self._cell_size), int(y / self._cell_size)

    def _get_hash_min_and_max(self,
                              x: float,
                              y: float,
                              radius: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return self._hash(x - radius, y - radius), self._hash(x + radius, y + radius)

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

                    x_diff = crater.x - x
                    y_diff = crater.y - y
                    distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)

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

                    x_diff = crater.x - x
                    y_diff = crater.y - y
                    distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)

                    if distance < crater.radius + radius:
                        results.add((crater, distance))

        return results

    def get_nearest_neighbor(self, crater: Crater) -> Tuple[Optional[Crater], float]:
        search_multiplier = 1.2
        search_distance = self._cell_size // 4
        nearest_crater = None
        closest_distance = self._max_search_distance

        last_min_point = None
        while True:
            min_point, max_point = self._get_hash_min_and_max(crater.x, crater.y, search_distance)
            if last_min_point != min_point:
                for candidate, distance in self.get_craters_with_centers_within_radius(crater.x,
                                                                                       crater.y,
                                                                                       search_distance):
                    if distance != 0 and distance < closest_distance:
                        nearest_crater = candidate
                        closest_distance = distance

                if nearest_crater:
                    return nearest_crater, closest_distance

            if search_distance == self._max_search_distance:
                break

            last_min_point = min_point
            search_distance *= search_multiplier
            if search_distance > self._max_search_distance:
                search_distance = self._max_search_distance

        return None, self._max_search_distance
