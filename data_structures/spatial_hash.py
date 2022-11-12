import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Tuple, Dict, List, Iterable, Optional

from saturation.datatypes import Crater


@dataclass(kw_only=True, frozen=True)
class IntPoint:
    x: int
    y: int


@dataclass(kw_only=True, frozen=True)
class FloatPoint:
    x: float
    y: float


@dataclass(kw_only=True, frozen=True)
class BoundingBox:
    min_point: FloatPoint
    max_point: FloatPoint


class SpatialHash:
    """
    Structure for fast collision checking.
    """
    def __init__(self, cell_size: int, max_search_distance: float):
        self._cell_size: int = cell_size
        self._max_search_distance = max_search_distance

        # Tracking of crater rims
        self._rim_contents: Dict[IntPoint, Set[Crater]] = defaultdict(lambda: set())
        self._buckets_for_crater_rims: Dict[Crater, List[Set[Crater]]] = defaultdict(lambda: [])

        # Tracking of crater centers
        self._center_contents: Dict[IntPoint, Set[Crater]] = defaultdict(lambda: set())
        self._buckets_for_crater_centers: Dict[Crater, Set[Crater]] = dict()

    def _hash(self, point: FloatPoint) -> IntPoint:
        return IntPoint(
            x=int(point.x / self._cell_size),
            y=int(point.y / self._cell_size)
        )

    @staticmethod
    def _get_bounding_box(x: float, y: float, radius: float) -> BoundingBox:
        return BoundingBox(
            min_point=FloatPoint(x=x - radius, y=y - radius),
            max_point=FloatPoint(x=x + radius, y=y + radius)
        )

    def _add_to_crater_rims(self, crater: Crater):
        box = self._get_bounding_box(crater.x, crater.y, crater.radius)
        min_point, max_point = self._hash(box.min_point), self._hash(box.max_point)

        buckets = []

        # iterate over the rectangular region
        for i in range(min_point.x, max_point.x + 1):
            for j in range(min_point.y, max_point.y + 1):
                # append to each intersecting cell
                bucket = self._rim_contents[IntPoint(x=i, y=j)]
                buckets.append(bucket)
                bucket.add(crater)

        self._buckets_for_crater_rims[crater] = buckets

    def _add_to_crater_centers(self, crater: Crater):
        point = self._hash(FloatPoint(x=crater.x, y=crater.y))
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
        box = self._get_bounding_box(x, y, radius)
        min_point, max_point = self._hash(box.min_point), self._hash(box.max_point)

        results: Set[Crater] = set()

        # iterate over the rectangular region
        for i in range(min_point.x, max_point.x + 1):
            for j in range(min_point.y, max_point.y + 1):
                candidates = self._rim_contents.get(IntPoint(x=i, y=j), set())

                for crater in candidates:
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
                                               radius: float) -> Iterable[Tuple[Crater, float]]:
        box = self._get_bounding_box(x, y, radius)
        min_point, max_point = self._hash(box.min_point), self._hash(box.max_point)

        results: Set[Tuple[Crater, float]] = set()

        # iterate over the rectangular region
        for i in range(min_point.x, max_point.x + 1):
            for j in range(min_point.y, max_point.y + 1):
                candidates = self._center_contents.get(IntPoint(x=i, y=j), set())

                for crater in candidates:
                    x_diff = crater.x - x
                    y_diff = crater.y - y
                    distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)

                    if distance < crater.radius + radius:
                        results.add((crater, distance))

        return results

    def get_nearest_neighbor(self, crater: Crater) -> Tuple[Optional[Crater], float]:
        search_multiplier = 2
        search_distance = self._cell_size // 2
        nearest_crater = None
        closest_distance = self._max_search_distance

        while True:
            for candidate, distance in self.get_craters_with_centers_within_radius(crater.x, crater.y, search_distance):
                if distance != 0 and distance < closest_distance:
                    nearest_crater = candidate
                    closest_distance = distance

            if nearest_crater:
                return nearest_crater, closest_distance

            if search_distance == self._max_search_distance:
                break

            search_distance *= search_multiplier
            if search_distance > self._max_search_distance:
                search_distance = self._max_search_distance

        return None, self._max_search_distance
