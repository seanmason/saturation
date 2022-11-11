import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Tuple, Dict, List, Iterable

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

    def __init__(self, cell_size: int):
        self._cell_size: int = cell_size

        # Tracking of crater rims
        self._rim_contents: Dict[IntPoint, Set[Crater]] = defaultdict(lambda: set())
        self._buckets_for_crater_rims: Dict[Crater, List[Set[Crater]]] = defaultdict(lambda: [])

        # Tracking of crater centers
        self._center_contents: Dict[IntPoint, Set[Crater]] = defaultdict(lambda: set())
        self._buckets_for_crater_centers: Dict[Crater, List[Set[Crater]]] = defaultdict(lambda: [])

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
        self._rim_contents[point].add(crater)

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

        for bucket in self._buckets_for_crater_centers[crater]:
            bucket.remove(crater)

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
                candidates = self._rim_contents[IntPoint(x=i, y=j)]

                for crater in candidates:
                    x_diff = crater.x - x
                    y_diff = crater.y - y
                    distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)

                    if distance < crater.radius + radius:
                        results.add(crater)

        return results

    def get_craters_with_centers_within_radius(self,
                                               x: float,
                                               y: float,
                                               radius: float) -> Iterable[Tuple[Crater, float]]:
        point = self._hash(FloatPoint(x=x, y=y))

        results: Set[Tuple[Crater, float]] = set()
        for crater in self._rim_contents[point]:
            x_diff = crater.x - x
            y_diff = crater.y - y
            distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)

            if distance < radius:
                results.add((crater, distance))

        return results
