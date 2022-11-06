from statistics import mean
from typing import Optional, Iterable

import numpy as np

from data_structures.kdtree import KDNode
from saturation.datatypes import Crater


class NearestNeighbor:
    REBALANCE_CADENCE = 500

    def __init__(self):
        self._tree: Optional[KDNode] = None
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

        # Rebalance on occasion
        self._rebalance_counter += 1
        if self._rebalance_counter % self.REBALANCE_CADENCE == 0 and not self._tree.is_balanced:
            self._tree = self._tree.rebalance()

    def remove(self, crater: Crater):
        point = (crater.x, crater.y)
        self._tree = self._tree.remove(point)

    def get_mean_nearest_neighbor_distance(self, craters: Iterable[Crater]) -> float:
        if self._tree is None:
            return 0.0

        return self._tree.get_mean_nn_distance(((crater.x, crater.y, crater.id, crater.radius) for crater in craters))
