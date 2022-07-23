from typing import List

import numpy as np

from saturation.datatypes import Crater


class NearestNeighborCalculator(object):
    """
    Calculates nearest neighbor distances between craters.
    """
    def get_nearest_neighbor_distances(self,
                                       from_craters: List[Crater],
                                       to_craters: List[Crater]) -> List[float]:
        return [
            self.get_nearest_neighbor_distance(x, to_craters)
            for x in from_craters
        ]

    def get_nearest_neighbor_distance(self, crater: Crater, to_craters: List[Crater]) -> float:
        return min((
            np.sqrt((x.x - crater.x)**2 + (x.y - crater.y)**2)
            for x in to_craters
            if x.id != crater.id
        ))
