from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

from saturation.geometry import get_intersection_arc, normalize_arcs, SortedArcList, merge_arcs


class CraterRecord(object):
    """
    Maintains the record of craters.
    """
    def __init__(self,
                 all_craters: pd.DataFrame,
                 min_crater_radius_for_stats: float,
                 min_rim_percentage: float,
                 effective_radius_multiplier: float):
        self._all_craters = all_craters
        self._min_crater_radius_for_stats = min_crater_radius_for_stats
        self._min_rim_percentage = min_rim_percentage
        self._effective_radius_multiplier = effective_radius_multiplier
        self._crater_ids = []

        # Contains distances between all craters in the record
        self._distances = defaultdict(lambda: dict())

        self._erased_crater_ids = []
        self._erased_arcs = defaultdict(lambda: SortedArcList())

    def _get_distance(self, first_id: int, second_id: int) -> float:
        first = self._all_craters.loc[first_id]
        second = self._all_craters.loc[second_id]

        return np.sqrt((first.x - second.x)**2 + (first.y - second.y)**2)

    def _update_rim_arcs_and_erased_craters(self, new_crater_id: int) -> List[int]:
        """
        Returns removed crater ids.
        """
        removed_crater_ids = []

        new_crater = self._all_craters.loc[new_crater_id]
        new_x = new_crater.x
        new_y = new_crater.y
        effective_radius = new_crater.radius * self._effective_radius_multiplier

        # Filter to only those craters that may be effected by the new crater.
        filtered = self._all_craters.loc[self._crater_ids]
        distance = np.sqrt((filtered.x - new_x) ** 2 + (filtered.y - new_y) ** 2)
        filtered = filtered[(distance < filtered.radius + effective_radius)
                            & (distance + effective_radius > filtered.radius)].reset_index()

        for x, y, radius, old_id in filtered[['x', 'y', 'radius', 'id']].values:
            arc = get_intersection_arc((x, y),
                                       radius,
                                       (new_x, new_y),
                                       effective_radius)

            normalized_arcs = normalize_arcs([(arc[0], arc[1])])
            self._erased_arcs[old_id].update(normalized_arcs)
            merged_arcs = merge_arcs(self._erased_arcs[old_id])
            self._erased_arcs[old_id] = merged_arcs

            # Remove old crater if necessary
            remaining_rim_percentage = 1 - sum([x[1] - x[0] for x in merged_arcs]) / (2 * np.pi)
            if remaining_rim_percentage < self._min_rim_percentage:
                removed_crater_ids.append(old_id)
                self._crater_ids.remove(old_id)
                self._erased_crater_ids.append(old_id)

        return removed_crater_ids

    def update(self, new_crater_id: int) -> List[int]:
        """
        Updates the crater record for the addition of the supplied crater id.
        :param new_crater_id: New crater to be added.
        :return: A list of crater ids that were removed.
        """
        crater = self._all_craters.loc[new_crater_id]

        removed_crater_ids = self._update_rim_arcs_and_erased_craters(new_crater_id)

        # Add the new crater if it is large enough.
        if crater.radius >= self._min_crater_radius_for_stats:
            self._crater_ids.append(new_crater_id)

            # # Update distances
            # for old_crater_id in self._crater_ids:
            #     distance = self._get_distance(new_crater_id, old_crater_id)
            #     self._distances[old_crater_id][new_crater_id] = distance
            #     self._distances[new_crater_id][old_crater_id] = distance

        return removed_crater_ids

    def get_distance(self, first_crater_id: int, second_crater_id: int) -> float:
        return self._distances[first_crater_id][second_crater_id]

    def get_craters(self) -> pd.DataFrame:
        return self._all_craters.loc[self._crater_ids]

    def get_crater_ids(self) -> List[int]:
        return self._crater_ids
