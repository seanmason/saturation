from collections import defaultdict
from typing import List, Set

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
                 effective_radius_multiplier: float,
                 terrain_size: int,
                 margin: int):
        self._all_craters = all_craters
        self._min_crater_radius_for_stats = min_crater_radius_for_stats
        self._min_rim_percentage = min_rim_percentage
        self._effective_radius_multiplier = effective_radius_multiplier

        self._min_x = margin
        self._min_y = margin
        self._max_x = terrain_size - margin - 1
        self._max_y = terrain_size - margin - 1

        # Find all crater ids that are both in bounds for stats and
        # greater than the minimum radius
        all_craters_for_stats = self._all_craters[
            (self._all_craters.radius >= self._min_crater_radius_for_stats)
            & (self._all_craters.x >= self._min_x) & (self._all_craters.x <= self._max_x)
            & (self._all_craters.y >= self._min_y) & (self._all_craters.y <= self._max_y)
        ]
        self._all_crater_ids_for_stats = set(all_craters_for_stats.reset_index().id)

        # All craters in the record
        self._crater_ids = set()

        # Craters in the record that are in bounds for stats
        self._crater_ids_for_stats = set()

        self._erased_crater_ids = []
        self._erased_arcs = defaultdict(lambda: SortedArcList())

        self._calculate_distances()

    def _calculate_distances(self):
        """
        Calculates the distances between all craters above the minimum radius.
        """
        filtered = self._all_craters[
            (self._all_craters.radius >= self._min_crater_radius_for_stats)
        ].reset_index()
        filtered_in_bounds = filtered[
            (filtered.x >= self._min_x) & (filtered.x <= self._max_x)
            & (filtered.y >= self._min_y) & (filtered.y <= self._max_y)
        ].reset_index()
        merged = pd.merge(filtered_in_bounds, filtered, how='cross', suffixes=('_old', '_new'))
        merged = merged[merged.id_new != merged.id_old]
        merged['distance'] = np.sqrt((merged.x_old - merged.x_new)**2 + (merged.y_old - merged.y_new)**2).astype('float16')
        self._distances = merged[['id_old', 'id_new', 'distance']].rename(columns={
            'id_old': 'first_id',
            'id_new': 'second_id'
        })

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
        filtered = self._all_craters.loc[list(self._crater_ids)]
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
                if old_id in self._crater_ids_for_stats:
                    self._crater_ids_for_stats.remove(old_id)
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
            self._crater_ids.add(new_crater_id)

            if new_crater_id in self._all_crater_ids_for_stats:
                self._crater_ids_for_stats.add(new_crater_id)

        return removed_crater_ids

    def get_nearest_neighbor_distances(self) -> np.array:
        """
        Returns nearest neighbor distances for craters in the record that fall within
        a central bounding box.
        """
        result = self._distances[self._distances.first_id.isin(self._crater_ids_for_stats)]
        result = result[result.second_id.isin(self._crater_ids)][['first_id', 'distance']]\
            .groupby(['first_id']).distance.min().values
        return result

    def get_craters(self) -> pd.DataFrame:
        """
        Returns a dataframe containing all craters in the record.
        """
        return self._all_craters.loc[self._crater_ids_for_stats]

    def get_crater_ids(self) -> Set[int]:
        """
        Returns the crater IDs currently in the record.
        """
        return self._crater_ids_for_stats
