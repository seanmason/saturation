from collections import defaultdict
from dataclasses import dataclass
from typing import List, Iterable
from scipy.optimize import fsolve

import numpy as np
import pandas as pd
from sortedcontainers import SortedList

from saturation.geometry import get_intersection_arc, normalize_arcs, SortedArcList, merge_arcs, \
    get_terrain_boundary_intersection_arc
from saturation.datatypes import Crater


@dataclass(frozen=True, kw_only=True)
class CraterDistance:
    crater_id: int
    distance: float


class CraterDictionary(object):
    """
    Convenience wrapper around a dictionary for Craters.
    """

    def __init__(self):
        self._craters = dict()

    def add(self, crater: Crater):
        self._craters[crater.id] = crater

    def remove(self, crater: Crater):
        del self._craters[crater.id]

    def __getitem__(self, crater_id: int) -> Crater:
        return self._craters[crater_id]

    def __contains__(self, crater_id: int) -> bool:
        return crater_id in self._craters

    def __iter__(self):
        return (x for x in self._craters.values())

    def __len__(self):
        return len(self._craters)


class CraterRecord(object):
    """
    Maintains the record of craters.
    """

    def __init__(self,
                 r_stat: float,
                 r_stat_multiplier: float,
                 min_rim_percentage: float,
                 effective_radius_multiplier: float,
                 observed_terrain_size: int,
                 terrain_padding: int):
        self._r_stat = r_stat
        self._r_stat_multiplier = r_stat_multiplier
        self._min_rim_percentage = min_rim_percentage
        self._effective_radius_multiplier = effective_radius_multiplier
        self._observed_terrain_size = observed_terrain_size
        self._terrain_padding = terrain_padding

        self._min_x = terrain_padding
        self._min_y = terrain_padding
        self._max_x = observed_terrain_size + terrain_padding - 1
        self._max_y = observed_terrain_size + terrain_padding - 1

        # Contains all craters with r > r_stat, may be outside the observed terrain
        self._all_craters_in_record = CraterDictionary()

        # Contains all craters with r > r_stat within the observed terrain
        self._craters_in_observation_area = CraterDictionary()

        # Duplicate of _craters_in_observation_area as a DataFrame
        # This is maintained as a performance optimization for calculating distances.
        self._craters_in_observation_area_dataframe = pd.DataFrame(columns=['id', 'x', 'y', 'radius'], dtype='float')
        self._craters_in_observation_area_dataframe['id'] = self._craters_in_observation_area_dataframe.id.astype('int')
        self._craters_in_observation_area_dataframe = self._craters_in_observation_area_dataframe.set_index(['id'])

        # Maintains distances for nearest neighbor calculations.
        # Contains pairwise distances from all craters in the observation area with r > r_stat
        # and all other craters with r > r_stat
        self._distances = defaultdict(lambda: SortedList(key=lambda x: x.distance))

        # Contains all craters, even those below r_stat and those outside the terrain
        self._all_craters = CraterDictionary()
        self._n_craters_added_in_observation_area = 0

        self._erased_arcs = defaultdict(lambda: SortedArcList())
        self._initial_rim_radians = dict()

    @property
    def n_craters_added_in_observation_area(self) -> int:
        return self._n_craters_added_in_observation_area

    @property
    def n_craters_in_observation_area(self) -> int:
        return len(self._craters_in_observation_area)

    def _add_crater_to_distances(self, crater: Crater):
        if crater.radius >= self._r_stat:
            for from_crater_id, distances in self._distances.items():
                if crater.id != from_crater_id:
                    from_crater = self._all_craters[from_crater_id]
                    distances = self._distances[from_crater.id]
                    distance = np.sqrt((from_crater.x - crater.x) ** 2 + (from_crater.y - crater.y) ** 2)
                    distances.add(CraterDistance(crater_id=crater.id, distance=distance))

            if self._min_x <= crater.x <= self._max_x and self._min_y <= crater.y <= self._max_y:
                distances = self._distances[crater.id]
                for to_crater in self._all_craters_in_record:
                    if crater != to_crater:
                        distance = np.sqrt((to_crater.x - crater.x) ** 2 + (to_crater.y - crater.y) ** 2)
                        distances.add(CraterDistance(crater_id=to_crater.id, distance=distance))

    def _remove_craters_from_distances(self, craters: List[Crater]):
        for crater in craters:
            if crater.id in self._distances:
                del self._distances[crater.id]

    def get_nearest_neighbor_distances(self) -> List[float]:
        result = []

        ids_in_record = set([x.id for x in self._all_craters_in_record])
        for distances in self._distances.values():
            while distances and distances[0].crater_id not in ids_in_record:
                distances.pop(0)

            if distances:
                result.append(distances[0].distance)

        return result

    def get_mean_neighbor_distance(self) -> float:
        total = 0.0
        counter = 0

        for distances in self._distances.values():
            if distances:
                total += distances[0].distance
                counter += 1

        return total / counter

    def _get_crater_ids_in_range(self, new_crater: Crater) -> Iterable[int]:
        """"
        Returns a list of crater IDs that may be affected by the addition of new_crater
        """
        effective_radius = new_crater.radius * self._effective_radius_multiplier

        distances = np.sqrt((self._craters_in_observation_area_dataframe.x - new_crater.x) ** 2
                            + (self._craters_in_observation_area_dataframe.y - new_crater.y) ** 2)
        return self._craters_in_observation_area_dataframe.index[
            (distances < self._craters_in_observation_area_dataframe.radius + effective_radius)
            & (distances > self._craters_in_observation_area_dataframe.radius - effective_radius)
            ]

    def _update_rim_arcs(self, new_crater: Crater):
        new_x = new_crater.x
        new_y = new_crater.y
        effective_radius = new_crater.radius * self._effective_radius_multiplier

        # If the new crater runs outside the observed terrain, remove those portions of its rim.
        lower_limit = self._terrain_padding
        upper_limit = self._terrain_padding + self._observed_terrain_size
        if new_crater.radius >= self._r_stat \
                and lower_limit <= new_x <= upper_limit \
                and lower_limit <= new_y <= upper_limit:
            arc = get_terrain_boundary_intersection_arc((new_x, new_y),
                                                        new_crater.radius,
                                                        self._observed_terrain_size,
                                                        self._terrain_padding)

            normalized_arcs = normalize_arcs([(arc[0], arc[1])])
            self._erased_arcs[new_crater.id].update(normalized_arcs)
            merged_arcs = merge_arcs(self._erased_arcs[new_crater.id])
            self._erased_arcs[new_crater.id] = merged_arcs
            self._initial_rim_radians[new_crater.id] = 2 * np.pi - sum([x[1] - x[0] for x in merged_arcs])

        crater_ids_in_range = self._get_crater_ids_in_range(new_crater)

        for old_crater_id in crater_ids_in_range:
            old_crater = self._all_craters_in_record[old_crater_id]

            # For a new crater to affect an old crater, (new crater radius) > (old crater radius) / r_stat_multiplier
            if new_crater.radius > old_crater.radius / self._r_stat_multiplier:
                arc = get_intersection_arc((old_crater.x, old_crater.y),
                                           old_crater.radius,
                                           (new_x, new_y),
                                           effective_radius)

                normalized_arcs = normalize_arcs([(arc[0], arc[1])])
                self._erased_arcs[old_crater.id].update(normalized_arcs)
                merged_arcs = merge_arcs(self._erased_arcs[old_crater.id])
                self._erased_arcs[old_crater.id] = merged_arcs

    def _remove_craters_with_destroyed_rims(self) -> List[Crater]:
        removed_craters = []

        for crater in list(self._craters_in_observation_area):
            removed_arcs = self._erased_arcs[crater.id]
            remaining_rim_percentage = 1 - (sum([x[1] - x[0] for x in removed_arcs]) / self._initial_rim_radians[crater.id])
            if remaining_rim_percentage < self._min_rim_percentage:
                removed_craters.append(crater)
                self._all_craters_in_record.remove(crater)
                self._craters_in_observation_area_dataframe.drop([crater.id], axis=0, inplace=True)
                if crater.id in self._craters_in_observation_area:
                    self._craters_in_observation_area.remove(crater)

        return removed_craters

    def add(self, crater: Crater) -> List[Crater]:
        """
        Adds the supplied crater to the record, possibly destroying other craters.
        :param crater: New crater to be added.
        :return: A list of craters that were erased as a result of the addition.
        """
        self._all_craters.add(crater)

        self._update_rim_arcs(crater)

        if crater.radius >= self._r_stat:
            self._all_craters_in_record.add(crater)

            if self._min_x <= crater.x <= self._max_x and self._min_y <= crater.y <= self._max_y:
                self._craters_in_observation_area.add(crater)
                self._craters_in_observation_area_dataframe = pd.concat(
                    [self._craters_in_observation_area_dataframe, pd.DataFrame([crater]).set_index(['id'])],
                    axis=0
                )
                self._n_craters_added_in_observation_area += 1

            self._add_crater_to_distances(crater)

        removed = self._remove_craters_with_destroyed_rims()
        if removed:
            self._remove_craters_from_distances(removed)

        return removed

    def get_crater(self, crater_id: int) -> Crater:
        """
        Returns the crater with the specified id.
        """
        return self._all_craters[crater_id]

    @property
    def all_craters(self) -> List[Crater]:
        """
        Returns a list of all craters ever seen.
        """
        return list(self._all_craters)

    @property
    def all_craters_in_record(self) -> List[Crater]:
        """
        Returns a list of all craters in the record.
        """
        return list(self._all_craters_in_record)

    @property
    def craters_in_observation_area(self) -> List[Crater]:
        """
        Returns a list of all craters in the record that are in the observation area.
        """
        return list(self._craters_in_observation_area)
