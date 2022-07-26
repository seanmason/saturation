from collections import defaultdict
from dataclasses import dataclass
from typing import List, Iterable

import numpy as np
import pandas as pd
from sortedcontainers import SortedList

from saturation.geometry import get_intersection_arc, normalize_arcs, SortedArcList, merge_arcs
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
                 min_rim_percentage: float,
                 effective_radius_multiplier: float,
                 observed_terrain_size: int,
                 terrain_padding: int):
        self._r_stat = r_stat
        self._min_rim_percentage = min_rim_percentage
        self._effective_radius_multiplier = effective_radius_multiplier

        self._min_x = terrain_padding
        self._min_y = terrain_padding
        self._max_x = observed_terrain_size + terrain_padding - 1
        self._max_y = observed_terrain_size + terrain_padding - 1

        self._craters_in_observation_area = CraterDictionary()
        self._all_craters_in_record = CraterDictionary()

        self._all_craters_in_record_dataframe = pd.DataFrame(columns=['id', 'x', 'y', 'radius'], dtype='float')
        self._all_craters_in_record_dataframe['id'] = self._all_craters_in_record_dataframe.id.astype('int')
        self._all_craters_in_record_dataframe = self._all_craters_in_record_dataframe.set_index(['id'])

        self._all_craters = CraterDictionary()
        self._n_craters_added_in_observation_area = 0

        self._erased_arcs = defaultdict(lambda: SortedArcList())

        self._distances = defaultdict(lambda: SortedList(key=lambda x: x.distance))

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
                    distance = np.sqrt((from_crater.x - crater.x)**2 + (from_crater.y - crater.y)**2)
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

        ids_in_record = set(self._all_craters_in_record_dataframe.index)
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
        new_x = new_crater.x
        new_y = new_crater.y
        effective_radius = new_crater.radius * self._effective_radius_multiplier

        distances = np.sqrt((self._all_craters_in_record_dataframe.x - new_crater.x) ** 2 + (self._all_craters_in_record_dataframe.y - new_crater.y) ** 2)
        return self._all_craters_in_record_dataframe.index[
            (distances < self._all_craters_in_record_dataframe.radius + effective_radius)
            & (distances > self._all_craters_in_record_dataframe.radius - effective_radius)
        ]

    def _update_rim_arcs_and_erased_craters(self, new_crater: Crater) -> List[Crater]:
        """
        Returns removed craters.
        """
        removed_craters = []

        new_x = new_crater.x
        new_y = new_crater.y
        effective_radius = new_crater.radius * self._effective_radius_multiplier

        crater_ids_in_range = self._get_crater_ids_in_range(new_crater)

        # for old_crater in craters_in_range:
        for old_crater_id in crater_ids_in_range:
            old_crater = self._all_craters_in_record[old_crater_id]
            arc = get_intersection_arc((old_crater.x, old_crater.y),
                                       old_crater.radius,
                                       (new_x, new_y),
                                       effective_radius)

            normalized_arcs = normalize_arcs([(arc[0], arc[1])])
            self._erased_arcs[old_crater.id].update(normalized_arcs)
            merged_arcs = merge_arcs(self._erased_arcs[old_crater.id])
            self._erased_arcs[old_crater.id] = merged_arcs

            # Remove old crater if necessary
            remaining_rim_percentage = 1 - sum([x[1] - x[0] for x in merged_arcs]) / (2 * np.pi)
            if remaining_rim_percentage < self._min_rim_percentage:
                removed_craters.append(old_crater)
                self._all_craters_in_record.remove(old_crater)
                self._all_craters_in_record_dataframe.drop([old_crater.id], axis=0, inplace=True)
                if old_crater.id in self._craters_in_observation_area:
                    self._craters_in_observation_area.remove(old_crater)

        return removed_craters

    def add(self, crater: Crater) -> List[Crater]:
        """
        Adds the supplied crater to the record, possibly destroying other craters.
        :param crater: New crater to be added.
        :return: A list of craters that were erased as a result of the addition.
        """
        self._all_craters.add(crater)

        erased = self._update_rim_arcs_and_erased_craters(crater)

        if crater.radius >= self._r_stat:
            self._all_craters_in_record.add(crater)
            self._all_craters_in_record_dataframe = pd.concat(
                [self._all_craters_in_record_dataframe, pd.DataFrame([crater]).set_index(['id'])],
                axis=0
            )

            if self._min_x <= crater.x <= self._max_x and self._min_y <= crater.y <= self._max_y:
                self._craters_in_observation_area.add(crater)
                self._n_craters_added_in_observation_area += 1

        self._add_crater_to_distances(crater)
        if erased:
            self._remove_craters_from_distances(erased)

        return erased

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
