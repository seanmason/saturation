import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Iterable, Dict

import numpy as np
from sortedcontainers import SortedList

from saturation.geometry import get_intersection_arc, normalize_arcs, SortedArcList, merge_arcs, \
    get_study_region_boundary_intersection_arc
from saturation.datatypes import Crater, Arc


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
    MAX_N_DISTANCES = 500

    def __init__(self,
                 r_stat: float,
                 r_stat_multiplier: float,
                 min_rim_percentage: float,
                 effective_radius_multiplier: float,
                 study_region_size: int,
                 study_region_padding: int,
                 max_r: float):
        self._r_stat = r_stat
        self._r_stat_multiplier = r_stat_multiplier
        self._min_rim_percentage = min_rim_percentage
        self._effective_radius_multiplier = effective_radius_multiplier
        self._study_region_size = study_region_size
        self._study_region_padding = study_region_padding
        self._max_distance = max_r + max_r * effective_radius_multiplier

        self._min_x = study_region_padding
        self._min_y = study_region_padding
        self._max_x = study_region_size + study_region_padding - 1
        self._max_y = study_region_size + study_region_padding - 1

        # Contains all craters with r > r_stat, may be outside the study region
        self._all_craters_in_record = CraterDictionary()

        # Contains all craters with r > r_stat within the study region
        self._craters_in_study_region = CraterDictionary()

        # Maintains distances for nearest neighbor calculations.
        # Contains pairwise distances from all craters in the study region with r > r_stat
        # and all other craters with r > r_stat
        self._distances: Dict[int, SortedList[CraterDistance]] = defaultdict(
            lambda: SortedList(key=lambda x: x.distance))

        # Contains all craters, even those below r_stat and those outside the study region
        self._all_craters = CraterDictionary()
        self._n_craters_added_in_study_region = 0

        self._erased_arcs = defaultdict(lambda: SortedArcList())
        self._initial_rim_radians = defaultdict(lambda: math.pi * 2)

    @property
    def n_craters_added_in_study_region(self) -> int:
        return self._n_craters_added_in_study_region

    @property
    def n_craters_in_study_region(self) -> int:
        return len(self._craters_in_study_region)

    @staticmethod
    def _get_distance(crater1: Crater, crater2: Crater) -> float:
        return math.sqrt((crater1.x - crater2.x) ** 2 + (crater1.y - crater2.y) ** 2)

    def _add_crater_to_distances(self, crater: Crater):
        if crater.radius >= self._r_stat:
            # Crater is large enough to be added to the record.
            # Add it as a "destination" in the crater distances record.
            for from_crater_id, distances in self._distances.items():
                if crater.id != from_crater_id:
                    from_crater = self._all_craters[from_crater_id]
                    distances = self._distances[from_crater.id]
                    distance = self._get_distance(from_crater, crater)
                    if distance < self._max_distance:
                        distances.add(CraterDistance(crater_id=crater.id, distance=distance))

            # Add the new crater as a "source" in the crater distances record.
            distances = self._distances[crater.id]
            for to_crater in self._all_craters_in_record:
                if crater != to_crater:
                    distance = self._get_distance(to_crater, crater)
                    if distance < self._max_distance:
                        distances.add(CraterDistance(crater_id=to_crater.id, distance=distance))

    def _remove_craters_from_distances(self, craters: List[Crater]):
        for crater in craters:
            if crater.id in self._distances:
                del self._distances[crater.id]

        crater_ids = {x.id for x in craters}
        for distance_list in self._distances.values():
            for distance in [x for x in distance_list if x.crater_id in crater_ids]:
                distance_list.remove(distance)

    def _cleanup_distances(self):
        """
        Removes from distances lists if the distances list is very long
        """
        for distance_list in self._distances.values():
            if not distance_list:
                continue

            while len(distance_list) > self.MAX_N_DISTANCES:
                distance_list.pop(-1)

    def get_nearest_neighbor_distances(self) -> List[float]:
        result = []

        ids_in_study_region = set([x.id for x in self._craters_in_study_region])
        for crater_id, distances in self._distances.items():
            if crater_id in ids_in_study_region:
                if distances:
                    result.append(distances[0].distance)

        return result

    def _get_craters_with_overlapping_rims(self, new_crater: Crater) -> Iterable[int]:
        """"
        Returns a list of crater IDs that may be affected by the addition of new_crater
        """
        effective_radius = new_crater.radius * self._effective_radius_multiplier
        distances = self._distances[new_crater.id]
        for distance in distances:
            existing_crater = self._all_craters[distance.crater_id]
            if distance.distance < existing_crater.radius + effective_radius:
                # Craters that may overlap
                if distance.distance > existing_crater.radius - effective_radius:
                    # The new crater's rim is not entirely within the existing crater's bowl
                    yield distance.crater_id

    def _update_rim_arcs(self, new_crater: Crater):
        new_x = new_crater.x
        new_y = new_crater.y
        effective_radius = new_crater.radius * self._effective_radius_multiplier

        # If the new crater runs outside the study region, remove those portions of its rim.
        lower_limit = self._study_region_padding
        upper_limit = self._study_region_padding + self._study_region_size
        if new_crater.radius >= self._r_stat \
                and lower_limit <= new_x <= upper_limit \
                and lower_limit <= new_y <= upper_limit:
            arc = get_study_region_boundary_intersection_arc((new_x, new_y),
                                                             new_crater.radius,
                                                             self._study_region_size,
                                                             self._study_region_padding)
            if arc:
                normalized_arcs = normalize_arcs([(arc[0], arc[1])])
                self._erased_arcs[new_crater.id].update(normalized_arcs)
                merged_arcs = merge_arcs(self._erased_arcs[new_crater.id])
                self._erased_arcs[new_crater.id] = merged_arcs
                self._initial_rim_radians[new_crater.id] = 2 * np.pi - sum([x[1] - x[0] for x in merged_arcs])

        crater_ids_in_range = self._get_craters_with_overlapping_rims(new_crater)

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

        for crater in list(self._all_craters_in_record):
            remaining_rim_percentage = self.get_remaining_rim_percent(crater.id)
            if remaining_rim_percentage < self._min_rim_percentage:
                removed_craters.append(crater)
                self._all_craters_in_record.remove(crater)
                if crater.id in self._craters_in_study_region:
                    self._craters_in_study_region.remove(crater)

        return removed_craters

    def add(self, crater: Crater) -> List[Crater]:
        """
        Adds the supplied crater to the record, possibly destroying other craters.
        :param crater: New crater to be added.
        :return: A list of craters that were erased as a result of the addition.
        """
        self._all_craters.add(crater)

        if crater.radius >= self._r_stat:
            self._add_crater_to_distances(crater)

        self._update_rim_arcs(crater)

        if crater.radius >= self._r_stat:
            self._all_craters_in_record.add(crater)

            if self._min_x <= crater.x <= self._max_x and self._min_y <= crater.y <= self._max_y:
                self._craters_in_study_region.add(crater)
                self._n_craters_added_in_study_region += 1

        removed = self._remove_craters_with_destroyed_rims()
        if removed:
            self._remove_craters_from_distances(removed)
            self._cleanup_distances()

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
    def craters_in_study_region(self) -> List[Crater]:
        """
        Returns a list of all craters in the record that are in the study region.
        """
        return list(self._craters_in_study_region)

    def get_erased_rim_segments(self, crater_id: int) -> List[Arc]:
        """
        Returns the erased rim segments for the specified crater.
        """
        return list(self._erased_arcs[crater_id])

    def get_remaining_rim_percent(self, crater_id: int) -> float:
        erased = self.get_erased_rim_segments(crater_id)
        initial_rim_radians = self._initial_rim_radians[crater_id]
        return 1 - ((sum([x[1] - x[0] for x in erased]) - 2 * np.pi + initial_rim_radians) / initial_rim_radians)
