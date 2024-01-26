from typing import List, Iterable, Dict
from collections import OrderedDict

import numpy as np
import numba as nb
from numba.experimental import jitclass
from saturation.geometry import get_intersection_arc, add_arc
from saturation.datatypes import Crater, Arc
from saturation.distances import Distances


crater_type = nb.typeof(Crater(np.int64(1), np.float32(1.0), np.float32(1.0), np.float32(1.0)))
int_to_crater_dict_type = nb.types.DictType(
    keyty=nb.types.int64,
    valty=crater_type
)


crater_dictionary_spec = OrderedDict({
    "_craters": int_to_crater_dict_type,
})


@jitclass(spec=crater_dictionary_spec)
class CraterDictionary(object):
    """
    Convenience wrapper around a dictionary for Craters.
    """
    def __init__(self):
        self._craters: Dict[int, Crater] = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=crater_type
        )

    def add(self, crater: Crater):
        self._craters[crater.id] = crater

    def remove(self, crater: Crater):
        del self._craters[crater.id]

    def get_craters(self) -> List[Crater]:
        return nb.typed.List(self._craters.values())

    def __getitem__(self, crater_id: int) -> Crater:
        return self._craters[crater_id]

    def __contains__(self, crater_id: int) -> bool:
        return crater_id in self._craters

    def __len__(self):
        return len(self._craters)


arc_type = nb.types.UniTuple(nb.float64, 2)
erased_arcs_type = nb.types.DictType(
    keyty=nb.int64,
    valty=nb.types.ListType(arc_type)
)
remaining_rim_percentages_type = nb.types.DictType(
    keyty=nb.int64,
    valty=nb.float64
)
arc_list_type = nb.types.ListType(arc_type)
crater_list_type = nb.types.ListType(crater_type)

crater_record_spec = OrderedDict({
    "_r_stat": nb.float64,
    "_r_stat_multiplier": nb.float64,
    "_min_rim_percentage": nb.float64,
    "_effective_radius_multiplier": nb.float64,
    "_study_region_size": nb.int64,
    "_study_region_padding": nb.int64,
    "_min_x": nb.int64,
    "_min_y": nb.int64,
    "_max_x": nb.int64,
    "_max_y": nb.int64,
    "_distances": Distances.class_type.instance_type,
    "_all_craters_in_record": CraterDictionary.class_type.instance_type,
    "_craters_in_study_region": CraterDictionary.class_type.instance_type,
    "_n_craters_added_in_study_region": nb.int64,
    "_erased_arcs": erased_arcs_type,
    "_remaining_rim_percentages": remaining_rim_percentages_type,
    "_craters_to_remove": crater_list_type,
    "_sum_tracked_radii": nb.float64,
    "_sum_tracked_squared_radii": nb.float64,
})


@jitclass(spec=crater_record_spec)
class CraterRecord(object):
    """
    Maintains the record of craters.
    """
    def __init__(self,
                 r_stat: float,
                 r_stat_multiplier: float,
                 min_rim_percentage: float,
                 effective_radius_multiplier: float,
                 study_region_size: int,
                 study_region_padding: int,
                 cell_size: int):
        self._r_stat = r_stat
        self._r_stat_multiplier = r_stat_multiplier
        self._min_rim_percentage = min_rim_percentage
        self._effective_radius_multiplier = effective_radius_multiplier
        self._study_region_size = study_region_size
        self._study_region_padding = study_region_padding

        self._min_x = study_region_padding
        self._min_y = study_region_padding
        self._max_x = study_region_size + study_region_padding - 1
        self._max_y = study_region_size + study_region_padding - 1

        self._distances = Distances(
            cell_size,
            0,
            study_region_size + 2 * study_region_padding
        )

        # Contains all craters with r > r_stat, may be outside the study region
        self._all_craters_in_record = CraterDictionary()

        # Contains all craters with r > r_stat within the study region
        self._craters_in_study_region = CraterDictionary()

        # Contains all craters, even those below r_stat and those outside the study region
        self._n_craters_added_in_study_region = 0

        self._erased_arcs: Dict[int, List[Arc]] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=arc_list_type
        )
        self._remaining_rim_percentages: Dict[int, float] = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.float64
        )

        self._craters_to_remove: List[Crater] = nb.typed.List.empty_list(crater_type)

        self._sum_tracked_radii: float = 0.0
        self._sum_tracked_squared_radii: float = 0.0

    @property
    def n_craters_added_in_study_region(self) -> int:
        return self._n_craters_added_in_study_region

    @property
    def n_craters_in_study_region(self) -> int:
        return len(self._craters_in_study_region)

    def get_center_to_center_nearest_neighbor_distance_mean(self) -> float:
        return self._distances.get_center_to_center_nearest_neighbor_distance_mean()

    def get_center_to_center_nearest_neighbor_distance_stdev(self) -> float:
        return self._distances.get_center_to_center_nearest_neighbor_distance_stdev()

    def get_center_to_center_nearest_neighbor_distance_min(self) -> float:
        return self._distances.get_center_to_center_nearest_neighbor_distance_min()

    def get_center_to_center_nearest_neighbor_distance_max(self) -> float:
        return self._distances.get_center_to_center_nearest_neighbor_distance_max()

    def _update_rim_arcs(self, new_crater: Crater):
        new_x = new_crater.x
        new_y = new_crater.y
        effective_radius = new_crater.radius * self._effective_radius_multiplier

        if new_crater.radius >= self._r_stat:
            self._remaining_rim_percentages[new_crater.id] = 1.

        craters_in_range = self._distances.get_craters_with_overlapping_rims(new_x,
                                                                             new_y,
                                                                             effective_radius)
        for old_crater_id in craters_in_range:
            if old_crater_id == new_crater.id:
                continue

            old_crater = self._all_craters_in_record[old_crater_id]

            # For a new crater to affect an old crater, (new crater radius) > (old crater radius) / r_stat_multiplier
            if new_crater.radius >= old_crater.radius / self._r_stat_multiplier:
                arc = get_intersection_arc((old_crater.x, old_crater.y),
                                           old_crater.radius,
                                           (new_x, new_y),
                                           effective_radius)

                erased_arcs = self._erased_arcs.setdefault(old_crater.id, nb.typed.List.empty_list(arc_type))
                add_arc(arc, erased_arcs)

                remaining_rim_percentage = 1.0 - sum([x[1] - x[0] for x in erased_arcs]) / (2 * np.pi)
                self._remaining_rim_percentages[old_crater.id] = remaining_rim_percentage

                if remaining_rim_percentage < self._min_rim_percentage:
                    self._craters_to_remove.append(old_crater)

    def _remove_craters_with_destroyed_rims(self) -> List[Crater]:
        for crater in self._craters_to_remove:
            del self._erased_arcs[crater.id]
            self._all_craters_in_record.remove(crater)
            del self._remaining_rim_percentages[crater.id]
            if crater.id in self._craters_in_study_region:
                self._craters_in_study_region.remove(crater)
                self._sum_tracked_radii -= crater.radius
                self._sum_tracked_squared_radii -= crater.radius ** 2

        result = self._craters_to_remove
        self._craters_to_remove = nb.typed.List.empty_list(crater_type)
        return result

    def add(self, crater: Crater) -> List[Crater]:
        """
        Adds the supplied crater to the record, possibly destroying other craters.
        :param crater: New crater to be added.
        :return: A list of craters that were erased as a result of the addition.
        """
        in_study_region = self._min_x <= crater.x <= self._max_x and self._min_y <= crater.y <= self._max_y
        if crater.radius >= self._r_stat:
            self._distances.add(crater, in_study_region)

        self._update_rim_arcs(crater)

        if crater.radius >= self._r_stat:
            self._all_craters_in_record.add(crater)

            if in_study_region:
                self._craters_in_study_region.add(crater)
                self._n_craters_added_in_study_region += 1
                self._sum_tracked_radii += crater.radius
                self._sum_tracked_squared_radii += crater.radius ** 2

        removed = self._remove_craters_with_destroyed_rims()
        if removed:
            self._distances.remove(removed)

        return removed

    def get_crater(self, crater_id: int) -> Crater:
        """
        Returns the crater with the specified id.
        """
        return self._all_craters_in_record[crater_id]

    @property
    def all_craters_in_record(self) -> List[Crater]:
        """
        Returns a list of all craters in the record.
        """
        return self._all_craters_in_record.get_craters()

    @property
    def craters_in_study_region(self) -> List[Crater]:
        """
        Returns a list of all craters in the record that are in the study region.
        """
        return self._craters_in_study_region.get_craters()

    def get_erased_rim_segments(self, crater_id: int) -> Iterable[Arc]:
        """
        Returns the erased rim segments for the specified crater.
        """
        return self._erased_arcs[crater_id]

    def get_remaining_rim_percent(self, crater_id: int) -> float:
        return self._remaining_rim_percentages.get(crater_id, 1.0)

    def get_mean_radius(self) -> float:
        n = self.n_craters_in_study_region
        if not n:
            return 0.0

        return self._sum_tracked_squared_radii / n

    def get_radius_stdev(self) -> float:
        n = self.n_craters_in_study_region

        sqr = self._sum_tracked_squared_radii
        r = self._sum_tracked_radii
        numerator = n * sqr - r ** 2
        if n < 2 or numerator < 0.0:
            return 0.0

        return np.sqrt(numerator / (n * (n - 1)))
