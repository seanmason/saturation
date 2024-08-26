from typing import List, Iterable, Dict, Callable
from collections import defaultdict

import numpy as np
import numba as nb
from saturation.geometry import get_intersection_arc, add_arc
from saturation.datatypes import Crater, Arc
from saturation.distances import Distances


class CraterDictionary(object):
    """
    Convenience wrapper around a dictionary for Craters.
    """
    def __init__(self):
        self._craters: Dict[int, Crater] = dict()

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


class CraterRecord(object):
    """
    Maintains the record of craters.
    """
    def __init__(self,
                 r_stat: float,
                 rim_erasure_effectiveness_function: Callable[[float, float], bool],
                 mrp: float,
                 rmult: float,
                 study_region_size: int,
                 study_region_padding: int,
                 cell_size: int):
        self._r_stat = r_stat
        self._rim_erasure_effectiveness_function = rim_erasure_effectiveness_function
        self._mrp = mrp
        self._rmult = rmult
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

        self._ntot = 0

        self._erased_arcs: Dict[int, List[Arc]] = defaultdict(
            lambda: nb.typed.List.empty_list(nb.types.UniTuple(nb.float64, 2))
        )
        self._remaining_rim_percentages: Dict[int, float] = dict()

        self._craters_to_remove: List[Crater] = []

        self._sum_tracked_radii: float = 0.0
        self._sum_tracked_squared_radii: float = 0.0

    @property
    def ntot(self) -> int:
        return self._ntot

    @property
    def nobs(self) -> int:
        return len(self._craters_in_study_region)

    def get_mnnd(self) -> float:
        return self._distances.get_mnnd()

    def get_nnd_stdev(self) -> float:
        return self._distances.get_nnd_stdev()

    def get_nnd_min(self) -> float:
        return self._distances.get_min_nnd()

    def get_nnd_max(self) -> float:
        return self._distances.get_max_nnd()

    def _update_rim_arcs(self, new_crater: Crater):
        new_x = new_crater.x
        new_y = new_crater.y
        effective_radius = new_crater.radius * self._rmult

        if new_crater.radius >= self._r_stat:
            self._remaining_rim_percentages[new_crater.id] = 1.

        craters_in_range = self._distances.get_craters_with_overlapping_rims(new_x,
                                                                             new_y,
                                                                             effective_radius)
        for old_crater_id in craters_in_range:
            if old_crater_id == new_crater.id:
                continue

            old_crater = self._all_craters_in_record[old_crater_id]

            # For a new crater to affect an old crater, (new crater radius) > func(old crater radius)
            can_affect_old = self._rim_erasure_effectiveness_function(new_crater.radius, old_crater.radius)
            if can_affect_old:
                arc = get_intersection_arc((old_crater.x, old_crater.y),
                                           old_crater.radius,
                                           (new_x, new_y),
                                           effective_radius)

                erased_arcs = self._erased_arcs[old_crater.id]
                add_arc(arc, erased_arcs)

                remaining_rim_percentage = 1.0 - sum([x[1] - x[0] for x in erased_arcs]) / (2 * np.pi)
                self._remaining_rim_percentages[old_crater.id] = remaining_rim_percentage

                if remaining_rim_percentage < self._mrp:
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
        self._craters_to_remove = []
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
                self._ntot += 1
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
        n = self.nobs
        if not n:
            return 0.0

        return self._sum_tracked_radii / n

    def get_radius_stdev(self) -> float:
        n = self.nobs

        sqr = self._sum_tracked_squared_radii
        r = self._sum_tracked_radii
        numerator = n * sqr - r ** 2
        if n < 2 or numerator < 0.0:
            return 0.0

        return np.sqrt(numerator / (n * (n - 1)))
