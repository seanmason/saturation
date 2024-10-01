from typing import List, Dict

import numpy as np
import numba as nb
from saturation.datatypes import Crater
from saturation.distances import Distances
from saturation.initial_rim_state_calculators import InitialRimStateCalculator
from saturation.rim_erasure_calculators import RimErasureCalculator


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
                 rim_erasure_calculator: RimErasureCalculator,
                 initial_rim_state_calculator: InitialRimStateCalculator,
                 mrp: float,
                 rmult: float,
                 study_region_size: int,
                 study_region_padding: int,
                 cell_size: int,
                 calculate_nearest_neighbor_stats: bool):
        self._r_stat = r_stat
        self._rim_erasure_calculator = rim_erasure_calculator
        self._initial_rim_state_calculator = initial_rim_state_calculator
        self._mrp = mrp
        self._rmult = rmult
        self._study_region_size = study_region_size
        self._study_region_padding = study_region_padding

        self._min_x = study_region_padding
        self._min_y = study_region_padding
        self._max_x = study_region_size + study_region_padding - 1
        self._max_y = study_region_size + study_region_padding - 1
        self._calculate_nearest_neighbor_stats = calculate_nearest_neighbor_stats

        self._distances = Distances(
            cell_size,
            0,
            study_region_size + 2 * study_region_padding,
            calculate_nearest_neighbor_stats
        )

        # Contains all craters with r > r_stat, may be outside the study region
        self._all_craters_in_record = CraterDictionary()

        # Contains all craters with r > r_stat within the study region
        self._craters_in_study_region = CraterDictionary()

        self._ntot = 0

        self._initial_rims: Dict[int, float] = dict()
        self._remaining_rims: Dict[int, float] = dict()
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
            initial = self._initial_rim_state_calculator.calculate(new_crater)
            self._remaining_rims[new_crater.id] = initial
            self._initial_rims[new_crater.id] = initial

        craters_in_range = self._distances.get_craters_with_overlapping_rims(
            new_x,
            new_y,
            effective_radius
        )
        for old_crater_id in craters_in_range:
            if old_crater_id == new_crater.id:
                continue

            old_crater = self._all_craters_in_record[old_crater_id]

            initial_rim = self._initial_rims[old_crater.id]
            existing_rim = self._remaining_rims[old_crater_id]
            updated_rim = self._rim_erasure_calculator.calculate_new_rim_state(old_crater, existing_rim, new_crater)
            self._remaining_rims[old_crater_id] = updated_rim
            if updated_rim / initial_rim < self._mrp:
                self._craters_to_remove.append(old_crater)

    def _remove_craters_with_destroyed_rims(self) -> List[Crater]:
        for crater in self._craters_to_remove:
            del self._remaining_rims[crater.id]
            del self._initial_rims[crater.id]
            self._all_craters_in_record.remove(crater)
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

    def get_remaining_rim(self, crater_id: int) -> float:
        return self._remaining_rims[crater_id]

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
