from collections import OrderedDict

from typing import List, Tuple

import numpy as np

from saturation.numba_utils import *
from saturation.datatypes import Crater, CraterType
from saturation.distances import Distances
from saturation.initial_rim_state_calculators import InitialRimStateCalculator, CircumferenceInitialRimStateCalculator
from saturation.rim_erasure_calculators import (
    RimErasureCalculator, ExponentRadiusConditionalRimOverlapRimErasureCalculator,
)


@jitclass({
    "_craters": nb.types.DictType(nb.int64, CraterType),
})
class CraterDictionary(object):
    """
    Convenience wrapper around a dictionary for Craters.
    """
    def __init__(self):
        self._craters = nb.typed.Dict.empty(key_type=nb.int64, value_type=CraterType)

    def add(self, crater: Crater) -> None:
        self._craters[crater.id] = crater

    def remove(self, crater_id: int) -> None:
        del self._craters[crater_id]

    def get_craters(self) -> nb.typed.List[Crater]:
        return nb.typed.List(self._craters.values())

    def __getitem__(self, crater_id: int) -> Crater:
        return self._craters[crater_id]

    def __contains__(self, crater_id: int) -> bool:
        return crater_id in self._craters

    def __len__(self) -> int:
        return len(self._craters)


if "DISABLE_NUMBA" not in os.environ:
    spec = OrderedDict({
        "_rstat": nb.float64,
        "_rim_erasure_calculator": ExponentRadiusConditionalRimOverlapRimErasureCalculator.class_type.instance_type,
        "_initial_rim_state_calculator": CircumferenceInitialRimStateCalculator.class_type.instance_type,
        "_mrp": nb.float64,
        "_rmult": nb.float64,
        "_study_region_size": nb.int64,
        "_study_region_padding": nb.int64,
        "_min_x": nb.int64,
        "_min_y": nb.int64,
        "_max_x": nb.int64,
        "_max_y": nb.int64,
        "_calculate_nearest_neighbor_stats": nb.boolean,
        "_distances": Distances.class_type.instance_type,
        "_all_craters_in_record": CraterDictionary.class_type.instance_type,
        "_craters_in_study_region": CraterDictionary.class_type.instance_type,
        "_nstat": nb.int64,
        "_initial_rims": nb.types.DictType(nb.int64, nb.float64),
        "_remaining_rims": nb.types.DictType(nb.int64, nb.float64),
        "_crater_ids_to_remove": nb.types.DictType(nb.int64, nb.int64),
        "_sum_tracked_radii": nb.float64,
        "_sum_tracked_squared_radii": nb.float64,
    })
else:
    spec = OrderedDict({})

@jitclass(spec)
class CraterRecord(object):
    """
    Maintains the record of craters.
    """
    def __init__(
        self,
        rstat: float,
        rim_erasure_calculator: RimErasureCalculator,
        initial_rim_state_calculator: InitialRimStateCalculator,
        mrp: float,
        rmult: float,
        study_region_size: int,
        study_region_padding: int,
        cell_size: int,
        calculate_nearest_neighbor_stats: bool
    ):
        self._rstat = rstat
        self._rim_erasure_calculator = rim_erasure_calculator
        self._initial_rim_state_calculator = initial_rim_state_calculator
        self._mrp = mrp
        self._rmult = rmult
        self._study_region_size = study_region_size
        self._study_region_padding = study_region_padding

        self._min_x = study_region_padding
        self._min_y = study_region_padding
        self._max_x = study_region_size + study_region_padding
        self._max_y = study_region_size + study_region_padding
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

        self._nstat = 0

        self._initial_rims = nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.float64)
        self._remaining_rims = nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.float64)
        self._crater_ids_to_remove = nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.int64)

        self._sum_tracked_radii: float = 0.0
        self._sum_tracked_squared_radii: float = 0.0

    @property
    def nstat(self) -> int:
        return self._nstat

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

    def _update_rim_arcs(self, new_craters: nb.typed.List[Crater]):
        for new_crater in new_craters:
            if new_crater.radius >= self._rstat:
                initial = self._initial_rim_state_calculator.calculate(new_crater)
                self._remaining_rims[new_crater.id] = initial
                self._initial_rims[new_crater.id] = initial

        for new_crater in new_craters:
            craters_in_range = self._distances.get_craters_with_overlapping_rims(
                new_crater.x,
                new_crater.y,
                new_crater.radius * self._rmult
            )
            for old_crater_id in craters_in_range:
                if old_crater_id == new_crater.id:
                    continue

                if old_crater_id in self._crater_ids_to_remove:
                    continue

                initial_rim = self._initial_rims[old_crater_id]
                existing_rim = self._remaining_rims[old_crater_id]

                old_crater = self._all_craters_in_record[old_crater_id]
                updated_rim = self._rim_erasure_calculator.calculate_new_rim_state(old_crater, existing_rim, new_crater)

                self._remaining_rims[old_crater_id] = updated_rim
                if updated_rim / initial_rim < self._mrp:
                    self._crater_ids_to_remove[old_crater_id] = new_crater.id

    def _remove_craters_with_destroyed_rims(self) -> Tuple[List[Crater], List[int]]:
        removed_craters = nb.typed.List()
        removed_by_ids = nb.typed.List()

        for crater_id in self._crater_ids_to_remove:
            removed_by_id = self._crater_ids_to_remove[crater_id]
            crater = self._all_craters_in_record[crater_id]
            removed_craters.append(crater)
            removed_by_ids.append(removed_by_id)

            del self._remaining_rims[crater.id]
            del self._initial_rims[crater.id]
            self._all_craters_in_record.remove(crater.id)
            if crater.id in self._craters_in_study_region:
                self._craters_in_study_region.remove(crater.id)
                self._sum_tracked_radii -= crater.radius
                self._sum_tracked_squared_radii -= crater.radius ** 2

        self._crater_ids_to_remove = nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.int64)
        return removed_craters, removed_by_ids

    def add_craters_smaller_than_rstat(self, craters: List[Crater]) -> Tuple[List[Crater], List[int]]:
        """
        Adds the supplied crater to the record, possibly destroying other craters.
        All craters must be smaller than rstat.
        :param craters: New craters to be added.
        :return: A list of craters that were erased as a result of the additions.
        """
        self._update_rim_arcs(craters)

        removed_craters, removed_by_ids = self._remove_craters_with_destroyed_rims()
        if len(removed_craters) > 0:
            self._distances.remove(removed_craters)

        return removed_craters, removed_by_ids

    def add_crater_geq_rstat(self, crater: Crater) -> Tuple[List[Crater], List[int]]:
        """
        Adds the supplied crater to the record, possibly destroying other craters.
        The crater must be larger than rstat.
        :param crater: New crater to be added.
        :return: A list of craters that were erased as a result of the additions.
        """
        is_in_study_region = self._is_in_study_region(crater)
        self._distances.add(crater, is_in_study_region)

        self._all_craters_in_record.add(crater)
        if is_in_study_region:
            self._craters_in_study_region.add(crater)
            self._nstat += 1
            self._sum_tracked_radii += crater.radius
            self._sum_tracked_squared_radii += crater.radius ** 2

        self._update_rim_arcs(nb.typed.List([crater]))

        removed_craters, removed_by_ids = self._remove_craters_with_destroyed_rims()
        if len(removed_craters) > 0:
            self._distances.remove(removed_craters)

        return removed_craters, removed_by_ids

    def _is_in_study_region(self, crater: Crater) -> bool:
        return (
            (self._min_x <= crater.x <= self._max_x)
            and (self._min_y <= crater.y <= self._max_y)
        )

    def get_crater(self, crater_id: int) -> Crater:
        return self._all_craters_in_record[crater_id]

    @property
    def all_craters_in_record(self) -> nb.typed.List[Crater]:
        return self._all_craters_in_record.get_craters()

    @property
    def craters_in_study_region(self) -> nb.typed.List[Crater]:
        return self._craters_in_study_region.get_craters()

    def get_remaining_rim(self, crater_id: int) -> float:
        return self._remaining_rims[crater_id]

    def get_mean_radius(self) -> float:
        n = self.nobs
        if n == 0:
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
