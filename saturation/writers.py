from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, List

import pandas as pd

from saturation.crater_record import CraterRecord
from saturation.datatypes import Crater


class StateRow(NamedTuple):
    last_crater_id: int
    ntot: int
    crater_id: int
    x: float
    y: float
    radius: float
    rim_percent_remaining: float


@dataclass(frozen=True, kw_only=True, slots=True)
class StatisticsRow:
    crater_id: int
    ntot: int
    nobs: int
    areal_density: float
    nnd_mean: float
    nnd_stdev: float
    nnd_min: float
    nnd_max: float
    radius_mean: float
    radius_stdev: float
    z: float
    za: float


class StatisticsWriter:
    def __init__(self, simulation_id: int, output_path: Path, output_cadence: int):
        self._simulation_id = simulation_id
        self._output_path = output_path
        self._output_cadence = output_cadence

        self._statistics_rows: List[StatisticsRow] = []
        self._total_statistics_rows: int = 0

    def _flush(self) -> None:
        if self._statistics_rows:
            out_df = pd.DataFrame(self._statistics_rows)
            out_df["simulation_id"] = self._simulation_id

            output_filename = f"{self._output_path}/statistics_{self._total_statistics_rows}.parquet"
            out_df.to_parquet(output_filename)

            self._statistics_rows = []

    def write(self, statistics_row: StatisticsRow) -> None:
        if self._output_cadence != 0:
            self._statistics_rows.append(statistics_row)
            self._total_statistics_rows += 1

            if self._total_statistics_rows % self._output_cadence == 0:
                self._flush()

    def close(self):
        self._flush()


class StateSnapshotWriter:
    def __init__(self, simulation_id: int, output_path: Path):
        self._simulation_id = simulation_id
        self._output_path = output_path

    def write_state_snapshot(self,
                             crater_record: CraterRecord,
                             last_crater: Crater,
                             n_craters_current: int) -> None:
        state_rows = []
        for report_crater in crater_record.all_craters_in_record:
            state_rows.append(StateRow(
                last_crater_id=last_crater.id,
                n_craters_added_in_study_region=n_craters_current,
                crater_id=report_crater.id,
                x=report_crater.x,
                y=report_crater.y,
                radius=report_crater.radius,
                rim_percent_remaining=crater_record.get_remaining_rim_percent(report_crater.id)
            ))

        state_filename = f'{self._output_path}/state_{n_craters_current}.parquet'
        state_df = pd.DataFrame(state_rows)
        state_df["simulation_id"] = self._simulation_id
        state_df.to_parquet(state_filename, index=False)


class CraterWriter:
    """
    Writes records of craters generated.
    """
    def __init__(self, output_cadence: int, simulation_id: int, output_path: Path):
        self._output_cadence = output_cadence
        self._simulation_id = simulation_id
        self._output_path = output_path

        self._total_craters = 0
        self._craters: List[Crater] = []

    def _flush(self) -> None:
        if self._craters:
            out_df = pd.DataFrame([x.to_dict() for x in self._craters])
            out_df["simulation_id"] = self._simulation_id
            output_filename = f"{self._output_path}/craters_{self._total_craters}.parquet"
            out_df.to_parquet(output_filename)

            self._craters = []

    def write(self, crater: Crater) -> None:
        if self._output_cadence != 0:
            self._craters.append(crater)
            self._total_craters += 1

            if self._total_craters % self._output_cadence == 0:
                self._flush()

    def close(self):
        self._flush()


@dataclass(frozen=True, kw_only=True, slots=True)
class CraterRemoval:
    removed_crater_id: int
    removed_by_crater_id: int


class CraterRemovalWriter:
    """
    Writes records of craters removed.
    """
    def __init__(self, output_cadence: int, simulation_id: int, output_path: Path):
        self._output_cadence = output_cadence
        self._simulation_id = simulation_id
        self._output_path = output_path

        self._total_removals = 0
        self._removals: List[CraterRemoval] = []

    def _flush(self) -> None:
        if self._removals:
            out_df = pd.DataFrame(self._removals)
            out_df["simulation_id"] = self._simulation_id
            output_filename = f"{self._output_path}/crater_removals_{self._total_removals}.parquet"
            out_df.to_parquet(output_filename)

            self._removals = []

    def write(self, removed_craters: List[Crater], removed_by_crater: Crater) -> None:
        if self._output_cadence != 0:
            for removed_crater in removed_craters:
                self._removals.append(CraterRemoval(removed_crater_id=removed_crater.id,
                                                    removed_by_crater_id=removed_by_crater.id))
                self._total_removals += 1

                if self._total_removals % self._output_cadence == 0:
                    self._flush()

    def close(self):
        self._flush()
