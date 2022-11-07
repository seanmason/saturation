from dataclasses import dataclass
from typing import NamedTuple, List

import pandas as pd

from saturation.crater_record import CraterRecord
from saturation.datatypes import Crater, Arc


class StateRow(NamedTuple):
    last_crater_id: int
    n_craters_added_in_study_region: int
    crater_id: int
    x: float
    y: float
    radius: float
    erased_rim_segments: List[Arc]
    rim_percent_remaining: float


@dataclass(frozen=True, kw_only=True)
class StatisticsRow:
    crater_id: int
    n_craters_added_in_study_region: int
    n_craters_in_study_region: int
    areal_density: float
    z: float
    za: float


class StatisticsWriter:
    def __init__(self, output_path: str):
        self._outfile = open(f"{output_path}/statistics.csv", "w")
        self._outfile.write("crater_id,n_craters_added_in_study_region,n_craters_in_study_region,areal_density,z,za\n")

    def write_row(self,
                  crater_id: int,
                  n_craters_added_in_study_region: int,
                  n_craters_in_study_region: int,
                  areal_density: float,
                  z: float,
                  za: float) -> None:
        self._outfile.write(f"{crater_id},{n_craters_added_in_study_region},{n_craters_in_study_region},"
                            f"{areal_density},{z},{za}\n")

    def close(self):
        self._outfile.close()


class RemovalsWriter:
    def __init__(self, output_path: str):
        self._outfile = open(f"{output_path}/removals.csv", "w")
        self._outfile.write("crater_id,removed_by_id\n")

    def write_row(self, crater_id: int, removed_by_id: int) -> None:
        self._outfile.write(f"{crater_id},{removed_by_id}\n")

    def close(self):
        self._outfile.close()


class CratersWriter:
    def __init__(self, output_path: str):
        self._outfile = open(f"{output_path}/all_craters.csv", "w")
        self._outfile.write("crater_id,n_craters_added_in_study_region,n_craters_in_study_region,areal_density,z,za\n")

    def write_row(self,
                  crater_id: int,
                  x: float,
                  y: float,
                  radius: float) -> None:
        self._outfile.write(f"{crater_id},{x},{y},{radius}\n")

    def close(self):
        self._outfile.close()


class StateSnapshotWriter:
    def __init__(self, output_path: str):
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
                erased_rim_segments=list(crater_record.get_erased_rim_segments(report_crater.id)),
                rim_percent_remaining=crater_record.get_remaining_rim_percent(report_crater.id)
            ))

        state_filename = f'{self._output_path}/state_{n_craters_current}.parquet'
        pd.DataFrame(state_rows).to_parquet(state_filename, index=False)
