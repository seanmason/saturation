from dataclasses import dataclass
from typing import NamedTuple, List

import pandas as pd

from saturation.crater_record import CraterRecord
from saturation.datatypes import Crater


class StateRow(NamedTuple):
    last_crater_id: int
    n_craters_added_in_study_region: int
    crater_id: int
    x: float
    y: float
    radius: float
    rim_percent_remaining: float


@dataclass(frozen=True, kw_only=True, slots=True)
class StatisticsRow:
    crater_id: int
    n_craters_added_in_study_region: int
    n_craters_in_study_region: int
    areal_density: float
    z: float
    za: float


class StatisticsWriter:
    def __init__(self, output_path: str, output_cadence: int):
        self._output_path = output_path
        self._output_cadence = output_cadence

        self._statistics_rows: List[StatisticsRow] = []
        self._total_statistics_rows: int = 0

    def _flush_rows(self) -> None:
        if self._statistics_rows:
            out_df = pd.DataFrame(self._statistics_rows)
            out_df.crater_id = out_df.crater_id.astype('uint32')
            out_df.n_craters_added_in_study_region = out_df.n_craters_added_in_study_region.astype('uint32')
            out_df.n_craters_in_study_region = out_df.n_craters_in_study_region.astype('uint32')
            out_df.areal_density = out_df.areal_density.astype('float32')
            out_df.z = out_df.z.astype('float32')
            out_df.za = out_df.za.astype('float32')

            output_filename = f"{self._output_path}/statistics_{self._total_statistics_rows}.parquet"
            out_df.to_parquet(output_filename)

            self._statistics_rows = []

    def write(self, statistics_row: StatisticsRow) -> None:
        if self._output_cadence != 0:
            self._statistics_rows.append(statistics_row)
            self._total_statistics_rows += 1

            if self._total_statistics_rows % self._output_cadence == 0:
                self._flush_rows()

    def close(self):
        self._flush_rows()


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
                rim_percent_remaining=crater_record.get_remaining_rim_percent(report_crater.id)
            ))

        state_filename = f'{self._output_path}/state_{n_craters_current}.parquet'
        state_df = pd.DataFrame(state_rows)
        state_df = state_df.astype({
            "last_crater_id": "uint32",
            "n_craters_added_in_study_region": "uint32",
            "crater_id": "uint32",
            "x": "uint32",
            "y": "uint32",
            "radius": "float32",
            "rim_percent_remaining": "float32",
        })
        state_df.to_parquet(state_filename, index=False)
