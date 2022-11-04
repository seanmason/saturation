from dataclasses import dataclass
from typing import Callable, Generator, Iterable, List, NamedTuple

import numpy as np
import pandas as pd

from saturation.areal_density import ArealDensityCalculator
from saturation.crater_record import CraterRecord
from saturation.distributions import ProbabilityDistribution
from saturation.plotting import save_study_region
from saturation.z_stats import calculate_z_statistic, calculate_za_statistic
from saturation.datatypes import Crater, Arc

# Type definitions
LocationFunc = Callable[[], np.array]


class CraterRow(NamedTuple):
    crater_id: int
    x: float
    y: float
    radius: float


@dataclass(frozen=True, kw_only=True)
class RemovalRow:
    crater_id: int
    removed_by_id: int


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


def get_crater_location() -> np.array:
    """
    Returns an (x, y) crater location, uniformly distributed on [0, 1]
    """
    return np.random.rand(2)


def get_craters(size_distribution: ProbabilityDistribution,
                full_region_size: float,
                location_func: LocationFunc = get_crater_location) -> Generator[Crater, None, None]:
    """
    Infinite generator for craters. full_region_size determines the size of the study region being impacted,
    including padding.
    """
    crater_id = 1
    while True:
        locations = location_func()
        yield Crater(
            id=crater_id,
            x=locations[0] * full_region_size,
            y=locations[1] * full_region_size,
            radius=size_distribution.pullback(np.random.rand(1)[0])
        )
        crater_id += 1


def run_simulation(crater_generator: Iterable[Crater],
                   n_craters: int,
                   r_stat: float,
                   r_stat_multiplier: float,
                   min_rim_percentage: float,
                   effective_radius_multiplier: float,
                   study_region_size: int,
                   study_region_padding: int,
                   output_path: str):
    """
    Runs a simulation.
    Writes several output files to the output directory:
    - A CSV of statistics after every crater added to the study region.
    - A CSV of the state after every crater is added to the study region.
    - A CSV recording when each crater is removed.
    - A CSV recording all craters generated.
    - Sample output images of the study region, as PNGs.

    :param crater_generator: Crater generator.
    :param n_craters: Number of craters above r_stat to generate.
    :param r_stat: Minimum crater radius for statistics
    :param r_stat_multiplier: r_stat multiplier
    :param min_rim_percentage: Minimum rim percentage for a crater to remain in the record.
    :param effective_radius_multiplier: Multiplier on a crater's radius when destroying other rims.
    :param study_region_size: Size of the study region.
    :param study_region_padding: Padding around the edges of the study region.
    :param output_path: Path to the output directory.
    """
    output_image_cadence = 50

    statistics_rows = []
    removals_rows = []
    all_craters_rows = []

    study_region_area = study_region_size ** 2

    # The crater record handles removal of craters rims and the record
    # of what craters remain at a given point in time.
    crater_record = CraterRecord(r_stat,
                                 r_stat_multiplier,
                                 min_rim_percentage,
                                 effective_radius_multiplier,
                                 study_region_size,
                                 study_region_padding)

    areal_density_calculator = ArealDensityCalculator(study_region_size, study_region_padding, r_stat)

    last_n_craters = 0
    for crater in crater_generator:
        n_craters_current = crater_record.n_craters_added_in_study_region

        all_craters_rows.append(CraterRow(
            crater_id=crater.id,
            x=crater.x,
            y=crater.y,
            radius=crater.radius
        ))

        removed_craters = crater_record.add(crater)

        areal_density_calculator.add_crater(crater)
        if removed_craters:
            areal_density_calculator.remove_craters(removed_craters)

            # Save removals
            for removed in removed_craters:
                removals_rows.append(RemovalRow(
                    crater_id=removed.id,
                    removed_by_id=crater.id
                ))

        # Only perform updates if the study region crater count ticked up
        if last_n_craters != n_craters_current:
            last_n_craters = n_craters_current

            areal_density = areal_density_calculator.areal_density

            if crater_record.n_craters_in_study_region > 1:
                mean_nn_distance = crater_record.get_mean_nearest_neighbor_distance()
                z = calculate_z_statistic(mean_nn_distance, crater_record.n_craters_in_study_region, study_region_area)
                za = calculate_za_statistic(mean_nn_distance,
                                            crater_record.n_craters_in_study_region,
                                            areal_density_calculator.area_covered,
                                            study_region_area)
            else:
                z = np.nan
                za = np.nan

            # Save stats
            statistics_rows.append(StatisticsRow(
                crater_id=crater.id,
                n_craters_added_in_study_region=n_craters_current,
                n_craters_in_study_region=crater_record.n_craters_in_study_region,
                areal_density=areal_density,
                z=z,
                za=za
            ))

            # Save state snapshot
            state_rows = []
            for report_crater in crater_record.all_craters_in_record:
                state_rows.append(StateRow(
                    last_crater_id=crater.id,
                    n_craters_added_in_study_region=n_craters_current,
                    crater_id=report_crater.id,
                    x=report_crater.x,
                    y=report_crater.y,
                    radius=report_crater.radius,
                    erased_rim_segments=list(crater_record.get_erased_rim_segments(report_crater.id)),
                    rim_percent_remaining=crater_record.get_remaining_rim_percent(report_crater.id)
                ))

            state_filename = f'{output_path}/state_{n_craters_current}.parquet'
            pd.DataFrame(state_rows).to_parquet(state_filename, index=False)

            if n_craters_current % output_image_cadence == 0:
                png_name = f'{output_path}/study_region_{n_craters_current}.png'
                save_study_region(areal_density_calculator, png_name)

        # Exit if we have generated our target number of craters.
        if n_craters_current == n_craters:
            break

    pd.DataFrame(statistics_rows).to_csv(f'{output_path}/statistics.csv', index=False)
    pd.DataFrame(removals_rows).to_csv(f'{output_path}/removals.csv', index=False)
    pd.DataFrame(all_craters_rows).to_csv(f'{output_path}/all_craters.csv', index=False)
