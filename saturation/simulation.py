import datetime
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Dict, List, Any

import numpy as np
import numba as nb
import yaml

from saturation.areal_density import ArealDensityCalculator
from saturation.crater_record import CraterRecord
from saturation.distributions import ProbabilityDistribution, ParetoProbabilityDistribution
from saturation.plotting import save_study_region
from saturation.stop_conditions import get_stop_condition
from saturation.writers import StatisticsWriter, StateSnapshotWriter, StatisticsRow, CraterWriter, CraterRemovalWriter
from saturation.z_stats import calculate_z_statistic, calculate_za_statistic
from saturation.datatypes import Crater

# Type definitions
LocationFunc = Callable[[], np.array]


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    simulation_id: int
    simulation_name: str
    random_seed: int
    slope: float
    rim_erasure_effectiveness_function: Dict[str, Any]
    mrp: float
    rmult: float
    study_region_size: int
    study_region_padding: int
    r_min: float
    r_stat: float
    r_max: float
    stop_condition: Dict
    write_statistics_cadence: int
    write_craters_cadence: int
    write_crater_removals_cadence: int
    write_state_cadence: int
    write_image_cadence: int
    write_image_points: List[int]
    spatial_hash_cell_size: int

    def to_dict(self) -> Dict:
        return {
            "simulation_id": self.simulation_id,
            "simulation_name": self.simulation_name,
            "random_seed": self.random_seed,
            "slope": self.slope,
            "rim_erasure_effectiveness_function": self.rim_erasure_effectiveness_function,
            "r_min": self.r_min,
            "r_stat": self.r_stat,
            "r_max": self.r_max,
            "mrp": self.mrp,
            "rmult": self.rmult,
            "study_region_size": self.study_region_size,
            "study_region_padding": self.study_region_padding,
            "stop_condition": self.stop_condition,
            "write_statistics_cadence": self.write_statistics_cadence,
            "write_craters_cadence": self.write_craters_cadence,
            "write_crater_removals_cadence": self.write_crater_removals_cadence,
            "write_state_cadence": self.write_state_cadence,
            "write_image_cadence": self.write_image_cadence,
            "write_image_points": self.write_image_points,
            "spatial_hash_cell_size": self.spatial_hash_cell_size,
        }


def get_multiplier_rim_erasure_effectiveness_function(multiplier: float) -> Callable[[float, float], bool]:
    @nb.njit
    def func(new_radius: float, existing_radius: float) -> bool:
        return new_radius > existing_radius * multiplier

    return func


@nb.njit
def log_rim_erasure_effectiveness_function(existing_radius: float, new_radius: float) -> bool:
    return np.log(new_radius) > existing_radius


@nb.njit
def sqrt_rim_erasure_effectiveness_function(existing_radius: float, new_radius: float) -> bool:
    return np.sqrt(new_radius) > existing_radius


def get_sin_log_rim_erasure_effectiveness_function(
        n_periods: float,
        min_r_period: float,
        max_r_period: float) -> Callable[[float, float], bool]:

    scale = 2 * np.pi / np.log(max_r_period / min_r_period) * n_periods

    @nb.njit
    def func(new_radius: float, existing_radius: float) -> bool:
        return new_radius > np.sin(np.log(existing_radius) * scale + 2) * existing_radius * 3

    return func


def get_rim_erasure_effectiveness_function(config: Dict[str, any]) -> Callable[[float, float], bool]:
    name = config["name"]

    result = None
    if name == "multiplier":
        result = get_multiplier_rim_erasure_effectiveness_function(config["multiplier"])
    elif name == "log":
        result = log_rim_erasure_effectiveness_function
    elif name == "sqrt":
        result = sqrt_rim_erasure_effectiveness_function
    elif name == "sin_log":
        result = get_sin_log_rim_erasure_effectiveness_function(config["n_periods"],
                                                                config["min_r_period"],
                                                                config["max_r_period"])

    return result


def get_craters(size_distribution: ProbabilityDistribution,
                full_region_size: float) -> Generator[Crater, None, None]:
    """
    Infinite generator for craters. full_region_size determines the size of the study region being impacted,
    including padding.
    """
    CHUNK_SIZE = 100000

    full_region_size = np.float32(full_region_size)

    crater_id = 1
    while True:
        index = (crater_id - 1) % CHUNK_SIZE
        if index == 0:
            xy = np.random.rand(CHUNK_SIZE, 2).astype("float32") * full_region_size
            radii = size_distribution.pullback(np.random.rand(CHUNK_SIZE)).astype("float32")

        yield Crater(
            id=crater_id,
            x=xy[index, 0],
            y=xy[index, 1],
            radius=radii[index]
        )
        crater_id += 1


def run_simulation(base_output_path: str, config: SimulationConfig):
    """
    Runs a simulation.
    Writes several output files to the output directory:
    - Parquet files of statistics after every crater added to the study region, output on a cadence.
    - Parquet files of the state after every crater is added to the study region, output on a cadence.
    - Sample output images of the study region, as PNGs.
    """
    print(f'Starting simulation {config.simulation_name}')
    sys.stdout.flush()

    output_path = Path(base_output_path) / config.simulation_name

    # Check if we should skip the run
    if os.path.exists(output_path / "completed.txt"):
        print(f'Found completion file for {config.simulation_name}, skipping...')
        return

    start_time = datetime.datetime.now()
    stop_condition = get_stop_condition(config.stop_condition)

    try:
        np.random.seed(config.random_seed)
        output_path.mkdir(parents=True, exist_ok=True)

        full_region_size = config.study_region_size + 2 * config.study_region_padding
        size_distribution = ParetoProbabilityDistribution(alpha=-config.slope,
                                                          x_min=config.r_min,
                                                          x_max=config.r_max)
        crater_generator = get_craters(size_distribution, full_region_size)

        start_time = datetime.datetime.now()

        # Write out the config file
        with open(output_path / "config.yaml", 'w') as config_output:
            yaml.dump(config.to_dict(), config_output)

        statistics_writer = StatisticsWriter(config.simulation_id,
                                             output_path,
                                             config.write_statistics_cadence)
        state_snapshot_writer = StateSnapshotWriter(config.simulation_id, output_path)
        crater_writer = CraterWriter(config.write_craters_cadence, config.simulation_id, output_path)
        crater_removals_writer = CraterRemovalWriter(config.write_crater_removals_cadence,
                                                     config.simulation_id,
                                                     output_path)

        study_region_area = config.study_region_size ** 2

        # The crater record handles removal of craters rims and the record
        # of what craters remain at a given point in time.
        r_stat = config.r_stat
        rim_erasure_effectiveness_function = get_rim_erasure_effectiveness_function(
            config.rim_erasure_effectiveness_function
        )
        crater_record = CraterRecord(config.r_stat,
                                     rim_erasure_effectiveness_function,
                                     config.mrp,
                                     config.rmult,
                                     config.study_region_size,
                                     config.study_region_padding,
                                     config.spatial_hash_cell_size)

        areal_density_calculator = ArealDensityCalculator((config.study_region_size, config.study_region_size),
                                                          (config.study_region_padding, config.study_region_padding),
                                                          r_stat)

        last_ntot = 0
        for crater in crater_generator:
            removed_craters = crater_record.add(crater)

            areal_density_calculator.add_crater(crater)
            if removed_craters:
                areal_density_calculator.remove_craters(removed_craters)
                crater_removals_writer.write(removed_craters, crater)

            if crater.radius >= r_stat:
                crater_writer.write(crater)

            # Only perform updates if the study region crater count ticked up
            ntot_current = crater_record.ntot
            if last_ntot != ntot_current:
                last_ntot = ntot_current

                areal_density = areal_density_calculator.areal_density

                if crater_record.nobs > 1:
                    mean_nn_distance = crater_record.get_mnnd()
                    z = calculate_z_statistic(mean_nn_distance, crater_record.nobs,
                                              study_region_area)
                    za = calculate_za_statistic(mean_nn_distance,
                                                crater_record.nobs,
                                                areal_density_calculator.area_covered,
                                                study_region_area)
                else:
                    z = np.nan
                    za = np.nan

                # Save stats
                statistics_row = StatisticsRow(
                    crater_id=crater.id,
                    ntot=ntot_current,
                    nobs=crater_record.nobs,
                    areal_density=areal_density,
                    mnnd=crater_record.get_mnnd(),
                    nnd_stdev=crater_record.get_nnd_stdev(),
                    nnd_min=crater_record.get_nnd_min(),
                    nnd_max=crater_record.get_nnd_max(),
                    radius_mean=crater_record.get_mean_radius(),
                    radius_stdev=crater_record.get_radius_stdev(),
                    z=z,
                    za=za
                )
                statistics_writer.write(statistics_row)

                if config.write_state_cadence != 0 and ntot_current % config.write_state_cadence == 0:
                    state_snapshot_writer.write_state_snapshot(crater_record, crater, ntot_current)

                if ntot_current in config.write_image_points \
                        or config.write_image_cadence != 0 and ntot_current % config.write_image_cadence == 0:
                    png_name = output_path / f"study_region_{ntot_current}.png"
                    save_study_region(areal_density_calculator, png_name)

                if stop_condition.should_stop(statistics_row):
                    break

        statistics_writer.close()
        crater_writer.close()
        crater_removals_writer.close()

        # Write out the completion file.
        with open(output_path / "completed.txt", "w") as completed_file:
            completed_file.write(f'duration: {(datetime.datetime.now() - start_time).total_seconds():.2f}')
    except:
        traceback.print_exc()

    duration = datetime.datetime.now() - start_time
    print(f'Finished simulation {config.simulation_name}, duration (seconds): {duration.total_seconds():.2f}')
    sys.stdout.flush()
