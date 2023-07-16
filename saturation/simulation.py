import datetime
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Dict, List

import numpy as np
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
    r_stat_multiplier: float
    min_rim_percentage: float
    effective_radius_multiplier: float
    study_region_size: int
    study_region_padding: int
    min_crater_radius: float
    max_crater_radius: float
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
            "r_stat_multiplier": self.r_stat_multiplier,
            "min_rim_percentage": self.min_rim_percentage,
            "effective_radius_multiplier": self.effective_radius_multiplier,
            "study_region_size": self.study_region_size,
            "study_region_padding": self.study_region_padding,
            "min_crater_radius": self.min_crater_radius,
            "max_crater_radius": self.max_crater_radius,
            "stop_condition": self.stop_condition,
            "write_statistics_cadence": self.write_statistics_cadence,
            "write_craters_cadence": self.write_craters_cadence,
            "write_crater_removals_cadence": self.write_crater_removals_cadence,
            "write_state_cadence": self.write_state_cadence,
            "write_image_cadence": self.write_image_cadence,
            "write_image_points": self.write_image_points,
            "spatial_hash_cell_size": self.spatial_hash_cell_size,
        }


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

        r_stat = config.min_crater_radius

        full_region_size = config.study_region_size + 2 * config.study_region_padding
        size_distribution = ParetoProbabilityDistribution(cdf_slope=config.slope,
                                                          x_min=config.min_crater_radius / config.r_stat_multiplier,
                                                          x_max=config.max_crater_radius)
        crater_generator = get_craters(size_distribution, full_region_size)

        start_time = datetime.datetime.now()

        output_path.mkdir(parents=True, exist_ok=True)

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
        crater_record = CraterRecord(r_stat,
                                     config.r_stat_multiplier,
                                     config.min_rim_percentage,
                                     config.effective_radius_multiplier,
                                     config.study_region_size,
                                     config.study_region_padding,
                                     config.spatial_hash_cell_size)

        areal_density_calculator = ArealDensityCalculator((config.study_region_size, config.study_region_size),
                                                          (config.study_region_padding, config.study_region_padding),
                                                          r_stat)

        last_n_craters = 0
        for crater in crater_generator:
            removed_craters = crater_record.add(crater)

            areal_density_calculator.add_crater(crater)
            if removed_craters:
                areal_density_calculator.remove_craters(removed_craters)
                crater_removals_writer.write(removed_craters, crater)

            if crater.radius >= r_stat:
                crater_writer.write(crater)

            # Only perform updates if the study region crater count ticked up
            n_craters_current = crater_record.n_craters_added_in_study_region
            if last_n_craters != n_craters_current:
                last_n_craters = n_craters_current

                areal_density = areal_density_calculator.areal_density

                if crater_record.n_craters_in_study_region > 1:
                    mean_nn_distance = crater_record.get_mean_nearest_neighbor_distance()
                    z = calculate_z_statistic(mean_nn_distance, crater_record.n_craters_in_study_region,
                                              study_region_area)
                    za = calculate_za_statistic(mean_nn_distance,
                                                crater_record.n_craters_in_study_region,
                                                areal_density_calculator.area_covered,
                                                study_region_area)
                else:
                    z = np.nan
                    za = np.nan

                # Save stats
                statistics_row = StatisticsRow(
                    crater_id=crater.id,
                    n_craters_added_in_study_region=n_craters_current,
                    n_craters_in_study_region=crater_record.n_craters_in_study_region,
                    areal_density=areal_density,
                    z=z,
                    za=za
                )
                statistics_writer.write(statistics_row)

                if config.write_state_cadence != 0 and n_craters_current % config.write_state_cadence == 0:
                    state_snapshot_writer.write_state_snapshot(crater_record, crater, n_craters_current)

                if n_craters_current in config.write_image_points \
                        or config.write_image_cadence != 0 and n_craters_current % config.write_image_cadence == 0:
                    png_name = output_path / f"study_region_{n_craters_current}.png"
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
