import datetime
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Dict, List, Any

import numpy as np
import yaml

from saturation.areal_density import ArealDensityCalculator
from saturation.crater_record import CraterRecord
from saturation.distributions import ProbabilityDistribution, ParetoProbabilityDistribution
from saturation.initial_rim_state_calculators import get_initial_rim_state_calculator
from saturation.rim_erasure_calculators import get_rim_erasure_calculator
from saturation.plotting import save_study_region
from saturation.stop_conditions import get_stop_condition
from saturation.writers import StatisticsWriter, StateSnapshotWriter, StatisticsRow, CraterWriter, CraterRemovalWriter
from saturation.z_stats import calculate_z_statistic, calculate_za_statistic
from saturation.datatypes import Crater


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    simulation_id: int
    simulation_name: str
    random_seed: int
    slope: float
    rim_erasure_method: Dict[str, Any]
    initial_rim_calculation_method: Dict[str, Any]
    mrp: float
    rmult: float
    study_region_size: int
    study_region_padding: int
    rmin: float
    rstat: float
    rmax: float
    stop_condition: Dict
    calculate_areal_density: bool
    calculate_nearest_neighbor_stats: bool
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
            "rim_erasure_method": self.rim_erasure_method,
            "initial_rim_calculation_method": self.initial_rim_calculation_method,
            "rmin": self.rmin,
            "rstat": self.rstat,
            "rmax": self.rmax,
            "mrp": self.mrp,
            "rmult": self.rmult,
            "study_region_size": self.study_region_size,
            "study_region_padding": self.study_region_padding,
            "stop_condition": self.stop_condition,
            "calculate_areal_density": self.calculate_areal_density,
            "calculate_nearest_neighbor_stats": self.calculate_nearest_neighbor_stats,
            "write_statistics_cadence": self.write_statistics_cadence,
            "write_craters_cadence": self.write_craters_cadence,
            "write_crater_removals_cadence": self.write_crater_removals_cadence,
            "write_state_cadence": self.write_state_cadence,
            "write_image_cadence": self.write_image_cadence,
            "write_image_points": self.write_image_points,
            "spatial_hash_cell_size": self.spatial_hash_cell_size,
        }


def get_craters(
    *,
    size_distribution: ProbabilityDistribution,
    region_size: float,
    min_radius_threshold: float,
    random_seed: int,
) -> Generator[Crater, None, None]:
    """
    Infinite generator for craters. Generates craters only if their radius is above min_radius_threshold.
    Assumes valid craters are extremely rare (e.g., 1 in 1000).
    """
    CHUNK_SIZE = int(1e6)
    BATCH_SIZE = CHUNK_SIZE
    DTYPE = np.float32

    rng = np.random.default_rng(seed=random_seed)

    uniform_threshold = DTYPE(size_distribution.cdf(min_radius_threshold))

    region_size = DTYPE(region_size)
    crater_id = 1
    index = CHUNK_SIZE

    xy = np.empty(shape=(CHUNK_SIZE, 2), dtype=DTYPE)
    radii = np.empty(shape=CHUNK_SIZE, dtype=DTYPE)
    uniform = np.empty(shape=BATCH_SIZE, dtype=DTYPE)
    crater_ids = np.empty(shape=BATCH_SIZE, dtype=np.int64)

    while True:
        if index >= CHUNK_SIZE:
            rng.random(dtype=DTYPE, out=xy)
            np.multiply(xy, region_size, out=xy)
            rng.random(dtype=DTYPE, out=uniform)
            count = 0

            while count < CHUNK_SIZE:
                rng.random(dtype=DTYPE, out=uniform)
                indexes_over_threshold = np.where(uniform >= uniform_threshold)[0]
                n_over_threshold = indexes_over_threshold.size

                if count + n_over_threshold > CHUNK_SIZE:
                    n_over_threshold = CHUNK_SIZE - count

                upper_index = min(count + n_over_threshold, CHUNK_SIZE)
                n_entries = upper_index - count
                radii[count:upper_index] = uniform[indexes_over_threshold][:n_entries]
                crater_ids[count:upper_index] = crater_id + indexes_over_threshold[:n_entries]

                count += n_entries
                crater_id += BATCH_SIZE

            radii = size_distribution.pullback(radii).astype("float32")
            index = 0

        while index < CHUNK_SIZE:
            yield Crater(
                id=crater_ids[index],
                x=xy[index, 0],
                y=xy[index, 1],
                radius=radii[index]
            )
            index += 1



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

    output_path = Path(base_output_path) / str(config.simulation_name)

    # Check if we should skip the run
    if os.path.exists(output_path / "completed.txt"):
        print(f'Found completion file for {config.simulation_name}, skipping...')
        return

    stop_condition = get_stop_condition(config.stop_condition)
    calculate_areal_density = config.calculate_areal_density
    calculate_nearest_neighbor_stats = config.calculate_nearest_neighbor_stats

    start_time = datetime.datetime.now()
    try:
        np.random.seed(config.random_seed)
        output_path.mkdir(parents=True, exist_ok=True)

        full_region_size = config.study_region_size + 2 * config.study_region_padding
        size_distribution = ParetoProbabilityDistribution(alpha=-config.slope,
                                                          x_min=config.rmin,
                                                          x_max=config.rmax)

        # The crater record handles removal of craters rims and the record
        # of what craters remain at a given point in time.
        rstat = config.rstat
        rim_erasure_calculator = get_rim_erasure_calculator(
            config=config.rim_erasure_method,
            rmult=config.rmult,
            rstat=rstat
        )

        min_radius_threshold = rim_erasure_calculator.get_min_radius_threshold()
        crater_generator = get_craters(
            size_distribution=size_distribution,
            region_size=full_region_size,
            min_radius_threshold=min_radius_threshold,
            random_seed=config.random_seed
        )

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

        initial_rim_state_calculator = get_initial_rim_state_calculator(config.initial_rim_calculation_method)
        crater_record = CraterRecord(
            rstat=config.rstat,
            rim_erasure_calculator=rim_erasure_calculator,
            initial_rim_state_calculator=initial_rim_state_calculator,
            mrp=config.mrp,
            rmult=config.rmult,
            study_region_size=config.study_region_size,
            study_region_padding=config.study_region_padding,
            cell_size=config.spatial_hash_cell_size,
            calculate_nearest_neighbor_stats=calculate_nearest_neighbor_stats
        )

        areal_density_calculator = ArealDensityCalculator(
            study_region_size=config.study_region_size,
            study_region_padding=config.study_region_padding,
            rstat=rstat
        )

        last_nstat = 0
        craters_smaller_than_rstat = []
        for crater in crater_generator:
            if crater.radius < rstat:
                craters_smaller_than_rstat.append(crater)
            else:
                if craters_smaller_than_rstat:
                    removed_craters = crater_record.add_craters_smaller_than_rstat(craters_smaller_than_rstat)
                    craters_smaller_than_rstat = []

                    if removed_craters:
                        if calculate_areal_density:
                            areal_density_calculator.remove_craters(removed_craters)
                        crater_removals_writer.write(removed_craters, crater)

                removed_craters = crater_record.add_crater_geq_rstat(crater)

                if calculate_areal_density:
                    areal_density_calculator.add_crater(crater)
                    if removed_craters:
                        areal_density_calculator.remove_craters(removed_craters)

                if removed_craters:
                    crater_removals_writer.write(removed_craters, crater)

                crater_writer.write(crater)

                # Only perform updates if the nstat ticked up
                nstat_current = crater_record.nstat
                if last_nstat != nstat_current:
                    last_nstat = nstat_current

                    areal_density = areal_density_calculator.areal_density

                    if crater_record.nobs > 1:
                        mean_nn_distance = crater_record.get_mnnd()
                        z = calculate_z_statistic(
                            mean_nn_distance, crater_record.nobs, study_region_area
                        )
                        za = calculate_za_statistic(
                            mean_nn_distance,
                            crater_record.nobs,
                            areal_density_calculator.area_covered,
                            study_region_area
                        )
                    else:
                        z = np.nan
                        za = np.nan

                    # Save stats
                    statistics_row = StatisticsRow(
                        crater_id=crater.id,
                        nstat=nstat_current,
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

                    if config.write_state_cadence != 0 and nstat_current % config.write_state_cadence == 0:
                        state_snapshot_writer.write_state_snapshot(crater_record, crater, nstat_current)

                    if nstat_current in config.write_image_points or config.write_image_cadence != 0 and nstat_current % config.write_image_cadence == 0:
                        png_name = output_path / f"study_region_{nstat_current}.png"
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
