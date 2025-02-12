from joblib import Parallel, delayed
import sys
from typing import Dict, List

import yaml

from saturation.simulation import run_simulation, SimulationConfig


def get_simulation_configs(config: Dict) -> List[SimulationConfig]:
    """
    Given a config, read directly from the YAML config file,
    creates a set of SimulationConfigs, one per run.
    """
    result = []

    # Order sims by their expected runtimes. Longer-running sims should be started first.
    # Sims that calculate nearest neighbor stats first
    # Then prioritize high rmax/rmin ratio
    # Then prioritize steep slopes
    key_func = lambda x: (not x[1]["calculate_nearest_neighbor_stats"], x[1]["rmin"] / x[1]["rmax"] , x[1]["slope"])
    run_configurations = sorted(config["run_configurations"].items(), key=key_func)
    for simulation_id, values in run_configurations:
        result.append(
            SimulationConfig(
                simulation_id=simulation_id,
                simulation_name=simulation_id,
                random_seed=values["random_seed"],
                slope=values["slope"],
                mrp=values["mrp"],
                rmult=values["rmult"],
                rim_erasure_method=values["rim_erasure_method"],
                initial_rim_calculation_method=values["initial_rim_calculation_method"],
                rmin=values["rmin"],
                rstat=values["rstat"],
                rmax=values["rmax"],
                study_region_size=values["study_region_size"],
                study_region_padding=values["study_region_padding"],
                stop_condition=values["stop_condition"],
                calculate_areal_density=values["calculate_areal_density"],
                calculate_nearest_neighbor_stats=values["calculate_nearest_neighbor_stats"],
                write_statistics_cadence=values["write_statistics_cadence"],
                write_craters_cadence=values["write_craters_cadence"],
                write_crater_removals_cadence=values["write_crater_removals_cadence"],
                write_state_cadence=values["write_state_cadence"],
                write_image_cadence=values["write_image_cadence"],
                write_image_points=values["write_image_points"],
                spatial_hash_cell_size=values["spatial_hash_cell_size"],
            )
        )

    return result


def main(base_output_path: str, config_filename: str):
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)

    n_workers = config["n_workers"]
    simulation_configs = get_simulation_configs(config)

    if n_workers > 1:
        Parallel(n_jobs=n_workers)(
            delayed(run_simulation)(base_output_path, simulation_config)
            for simulation_config in simulation_configs
        )
    else:
        for simulation_config in simulation_configs:
            run_simulation(base_output_path, simulation_config)



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
