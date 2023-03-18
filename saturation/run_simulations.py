import multiprocessing
import sys
from typing import Dict, List

import yaml

from saturation.simulation import run_simulation, SimulationConfig


def get_simulation_configs(config: Dict) -> List[SimulationConfig]:
    """
    Given a config, read directly from the YAML config file,
    creates a set of SimulationConfigs, one per run.
    """
    base_output_path = config['output_path']
    write_statistics_cadence = config['write_statistics_cadence']
    write_craters_cadence = config['write_craters_cadence']
    write_crater_removals_cadence = config['write_crater_removals_cadence']
    write_state_cadence = config['write_state_cadence']
    write_image_cadence = config['write_image_cadence']

    result = []
    run_configurations = sorted(config['run_configurations'], key=lambda x: x[list(x.keys())[0]]["slope"], reverse=True)
    for sim_group_config in run_configurations:
        simulation_id, values = list(sim_group_config.items())[0]
        simulation_name = values["simulation_name"]
        output_path = f'{base_output_path}/{simulation_name}/'

        result.append(SimulationConfig(
            simulation_id=simulation_id,
            simulation_name=simulation_name,
            output_path=output_path,
            slope=values['slope'],
            r_stat_multiplier=values['r_stat_multiplier'],
            min_rim_percentage=values['min_rim_percentage'],
            effective_radius_multiplier=values['effective_radius_multiplier'],
            study_region_size=values['study_region_size'],
            study_region_padding=values['study_region_padding'],
            min_crater_radius=values['min_crater_radius'],
            max_crater_radius=values['max_crater_radius'],
            stop_condition=values['stop_condition'],
            write_statistics_cadence=write_statistics_cadence,
            write_craters_cadence=write_craters_cadence,
            write_crater_removals_cadence=write_crater_removals_cadence,
            write_state_cadence=write_state_cadence,
            write_image_cadence=write_image_cadence,
            spatial_hash_cell_size=values['spatial_hash_cell_size'],
        ))

    return result


def main(config_filename: str):
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)

    n_workers = config['n_workers']
    simulation_configs = get_simulation_configs(config)

    if n_workers > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for simulation_config in simulation_configs:
                pool.apply_async(run_simulation, (simulation_config, ))

            pool.close()
            pool.join()
    else:
        for simulation_config in simulation_configs:
            run_simulation(simulation_config)


if __name__ == '__main__':
    main(sys.argv[1])
