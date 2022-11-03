import datetime
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import yaml

from saturation.distributions import ParetoProbabilityDistribution
from saturation.simulation import run_simulation, get_craters


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    simulation_name: str
    output_path: str
    simulation_id: int
    slope: float
    r_stat_multiplier: float
    min_rim_percentage: float
    effective_radius_multiplier: float
    study_region_size: int
    study_region_padding: int
    min_crater_radius: float
    max_crater_radius: float
    n_craters: int


def run_single_simulation(config: SimulationConfig):
    np.random.seed(config.simulation_id)

    r_stat = config.r_stat_multiplier * config.min_crater_radius

    full_region_size = config.study_region_size + 2 * config.study_region_padding
    size_distribution = ParetoProbabilityDistribution(cdf_slope=config.slope,
                                                      x_min=config.min_crater_radius,
                                                      x_max=config.max_crater_radius)
    crater_generator = get_craters(size_distribution, full_region_size)

    print(f'Starting simulation {config.simulation_name}')
    start_time = datetime.datetime.now()

    path = Path(config.output_path)
    path.mkdir(parents=True, exist_ok=True)

    run_simulation(crater_generator,
                   config.n_craters,
                   r_stat,
                   config.r_stat_multiplier,
                   config.min_rim_percentage,
                   config.effective_radius_multiplier,
                   config.study_region_size,
                   config.study_region_padding,
                   config.max_crater_radius,
                   config.output_path)

    duration = datetime.datetime.now() - start_time
    print(f'Finished simulation {config.simulation_name}, duration (seconds): {duration.total_seconds():.2f}')


def get_simulation_configs(config: Dict) -> List[SimulationConfig]:
    """
    Given a config, read directly from the YAML config file,
    creates a set of SimulationConfigs, one per run.
    """
    base_output_path = config['output_path']

    result = []
    for sim_group_config in config['run_configurations']:
        simulation_group_name, values = list(sim_group_config.items())[0]
        n_simulations = values['n_simulations']

        for simulation_id in range(1, n_simulations + 1):
            simulation_name = f'{simulation_group_name}_{simulation_id}'
            output_path = f'{base_output_path}/{simulation_group_name}/{simulation_id}/'

            result.append(SimulationConfig(
                simulation_name=simulation_name,
                output_path=output_path,
                simulation_id=simulation_id,
                slope=values['slope'],
                r_stat_multiplier=values['r_stat_multiplier'],
                min_rim_percentage=values['min_rim_percentage'],
                effective_radius_multiplier=values['effective_radius_multiplier'],
                study_region_size=values['study_region_size'],
                study_region_padding=values['study_region_padding'],
                min_crater_radius=values['min_crater_radius'],
                max_crater_radius=values['max_crater_radius'],
                n_craters=values['n_craters']
            ))

    return result


def main(config_filename: str):
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)

    n_workers = config['n_workers']
    simulation_configs = get_simulation_configs(config)

    # with multiprocessing.Pool(processes=n_workers) as pool:
    #     for simulation_config in simulation_configs:
    #         pool.apply_async(run_single_simulation, (simulation_config, ))
    #
    #     pool.close()
    #     pool.join()
    for simulation_config in simulation_configs:
        run_single_simulation(simulation_config)


if __name__ == '__main__':
    main(sys.argv[1])
