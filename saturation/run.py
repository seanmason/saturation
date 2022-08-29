import multiprocessing
import numpy as np
from functools import partial
import os

from saturation.distributions import ParetoProbabilityDistribution
from saturation.simulation import run_simulation, get_craters


def run_single_simulation(n_craters: int,
                          slope: float,
                          observed_terrain_size: int,
                          min_crater_radius: float,
                          r_stat_multiplier: float,
                          min_rim_percentage: float,
                          effective_radius_multiplier: float,
                          output_path: str,
                          simulation_id: int) -> int:
    np.random.seed(simulation_id)

    terrain_padding = int(observed_terrain_size * 0.125)
    max_crater_radius = observed_terrain_size // 4
    r_stat = r_stat_multiplier * min_crater_radius

    full_terrain_size = observed_terrain_size + 2 * terrain_padding
    size_distribution = ParetoProbabilityDistribution(cdf_slope=slope,
                                                      x_min=min_crater_radius,
                                                      x_max=max_crater_radius)
    crater_generator = get_craters(size_distribution, full_terrain_size)

    print(f'Start simulation {simulation_id}')

    simulation_output_path = os.path.join(output_path, str(simulation_id))
    if not os.path.exists(simulation_output_path):
        os.mkdir(simulation_output_path)

    run_simulation(crater_generator,
                   n_craters,
                   r_stat,
                   r_stat_multiplier,
                   min_rim_percentage,
                   effective_radius_multiplier,
                   observed_terrain_size,
                   terrain_padding,
                   simulation_output_path)

    return simulation_id


if __name__ == '__main__':
    N_WORKERS = 1
    N_SIMULATIONS = 5

    N_CRATERS = 20000
    SLOPE = 2.8
    OBSERVED_TERRAIN_SIZE = 10000
    MIN_CRATER_RADIUS = 2.5
    R_STAT_MULTIPLIER = 9
    MIN_RIM_PERCENTAGE = 0.6
    EFFECTIVE_RADIUS_MULTIPLIER = 1.9

    BASE_OUTPUT_PATH = '/home/mason/output/'
    OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, f'{SLOPE:0.2f}_{R_STAT_MULTIPLIER:0.2f}_{MIN_RIM_PERCENTAGE:0.2f}_{EFFECTIVE_RADIUS_MULTIPLIER:0.2f}')
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        it = pool.imap_unordered(
            partial(run_single_simulation,
                    N_CRATERS,
                    SLOPE,
                    OBSERVED_TERRAIN_SIZE,
                    MIN_CRATER_RADIUS,
                    R_STAT_MULTIPLIER,
                    MIN_RIM_PERCENTAGE,
                    EFFECTIVE_RADIUS_MULTIPLIER,
                    OUTPUT_PATH),
            list(range(1, N_SIMULATIONS + 1))
        )
        for x in it:
            print(f'Completed simulation {x}')
