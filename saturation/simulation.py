from typing import Callable, TextIO, Generator

import numpy as np

from saturation.areal_density import ArealDensityCalculator
from saturation.crater_record import CraterRecord
from saturation.distributions import ProbabilityDistribution, ParetoProbabilityDistribution
from saturation.statistics import calculate_z_statistic, calculate_za_statistic
from saturation.datatypes import Crater


# Type definitions
LocationFunc = Callable[[], np.array]


def get_crater_location() -> np.array:
    """
    Returns an (x, y) crater location, uniformly distributed on [0, 1]
    """
    return np.random.rand(2)


def get_craters(size_distribution: ProbabilityDistribution,
                full_terrain_size: float,
                location_func: LocationFunc = get_crater_location) -> Generator[Crater, None, None]:
    """
    Infinite generator for craters. Scale defines the maximum size of the terrain.
    """
    crater_id = 1
    while True:
        locations = location_func()
        # radius = size_distribution.pullback(np.random.rand(1)[0])
        yield Crater(
            id=crater_id,
            x=locations[0] * full_terrain_size,
            y=locations[1] * full_terrain_size,
            radius=size_distribution.pullback(np.random.rand(1)[0])
        )
        crater_id += 1


def run_simulation(n_craters: int,
                   r_stat: float,
                   min_rim_percentage: float,
                   effective_radius_multiplier: float,
                   observed_terrain_size: int,
                   terrain_padding: int,
                   size_distribution: ProbabilityDistribution,
                   output_file: TextIO):
    """
    Runs a simulation.

    :param n_craters: Number of craters above r_stat to generate.
    :param r_stat: Minimum crater radius for statistics
    :param min_rim_percentage: Minimum rim percentage for a crater to remain in the record.
    :param effective_radius_multiplier: Multiplier on a crater's radius when destroying other rims.
    :param observed_terrain_size: Size of the observed terrain.
    :param terrain_padding: Padding around the edges of the terrain.
    :param size_distribution: Probability distribution for crater sizes.
    :param output_file: Output file.
    """
    # Write the header
    output_file.write('crater_id,n_craters_generated,n_craters_in_observation_area,areal_density,z,za\n')

    full_terrain_size = observed_terrain_size + 2 * terrain_padding
    observed_terrain_area = observed_terrain_size**2

    # The crater record handles removal of craters rims and the record
    # of what craters remain at a given point in time.
    crater_record = CraterRecord(r_stat,
                                 min_rim_percentage,
                                 effective_radius_multiplier,
                                 observed_terrain_size,
                                 terrain_padding)

    areal_density_calculator = ArealDensityCalculator(observed_terrain_size, terrain_padding)

    last_n_craters = 0
    for crater in get_craters(size_distribution, full_terrain_size):
        # Exit if we have generated our target number of craters.
        if crater_record.n_craters_added_in_observation_area == n_craters:
            break

        removed_craters = crater_record.add(crater)

        if removed_craters:
            areal_density_calculator.remove_craters(removed_craters)

        # Only perform updates if the observable crater count ticked up
        if last_n_craters != crater_record.n_craters_added_in_observation_area:
            last_n_craters = crater_record.n_craters_added_in_observation_area

            areal_density_calculator.add_crater(crater)
            areal_density = areal_density_calculator.areal_density

            if crater_record.n_craters_in_observation_area > 1:
                nn_distances = crater_record.get_nearest_neighbor_distances()
                z = calculate_z_statistic(nn_distances, observed_terrain_area)
                za = calculate_za_statistic(nn_distances,
                                            areal_density_calculator.area_covered,
                                            observed_terrain_area)
            else:
                z = np.nan
                za = np.nan

            # Write out stats
            n_craters_generated = crater_record.n_craters_added_in_observation_area
            n_craters_in_observation_area = crater_record.n_craters_in_observation_area
            output_file.write(
                f'{crater.id},{n_craters_generated},{n_craters_in_observation_area},{areal_density},{z},{za}\n')


if __name__ == '__main__':
    n_craters = 5000
    slope = 1
    observed_terrain_size = 10000
    terrain_padding = int(observed_terrain_size * 0.125)
    min_crater_radius = 2.5
    r_stat_multiplier = 9
    min_rim_percentage = 0.6
    effective_radius_multiplier = 1.5
    max_crater_radius = observed_terrain_size // 4
    r_stat = r_stat_multiplier * min_crater_radius

    size_distribution = ParetoProbabilityDistribution(cdf_slope=slope,
                                                      x_min=min_crater_radius,
                                                      x_max=max_crater_radius)

    for simulation_number in range(100):
        print(f'Simulation number: {simulation_number}')
        with open(f'/home/mason/output/sim_run_1_9_0.6_1.5_{simulation_number}.txt', 'w') as output_file:
            run_simulation(n_craters,
                           r_stat,
                           min_rim_percentage,
                           effective_radius_multiplier,
                           observed_terrain_size,
                           terrain_padding,
                           size_distribution,
                           output_file)

    n_craters = 5000
    slope = 1
    observed_terrain_size = 10000
    terrain_padding = int(observed_terrain_size * 0.125)
    min_crater_radius = 2.5
    r_stat_multiplier = 9
    min_rim_percentage = 0.4
    effective_radius_multiplier = 1.1
    max_crater_radius = observed_terrain_size // 4
    r_stat = r_stat_multiplier * min_crater_radius

    size_distribution = ParetoProbabilityDistribution(cdf_slope=slope,
                                                      x_min=min_crater_radius,
                                                      x_max=max_crater_radius)

    for simulation_number in range(100):
        print(f'Simulation number: {simulation_number}')
        with open(f'/home/mason/output/sim_run_1_9_0.4_1.1_{simulation_number}.txt', 'w') as output_file:
            run_simulation(n_craters,
                           r_stat,
                           min_rim_percentage,
                           effective_radius_multiplier,
                           observed_terrain_size,
                           terrain_padding,
                           size_distribution,
                           output_file)
