from typing import Callable, TextIO, List

import pandas as pd
import numpy as np

from saturation.areal_density import ArealDensityCalculator
from saturation.crater_record import CraterRecord
from saturation.distributions import ProbabilityDistribution, PowerLawProbabilityDistribution

# Type definitions
from saturation.geometry import calculate_rim_percentage_remaining, get_erased_rim_arcs
from saturation.statistics import calculate_z_statistic, calculate_za_statistic

LocationFunc = Callable[[int], np.array]


def get_crater_locations(n_craters: int) -> np.array:
    """
    Returns n_craters crater locations, uniformly distributed on [0, 1]
    """
    return np.random.rand(n_craters, 2)


def get_craters(n_craters: int,
                size_distribution: ProbabilityDistribution,
                scale: float,
                location_func: LocationFunc = get_crater_locations) -> pd.DataFrame:
    """
    Returns a dataframe of n_craters, including (x, y) center locations and radii.
    Scale defines the maximum size of the terrain.
    """
    ids = np.arange(1, n_craters + 1, dtype=int)
    locations = location_func(n_craters) * scale
    radii = [size_distribution.uniform_to_density(x) for x in np.random.rand(n_craters)]
    data_dict = {
        'id': ids,
        'x': locations[:, 0],
        'y': locations[:, 1],
        'radius': radii
    }
    data = pd.DataFrame(data_dict).set_index(['id'])

    return data


def run_simulation(n_craters: int,
                   size_distribution: ProbabilityDistribution,
                   min_crater_radius: float,
                   min_rim_percentage: float,
                   effective_radius_multiplier: float,
                   min_crater_radius_multiplier_for_stats: float,
                   terrain_size: int,
                   output_file: TextIO):
    """
    Runs a simulation.

    :param n_craters: Number of craters to generate.
    :param size_distribution: Probability distribution for crater sizes.
    :param min_crater_radius: Minimum crater radius.
    :param min_rim_percentage: Minimum rim percentage for a crater to remain in the record.
    :param effective_radius_multiplier: Multiplier on a crater's radius when destroying other rims.
    :param min_crater_radius_multiplier_for_stats: Multiple of the minimum crater radius for consideration in stats calculation (r_stat).
    :param terrain_size: Size of the terrain.
    :param output_filename: Output filename.
    """
    # Write the header
    output_file.write('crater_id,n_stats_craters,crater_count,areal_density,z,za\n')

    min_crater_radius_for_stats = min_crater_radius * min_crater_radius_multiplier_for_stats

    margin = terrain_size // 10
    limited_terrain_size = terrain_size - 2 * margin
    areal_density_calculator = ArealDensityCalculator(terrain_size, margin)

    # Get craters according to the SFD
    craters = get_craters(n_craters, size_distribution, terrain_size)

    # The crater record handles removal of craters rims and the record
    # of what craters remain at a given point in time.
    crater_record = CraterRecord(craters,
                                 min_crater_radius_for_stats,
                                 min_rim_percentage,
                                 effective_radius_multiplier,
                                 terrain_size,
                                 margin)

    # Simulation loop, one new crater at a time.
    stats_calculated_crater_count = 0
    for crater_row in craters.itertuples():
        crater_id = crater_row.Index

        removed_crater_ids = crater_record.update(crater_id)

        # Calculate statistics
        if crater_row.radius >= min_crater_radius_for_stats:
            stats_calculated_crater_count += 1
            areal_density_calculator.update(craters.loc[[crater_id]], craters.loc[removed_crater_ids])
            areal_density = areal_density_calculator.get_areal_density()
            z = calculate_z_statistic(crater_record.get_nearest_neighbor_distances(), limited_terrain_size)
            za = calculate_za_statistic(crater_record.get_nearest_neighbor_distances(),
                                        areal_density * limited_terrain_size**2,
                                        limited_terrain_size)

            output_file.write(f'{crater_id},{stats_calculated_crater_count},{len(crater_record.get_crater_ids())},{areal_density},{z},{za}\n')

        if crater_id % 500 == 0:
            print(f'{crater_id},{stats_calculated_crater_count},{len(crater_record.get_crater_ids())},{areal_density},{z},{za}')


if __name__ == '__main__':
    n_craters = 30000
    slope = -2
    terrain_size = 12500
    min_crater_radius = 2.5
    min_rim_percentage = 0.4
    effective_radius_multiplier = 1.5
    min_crater_radius_multiplier_for_stats = 9
    max_crater_radius = (terrain_size * 0.8) // 4

    size_distribution = PowerLawProbabilityDistribution(slope=slope,
                                                        min_value=min_crater_radius,
                                                        max_value=max_crater_radius)

    for simulation_number in range(100):
        with open(f'/home/mason/output/simulation_run_{simulation_number}.txt', 'w') as output_file:
            run_simulation(n_craters,
                           size_distribution,
                           min_crater_radius,
                           min_rim_percentage,
                           effective_radius_multiplier,
                           min_crater_radius_multiplier_for_stats,
                           terrain_size,
                           output_file)
