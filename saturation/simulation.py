from typing import Callable, TextIO, Generator, Iterable

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
        yield Crater(
            id=crater_id,
            x=locations[0] * full_terrain_size,
            y=locations[1] * full_terrain_size,
            radius=size_distribution.pullback(np.random.rand(1)[0])
        )
        crater_id += 1


def run_simulation(crater_generator: Iterable[Crater],
                   n_craters: int,
                   r_stat: float,
                   r_stat_multiplier: float,
                   min_rim_percentage: float,
                   effective_radius_multiplier: float,
                   observed_terrain_size: int,
                   terrain_padding: int,
                   output_file: TextIO):
    """
    Runs a simulation.

    :param crater_generator: Crater generator.
    :param n_craters: Number of craters above r_stat to generate.
    :param r_stat: Minimum crater radius for statistics
    :param r_stat_multiplier: r_stat multiplier
    :param min_rim_percentage: Minimum rim percentage for a crater to remain in the record.
    :param effective_radius_multiplier: Multiplier on a crater's radius when destroying other rims.
    :param observed_terrain_size: Size of the observed terrain.
    :param terrain_padding: Padding around the edges of the terrain.
    :param output_file: Output file.
    """
    # Write the header
    output_file.write('crater_id,n_craters_generated,n_craters_in_observation_area,areal_density,z,za\n')

    observed_terrain_area = observed_terrain_size**2

    # The crater record handles removal of craters rims and the record
    # of what craters remain at a given point in time.
    crater_record = CraterRecord(r_stat,
                                 r_stat_multiplier,
                                 min_rim_percentage,
                                 effective_radius_multiplier,
                                 observed_terrain_size,
                                 terrain_padding)

    areal_density_calculator = ArealDensityCalculator(observed_terrain_size, terrain_padding, r_stat)

    last_n_craters = 0
    for crater in crater_generator:
        # Exit if we have generated our target number of craters.
        if crater_record.n_craters_added_in_observation_area == n_craters:
            break

        removed_craters = crater_record.add(crater)

        areal_density_calculator.add_crater(crater)
        if removed_craters:
            areal_density_calculator.remove_craters(removed_craters)

        # Only perform updates if the observable crater count ticked up
        if last_n_craters != crater_record.n_craters_added_in_observation_area:
            last_n_craters = crater_record.n_craters_added_in_observation_area

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
