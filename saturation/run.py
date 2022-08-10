from saturation.distributions import ParetoProbabilityDistribution
from saturation.simulation import run_simulation, get_craters

if __name__ == '__main__':
    n_craters = 5000
    slope = 2
    observed_terrain_size = 10000
    terrain_padding = int(observed_terrain_size * 0.125)
    min_crater_radius = 2.5
    r_stat_multiplier = 9
    min_rim_percentage = 0.4
    effective_radius_multiplier = 1.5
    max_crater_radius = observed_terrain_size // 4
    r_stat = r_stat_multiplier * min_crater_radius

    full_terrain_size = observed_terrain_size + 2 * terrain_padding

    size_distribution = ParetoProbabilityDistribution(cdf_slope=slope,
                                                      x_min=min_crater_radius,
                                                      x_max=max_crater_radius)
    crater_generator = get_craters(size_distribution, full_terrain_size)

    for simulation_number in range(30):
        print(f'Simulation number: {simulation_number}')
        with open(f'/home/mason/output/sim_run_2_9_0.4_1.5_{simulation_number}.txt', 'w') as output_file:
            run_simulation(crater_generator,
                           n_craters,
                           r_stat,
                           r_stat_multiplier,
                           min_rim_percentage,
                           effective_radius_multiplier,
                           observed_terrain_size,
                           terrain_padding,
                           output_file)


    n_craters = 5000
    slope = 1
    observed_terrain_size = 10000
    terrain_padding = int(observed_terrain_size * 0.125)
    min_crater_radius = 2.5
    r_stat_multiplier = 3
    min_rim_percentage = 0.4
    effective_radius_multiplier = 1.5
    max_crater_radius = observed_terrain_size // 4
    r_stat = r_stat_multiplier * min_crater_radius

    full_terrain_size = observed_terrain_size + 2 * terrain_padding

    size_distribution = ParetoProbabilityDistribution(cdf_slope=slope,
                                                      x_min=min_crater_radius,
                                                      x_max=max_crater_radius)
    crater_generator = get_craters(size_distribution, full_terrain_size)

    for simulation_number in range(55):
        print(f'Simulation number: {simulation_number}')
        with open(f'/home/mason/output/sim_run_1_3_0.4_1.5_{simulation_number}.txt', 'w') as output_file:
            run_simulation(crater_generator,
                           n_craters,
                           r_stat,
                           r_stat_multiplier,
                           min_rim_percentage,
                           effective_radius_multiplier,
                           observed_terrain_size,
                           terrain_padding,
                           output_file)

