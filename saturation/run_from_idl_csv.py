import pandas as pd
from typing import Iterable

from saturation.datatypes import Crater
from saturation.simulation import run_simulation


def get_craters_from_csv(filename: str, terrain_padding: int) -> Iterable[Crater]:
    data = pd.read_csv(filename, skiprows=3, header=None)
    data.columns = ['x', 'y', 'radius']

    id_counter = 1
    for item in data.itertuples():
        yield Crater(id=id_counter,
                     x=item.x + terrain_padding,
                     y=item.y + terrain_padding,
                     radius=item.radius)
        id_counter += 1


if __name__ == '__main__':
    filename = '/home/mason/code/kirchoff_saturation/output/run_-1.00_1.50_0.40_3.00_1_2.csv'

    n_craters = 5000
    observed_terrain_size = 10000
    terrain_padding = int(observed_terrain_size * 0.125)
    min_crater_radius = 2.5
    r_stat_multiplier = 3
    min_rim_percentage = 0.40
    effective_radius_multiplier = 1.5
    max_crater_radius = observed_terrain_size // 4
    r_stat = r_stat_multiplier * min_crater_radius

    crater_generator = get_craters_from_csv(filename, terrain_padding)

    with open(f'/home/mason/output/replication_sim_run_1_3_0.4_1.5.txt', 'w') as output_file:
        run_simulation(crater_generator,
                       n_craters,
                       r_stat,
                       r_stat_multiplier,
                       min_rim_percentage,
                       effective_radius_multiplier,
                       observed_terrain_size,
                       terrain_padding,
                       output_file)
