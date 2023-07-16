import sys
from typing import Dict
import itertools
import copy

import numpy as np
import yaml


def get_uniform_random(minimum: float, maximum: float) -> float:
    return np.random.random() * (maximum - minimum) + minimum


def get_random_normal(mean: float, stdev: float) -> float:
    return np.random.normal(loc=mean, scale=stdev)


def create_central_composite_design_configs(base_output_path: str, config: Dict):
    np.random.seed(config["random_seed"])

    run_configurations = []
    output_config = {
        "n_workers": config["n_workers"],
        "write_statistics_cadence": config["write_statistics_cadence"],
        "write_craters_cadence": config["write_craters_cadence"],
        "write_crater_removals_cadence": config["write_crater_removals_cadence"],
        "write_state_cadence": config["write_state_cadence"],
        "write_image_cadence": config["write_image_cadence"],
        "write_image_points": config["write_image_points"],
        "run_configurations": run_configurations
    }

    parameters = config["parameters"]
    n_levels = config["n_levels"]

    ranges = []
    for parameter_dict in parameters:
        name = list(parameter_dict.keys())[0]
        values = parameter_dict[name]

        min_value = values["min"]
        max_value = values["max"]
        value_range = max_value - min_value

        level_config = []
        for level in range(n_levels):
            level_config.append({
                "name": name,
                "value": min_value + level / (n_levels - 1) * value_range,
                "stdev": value_range * .005
            })

        ranges.append(level_config)

    # Add configuration per combination of all levels
    id_counter = 1
    ccd_points = itertools.product(*ranges)
    for ccd_point in ccd_points:
        for sim_number_for_location in range(config["n_simulations_per_location"]):
            ccd_point_parameters = []
            for parameter in ccd_point:
                if sim_number_for_location == 0:
                    value = parameter["value"]
                else:
                    value = get_random_normal(parameter["value"], parameter["stdev"])
                ccd_point_parameters.append((parameter["name"], value))

            ccd_point_parameters = sorted(ccd_point_parameters, key=lambda x: x[0])
            simulation_name = "ccd_" + "_".join([f"{x[1]:.3f}" for x in ccd_point_parameters])
            ccd_point_parameters.append(("simulation_name", simulation_name))

            new_config_section = copy.deepcopy(config["base_config"])
            config_addition = {x[0]: x[1] for x in ccd_point_parameters}
            new_config_section["random_seed"] = np.random.randint(0, 10000000)
            new_config_section.update(config_addition)

            run_configurations.append({
                id_counter: new_config_section
            })
            id_counter += 1

    output_path = f"{base_output_path}/ccd.yaml"
    with open(output_path, 'w') as output_file:
        yaml.dump(output_config, output_file)


def main(base_output_path: str, config_filename: str):
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)

    create_central_composite_design_configs(base_output_path, config)


if __name__ == "__main__":
    # Usage: python create_central_composite_design_configs.py <output path> <config filename>
    main(sys.argv[1], sys.argv[2])
