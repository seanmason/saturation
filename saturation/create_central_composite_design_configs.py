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


def create_central_composite_design_configs(config: Dict):
    np.random.seed(config["random_seed"])

    run_configurations = []
    output_config = {
        "n_workers": config["n_workers"],
        "output_path": config["run_output_path"],
        "write_statistics_cadence": config["write_statistics_cadence"],
        "write_state_cadence": config["write_state_cadence"],
        "write_image_cadence": config["write_image_cadence"],
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
    ccd_points = itertools.product(*ranges)
    for ccd_point in ccd_points:
        for _ in range(config["n_simulations_per_location"]):
            ccd_point_parameters = []
            for parameter in ccd_point:
                value = get_random_normal(parameter["value"], parameter["stdev"])
                ccd_point_parameters.append((parameter["name"], value))

            ccd_point_parameters = sorted(ccd_point_parameters, key=lambda x: x[0])
            new_config_section = copy.deepcopy(config["base_config"])
            config_addition = {x[0]: x[1] for x in ccd_point_parameters}
            new_config_section.update(config_addition)
            name = "central_composite_design_point_" + "_".join([f"{x[1]:.3f}" for x in ccd_point_parameters])

            run_configurations.append({
                name: new_config_section
            })

    output_path = f"{config['config_output_path']}/central_composite_design_points.yaml"
    with open(output_path, 'w') as output_file:
        yaml.dump(output_config, output_file)


def main(config_filename: str):
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)

    create_central_composite_design_configs(config)


if __name__ == "__main__":
    # Usage: python create_central_composite_design_configs.py <config filename>
    main(sys.argv[1])
