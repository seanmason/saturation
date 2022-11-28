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


def create_corner_point_and_center_configs(config: Dict):
    np.random.seed(config["random_seed"])

    run_configurations = []
    output_config = {
        "n_workers": config["n_workers"],
        "output_path": config["run_output_path"],
        "write_state_cadence": config["write_state_cadence"],
        "write_images": config["write_images"],
        "write_all_craters": config["write_all_craters"],
        "write_removals": config["write_removals"],
        "run_configurations": run_configurations
    }

    parameters = config["parameters"]

    ranges = []
    for parameter_dict in parameters:
        name = list(parameter_dict.keys())[0]
        values = parameter_dict[name]
        ranges.append([
            {
                "name": name,
                "value": values["min"],
                "stdev": values["stdev"]
            },
            {
                "name": name,
                "value": values["max"],
                "stdev": values["stdev"]
            }
        ])

    # Add corner points to run configurations
    corner_points = itertools.product(*ranges)
    for corner_point in corner_points:
        for _ in range(config["n_simulations_per_corner_point_and_center"]):
            corner_point_parameters = []
            for parameter in corner_point:
                value = get_random_normal(parameter["value"], parameter["stdev"])
                corner_point_parameters.append((parameter["name"], value))

            corner_point_parameters = sorted(corner_point_parameters, key=lambda x: x[0])
            new_config_section = copy.deepcopy(config["base_config"])
            config_addition = {x[0]: x[1] for x in corner_point_parameters}
            new_config_section.update(config_addition)
            name = "corner_point_" + "_".join([f"{x[1]:.3f}" for x in corner_point_parameters])

            run_configurations.append({
                name: new_config_section
            })

    # Add the center point
    for _ in range(config["n_simulations_per_corner_point_and_center"]):
        center_parameters = []
        for parameter_dict in parameters:
            name = list(parameter_dict.keys())[0]
            values = parameter_dict[name]

            midpoint = (values["min"] + values["max"]) / 2
            value = get_random_normal(midpoint, values["stdev"])
            center_parameters.append((name, value))

        center_parameters = sorted(center_parameters, key=lambda x: x[0])
        new_config_section = copy.deepcopy(config["base_config"])
        config_addition = {x[0]: x[1] for x in center_parameters}
        new_config_section.update(config_addition)
        name = "center_point_" + "_".join([f"{x[1]:.3f}" for x in center_parameters])

        run_configurations.append({
            name: new_config_section
        })

    output_path = f"{config['config_output_path']}/corner_and_center_points_config.yaml"
    with open(output_path, 'w') as output_file:
        yaml.dump(output_config, output_file)


def create_interior_point_configs(config: Dict):
    np.random.seed(config["random_seed"])

    run_configurations = []
    output_config = {
        "n_workers": config["n_workers"],
        "output_path": config["run_output_path"],
        "write_state_cadence": config["write_state_cadence"],
        "write_images": config["write_images"],
        "write_removals": config["write_removals"],
        "write_all_craters": config["write_all_craters"],
        "run_configurations": run_configurations
    }

    parameters = config["parameters"]

    for simulation_number in range(config["n_simulations_for_interior"]):
        if simulation_number != 0 and simulation_number % config["n_simulations_per_interior_config"] == 0:
            output_path = f"{config['config_output_path']}/interior_points_config_{simulation_number}.yaml"
            with open(output_path, 'w') as output_file:
                yaml.dump(output_config, output_file)

            run_configurations.clear()

        interior_parameters = []
        for parameter_dict in parameters:
            name = list(parameter_dict.keys())[0]
            values = parameter_dict[name]

            value = get_uniform_random(values["min"], values["max"])
            interior_parameters.append((name, value))

        interior_parameters = sorted(interior_parameters, key=lambda x: x[0])
        new_config_section = copy.deepcopy(config["base_config"])
        config_addition = {x[0]: x[1] for x in interior_parameters}
        new_config_section.update(config_addition)
        name = "interior_point_" + "_".join([f"{x[1]:.3f}" for x in interior_parameters])

        run_configurations.append({
            name: new_config_section
        })

    if run_configurations:
        output_path = f"{config['config_output_path']}/interior_points_config_{config['n_simulations_for_interior']}.yaml"
        with open(output_path, 'w') as output_file:
            yaml.dump(output_config, output_file)


def main(config_filename: str):
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)

    create_corner_point_and_center_configs(config)
    create_interior_point_configs(config)


if __name__ == "__main__":
    # Usage: python create_surface_sampling_configs.py <config filename>
    main(sys.argv[1])
