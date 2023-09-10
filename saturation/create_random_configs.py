import sys
from pathlib import Path
from typing import Dict
import copy

import numpy as np
import yaml


def get_uniform_random(minimum: float, maximum: float) -> float:
    return np.random.random() * (maximum - minimum) + minimum


def create_random_configs(base_output_path: str, config: Dict):
    np.random.seed(config["random_seed"])

    run_configurations = []
    output_config = {
        "n_workers": config["n_workers"],
        "write_statistics_cadence": config["write_statistics_cadence"],
        "write_state_cadence": config["write_state_cadence"],
        "write_craters_cadence": config["write_craters_cadence"],
        "write_image_points": config["write_image_points"],
        "write_crater_removals_cadence": config["write_crater_removals_cadence"],
        "write_image_cadence": config["write_image_cadence"],
        "run_configurations": run_configurations
    }

    parameters = config["parameters"]

    id_counter = 1
    for simulation_number in range(config["n_simulations"]):
        if simulation_number != 0 and simulation_number % config["n_simulations_per_config"] == 0:
            output_path = Path(f"{base_output_path}/config_{simulation_number:07d}.yaml")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as output_file:
                yaml.dump(output_config, output_file)

            run_configurations.clear()

        parameter_values = []
        for parameter_dict in parameters:
            name = list(parameter_dict.keys())[0]
            values = parameter_dict[name]

            value = get_uniform_random(values["min"], values["max"])
            parameter_values.append((name, value))

        parameter_values = sorted(parameter_values, key=lambda x: x[0])
        new_config_section = copy.deepcopy(config["base_config"])
        new_config_section["random_seed"] = np.random.randint(0, 10000000)
        config_addition = {x[0]: x[1] for x in parameter_values}
        new_config_section.update(config_addition)

        simulation_name = "_".join([f"{x[1]:.3f}" for x in parameter_values])
        new_config_section["simulation_name"] = simulation_name

        run_configurations.append({
            id_counter: new_config_section
        })
        id_counter += 1

    if run_configurations:
        output_path = f"{base_output_path}/config_{config['n_simulations']:07d}.yaml"
        with open(output_path, 'w') as output_file:
            yaml.dump(output_config, output_file)


def main(base_output_path: str, config_filename: str):
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)

    create_random_configs(base_output_path, config)


if __name__ == "__main__":
    # Usage: python create_random_configs.py <base output path> <config filename>
    main(sys.argv[1], sys.argv[2])
