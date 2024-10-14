import numpy as np

import itertools
import copy
from typing import List, Dict, Any

import yaml


def create_configs_for_product_of_parameters(
    *,
    slopes: List[float],
    rmults: List[float],
    mrps: List[float],
    rim_erasure_methods: List[Dict[str, Any]],
    stop_ntot: int,
    base_config: Dict[str, Any],
    overrides: Dict[str, Any] = None,
) -> List[Dict]:
    initial_rim_calculation_method = {
        "name": "circumference"
    }
    stop_condition = {
        "name": "ntot",
        "ntot": stop_ntot
    }

    result = []

    for slope, rim_erasure_method, rmult, mrp in itertools.product(slopes, rim_erasure_methods, rmults, mrps):
        config = copy.deepcopy(base_config)
        config["slope"] = slope
        config["rmult"] = rmult
        config["mrp"] = mrp
        config["rim_erasure_method"] = rim_erasure_method
        config["initial_rim_calculation_method"] = initial_rim_calculation_method
        config["stop_condition"] = stop_condition
        config["random_seed"] = np.random.randint(1, 2**32-1)

        if overrides:
            for k, v in overrides.items():
                config[k] = v

        result.append(config)

    return result


def add_configs(
    *,
    configs_to_add: List[Dict],
    configs: Dict[int, Dict],
) -> Dict[int, Dict]:
    id_counter = len(configs) + 1

    result = copy.deepcopy(configs)
    for config in configs_to_add:
        result[id_counter] = copy.deepcopy(config)
        id_counter += 1

    return result


def main():
    n_workers = 22
    base_config = {
        "r_min": 0.5,
        "r_max": 500,
        "r_stat": 3,
        "study_region_padding": 250,
        "study_region_size": 2000,
        "spatial_hash_cell_size": 10,
        "calculate_areal_density": False,
        "calculate_nearest_neighbor_stats": False,
        "write_crater_removals_cadence": 50000,
        "write_craters_cadence": 50000,
        "write_image_cadence": 0,
        "write_image_points": [],
        "write_state_cadence": 0,
        "write_statistics_cadence": 50000,
    }

    run_configurations = dict()
    np.random.seed(123)

    # Add configs for steep vs shallow slope
    erat = 5.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-1.0, -4.0],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": erat}],
        rmults=[1.5],
        mrps=[0.5],
        stop_ntot=2500000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"] / erat}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # Add configs for high destruction
    erat = 11.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-2.5],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": erat}],
        rmults=[1.9],
        mrps=[0.75],
        stop_ntot=2500000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"] / erat}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # Add configs for low destruction
    erat = 3.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-2.5],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": erat}],
        rmults=[1.1],
        mrps=[0.25],
        stop_ntot=2500000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"] / erat}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # Add configs for an "infinite" erat
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-1.0, -4.0],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": 1000000.0}],
        rmults=[1.0],
        mrps=[0.5],
        stop_ntot=2500000,
        base_config=base_config,
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # Add config for a range of slopes, 15 simulations from slope -1.0 to -4.0
    # Lower erat
    n_sims = 15
    min_slope = -4.0
    max_slope = -1.0
    step_size = (max_slope - min_slope) / n_sims
    erat = 3.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[min_slope + x * step_size for x in range(n_sims + 1)],
        rim_erasure_methods=[
            {"name": "radius_ratio", "ratio": erat},
        ],
        rmults=[1.1, 1.9],
        mrps=[0.25, 0.75],
        stop_ntot=1000000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"] / erat}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # Add config for a range of slopes, 15 simulations from slope -1.0 to -4.0
    # Higher erat
    erat = 7.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[min_slope + x * step_size for x in range(n_sims + 1)],
        rim_erasure_methods=[
            {"name": "radius_ratio", "ratio": erat},
        ],
        rmults=[1.1, 1.9],
        mrps=[0.25, 0.75],
        stop_ntot=1000000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"] / erat}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # Add config for a long-running simulation with shallow slope
    erat = 5.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-1.0],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": erat}],
        rmults=[1.5],
        mrps=[0.5],
        stop_ntot=50000000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"] / erat}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # Add configs for alternate rim erasure methods
    exponent = 0.5
    rim_erasure_methods = [
        {"name": "exponent", "exponent": exponent},
    ]
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-3.5],
        rim_erasure_methods=rim_erasure_methods,
        rmults=[1.5],
        mrps=[0.5],
        stop_ntot=2500000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"]**exponent}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    exponent = 0.25
    rim_erasure_methods = [
        {"name": "exponent", "exponent": exponent},
    ]
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-3.5],
        rim_erasure_methods=rim_erasure_methods,
        rmults=[1.5],
        mrps=[0.5],
        stop_ntot=2500000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"]**exponent}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    rim_erasure_methods = [
        {"name": "log"},
    ]
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-3.5],
        rim_erasure_methods=rim_erasure_methods,
        rmults=[1.5],
        mrps=[0.5],
        stop_ntot=2500000,
        base_config=base_config,
        overrides={"r_min": float(np.log(base_config["r_stat"]))}
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )


    # Add configs for an extended "infinite" erat
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-1.0, -3.0, -3.5, -4.0],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": 1000000.0}],
        rmults=[1.5],
        mrps=[0.5],
        stop_ntot=10000000,
        base_config=base_config,
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-1.0, -3.0, -3.5, -4.0],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": 1000000.0}],
        rmults=[1.5],
        mrps=[0.5],
        stop_ntot=10000000,
        base_config=base_config,
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    final_config = {
        "n_workers": min(n_workers, len(run_configurations)),
        "run_configurations": run_configurations,

    }
    print(yaml.dump(final_config))


if __name__ == "__main__":
    main()



