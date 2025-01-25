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
    n_workers = 16
    base_config = {
        "r_min": 0.5,
        "r_max": 500,
        "r_stat": 3,
        "study_region_padding": 250,
        "study_region_size": 2000,
        "spatial_hash_cell_size": 5,
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
        overrides={
            "r_min": base_config["r_stat"] / erat,
            "calculate_areal_density": True,
            "calculate_nearest_neighbor_stats": True,
        }
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # Add configs steep vs shallow slope, high destruction
    erat = 11.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-1, -4],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": erat}],
        rmults=[1.9],
        mrps=[0.75],
        stop_ntot=2500000,
        base_config=base_config,
        overrides={
            "r_min": base_config["r_stat"] / erat,
            "calculate_areal_density": True,
            "calculate_nearest_neighbor_stats": True,
            "random_seed": 123
        }
    )
    run_configurations = add_configs(
        configs=run_configurations, configs_to_add=configs_to_add
    )

    # Add configs steep vs shallow slope, low destruction
    erat = 3.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-1, -4],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": erat}],
        rmults=[1.1],
        mrps=[0.25],
        stop_ntot=2500000,
        base_config=base_config,
        overrides={
            "r_min": base_config["r_stat"] / erat,
            "calculate_areal_density": True,
            "calculate_nearest_neighbor_stats": True,
            "random_seed": 123
        }
    )
    run_configurations = add_configs(
        configs=run_configurations, configs_to_add=configs_to_add
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

    # Add configs for infinite erat (no erasure threshold) by a range of slopes
    n_sims = 12
    min_slope = -5.0
    max_slope = -1.0
    step_size = (max_slope - min_slope) / n_sims
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[min_slope + x * step_size for x in range(n_sims + 1)],
        rim_erasure_methods=[{"name": "radius_ratio", "ratio": 1000000.0}],
        rmults=[1.5],
        mrps=[0.5],
        stop_ntot=1000000,
        base_config=base_config,
        overrides={"r_min": base_config["r_stat"] / erat}
    )
    run_configurations = add_configs(
        configs=run_configurations, configs_to_add=configs_to_add
    )

    # Add configs for exponents and slopes, restricting rmin, smaller area
    ratio = 2.0
    min_exponent = 0.1
    max_exponent = 1.0
    n_exponents = 10
    exponents = [round(min_exponent + x * (max_exponent - min_exponent) / (n_exponents - 1), 3) for x in
                 range(n_exponents)]
    n_sims = 21
    min_slope = -5.0
    max_slope = -1.0
    step_size = (max_slope - min_slope) / (n_sims - 1)
    for exponent in exponents:
        rim_erasure_methods = [{
            "name": "exponent",
            "exponent": exponent,
            "ratio": ratio
        }, ]
        configs_to_add = create_configs_for_product_of_parameters(
            slopes=[round(min_slope + x * step_size, 3) for x in range(n_sims)],
            rim_erasure_methods=rim_erasure_methods,
            rmults=[1.0],
            mrps=[0.5],
            stop_ntot=1000000,
            base_config=base_config,
            overrides={
                "r_min": 1.1 / ratio,
                "study_region_padding": 125,
                "study_region_size": 1000,
                "r_max": 250,
            }
        )
        run_configurations = add_configs(
            configs=run_configurations, configs_to_add=configs_to_add
        )

    # Add configs for simulations to "test" the developed kappa model.
    ratio = 2.0
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-3.3, -4.3],
        rim_erasure_methods=[{
            "name": "exponent",
            "exponent": 0.65,
            "ratio": ratio
        }, ],
        rmults=[1.0],
        mrps=[0.5],
        stop_ntot=1000000,
        base_config=base_config,
        overrides={
            "r_min": 1.1 / ratio,
            "study_region_padding": 125,
            "study_region_size": 1000,
            "r_max": 250,
        }
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-3.3, -4.3],
        rim_erasure_methods=[{
            "name": "exponent",
            "exponent": 0.35,
            "ratio": ratio
        }, ],
        rmults=[1.0],
        mrps=[0.5],
        stop_ntot=1000000,
        base_config=base_config,
        overrides={
            "r_min": 1.1 / ratio,
            "study_region_padding": 125,
            "study_region_size": 1000,
            "r_max": 250,
        }
    )
    run_configurations = add_configs(
        configs=run_configurations,
        configs_to_add=configs_to_add
    )

    # A last simulation that has a steeper slope
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-5.5, -5.25, -4.85],
        rim_erasure_methods=[{
            "name": "exponent",
            "exponent": 0.09,
            "ratio": ratio
        },
        {
            "name": "exponent",
            "exponent": 0.15,
            "ratio": ratio
        },],
        rmults=[1.0],
        mrps=[0.5],
        stop_ntot=1000000,
        base_config=base_config,
        overrides={
            "r_min": 1.1 / ratio,
            "study_region_padding": 125,
            "study_region_size": 1000,
            "r_max": 250,
        }
    )
    run_configurations = add_configs(
        configs=run_configurations, configs_to_add=configs_to_add
    )


    final_config = {
        "n_workers": min(n_workers, len(run_configurations)),
        "run_configurations": run_configurations,

    }
    print(yaml.dump(final_config))


if __name__ == "__main__":
    main()



