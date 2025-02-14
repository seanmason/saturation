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
    nstop: int,
    base_config: Dict[str, Any],
    overrides: Dict[str, Any] = None,
) -> List[Dict]:
    initial_rim_calculation_method = {
        "name": "circumference"
    }
    stop_condition = {
        "name": "nstat",
        "nstat": nstop
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
    n_workers = 28

    rstat = 3.0
    ratio = 3.0
    base_config = {
        "rstat": rstat,
        "rmin": rstat / ratio,
        # "rmax": 3000,
        # "study_region_padding": 1500,
        # "study_region_size": 12000,
        "rmax": 500,
        "study_region_padding": 125,
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
    nstop = 5000000

    run_configurations = dict()
    np.random.seed(123)

    # Add configs for steep vs shallow slope
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-1.0, -5.0],
        rim_erasure_methods=[{
                                 "name": "exponent_radius_ratio",
                                 "ratio": ratio,
                                 "exponent": 1.0
                             }],
        rmults=[1.0],
        mrps=[0.5],
        nstop=nstop,
        base_config=base_config,
        overrides={
            "rmin": base_config["rstat"] / ratio,
            "calculate_areal_density": True,
            "calculate_nearest_neighbor_stats": True,
        }
    )
    run_configurations = add_configs(configs=run_configurations, configs_to_add=configs_to_add)

    # Add configs for infinite ratio (no erasure threshold) by a range of slopes
    n_sims = 12
    min_slope = -5.0
    max_slope = -1.0
    step_size = (max_slope - min_slope) / n_sims
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[min_slope + x * step_size for x in range(n_sims + 1)],
        rim_erasure_methods=[{
                                 "name": "exponent_radius_ratio",
                                 "ratio": 1000000.0,
                                 "exponent": 1.0
                             }],
        rmults=[1.0],
        mrps=[0.5],
        nstop=nstop,
        base_config=base_config,
        overrides={"rmin": base_config["rstat"] / ratio}
    )
    run_configurations = add_configs(configs=run_configurations, configs_to_add=configs_to_add)

    # Add configs for exponents and slopes, restricting rmin, smaller area
    min_exponent = 0.1
    max_exponent = 1.0
    n_exponents = 10
    exponents = [
        round(min_exponent + x * (max_exponent - min_exponent) / (n_exponents - 1), 3)
        for x in range(n_exponents)
    ]
    n_sims = 21
    min_slope = -5.0
    max_slope = -1.0
    step_size = (max_slope - min_slope) / (n_sims - 1)
    for exponent in exponents:
        rim_erasure_methods = [{
            "name": "exponent_radius_ratio",
            "exponent": exponent,
            "ratio": ratio
        }, ]
        configs_to_add = create_configs_for_product_of_parameters(
            slopes=[round(min_slope + x * step_size, 3) for x in range(n_sims)],
            rim_erasure_methods=rim_erasure_methods,
            rmults=[1.0],
            mrps=[0.5],
            nstop=nstop,
            base_config=base_config,
            overrides={
                "rmin": 0.22,
            }
        )
        run_configurations = add_configs(configs=run_configurations, configs_to_add=configs_to_add)

    # Add configs for simulations to "test" the developed kappa model.
    configs_to_add = create_configs_for_product_of_parameters(
        slopes=[-3.3, -4.3, -4.85, -5.25, -5.5],
        rim_erasure_methods=[
            {
                "name": "exponent_radius_ratio",
                "exponent": x,
                "ratio": ratio
            }
            for x in [0.15, 0.35, 0.65]
        ],
        rmults=[1.0],
        mrps=[0.5],
        nstop=nstop,
        base_config=base_config,
        overrides={
            "rmin": 0.22,
        }
    )
    run_configurations = add_configs(configs=run_configurations, configs_to_add=configs_to_add)

    final_config = {
        "n_workers": min(n_workers, len(run_configurations)),
        "run_configurations": run_configurations,

    }
    print(yaml.dump(final_config))


if __name__ == "__main__":
    main()



