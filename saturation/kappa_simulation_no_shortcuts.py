import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads


DTYPE = np.float64
RANDOM_CHUNK_SIZE = 100000


@njit(fastmath=True)
def inv_trunc_pareto(
    u: float,
    alpha: float,
    r_dist_min: float,
    r_dist_max: float
):
    cdf_max = 1.0 - (r_dist_min / r_dist_max) ** alpha
    return r_dist_min / (1.0 - u * cdf_max) ** (1.0 / alpha)

@njit(fastmath=True)
def overlap_arc_length(
    r_subject: float,
    r_new: float,
    distance: float
):
    if distance >= r_subject + r_new:
        return 0.0

    if r_subject < r_new and distance <= r_new - r_subject:
        return 2.0 * np.pi * r_subject

    if r_subject > r_new and distance <= r_subject - r_new:
        return 0.0

    x = (distance**2 + r_subject**2 - r_new**2) / (2.0 * distance * r_subject)
    if x > 1.0:
        x = 1.0

    if x < -1.0:
        x = -1.0

    return 2.0 * np.arccos(x) * r_subject


@njit(fastmath=True)
def generate_random_chunks(
    alpha: float,
    r_dist_min: float,
    r_dist_max: float,
    surface_size: float
):
    r_news = inv_trunc_pareto(
        u=np.random.random(RANDOM_CHUNK_SIZE).astype(DTYPE),
        alpha=alpha,
        r_dist_min=r_dist_min,
        r_dist_max=r_dist_max
    )
    distances = np.sqrt(
        (surface_size / 2 * np.sqrt(np.random.random(RANDOM_CHUNK_SIZE).astype(DTYPE))) ** 2
        + (surface_size / 2 * np.sqrt(np.random.random(RANDOM_CHUNK_SIZE).astype(DTYPE))) ** 2
    )

    return r_news, distances

@njit(fastmath=True)
def run_single_sim(
    r_subject: float,
    alpha: float,
    tau: float,
    r_dist_min: float,
    r_dist_max: float,
    surface_size: float
):
    rim_original = 2.0 * np.pi * r_subject
    rim_state = rim_original

    count = 0.0
    index_counter = 0
    r_news, distances = generate_random_chunks(
        alpha=alpha,
        r_dist_min=r_dist_min,
        r_dist_max=r_dist_max,
        surface_size=surface_size
    )
    while rim_state > 0.5 * rim_original:
        count += 1

        r_new = r_news[index_counter]
        distance = distances[index_counter]

        if distance <= r_subject + r_new and r_new >= r_subject**tau / 2.0:
            arc_len = overlap_arc_length(r_subject=r_subject, r_new=r_new, distance=distance)
            if arc_len > 0.0:
                frac = arc_len / (2.0 * np.pi * r_subject)
                rim_state *= (1.0 - frac)


        index_counter += 1
        if index_counter == RANDOM_CHUNK_SIZE:
            r_news, distances = generate_random_chunks(
                alpha=alpha,
                r_dist_min=r_dist_min,
                r_dist_max=r_dist_max,
                surface_size=surface_size
            )
            index_counter = 0

    return int(np.ceil(count))


@njit(parallel=True, fastmath=True)
def run_simulations(
    n_r_values: int,
    n_alpha_values: int,
    n_tau_values: int,
    n_repeats_per_scenario: int,
    r_min: float,
    r_max: float,
    r_dist_min: float,
    r_dist_max: float,
    alpha_min: float,
    alpha_max: float,
    tau_min: float,
    tau_max: float,
    surface_size: float,
    seed: int
):
    np.random.seed(seed)
    r_values = np.linspace(r_min, r_max, n_r_values).astype(DTYPE)
    alpha_values = np.linspace(alpha_min, alpha_max, n_alpha_values).astype(DTYPE)
    tau_values = np.linspace(tau_min, tau_max, n_tau_values + 1).astype(DTYPE)[1:]

    n_total = n_r_values * n_alpha_values * n_tau_values * n_repeats_per_scenario
    params = np.zeros((n_total, 3), dtype=DTYPE)
    kappa = np.zeros(n_total, dtype=np.int64)

    for i in prange(n_r_values):
        for j in prange(n_alpha_values):
            for k in prange(n_tau_values):
                for m in prange(n_repeats_per_scenario):
                    idx = (
                        i * (n_alpha_values * n_tau_values * n_repeats_per_scenario)
                        + j * (n_tau_values * n_repeats_per_scenario)
                        + k * n_repeats_per_scenario
                        + m
                    )
                    out = run_single_sim(
                        r_subject=r_values[i],
                        alpha=alpha_values[j],
                        tau=tau_values[k],
                        r_dist_min=r_dist_min,
                        r_dist_max=r_dist_max,
                        surface_size=surface_size
                    )
                    params[idx, 0] = r_values[i]
                    params[idx, 1] = alpha_values[j]
                    params[idx, 2] = tau_values[k]
                    kappa[idx] = out

    return params, kappa


def main():
    n_tau_values = 2
    r_min = 10.0
    r_max = 50.0
    r_dist_max = r_max * 4
    tau_min = 0.0
    tau_max = 1.0

    params, kappa = run_simulations(
        n_r_values = 5,
        n_alpha_values = 5,
        n_tau_values = n_tau_values,
        n_repeats_per_scenario = 5,
        r_min = r_min,
        r_max = r_max,
        r_dist_min = r_min**((tau_max - tau_min) / n_tau_values),
        r_dist_max = r_dist_max,
        alpha_min = 3.0,
        alpha_max = 4.0,
        tau_min = tau_min,
        tau_max = tau_max,
        surface_size = r_max + r_dist_max,
        seed = 123
    )

    df = pd.DataFrame(
        np.column_stack((params, kappa)),
        columns=["r", "alpha", "tau", "kappa"]
    )
    df.to_parquet("kappa_simulation_results_no_shortcuts.parquet")


if __name__ == "__main__":
    set_num_threads(28)
    main()
