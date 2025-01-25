import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads


DTYPE = np.float64
RANDOM_CHUNK_SIZE = 100000


@njit(fastmath=True)
def inv_trunc_pareto(u, alpha, r_min, r_max):
    cdf_max = 1.0 - (r_min / r_max) ** alpha
    return r_min / (1.0 - u * cdf_max) ** (1.0 / alpha)


@njit(fastmath=True)
def overlap_arc_length(r_subj, r_new, d):
    if d >= r_subj + r_new:
        return 0.0

    if r_subj < r_new and d <= r_new - r_subj:
        return 2.0 * np.pi * r_subj

    if r_subj > r_new and d <= r_subj - r_new:
        return 0.0

    x = (d**2 + r_subj**2 - r_new**2) / (2.0 * d * r_subj)
    if x > 1.0:
        x = 1.0

    if x < -1.0:
        x = -1.0

    return 2.0 * np.arccos(x) * r_subj


@njit(fastmath=True)
def run_single_sim(r_subj, alpha, tau, r_dist_min, r_dist_max, surface_size):
    rim_original = 2.0 * np.pi * r_subj
    rim_state = rim_original
    r_cut = (r_subj ** tau) / 2.0

    cdf_max = 1.0 - (r_dist_min / r_dist_max) ** alpha
    cdf_cut = 1.0 - (r_dist_min / r_cut) ** alpha
    frac_in_range = 1 - cdf_cut / cdf_max

    bound_radius = r_subj + r_cut
    area_bounding = np.pi * bound_radius * bound_radius
    area_total = surface_size * surface_size
    frac_in_center = area_bounding / area_total

    scale_factor = 1.0 / (frac_in_range * frac_in_center)

    count = 0.0
    index_counter = 0
    r_news = inv_trunc_pareto(np.random.random(RANDOM_CHUNK_SIZE).astype(DTYPE), alpha, r_cut, r_dist_max)
    distances = bound_radius * np.sqrt(np.random.random(RANDOM_CHUNK_SIZE).astype(DTYPE))
    while rim_state > 0.5 * rim_original:
        r_new = r_news[index_counter]
        distance = distances[index_counter]

        arc_len = overlap_arc_length(r_subj, r_new, distance)
        if arc_len > 0.0:
            frac = arc_len / (2.0 * np.pi * r_subj)
            rim_state *= (1.0 - frac)

        count += scale_factor

        index_counter += 1
        if index_counter == RANDOM_CHUNK_SIZE:
            r_news = inv_trunc_pareto(np.random.random(RANDOM_CHUNK_SIZE).astype(DTYPE), alpha, r_cut, r_dist_max)
            distances = bound_radius * np.sqrt(np.random.random(RANDOM_CHUNK_SIZE).astype(DTYPE))
            index_counter = 0

    return int(np.ceil(count))


@njit(parallel=True, fastmath=True)
def run_simulations(n_r_values, n_alpha_values, n_tau_values, n_repeats_per_scenario,
                    r_min, r_max, r_dist_min, r_dist_max, alpha_min, alpha_max, tau_min, tau_max,
                    surface_size, seed):
    np.random.seed(seed)
    r_values = np.exp(np.linspace(np.log(r_min), np.log(r_max), num=n_r_values)).astype(DTYPE)
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
                        r_values[i],
                        alpha_values[j],
                        tau_values[k],
                        r_dist_min,
                        r_dist_max,
                        surface_size
                    )
                    params[idx, 0] = r_values[i]
                    params[idx, 1] = alpha_values[j]
                    params[idx, 2] = tau_values[k]
                    kappa[idx] = out

    return params, kappa


def main():
    n_tau_values = 20
    r_min = 5.0
    r_max = 1000.0
    r_dist_max = r_max * 4
    tau_min = 0.1
    tau_max = 1.0

    params, kappa = run_simulations(
        n_r_values=20,
        n_alpha_values=20,
        n_tau_values=n_tau_values,
        n_repeats_per_scenario=1000,
        r_min=r_min,
        r_max=r_max,
        r_dist_min=r_min ** (tau_min + (tau_max - tau_min) / n_tau_values) / 2.0,
        r_dist_max=r_dist_max,
        alpha_min=0.1,
        alpha_max=5.0,
        tau_min=tau_min,
        tau_max=tau_max,
        surface_size=r_max + r_dist_max,
        seed=123
    )

    df = pd.DataFrame(
        np.column_stack((params, kappa)),
        columns=["r", "alpha", "tau", "kappa"]
    )
    df.to_parquet("kappa_simulation_results.parquet")


if __name__ == "__main__":
    # alpha = 5.0
    # for x in range(100):
    #     r = 3.0 + x / 2
    #
    #     res = 0
    #     for _ in range(100):
    #         res += run_single_sim(
    #             r,
    #             alpha,
    #             1.0,
    #             1.0,
    #             100.0,
    #             400.0
    #         )
    #     expected = r**(alpha - 2) * (alpha - 2) / alpha
    #     print(
    #         np.log(r),
    #         ", ",
    #         np.log(res / 100),
    #         ", ",
    #         np.log(res / 100) / np.log(r),
    #         ", ",
    #         expected * 1000 / (res / 100)
    #     )
    set_num_threads(28)
    main()
