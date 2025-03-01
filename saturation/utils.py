import glob

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import *

import matplotlib
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame


def read_config(path: Path) -> Dict:
    with path.open("r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def read_configs(base_path: str, spark_session: SparkSession, completed_only: bool=True) -> pyspark.RDD:
    if completed_only:
        completed_filenames = list(Path(base_path).glob("*/completed.txt"))
    else:
        completed_filenames = list(Path(base_path).glob("*/config.yaml"))
    configs = map(lambda x: x.parent / "config.yaml", completed_filenames)
    configs = map(read_config, configs)
    return spark_session.sparkContext.parallelize(configs)


def create_configs_df(configs: pyspark.RDD) -> DataFrame:
    config_columns = [
        "simulation_id",
        "slope",
        "rmult",
        "rstat",
        "mrp",
        "rmin",
        "study_region_size",
        "study_region_padding",
        "rim_erasure_method",
        "stop_condition"
    ]
    return configs.map(lambda x: {k: v for k, v in x.items() if k in config_columns}).toDF().cache()


def join_configs(data: DataFrame, configs: DataFrame, spark: SparkSession) -> DataFrame:
    data.createOrReplaceTempView("data")
    configs.createOrReplaceTempView("configs")
    
    # Join data and config
    query = f"""
    SELECT
        configs.slope,
        configs.mrp,
        configs.rmult,
        configs.rim_erasure_method,
        data.*
    FROM
        data
        INNER JOIN configs 
            ON data.simulation_id = configs.simulation_id
    """
    return spark.sql(query)


def get_configs(
    *,
    base_path: str,
    spark: SparkSession,
    completed_only: bool=False,
) -> Tuple[pd.DataFrame, DataFrame, Dict]:
    configs_df = create_configs_df(
        read_configs(
            base_path,
            spark,
            completed_only=completed_only
        )
    ).cache()
    configs_pdf = configs_df.toPandas()

    configs_pdf = configs_pdf[~configs_pdf.simulation_id.isna()].copy()
    configs_pdf["rim_erasure_exponent"] = configs_pdf.rim_erasure_method.apply(lambda x: x.get("exponent", -1))
    configs_pdf["rim_erasure_radius_ratio"] = configs_pdf.rim_erasure_method.apply(lambda x: x.get("ratio", -1))

    configs_dict = dict()
    for config_file in glob.glob(f"{base_path}/config/*config*.yaml"):
        configs_dict.update(read_config(Path(config_file))["run_configurations"])

    return configs_pdf, configs_df, configs_dict


def get_scientific_notation(
    number: float,
    sig_fig: int
):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")

    # remove leading "+" and strip leading zeros
    b = int(b)

    return f"{a} \\cdot 10^{b}"


def get_state_at_time(
    *,
    stats_df: DataFrame,
    craters_df: DataFrame,
    removals_df: DataFrame,
    target_nstat: int,
    study_region_size: int,
    study_region_padding: int,
    spark: SparkSession,
    max_radius: Optional[float] = None,
    result_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if max_radius:
        craters_df = craters_df.where(F.col("radius") <= max_radius)

    stats_df.createOrReplaceTempView("stats")
    craters_df.createOrReplaceTempView("craters")
    removals_df.createOrReplaceTempView("removals")

    query = f"""
    SELECT
        c.simulation_id,
        c.x,
        c.y,
        c.radius
    FROM
        stats s
        INNER JOIN craters c ON
            c.id = s.crater_id
            AND c.simulation_id = s.simulation_id
        LEFT JOIN removals r ON
            r.simulation_id = c.simulation_id
            AND r.removed_crater_id = c.id
            AND r.simulation_id = s.simulation_id
    WHERE
        c.id <= (
            SELECT
                MAX(crater_id)
            FROM
                stats
            WHERE
                nstat <= {target_nstat}
                AND stats.simulation_id = s.simulation_id
        )
        AND (r.removed_by_crater_id IS NULL OR r.removed_by_crater_id >
            (
                SELECT
                    MAX(crater_id)
                FROM
                    stats
                WHERE
                    nstat <= {target_nstat}
                    AND stats.simulation_id = s.simulation_id
            )
        )
        AND c.x >= {study_region_padding} AND c.x < {study_region_padding + study_region_size}
        AND c.y >= {study_region_padding} AND c.y < {study_region_padding + study_region_size}
    """
    result = spark.sql(query)
    if result_columns:
        result = result.select(*result_columns)
    return result.toPandas()


def estimate_cumulative_slope(
    *,
    radii: List[float],
    min_radius: float,
    max_radius: float,
    min_search_slope: float = -10.0,
    max_search_slope: float = 0.0
) -> Tuple[float, float]:
    """
    Returns a tuple of the estimated slope and sigma.
    """
    N_GUESSES = 100000

    # Filter craters to only those between min and max
    radii = np.array([x for x in radii if min_radius <= x <= max_radius])

    summation = np.sum(np.log(radii / min_radius))
    nobs = radii.shape[0]

    guesses = min_search_slope + np.array([x * (max_search_slope - min_search_slope) / N_GUESSES for x in range(1, N_GUESSES + 1)])

    min_max_ratio = min_radius / max_radius
    guesses = nobs / guesses + nobs * min_max_ratio**guesses * np.log(min_max_ratio) / (1 - min_max_ratio**guesses) - summation
    
    min_index = np.argmin(np.abs(guesses))
    alpha = min_search_slope + min_index * (max_search_slope - min_search_slope) / N_GUESSES
    cumulative_slope = -alpha
    
    sigma = min_max_ratio**alpha * np.log(min_max_ratio)**2 / (1 - min_max_ratio**alpha)**2
    sigma = np.sqrt(1 / (1 / alpha**2 - sigma) / nobs)
    
    return cumulative_slope, sigma


def get_states_at_ntats(
    *,
    simulation_id: int,
    configs_dict: Dict,
    base_path: str,
    spark: SparkSession,
    target_nstats: Optional[List[int]] = None,
    max_nstat: Optional[int] = None,
    n_states: int = 25,
    max_radius: Optional[float] = None
) -> Dict[int, pd.DataFrame]:
    """
    Returns a dict from values of nstat (a moment in time) to a dataframe of crater locations and radii at that time.
    """
    if not target_nstats:
        target_nstats = [
            int(10 ** 2 * 10 ** ((x + 1) / n_states * (np.log10(max_nstat) - 2)))
            for x in range(n_states)
        ]

    study_region_size = configs_dict[simulation_id]["study_region_size"]
    study_region_padding = configs_dict[simulation_id]["study_region_padding"]

    sim_path = f"{base_path}/{simulation_id}"

    stats_df = spark.read.parquet(f"{sim_path}/statistics_*.parquet")
    craters_df = spark.read.parquet( f"{sim_path}/craters_*.parquet")
    removals_df = spark.read.parquet(f"{sim_path}/crater_removals_*.parquet")

    return {
        x: get_state_at_time(
            stats_df=stats_df,
            craters_df=craters_df,
            removals_df=removals_df,
            target_nstat=x,
            study_region_size=study_region_size,
            study_region_padding=study_region_padding,
            spark=spark,
            max_radius=max_radius
        )
        for x in target_nstats}


def estimate_slopes_for_states(
    states: Dict[int, pd.DataFrame],
    rmin: float
) -> pd.DataFrame:
    estimates = []
    for nstat, state in states.items():
        state = states[nstat]
        alpha, sigma = estimate_cumulative_slope(
            radii=state.radius,
            min_radius=rmin,
            max_radius=state.radius.max(),
            min_search_slope=0.0,
            max_search_slope=10.0
        )
        estimates.append({"nstat": nstat, "alpha": alpha, "sigma": sigma})
    return pd.DataFrame(estimates)


def estimate_intercept(radii: pd.Series, slope: float) -> float:
    def create_sfd_loss(radii: pd.Series, slope: float) -> Callable[[float], float]:
        r = radii.sort_values()
        
        def loss_func(intercept: float) -> float:
            expected = intercept * r ** -slope
            actual = np.flip(range(r.shape[0]))
            loss = ((actual - expected)**2).sum()
            return loss
    
        return loss_func

    return minimize_scalar(create_sfd_loss(radii, -slope), tol=1e-10).x


def calculate_areal_density(
    craters: pd.DataFrame,
    study_region_size: int,
    study_region_padding: int,
    rstat: float
) -> float:
    from saturation.areal_density import ArealDensityCalculator
    from saturation.datatypes import Crater

    ad_calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        rstat=rstat
    )
    for idx, row in craters.iterrows():
        ad_calculator.add_crater(
            Crater(
                idx,
                row.x,
                row.y,
                row.radius
            )
        )

    return ad_calculator.areal_density
    

def plot_csfd(data: pd.DataFrame):
    font_size = 24

    radii = data.radius.sort_values()

    fig = plt.figure(figsize=(9, 6), dpi=400)
    ax = fig.add_subplot(111)

    ax.plot(radii, range(len(radii) + 1, 1, -1), label="Observed")
    ax.set_xlabel("$r$", fontsize=font_size)
    ax.set_ylabel("$N(\\geq r)$", fontsize=font_size)

    ax.set_xscale("log")
    ax.set_yscale("log")

    return fig


def plot_csfd_with_slope(
    data: pd.DataFrame,
    slope: float,
    intercept: float = 1
):
    fig = plot_csfd(data)
    ax = fig.axes[0]

    radii = data.radius.sort_values()
    expected = intercept * radii ** -slope
    ax.plot(radii, expected, label="Estimated", ls="--", c="green")

    ax.legend()

    return fig


def plot_csfds_for_multiple_nstat(
    states: dict[int, pd.DataFrame],
    slope_intercept_line_styles: List[Tuple[float, float, str]]
):
    """
    Plots CSFDs for multiple values of nstat
    """
    colors = [
        "blue",
        "red",
        "orange",
        "black",
        "green",
    ]

    fig = plt.figure(figsize=(6, 4), dpi=400)
    ax = fig.add_subplot(111)

    radii = None
    for idx, (nstat, data) in enumerate(states.items()):
        radii = data.radius.sort_values()
        nstat_string = nstat if nstat < 1e5 else get_scientific_notation(nstat, 2)
        ax.plot(radii, range(len(radii) + 1, 1, -1), label="$N_{\\text{stat}}" + f"={nstat_string}$", c=colors[idx % len(colors)])

    for slope, intercept, line_style, label in slope_intercept_line_styles:
        expected = intercept * radii ** slope
        ax.plot(
            radii[expected > 1],
            expected[expected > 1],
            ls=line_style,
            c="black",
            label=label
        )

    ax.set_xlabel("$r$", fontsize=14)
    ax.set_ylabel("$N(\\geq r)$", fontsize=14)

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')

    return fig


def plot_circle(
    center: Tuple[float, float],
    radius: float,
    axes_subplot,
    fill: bool = False,
    color: str = 'black',
    lw: float = 1,
    antialiased: bool = True
):
    """
    Plots the specified circle on the supplied subplot.
    """
    axes_subplot.add_patch(
        matplotlib.patches.Circle(
            center,
            radius=radius,
            color=color,
            fill=fill,
            lw=lw,
            antialiased=antialiased
        )
    )


def plot_terrain(data: pd.DataFrame):
    x_range = data.x.max() - data.x.min()
    y_range = data.y.max() - data.y.min()
    xy_range = max(x_range, y_range)
    
    figsize = 5
    fig, ax = plt.subplots(figsize=(figsize * x_range / xy_range, figsize * y_range / xy_range))

    # x, y are in km, radius is in m, let's make them consistent
    data = data.copy()

    ax.set_xlim([data.x.min() - 15, data.x.max() + 15])
    ax.set_ylim([data.y.min() - 15, data.y.max() + 15])
    
    # Plot craters
    for idx, row in data.iterrows():
        plot_circle((row.x, row.y), row.radius, ax)

    plt.show()


def get_lifespans_for_simulation(
    *,
    simulation_id: int,
    spark: SparkSession,
    craters: DataFrame=None,
    removals: DataFrame=None,
    configs_df: DataFrame=None,
) -> pd.DataFrame:
    configs_df.createOrReplaceTempView("config")
    craters.createOrReplaceTempView("craters")
    removals.createOrReplaceTempView("removals")

    query = f"""
    WITH lifespans AS
    (
        SELECT
            simulation_id,
            removed_crater_id AS id,
            removed_by_crater_id - removed_crater_id AS lifespan
        FROM
            removals
    ),
    craters AS
    (
        SELECT
            c.simulation_id,
            c.id,
            radius
        FROM
            craters c
            INNER JOIN config cfg ON
                c.simulation_id = cfg.simulation_id
        WHERE
            1=1
            AND c.x BETWEEN study_region_padding AND study_region_size + study_region_padding
            AND c.y BETWEEN study_region_padding AND study_region_size + study_region_padding
    )
    SELECT
        radius,
        lifespan
    FROM
        lifespans l
        INNER JOIN craters c ON
            c.id = l.id
            AND c.simulation_id = l.simulation_id
    WHERE
        l.simulation_id = {simulation_id}
    ORDER BY
        radius
    """
    return spark.sql(query).toPandas()


def plot_metric(
    data: pd.DataFrame, x_var: str, x_label: str, y_var: str, y_label: str, dotted_horizontal_lines: list[float] = None
):
    font_size = 16

    fig = plt.figure(figsize=(6, 4), dpi=400)
    ax = fig.add_subplot(111)

    simulation_ids = data.simulation_id.drop_duplicates()
    for idx, simulation_id in enumerate(simulation_ids):
        data_subset = data[data.simulation_id == simulation_id].sort_values("nstat")
        ax.plot(
            data_subset[x_var], data_subset[y_var], c=colors[idx % len(colors)], ls=line_styles[idx % len(line_styles)]
        )

    if dotted_horizontal_lines:
        for y_val in dotted_horizontal_lines:
            ax.axhline(y_val, color="r", linestyle="--")

    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

    return fig


def plot_metrics(
    *,
    df: pd.DataFrame,
    scenario_name: str,
    nstat_bound_saturation: int,
    show_plots: bool = False
):
    ad_line = df[df.nstat > nstat_bound_saturation].ad.mean()
    print(f"AD line: {ad_line}")
    fig = plot_metric(
        df, "nstat", "$N_{\\text{stat}}$", "ad", "$A_d$", dotted_horizontal_lines=[ad_line]
    )
    if show_plots:
        plt.show()
    fig.savefig(f"figures/{scenario_name}_nstat_ad.png", bbox_inches="tight")

    log_mnnd_line = df[df.nstat > nstat_bound_saturation].log_mnnd.mean()
    print(f"log_mnnd line: {log_mnnd_line}")
    fig = plot_metric(
        df, "nstat", "$N_{\\text{stat}}$", "log_mnnd", "$log_{10}(\\overline{NN}_d)$", dotted_horizontal_lines=[log_mnnd_line]
    )
    plt.show()
    fig.savefig(f"figures/{scenario_name}_nstat_mnnd.png", bbox_inches="tight")

    fig = plot_metric(
        df, "nstat", "$N_{\\text{stat}}$", "z", "$Z$", dotted_horizontal_lines=[-1.96, 1.96]
    )
    if show_plots:
        plt.show()
    fig.savefig(f"figures/{scenario_name}_nstat_z.png", bbox_inches="tight")

    fig = plot_metric(
        df, "nstat", "$N_{\\text{stat}}$", "za", "$Z_a$", dotted_horizontal_lines=[-1.96, 1.96]
    )
    if show_plots:
        plt.show()
    fig.savefig(f"figures/{scenario_name}_nstat_za.png", bbox_inches="tight")

    radius_mean_line = df[df.nstat > nstat_bound_saturation].radius_mean.mean()
    print(f"radius_mean line: {radius_mean_line}")
    fig = plot_metric(
        df, "nstat", "$N_{\\text{stat}}$", "radius_mean", "$\\overline{r}$", dotted_horizontal_lines=[radius_mean_line]
    )
    if show_plots:
        plt.show()
    fig.savefig(f"figures/{scenario_name}_nstat_radius_mean.png", bbox_inches="tight")

    radius_stdev_line = df[df.nstat > nstat_bound_saturation].radius_stdev.mean()
    print(f"radius_stdev line: {radius_stdev_line}")
    fig = plot_metric(
        df, "nstat", "$N_{\\text{stat}}$", "radius_stdev", "$\\sigma_r$", dotted_horizontal_lines=[radius_stdev_line]
    )
    if show_plots:
        plt.show()
    fig.savefig(f"figures/{scenario_name}_nstat_radius_stdev.png", bbox_inches="tight")


def plot_slope_estimates(estimates_df: pd.DataFrame):
    font_size = 16

    fig = plt.figure(figsize=(6, 4), dpi=400)
    ax = fig.add_subplot(111)

    ax.errorbar(
        estimates_df.nstat, estimates_df.alpha, estimates_df.sigma, ls="None", marker="+"
    )
    ax.set_xlabel("$N_{\\text{stat}}$", fontsize=font_size)
    ax.set_ylabel("$b$", fontsize=font_size)
    ax.set_xscale("log")

    return fig


def get_statistics_with_lifespans_for_simulations(
    *,
    simulation_ids: List[int],
    base_path: str,
    configs_df: DataFrame,
    spark: SparkSession,
    n_samples_per_sim: Optional[int]=None,
    max_radius: Optional[float]=None,
    max_nstat: Optional[int]=None,
    result_columns: Optional[List[str]]=None,
) -> pd.DataFrame:
    """
    Returns statistics and lifespans for each simulation id specified.
    Returns only craters with centers in the study region.
    Optionally sampled.
    """
    F.broadcast(configs_df).createOrReplaceTempView("configs")

    result = None
    for simulation_id in simulation_ids:
        craters = spark.read.parquet(f"{base_path}/{simulation_id}/craters_*.parquet")
        removals = spark.read.parquet(f"{base_path}/{simulation_id}/crater_removals_*.parquet")
        statistics = spark.read.parquet(f"{base_path}/{simulation_id}/statistics_*.parquet")

        statistics.createOrReplaceTempView("statistics")
        craters.createOrReplaceTempView("craters")
        removals.createOrReplaceTempView("removals")

        query = f"""
        SELECT
            c.x,
            c.y,
            c.radius,
            s.simulation_id,
            s.nstat,
            s.nobs,
            s.areal_density,
            configs.slope,
            configs.mrp,
            configs.rmult,
            configs.rim_erasure_method['name'] AS rim_erasure_method_name,
            COALESCE(configs.rim_erasure_method['ratio'], 0) AS rim_erasure_radius_ratio,
            COALESCE(configs.rim_erasure_method['exponent'], 0) AS rim_erasure_exponent,
            CASE
                WHEN r.removed_by_crater_id IS NULL THEN NULL
                ELSE r.removed_by_crater_id - c.id
            END AS lifespan
        FROM
            statistics s
            INNER JOIN craters c ON
                c.id = s.crater_id
            LEFT JOIN removals r ON
                r.removed_crater_id = c.id
            INNER JOIN configs ON
                configs.simulation_id = s.simulation_id
        """
        result_for_simulation = spark.sql(query)
        if max_radius:
            result_for_simulation = result_for_simulation.where(F.col("radius") <= max_radius)
        if max_nstat:
            result_for_simulation = result_for_simulation.where(F.col("nstat") <= max_nstat)
        if n_samples_per_sim:
            result_for_simulation = (
                result_for_simulation.orderBy(F.rand())
                .limit(n_samples_per_sim)
                .select(*result_columns)
            )

        if result is None:
            result = result_for_simulation.toPandas()
        else:
            result = pd.concat([result, result_for_simulation.toPandas()])

    return result