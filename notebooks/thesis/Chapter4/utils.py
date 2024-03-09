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
from pyspark.sql import SparkSession, DataFrame, Window


def read_config(path: Path) -> Dict:
    with path.open("r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def read_configs(base_path: str, spark_session: SparkSession) -> pyspark.RDD:
    completed_filenames = list(Path(base_path).glob("*/completed.txt"))
    configs = map(lambda x: x.parent / "config.yaml", completed_filenames)
    configs = map(read_config, configs)
    return spark_session.sparkContext.parallelize(configs)


def create_configs_df(configs: pyspark.RDD) -> DataFrame:
    config_columns = [
        "simulation_id",
        "slope",
        "r_stat_multiplier",
        "effective_radius_multiplier",
        "min_rim_percentage"
    ]
    return configs.map(lambda x: {k: v for k, v in x.items() if k in config_columns}).toDF().cache()


def join_configs(data: DataFrame, configs: DataFrame, spark: SparkSession) -> DataFrame:
    data.createOrReplaceTempView("data")
    configs.createOrReplaceTempView("configs")
    
    # Join data and config
    query = f"""
    SELECT
        configs.slope,
        configs.min_rim_percentage,
        configs.effective_radius_multiplier,
        configs.r_stat_multiplier,
        data.*
    FROM
        data
        INNER JOIN configs 
            ON data.simulation_id = configs.simulation_id
    """
    return spark.sql(query)


def get_state_at_time(stats_df: DataFrame,
                      craters_df: DataFrame,
                      removals_df: DataFrame,
                      simulation_id: int,
                      target_n_craters_added_in_study_region: int,
                      study_region_size: int,
                      study_region_padding: int,
                      spark: SparkSession) -> pd.DataFrame:
    max_crater_id = stats_df.where(
        (F.col("simulation_id") == F.lit(simulation_id))
        & (F.col("n_craters_added_in_study_region") <= F.lit(target_n_craters_added_in_study_region))
    ).select(F.max("crater_id")).collect()[0][0]
    
    stats_df.createOrReplaceTempView("stats")
    craters_df.createOrReplaceTempView("craters")
    removals_df.createOrReplaceTempView("removals")

    query = f"""
    SELECT
        c.x,
        c.y,
        c.radius
    FROM
        craters c
        LEFT JOIN removals r ON
            r.simulation_id = c.simulation_id
            AND r.removed_crater_id = c.id
    WHERE
        c.simulation_id == {simulation_id}
        AND c.id <= {max_crater_id}
        AND (r.removed_by_crater_id IS NULL OR r.removed_by_crater_id > {max_crater_id})
        AND c.x >= {study_region_padding} AND c.x < {study_region_padding + study_region_size}
        AND c.y >= {study_region_padding} AND c.y < {study_region_padding + study_region_size}
    """
    return spark.sql(query).toPandas()


def estimate_cumulative_slope(diameters: List[float],
                              min_diameter: float,
                              max_diameter: float,
                              min_search_slope: float = 0.0,
                              max_search_slope: float = 10.0) -> float:
    N_GUESSES = 100000

    # Filter craters to only those between min and max
    diameters = np.array([x for x in diameters if min_diameter <= x <= max_diameter])

    summation = np.sum(np.log(diameters / min_diameter))
    n_craters = diameters.shape[0]

    guesses = min_search_slope + np.array([x * (max_search_slope - min_search_slope) / N_GUESSES for x in range(1, N_GUESSES + 1)])

    min_max_ratio = min_diameter / max_diameter
    guesses = n_craters / guesses + n_craters * min_max_ratio**guesses * np.log(min_max_ratio) / (1 - min_max_ratio**guesses) - summation
    
    min_index = np.argmin(np.abs(guesses))
    alpha = min_search_slope + min_index * (max_search_slope - min_search_slope) / N_GUESSES
    cumulative_slope = -alpha
    
    sigma = min_max_ratio**alpha * np.log(min_max_ratio)**2 / (1 - min_max_ratio**alpha)**2
    sigma = np.sqrt(1 / (1 / alpha**2 - sigma) / n_craters)
    
    return cumulative_slope, sigma


def estimate_intercept(radii: pd.Series, slope: float) -> float:
    def create_sfd_loss(radii: pd.Series, slope: float) -> Callable[float, float]:
        radii = radii.sort_values()
        
        def loss_func(intercept: float) -> float:
            expected = intercept * radii ** -slope
            actual = np.flip(range(radii.shape[0]))
            loss = ((actual - expected)**2).sum()
            return loss
    
        return loss_func

    return minimize_scalar(create_sfd_loss(radii, -slope), tol=1e-10).x


def calculate_areal_density(craters: pd.DataFrame,
                            study_region_size: float,
                            study_region_padding: float,
                            r_stat: float) -> float:
    from saturation.areal_density import ArealDensityCalculator
    from saturation.datatypes import Crater

    ad_calculator = ArealDensityCalculator(
        (study_region_size, study_region_size),
        (study_region_padding, study_region_padding),
        r_stat)
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
    

def plot_sfds(data: pd.DataFrame, slope: float, intercept: float = 1):
    radii = data.radius.sort_values()
    
    # Track min and max radii
    min_radius = radii.min()
    max_radius = radii.max()

    plt.plot(radii, range(len(radii) + 1, 1, -1), label="Observed")
    plt.xlabel("Crater Radius")
    plt.ylabel("N(>=R)")

    expected = intercept * radii ** -slope
    plt.plot(radii, expected, label="Estimated", ls="--")

    plt.subplots_adjust(right=0.7)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def plot_circle(center: Tuple[float, float],
                radius: float,
                axes_subplot,
                fill: bool = False,
                color: str = 'black',
                lw: float = 1,
                antialiased: bool = True):
    """
    Plots the specified circle on the supplied subplot.
    """
    axes_subplot.add_patch(matplotlib.patches.Circle(center,
                                                     radius=radius,
                                                     color=color,
                                                     fill=fill,
                                                     lw=lw,
                                                     antialiased=antialiased))

def plot_terrain(data: pd.DataFrame):
    x_range = data.x.max() - data.x.min()
    y_range = data.y.max() - data.y.min()
    xy_range = max(x_range, y_range)
    
    figsize = 5
    fig, ax = plt.subplots(figsize=(figsize * x_range / xy_range, figsize * y_range / xy_range))

    # x, y are in km, radius is in m, let's make them consistent
    data = data.copy()

    # ax.set_xlim([data.x.min() - 15, data.x.min() + xy_range + 15])
    # ax.set_ylim([data.y.min() - 15, data.y.min() + xy_range + 15])

    ax.set_xlim([data.x.min() - 15, data.x.max() + 15])
    ax.set_ylim([data.y.min() - 15, data.y.max() + 15])
    
    # Plot craters
    for idx, row in data.iterrows():
        plot_circle((row.x, row.y), row.radius, ax)

    plt.show()


def setup_dataset(data: DataFrame,
                  configs: DataFrame,
                  predictor_variables: List[str],
                  target: str,
                  spark: SparkSession):
    data.createOrReplaceTempView("data")
    configs.createOrReplaceTempView("configs")
    
    # Join data and config
    data_and_config_select_clause = ",\n".join(["data.simulation_id AS simulation_id"] + predictor_variables)
    data_and_config_select_clause += f",\n {target} as target"
    query = f"""
    SELECT
        {data_and_config_select_clause}
    FROM
        data
        INNER JOIN configs
            ON data.simulation_id = configs.simulation_id
    """
    data_and_configs = spark.sql(query)

    return data_and_configs


def setup_datasets_for_model(data: DataFrame,
                             configs: DataFrame,
                             test_simulations_fraction: float,
                             predictor_variables: List[str],
                             target: str,
                             train_sample_fraction: float,
                             n_test_observations: int,
                             spark: SparkSession,
                             cache_train: bool = True,
                             cache_test: bool = True):
    """
    Joins data and configs, splits data into train and test datasets.
    `train_sample_fraction` specifies the fraction of simulations to use in the test set.
    `n_test_observations` specifies the maximum number of observations to use in the test set.
    """
    simulation_ids = list(configs.select("simulation_id").toPandas().drop_duplicates().simulation_id.sort_values())
    test_simulation_ids = set(np.random.choice(simulation_ids, int(configs.count() * test_simulations_fraction), replace=False))
    
    train = setup_dataset(data, configs.where(~F.col("simulation_id").isin(test_simulation_ids)), predictor_variables, target, spark)
    train = train.sample(train_sample_fraction)

    if cache_train:
        train = train.cache()
        train.count()
    train.createOrReplaceTempView("train")

    test = setup_dataset(data, configs.where(F.col("simulation_id").isin(test_simulation_ids)), predictor_variables, target, spark)
    test = test.drop("simulation_id")
    test_count = test.count()

    # Need to sample a few more because Spark's sampling is not precise
    test = test.sample(min(n_test_observations / test_count * 1.5, 1.0)).limit(n_test_observations)
    
    if cache_test:
        test = test.cache()
        test.count()
    test.createOrReplaceTempView("test")
    
    return train, test