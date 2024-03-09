import sys

import numpy as np
from pathlib import Path
import yaml
from typing import *

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
    return configs.map(lambda x: {k: v for k, v in x.items() if k in config_columns}).toDF()


def sample_by_simulation(data: DataFrame,
                         n_samples_per_simulation: int) -> DataFrame:
    """
    Samples n_samples_per_simulation samples from each simulation.
    """
    window = Window.partitionBy("simulation_id").orderBy("rnd_")

    filtered = (
        data
        .withColumn("rnd_", F.rand())
        .withColumn("rn_", F.row_number().over(window))
        .where(F.col("rn_") <= n_samples_per_simulation)
        .drop("rn_")
        .drop("rnd_")
    )

    return filtered


def add_post_saturation_percentiles(data: DataFrame, column: str):
    """
    Calculates the post-saturation percentile of a given column.
    """
    col_dtype = dict(data.dtypes)[column]

    # Select all points post-saturation - last 1/3 of each simulation
    window = Window.partitionBy("simulation_id").orderBy(F.col("n_craters_added_in_study_region"))
    with_row_number = data.withColumn("row_number", F.row_number().over(window))

    saturation_points = with_row_number.groupby("simulation_id").agg(F.max("row_number").alias("n_rows"))
    saturation_points = saturation_points.withColumn("saturation_point", (F.col("n_rows") / 3 * 2).cast("int"))

    with_saturation_points = with_row_number.join(saturation_points, on="simulation_id", how="inner")
    post_saturation = (
        with_saturation_points
        .filter(F.col("row_number") - F.col("saturation_point") >= 0)
        .drop("row_number")
        .drop("saturation_point")
        .drop("n_rows")
    )

    # Calculate post-saturation percentiles for each simulation
    # Create a "lookup table" of percentiles by simulation to join to
    percentile_lookup = (
        post_saturation
        .groupby("simulation_id")
        .agg(
            F.percentile_approx(column, F.array(*[F.lit(x / 100.0) for x in range(1, 100)]), 10000).alias("percentiles")
        )
        .select(
            "simulation_id",
            F.explode(
                F.arrays_zip(
                    F.array(*[F.lit(x / 100) for x in range(0, 100)]),
                    F.array_insert("percentiles", 1, F.lit(-2 ** 33).cast(col_dtype)),
                    F.array_insert("percentiles", 100, F.lit(2 ** 33).cast(col_dtype)),
                )
            ).alias("percentile_array")
        )
        .select(
            "simulation_id",
            F.col("percentile_array")["0"].alias(f"post_saturation_{column}_percentile"),
            F.col("percentile_array")["1"].alias("lower"),
            F.col("percentile_array")["2"].alias("upper"),
        )
    ).cache()

    # Join back to the full dataframe to add percentiles to each observation
    result = (
        data
        .join(percentile_lookup, on="simulation_id")
        .filter(data[column] >= percentile_lookup.lower)
        .filter(data[column] < percentile_lookup.upper)
        .drop("lower", "upper")
    )

    return result


def main(base_path: str, n_simulations: int, n_cores: int):
    n_samples_per_simulation = [
        100,
        250,
        # 500,
        # 5000,
        # 10000
    ]

    spark = (SparkSession.builder
             .master(f"local[{n_cores}]")
             .appName("Saturation")
             .config("spark.sql.shuffle.partitions", "1000")
             .config("spark.driver.memory", "64g")
             .config("spark.driver.maxResultSize", "8g")
             .getOrCreate())

    configs_df = create_configs_df(read_configs(base_path, spark))

    # Select at most n_simulations from all available simulations
    simulation_ids = list(configs_df.select("simulation_id").toPandas()["simulation_id"])
    n_simulations = min(n_simulations, len(simulation_ids))
    simulation_ids = set(np.random.choice(simulation_ids, replace=False, size=n_simulations))

    data = spark.read.parquet(f"{base_path}/*/statistics_*.parquet")
    data = data.filter(data.simulation_id.isin(simulation_ids))
    data = data.withColumn("information_remaining",
                           F.col("n_craters_in_study_region") / F.col("n_craters_added_in_study_region"))

    data = add_post_saturation_percentiles(data, "n_craters_in_study_region")
    data = add_post_saturation_percentiles(data, "areal_density")
    data = data.join(F.broadcast(configs_df), on="simulation_id")

    # Split into train and test sets
    train_simulation_ids = set(np.random.choice(list(simulation_ids), replace=False, size=int(n_simulations * 0.8)))
    test_simulation_ids = set([x for x in simulation_ids if x not in train_simulation_ids])
    train_df = data.filter(data.simulation_id.isin(train_simulation_ids))
    test_df = data.filter(data.simulation_id.isin(test_simulation_ids))

    for n in n_samples_per_simulation:
        print(f"Running for {n} samples per simulation...")

        sample = sample_by_simulation(train_df, n)
        sample.write.parquet(f"{base_path}/train_{n_simulations}_{n}.parquet")

        sample = sample_by_simulation(test_df, n)
        sample.write.parquet(f"{base_path}/test_{n_simulations}_{n}.parquet")


if __name__ == "__main__":
    """
    Usage: python summarize_simulations.py <base path of simulations> <n simulations to sample> <n cores to use>
    """
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
