import itertools

def main():
    format = """- {id}:
    slope: {slope}
    erat: {erat}
    mrp: {mrp}
    rmult: 1.0
    max_crater_radius: 250
    min_crater_radius: 3
    random_seed: {random_seed}
    simulation_name: "{id}"
    spatial_hash_cell_size: 3
    stop_condition:
      ntot: 50000000
      name: ntot
    study_region_padding: 125
    study_region_size: 1000
"""

    result = ""

    slopes = [-2.5, -3.5, -4.0]
    erats = [1.0, 3.0]
    mrps = [0.25, 0.75]
    random_seeds = [123, 234]

    for idx, (slope, erat, mrp, random_seed) in enumerate(itertools.product(slopes, erats, mrps, random_seeds)):
        result += format.format(
            id=idx,
            slope=slope,
            mrp=mrp,
            erat=erat,
            random_seed=random_seed
        )

    result += format.format(
        id=1000,
        slope=-1.25,
        erat=1,
        mrp=0.5,
        random_seed=123
    )

    result += format.format(
        id=2000,
        slope=-5,
        erat=1,
        mrp=0.75,
        random_seed=123
    )

    print(result)


if __name__ == "__main__":
    main()



