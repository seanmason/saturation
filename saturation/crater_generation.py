from typing import Generator, List, Tuple
import numpy as np
from numba import njit
from numba.typed import List as TypedList
from saturation.datatypes import Crater, CraterType
from saturation.distributions import ProbabilityDistribution


@njit
def _generate_craters_chunk(
    size_distribution: ProbabilityDistribution,
    region_size: float,
    min_radius_threshold: float,
    start_crater_id: int,
    rng: np.random.Generator,
) -> Tuple[TypedList, int]:
    CHUNK_SIZE = int(1e6)
    DTYPE = np.float32

    uniform_threshold = DTYPE(size_distribution.cdf(min_radius_threshold))

    region_size = DTYPE(region_size)
    crater_id = start_crater_id - 1

    craters = TypedList()

    while len(craters) < CHUNK_SIZE:
        crater_id += 1

        uniform = rng.random(dtype=DTYPE)
        if uniform > uniform_threshold:
            radius = size_distribution.pullback(uniform)
            x, y = rng.random(size=2, dtype=DTYPE) * region_size
            craters.append(Crater(
                crater_id,
                x,
                y,
                radius
            ))

    return craters, crater_id


@njit
def _group_craters(
    craters: TypedList,
    rstat: float,
) -> Tuple[TypedList, TypedList]:
    groups = TypedList()
    checks = TypedList()

    current_group = TypedList.empty_list(item_type=CraterType)
    current_group_geq_rstat = craters[0].radius >= rstat

    for crater in craters:
        geq_rstat = crater.radius >= rstat
        if geq_rstat == current_group_geq_rstat:
            current_group.append(crater)
        else:
            groups.append(current_group)
            checks.append(current_group_geq_rstat)

            current_group = TypedList.empty_list(item_type=CraterType)
            current_group.append(crater)
            current_group_geq_rstat = geq_rstat

    if current_group:
        groups.append(current_group)
        checks.append(current_group_geq_rstat)

    return groups, checks


def get_grouped_craters(
    size_distribution: ProbabilityDistribution,
    rstat: float,
    region_size: float,
    min_radius_threshold: float,
    random_seed: int
) -> Generator[Tuple[List[Crater], bool], None, None]:
    """
    Generates groups of craters from a single chunk. For each group, all consecutive
    craters share the same Boolean value of (crater.radius > rstat). Each yielded tuple
    contains a list of craters (as a Numba typed list) and the Boolean check value.
    """
    rng = np.random.default_rng(seed=random_seed)
    start_crater_id = 0

    while True:
        craters, start_crater_id = _generate_craters_chunk(
            size_distribution,
            region_size,
            min_radius_threshold,
            start_crater_id,
            rng
        )
        yield from zip(*_group_craters(craters, rstat))
