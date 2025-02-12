import numpy as np

from saturation.areal_density import ArealDensityCalculator
from saturation.crater_record import CraterRecord
from saturation.distributions import ParetoProbabilityDistribution
from saturation.initial_rim_state_calculators import CircumferenceInitialRimStateCalculator
from saturation.rim_erasure_calculators import get_rim_erasure_calculator
from saturation.simulation import get_craters


def test_crater_record_integration():
    """
    Integration/regression test for the crater record.
    """
    # Arrange
    ntot = 5000
    study_region_size = 1000
    study_region_padding = 0
    r_stat = 15
    ratio = 5.0
    rmult=1.5
    exponent = 0.5

    rim_erasure_calculator = get_rim_erasure_calculator(
        config={
            "name": "exponent_radius_ratio",
            "ratio": ratio,
            "exponent": exponent
        },
        rmult=rmult,
        r_stat=r_stat,
    )
    min_radius_threshold = rim_erasure_calculator.get_min_radius_threshold()

    distribution = ParetoProbabilityDistribution(alpha=1.5, x_min=min_radius_threshold / 2, x_max=250)
    crater_generator = get_craters(
        size_distribution=distribution,
        region_size=study_region_size + study_region_padding,
        min_radius_threshold=min_radius_threshold,
        random_seed=123
    )
    record = CraterRecord(
        r_stat=r_stat,
        rim_erasure_calculator=rim_erasure_calculator,
        initial_rim_state_calculator=CircumferenceInitialRimStateCalculator(),
        mrp=0.5,
        rmult=rmult,
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        cell_size=50,
        calculate_nearest_neighbor_stats=True
    )
    areal_density_calculator = ArealDensityCalculator(
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        r_stat=r_stat
    )

    # Act
    counter = 0
    removed_counter = 0
    for crater in crater_generator:
        removed_craters = record.add(crater)

        areal_density_calculator.add_crater(crater)
        if removed_craters:
            removed_counter += len(removed_craters)
            areal_density_calculator.remove_craters(removed_craters)

        counter += 1
        if record.ntot == ntot:
            break

    # Assert
    print(f"{counter}, {removed_counter}, {record.nobs}, {record.ntot}, {areal_density_calculator.areal_density}, {crater.id}")
    assert record.ntot == ntot
    assert record.nobs == 4933
    assert removed_counter == 67
    assert areal_density_calculator.areal_density == 0.289637
    assert crater.id == 1295580
