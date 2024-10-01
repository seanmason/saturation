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
    np.random.seed(123)

    ntot = 500
    study_region_size = 1000
    study_region_padding = 125
    r_stat = 15

    rim_erasure_calculator = get_rim_erasure_calculator({
        "name": "radius_ratio",
        "ratio": 5.0
    }, 1.5)

    distribution = ParetoProbabilityDistribution(alpha=1.5, x_min=5, x_max=250)
    crater_generator = get_craters(distribution, study_region_size + study_region_padding)
    record = CraterRecord(
        r_stat=r_stat,
        rim_erasure_calculator=rim_erasure_calculator,
        initial_rim_state_calculator=CircumferenceInitialRimStateCalculator(),
        mrp=0.5,
        rmult=1.5,
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        cell_size=50,
        calculate_nearest_neighbor_stats=True
    )
    areal_density_calculator = ArealDensityCalculator((study_region_size, study_region_size),
                                                      (study_region_padding, study_region_padding),
                                                      r_stat)

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
        if counter == ntot:
            break

    # Assert
    print(f"{removed_counter}, {record.nobs}, {areal_density_calculator.areal_density}")
    assert removed_counter == 21
    assert record.nobs == 48
    assert areal_density_calculator.areal_density == 0.237018
