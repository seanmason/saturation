import numpy as np

from saturation.areal_density import ArealDensityCalculator
from saturation.crater_record import CraterRecord
from saturation.distributions import ParetoProbabilityDistribution
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

    distribution = ParetoProbabilityDistribution(alpha=1.5, x_min=5, x_max=250)
    crater_generator = get_craters(distribution, study_region_size + study_region_padding)
    record = CraterRecord(
        r_stat=r_stat,
        erat=3,
        mrp=0.5,
        rmult=1.5,
        study_region_size=study_region_size,
        study_region_padding=study_region_padding,
        cell_size=50
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
    assert removed_counter == 19
    assert record.nobs == 49
    assert areal_density_calculator.areal_density == 0.23791
