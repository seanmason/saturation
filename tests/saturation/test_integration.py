from saturation.areal_density import ArealDensityCalculator
from saturation.crater_record import CraterRecord
from saturation.distributions import ParetoProbabilityDistribution
from saturation.initial_rim_state_calculators import CircumferenceInitialRimStateCalculator
from saturation.rim_erasure_calculators import get_rim_erasure_calculator
from saturation.crater_generation import get_grouped_craters


def test_crater_record_integration():
    """
    Integration/regression test for the crater record.
    """
    # Arrange
    nstop = 5000
    study_region_size = 1000
    study_region_padding = 125
    rstat = 15
    ratio = 5.0
    rmult = 1.5
    exponent = 0.5

    rim_erasure_calculator = get_rim_erasure_calculator(
        config={
            "name": "exponent_radius_ratio",
            "ratio": ratio,
            "exponent": exponent
        },
        rmult=rmult,
        rstat=rstat,
    )
    min_radius_threshold = rim_erasure_calculator.get_min_radius_threshold()

    distribution = ParetoProbabilityDistribution(alpha=1.5, x_min=min_radius_threshold / 2, x_max=250)
    crater_generator = get_grouped_craters(
        size_distribution=distribution,
        region_size=study_region_size + study_region_padding,
        min_radius_threshold=min_radius_threshold,
        rstat=rstat,
        random_seed=123
    )
    crater_record = CraterRecord(
        rstat=rstat,
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
        rstat=rstat
    )

    # Act
    counter = 0
    removed_counter = 0
    should_exit = False
    for crater_group, geq_rstat in crater_generator:
        if should_exit:
            break

        if not geq_rstat:
            removed_craters, removed_by_ids = crater_record.add_craters_smaller_than_rstat(crater_group)
            removed_counter += len(removed_craters)

            if len(removed_craters) > 0:
                areal_density_calculator.remove_craters(removed_craters)
        else:
            for crater in crater_group:
                removed_craters, removed_by_ids = crater_record.add_crater_geq_rstat(crater)
                removed_counter += len(removed_craters)

                areal_density_calculator.add_crater(crater)
                if len(removed_craters) > 0:
                    areal_density_calculator.remove_craters(removed_craters)

                counter += 1
                if crater_record.nstat == nstop:
                    should_exit = True
                    break

    # Assert
    print(len(crater_record.all_craters_in_record))
    print(f"{counter}, {removed_counter}, {crater_record.nobs}, {crater_record.nstat}, {areal_density_calculator.areal_density}, {crater.id}")
    assert crater_record.nstat == nstop
    assert crater_record.nobs == 68
    assert removed_counter == 6141
    assert areal_density_calculator.areal_density == 0.286203
    assert crater.id == 1512725
