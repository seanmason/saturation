import numpy as np
from sklearn.linear_model import LinearRegression

from saturation.distributions import PowerLawProbabilityDistribution


def test_power_law_distribution_round_trip():
    # Arrange
    pdf = PowerLawProbabilityDistribution(slope=-3, min_value=1)

    # Act
    p = pdf.value_to_probability(1.5)
    result = pdf.probability_to_value(p)

    # Assert
    assert result == 1.5


def test_power_law_distribution_slope():
    """
    The slope in log-log space should equal the CDF's slope.
    """
    # Arrange
    pdf = PowerLawProbabilityDistribution(slope=-2.8, min_value=1)

    # Act
    n_samples = 500000
    y = np.arange(n_samples, 0, -1)

    x = [pdf.uniform_to_value(x) for x in np.random.rand(n_samples)]
    x = sorted(x)

    ln_x = np.reshape(np.log(x), newshape=(len(x), 1))
    ln_y = np.log(y)

    reg = LinearRegression().fit(ln_x, ln_y)

    # Assert
    # Within a tolerance of 0.5%
    assert abs(1 - reg.coef_[0] / -1.8) < 0.01


def test_power_law_distribution_sums_to_1():
    # Arrange
    min_value = 5
    step_size = 0.001
    steps = 1000000
    pdf = PowerLawProbabilityDistribution(slope=-2.8, min_value=min_value)

    # Act
    # Analytical integral via right-handed Riemann sum
    test_values = [min_value + (x + 1) * step_size for x in range(steps)]
    result = sum([pdf.value_to_probability(x) * step_size for x in test_values])

    # Assert
    # Should be within 0.1%
    assert abs(result - 1) < 1e-3
