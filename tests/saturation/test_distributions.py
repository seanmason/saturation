import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LinearRegression

from saturation.distributions import PowerLawProbabilityDistribution


def test_power_law_distribution_round_trip():
    # Arrange
    distribution = PowerLawProbabilityDistribution(slope=-3, min_value=1)

    # Act
    cumulative = distribution.cdf(1.5)
    result = distribution.uniform_to_density(cumulative)

    # Assert
    assert result == 1.5


def test_power_law_distribution_inverse_cdf_max_value_respected():
    # Arrange
    distribution = PowerLawProbabilityDistribution(slope=-2.8, min_value=1, max_value=10)

    # Act
    result = distribution.uniform_to_density(1)

    # Assert
    assert_almost_equal(result, 10)


def test_power_law_distribution_slope():
    """
    The slope in log-log space should equal the CDF's slope.
    """
    # Arrange
    distribution = PowerLawProbabilityDistribution(slope=-2.8, min_value=1)

    # Act
    n_samples = 500000
    y = np.arange(n_samples, 0, -1)

    x = [distribution.uniform_to_density(x) for x in np.random.rand(n_samples)]
    x = sorted(x)

    ln_x = np.reshape(np.log(x), newshape=(len(x), 1))
    ln_y = np.log(y)

    reg = LinearRegression().fit(ln_x, ln_y)

    # Assert
    # Within a tolerance of 0.5%
    assert abs(1 - reg.coef_[0] / -1.8) < 0.01


def test_power_law_distribution_pdf_sums_to_1():
    # Arrange
    min_value = 5
    max_value = 100
    steps = 100000
    distribution = PowerLawProbabilityDistribution(slope=-2.8, min_value=min_value, max_value=max_value)

    # Act
    # Analytical integral via right-handed Riemann sum
    step_size = (max_value - min_value) / steps
    test_values = [min_value + (x + 1) * step_size for x in range(steps)]
    result = sum([distribution.pdf(x) * step_size for x in test_values])

    # Assert
    # Should be within 0.1%
    assert abs(result - 1) < 1e-3
