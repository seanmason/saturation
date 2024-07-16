import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LinearRegression

from saturation.distributions import ParetoProbabilityDistribution


def test_pareto_distribution_uniform_to_density_max_value_respected():
    # Arrange
    distribution = ParetoProbabilityDistribution(alpha=2.8, x_min=1, x_max=25)

    # Act
    result = distribution.pullback(1)

    # Assert
    assert_almost_equal(result, 25)


def test_pareto_distribution_slope():
    """
    The slope in log-log space should equal the CDF's slope.
    """
    # Arrange
    distribution = ParetoProbabilityDistribution(alpha=2.8, x_min=1, x_max=25)

    # Act
    n_samples = 500000
    y = np.arange(n_samples, 0, -1)

    x = [distribution.pullback(x) for x in np.random.rand(n_samples)]
    x = sorted(x)

    ln_x = np.reshape(np.log(x), newshape=(len(x), 1))
    ln_y = np.log(y)

    reg = LinearRegression().fit(ln_x, ln_y)

    # Assert
    # Within a tolerance of .5%
    assert_almost_equal(reg.coef_[0], -2.8, 1)


def test_pareto_distribution_pdf_sums_to_1():
    # Arrange
    min_value = 5
    max_value = 100
    steps = 100000
    distribution = ParetoProbabilityDistribution(alpha=2.8, x_min=min_value, x_max=max_value)

    # Act
    # Analytical integral via right-handed Riemann sum
    step_size = (max_value - min_value) / steps
    test_values = [min_value + (x + 1) * step_size for x in range(steps)]
    result = sum([distribution.pdf(x) * step_size for x in test_values])

    # Assert
    # Should be within .1%
    assert_almost_equal(result, 1, 3)
