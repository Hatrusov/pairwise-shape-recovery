import numpy as np
import pytest

from pairwise_shape_recovery.analytic import (
    analytic_inverse_kurtosis_from_half_sum,
    analytic_inverse_skewness_from_half_sum,
)
from pairwise_shape_recovery.inference import (
    EXCLUDED,
    covariance_from_moments,
    direct_covariance,
    recover_from_pairs,
    standardized_moments,
)


@pytest.mark.parametrize("n", [3, 5, 6, 7, 10, 20, 30, 50, 100, 129, 256])
def test_pair_only_matches_direct(n):
    x = np.random.default_rng(761 + n).exponential(size=n)
    i, j = np.triu_indices(n, 1)
    s = (x[i] + x[j]) / 2
    r = recover_from_pairs(s, n)
    np.testing.assert_allclose(r["C"], direct_covariance(x), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        r["moments"], standardized_moments(x), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        recover_from_pairs(s[::-1], n)["C"], r["C"], rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize("n", EXCLUDED)
def test_exceptional_size_diagnostic(n):
    with pytest.raises(ValueError, match="Universal guarantee"):
        recover_from_pairs(np.zeros(n * (n - 1) // 2), n)


def test_validation_and_polynomial_covariance():
    for s, n in [([1, 2], 3), ([1, 1, 1], 3), ([1, 2, float("nan")], 3)]:
        with pytest.raises(ValueError):
            recover_from_pairs(s, n)
    x = np.array([-3.0, -1.0, 0.0, 1.0, 7.0])
    np.testing.assert_allclose(
        covariance_from_moments(standardized_moments(x)),
        direct_covariance(x),
        atol=1e-12,
    )


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 16])
def test_small_size_shape_inversions_and_difference_identity(n):
    x = np.random.default_rng(n).normal(size=n)
    i, j = np.triu_indices(n, 1)
    s = (x[i] + x[j]) / 2

    def shape(a):
        z = a - a.mean()
        m2 = np.mean(z * z)
        return np.mean(z**3) / m2**1.5 * np.sqrt(len(a) * (len(a) - 1)) / (
            len(a) - 2
        ), np.mean(z**4) / m2**2

    g, b = shape(x)
    gs, bs = shape(s)
    if n != 4:
        assert np.isclose(analytic_inverse_skewness_from_half_sum(gs, n), g)
    else:
        assert abs(gs) < 1e-12
    if n != 8:
        assert np.isclose(analytic_inverse_kurtosis_from_half_sum(bs, n), b)
    else:
        assert np.isclose(bs, 7 / 3)
    d = np.abs(x[i] - x[j])
    assert np.isclose(2 * n / (n - 1) * np.mean(d**4) / np.mean(d * d) ** 2 - 3, b)
