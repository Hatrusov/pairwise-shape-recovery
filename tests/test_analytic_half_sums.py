from __future__ import annotations

import numpy as np
import pytest

from pairwise_shape_recovery.analytic import (
    analytic_inverse_kurtosis_from_half_sum,
    analytic_inverse_skewness_from_half_sum,
    half_sum_kurtosis_inverse_intercept,
    half_sum_kurtosis_inverse_slope,
    half_sum_skewness_inverse_slope,
    validate_half_sum_identity_config,
)
from pairwise_shape_recovery.config import PipelineConfig, StatisticsConfig
from pairwise_shape_recovery.distributions import generate_sample
from pairwise_shape_recovery.statistics import compute_characteristics, normalize_sample
from pairwise_shape_recovery.transforms import pairwise_half_sums


def test_half_sum_identities_across_distribution_families() -> None:
    rng = np.random.default_rng(12345)
    pipeline_config = PipelineConfig()
    stats_config = StatisticsConfig()
    distributions = [
        "uniform",
        "normal",
        "cauchy",
        "student_t5",
        "lognormal",
        "gamma",
        "weibull",
        "skewnormal",
    ]

    for distribution in distributions:
        for n_value in [10, 20, 30, 40, 50]:
            x = generate_sample(
                distribution,
                n_value,
                rng,
                pipeline_config.distribution_parameters,
            )
            x = normalize_sample(x, stats_config)
            half = pairwise_half_sums(x)
            original_stats = compute_characteristics(x, stats_config)
            half_stats = compute_characteristics(half, stats_config)

            predicted_skew = analytic_inverse_skewness_from_half_sum(
                half_stats["skewness"], n_value
            )
            predicted_kurtosis = analytic_inverse_kurtosis_from_half_sum(
                half_stats["kurtosis"], n_value
            )

            assert np.isclose(
                predicted_skew, original_stats["skewness"], atol=1e-10, rtol=1e-10
            )
            assert np.isclose(
                predicted_kurtosis,
                original_stats["kurtosis"],
                atol=1e-10,
                rtol=1e-10,
            )


def test_vectorized_analytic_inverse_functions_return_expected_shape() -> None:
    values = np.asarray([1.0, 2.0, 3.0])

    skew = analytic_inverse_skewness_from_half_sum(values, 10)
    kurtosis = analytic_inverse_kurtosis_from_half_sum(values, 10)

    assert isinstance(skew, np.ndarray)
    assert isinstance(kurtosis, np.ndarray)
    assert skew.shape == values.shape
    assert kurtosis.shape == values.shape


def test_boundary_errors() -> None:
    with pytest.raises(ValueError):
        half_sum_kurtosis_inverse_intercept(8)
    with pytest.raises(ValueError):
        half_sum_kurtosis_inverse_slope(8)
    with pytest.raises(ValueError):
        half_sum_skewness_inverse_slope(4)


@pytest.mark.parametrize(
    "stats_config",
    [
        StatisticsConfig(skew_mode="moment"),
        StatisticsConfig(kurtosis_mode="excess"),
        StatisticsConfig(sd_mode="sample"),
    ],
)
def test_invalid_half_sum_identity_config_raises(
    stats_config: StatisticsConfig,
) -> None:
    with pytest.raises(ValueError):
        validate_half_sum_identity_config(stats_config)
