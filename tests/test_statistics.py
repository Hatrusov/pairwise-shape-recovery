from __future__ import annotations

import numpy as np
import pytest

from pairwise_shape_recovery.config import StatisticsConfig
from pairwise_shape_recovery.statistics import compute_characteristics, normalize_sample


def test_compute_characteristics_population_moments() -> None:
    x = np.asarray([1.0, 2.0, 3.0])
    stats = compute_characteristics(x, StatisticsConfig())

    assert stats["mean"] == 2.0
    assert np.isclose(stats["sd"], np.sqrt(2.0 / 3.0))
    assert np.isclose(stats["skewness"], 0.0)
    assert np.isclose(stats["kurtosis"], 1.5)


def test_skew_moment_mode_and_adjusted_mode() -> None:
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    moment = compute_characteristics(x, StatisticsConfig(skew_mode="moment"))
    adjusted = compute_characteristics(x, StatisticsConfig())
    factor = np.sqrt(x.size * (x.size - 1)) / (x.size - 2)

    assert np.isclose(adjusted["skewness"], factor * moment["skewness"])


def test_kurtosis_excess_mode() -> None:
    x = np.asarray([1.0, 2.0, 3.0])
    default_stats = compute_characteristics(x, StatisticsConfig())
    excess_stats = compute_characteristics(
        x,
        StatisticsConfig(kurtosis_mode="excess"),
    )

    assert np.isclose(default_stats["kurtosis"], excess_stats["kurtosis"] + 3.0)


def test_unsupported_modes_raise() -> None:
    x = np.asarray([1.0, 2.0, 4.0])

    with pytest.raises(ValueError):
        compute_characteristics(x, StatisticsConfig(sd_mode="sample"))
    with pytest.raises(ValueError):
        compute_characteristics(x, StatisticsConfig(skew_mode="unknown"))
    with pytest.raises(ValueError):
        compute_characteristics(x, StatisticsConfig(kurtosis_mode="unknown"))


def test_near_zero_variance_behavior() -> None:
    x = np.asarray([1.0, 1.0, 1.0])
    stats = compute_characteristics(x, StatisticsConfig())

    assert stats["sd"] == 0.0
    assert np.isnan(stats["skewness"])
    assert np.isnan(stats["kurtosis"])
    with pytest.raises(ValueError):
        normalize_sample(x, StatisticsConfig())


def test_invalid_array_shapes_raise() -> None:
    with pytest.raises(ValueError):
        compute_characteristics(np.ones((2, 2)), StatisticsConfig())
    with pytest.raises(ValueError):
        compute_characteristics(np.asarray([]), StatisticsConfig())


def test_normalize_sample_uses_population_sd_and_preserves_shape_moments() -> None:
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    before = compute_characteristics(x, StatisticsConfig())
    normalized = normalize_sample(x, StatisticsConfig())
    after = compute_characteristics(normalized, StatisticsConfig())

    assert np.isclose(after["mean"], 0.0)
    assert np.isclose(after["sd"], 1.0)
    assert np.isclose(after["skewness"], before["skewness"])
    assert np.isclose(after["kurtosis"], before["kurtosis"])


def test_normalize_rejects_non_population_sd_mode() -> None:
    with pytest.raises(ValueError):
        normalize_sample(
            np.asarray([1.0, 2.0, 3.0]),
            StatisticsConfig(sd_mode="sample"),
        )

