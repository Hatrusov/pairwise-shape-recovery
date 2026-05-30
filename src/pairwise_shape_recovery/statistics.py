"""Moment calculations used by the reproducible protocol."""

from __future__ import annotations

from typing import Any

import numpy as np

from pairwise_shape_recovery.config import StatisticsConfig


NEAR_ZERO_VARIANCE = 1e-14


def _as_1d_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    if arr.size == 0:
        raise ValueError("Expected a non-empty array.")
    return arr


def compute_characteristics(
    x: np.ndarray, mode: StatisticsConfig | None = None
) -> dict[str, Any]:
    """Compute mean, population SD, skewness, and kurtosis.

    Kurtosis defaults to the standardized fourth central moment beta2 and does
    not subtract 3. Skewness defaults to adjusted Fisher-Pearson G1.
    """

    stats_mode = mode or StatisticsConfig()
    arr = _as_1d_float_array(x)
    n = int(arr.size)

    mean_x = float(np.mean(arr))
    centered = arr - mean_x

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m2 = float(np.mean(centered**2))

    if stats_mode.sd_mode != "population":
        raise ValueError(f"Unsupported sd_mode: {stats_mode.sd_mode}")

    sd = float(np.sqrt(m2)) if m2 >= 0.0 else np.nan

    if not np.isfinite(m2) or m2 <= NEAR_ZERO_VARIANCE:
        skewness = np.nan
        kurtosis = np.nan
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            m3 = float(np.mean(centered**3))
            m4 = float(np.mean(centered**4))
            g1 = m3 / (m2 ** 1.5)
            beta2 = m4 / (m2**2)

        if stats_mode.skew_mode == "adjusted_fisher_pearson":
            skewness = (
                float(np.sqrt(n * (n - 1)) / (n - 2) * g1)
                if n > 2
                else np.nan
            )
        elif stats_mode.skew_mode == "moment":
            skewness = float(g1)
        else:
            raise ValueError(f"Unsupported skew_mode: {stats_mode.skew_mode}")

        if stats_mode.kurtosis_mode == "standardized_fourth_moment_no_minus_3":
            kurtosis = float(beta2)
        elif stats_mode.kurtosis_mode == "excess":
            kurtosis = float(beta2 - 3.0)
        else:
            raise ValueError(
                f"Unsupported kurtosis_mode: {stats_mode.kurtosis_mode}"
            )

    return {
        "mean": mean_x,
        "sd": sd,
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "n": n,
    }


def normalize_sample(
    x: np.ndarray, stats_config: StatisticsConfig | None = None
) -> np.ndarray:
    """Normalize one finite sample by its own mean and population SD."""

    arr = _as_1d_float_array(x)
    mode = stats_config or StatisticsConfig()
    if mode.sd_mode != "population":
        raise ValueError("normalize_sample requires sd_mode='population'.")
    mean_x = float(np.mean(arr))
    centered = arr - mean_x
    with np.errstate(over="ignore", invalid="ignore"):
        m2 = float(np.mean(centered**2))
    sd = float(np.sqrt(m2)) if m2 >= 0.0 else np.nan
    if not np.isfinite(sd) or sd <= NEAR_ZERO_VARIANCE:
        raise ValueError("Cannot normalize sample with non-finite or near-zero SD.")
    return centered / sd
