"""Analytical identities for pairwise half-sums."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairwise_shape_recovery.config import StatisticsConfig
from pairwise_shape_recovery.statistics import compute_characteristics
from pairwise_shape_recovery.transforms import pairwise_half_sums


def validate_half_sum_identity_config(stats_config: StatisticsConfig) -> None:
    """Ensure moment definitions match the analytical half-sum identities."""

    if stats_config.sd_mode != "population":
        raise ValueError("Half-sum identities require sd_mode='population'.")
    if stats_config.skew_mode != "adjusted_fisher_pearson":
        raise ValueError(
            "Half-sum skewness identity requires skew_mode='adjusted_fisher_pearson'."
        )
    if stats_config.kurtosis_mode != "standardized_fourth_moment_no_minus_3":
        raise ValueError(
            "Half-sum kurtosis identity requires "
            "kurtosis_mode='standardized_fourth_moment_no_minus_3'."
        )


def half_sum_kurtosis_inverse_intercept(n: int) -> float:
    """Intercept in beta2_x = a + b * beta2_half_sum for N > 8."""

    if n <= 8:
        raise ValueError("Half-sum kurtosis inverse identity requires N > 8.")
    return float(-3.0 * n / (n - 8.0))


def half_sum_kurtosis_inverse_slope(n: int) -> float:
    """Slope in beta2_x = a + b * beta2_half_sum for N > 8."""

    if n <= 8:
        raise ValueError("Half-sum kurtosis inverse identity requires N > 8.")
    return float(2.0 * (n - 2.0) ** 2 / ((n - 1.0) * (n - 8.0)))


def half_sum_skewness_inverse_slope(n: int) -> float:
    """Slope linking adjusted skewness of x to adjusted skewness of half-sums."""

    if n <= 4:
        raise ValueError("Half-sum skewness inverse identity requires N > 4.")
    m = n * (n - 1) / 2.0
    numerator = n * np.sqrt((n - 1.0) * (n - 2.0)) * (m - 2.0)
    denominator = m * np.sqrt(m - 1.0) * (n - 4.0)
    return float(numerator / denominator)


def analytic_inverse_kurtosis_from_half_sum(beta2_s: float | np.ndarray, n: int):
    """Predict original beta2 from half-sum beta2."""

    return half_sum_kurtosis_inverse_intercept(n) + half_sum_kurtosis_inverse_slope(
        n
    ) * np.asarray(beta2_s)


def analytic_inverse_skewness_from_half_sum(g1_s: float | np.ndarray, n: int):
    """Predict original adjusted Fisher-Pearson skewness from half-sum skewness."""

    return half_sum_skewness_inverse_slope(n) * np.asarray(g1_s)


def quick_half_sum_identity_check(
    sample: np.ndarray,
    stats_config: StatisticsConfig,
) -> dict[str, float]:
    """Check analytical half-sum identities for one normalized sample."""

    validate_half_sum_identity_config(stats_config)
    n = int(sample.size)
    original = compute_characteristics(sample, stats_config)
    half = compute_characteristics(pairwise_half_sums(sample), stats_config)
    predicted_skew = float(analytic_inverse_skewness_from_half_sum(half["skewness"], n))
    predicted_kurtosis = float(
        analytic_inverse_kurtosis_from_half_sum(half["kurtosis"], n)
    )
    return {
        "skew_abs_error": abs(predicted_skew - original["skewness"]),
        "kurtosis_abs_error": abs(predicted_kurtosis - original["kurtosis"]),
    }


def build_half_sum_analytic_checks(
    result_matrices: dict[tuple, pd.DataFrame],
    inverse_regressions: pd.DataFrame,
    stats_config: StatisticsConfig,
) -> pd.DataFrame:
    """Compare analytical half-sum identities against simulated result matrices."""

    validate_half_sum_identity_config(stats_config)
    rows: list[dict[str, float | int | str]] = []
    for key, matrix in sorted(result_matrices.items()):
        if len(key) == 2:
            regime, n = key
            distribution = None
        elif len(key) == 3:
            regime, distribution, n = key
        else:
            raise ValueError(f"Unsupported result matrix key: {key}")
        skew_pred = analytic_inverse_skewness_from_half_sum(
            matrix["half_sum_skewness"].to_numpy(dtype=float), n
        )
        kurt_pred = analytic_inverse_kurtosis_from_half_sum(
            matrix["half_sum_kurtosis"].to_numpy(dtype=float), n
        )
        skew_errors = np.abs(skew_pred - matrix["original_skewness"].to_numpy(float))
        kurt_errors = np.abs(
            kurt_pred - matrix["original_kurtosis"].to_numpy(dtype=float)
        )

        skew_ols = inverse_regressions[
            (inverse_regressions["regime"] == regime)
            & (inverse_regressions["N"] == n)
            & (inverse_regressions["dependent_variable"] == "original_skewness")
        ]
        kurt_ols = inverse_regressions[
            (inverse_regressions["regime"] == regime)
            & (inverse_regressions["N"] == n)
            & (inverse_regressions["dependent_variable"] == "original_kurtosis")
        ]
        if distribution is not None:
            skew_ols = skew_ols[skew_ols["distribution"] == distribution]
            kurt_ols = kurt_ols[kurt_ols["distribution"] == distribution]
        row: dict[str, float | int | str] = {
            "regime": regime,
            "N": n,
            "n_observations": int(len(matrix)),
            "analytic_skew_slope": half_sum_skewness_inverse_slope(n),
            "ols_skew_intercept": float(skew_ols["intercept"].iloc[0]),
            "ols_skew_slope": float(skew_ols["slope"].iloc[0]),
            "skew_max_abs_error": float(np.max(skew_errors)),
            "skew_mean_abs_error": float(np.mean(skew_errors)),
            "skew_median_abs_error": float(np.median(skew_errors)),
            "analytic_kurtosis_intercept": half_sum_kurtosis_inverse_intercept(n),
            "analytic_kurtosis_slope": half_sum_kurtosis_inverse_slope(n),
            "ols_kurtosis_intercept": float(kurt_ols["intercept"].iloc[0]),
            "ols_kurtosis_slope": float(kurt_ols["slope"].iloc[0]),
            "kurtosis_max_abs_error": float(np.max(kurt_errors)),
            "kurtosis_mean_abs_error": float(np.mean(kurt_errors)),
            "kurtosis_median_abs_error": float(np.median(kurt_errors)),
        }
        if distribution is not None:
            row["distribution"] = distribution
        rows.append(row)
    return pd.DataFrame(rows)
