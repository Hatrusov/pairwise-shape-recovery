"""OLS regression utilities."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    warnings: MutableSequence[str] | None = None,
    context: str = "",
) -> float:
    """Weighted absolute percentage error."""

    denominator = float(np.sum(np.abs(y_true)))
    if np.isclose(denominator, 0.0):
        message = "WAPE denominator is close to zero"
        if context:
            message = f"{message}: {context}"
        if warnings is not None:
            warnings.append(message)
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def fit_ols_regression(
    data: pd.DataFrame,
    dependent_variable: str,
    independent_variable: str,
    regime: str,
    n_value: int,
    method: str,
    warnings: MutableSequence[str] | None = None,
) -> dict[str, Any]:
    """Fit one simple OLS regression and return a flat result record."""

    subset = data[[dependent_variable, independent_variable]].dropna()
    y = subset[dependent_variable].to_numpy(dtype=float)
    x = subset[independent_variable].to_numpy(dtype=float)
    design = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, design).fit()
    y_pred = model.predict(design)
    residuals = y - y_pred
    n_obs = int(model.nobs)
    regression_standard_error = (
        float(np.sqrt(np.sum(residuals**2) / (n_obs - 2))) if n_obs > 2 else np.nan
    )
    r = float(np.corrcoef(x, y)[0, 1]) if n_obs > 1 else np.nan
    ci = model.conf_int(alpha=0.05)

    return {
        "regime": regime,
        "N": n_value,
        "method": method,
        "dependent_variable": dependent_variable,
        "independent_variable": independent_variable,
        "n_observations": n_obs,
        "intercept": float(model.params[0]),
        "slope": float(model.params[1]),
        "R": r,
        "R2": float(model.rsquared),
        "adjusted_R2": float(model.rsquared_adj),
        "regression_standard_error": regression_standard_error,
        "F_statistic": float(model.fvalue) if model.fvalue is not None else np.nan,
        "p_value_F": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "p_value_intercept": float(model.pvalues[0]),
        "p_value_slope": float(model.pvalues[1]),
        "ci95_intercept_low": float(ci[0, 0]),
        "ci95_intercept_high": float(ci[0, 1]),
        "ci95_slope_low": float(ci[1, 0]),
        "ci95_slope_high": float(ci[1, 1]),
        "WAPE": wape(
            y,
            y_pred,
            warnings,
            f"{regime} N={n_value} {dependent_variable} ~ {independent_variable}",
        ),
        "MAE": float(np.mean(np.abs(residuals))),
        "RMSE": float(np.sqrt(np.mean(residuals**2))),
    }


def _matrix_key_parts(key: tuple) -> tuple[str, str | None, int]:
    if len(key) == 2:
        regime, n_value = key
        return str(regime), None, int(n_value)
    if len(key) == 3:
        regime, distribution, n_value = key
        return str(regime), str(distribution), int(n_value)
    raise ValueError(f"Unsupported result matrix key: {key}")


def build_direct_kurtosis_regressions(
    result_matrices: dict[tuple, pd.DataFrame],
    warnings: MutableSequence[str] | None = None,
) -> pd.DataFrame:
    """Build original-kurtosis regressions on transformed-sequence kurtosis."""

    predictors = [
        ("ordered_difference", "ordered_difference_kurtosis"),
        ("direct_difference", "direct_difference_kurtosis"),
        ("half_sum", "half_sum_kurtosis"),
    ]
    rows = []
    for key, matrix in sorted(result_matrices.items()):
        regime, distribution, n_value = _matrix_key_parts(key)
        for method, predictor in predictors:
            row = fit_ols_regression(
                matrix,
                "original_kurtosis",
                predictor,
                regime,
                n_value,
                method,
                warnings,
            )
            if distribution is not None:
                row["distribution"] = distribution
            rows.append(row)
    return pd.DataFrame(rows)


def build_inverse_half_sum_regressions(
    result_matrices: dict[tuple, pd.DataFrame],
    warnings: MutableSequence[str] | None = None,
) -> pd.DataFrame:
    """Build inverse regressions from half-sum skewness and kurtosis."""

    specifications = [
        ("original_skewness", "half_sum_skewness"),
        ("original_kurtosis", "half_sum_kurtosis"),
    ]
    rows = []
    for key, matrix in sorted(result_matrices.items()):
        regime, distribution, n_value = _matrix_key_parts(key)
        for dependent, predictor in specifications:
            row = fit_ols_regression(
                matrix,
                dependent,
                predictor,
                regime,
                n_value,
                "half_sum",
                warnings,
            )
            if distribution is not None:
                row["distribution"] = distribution
            rows.append(row)
    return pd.DataFrame(rows)
