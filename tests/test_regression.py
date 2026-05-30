from __future__ import annotations

import numpy as np
import pandas as pd

from pairwise_shape_recovery.regression import (
    build_direct_kurtosis_regressions,
    build_inverse_half_sum_regressions,
    fit_ols_regression,
    wape,
)
from pairwise_shape_recovery.simulation import RESULT_COLUMNS


def test_wape_simple_example() -> None:
    y_true = np.asarray([2.0, 4.0])
    y_pred = np.asarray([1.0, 5.0])

    assert np.isclose(wape(y_true, y_pred), 2.0 / 6.0)


def test_wape_zero_denominator_warns() -> None:
    warnings: list[str] = []
    value = wape(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]), warnings)

    assert np.isnan(value)
    assert warnings


def test_fit_ols_regression_exact_line() -> None:
    x = np.arange(5, dtype=float)
    data = pd.DataFrame({"x": x, "y": 2.0 + 3.0 * x})
    row = fit_ols_regression(data, "y", "x", "regime", 10, "method")

    assert np.isclose(row["intercept"], 2.0)
    assert np.isclose(row["slope"], 3.0)
    assert np.isclose(row["R2"], 1.0)
    for key in [
        "regime",
        "N",
        "method",
        "dependent_variable",
        "independent_variable",
        "n_observations",
        "intercept",
        "slope",
        "R",
        "R2",
        "adjusted_R2",
        "regression_standard_error",
        "F_statistic",
        "p_value_F",
        "p_value_intercept",
        "p_value_slope",
        "ci95_intercept_low",
        "ci95_intercept_high",
        "ci95_slope_low",
        "ci95_slope_high",
        "WAPE",
        "MAE",
        "RMSE",
    ]:
        assert key in row


def test_regression_builders_row_counts() -> None:
    data = pd.DataFrame(
        {
            column: np.linspace(1.0, 5.0, 5) + index
            for index, column in enumerate(RESULT_COLUMNS)
        }
    )
    matrices = {
        ("low_skew_selected", 10): data,
        ("moderate_shape_selected", 10): data,
    }

    direct = build_direct_kurtosis_regressions(matrices)
    inverse = build_inverse_half_sum_regressions(matrices)

    assert len(direct) == 6
    assert len(inverse) == 4
