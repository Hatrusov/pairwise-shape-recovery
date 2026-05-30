"""Pairwise transformations of one-dimensional samples."""

from __future__ import annotations

import numpy as np


def _validate_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    if arr.size < 2:
        raise ValueError("At least two observations are required.")
    return arr


def ordered_pairwise_differences(x: np.ndarray) -> np.ndarray:
    """Ranked pairwise differences after sorting the sample descending."""

    arr = np.sort(_validate_vector(x))[::-1]
    n = arr.size
    values = [arr[i] - arr[j] for i in range(n - 1) for j in range(i + 1, n)]
    result = np.asarray(values, dtype=float)
    expected = n * (n - 1) // 2
    if result.size != expected:
        raise RuntimeError("Unexpected ordered pairwise difference length.")
    return result


def direct_pairwise_differences(x: np.ndarray) -> np.ndarray:
    """Order-dependent signed pairwise differences without sorting."""

    arr = _validate_vector(x)
    n = arr.size
    values = [arr[i] - arr[j] for i in range(n - 1) for j in range(i + 1, n)]
    result = np.asarray(values, dtype=float)
    expected = n * (n - 1) // 2
    if result.size != expected:
        raise RuntimeError("Unexpected direct pairwise difference length.")
    return result


def pairwise_half_sums(x: np.ndarray) -> np.ndarray:
    """Pairwise half-sums for all pairs i < j."""

    arr = _validate_vector(x)
    n = arr.size
    values = [(arr[i] + arr[j]) / 2.0 for i in range(n - 1) for j in range(i + 1, n)]
    result = np.asarray(values, dtype=float)
    expected = n * (n - 1) // 2
    if result.size != expected:
        raise RuntimeError("Unexpected pairwise half-sum length.")
    return result

