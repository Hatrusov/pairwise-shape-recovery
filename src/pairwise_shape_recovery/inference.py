"""Empirical shape and influence covariance from complete unlabeled half-sums.

The covariance returned is C, not C/n and not exact sampling covariance.
Only valid complete half-sum multisets are covered by the recovery theorem.
Length checks alone cannot establish that an arbitrary input is such a multiset.
"""

from math import comb

import numpy as np

EXCLUDED = (4, 8, 16, 32, 64, 128)


def standardized_moments(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < 3:
        raise ValueError("A finite one-dimensional sample of length >= 3 is required")
    z = x - x.mean()
    v = np.mean(z * z)
    if v <= 0:
        raise ValueError("Positive variance required")
    t = z / np.sqrt(v)
    return np.array([np.mean(t**r) for r in range(9)])


def covariance_from_moments(lam):
    g, b = lam[3:5]
    # Independent polynomial-product implementation of E[psi_i psi_j].
    a = np.array([g / 2, -3, -1.5 * g, 1.0])
    c = np.array([b, -4 * g, -2 * b, 0.0, 1.0])

    def product(p, q):
        v = np.convolve(p, q)
        return v @ lam[: len(v)]

    return np.array([[product(a, a), product(a, c)], [product(a, c), product(c, c)]])


def direct_covariance(x):
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    t = z / np.sqrt(np.mean(z * z))
    g, b = np.mean(t**3), np.mean(t**4)
    psi = np.array(
        [t**3 - 1.5 * g * t * t - 3 * t + g / 2, t**4 - 2 * b * t * t - 4 * g * t + b]
    )
    return psi @ psi.T / len(x)


def recover_from_pairs(s, n):
    if not isinstance(n, (int, np.integer)) or n < 3:
        raise ValueError("n must be an integer >= 3")
    if n in EXCLUDED:
        raise ValueError(
            "Universal guarantee fails; this does not prove ambiguity of this input"
        )
    s = np.asarray(s, dtype=float)
    if s.ndim != 1 or len(s) != n * (n - 1) // 2 or not np.isfinite(s).all():
        raise ValueError("Incorrect length, dimension, or nonfinite pair values")
    y = s - s.mean()
    scale = np.sqrt(np.mean(y * y))
    if scale <= 0:
        raise ValueError("Positive variance required")
    y = y / scale
    A = np.zeros(9)
    A[0] = n
    cancellation = []
    for r in range(2, 9):
        lhs = 2**r * np.sum(y**r)
        cross = sum(comb(r, k) * A[k] * A[r - k] for k in range(1, r)) / 2
        A[r] = (lhs - cross) / (n - 2 ** (r - 1))
        cancellation.append(
            (abs(lhs) + abs(cross)) / max(abs(lhs - cross), np.finfo(float).tiny)
        )
    if A[2] <= 0:
        raise ValueError("Recovered variance is nonpositive")
    lam = (A / n) / (A[2] / n) ** (np.arange(9) / 2)
    C = covariance_from_moments(lam)
    return {
        "skewness": lam[3],
        "pearson_kurtosis": lam[4],
        "C": C,
        "estimated_sampling_covariance": C / n,
        "moments": lam,
        "min_eigenvalue": np.linalg.eigvalsh(C)[0],
        "max_cancellation_ratio": max(cancellation),
    }
