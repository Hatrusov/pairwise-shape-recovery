"""Random sample generators for the simulation protocol."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
from scipy import stats

from pairwise_shape_recovery.statistics import NEAR_ZERO_VARIANCE


SUPPORTED_DISTRIBUTIONS = {
    "uniform",
    "normal",
    "cauchy",
    "student_t5",
    "lognormal",
    "gamma",
    "weibull",
    "skewnormal",
}


def _increment_rejection(
    metadata: MutableMapping[str, Any] | None,
    distribution_name: str,
    n: int,
    reason: str,
) -> None:
    if metadata is None:
        return
    key = f"{distribution_name}_N{n}_{reason}"
    rejections = metadata.setdefault("generation_rejections", {})
    rejections[key] = int(rejections.get(key, 0)) + 1


def _draw_once(
    distribution_name: str,
    n: int,
    rng: np.random.Generator,
    parameters: Mapping[str, float],
) -> np.ndarray:
    if distribution_name == "uniform":
        return rng.uniform(parameters["low"], parameters["high"], size=n)
    if distribution_name == "normal":
        return rng.normal(parameters["loc"], parameters["scale"], size=n)
    if distribution_name == "cauchy":
        return stats.cauchy.rvs(
            loc=parameters["loc"],
            scale=parameters["scale"],
            size=n,
            random_state=rng,
        )
    if distribution_name == "student_t5":
        return stats.t.rvs(df=parameters["df"], size=n, random_state=rng)
    if distribution_name == "lognormal":
        return rng.lognormal(
            mean=parameters["mean"], sigma=parameters["sigma"], size=n
        )
    if distribution_name == "gamma":
        return stats.gamma.rvs(
            a=parameters["shape"],
            scale=parameters["scale"],
            size=n,
            random_state=rng,
        )
    if distribution_name == "weibull":
        return stats.weibull_min.rvs(
            c=parameters["shape"], size=n, random_state=rng
        )
    if distribution_name == "skewnormal":
        return stats.skewnorm.rvs(
            a=parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
            size=n,
            random_state=rng,
        )
    raise ValueError(f"Unsupported distribution: {distribution_name}")


def generate_sample(
    distribution_name: str,
    n: int,
    rng: np.random.Generator,
    distribution_parameters: Mapping[str, Mapping[str, float]],
    metadata: MutableMapping[str, Any] | None = None,
    max_attempts: int = 10000,
) -> np.ndarray:
    """Generate one finite one-dimensional sample with non-degenerate variance."""

    if distribution_name not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution: {distribution_name}")
    if distribution_name not in distribution_parameters:
        raise ValueError(f"Missing parameters for distribution: {distribution_name}")
    if n <= 1:
        raise ValueError("Sample size must be greater than one.")

    for _attempt in range(max_attempts):
        sample = np.asarray(
            _draw_once(
                distribution_name,
                n,
                rng,
                distribution_parameters[distribution_name],
            ),
            dtype=float,
        )
        if sample.ndim != 1 or sample.size != n:
            _increment_rejection(metadata, distribution_name, n, "bad_shape")
            continue
        if not np.all(np.isfinite(sample)):
            _increment_rejection(metadata, distribution_name, n, "non_finite")
            continue

        centered = sample - float(np.mean(sample))
        with np.errstate(over="ignore", invalid="ignore"):
            m2 = float(np.mean(centered**2))
        if not np.isfinite(m2) or m2 <= NEAR_ZERO_VARIANCE:
            _increment_rejection(metadata, distribution_name, n, "near_zero_sd")
            continue
        return sample

    raise RuntimeError(
        f"Could not generate a valid {distribution_name} sample of size {n} "
        f"after {max_attempts} attempts."
    )
