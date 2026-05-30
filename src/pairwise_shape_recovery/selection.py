"""Finite-sample selection procedures."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
import pandas as pd

from pairwise_shape_recovery.config import (
    LowSkewSelectionConfig,
    ModerateShapeSelectionConfig,
    StatisticsConfig,
)
from pairwise_shape_recovery.distributions import generate_sample
from pairwise_shape_recovery.statistics import compute_characteristics, normalize_sample


def _summary(values: list[float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"min": np.nan, "median": np.nan, "max": np.nan}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _candidate_record(
    distribution_name: str,
    n: int,
    raw: np.ndarray,
    normalized: np.ndarray,
    stats_config: StatisticsConfig,
) -> dict[str, Any] | None:
    norm_stats = compute_characteristics(normalized, stats_config)
    if not np.isfinite(norm_stats["skewness"]) or not np.isfinite(
        norm_stats["kurtosis"]
    ):
        return None

    raw_stats = compute_characteristics(raw, stats_config)
    return {
        "distribution": distribution_name,
        "N": n,
        "raw_sample": raw.tolist(),
        "normalized_sample": normalized.tolist(),
        "mean_raw": raw_stats["mean"],
        "sd_raw": raw_stats["sd"],
        "skew_raw": raw_stats["skewness"],
        "kurtosis_raw": raw_stats["kurtosis"],
        "mean_norm": norm_stats["mean"],
        "sd_norm": norm_stats["sd"],
        "skew_norm": norm_stats["skewness"],
        "kurtosis_norm": norm_stats["kurtosis"],
        "abs_skew": abs(norm_stats["skewness"]),
    }


def _generate_candidates(
    distribution_name: str,
    n: int,
    count: int,
    rng: np.random.Generator,
    stats_config: StatisticsConfig,
    distribution_parameters: Mapping[str, Mapping[str, float]],
    metadata: MutableMapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], int, int]:
    records: list[dict[str, Any]] = []
    generated = 0
    skipped_nonfinite_stats = 0
    while generated < count:
        generated += 1
        raw = generate_sample(
            distribution_name, n, rng, distribution_parameters, metadata
        )
        try:
            normalized = normalize_sample(raw, stats_config)
        except ValueError:
            skipped_nonfinite_stats += 1
            continue
        record = _candidate_record(distribution_name, n, raw, normalized, stats_config)
        if record is None:
            skipped_nonfinite_stats += 1
            continue
        records.append(record)
    return records, generated, skipped_nonfinite_stats


def select_low_skew_samples(
    distribution_name: str,
    n: int,
    n_selected: int,
    rng: np.random.Generator,
    config: LowSkewSelectionConfig,
    stats_config: StatisticsConfig,
    distribution_parameters: Mapping[str, Mapping[str, float]],
    metadata: MutableMapping[str, Any] | None = None,
    metadata_regime: str = "low_skew_selected",
) -> pd.DataFrame:
    """Select finite samples with small absolute adjusted skewness."""

    candidates: list[dict[str, Any]] = []
    generated_total = 0
    skipped_total = 0
    next_batch = int(config.initial_pool_size)

    while generated_total < config.max_pool_size:
        batch_size = min(next_batch, config.max_pool_size - generated_total)
        batch, generated, skipped = _generate_candidates(
            distribution_name,
            n,
            batch_size,
            rng,
            stats_config,
            distribution_parameters,
            metadata,
        )
        candidates.extend(batch)
        generated_total += generated
        skipped_total += skipped

        passing_count = sum(
            candidate["abs_skew"] <= config.skew_abs_threshold
            for candidate in candidates
        )
        if passing_count >= n_selected:
            break
        next_batch = min(max(next_batch * 2, 1), config.max_pool_size - generated_total)
        if next_batch <= 0:
            break

    sorted_candidates = sorted(candidates, key=lambda item: item["abs_skew"])
    selected = [
        candidate
        for candidate in sorted_candidates
        if candidate["abs_skew"] <= config.skew_abs_threshold
    ][:n_selected]
    if len(selected) < n_selected:
        raise RuntimeError(
            f"Only selected {len(selected)} low-skew samples for "
            f"{distribution_name}, N={n}; needed {n_selected}."
        )

    for rank, record in enumerate(selected, start=1):
        record["selection_score"] = record["abs_skew"]
        record["selection_rank"] = rank
        record["selection_rule"] = config.strategy

    if metadata is not None:
        abs_skews = [candidate["abs_skew"] for candidate in candidates]
        kurtoses = [candidate["kurtosis_norm"] for candidate in candidates]
        metadata.setdefault("selection_metadata", []).append(
            {
                "regime": metadata_regime,
                "selection_regime": metadata_regime,
                "selection_algorithm": "low_skew_threshold",
                "distribution_name": distribution_name,
                "N": n,
                "generated_candidates": generated_total,
                "valid_candidates": len(candidates),
                "skipped_nonfinite_stats": skipped_total,
                "passed_threshold": sum(
                    value <= config.skew_abs_threshold for value in abs_skews
                ),
                "selected": len(selected),
                "skew_abs_threshold": config.skew_abs_threshold,
                "strategy": config.strategy,
                "abs_skew_min": _summary(abs_skews)["min"],
                "abs_skew_median": _summary(abs_skews)["median"],
                "abs_skew_max": _summary(abs_skews)["max"],
                "kurtosis_min": _summary(kurtoses)["min"],
                "kurtosis_median": _summary(kurtoses)["median"],
                "kurtosis_max": _summary(kurtoses)["max"],
            }
        )

    return pd.DataFrame(selected)


def select_moderate_shape_samples(
    distribution_name: str,
    n: int,
    n_selected: int,
    rng: np.random.Generator,
    config: ModerateShapeSelectionConfig,
    stats_config: StatisticsConfig,
    distribution_parameters: Mapping[str, Mapping[str, float]],
    metadata: MutableMapping[str, Any] | None = None,
    metadata_regime: str = "moderate_shape_selected",
) -> pd.DataFrame:
    """Select finite samples by the configured moderate-shape score."""

    if config.decile_indexing != "one_based":
        raise ValueError("Only one_based decile_indexing is supported.")
    if config.score != "abs_skew_times_kurtosis":
        raise ValueError("Only abs_skew_times_kurtosis score is supported.")

    start = (config.decile - 1) * n_selected
    stop = config.decile * n_selected
    required_pool = max(config.pool_size, stop)

    candidates: list[dict[str, Any]] = []
    generated_total = 0
    skipped_total = 0
    while len(candidates) < stop:
        to_generate = required_pool - generated_total
        if to_generate <= 0:
            to_generate = stop - len(candidates)
        batch, generated, skipped = _generate_candidates(
            distribution_name,
            n,
            to_generate,
            rng,
            stats_config,
            distribution_parameters,
            metadata,
        )
        generated_total += generated
        skipped_total += skipped
        for record in batch:
            record["selection_score"] = record["abs_skew"] * record["kurtosis_norm"]
            candidates.append(record)

    sorted_candidates = sorted(candidates, key=lambda item: item["selection_score"])
    selected = sorted_candidates[start:stop]
    if len(selected) < n_selected:
        raise RuntimeError(
            f"Only selected {len(selected)} moderate-shape samples for "
            f"{distribution_name}, N={n}; needed {n_selected}."
        )

    for rank, record in enumerate(sorted_candidates, start=1):
        record["selection_rank"] = rank
        record["selection_rule"] = (
            "decile_abs_skew_times_kurtosis_moderate_shape"
        )

    if metadata is not None:
        skewness = [candidate["skew_norm"] for candidate in candidates]
        abs_skews = [candidate["abs_skew"] for candidate in candidates]
        kurtoses = [candidate["kurtosis_norm"] for candidate in candidates]
        scores = [candidate["selection_score"] for candidate in candidates]
        metadata.setdefault("selection_metadata", []).append(
            {
                "regime": metadata_regime,
                "selection_regime": metadata_regime,
                "selection_algorithm": "moderate_shape_score_decile",
                "distribution_name": distribution_name,
                "N": n,
                "pool_size_actual": generated_total,
                "valid_candidates": len(candidates),
                "skipped_nonfinite_stats": skipped_total,
                "decile": config.decile,
                "decile_indexing": config.decile_indexing,
                "start_index_zero_based": start,
                "stop_index_zero_based_exclusive": stop,
                "selected": len(selected),
                "score": config.score,
                "skewness_min": _summary(skewness)["min"],
                "skewness_median": _summary(skewness)["median"],
                "skewness_max": _summary(skewness)["max"],
                "abs_skew_min": _summary(abs_skews)["min"],
                "abs_skew_median": _summary(abs_skews)["median"],
                "abs_skew_max": _summary(abs_skews)["max"],
                "kurtosis_min": _summary(kurtoses)["min"],
                "kurtosis_median": _summary(kurtoses)["median"],
                "kurtosis_max": _summary(kurtoses)["max"],
                "xi_min": _summary(scores)["min"],
                "xi_median": _summary(scores)["median"],
                "xi_max": _summary(scores)["max"],
            }
        )

    return pd.DataFrame(selected)
