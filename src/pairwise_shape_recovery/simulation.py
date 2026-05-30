"""End-to-end simulation pipeline."""

from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pairwise_shape_recovery import __version__
from pairwise_shape_recovery.analytic import (
    build_half_sum_analytic_checks,
    validate_half_sum_identity_config,
)
from pairwise_shape_recovery.config import (
    PipelineConfig,
    config_to_dict,
    dump_config,
)
from pairwise_shape_recovery.io import (
    compute_file_sha256,
    dataframe_to_records,
    ensure_output_tree,
    get_git_info,
    save_dataframe_csv_parquet,
    save_json,
    save_parquet,
    setup_run_logger,
    write_summary_workbook,
)
from pairwise_shape_recovery.plotting import plot_correlation_heatmap
from pairwise_shape_recovery.regression import (
    build_direct_kurtosis_regressions,
    build_inverse_half_sum_regressions,
)
from pairwise_shape_recovery.selection import (
    select_low_skew_samples,
    select_moderate_shape_samples,
)
from pairwise_shape_recovery.statistics import compute_characteristics
from pairwise_shape_recovery.transforms import (
    direct_pairwise_differences,
    ordered_pairwise_differences,
    pairwise_half_sums,
)


RESULT_COLUMNS = [
    "original_skewness",
    "original_kurtosis",
    "ordered_difference_skewness",
    "ordered_difference_kurtosis",
    "direct_difference_skewness",
    "direct_difference_kurtosis",
    "half_sum_skewness",
    "half_sum_kurtosis",
]

RESULT_METADATA_COLUMNS = [
    "global_sample_id",
    "distribution",
    "population_family",
    "population_symmetry",
    "selection_regime",
    "selection_algorithm",
]

PRIMARY_PROTOCOL_REGIMES = [
    ("low_skew_threshold", "low_skew_selected"),
    ("moderate_shape_score_decile", "moderate_shape_selected"),
]

EXTENDED_PROTOCOL_REGIMES = PRIMARY_PROTOCOL_REGIMES

POPULATION_SYMMETRY = {
    "uniform": "symmetric",
    "normal": "symmetric",
    "cauchy": "symmetric",
    "student_t5": "symmetric",
    "lognormal": "asymmetric",
    "gamma": "asymmetric",
    "weibull": "asymmetric",
    "skewnormal": "asymmetric",
}


def compute_correlation_matrix(result_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute a Pearson correlation matrix for the result matrix."""

    return result_matrix.corr(method="pearson")


def build_result_matrix(samples: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Build the 8-column result matrix for selected normalized samples."""

    rows: list[dict[str, float]] = []
    for sample in samples["normalized_sample"]:
        x_norm = np.asarray(sample, dtype=float)

        original_stats = compute_characteristics(x_norm, config.statistics)
        ordered_stats = compute_characteristics(
            ordered_pairwise_differences(x_norm), config.statistics
        )
        direct_stats = compute_characteristics(
            direct_pairwise_differences(x_norm), config.statistics
        )
        half_sum_stats = compute_characteristics(
            pairwise_half_sums(x_norm), config.statistics
        )

        rows.append(
            {
                "original_skewness": original_stats["skewness"],
                "original_kurtosis": original_stats["kurtosis"],
                "ordered_difference_skewness": ordered_stats["skewness"],
                "ordered_difference_kurtosis": ordered_stats["kurtosis"],
                "direct_difference_skewness": direct_stats["skewness"],
                "direct_difference_kurtosis": direct_stats["kurtosis"],
                "half_sum_skewness": half_sum_stats["skewness"],
                "half_sum_kurtosis": half_sum_stats["kurtosis"],
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def attach_result_metadata(
    samples: pd.DataFrame, result_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Attach sample identifiers to a result matrix without changing row order."""

    metadata = samples[RESULT_METADATA_COLUMNS].reset_index(drop=True)
    return pd.concat([metadata, result_matrix.reset_index(drop=True)], axis=1)


def _prepare_samples_for_saving(
    samples: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    df = samples.copy()
    if not config.outputs.save_raw_samples and "raw_sample" in df:
        df = df.drop(columns=["raw_sample"])
    if not config.outputs.save_normalized_samples and "normalized_sample" in df:
        df = df.drop(columns=["normalized_sample"])
    return df


def _assemble_regime_samples(
    selection_algorithm: str,
    output_regime: str,
    n_value: int,
    distributions: list[str],
    n_selected: int,
    rng: np.random.Generator,
    config: PipelineConfig,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    frames = []
    for distribution in distributions:
        if selection_algorithm == "low_skew_threshold":
            selected = select_low_skew_samples(
                distribution,
                n_value,
                n_selected,
                rng,
                config.low_skew_selection,
                config.statistics,
                config.distribution_parameters,
                metadata,
                metadata_regime=output_regime,
            )
        elif selection_algorithm == "moderate_shape_score_decile":
            selected = select_moderate_shape_samples(
                distribution,
                n_value,
                n_selected,
                rng,
                config.moderate_shape_selection,
                config.statistics,
                config.distribution_parameters,
                metadata,
                metadata_regime=output_regime,
            )
        else:
            raise ValueError(f"Unknown selection algorithm: {selection_algorithm}")
        frames.append(selected)

    samples = pd.concat(frames, ignore_index=True)
    samples.insert(0, "regime", output_regime)
    samples.insert(1, "selection_regime", output_regime)
    samples.insert(2, "selection_algorithm", selection_algorithm)
    samples.insert(3, "population_family", samples["distribution"])
    samples.insert(
        4,
        "population_symmetry",
        samples["distribution"].map(POPULATION_SYMMETRY).fillna("unknown"),
    )
    samples.insert(5, "global_sample_id", "")
    samples["replicate_id"] = 0
    distribution_counters: dict[str, int] = {}
    global_ids = []
    replicate_ids = []
    for _, row in samples.iterrows():
        distribution = str(row["distribution"])
        index = distribution_counters.get(distribution, 0)
        distribution_counters[distribution] = index + 1
        replicate_ids.append(index)
        global_ids.append(f"{output_regime}_N{n_value}_{distribution}_{index:04d}")
    samples["global_sample_id"] = global_ids
    samples["replicate_id"] = replicate_ids
    return samples


def _correlation_summary_records(
    correlations: dict[tuple, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for key, matrix in sorted(correlations.items()):
        if len(key) == 2:
            regime, n_value = key
            distribution = None
        elif len(key) == 3:
            regime, distribution, n_value = key
        else:
            raise ValueError(f"Unsupported correlation key: {key}")
        for variable_1 in matrix.index:
            for variable_2 in matrix.columns:
                row = {
                    "regime": regime,
                    "N": n_value,
                    "variable_1": variable_1,
                    "variable_2": variable_2,
                    "correlation": float(matrix.loc[variable_1, variable_2]),
                }
                if distribution is not None:
                    row["distribution"] = distribution
                rows.append(row)
    return pd.DataFrame(rows)


def _run_protocol(
    config: PipelineConfig,
    output_root: Path,
    distributions: list[str],
    sample_sizes: list[int],
    n_selected: int,
    metadata: dict[str, Any],
    regimes: list[tuple[str, str]],
) -> dict[str, Any]:
    directories = ensure_output_tree(output_root)
    logger = setup_run_logger(output_root)
    rng = np.random.default_rng(config.seed)
    result_matrices: dict[tuple[str, int], pd.DataFrame] = {}
    result_matrices_by_distribution: dict[tuple[str, str, int], pd.DataFrame] = {}
    correlations: dict[tuple[str, int], pd.DataFrame] = {}
    correlations_by_distribution: dict[tuple[str, str, int], pd.DataFrame] = {}

    for n_value in sample_sizes:
        for selection_algorithm, output_regime in regimes:
            logger.info("Selecting %s samples for N=%s", output_regime, n_value)
            samples = _assemble_regime_samples(
                selection_algorithm,
                output_regime,
                n_value,
                distributions,
                n_selected,
                rng,
                config,
                metadata,
            )
            sample_path = (
                directories["samples"] / f"{output_regime}_N{n_value}_samples.parquet"
            )
            save_parquet(_prepare_samples_for_saving(samples, config), sample_path)

            logger.info("Building result matrix for %s N=%s", output_regime, n_value)
            result_matrix = build_result_matrix(samples, config)
            result_matrix_with_metadata = attach_result_metadata(
                samples, result_matrix
            )
            result_matrices[(output_regime, n_value)] = result_matrix
            if config.outputs.save_result_matrices:
                save_dataframe_csv_parquet(
                    result_matrix,
                    directories["result_matrices"]
                    / f"{output_regime}_N{n_value}_result_matrix.csv",
                    directories["result_matrices"]
                    / f"{output_regime}_N{n_value}_result_matrix.parquet",
                )
                save_dataframe_csv_parquet(
                    result_matrix_with_metadata,
                    directories["result_matrices"]
                    / f"{output_regime}_N{n_value}_result_matrix_with_metadata.csv",
                    directories["result_matrices"]
                    / f"{output_regime}_N{n_value}_result_matrix_with_metadata.parquet",
                )

            correlation = compute_correlation_matrix(result_matrix)
            correlations[(output_regime, n_value)] = correlation
            correlation.to_csv(
                directories["correlations"]
                / f"{output_regime}_N{n_value}_correlation.csv"
            )
            correlation.to_excel(
                directories["correlations"]
                / f"{output_regime}_N{n_value}_correlation.xlsx"
            )
            if config.outputs.save_figures:
                plot_correlation_heatmap(
                    correlation,
                    directories["figures"]
                    / f"correlation_{output_regime}_N{n_value}.png",
                )
                plot_correlation_heatmap(
                    correlation,
                    directories["figures"]
                    / f"correlation_{output_regime}_N{n_value}.pdf",
                )

            for distribution, distribution_samples in samples.groupby("distribution"):
                distribution_matrix = build_result_matrix(distribution_samples, config)
                result_matrices_by_distribution[
                    (output_regime, str(distribution), n_value)
                ] = distribution_matrix
                if config.outputs.save_result_matrices:
                    save_dataframe_csv_parquet(
                        distribution_matrix,
                        directories["result_matrices_by_distribution"]
                        / (
                            f"{output_regime}_{distribution}_N{n_value}"
                            "_result_matrix.csv"
                        ),
                        directories["result_matrices_by_distribution"]
                        / (
                            f"{output_regime}_{distribution}_N{n_value}"
                            "_result_matrix.parquet"
                        ),
                    )
                distribution_correlation = compute_correlation_matrix(
                    distribution_matrix
                )
                correlations_by_distribution[
                    (output_regime, str(distribution), n_value)
                ] = distribution_correlation
                distribution_correlation.to_csv(
                    directories["correlations_by_distribution"]
                    / f"{output_regime}_{distribution}_N{n_value}_correlation.csv"
                )

    warnings = metadata.setdefault("warnings", [])
    direct_regressions = build_direct_kurtosis_regressions(result_matrices, warnings)
    inverse_regressions = build_inverse_half_sum_regressions(result_matrices, warnings)
    direct_regressions_by_distribution = build_direct_kurtosis_regressions(
        result_matrices_by_distribution, warnings
    )
    inverse_regressions_by_distribution = build_inverse_half_sum_regressions(
        result_matrices_by_distribution, warnings
    )
    direct_regressions.to_csv(
        directories["regressions"] / "direct_kurtosis_regressions.csv", index=False
    )
    inverse_regressions.to_csv(
        directories["regressions"] / "inverse_half_sum_regressions.csv", index=False
    )
    direct_regressions_by_distribution.to_csv(
        directories["regressions_by_distribution"]
        / "direct_kurtosis_regressions_by_distribution.csv",
        index=False,
    )
    inverse_regressions_by_distribution.to_csv(
        directories["regressions_by_distribution"]
        / "inverse_half_sum_regressions_by_distribution.csv",
        index=False,
    )

    if config.validation.run_analytic_half_sum_checks:
        validate_half_sum_identity_config(config.statistics)
        analytic_checks = build_half_sum_analytic_checks(
            result_matrices, inverse_regressions, config.statistics
        )
        analytic_checks_by_distribution = build_half_sum_analytic_checks(
            result_matrices_by_distribution,
            inverse_regressions_by_distribution,
            config.statistics,
        )
        tolerance = config.validation.tolerance
        for _, row in pd.concat(
            [analytic_checks, analytic_checks_by_distribution], ignore_index=True
        ).iterrows():
            if (
                row["skew_max_abs_error"] > tolerance
                or row["kurtosis_max_abs_error"] > tolerance
            ):
                warnings.append(
                    "Half-sum identity error above tolerance for "
                    f"{row['regime']} N={row['N']}: "
                    f"skew={row['skew_max_abs_error']}, "
                    f"kurtosis={row['kurtosis_max_abs_error']}"
                )
    else:
        analytic_checks = pd.DataFrame()
        analytic_checks_by_distribution = pd.DataFrame()
    analytic_checks.to_csv(
        directories["analytic"] / "half_sum_analytic_checks.csv", index=False
    )
    analytic_checks_by_distribution.to_csv(
        directories["analytic"] / "half_sum_analytic_checks_by_distribution.csv",
        index=False,
    )

    correlation_summary = _correlation_summary_records(correlations)
    correlation_summary_by_distribution = _correlation_summary_records(
        correlations_by_distribution
    )
    correlation_summary_by_distribution.to_csv(
        directories["correlations_by_distribution"]
        / "correlation_summary_by_distribution.csv",
        index=False,
    )
    selection_metadata = pd.DataFrame(metadata.get("selection_metadata", []))

    if config.outputs.save_excel_workbook:
        config_dict = config_to_dict(config)
        write_summary_workbook(
            directories["regressions"] / "regression_summary.xlsx",
            config_dict,
            direct_regressions,
            inverse_regressions,
            correlation_summary,
            analytic_checks,
            selection_metadata,
            direct_regressions_by_distribution,
            inverse_regressions_by_distribution,
            analytic_checks_by_distribution,
            correlation_summary_by_distribution,
        )
        write_summary_workbook(
            directories["workbook"] / "full_results.xlsx",
            config_dict,
            direct_regressions,
            inverse_regressions,
            correlation_summary,
            analytic_checks,
            selection_metadata,
            direct_regressions_by_distribution,
            inverse_regressions_by_distribution,
            analytic_checks_by_distribution,
            correlation_summary_by_distribution,
        )

    return {
        "result_matrices": result_matrices,
        "result_matrices_by_distribution": result_matrices_by_distribution,
        "correlations": correlations,
        "correlations_by_distribution": correlations_by_distribution,
        "direct_regressions": direct_regressions,
        "inverse_regressions": inverse_regressions,
        "direct_regressions_by_distribution": direct_regressions_by_distribution,
        "inverse_regressions_by_distribution": inverse_regressions_by_distribution,
        "analytic_checks": analytic_checks,
        "analytic_checks_by_distribution": analytic_checks_by_distribution,
        "correlation_summary": correlation_summary,
        "correlation_summary_by_distribution": correlation_summary_by_distribution,
        "selection_metadata": selection_metadata,
    }


def _new_metadata(
    config: PipelineConfig,
    output_root: Path,
    config_path: str | Path | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": ".".join(map(str, __import__("sys").version_info[:3])),
        "package_version": __version__,
        "seed": config.seed,
        "distribution_parameters": config.distribution_parameters,
        "warnings": [],
        "selection_metadata": [],
        "generation_rejections": {},
        "output_root": str(output_root),
        "config_source": str(config_path) if config_path is not None else None,
    }
    return metadata


def _prepare_metadata_and_config(
    config: PipelineConfig,
    output_root: Path,
    config_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    directories = ensure_output_tree(output_root)
    metadata = _new_metadata(config, output_root, config_path)
    config_used_path = output_root / "config_used.yaml"
    dump_config(config, config_used_path)
    metadata["config_hash"] = compute_file_sha256(config_used_path)
    metadata.update(get_git_info(Path.cwd()))
    return metadata, directories


def _finalize_metadata(
    metadata: dict[str, Any],
    directories: dict[str, Path],
    started: float,
) -> None:
    metadata["runtime_seconds"] = float(time.perf_counter() - started)
    metadata["selection_metadata_records"] = dataframe_to_records(
        pd.DataFrame(metadata.get("selection_metadata", []))
    )
    save_json(metadata, directories["root"] / "metadata.json")


def run_experiment(
    config: PipelineConfig, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Run the full reproducible experiment and write all configured outputs."""

    started = time.perf_counter()
    output_root = Path(config.outputs.root_dir)
    metadata, directories = _prepare_metadata_and_config(
        config, output_root, config_path
    )
    logger = setup_run_logger(output_root)
    logger.info("Starting pairwise shape recovery run at %s", output_root)

    main_results = _run_protocol(
        config,
        output_root,
        config.distributions,
        config.sample_sizes,
        config.n_selected_per_distribution,
        metadata,
        PRIMARY_PROTOCOL_REGIMES,
    )

    if config.extended_protocol.enabled:
        extended_started = time.perf_counter()
        extended_config = deepcopy(config)
        extended_config.outputs.root_dir = str(
            output_root / config.extended_protocol.root_subdir
        )
        extended_n_selected = (
            config.extended_protocol.n_selected_per_distribution
            or config.n_selected_per_distribution
        )
        extended_sample_sizes = (
            config.extended_protocol.sample_sizes or config.sample_sizes
        )
        extended_output_root = Path(extended_config.outputs.root_dir)
        extended_metadata, extended_directories = _prepare_metadata_and_config(
            extended_config, extended_output_root, config_path
        )
        extended_metadata["parent_output_root"] = str(output_root)
        extended_metadata["extended_protocol"] = {"enabled": True}
        try:
            _run_protocol(
                extended_config,
                extended_output_root,
                config.extended_protocol.distributions,
                extended_sample_sizes,
                extended_n_selected,
                extended_metadata,
                EXTENDED_PROTOCOL_REGIMES,
            )
            extended_metadata["status"] = "completed"
        except Exception:
            extended_metadata["status"] = "failed"
            _finalize_metadata(
                extended_metadata, extended_directories, extended_started
            )
            raise
        _finalize_metadata(extended_metadata, extended_directories, extended_started)
        metadata["extended_protocol"] = {
            "enabled": True,
            "root_dir": str(extended_output_root),
            "runtime": extended_metadata["runtime_seconds"],
            "status": extended_metadata["status"],
        }
    else:
        metadata["extended_protocol"] = {
            "enabled": False,
            "root_dir": None,
            "runtime": None,
            "status": "skipped",
        }

    _finalize_metadata(metadata, directories, started)
    logger = setup_run_logger(output_root)
    logger.info("Finished run in %.2f seconds", metadata["runtime_seconds"])

    return {
        "output_root": output_root,
        "metadata": metadata,
        **main_results,
    }
