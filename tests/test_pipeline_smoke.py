from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pandas as pd
import yaml

from pairwise_shape_recovery.config import PipelineConfig, config_to_dict, load_config
from pairwise_shape_recovery.simulation import RESULT_COLUMNS, run_experiment


def _smoke_config(output_root: Path, extended: bool = False) -> dict:
    config_data = config_to_dict(PipelineConfig())
    config_data.update(
        {
            "seed": 123,
            "sample_sizes": [10],
            "distributions": ["normal"],
            "n_selected_per_distribution": 5,
        }
    )
    config_data["low_skew_selection"].update(
        {
            "skew_abs_threshold": 2.0,
            "initial_pool_size": 50,
            "max_pool_size": 50,
        }
    )
    config_data["moderate_shape_selection"].update(
        {
            "pool_size": 50,
            "decile": 4,
            "n_selected_per_distribution": 5,
        }
    )
    config_data["outputs"].update(
        {
            "root_dir": str(output_root),
            "save_excel_workbook": True,
            "save_figures": False,
        }
    )
    config_data["validation"].update(
        {
            "run_analytic_half_sum_checks": True,
            "tolerance": 1e-10,
        }
    )
    config_data["extended_protocol"].update(
        {
            "enabled": extended,
            "distributions": ["lognormal"],
            "n_selected_per_distribution": 5,
            "sample_sizes": [10],
        }
    )
    return config_data


def test_pipeline_smoke_outputs_and_metadata(tmp_path: Path) -> None:
    output_root = tmp_path / "run_smoke"
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(_smoke_config(output_root), sort_keys=False),
        encoding="utf-8",
    )

    result = run_experiment(load_config(config_path), config_path)

    low_skew_matrix = (
        output_root / "result_matrices" / "low_skew_selected_N10_result_matrix.csv"
    )
    moderate_shape_matrix = (
        output_root
        / "result_matrices"
        / "moderate_shape_selected_N10_result_matrix.csv"
    )
    assert low_skew_matrix.exists()
    assert moderate_shape_matrix.exists()
    assert (output_root / "analytic" / "half_sum_analytic_checks.csv").exists()
    assert (output_root / "regressions" / "inverse_half_sum_regressions.csv").exists()
    assert (output_root / "regressions" / "regression_summary.xlsx").exists()
    assert (output_root / "workbook" / "full_results.xlsx").exists()
    assert (output_root / "config_used.yaml").exists()
    assert (output_root / "logs" / "run.log").exists()
    assert (
        output_root
        / "result_matrices"
        / "by_distribution"
        / "low_skew_selected_normal_N10_result_matrix.csv"
    ).exists()
    assert (
        output_root
        / "correlations"
        / "by_distribution"
        / "low_skew_selected_normal_N10_correlation.csv"
    ).exists()
    assert (
        output_root
        / "regressions"
        / "by_distribution"
        / "direct_kurtosis_regressions_by_distribution.csv"
    ).exists()
    assert (
        output_root
        / "correlations"
        / "by_distribution"
        / "correlation_summary_by_distribution.csv"
    ).exists()

    matrix = pd.read_csv(low_skew_matrix)
    matrix_with_metadata = pd.read_csv(
        output_root
        / "result_matrices"
        / "low_skew_selected_N10_result_matrix_with_metadata.csv"
    )
    samples = pd.read_parquet(
        output_root / "samples" / "low_skew_selected_N10_samples.parquet"
    )
    assert matrix.shape == (5, 8)
    assert list(matrix.columns) == RESULT_COLUMNS
    assert matrix_with_metadata.shape[0] == matrix.shape[0]
    assert (
        matrix_with_metadata["global_sample_id"].tolist()
        == samples["global_sample_id"].tolist()
    )
    assert list(matrix_with_metadata[RESULT_COLUMNS].columns) == RESULT_COLUMNS
    assert pd.read_csv(moderate_shape_matrix).shape == (5, 8)
    assert pd.read_csv(
        output_root / "correlations" / "low_skew_selected_N10_correlation.csv",
        index_col=0,
    ).shape == (8, 8)

    analytic = pd.read_csv(output_root / "analytic" / "half_sum_analytic_checks.csv")
    assert analytic["skew_max_abs_error"].max() < 1e-10
    assert analytic["kurtosis_max_abs_error"].max() < 1e-10

    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    for key in [
        "seed",
        "config_hash",
        "runtime_seconds",
        "selection_metadata_records",
        "generation_rejections",
        "warnings",
        "distribution_parameters",
    ]:
        assert key in metadata
    assert metadata["seed"] == 123
    assert metadata["extended_protocol"]["status"] == "skipped"
    assert metadata["git_commit_hash"] is None or isinstance(
        metadata["git_commit_hash"], str
    )
    assert "git_available" in metadata
    assert "git_error" in metadata
    assert result["metadata"]["seed"] == 123

    workbook = openpyxl.load_workbook(
        output_root / "workbook" / "full_results.xlsx", read_only=True
    )
    assert all(len(sheet_name) <= 31 for sheet_name in workbook.sheetnames)
    assert "corr_by_dist" in workbook.sheetnames
    config_sheet_values = [
        row[0] for row in workbook["config"].iter_rows(min_row=2, values_only=True)
    ]
    assert "seed" in config_sheet_values
    assert "distribution_parameters.normal.loc" in config_sheet_values
    assert "extended_protocol.enabled" in config_sheet_values
    assert "sample_sizes" in config_sheet_values


def test_extended_validation_smoke_uses_separate_metadata(tmp_path: Path) -> None:
    output_root = tmp_path / "run_extended_smoke"
    config_path = tmp_path / "extended_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(_smoke_config(output_root, extended=True), sort_keys=False),
        encoding="utf-8",
    )

    run_experiment(load_config(config_path), config_path)

    extended_root = output_root / "extended_protocol"
    assert (extended_root / "metadata.json").exists()
    assert (extended_root / "config_used.yaml").exists()
    assert (
        extended_root / "result_matrices" / "low_skew_selected_N10_result_matrix.csv"
    ).exists()
    assert (
        extended_root
        / "result_matrices"
        / "by_distribution"
        / "low_skew_selected_lognormal_N10_result_matrix.csv"
    ).exists()
    assert (extended_root / "analytic" / "half_sum_analytic_checks.csv").exists()

    main_metadata = json.loads(
        (output_root / "metadata.json").read_text(encoding="utf-8")
    )
    extended_metadata = json.loads(
        (extended_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert main_metadata["extended_protocol"]["status"] == "completed"
    assert main_metadata["extended_protocol"]["root_dir"] == str(extended_root)
    assert {
        record["distribution_name"]
        for record in main_metadata["selection_metadata_records"]
    } == {"normal"}
    assert {
        record["distribution_name"]
        for record in extended_metadata["selection_metadata_records"]
    } == {"lognormal"}
    assert {
        record["selection_regime"]
        for record in extended_metadata["selection_metadata_records"]
    } == {"low_skew_selected", "moderate_shape_selected"}
    assert {
        record["regime"] for record in extended_metadata["selection_metadata_records"]
    } == {"low_skew_selected", "moderate_shape_selected"}


def test_main_and_extended_outputs_do_not_overlap_for_different_families(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "run_non_overlap"
    config_path = tmp_path / "non_overlap.yaml"
    config_data = _smoke_config(output_root, extended=True)
    config_data["sample_sizes"] = [50]
    config_data["extended_protocol"]["sample_sizes"] = [50]
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )

    run_experiment(load_config(config_path), config_path)

    main_path = (
        output_root / "result_matrices" / "low_skew_selected_N50_result_matrix.csv"
    )
    extended_path = (
        output_root
        / "extended_protocol"
        / "result_matrices"
        / "low_skew_selected_N50_result_matrix.csv"
    )
    assert main_path.exists()
    assert extended_path.exists()
    assert main_path != extended_path
    assert not pd.read_csv(main_path).equals(pd.read_csv(extended_path))

    main_metadata = json.loads(
        (output_root / "metadata.json").read_text(encoding="utf-8")
    )
    extended_metadata = json.loads(
        (output_root / "extended_protocol" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        record["distribution_name"]
        for record in main_metadata["selection_metadata_records"]
    } == {"normal"}
    assert {
        record["distribution_name"]
        for record in extended_metadata["selection_metadata_records"]
    } == {"lognormal"}
