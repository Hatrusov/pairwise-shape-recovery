"""Input/output helpers for reproducible experiment runs."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_output_tree(root: str | Path) -> dict[str, Path]:
    """Create and return all standard output directories."""

    root_path = Path(root)
    directories = {
        "root": root_path,
        "samples": root_path / "samples",
        "result_matrices": root_path / "result_matrices",
        "result_matrices_by_distribution": root_path
        / "result_matrices"
        / "by_distribution",
        "correlations": root_path / "correlations",
        "correlations_by_distribution": root_path / "correlations" / "by_distribution",
        "regressions": root_path / "regressions",
        "regressions_by_distribution": root_path / "regressions" / "by_distribution",
        "analytic": root_path / "analytic",
        "figures": root_path / "figures",
        "workbook": root_path / "workbook",
        "logs": root_path / "logs",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""

    return json.loads(df.to_json(orient="records"))


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=True)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save Parquet with a clear dependency message when pyarrow is missing."""

    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Parquet output requires pyarrow. Install with "
            "`python -m pip install -e .[dev]` or install `pyarrow`."
        ) from exc


def save_dataframe_csv_parquet(
    df: pd.DataFrame, csv_path: str | Path, parquet_path: str | Path
) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    save_parquet(df, parquet_path)


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_info(repo_root: str | Path) -> dict[str, Any]:
    """Return git provenance without treating missing git as a runtime warning."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - depends on local git setup
        return {
            "git_commit_hash": None,
            "git_available": False,
            "git_error": str(exc),
        }
    return {
        "git_commit_hash": completed.stdout.strip(),
        "git_available": True,
        "git_error": None,
    }


def setup_run_logger(output_root: str | Path) -> logging.Logger:
    """Create a run logger that writes to logs/run.log."""

    logger = logging.getLogger("pairwise_shape_recovery")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path = Path(output_root) / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)
    return logger


def flatten_config(config: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    """Flatten nested config dicts for workbook export."""

    rows: list[dict[str, Any]] = []
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(flatten_config(value, full_key))
        else:
            rows.append(
                {"key": full_key, "value": json.dumps(value, ensure_ascii=False)}
            )
    return rows


def write_summary_workbook(
    path: str | Path,
    config: dict[str, Any],
    direct_regressions: pd.DataFrame,
    inverse_regressions: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    analytic_checks: pd.DataFrame,
    selection_metadata: pd.DataFrame,
    direct_regressions_by_distribution: pd.DataFrame | None = None,
    inverse_regressions_by_distribution: pd.DataFrame | None = None,
    analytic_checks_by_distribution: pd.DataFrame | None = None,
    correlation_summary_by_distribution: pd.DataFrame | None = None,
) -> None:
    """Write the requested multi-sheet XLSX workbook."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(flatten_config(config)).to_excel(
            writer, sheet_name="config", index=False
        )
        direct_regressions.to_excel(
            writer, sheet_name="direct_kurtosis_regressions", index=False
        )
        inverse_regressions.to_excel(
            writer, sheet_name="inverse_half_sum_regressions", index=False
        )
        correlation_summary.to_excel(
            writer, sheet_name="correlation_summary", index=False
        )
        analytic_checks.to_excel(
            writer, sheet_name="analytic_half_sum_checks", index=False
        )
        selection_metadata.to_excel(
            writer, sheet_name="selection_metadata", index=False
        )
        if direct_regressions_by_distribution is not None:
            direct_regressions_by_distribution.to_excel(
                writer,
                sheet_name="direct_by_dist",
                index=False,
            )
        if inverse_regressions_by_distribution is not None:
            inverse_regressions_by_distribution.to_excel(
                writer,
                sheet_name="inverse_by_dist",
                index=False,
            )
        if analytic_checks_by_distribution is not None:
            analytic_checks_by_distribution.to_excel(
                writer,
                sheet_name="analytic_by_dist",
                index=False,
            )
        if correlation_summary_by_distribution is not None:
            correlation_summary_by_distribution.to_excel(
                writer,
                sheet_name="corr_by_dist",
                index=False,
            )
