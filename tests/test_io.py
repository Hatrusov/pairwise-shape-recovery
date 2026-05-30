from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pandas as pd

from pairwise_shape_recovery.io import (
    compute_file_sha256,
    ensure_output_tree,
    save_dataframe_csv_parquet,
    save_json,
    write_summary_workbook,
)


def test_ensure_output_tree_creates_directories(tmp_path: Path) -> None:
    directories = ensure_output_tree(tmp_path / "run")

    for directory in directories.values():
        assert directory.exists()
        assert directory.is_dir()


def test_save_json_writes_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    save_json({"seed": 123}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"seed": 123}


def test_save_dataframe_csv_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "data.csv"
    parquet_path = tmp_path / "data.parquet"

    save_dataframe_csv_parquet(df, csv_path, parquet_path)

    assert csv_path.exists()
    assert parquet_path.exists()
    assert pd.read_csv(csv_path).shape == (2, 2)
    assert pd.read_parquet(parquet_path).shape == (2, 2)


def test_write_summary_workbook_sheets(tmp_path: Path) -> None:
    path = tmp_path / "summary.xlsx"
    df = pd.DataFrame({"a": [1.0]})

    write_summary_workbook(
        path,
        {"seed": 123},
        df,
        df,
        df,
        df,
        df,
        df,
        df,
        df,
        df,
    )

    workbook = openpyxl.load_workbook(path, read_only=True)
    assert {
        "config",
        "direct_kurtosis_regressions",
        "inverse_half_sum_regressions",
        "correlation_summary",
        "analytic_half_sum_checks",
        "selection_metadata",
        "direct_by_dist",
        "inverse_by_dist",
        "analytic_by_dist",
        "corr_by_dist",
    }.issubset(set(workbook.sheetnames))
    assert all(len(sheet_name) <= 31 for sheet_name in workbook.sheetnames)


def test_compute_file_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("abc", encoding="utf-8")

    assert compute_file_sha256(path) == compute_file_sha256(path)
