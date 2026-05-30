"""Command-line interface for pairwise-shape-recovery."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from pairwise_shape_recovery.analytic import (
    quick_half_sum_identity_check,
    validate_half_sum_identity_config,
)
from pairwise_shape_recovery.config import load_config
from pairwise_shape_recovery.io import ensure_output_tree
from pairwise_shape_recovery.simulation import run_experiment
from pairwise_shape_recovery.statistics import normalize_sample


def _find_project_root(start: Path | None = None) -> Path:
    """Find the nearest project root containing pyproject.toml."""

    search_roots = [start or Path.cwd(), Path(__file__).resolve()]
    for root in search_roots:
        current = root if root.is_dir() else root.parent
        for candidate in [current, *current.parents]:
            if (candidate / "pyproject.toml").exists():
                return candidate
    return Path.cwd()


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output is not None:
        config.outputs.root_dir = args.output
    result = run_experiment(config, args.config)
    metadata = result["metadata"]
    print("Run complete")
    print(f"Output directory: {result['output_root']}")
    print(f"Runtime seconds: {metadata['runtime_seconds']:.2f}")
    print(f"Warnings: {len(metadata.get('warnings', []))}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_half_sum_identity_config(config.statistics)
    rng = np.random.default_rng(config.seed)
    tolerance = config.validation.tolerance
    max_skew_error = 0.0
    max_kurtosis_error = 0.0
    for n_value in config.sample_sizes:
        for _ in range(5):
            x = rng.normal(size=n_value)
            x_norm = normalize_sample(x, config.statistics)
            errors = quick_half_sum_identity_check(x_norm, config.statistics)
            max_skew_error = max(max_skew_error, errors["skew_abs_error"])
            max_kurtosis_error = max(max_kurtosis_error, errors["kurtosis_abs_error"])
    print(f"Max skew identity error: {max_skew_error:.3e}")
    print(f"Max kurtosis identity error: {max_kurtosis_error:.3e}")
    if max_skew_error > tolerance * 10 or max_kurtosis_error > tolerance * 10:
        raise SystemExit("Validation failed: analytical identity error is too large.")
    print("Validation passed")
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    project_root = Path(args.cwd).resolve() if args.cwd else _find_project_root()
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = project_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_output_tree(project_root / "outputs")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    report = completed.stdout
    if completed.stderr:
        report = f"{report}\n{completed.stderr}"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="" if report.endswith("\n") else "\n")
    print(f"Pytest report saved to {report_path}")
    return int(completed.returncode)


def _cmd_clean(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if not output.exists():
        print(f"Output directory does not exist: {output}")
        return 0
    answer = input(f"Delete output directory {output}? Type 'yes' to confirm: ")
    if answer.strip().lower() != "yes":
        print("Clean cancelled")
        return 1
    shutil.rmtree(output)
    print(f"Deleted {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pairwise-shape-recovery")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full experiment.")
    run_parser.add_argument("--config", required=True, help="Path to YAML config.")
    run_parser.add_argument("--output", help="Optional output directory override.")
    run_parser.set_defaults(func=_cmd_run)

    validate_parser = subparsers.add_parser(
        "validate", help="Run fast analytical checks."
    )
    validate_parser.add_argument("--config", required=True, help="Path to YAML config.")
    validate_parser.set_defaults(func=_cmd_validate)

    test_parser = subparsers.add_parser(
        "test", help="Run pytest -q and save the report."
    )
    test_parser.add_argument(
        "--report",
        default="outputs/logs/test_report.txt",
        help="Path for pytest report output.",
    )
    test_parser.add_argument(
        "--cwd",
        help="Project root to run pytest from. Defaults to pyproject.toml root.",
    )
    test_parser.set_defaults(func=_cmd_test)

    clean_parser = subparsers.add_parser("clean", help="Delete an output directory.")
    clean_parser.add_argument(
        "--output", required=True, help="Output directory to delete."
    )
    clean_parser.set_defaults(func=_cmd_clean)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
