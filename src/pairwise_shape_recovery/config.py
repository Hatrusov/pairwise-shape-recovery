"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DISTRIBUTION_PARAMETERS: dict[str, dict[str, float]] = {
    "uniform": {"low": -1.7320508075688772, "high": 1.7320508075688772},
    "normal": {"loc": 0.0, "scale": 1.0},
    "cauchy": {"loc": 0.0, "scale": 1.0},
    "student_t5": {"df": 5.0},
    "lognormal": {"mean": 0.0, "sigma": 0.5},
    "gamma": {"shape": 2.0, "scale": 1.0},
    "weibull": {"shape": 1.5},
    "skewnormal": {"shape": 8.0, "loc": 0.0, "scale": 1.0},
}

ALLOWED_DISTRIBUTION_PARAMETER_KEYS: dict[str, set[str]] = {
    "uniform": {"low", "high"},
    "normal": {"loc", "scale"},
    "cauchy": {"loc", "scale"},
    "student_t5": {"df"},
    "lognormal": {"mean", "sigma"},
    "gamma": {"shape", "scale"},
    "weibull": {"shape"},
    "skewnormal": {"shape", "loc", "scale"},
}


@dataclass(slots=True)
class StatisticsConfig:
    """Moment estimator choices used across the protocol."""

    sd_mode: str = "population"
    skew_mode: str = "adjusted_fisher_pearson"
    kurtosis_mode: str = "standardized_fourth_moment_no_minus_3"


@dataclass(slots=True)
class LowSkewSelectionConfig:
    """Selection settings for finite samples with low sample skewness."""

    skew_abs_threshold: float = 0.1
    initial_pool_size: int = 1000
    max_pool_size: int = 200000
    strategy: str = "sort_by_abs_skew_then_take_first_satisfying_threshold"


@dataclass(slots=True)
class ModerateShapeSelectionConfig:
    """Selection settings based on a finite-sample shape score."""

    pool_size: int = 1000
    decile: int = 4
    decile_indexing: str = "one_based"
    score: str = "abs_skew_times_kurtosis"
    n_selected_per_distribution: int = 100


@dataclass(slots=True)
class OutputConfig:
    """Output locations and file switches."""

    root_dir: str = "outputs/primary_protocol"
    save_raw_samples: bool = True
    save_normalized_samples: bool = True
    save_derived_sequences: bool = False
    save_result_matrices: bool = True
    save_excel_workbook: bool = True
    save_figures: bool = True


@dataclass(slots=True)
class ValidationConfig:
    """Validation settings."""

    run_analytic_half_sum_checks: bool = True
    tolerance: float = 1e-10


@dataclass(slots=True)
class ExtendedProtocolConfig:
    """Optional protocol on additional distribution families."""

    enabled: bool = False
    root_subdir: str = "extended_protocol"
    distributions: list[str] = field(
        default_factory=lambda: ["lognormal", "gamma", "weibull", "skewnormal"]
    )
    n_selected_per_distribution: int | None = None
    sample_sizes: list[int] | None = None


@dataclass(slots=True)
class PipelineConfig:
    """Top-level experiment configuration."""

    seed: int = 20261225
    sample_sizes: list[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    distributions: list[str] = field(
        default_factory=lambda: ["uniform", "normal", "cauchy", "student_t5"]
    )
    distribution_parameters: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            name: dict(parameters)
            for name, parameters in DEFAULT_DISTRIBUTION_PARAMETERS.items()
        }
    )
    n_selected_per_distribution: int = 100
    low_skew_selection: LowSkewSelectionConfig = field(
        default_factory=LowSkewSelectionConfig
    )
    moderate_shape_selection: ModerateShapeSelectionConfig = field(
        default_factory=ModerateShapeSelectionConfig
    )
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    extended_protocol: ExtendedProtocolConfig = field(
        default_factory=ExtendedProtocolConfig
    )


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _filter_dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")
    return {key: values[key] for key in allowed if key in values}


def _parse_float(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "sqrt(3)":
            return 3.0**0.5
        if lowered == "-sqrt(3)":
            return -(3.0**0.5)
    return float(value)


def _coerce_distribution_parameters(
    values: dict[str, Any]
) -> dict[str, dict[str, float]]:
    parameters: dict[str, dict[str, float]] = {}
    for distribution_name, raw_params in values.items():
        if distribution_name not in ALLOWED_DISTRIBUTION_PARAMETER_KEYS:
            raise ValueError(
                f"Unknown distribution in distribution_parameters: {distribution_name}"
            )
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"Parameters for {distribution_name} must be a mapping."
            )
        allowed = ALLOWED_DISTRIBUTION_PARAMETER_KEYS[distribution_name]
        unknown = sorted(set(raw_params) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown parameters for {distribution_name}: {unknown}"
            )
        missing = sorted(allowed - set(raw_params))
        if missing:
            raise ValueError(
                f"Missing parameters for {distribution_name}: {missing}"
            )
        parameters[distribution_name] = {
            key: _parse_float(value) for key, value in raw_params.items()
        }
    return parameters


def _validate_distribution_names(names: list[str], context: str) -> list[str]:
    supported = set(ALLOWED_DISTRIBUTION_PARAMETER_KEYS)
    unknown = sorted(set(names) - supported)
    if unknown:
        raise ValueError(f"Unknown distribution names in {context}: {unknown}")
    return names


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Return a JSON/YAML-serializable dict representation."""

    return asdict(config)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config and merge it over dataclass defaults."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults = config_to_dict(PipelineConfig())
    unknown_top_level = sorted(set(raw) - set(defaults))
    if unknown_top_level:
        raise ValueError(f"Unknown top-level config keys: {unknown_top_level}")
    if "distribution_parameters" in raw:
        _coerce_distribution_parameters(raw["distribution_parameters"])
    merged = _deep_update(defaults, raw)

    stats_values = _filter_dataclass_kwargs(StatisticsConfig, merged["statistics"])
    low_skew_values = _filter_dataclass_kwargs(
        LowSkewSelectionConfig, merged["low_skew_selection"]
    )
    moderate_shape_values = _filter_dataclass_kwargs(
        ModerateShapeSelectionConfig, merged["moderate_shape_selection"]
    )
    output_values = _filter_dataclass_kwargs(OutputConfig, merged["outputs"])
    validation_values = _filter_dataclass_kwargs(ValidationConfig, merged["validation"])
    extended_values = _filter_dataclass_kwargs(
        ExtendedProtocolConfig, merged["extended_protocol"]
    )

    stats = StatisticsConfig(**stats_values)
    low_skew = LowSkewSelectionConfig(
        skew_abs_threshold=float(low_skew_values["skew_abs_threshold"]),
        initial_pool_size=int(low_skew_values["initial_pool_size"]),
        max_pool_size=int(low_skew_values["max_pool_size"]),
        strategy=str(low_skew_values["strategy"]),
    )
    moderate_shape = ModerateShapeSelectionConfig(
        pool_size=int(moderate_shape_values["pool_size"]),
        decile=int(moderate_shape_values["decile"]),
        decile_indexing=str(moderate_shape_values["decile_indexing"]),
        score=str(moderate_shape_values["score"]),
        n_selected_per_distribution=int(
            moderate_shape_values["n_selected_per_distribution"]
        ),
    )
    outputs = OutputConfig(
        root_dir=str(output_values["root_dir"]),
        save_raw_samples=bool(output_values["save_raw_samples"]),
        save_normalized_samples=bool(output_values["save_normalized_samples"]),
        save_derived_sequences=bool(output_values["save_derived_sequences"]),
        save_result_matrices=bool(output_values["save_result_matrices"]),
        save_excel_workbook=bool(output_values["save_excel_workbook"]),
        save_figures=bool(output_values["save_figures"]),
    )
    validation = ValidationConfig(
        run_analytic_half_sum_checks=bool(
            validation_values["run_analytic_half_sum_checks"]
        ),
        tolerance=float(validation_values["tolerance"]),
    )
    extended_protocol = ExtendedProtocolConfig(
        enabled=bool(extended_values["enabled"]),
        root_subdir=str(extended_values["root_subdir"]),
        distributions=_validate_distribution_names(
            [str(value) for value in extended_values["distributions"]],
            "extended_protocol.distributions",
        ),
        n_selected_per_distribution=(
            None
            if extended_values.get("n_selected_per_distribution") is None
            else int(extended_values["n_selected_per_distribution"])
        ),
        sample_sizes=(
            None
            if extended_values.get("sample_sizes") is None
            else [int(value) for value in extended_values["sample_sizes"]]
        ),
    )

    return PipelineConfig(
        seed=int(merged["seed"]),
        sample_sizes=[int(value) for value in merged["sample_sizes"]],
        distributions=_validate_distribution_names(
            [str(value) for value in merged["distributions"]],
            "distributions",
        ),
        distribution_parameters=_coerce_distribution_parameters(
            merged["distribution_parameters"]
        ),
        n_selected_per_distribution=int(merged["n_selected_per_distribution"]),
        low_skew_selection=low_skew,
        moderate_shape_selection=moderate_shape,
        statistics=stats,
        outputs=outputs,
        validation=validation,
        extended_protocol=extended_protocol,
    )


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    """Write a configuration to YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config_to_dict(config),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
