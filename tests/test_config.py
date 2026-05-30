from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pairwise_shape_recovery.config import (
    PipelineConfig,
    config_to_dict,
    dump_config,
    load_config,
)
from pairwise_shape_recovery.distributions import generate_sample


def test_default_yaml_loads() -> None:
    config = load_config("configs/default.yaml")

    assert config.seed == 20261225
    assert config.distribution_parameters["student_t5"]["df"] == 5.0


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    data = config_to_dict(PipelineConfig())
    data["unknown"] = True
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_nested_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_nested.yaml"
    data = config_to_dict(PipelineConfig())
    data["statistics"]["unknown"] = True
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_distribution_parameter_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_params.yaml"
    data = config_to_dict(PipelineConfig())
    data["distribution_parameters"]["normal"]["bad"] = 1.0
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    ("distribution", "missing_key"),
    [
        ("normal", "scale"),
        ("student_t5", "df"),
        ("gamma", "shape"),
        ("skewnormal", "shape"),
    ],
)
def test_missing_distribution_parameter_raises(
    tmp_path: Path, distribution: str, missing_key: str
) -> None:
    path = tmp_path / "missing_params.yaml"
    data = config_to_dict(PipelineConfig())
    data["distribution_parameters"][distribution].pop(missing_key)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_generate_sample_missing_distribution_parameters_raises() -> None:
    with pytest.raises(ValueError):
        generate_sample(
            "normal",
            10,
            np.random.default_rng(123),
            {"uniform": {"low": 0.0, "high": 1.0}},
        )


@pytest.mark.parametrize(
    "section",
    ["distributions", "extended_protocol.distributions"],
)
def test_invalid_distribution_name_raises(tmp_path: Path, section: str) -> None:
    path = tmp_path / "bad_distribution.yaml"
    data = config_to_dict(PipelineConfig())
    if section == "distributions":
        data["distributions"] = ["normal", "unknown_distribution"]
    else:
        data["extended_protocol"]["distributions"] = ["lognormal", "unknown"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_extended_validation_config_loads() -> None:
    config = load_config("configs/extended_validation.yaml")

    assert config.extended_protocol.enabled is True
    assert config.extended_protocol.distributions == [
        "lognormal",
        "gamma",
        "weibull",
        "skewnormal",
    ]


def test_distribution_parameters_change_generation() -> None:
    config = PipelineConfig()
    changed = config_to_dict(config)["distribution_parameters"]
    changed["uniform"] = {"low": 10.0, "high": 11.0}

    default_sample = generate_sample(
        "uniform",
        20,
        np.random.default_rng(123),
        config.distribution_parameters,
    )
    changed_sample = generate_sample(
        "uniform",
        20,
        np.random.default_rng(123),
        changed,
    )

    assert not np.allclose(default_sample, changed_sample)
    assert np.min(changed_sample) >= 10.0
    assert np.max(changed_sample) <= 11.0


def test_config_dump_load_roundtrip(tmp_path: Path) -> None:
    config = PipelineConfig(seed=99, sample_sizes=[10])
    path = tmp_path / "roundtrip.yaml"

    dump_config(config, path)
    loaded = load_config(path)

    assert loaded.seed == 99
    assert loaded.sample_sizes == [10]
    assert loaded.distribution_parameters == config.distribution_parameters
