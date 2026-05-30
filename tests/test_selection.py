from __future__ import annotations

import numpy as np
import pytest

from pairwise_shape_recovery.config import (
    LowSkewSelectionConfig,
    ModerateShapeSelectionConfig,
    PipelineConfig,
    StatisticsConfig,
)
from pairwise_shape_recovery.selection import (
    select_low_skew_samples,
    select_moderate_shape_samples,
)


def test_low_skew_selection_metadata_and_threshold() -> None:
    metadata: dict = {}
    config = PipelineConfig()
    selected = select_low_skew_samples(
        "normal",
        10,
        5,
        np.random.default_rng(123),
        LowSkewSelectionConfig(
            skew_abs_threshold=2.0,
            initial_pool_size=20,
            max_pool_size=20,
        ),
        StatisticsConfig(),
        config.distribution_parameters,
        metadata,
    )

    assert len(selected) == 5
    assert (selected["abs_skew"] <= 2.0).all()
    assert selected["selection_rank"].tolist() == [1, 2, 3, 4, 5]
    record = metadata["selection_metadata"][0]
    for key in [
        "generated_candidates",
        "valid_candidates",
        "passed_threshold",
        "selected",
    ]:
        assert key in record
    assert record["selected"] == 5


def test_moderate_shape_selection_decile_and_scores() -> None:
    metadata: dict = {}
    config = PipelineConfig()
    selected = select_moderate_shape_samples(
        "normal",
        10,
        5,
        np.random.default_rng(123),
        ModerateShapeSelectionConfig(pool_size=20, decile=2),
        StatisticsConfig(),
        config.distribution_parameters,
        metadata,
    )

    assert len(selected) == 5
    np.testing.assert_allclose(
        selected["selection_score"],
        selected["abs_skew"] * selected["kurtosis_norm"],
    )
    assert selected["selection_rank"].tolist() == [6, 7, 8, 9, 10]
    record = metadata["selection_metadata"][0]
    assert record["start_index_zero_based"] == 5
    assert record["stop_index_zero_based_exclusive"] == 10


def test_moderate_shape_selection_invalid_config() -> None:
    config = PipelineConfig()

    with pytest.raises(ValueError):
        select_moderate_shape_samples(
            "normal",
            10,
            5,
            np.random.default_rng(123),
            ModerateShapeSelectionConfig(decile_indexing="zero_based"),
            StatisticsConfig(),
            config.distribution_parameters,
        )
    with pytest.raises(ValueError):
        select_moderate_shape_samples(
            "normal",
            10,
            5,
            np.random.default_rng(123),
            ModerateShapeSelectionConfig(score="unknown"),
            StatisticsConfig(),
            config.distribution_parameters,
        )
