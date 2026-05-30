from __future__ import annotations

import numpy as np
import pytest

from pairwise_shape_recovery.transforms import (
    direct_pairwise_differences,
    ordered_pairwise_differences,
    pairwise_half_sums,
)


def test_transform_values_for_n3() -> None:
    x = np.asarray([1.0, 2.0, 4.0])

    np.testing.assert_allclose(ordered_pairwise_differences(x), [2.0, 3.0, 1.0])
    np.testing.assert_allclose(direct_pairwise_differences(x), [-1.0, -3.0, -2.0])
    np.testing.assert_allclose(pairwise_half_sums(x), [1.5, 2.5, 3.0])


def test_direct_differences_are_order_dependent() -> None:
    x = np.asarray([1.0, 2.0, 4.0])
    permuted = np.asarray([4.0, 2.0, 1.0])

    np.testing.assert_allclose(direct_pairwise_differences(x), [-1.0, -3.0, -2.0])
    np.testing.assert_allclose(direct_pairwise_differences(permuted), [2.0, 3.0, 1.0])
    assert not np.allclose(
        direct_pairwise_differences(x),
        direct_pairwise_differences(permuted),
    )


def test_ordered_differences_and_half_sums_are_permutation_invariant_as_sets() -> None:
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    permuted = np.asarray([4.0, 1.0, 8.0, 2.0])

    np.testing.assert_allclose(
        np.sort(ordered_pairwise_differences(x)),
        np.sort(ordered_pairwise_differences(permuted)),
    )
    np.testing.assert_allclose(
        np.sort(pairwise_half_sums(x)),
        np.sort(pairwise_half_sums(permuted)),
    )


def test_transform_lengths_for_protocol_sizes() -> None:
    for n_value in [10, 20, 30, 40, 50]:
        x = np.arange(n_value, dtype=float)
        expected_length = n_value * (n_value - 1) // 2

        assert ordered_pairwise_differences(x).size == expected_length
        assert direct_pairwise_differences(x).size == expected_length
        assert pairwise_half_sums(x).size == expected_length


def test_transform_input_validation() -> None:
    for transform in [
        ordered_pairwise_differences,
        direct_pairwise_differences,
        pairwise_half_sums,
    ]:
        with pytest.raises(ValueError):
            transform(np.asarray([1.0]))
        with pytest.raises(ValueError):
            transform(np.ones((2, 2)))

