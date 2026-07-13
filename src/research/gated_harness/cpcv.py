"""Combinatorial purged cross-validation for path-dependent research."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import numpy as np

from .purging import purge_train_indices


def combinatorial_purged_splits(
    n_observations: int,
    *,
    n_groups: int = 5,
    n_test_groups: int = 2,
    purge_gap: int = 0,
    embargo_gap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield every declared test-group combination with adjacent rows removed."""
    if n_groups < 2 or not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups)")
    if n_observations < n_groups:
        raise ValueError("n_observations must cover every CPCV group")
    all_indices = np.arange(n_observations, dtype=int)
    groups = np.array_split(all_indices, n_groups)
    for chosen in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([groups[index] for index in chosen])
        train = all_indices[~np.isin(all_indices, test)]
        yield purge_train_indices(
            train, test, purge_gap=purge_gap, embargo_gap=embargo_gap
        ), test
