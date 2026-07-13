"""Purging and embargo primitives for overlapping financial labels."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def purge_train_indices(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    purge_gap: int,
    embargo_gap: int = 0,
) -> np.ndarray:
    """Remove training observations near every contiguous test interval."""
    if purge_gap < 0 or embargo_gap < 0:
        raise ValueError("purge and embargo gaps must be non-negative")
    train = np.asarray(train_indices, dtype=int)
    test = np.sort(np.asarray(test_indices, dtype=int))
    if test.size == 0:
        raise ValueError("test_indices cannot be empty")
    blocked = np.zeros(train.shape, dtype=bool)
    boundaries = np.flatnonzero(np.diff(test) > 1) + 1
    for group in np.split(test, boundaries):
        lo = int(group[0]) - purge_gap
        hi = int(group[-1]) + purge_gap + embargo_gap
        blocked |= (train >= lo) & (train <= hi)
    return train[~blocked]


def purged_kfold(
    n_observations: int,
    *,
    n_splits: int = 5,
    purge_gap: int = 0,
    embargo_gap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2 or n_observations < n_splits:
        raise ValueError("need at least two non-empty folds")
    all_indices = np.arange(n_observations, dtype=int)
    for test in np.array_split(all_indices, n_splits):
        train = all_indices[~np.isin(all_indices, test)]
        yield purge_train_indices(
            train, test, purge_gap=purge_gap, embargo_gap=embargo_gap
        ), test

