"""Chronological splits that preserve an untouched temporal tail."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from src.evidence.canonical import canonical_bytes


@dataclass(frozen=True)
class TemporalHoldout:
    train: tuple[int, ...]
    holdout: tuple[int, ...]
    embargoed: tuple[int, ...]

    @property
    def split_digest(self) -> str:
        """Bind reports to the exact train/embargo/holdout allocation."""
        payload = {
            "train": self.train,
            "embargoed": self.embargoed,
            "holdout": self.holdout,
        }
        return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def chronological_holdout(
    n_observations: int,
    *,
    holdout_fraction: float = 0.2,
    embargo_observations: int = 0,
    minimum_train: int = 2,
    minimum_holdout: int = 2,
) -> TemporalHoldout:
    """Reserve the final observations once; embargoed rows belong to neither side."""
    if n_observations < minimum_train + minimum_holdout + embargo_observations:
        raise ValueError("insufficient observations for train, embargo, and untouched holdout")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    if embargo_observations < 0:
        raise ValueError("embargo_observations must be non-negative")
    holdout_size = max(minimum_holdout, int(np.ceil(n_observations * holdout_fraction)))
    holdout_start = n_observations - holdout_size
    train_end = holdout_start - embargo_observations
    if train_end < minimum_train:
        raise ValueError("holdout allocation leaves too little training history")
    return TemporalHoldout(
        train=tuple(range(train_end)),
        embargoed=tuple(range(train_end, holdout_start)),
        holdout=tuple(range(holdout_start, n_observations)),
    )
