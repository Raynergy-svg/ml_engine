"""Smoke tests for RL training components.

These tests must never call `exit()` at import time (pytest collection safety).
RL dependencies (stable-baselines3, gymnasium/gym) are optional in some envs.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_rl_training_smoke():
    from rl_position_sizing import GYM_AVAILABLE, SB3_AVAILABLE, RLConfig, RLPositionSizer

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        pytest.skip("Optional RL dependencies not installed (SB3/gym).")

    # Create synthetic training data (small + fast)
    n_samples = 120  # > min_trades=100 in default config
    features = np.random.randn(n_samples, 20)
    predictions = np.column_stack(
        [
            np.random.uniform(0.45, 0.65, n_samples),
            np.random.uniform(0.4, 0.7, n_samples),
        ]
    )
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.3)

    config = RLConfig(total_timesteps=256)
    sizer = RLPositionSizer(config)
    sizer.train(features, predictions, prices)

    test_features = np.random.randn(20)
    test_pred = np.array([0.6, 0.7])
    position_size = sizer.get_position_size(test_features, test_pred)

    assert np.isfinite(position_size)
