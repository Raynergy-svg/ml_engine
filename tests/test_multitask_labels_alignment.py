import pytest

pytest.importorskip("multitask_labels")

import numpy as np
import pandas as pd

from multitask_labels import build_multitask_targets


def test_direction_uses_horizon_move_target_shift():
    # Close series: up then down; horizon=2 should reflect move from t0 to t0+2
    close = np.array([100, 101, 102, 101, 100, 99, 100], dtype=float)
    df = pd.DataFrame({"close": close})

    seq_len = 3
    target_shift = 2
    targets = build_multitask_targets(
        df,
        sequence_length=seq_len,
        target_shift=target_shift,
        close_col="close",
        risk_window=3,
        state_classes=3,
    )

    # For each i, base index = i+seq_len-1, target index = base+target_shift
    n = len(close) - seq_len - target_shift + 1
    base_idx = np.arange(n) + seq_len - 1
    tgt_idx = base_idx + target_shift
    expected_return = (close[tgt_idx] / close[base_idx]) - 1.0
    expected_dir = (expected_return > 0).astype(np.float32)

    assert targets.trend.shape == (n,)
    assert targets.direction.shape == (n,)
    assert np.allclose(targets.trend, expected_return.astype(np.float32))
    assert np.array_equal(targets.direction, expected_dir)
# — Raynergy-svg —
