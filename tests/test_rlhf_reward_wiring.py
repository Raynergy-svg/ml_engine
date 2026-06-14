"""Integration test: RLHFRewardShaper wiring into PostTradeLoop.run().

US-011: After computing the base shaped reward, PostTradeLoop should call
RLHFRewardShaper.shape_reward() and append the shaped reward delta.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.scanner.feedback.post_trade_loop import PostTradeLoop


def test_rlhf_shapes_reward_when_enabled(tmp_path, monkeypatch):
    """When RLHF is enabled and trained, shaped reward should include learned component."""
    loop = PostTradeLoop()

    # Mock RLHF shaper
    mock_rlhf = MagicMock()
    mock_rlhf._trained = True
    mock_rlhf.shape_reward.return_value = 42.5
    loop._rlhf_shaper = mock_rlhf

    # Stub journal read and feedback append
    monkeypatch.setattr(loop, "_already_processed", lambda _tid: False)
    monkeypatch.setattr(loop, "_load_journal_entry", lambda _tid: {
        "trade_id": "T123",
        "pair": "EUR_USD",
        "pnl_pips": 15.0,
        "slippage_pips": 0.5,
        "mae_pips": 8.0,
        "volatility_regime": "NORMAL",
        "sl_pips": 20.0,
    })
    monkeypatch.setattr(loop, "_append_feedback_record", lambda **kw: None)
    monkeypatch.setattr(loop, "_append_learning", lambda **kw: None)

    result = loop.run({"trade_id": "T123", "source": "test"})

    assert result["status"] == "processed"
    mock_rlhf.shape_reward.assert_called_once()
    # The shaped_reward in result should reflect RLHF value
    assert result.get("shaped_reward") == pytest.approx(42.5, abs=0.1)


def test_rlhf_not_called_when_untrained():
    loop = PostTradeLoop()
    mock_rlhf = MagicMock()
    mock_rlhf._trained = False
    loop._rlhf_shaper = mock_rlhf

    # Should not call shape_reward if not trained
    result = loop.run({"trade_id": "", "source": "test"})
    mock_rlhf.shape_reward.assert_not_called()
