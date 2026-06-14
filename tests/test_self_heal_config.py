"""Tests for self-healing config updater (US-016)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.autonomy.self_heal_config import SelfHealConfig, SelfHealConfigUpdater


class TestSelfHealConfigUpdater:
    def test_init(self):
        updater = SelfHealConfigUpdater()
        assert updater.cfg.min_win_rate_threshold == 0.45

    def test_insufficient_history(self):
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=10))
        result = updater.update({"min_confidence": 42.0}, [])
        assert result["reason"] == "insufficient_trade_history"
        assert len(result["changes"]) == 0

    def test_low_win_rate_adjustment(self):
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=5))
        trades = [{"pnl": -1}] * 4 + [{"pnl": 1}]
        config = {"min_confidence": 42.0, "risk_per_trade_pct": 0.05, "compound_gate_floor": 0.1}
        result = updater.update(config, trades)
        assert len(result["changes"]) > 0
        assert result["changes"][0]["param"] == "min_confidence"
        assert result["changes"][0]["new"] < result["changes"][0]["old"]

    def test_history_tracking(self):
        updater = SelfHealConfigUpdater()
        trades = [{"pnl": 1}] * 20
        updater.update({"min_confidence": 42.0}, trades)
        assert len(updater.get_history()) == 1
