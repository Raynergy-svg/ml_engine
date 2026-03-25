"""
Tests for PairPerformanceTracker & Blacklisting (US-303, Phase 48).

Covers: trade recording, blacklist triggers, cooldown, persistence,
disabled filter, summary, edge cases.
"""

import json
import os
import pytest
import tempfile

from src.scanner.pair_blacklist import (
    PairPerformanceTracker,
    PairBlacklistConfig,
    PairMetrics,
    BlacklistCheckResult,
    create_default_pair_tracker,
)


class TestPairMetrics:
    def test_win_rate(self):
        m = PairMetrics(pair="EUR_USD", wins=6, losses=4, trade_count=10)
        assert m.win_rate == pytest.approx(0.6)

    def test_win_rate_zero(self):
        m = PairMetrics(pair="EUR_USD")
        assert m.win_rate == 0.0

    def test_profit_factor(self):
        m = PairMetrics(pair="EUR_USD", total_profit=200.0, total_loss=100.0)
        assert m.profit_factor == pytest.approx(2.0)

    def test_profit_factor_no_losses(self):
        m = PairMetrics(pair="EUR_USD", total_profit=100.0, total_loss=0.0)
        assert m.profit_factor == 3.0  # Capped

    def test_sharpe_estimate_insufficient_data(self):
        m = PairMetrics(pair="EUR_USD", trade_count=3)
        assert m.sharpe_estimate == 0.0


class TestPairBlacklistConfig:
    def test_defaults(self):
        config = PairBlacklistConfig()
        assert config.max_consecutive_losses == 4
        assert config.min_trades == 10
        assert config.cooldown_cycles == 2

    def test_validate_valid(self):
        PairBlacklistConfig().validate()

    def test_validate_bad_min_trades(self):
        with pytest.raises(ValueError, match="min_trades"):
            PairBlacklistConfig(min_trades=1).validate()

    def test_validate_bad_cooldown(self):
        with pytest.raises(ValueError, match="cooldown_cycles"):
            PairBlacklistConfig(cooldown_cycles=0).validate()


class TestPairPerformanceTracker:
    def test_no_trades_not_blacklisted(self):
        t = PairPerformanceTracker()
        result = t.check_pair("EUR_USD")
        assert result.is_blacklisted is False

    def test_winning_pair_not_blacklisted(self):
        t = PairPerformanceTracker()
        for _ in range(12):
            t.update_from_trade("EUR_USD", pnl=15.0, won=True)
        result = t.check_pair("EUR_USD")
        assert result.is_blacklisted is False

    def test_consecutive_losses_trigger_blacklist(self):
        config = PairBlacklistConfig(min_trades=5, max_consecutive_losses=4)
        t = PairPerformanceTracker(config)
        # 5 wins then 4 consecutive losses
        for _ in range(5):
            t.update_from_trade("EUR_USD", pnl=10.0, won=True)
        for _ in range(4):
            t.update_from_trade("EUR_USD", pnl=-15.0, won=False)

        result = t.check_pair("EUR_USD")
        assert result.is_blacklisted is True
        assert result.cooldown_remaining == 2

    def test_insufficient_trades_no_blacklist(self):
        config = PairBlacklistConfig(min_trades=10)
        t = PairPerformanceTracker(config)
        for _ in range(5):
            t.update_from_trade("EUR_USD", pnl=-20.0, won=False)

        result = t.check_pair("EUR_USD")
        assert result.is_blacklisted is False  # Only 5 trades < 10 min

    def test_cooldown_tick(self):
        config = PairBlacklistConfig(min_trades=5, max_consecutive_losses=3, cooldown_cycles=2)
        t = PairPerformanceTracker(config)
        for _ in range(5):
            t.update_from_trade("EUR_USD", pnl=10.0, won=True)
        for _ in range(3):
            t.update_from_trade("EUR_USD", pnl=-20.0, won=False)

        assert t.check_pair("EUR_USD").is_blacklisted is True

        removed = t.tick_cooldown()
        assert len(removed) == 0
        assert t.check_pair("EUR_USD").is_blacklisted is True  # 1 cycle left

        removed = t.tick_cooldown()
        assert "EUR_USD" in removed
        assert t.check_pair("EUR_USD").is_blacklisted is False

    def test_disabled_filter(self):
        config = PairBlacklistConfig(enabled=False)
        t = PairPerformanceTracker(config)
        result = t.check_pair("EUR_USD")
        assert result.is_blacklisted is False
        assert "disabled" in result.reason

    def test_different_pairs_independent(self):
        config = PairBlacklistConfig(min_trades=5, max_consecutive_losses=3)
        t = PairPerformanceTracker(config)
        for _ in range(8):
            t.update_from_trade("EUR_USD", pnl=15.0, won=True)
        for _ in range(5):
            t.update_from_trade("GBP_USD", pnl=10.0, won=True)
        for _ in range(3):
            t.update_from_trade("GBP_USD", pnl=-20.0, won=False)

        assert t.check_pair("EUR_USD").is_blacklisted is False
        assert t.check_pair("GBP_USD").is_blacklisted is True

    def test_get_blacklisted_pairs(self):
        config = PairBlacklistConfig(min_trades=5, max_consecutive_losses=3)
        t = PairPerformanceTracker(config)
        for _ in range(5):
            t.update_from_trade("EUR_USD", pnl=5.0, won=True)
        for _ in range(3):
            t.update_from_trade("EUR_USD", pnl=-20.0, won=False)

        bl = t.get_blacklisted_pairs()
        assert "EUR_USD" in bl

    def test_get_pair_summary(self):
        t = PairPerformanceTracker()
        for _ in range(5):
            t.update_from_trade("EUR_USD", pnl=10.0, won=True)
        summary = t.get_pair_summary()
        assert "EUR_USD" in summary
        assert summary["EUR_USD"]["trades"] == 5
        assert summary["EUR_USD"]["win_rate"] == 1.0

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "pair.json")
            config = PairBlacklistConfig(state_file=state_file)
            t = PairPerformanceTracker(config)

            for _ in range(8):
                t.update_from_trade("EUR_USD", pnl=10.0, won=True)
            t.save_state()

            t2 = PairPerformanceTracker(config)
            assert "EUR_USD" in t2.metrics
            assert t2.metrics["EUR_USD"].trade_count == 8

    def test_drawdown_tracking(self):
        config = PairBlacklistConfig(min_trades=5, max_drawdown_pct=10.0)
        t = PairPerformanceTracker(config)
        # Build up profit then lose a lot
        for _ in range(5):
            t.update_from_trade("EUR_USD", pnl=50.0, won=True)  # peak = 250
        for _ in range(3):
            t.update_from_trade("EUR_USD", pnl=-40.0, won=False)  # total = 130, dd from 250

        m = t.metrics["EUR_USD"]
        assert m.max_drawdown_pct > 0


class TestPairTrackerFactory:
    def test_create_default(self):
        t = create_default_pair_tracker()
        assert isinstance(t, PairPerformanceTracker)

    def test_create_with_config(self):
        config = PairBlacklistConfig(cooldown_cycles=5)
        t = create_default_pair_tracker(config=config)
        assert t.config.cooldown_cycles == 5
