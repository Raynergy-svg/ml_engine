"""
Tests for StrategyFitnessDetector (US-302, Phase 48).

Covers: fitness scoring, trade recording, thresholds, persistence,
disabled filter, insufficient data, edge cases.
"""

import json
import os
import pytest
import tempfile
import time

from src.scanner.strategy_fitness import (
    StrategyFitnessDetector,
    StrategyFitnessConfig,
    FitnessMetrics,
    FitnessCheckResult,
    TradeRecord,
    create_default_fitness_detector,
)


def _make_trade(pair="EUR_USD", regime="NORMAL", pnl=10.0, won=True,
                predicted="BUY", actual="BUY", disagreement=0.15):
    return TradeRecord(
        pair=pair, regime=regime, pnl=pnl, won=won,
        predicted_direction=predicted, actual_direction=actual,
        agent_disagreement=disagreement, timestamp=time.time(),
    )


class TestFitnessMetrics:
    def test_win_rate(self):
        m = FitnessMetrics(trade_count=10, win_count=6)
        assert m.win_rate == pytest.approx(0.6)

    def test_win_rate_zero_trades(self):
        m = FitnessMetrics(trade_count=0)
        assert m.win_rate == 0.0

    def test_fitness_score_good(self):
        m = FitnessMetrics(
            rolling_profit_factor=2.0,
            prediction_correlation=0.8,
            avg_agent_disagreement=0.1,
        )
        assert m.fitness_score > 0.7

    def test_fitness_score_bad(self):
        m = FitnessMetrics(
            rolling_profit_factor=0.5,
            prediction_correlation=0.3,
            avg_agent_disagreement=0.5,
        )
        assert m.fitness_score < 0.4

    def test_fitness_score_bounded(self):
        m = FitnessMetrics(
            rolling_profit_factor=10.0,
            prediction_correlation=1.0,
            avg_agent_disagreement=0.0,
        )
        assert m.fitness_score <= 1.0


class TestStrategyFitnessConfig:
    def test_defaults(self):
        config = StrategyFitnessConfig()
        assert config.window_size == 20
        assert config.min_trades == 10
        assert config.skip_threshold == 0.40
        assert config.reduce_threshold == 0.60

    def test_validate_valid(self):
        StrategyFitnessConfig().validate()

    def test_validate_bad_window(self):
        with pytest.raises(ValueError, match="window_size"):
            StrategyFitnessConfig(window_size=2).validate()

    def test_validate_thresholds_inverted(self):
        with pytest.raises(ValueError, match="skip_threshold"):
            StrategyFitnessConfig(skip_threshold=0.7, reduce_threshold=0.5).validate()


class TestStrategyFitnessDetector:
    def test_insufficient_data_normal(self):
        d = StrategyFitnessDetector()
        result = d.check_fitness("EUR_USD", "NORMAL")
        assert result.action == "NORMAL"
        assert "insufficient_data" in result.reason

    def test_good_fitness_normal(self):
        d = StrategyFitnessDetector()
        for _ in range(15):
            d.record_trade(_make_trade(pnl=20.0, won=True, disagreement=0.1))
        for _ in range(5):
            d.record_trade(_make_trade(pnl=-8.0, won=False, disagreement=0.1))

        result = d.check_fitness("EUR_USD", "NORMAL")
        assert result.action == "NORMAL"
        assert result.fitness_score > 0.6

    def test_bad_fitness_skip(self):
        d = StrategyFitnessDetector()
        # Mostly losses with high disagreement and bad prediction correlation
        for _ in range(3):
            d.record_trade(_make_trade(pnl=5.0, won=True, predicted="BUY",
                                       actual="BUY", disagreement=0.5))
        for _ in range(12):
            d.record_trade(_make_trade(pnl=-15.0, won=False, predicted="BUY",
                                       actual="SELL", disagreement=0.5))

        result = d.check_fitness("EUR_USD", "NORMAL")
        assert result.action == "SKIP"
        assert result.fitness_score < 0.40

    def test_mediocre_fitness_reduce(self):
        config = StrategyFitnessConfig(skip_threshold=0.30, reduce_threshold=0.60)
        d = StrategyFitnessDetector(config)
        # Mixed results — mediocre
        for _ in range(6):
            d.record_trade(_make_trade(pnl=10.0, won=True, predicted="BUY",
                                       actual="BUY", disagreement=0.25))
        for _ in range(6):
            d.record_trade(_make_trade(pnl=-12.0, won=False, predicted="BUY",
                                       actual="SELL", disagreement=0.35))

        result = d.check_fitness("EUR_USD", "NORMAL")
        assert result.action in ("REDUCE_SIZE", "SKIP")

    def test_disabled_filter(self):
        config = StrategyFitnessConfig(enabled=False)
        d = StrategyFitnessDetector(config)
        result = d.check_fitness("EUR_USD", "NORMAL")
        assert result.action == "NORMAL"
        assert "disabled" in result.reason

    def test_different_pairs_tracked_separately(self):
        d = StrategyFitnessDetector()
        for _ in range(12):
            d.record_trade(_make_trade(pair="EUR_USD", pnl=20.0, won=True))
        for _ in range(12):
            d.record_trade(_make_trade(pair="GBP_USD", pnl=-15.0, won=False,
                                       predicted="BUY", actual="SELL",
                                       disagreement=0.5))

        r1 = d.check_fitness("EUR_USD", "NORMAL")
        r2 = d.check_fitness("GBP_USD", "NORMAL")
        assert r1.action == "NORMAL"
        assert r2.action in ("SKIP", "REDUCE_SIZE")

    def test_rolling_window_bounded(self):
        config = StrategyFitnessConfig(window_size=10)
        d = StrategyFitnessDetector(config)
        # Fill with 20 trades (window=10, should only keep last 10)
        for _ in range(20):
            d.record_trade(_make_trade(pnl=10.0, won=True))

        key = d._key("EUR_USD", "NORMAL")
        assert len(d._trades[key]) == 10

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "fitness.json")
            config = StrategyFitnessConfig(state_file=state_file)
            d = StrategyFitnessDetector(config)

            for _ in range(12):
                d.record_trade(_make_trade(pnl=15.0, won=True))
            d.save_state()

            d2 = StrategyFitnessDetector(config)
            key = d2._key("EUR_USD", "NORMAL")
            assert key in d2._trades
            assert len(d2._trades[key]) == 12

    def test_get_all_fitness(self):
        d = StrategyFitnessDetector()
        for _ in range(12):
            d.record_trade(_make_trade(pair="EUR_USD", pnl=10.0, won=True))
        for _ in range(12):
            d.record_trade(_make_trade(pair="GBP_USD", regime="HIGH",
                                       pnl=-5.0, won=False))

        results = d.get_all_fitness()
        assert "EUR_USD:NORMAL" in results
        assert "GBP_USD:HIGH" in results


class TestFitnessFactory:
    def test_create_default(self):
        d = create_default_fitness_detector()
        assert isinstance(d, StrategyFitnessDetector)

    def test_create_with_config(self):
        config = StrategyFitnessConfig(window_size=50)
        d = create_default_fitness_detector(config=config)
        assert d.config.window_size == 50
