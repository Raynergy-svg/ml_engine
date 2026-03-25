"""
Tests for WalkForwardOptimizer (US-305, Phase 48).

Covers: grid search, WFE calculation, overfitting detection, trade recording,
optimization trigger, recommendations, persistence.
"""

import json
import os
import pytest
import tempfile
import time

from src.scanner.walkforward_optimizer import (
    WalkForwardOptimizer,
    WalkForwardConfig,
    TradeOutcome,
    ParameterSet,
    OptimizationResult,
    create_default_walkforward_optimizer,
)


def _make_outcome(
    pair="EUR_USD", pnl=10.0, won=True, confidence=0.60,
    momentum=0.55, regime="NORMAL",
):
    return TradeOutcome(
        pair=pair, direction="BUY", pnl=pnl,
        confidence=confidence, momentum_score=momentum,
        atr_sl_pips=15.0, atr_tp_pips=25.0,
        regime=regime, won=won, timestamp=time.time(),
    )


class TestParameterSet:
    def test_as_dict(self):
        p = ParameterSet(confidence_threshold=0.55, momentum_threshold=0.45)
        d = p.as_dict()
        assert d["confidence_threshold"] == 0.55
        assert d["momentum_threshold"] == 0.45

    def test_defaults(self):
        p = ParameterSet()
        assert p.confidence_threshold == 0.50
        assert p.atr_tp_multiplier == 1.5


class TestWalkForwardConfig:
    def test_defaults(self):
        c = WalkForwardConfig()
        assert c.in_sample_size == 60
        assert c.out_of_sample_size == 15
        assert c.trigger_interval == 75

    def test_validate_valid(self):
        WalkForwardConfig().validate()

    def test_validate_bad_is_size(self):
        with pytest.raises(ValueError, match="in_sample_size"):
            WalkForwardConfig(in_sample_size=5).validate()

    def test_validate_bad_oos_size(self):
        with pytest.raises(ValueError, match="out_of_sample_size"):
            WalkForwardConfig(out_of_sample_size=2).validate()


class TestWalkForwardOptimizer:
    def _config(self, **kwargs):
        """Small config for fast testing."""
        defaults = dict(
            in_sample_size=20,
            out_of_sample_size=5,
            trigger_interval=25,
            confidence_range=(0.40, 0.50, 0.60),
            momentum_range=(0.40, 0.50),
            atr_sl_range=(1.0, 1.2),
            atr_tp_range=(1.5, 2.0),
        )
        defaults.update(kwargs)
        return WalkForwardConfig(**defaults)

    def test_should_optimize_insufficient_data(self):
        opt = WalkForwardOptimizer(self._config())
        for i in range(10):
            opt.record_outcome(_make_outcome())
        assert opt.should_optimize() is False

    def test_should_optimize_enough_data(self):
        opt = WalkForwardOptimizer(self._config())
        for i in range(30):
            opt.record_outcome(_make_outcome())
        assert opt.should_optimize() is True

    def test_should_optimize_disabled(self):
        config = self._config(enabled=False)
        opt = WalkForwardOptimizer(config)
        for i in range(30):
            opt.record_outcome(_make_outcome())
        assert opt.should_optimize() is False

    def test_run_optimization_winning_trades(self):
        opt = WalkForwardOptimizer(self._config())
        # 20 IS wins + 5 OOS wins
        for i in range(25):
            opt.record_outcome(_make_outcome(pnl=15.0, won=True, confidence=0.65, momentum=0.55))

        result = opt.run_optimization()
        assert result.is_profit_factor > 1.0
        assert result.oos_profit_factor > 0
        assert isinstance(result.wfe, float)
        assert result.recommendation in ("APPLY", "REVIEW", "REJECT")

    def test_run_optimization_losing_trades(self):
        opt = WalkForwardOptimizer(self._config())
        # Mostly losses
        for i in range(5):
            opt.record_outcome(_make_outcome(pnl=5.0, won=True, confidence=0.65, momentum=0.55))
        for i in range(20):
            opt.record_outcome(_make_outcome(pnl=-10.0, won=False, confidence=0.65, momentum=0.55))

        result = opt.run_optimization()
        # Should still complete without error
        assert isinstance(result, OptimizationResult)

    def test_run_optimization_insufficient_data(self):
        opt = WalkForwardOptimizer(self._config())
        for i in range(5):
            opt.record_outcome(_make_outcome())

        result = opt.run_optimization()
        assert result.recommendation == "REJECT"
        assert result.is_trade_count == 0

    def test_overfitting_detection(self):
        """When IS is great but OOS is terrible, WFE should be low."""
        config = self._config()
        opt = WalkForwardOptimizer(config)

        # IS trades: all wins with high confidence
        for i in range(20):
            opt.record_outcome(_make_outcome(pnl=20.0, won=True, confidence=0.70, momentum=0.60))
        # OOS trades: all losses
        for i in range(5):
            opt.record_outcome(_make_outcome(pnl=-20.0, won=False, confidence=0.70, momentum=0.60))

        result = opt.run_optimization()
        # OOS PF should be low relative to IS PF
        assert result.oos_profit_factor <= result.is_profit_factor

    def test_trades_since_last_opt_reset(self):
        opt = WalkForwardOptimizer(self._config())
        for i in range(30):
            opt.record_outcome(_make_outcome())
        assert opt._trades_since_last_opt == 30

        opt.run_optimization()
        assert opt._trades_since_last_opt == 0

    def test_outcome_buffer_bounded(self):
        config = self._config(in_sample_size=20, out_of_sample_size=5)
        opt = WalkForwardOptimizer(config)
        for i in range(100):
            opt.record_outcome(_make_outcome())
        # Max size = (20+5)*2 = 50
        assert len(opt._outcomes) <= 50

    def test_evaluate_params_filters_low_confidence(self):
        opt = WalkForwardOptimizer(self._config())
        # Trades with confidence below threshold should be filtered
        trades = [
            _make_outcome(pnl=-10.0, won=False, confidence=0.30, momentum=0.55),
            _make_outcome(pnl=20.0, won=True, confidence=0.70, momentum=0.55),
        ]
        params = ParameterSet(confidence_threshold=0.50, momentum_threshold=0.40)
        pf = opt._evaluate_params(params, trades)
        # Only the winning trade passes confidence gate
        assert pf == 3.0  # All profit, no loss (capped at 3.0)

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "wf_state.json")
            config = self._config(state_file=state_file)
            opt = WalkForwardOptimizer(config)

            for i in range(10):
                opt.record_outcome(_make_outcome())
            opt.save_state()

            opt2 = WalkForwardOptimizer(config)
            assert len(opt2._outcomes) == 10

    def test_result_as_dict(self):
        result = OptimizationResult(
            best_params=ParameterSet(),
            is_profit_factor=1.5,
            oos_profit_factor=1.2,
            wfe=0.80,
            is_overfitting=False,
            is_trade_count=60,
            oos_trade_count=15,
            recommendation="APPLY",
            timestamp=time.time(),
        )
        d = result.as_dict()
        assert d["wfe"] == 0.8
        assert d["recommendation"] == "APPLY"
        assert "best_params" in d

    def test_write_recommendation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_file = os.path.join(tmpdir, "config_adj.json")
            config = self._config(recommendations_file=rec_file)
            opt = WalkForwardOptimizer(config)

            for i in range(25):
                opt.record_outcome(_make_outcome(pnl=10.0, won=True, confidence=0.65, momentum=0.55))

            result = opt.run_optimization()

            # Check recommendation was written
            assert os.path.exists(rec_file)
            with open(rec_file) as f:
                data = json.load(f)
            assert "walkforward_recommendations" in data
            assert len(data["walkforward_recommendations"]) >= 1


class TestWalkForwardFactory:
    def test_create_default(self):
        opt = create_default_walkforward_optimizer()
        assert isinstance(opt, WalkForwardOptimizer)

    def test_create_with_config(self):
        config = WalkForwardConfig(in_sample_size=100)
        opt = create_default_walkforward_optimizer(config=config)
        assert opt.config.in_sample_size == 100
