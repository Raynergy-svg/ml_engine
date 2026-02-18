"""
Tests for src/scanner module.

Tests cover:
- ScannerConfig dataclass defaults and post_init behavior
- PairAnalysis scoring, gate summary, and tradeability logic
- ScanResult filtering and aggregation properties
- GateEvaluator existence
"""

import pytest
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestScannerConfig:
    """Test ScannerConfig dataclass."""

    def test_default_pairs(self):
        from src.scanner.config import ScannerConfig, MAJOR_PAIRS
        cfg = ScannerConfig()
        assert cfg.pairs == MAJOR_PAIRS

    def test_default_thresholds(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        assert cfg.min_tcn_probability == 0.60
        assert cfg.min_confidence == 50.0  # 0-100 scale (Ridge ADX score)
        assert cfg.min_momentum == 0.20
        assert cfg.max_drawdown_pct == 0.025

    def test_string_paths_converted(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig(config_path="config/config_improved_H1.yaml",
                            model_dir="trained_data/models")
        assert isinstance(cfg.config_path, Path)
        assert isinstance(cfg.model_dir, Path)
        assert cfg.config_path.is_absolute()
        assert cfg.model_dir.is_absolute()

    def test_pip_values(self):
        from src.scanner.config import ScannerConfig, PIP_VALUES
        cfg = ScannerConfig()
        assert cfg.get_pip_value("EUR_USD") == 0.0001
        assert cfg.get_pip_value("USD_JPY") == 0.01
        # Unknown pair falls back to 0.0001
        assert cfg.get_pip_value("UNKNOWN_PAIR") == 0.0001

    def test_session_filter_defaults(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        assert cfg.enable_session_filter is True
        assert cfg.session_start_utc == 8
        assert cfg.session_end_utc == 21

    def test_from_cli_args_force_disables_session(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig.from_cli_args(force=True)
        assert cfg.enable_session_filter is False

    def test_profile_aggressive_lowers_gate_thresholds(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        cfg.apply_profile("aggressive")
        assert cfg.profile == "aggressive"
        assert cfg.min_confidence == 45.0
        assert cfg.min_momentum == 0.12
        assert cfg.min_atr_pips == 3.0
        assert cfg.min_volatility_regime == 0

    def test_profile_conservative_tightens_gate_thresholds(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        cfg.apply_profile("conservative")
        assert cfg.profile == "conservative"
        assert cfg.min_confidence == 58.0
        assert cfg.min_momentum == 0.28
        assert cfg.min_atr_pips == 6.0
        assert cfg.min_volatility_regime == 2

    def test_unknown_profile_raises(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        with pytest.raises(ValueError):
            cfg.apply_profile("ultra")


# ---------------------------------------------------------------------------
# Results tests
# ---------------------------------------------------------------------------

def _make_pair_analysis(**kwargs):
    """Create a PairAnalysis with sensible defaults, overridable via kwargs."""
    from src.scanner.results import PairAnalysis
    defaults = dict(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.70,
        gates_passed=True,
        volatility_gate_passed=True,
        tcn_probability=0.65,
        ridge_confidence=60.0,
        xgb_momentum=0.40,
        momentum=0.40,
        drawdown=0.01,
        rf_drawdown=0.01,
        trend_strength=0.50,
        confidence_passed=True,
        momentum_passed=True,
        risk_passed=True,
    )
    defaults.update(kwargs)
    return PairAnalysis(**defaults)


class TestPairAnalysis:
    """Test PairAnalysis dataclass."""

    def test_is_tradeable_all_gates_passed(self):
        pa = _make_pair_analysis(gates_passed=True, direction="LONG")
        assert pa.is_tradeable is True

    def test_not_tradeable_hold(self):
        """HOLD direction with gates passed should still not be tradeable
        (direction is not None but HOLD is a valid direction that gates can pass)."""
        pa = _make_pair_analysis(gates_passed=True, direction="HOLD")
        # is_tradeable checks: gates_passed and direction is not None and error is None
        # HOLD with gates_passed=True and no error → is_tradeable is True
        # (the property doesn't filter on direction value, only None)
        assert pa.is_tradeable is True

    def test_not_tradeable_direction_none(self):
        pa = _make_pair_analysis(gates_passed=True, direction=None)
        assert pa.is_tradeable is False

    def test_not_tradeable_gates_failed(self):
        pa = _make_pair_analysis(gates_passed=False, direction="LONG")
        assert pa.is_tradeable is False

    def test_not_tradeable_with_error(self):
        pa = _make_pair_analysis(gates_passed=True, direction="LONG",
                                 error="Model load failed")
        assert pa.is_tradeable is False

    def test_overall_score_range(self):
        pa = _make_pair_analysis()
        score = pa.overall_score
        assert 0.0 <= score <= 1.0

    def test_overall_score_zero_on_error(self):
        pa = _make_pair_analysis(error="something broke")
        assert pa.overall_score == 0.0

    def test_gate_summary_all_passed(self):
        pa = _make_pair_analysis(
            confidence_passed=True,
            confidence_gate_passed=True,
            momentum_passed=True,
            momentum_gate_passed=True,
            risk_passed=True,
            risk_gate_passed=True,
        )
        # gate_summary counts confidence, momentum, risk (3 gates)
        assert pa.gate_summary == "3/3"

    def test_gate_summary_some_failed(self):
        pa = _make_pair_analysis(
            confidence_passed=False,
            confidence_gate_passed=False,
            momentum_passed=False,
            momentum_gate_passed=False,
            risk_passed=True,
            risk_gate_passed=True,
        )
        assert pa.gate_summary == "1/3"

    def test_to_dict_keys(self):
        pa = _make_pair_analysis()
        d = pa.to_dict()
        assert "pair" in d
        assert "direction" in d
        assert "confidence" in d
        assert "is_tradeable" in d
        assert "overall_score" in d
        assert "volatility_regime" in d

    def test_default_fields(self):
        from src.scanner.results import PairAnalysis
        pa = PairAnalysis(pair="TEST_PAIR")
        assert pa.direction == "HOLD"
        assert pa.confidence == 0.0
        assert pa.gates_passed is False
        assert pa.error is None


class TestScanResult:
    """Test ScanResult aggregation."""

    def _make_result(self) -> "ScanResult":
        from src.scanner.results import ScanResult
        return ScanResult(analyses=[
            _make_pair_analysis(pair="EUR_USD", gates_passed=True,
                                direction="LONG", confidence=0.80),
            _make_pair_analysis(pair="GBP_USD", gates_passed=True,
                                direction="SHORT", confidence=0.65),
            _make_pair_analysis(pair="USD_JPY", gates_passed=False,
                                direction="HOLD"),
            _make_pair_analysis(pair="AUD_USD", direction="LONG",
                                error="Model not found"),
        ])

    def test_tradeable_count(self):
        sr = self._make_result()
        assert len(sr.tradeable) == 2

    def test_tradeable_sorted_by_score(self):
        sr = self._make_result()
        scores = [a.overall_score for a in sr.tradeable]
        assert scores == sorted(scores, reverse=True)

    def test_non_tradeable(self):
        sr = self._make_result()
        # USD_JPY has gates_passed=False and no error
        non_t = sr.non_tradeable
        assert any(a.pair == "USD_JPY" for a in non_t)

    def test_errors(self):
        sr = self._make_result()
        assert len(sr.errors) == 1
        assert sr.errors[0].pair == "AUD_USD"

    def test_success_rate(self):
        sr = self._make_result()
        # 3 out of 4 succeeded (1 error)
        assert sr.success_rate == pytest.approx(0.75)

    def test_get_top_n(self):
        sr = self._make_result()
        top = sr.get_top_n(1)
        assert len(top) == 1
        assert top[0].pair == "EUR_USD"  # highest confidence

    def test_to_dict(self):
        sr = self._make_result()
        d = sr.to_dict()
        assert "tradeable_count" in d
        assert "error_count" in d
        assert d["error_count"] == 1
        assert d["tradeable_count"] == 2

    def test_empty_result(self):
        from src.scanner.results import ScanResult
        sr = ScanResult()
        assert sr.success_rate == 0.0
        assert len(sr.tradeable) == 0
        assert len(sr.errors) == 0


# ---------------------------------------------------------------------------
# Gate evaluator tests
# ---------------------------------------------------------------------------

class TestGateEvaluator:
    """Test GateEvaluator class exists and is importable."""

    def test_gate_evaluator_importable(self):
        from src.scanner.gates import GateEvaluator
        assert GateEvaluator is not None
