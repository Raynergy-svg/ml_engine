"""
Tests for src/scanner module.

Tests cover:
- ScannerConfig dataclass defaults and post_init behavior
- PairAnalysis scoring, gate summary, and tradeability logic
- ScanResult filtering and aggregation properties
- Helper functions (_normalize_confidence, _format_confidence_pct)
- ScannerError exception
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
        assert cfg.min_confidence == 0.50
        assert cfg.min_momentum == 0.20
        assert cfg.max_drawdown_pct == 0.025
        assert cfg.min_meta_confidence == 0.55

    def test_string_paths_converted(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig(config_path="config/config_improved_H1.yaml",
                            model_dir="trained_data/models")
        assert isinstance(cfg.config_path, Path)
        assert isinstance(cfg.model_dir, Path)
        assert cfg.config_path.is_absolute()
        assert cfg.model_dir.is_absolute()

    def test_position_multiplier_tiers(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        # Low tier
        assert cfg.get_position_multiplier(0.55) == cfg.position_multiplier_low
        # Medium tier
        assert cfg.get_position_multiplier(0.70) == cfg.position_multiplier_medium
        # High tier
        assert cfg.get_position_multiplier(0.90) == cfg.position_multiplier_high

    def test_pip_values(self):
        from src.scanner.config import ScannerConfig, PIP_VALUES
        cfg = ScannerConfig()
        assert cfg.get_pip_value("EUR_USD") == 0.0001
        assert cfg.get_pip_value("USD_JPY") == 0.01
        # Unknown pair falls back to 0.0001
        assert cfg.get_pip_value("UNKNOWN_PAIR") == 0.0001

    def test_format_confidence_pct(self):
        from src.scanner.config import ScannerConfig
        assert ScannerConfig.format_confidence_pct(0.65) == "65%"
        assert ScannerConfig.format_confidence_pct(1.0) == "100%"
        assert ScannerConfig.format_confidence_pct(0.0) == "0%"

    def test_normalize_confidence(self):
        from src.scanner.config import ScannerConfig
        assert ScannerConfig.normalize_confidence(50, from_scale=100) == 0.5
        assert ScannerConfig.normalize_confidence(100, from_scale=100) == 1.0
        assert ScannerConfig.normalize_confidence(0, from_scale=100) == 0.0


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
        volatility_regime=2,           # ACTIVE_NEXT
        volatility_regime_name="ACTIVE_NEXT",
        regime_confidence=0.80,
        extreme_warning=False,
        volatility_gate_passed=True,
        tcn_probability=0.65,
        ridge_confidence=0.60,
        momentum=0.40,
        drawdown=0.01,
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
        pa = _make_pair_analysis(gates_passed=True, direction="HOLD")
        assert pa.is_tradeable is False

    def test_not_tradeable_gates_failed(self):
        pa = _make_pair_analysis(gates_passed=False, direction="LONG")
        assert pa.is_tradeable is False

    def test_not_tradeable_with_error(self):
        pa = _make_pair_analysis(gates_passed=True, direction="LONG",
                                 error="Model load failed")
        assert pa.is_tradeable is False

    def test_not_tradeable_with_block_reason(self):
        pa = _make_pair_analysis(gates_passed=True, direction="SHORT",
                                 block_reason="QUIET_NEXT volatility")
        assert pa.is_tradeable is False

    def test_is_tradeable_override(self):
        pa = _make_pair_analysis(gates_passed=False, direction="HOLD",
                                 is_tradeable_override=True)
        assert pa.is_tradeable is True

    def test_overall_score_range(self):
        pa = _make_pair_analysis()
        score = pa.overall_score
        assert 0.0 <= score <= 1.0

    def test_overall_score_zero_on_error(self):
        pa = _make_pair_analysis(error="something broke")
        assert pa.overall_score == 0.0

    def test_overall_score_zero_on_block(self):
        pa = _make_pair_analysis(block_reason="blocked")
        assert pa.overall_score == 0.0

    def test_gate_summary_all_passed(self):
        pa = _make_pair_analysis(
            volatility_gate_passed=True,
            confidence_passed=True,
            momentum_passed=True,
            risk_passed=True,
        )
        assert pa.gate_summary == "4/4"

    def test_gate_summary_some_failed(self):
        pa = _make_pair_analysis(
            volatility_gate_passed=True,
            confidence_passed=False,
            confidence_gate_passed=False,
            momentum_passed=False,
            momentum_gate_passed=False,
            risk_passed=True,
        )
        assert pa.gate_summary == "2/4"

    def test_format_confidence(self):
        pa = _make_pair_analysis(confidence=0.72)
        assert pa.format_confidence() == "72%"

    def test_to_dict_keys(self):
        pa = _make_pair_analysis()
        d = pa.to_dict()
        assert "pair" in d
        assert "direction" in d
        assert "confidence" in d
        assert "is_tradeable" in d
        assert "overall_score" in d
        assert "volatility_regime_name" in d


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
                                direction="HOLD", volatility_regime=0),
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
# Gate helper tests
# ---------------------------------------------------------------------------

class TestGateHelpers:
    """Test gate module helper functions."""

    def test_normalize_confidence_standard(self):
        from src.scanner.gates import _normalize_confidence
        assert _normalize_confidence(50, from_scale=100) == 0.5
        assert _normalize_confidence(100, from_scale=100) == 1.0

    def test_normalize_confidence_clamps(self):
        from src.scanner.gates import _normalize_confidence
        assert _normalize_confidence(150, from_scale=100) == 1.0
        assert _normalize_confidence(-10, from_scale=100) == 0.0

    def test_normalize_confidence_zero_scale(self):
        from src.scanner.gates import _normalize_confidence
        assert _normalize_confidence(50, from_scale=0) == 0.0

    def test_format_confidence_pct(self):
        from src.scanner.gates import _format_confidence_pct
        assert _format_confidence_pct(0.65) == "65%"
        assert _format_confidence_pct(0.0) == "0%"
        assert _format_confidence_pct(1.0) == "100%"

    def test_scanner_error_is_exception(self):
        from src.scanner.gates import ScannerError
        assert issubclass(ScannerError, Exception)
        with pytest.raises(ScannerError):
            raise ScannerError("TCN model missing")


# ---------------------------------------------------------------------------
# Blocked-by-volatility tests
# ---------------------------------------------------------------------------

class TestVolatilityFiltering:
    """Test volatility regime filtering on ScanResult."""

    def test_quiet_regime_blocked(self):
        """QUIET_NEXT (0) should be blocked."""
        from src.scanner.results import ScanResult
        sr = ScanResult(analyses=[
            _make_pair_analysis(pair="EUR_USD", volatility_regime=0),
        ])
        assert len(sr.blocked_by_volatility) == 1

    def test_stable_regime_allowed(self):
        """STABLE_NEXT (1) should pass."""
        from src.scanner.results import ScanResult
        sr = ScanResult(analyses=[
            _make_pair_analysis(pair="EUR_USD", volatility_regime=1),
        ])
        assert len(sr.blocked_by_volatility) == 0

    def test_active_regime_allowed(self):
        """ACTIVE_NEXT (2) should pass."""
        from src.scanner.results import ScanResult
        sr = ScanResult(analyses=[
            _make_pair_analysis(pair="EUR_USD", volatility_regime=2),
        ])
        assert len(sr.blocked_by_volatility) == 0

    def test_extreme_regime_blocked(self):
        """EXTREME_NEXT (3) should be blocked."""
        from src.scanner.results import ScanResult
        sr = ScanResult(analyses=[
            _make_pair_analysis(pair="EUR_USD", volatility_regime=3,
                                extreme_warning=True),
        ])
        assert len(sr.blocked_by_volatility) == 1
