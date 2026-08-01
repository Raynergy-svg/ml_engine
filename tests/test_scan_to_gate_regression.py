"""Phase 37 (US-241): Regression tests for full scan-to-gate pipeline.

End-to-end regression: Scanner._scan_pair → _run_inference → GateEvaluator
→ PairAnalysis with mock models. Verifies the pipeline produces valid gate
results without real OANDA data or trained models.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.scanner.config import ScannerConfig
from src.scanner.results import PairAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ohlcv(rows=200, pair="EUR_USD"):
    """Create realistic OHLCV data matching what OANDA would return."""
    np.random.seed(42)
    close = 1.1000 + np.cumsum(np.random.randn(rows) * 0.0005)
    high = close + np.abs(np.random.randn(rows) * 0.001)
    low = close - np.abs(np.random.randn(rows) * 0.001)
    open_ = close + np.random.randn(rows) * 0.0003
    volume = np.random.randint(500, 5000, rows)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_config(**overrides):
    """Create a ScannerConfig suitable for testing."""
    defaults = dict(
        enable_execution=False,
        enable_session_filter=False,
        blocked_pairs=[],
        incremental_enabled=False,
        pairs=["EUR_USD"],
        profile="balanced",
    )
    defaults.update(overrides)
    return ScannerConfig(**defaults)


def _make_gate_result(
    all_passed=True,
    confidence=65.0,
    momentum=0.72,
    drawdown=0.02,
    transformer_direction="LONG",
    transformer_prob=0.78,
):
    """Create a mock gate evaluation result dict."""
    return {
        "all_passed": all_passed,
        "confidence": confidence,
        "confidence_passed": confidence >= 25.0,
        "momentum": momentum,
        "momentum_acceleration": momentum > 0.6,
        "momentum_passed": momentum >= 0.5,
        "risk_passed": drawdown < 0.05,
        "drawdown": drawdown,
        "transformer_direction": transformer_direction,
        "transformer_prob": transformer_prob,
    }


# ---------------------------------------------------------------------------
# Tests: Scanner initialization (no OANDA)
# ---------------------------------------------------------------------------
class TestScannerInit:
    def test_scanner_creates_with_config(self):
        """Scanner initializes without OANDA when config is provided."""
        cfg = _make_config()
        from src.scanner.engine import Scanner
        scanner = Scanner(config=cfg, oanda_client=MagicMock())
        assert scanner.config.enable_execution is False
        assert scanner._oanda is not None

    def test_scanner_defaults_to_balanced_profile(self):
        from src.scanner.engine import Scanner
        scanner = Scanner(config=_make_config(), oanda_client=MagicMock())
        assert scanner.config.profile == "balanced"


# ---------------------------------------------------------------------------
# Tests: _run_inference pipeline paths
# ---------------------------------------------------------------------------
class TestRunInferencePipeline:
    """Test _run_inference falls through ensemble → gates → technicals."""

    def _get_scanner(self):
        from src.scanner.engine import Scanner
        cfg = _make_config()
        scanner = Scanner(config=cfg, oanda_client=MagicMock())
        return scanner

    def test_technical_fallback_produces_direction(self):
        """Technicals provide direction from RSI when the fallback is ENABLED.

        2026-07-31: enable_technical_fallback now defaults to False (fail-closed)
        after the operator directive to remove the no-ML direction fallbacks, so
        this test opts in explicitly. The second half asserts the new default —
        with the flag off, a dead ML stack must abstain (HOLD) rather than trade
        an RSI heuristic.
        """
        scanner = self._get_scanner()
        df_raw = _make_ohlcv(rows=200)
        df_feat = df_raw.copy()
        # Add RSI column — oversold = LONG signal
        df_feat["rsi"] = 25.0
        df_feat["macd"] = 0.001

        scanner.config.enable_technical_fallback = True

        # Disable ensemble and gates via lazy-init patches
        with patch.object(scanner, '_init_modular_ensemble', return_value=False):
            with patch.object(scanner, '_init_gate_evaluator', return_value=False):
                with patch.object(scanner, '_check_volatility_regime', return_value=(True, 1)):
                    direction, conf, tcn, ridge, gates, regime, succeeded, details = \
                        scanner._run_inference(df_raw, df_feat, "EUR_USD")

        assert direction == "LONG"
        assert conf > 0.5  # Oversold RSI → above-baseline confidence
        assert succeeded is True
        assert gates is False  # No gates ran

        # Fail-closed default: no ML direction => no trade.
        scanner.config.enable_technical_fallback = False
        with patch.object(scanner, '_init_modular_ensemble', return_value=False):
            with patch.object(scanner, '_init_gate_evaluator', return_value=False):
                with patch.object(scanner, '_check_volatility_regime', return_value=(True, 1)):
                    direction_off, _c, _t, _r, _g, _rg, succeeded_off, _d = \
                        scanner._run_inference(df_raw, df_feat, "EUR_USD")

        assert direction_off == "HOLD"
        assert succeeded_off is False

    def test_overbought_rsi_gives_short(self):
        """RSI > 70 produces SHORT direction when the technical fallback is on.

        Opts in explicitly — enable_technical_fallback defaults to False as of
        2026-07-31 (fail-closed).
        """
        scanner = self._get_scanner()
        df_raw = _make_ohlcv(rows=200)
        df_feat = df_raw.copy()
        df_feat["rsi"] = 80.0
        scanner.config.enable_technical_fallback = True

        with patch.object(scanner, '_init_modular_ensemble', return_value=False):
            with patch.object(scanner, '_init_gate_evaluator', return_value=False):
                with patch.object(scanner, '_check_volatility_regime', return_value=(True, 1)):
                    direction, conf, *_ = scanner._run_inference(df_raw, df_feat, "EUR_USD")
        assert direction == "SHORT"
        assert conf > 0.5

    def test_gate_evaluator_path(self):
        """When gate evaluator is available, gates run and populate details."""
        scanner = self._get_scanner()
        df_raw = _make_ohlcv(rows=200)
        df_feat = df_raw.copy()
        df_feat["rsi"] = 45.0
        df_feat["macd"] = 0.001
        df_feat["adx"] = 30.0
        df_feat["atr"] = 0.0015

        # Mock gate evaluator
        mock_gate = MagicMock()
        mock_gate.evaluate_all_gates.return_value = _make_gate_result(
            all_passed=True, confidence=70.0, momentum=0.75,
            transformer_direction="LONG", transformer_prob=0.82,
        )
        scanner._gate_evaluator = mock_gate

        # Disable ensemble so we fall through to gates
        scanner._modular_ensemble = None

        # Mock _init_modular_ensemble to return False so it doesn't re-init the ensemble
        with patch.object(scanner, '_init_modular_ensemble', return_value=False):
            # Mock _init_gate_evaluator to return True (already initialized)
            with patch.object(scanner, '_init_gate_evaluator', return_value=True):
                # Mock volatility check to allow
                with patch.object(scanner, '_check_volatility_regime', return_value=(True, 1)):
                    direction, conf, tcn, ridge, gates, regime, succeeded, details = \
                        scanner._run_inference(df_raw, df_feat, "EUR_USD")

        assert direction == "LONG"
        assert gates is True
        assert conf > 0
        assert conf <= 0.95  # Clamped
        assert succeeded is True
        assert details["momentum_passed"] is True
        assert details["confidence_passed"] is True

    def test_gates_failed_still_returns_result(self):
        """When gates fail, pipeline returns result with gates_passed=False."""
        scanner = self._get_scanner()
        df_raw = _make_ohlcv(rows=200)
        df_feat = df_raw.copy()
        df_feat["rsi"] = 50.0
        df_feat["adx"] = 10.0  # Weak trend

        mock_gate = MagicMock()
        mock_gate.evaluate_all_gates.return_value = _make_gate_result(
            all_passed=False, confidence=15.0, momentum=0.25,
            drawdown=0.08,  # High drawdown
            transformer_direction="SHORT", transformer_prob=0.55,
        )
        scanner._gate_evaluator = mock_gate

        with patch.object(scanner, '_init_modular_ensemble', return_value=False):
            with patch.object(scanner, '_init_gate_evaluator', return_value=True):
                with patch.object(scanner, '_check_volatility_regime', return_value=(True, 1)):
                    direction, conf, tcn, ridge, gates, regime, succeeded, details = \
                        scanner._run_inference(df_raw, df_feat, "EUR_USD")

        assert direction == "SHORT"
        assert gates is False
        assert succeeded is True


# ---------------------------------------------------------------------------
# Tests: _scan_pair end-to-end
# ---------------------------------------------------------------------------
class TestScanPairEndToEnd:
    def _get_scanner(self):
        from src.scanner.engine import Scanner
        cfg = _make_config(
            enable_session_filter=False,
            blocked_pairs=[],
            incremental_enabled=False,
        )
        scanner = Scanner(config=cfg, oanda_client=MagicMock())
        return scanner

    def test_blocked_pair_returns_hold(self):
        """Blocked pairs get HOLD with error message."""
        from src.scanner.engine import Scanner
        cfg = _make_config(blocked_pairs=["EUR_USD"])
        scanner = Scanner(config=cfg, oanda_client=MagicMock())

        result = scanner._scan_pair("EUR_USD")
        assert result is not None
        assert result.pair == "EUR_USD"
        assert result.direction == "HOLD"
        assert result.confidence == 0.0
        assert "blocked" in result.error.lower()

    def test_no_data_returns_hold_error(self):
        """When OANDA returns no data, pair gets error PairAnalysis."""
        scanner = self._get_scanner()
        # Ensure EUR_USD is not blocked
        scanner.config.blocked_pairs = []

        with patch.object(scanner, '_fetch_pair_data', return_value=None):
            result = scanner._scan_pair("EUR_USD")

        assert result is not None
        assert result.direction == "HOLD"
        assert "no data" in result.error.lower()

    def test_full_pipeline_with_mocked_data(self):
        """Full _scan_pair with mocked OANDA data produces valid PairAnalysis."""
        scanner = self._get_scanner()
        df_raw = _make_ohlcv(rows=300)

        # Mock gate evaluator for the gate path
        mock_gate = MagicMock()
        mock_gate.evaluate_all_gates.return_value = _make_gate_result(
            all_passed=True, confidence=72.0, momentum=0.8,
            transformer_direction="LONG", transformer_prob=0.85,
        )
        scanner._gate_evaluator = mock_gate

        with patch.object(scanner, '_fetch_pair_data', return_value=df_raw):
            with patch.object(scanner, '_init_gate_evaluator', return_value=True):
                with patch.object(scanner, '_check_volatility_regime', return_value=(True, 1)):
                    result = scanner._scan_pair("EUR_USD")

        assert result is not None
        assert isinstance(result, PairAnalysis)
        assert result.pair == "EUR_USD"
        assert result.direction in ("LONG", "SHORT", "HOLD")
        # Confidence should be in valid range
        assert 0.0 <= result.confidence <= 1.0
        # ATR should be computed
        assert result.atr >= 0.0
        # SL/TP should be positive
        assert result.sl_pips >= 0.0
        assert result.tp_pips >= 0.0

    def test_pipeline_result_scores_in_valid_ranges(self):
        """Verify all numeric scores in PairAnalysis are within expected bounds."""
        scanner = self._get_scanner()
        df_raw = _make_ohlcv(rows=300)

        mock_gate = MagicMock()
        mock_gate.evaluate_all_gates.return_value = _make_gate_result(
            all_passed=True, confidence=60.0, momentum=0.65,
            transformer_direction="SHORT", transformer_prob=0.70,
        )
        scanner._gate_evaluator = mock_gate

        with patch.object(scanner, '_fetch_pair_data', return_value=df_raw):
            with patch.object(scanner, '_init_gate_evaluator', return_value=True):
                with patch.object(scanner, '_check_volatility_regime', return_value=(True, 1)):
                    result = scanner._scan_pair("EUR_USD")

        assert result is not None
        # Core confidence and probability ranges
        assert 0.0 <= result.confidence <= 1.0
        assert result.tcn_confidence >= 0.0
        assert result.ridge_confidence >= 0.0
        # Momentum score range
        assert result.momentum >= 0.0
        # Risk percentage must be non-negative
        assert result.risk_pct >= 0.0
        # Drawdown can't be negative
        assert result.drawdown >= 0.0


# ---------------------------------------------------------------------------
# Tests: PairAnalysis data integrity
# ---------------------------------------------------------------------------
class TestPairAnalysisDataIntegrity:
    def test_default_pair_analysis(self):
        """PairAnalysis defaults are safe (HOLD, zero confidence, no gates)."""
        pa = PairAnalysis(pair="EUR_USD")
        assert pa.direction == "HOLD"
        assert pa.confidence == 0.0
        assert pa.gates_passed is False
        assert pa.error is None

    def test_pair_analysis_with_gate_results(self):
        """PairAnalysis correctly stores gate evaluation results."""
        pa = PairAnalysis(
            pair="GBP_USD",
            direction="LONG",
            confidence=0.78,
            gates_passed=True,
            momentum=0.72,
            momentum_passed=True,
            confidence_score=65.0,
            confidence_passed=True,
            risk_passed=True,
            drawdown=0.015,
        )
        assert pa.gates_passed is True
        assert pa.momentum_passed is True
        assert pa.confidence_passed is True
        assert pa.risk_passed is True
        assert pa.drawdown == 0.015
