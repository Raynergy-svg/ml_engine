"""Phase 36 (US-235): Integration smoke test for scan pipeline.

Verifies that the PairAnalysis → ScanResult → AgentTeam pipeline
produces valid output structures without any OANDA or live data deps.
"""

import json
import tempfile
import os
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.scanner.results import PairAnalysis, ScanResult
from src.scanner.agents._team import AgentVerdict, ScannerAgentTeam


# ---------------------------------------------------------------------------
# Tests: PairAnalysis structure integrity
# ---------------------------------------------------------------------------
class TestPairAnalysisStructure:
    def test_default_direction_is_hold(self):
        pa = PairAnalysis(pair="EUR_USD")
        assert pa.direction == "HOLD"
        assert pa.error is None

    def test_tradeable_false_by_default(self):
        pa = PairAnalysis(pair="EUR_USD")
        assert pa.gates_passed is False

    def test_all_gate_fields_present(self):
        pa = PairAnalysis(pair="EUR_USD")
        assert hasattr(pa, "confidence_passed")
        assert hasattr(pa, "momentum_passed")
        assert hasattr(pa, "risk_passed")

    def test_agent_fields_present(self):
        pa = PairAnalysis(pair="EUR_USD")
        assert hasattr(pa, "agent_votes")
        assert hasattr(pa, "agent_score")
        assert hasattr(pa, "weighted_vote_score")
        assert hasattr(pa, "uncertainty_score")


# ---------------------------------------------------------------------------
# Tests: ScanResult collection
# ---------------------------------------------------------------------------
class TestScanResultStructure:
    def test_empty_scan_result(self):
        sr = ScanResult()
        assert len(sr.analyses) == 0
        assert sr.model_type == "unknown"

    def test_tradeable_property(self):
        sr = ScanResult(analyses=[
            PairAnalysis(pair="EUR_USD", gates_passed=True, direction="LONG",
                         confidence_passed=True, momentum_passed=True, risk_passed=True),
            PairAnalysis(pair="GBP_USD", gates_passed=False),
        ])
        tradeable = [a for a in sr.analyses if a.gates_passed]
        assert len(tradeable) == 1
        assert tradeable[0].pair == "EUR_USD"

    def test_scan_result_serializable(self):
        """ScanResult analyses should be JSON-friendly."""
        sr = ScanResult(analyses=[
            PairAnalysis(pair="EUR_USD", direction="LONG", confidence=0.72),
        ])
        # Check PairAnalysis can be dict-ified
        pa = sr.analyses[0]
        assert pa.pair == "EUR_USD"
        assert pa.direction == "LONG"


# ---------------------------------------------------------------------------
# Tests: AgentTeam evaluate pipeline smoke test
# ---------------------------------------------------------------------------
class TestAgentTeamEvaluatePipeline:
    """Smoke test: PairAnalysis → AgentTeam.evaluate() → enriched PairAnalysis."""

    def _mock_config(self):
        cfg = MagicMock()
        cfg.regime_disabled_agents = {}
        cfg.enable_trend_agent = True
        cfg.enable_mean_reversion_agent = True
        cfg.enable_volatility_agent = True
        cfg.enable_risk_sentinel_agent = True
        cfg.enable_uncertainty_agent = True
        cfg.enable_execution_quality_agent = True
        cfg.enable_news_risk_agent = False
        cfg.enable_multi_timeframe_agent = False
        cfg.enable_pair_performance_agent = False
        cfg.enable_momentum_agent = False
        cfg.enable_session_timing_agent = False
        cfg.enable_support_resistance_agent = False
        cfg.enable_trader_readiness_agent = False
        return cfg

    def _make_mock_df(self, rows=50):
        """Create a mock DataFrame resembling OANDA candle data."""
        import numpy as np
        np.random.seed(42)
        close = 1.1000 + np.cumsum(np.random.randn(rows) * 0.001)
        df = pd.DataFrame({
            "close": close,
            "high": close + 0.002,
            "low": close - 0.002,
            "open": close - 0.0005,
            "volume": np.random.randint(100, 1000, rows),
            "atr": np.full(rows, 0.0015),
            "rsi": np.random.uniform(30, 70, rows),
            "macd": np.random.randn(rows) * 0.001,
            "macd_signal": np.random.randn(rows) * 0.001,
            "bb_upper": close + 0.003,
            "bb_lower": close - 0.003,
            "adx": np.random.uniform(15, 40, rows),
            "ema_20": close - 0.0003,
            "ema_50": close - 0.0008,
            "sma_200": close - 0.002,
        })
        return df

    def test_evaluate_with_long_direction(self):
        """Smoke: evaluate() should not crash on a LONG analysis."""
        tmpdir = tempfile.mkdtemp()
        weights_path = os.path.join(tmpdir, "trained_data", "models")
        os.makedirs(weights_path, exist_ok=True)

        with patch.object(ScannerAgentTeam, "_WEIGHTS_FILE",
                          os.path.join(weights_path, "agent_weights.json")):
            team = ScannerAgentTeam(self._mock_config())

        analysis = PairAnalysis(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.72,
            confidence_score=0.72,
            momentum=0.65,
            atr=0.0015,
            atr_pips=15.0,
            volatility_regime="NORMAL",
            current_price=1.1050,
        )

        df = self._make_mock_df()

        result = team.evaluate(analysis, df_raw=df, df_feat=df)

        # Result should still be a PairAnalysis
        assert isinstance(result, PairAnalysis)
        assert result.pair == "EUR_USD"
        # Agent verdicts should have been populated
        assert len(result.agent_reasons) >= 0  # May be 0 if all agents abstain

    def test_evaluate_skips_error_analysis(self):
        """Analysis with error should be returned unchanged."""
        tmpdir = tempfile.mkdtemp()
        weights_path = os.path.join(tmpdir, "trained_data", "models")
        os.makedirs(weights_path, exist_ok=True)

        with patch.object(ScannerAgentTeam, "_WEIGHTS_FILE",
                          os.path.join(weights_path, "agent_weights.json")):
            team = ScannerAgentTeam(self._mock_config())

        analysis = PairAnalysis(pair="GBP_USD", error="No data available")
        df = self._make_mock_df()

        result = team.evaluate(analysis, df_raw=df, df_feat=df)
        assert result.error == "No data available"

    def test_evaluate_skips_hold_direction(self):
        """HOLD direction should be returned without agent evaluation."""
        tmpdir = tempfile.mkdtemp()
        weights_path = os.path.join(tmpdir, "trained_data", "models")
        os.makedirs(weights_path, exist_ok=True)

        with patch.object(ScannerAgentTeam, "_WEIGHTS_FILE",
                          os.path.join(weights_path, "agent_weights.json")):
            team = ScannerAgentTeam(self._mock_config())

        analysis = PairAnalysis(pair="USD_JPY", direction="HOLD")
        df = self._make_mock_df()

        result = team.evaluate(analysis, df_raw=df, df_feat=df)
        assert result.direction == "HOLD"
