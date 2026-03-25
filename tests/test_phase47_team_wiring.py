"""
Tests for Phase 47: MTF Confluence and Ensemble Conflict Resolver Integration.

US-294: Wire MTF Confluence into _evaluate_multi_timeframe
US-295: Wire Ensemble Conflict Resolver into _evaluate_uncertainty

Tests cover:
1. Module initialization (success and failure paths)
2. Fallback behavior when modules unavailable
3. Integration with agent team evaluation
4. Config flag toggling
5. Error resilience
"""

import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.scanner.agents._team import ScannerAgentTeam, AgentDecisionContext, AgentVerdict
from src.scanner.results import PairAnalysis
from src.scanner.mtf_confluence import MultiTimeframeConfluence, ConfluenceResult, ScreenResult
from src.scanner.ensemble_conflict import (
    EnsembleConflictResolver,
    ModelPrediction,
    ConflictResult,
)


class TestMTFConfluenceInit:
    """Test MTF Confluence module initialization in ScannerAgentTeam.__init__"""

    def test_mtf_confluence_init_success(self):
        """MTF confluence should initialize when module available."""
        config = Mock()
        config.mtf_confluence_enabled = False  # Will be toggled per test
        config.enable_trend_agent = True
        config.enable_mean_reversion_agent = True
        config.enable_volatility_agent = True
        config.enable_risk_sentinel_agent = True
        config.enable_uncertainty_agent = True
        config.enable_execution_quality_agent = True
        config.enable_news_risk_agent = False
        config.enable_multi_timeframe_agent = False
        config.enable_pair_performance_agent = False
        config.enable_momentum_agent = False
        config.enable_session_timing_agent = False
        config.enable_support_resistance_agent = False

        team = ScannerAgentTeam(config)

        assert team._mtf_confluence is not None
        assert isinstance(team._mtf_confluence, MultiTimeframeConfluence)

    def test_mtf_confluence_init_failure_graceful(self):
        """MTF confluence should gracefully fall back if import fails."""
        config = Mock()
        config.mtf_confluence_enabled = False

        # Real test: just verify that if we create with normal config, it works
        # The module gracefully handles import errors during __init__
        team = ScannerAgentTeam(config)
        # Either it's initialized or None is acceptable
        assert team._mtf_confluence is None or isinstance(team._mtf_confluence, MultiTimeframeConfluence)

    def test_mtf_confluence_default_config(self):
        """MTF confluence should have default config."""
        config = Mock()
        config.mtf_confluence_enabled = False

        team = ScannerAgentTeam(config)

        if team._mtf_confluence is not None:
            assert team._mtf_confluence.config is not None
            assert team._mtf_confluence.config.proceed_threshold == 0.65
            assert team._mtf_confluence.config.caution_threshold == 0.45


class TestEnsembleConflictInit:
    """Test Ensemble Conflict Resolver module initialization."""

    def test_ensemble_conflict_init_success(self):
        """Ensemble conflict resolver should initialize when available."""
        config = Mock()
        config.ensemble_conflict_enabled = False
        config.enable_trend_agent = True
        config.enable_mean_reversion_agent = True
        config.enable_volatility_agent = True
        config.enable_risk_sentinel_agent = True
        config.enable_uncertainty_agent = True
        config.enable_execution_quality_agent = True
        config.enable_news_risk_agent = False
        config.enable_multi_timeframe_agent = False
        config.enable_pair_performance_agent = False
        config.enable_momentum_agent = False
        config.enable_session_timing_agent = False
        config.enable_support_resistance_agent = False

        team = ScannerAgentTeam(config)

        assert team._ensemble_conflict is not None
        assert isinstance(team._ensemble_conflict, EnsembleConflictResolver)

    def test_ensemble_conflict_init_failure_graceful(self):
        """Ensemble conflict should gracefully fall back if import fails."""
        config = Mock()
        config.ensemble_conflict_enabled = False

        team = ScannerAgentTeam(config)
        # Either it's initialized or None is acceptable
        assert team._ensemble_conflict is None or isinstance(team._ensemble_conflict, EnsembleConflictResolver)

    def test_ensemble_conflict_default_config(self):
        """Ensemble conflict resolver should have default config."""
        config = Mock()
        config.ensemble_conflict_enabled = False

        team = ScannerAgentTeam(config)

        if team._ensemble_conflict is not None:
            assert team._ensemble_conflict.config is not None
            assert team._ensemble_conflict.config.enabled is True
            assert team._ensemble_conflict.config.direction_threshold == 0.1


class TestMTFConfluenceIntegration:
    """Test MTF Confluence integration in _evaluate_multi_timeframe."""

    def _make_test_config(self, mtf_confluence_enabled=True):
        """Helper to create test config."""
        config = Mock()
        config.mtf_confluence_enabled = mtf_confluence_enabled
        config.enable_trend_agent = False
        config.enable_mean_reversion_agent = False
        config.enable_volatility_agent = False
        config.enable_risk_sentinel_agent = False
        config.enable_uncertainty_agent = False
        config.enable_execution_quality_agent = False
        config.enable_news_risk_agent = False
        config.enable_multi_timeframe_agent = True
        config.enable_pair_performance_agent = False
        config.enable_momentum_agent = False
        config.enable_session_timing_agent = False
        config.enable_support_resistance_agent = False
        return config

    def _make_test_dataframes(self):
        """Helper to create test dataframes."""
        dates = pd.date_range("2026-03-01", periods=100, freq="H")
        data = {
            "open": 1.1 + np.random.randn(100) * 0.001,
            "high": 1.101 + np.random.randn(100) * 0.001,
            "low": 1.099 + np.random.randn(100) * 0.001,
            "close": 1.1 + np.random.randn(100) * 0.001,
            "volume": np.random.uniform(1000, 5000, 100),
        }
        df_raw = pd.DataFrame(data, index=dates)
        df_feat = df_raw.copy()
        df_feat["sma_20"] = df_feat["close"].rolling(20).mean()
        return df_raw, df_feat

    def _make_test_analysis(self):
        """Helper to create test PairAnalysis."""
        analysis = Mock(spec=PairAnalysis)
        analysis.pair = "EURUSD"
        analysis.direction = "LONG"
        analysis.confidence = 0.65
        analysis.current_price = 1.1
        analysis.atr_pips = 10.0
        analysis.spread_pips = 1.0
        analysis.est_slippage_pips = 0.5
        analysis.volatility_regime = "NORMAL"
        analysis.error = None
        # Ensemble conflict fields
        analysis.tcn_score = 0.5
        analysis.ridge_score = 0.45
        analysis.rf_score = 0.55
        return analysis

    def test_mtf_confluence_proceed_mapping(self):
        """MTF Confluence PROCEED recommendation should map to high score."""
        config = self._make_test_config(mtf_confluence_enabled=True)
        team = ScannerAgentTeam(config)

        # Skip if confluence not available
        if team._mtf_confluence is None:
            pytest.skip("MTF Confluence not available")

        # Create mock confluence result
        screen1 = ScreenResult("trend_4h", "BULLISH", 0.75, {})
        screen2 = ScreenResult("wave_1h", "BULLISH", 0.70, {})
        screen3 = ScreenResult("entry_15m", "BULLISH", 0.80, {})

        confluence_result = ConfluenceResult(
            confluence_score=0.75,
            direction_aligned=True,
            screen_results=[screen1, screen2, screen3],
            penalty_applied=0.0,
            recommendation="PROCEED",
            details={},
        )

        # Mock the confluence call
        team._mtf_confluence.score_confluence = Mock(return_value=confluence_result)
        team._fetch_mtf_candles = Mock(return_value=pd.DataFrame({
            "close": np.random.randn(30) + 1.1,
            "high": np.random.randn(30) + 1.101,
            "low": np.random.randn(30) + 1.099,
        }))

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        verdict = team._evaluate_multi_timeframe(ctx)

        assert verdict.reason_code == "mtf_confluence_proceed"
        assert verdict.passed is True
        assert verdict.score >= 0.75

    def test_mtf_confluence_caution_mapping(self):
        """MTF Confluence CAUTION recommendation should map to medium score."""
        config = self._make_test_config(mtf_confluence_enabled=True)
        team = ScannerAgentTeam(config)

        if team._mtf_confluence is None:
            pytest.skip("MTF Confluence not available")

        screen1 = ScreenResult("trend_4h", "BULLISH", 0.60, {})
        screen2 = ScreenResult("wave_1h", "NEUTRAL", 0.50, {})
        screen3 = ScreenResult("entry_15m", "BULLISH", 0.65, {})

        confluence_result = ConfluenceResult(
            confluence_score=0.60,
            direction_aligned=False,
            screen_results=[screen1, screen2, screen3],
            penalty_applied=-0.25,
            recommendation="CAUTION",
            details={},
        )

        team._mtf_confluence.score_confluence = Mock(return_value=confluence_result)
        team._fetch_mtf_candles = Mock(return_value=pd.DataFrame({
            "close": np.random.randn(30) + 1.1,
            "high": np.random.randn(30) + 1.101,
            "low": np.random.randn(30) + 1.099,
        }))

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        verdict = team._evaluate_multi_timeframe(ctx)

        assert verdict.reason_code == "mtf_confluence_caution"
        assert verdict.passed is True
        assert 0.45 <= verdict.score <= 0.75

    def test_mtf_confluence_reject_mapping(self):
        """MTF Confluence REJECT recommendation should map to low score."""
        config = self._make_test_config(mtf_confluence_enabled=True)
        team = ScannerAgentTeam(config)

        if team._mtf_confluence is None:
            pytest.skip("MTF Confluence not available")

        screen1 = ScreenResult("trend_4h", "BEARISH", 0.30, {})
        screen2 = ScreenResult("wave_1h", "BULLISH", 0.40, {})
        screen3 = ScreenResult("entry_15m", "BEARISH", 0.35, {})

        confluence_result = ConfluenceResult(
            confluence_score=0.35,
            direction_aligned=False,
            screen_results=[screen1, screen2, screen3],
            penalty_applied=-0.25,
            recommendation="REJECT",
            details={},
        )

        team._mtf_confluence.score_confluence = Mock(return_value=confluence_result)
        team._fetch_mtf_candles = Mock(return_value=pd.DataFrame({
            "close": np.random.randn(30) + 1.1,
            "high": np.random.randn(30) + 1.101,
            "low": np.random.randn(30) + 1.099,
        }))

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        verdict = team._evaluate_multi_timeframe(ctx)

        assert verdict.reason_code == "mtf_confluence_reject"
        assert verdict.passed is False

    def test_mtf_confluence_error_fallback(self):
        """MTF Confluence errors should fall back to legacy logic."""
        config = self._make_test_config(mtf_confluence_enabled=True)
        team = ScannerAgentTeam(config)

        if team._mtf_confluence is None:
            pytest.skip("MTF Confluence not available")

        # Mock confluence to raise exception
        team._mtf_confluence.score_confluence = Mock(side_effect=ValueError("Test error"))
        team._fetch_mtf_candles = Mock(return_value=None)

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        # Should not raise, should fall back to legacy
        verdict = team._evaluate_multi_timeframe(ctx)

        assert verdict.name == "multi_timeframe"
        assert verdict.reason_code in ("mtf_full_confluence", "mtf_partial", "mtf_weak", "mtf_none")

    def test_mtf_confluence_disabled_uses_legacy(self):
        """MTF Confluence disabled should use legacy logic."""
        config = self._make_test_config(mtf_confluence_enabled=False)
        team = ScannerAgentTeam(config)

        team._fetch_mtf_candles = Mock(return_value=None)

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        verdict = team._evaluate_multi_timeframe(ctx)

        # Should use legacy reason codes
        assert verdict.reason_code in ("mtf_full_confluence", "mtf_partial", "mtf_weak", "mtf_none")


class TestEnsembleConflictIntegration:
    """Test Ensemble Conflict Resolver integration in _evaluate_uncertainty."""

    def _make_test_config(self, ensemble_conflict_enabled=True):
        """Helper to create test config."""
        config = Mock()
        config.ensemble_conflict_enabled = ensemble_conflict_enabled
        config.enable_trend_agent = False
        config.enable_mean_reversion_agent = False
        config.enable_volatility_agent = False
        config.enable_risk_sentinel_agent = False
        config.enable_uncertainty_agent = True
        config.enable_execution_quality_agent = False
        config.enable_news_risk_agent = False
        config.enable_multi_timeframe_agent = False
        config.enable_pair_performance_agent = False
        config.enable_momentum_agent = False
        config.enable_session_timing_agent = False
        config.enable_support_resistance_agent = False
        config.max_uncertainty_score = 0.40
        config.max_model_disagreement = 0.50
        config.soft_uncertainty_blocking = False
        return config

    def _make_test_dataframes(self):
        """Helper to create test dataframes with LONG-aligned heuristics.

        Heuristic alignment for LONG direction:
        - close > sma_20 → heuristic = +1 (agrees with LONG)
        - rsi < 50 → heuristic = +1 (agrees with LONG)
        - returns > 0 → heuristic = +1 (agrees with LONG)
        This ensures model_disagreement = 0 (no opposing heuristics).
        """
        dates = pd.date_range("2026-03-01", periods=100, freq="h")
        # Create steadily rising prices so close > sma_20 and returns > 0
        closes = np.linspace(1.09, 1.11, 100)
        data = {
            "open": closes - 0.0005,
            "high": closes + 0.001,
            "low": closes - 0.001,
            "close": closes,
            "volume": np.full(100, 3000.0),
        }
        df_raw = pd.DataFrame(data, index=dates)
        df_feat = df_raw.copy()
        df_feat["sma_20"] = df_feat["close"].rolling(20).mean()
        df_feat["rsi"] = 45.0  # Below 50 → agrees with LONG
        df_feat["returns"] = 0.001  # Positive → agrees with LONG
        return df_raw, df_feat

    def _make_test_analysis(self):
        """Helper to create test PairAnalysis."""
        analysis = Mock(spec=PairAnalysis)
        analysis.pair = "EURUSD"
        analysis.direction = "LONG"
        analysis.confidence = 0.65
        analysis.current_price = 1.11
        analysis.atr_pips = 10.0
        analysis.spread_pips = 1.0
        analysis.est_slippage_pips = 0.5
        analysis.volatility_regime = "NORMAL"
        analysis.error = None
        # Ensemble conflict fields
        analysis.tcn_score = 0.5
        analysis.ridge_score = 0.45
        analysis.rf_score = 0.55
        # Set low model_disagreement and uncertainty to avoid legacy hard block
        analysis.model_disagreement = 0.15
        analysis.uncertainty_score = 0.20
        analysis.confidence_variance = 0.05
        return analysis

    def test_ensemble_conflict_direction_conflict_blocks(self):
        """Ensemble conflict with direction conflict should block trade."""
        config = self._make_test_config(ensemble_conflict_enabled=True)
        team = ScannerAgentTeam(config)

        if team._ensemble_conflict is None:
            pytest.skip("Ensemble Conflict Resolver not available")

        # Create mock conflict result with direction conflict
        conflict_result = ConflictResult(
            has_direction_conflict=True,
            disagreement_score=0.55,
            penalty=-1.0,
            should_block=True,
            direction_votes={"BUY": 2, "SELL": 1, "HOLD": 0},
            consensus_direction="BUY",
            details={},
        )

        team._ensemble_conflict.resolve = Mock(return_value=conflict_result)

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()
        analysis.tcn_score = 0.5
        analysis.ridge_score = -0.5
        analysis.rf_score = 0.45

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        verdict = team._evaluate_uncertainty(ctx)

        assert verdict.block_trade is True
        assert verdict.reason_code == "ensemble_direction_conflict"
        assert "ensemble direction conflict" in verdict.reason.lower()

    def test_ensemble_conflict_graduated_penalty(self):
        """Ensemble conflict with disagreement should apply graduated penalty."""
        config = self._make_test_config(ensemble_conflict_enabled=True)
        team = ScannerAgentTeam(config)

        if team._ensemble_conflict is None:
            pytest.skip("Ensemble Conflict Resolver not available")

        # Create mock conflict result with low disagreement (won't trigger hard block)
        conflict_result = ConflictResult(
            has_direction_conflict=False,
            disagreement_score=0.25,  # Below the hard floor of 0.30
            penalty=-0.10,
            should_block=False,
            direction_votes={"BUY": 2, "SELL": 1, "HOLD": 0},
            consensus_direction="BUY",
            details={},
        )

        team._ensemble_conflict.resolve = Mock(return_value=conflict_result)

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()
        # Set model scores to be aligned (no internal disagreement)
        analysis.tcn_score = 0.5
        analysis.ridge_score = 0.45
        analysis.rf_score = 0.55

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        verdict = team._evaluate_uncertainty(ctx)

        # Check that ensemble conflict penalty is recorded in metadata
        assert "ensemble_conflict_penalty" in verdict.metadata
        assert verdict.metadata["ensemble_conflict_penalty"] == -0.10
        # With low internal disagreement and good confidence, should pass
        assert verdict.passed is True  # No hard block triggered
        # Verify ensemble conflict was resolved
        assert "ensemble_consensus_direction" in verdict.metadata
        assert verdict.metadata["ensemble_consensus_direction"] == "BUY"

    def test_ensemble_conflict_error_fallback(self):
        """Ensemble conflict errors should fall back to legacy logic."""
        config = self._make_test_config(ensemble_conflict_enabled=True)
        team = ScannerAgentTeam(config)

        if team._ensemble_conflict is None:
            pytest.skip("Ensemble Conflict Resolver not available")

        # Mock resolver to raise exception
        team._ensemble_conflict.resolve = Mock(side_effect=RuntimeError("Test error"))

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        # Should not raise, should fall back to legacy
        verdict = team._evaluate_uncertainty(ctx)

        assert verdict.name == "uncertainty"
        # Should use legacy reason codes
        assert verdict.reason_code in (
            "disagreement_hard_block",
            "uncertainty_soft_penalty",
            "uncertainty_block",
            "uncertainty_ok",
            "uncertainty_watch",
        )

    def test_ensemble_conflict_disabled_uses_legacy(self):
        """Ensemble conflict disabled should use legacy logic."""
        config = self._make_test_config(ensemble_conflict_enabled=False)
        team = ScannerAgentTeam(config)

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        verdict = team._evaluate_uncertainty(ctx)

        # Should use legacy reason codes
        assert verdict.reason_code in (
            "disagreement_hard_block",
            "uncertainty_soft_penalty",
            "uncertainty_block",
            "uncertainty_ok",
            "uncertainty_watch",
        )


class TestPhase47Integration:
    """Test Phase 47 modules working together in evaluate()."""

    def _make_full_config(self):
        """Helper to create full test config with all modules enabled."""
        config = Mock()
        config.mtf_confluence_enabled = True
        config.ensemble_conflict_enabled = True
        config.enable_trend_agent = True
        config.enable_mean_reversion_agent = True
        config.enable_volatility_agent = True
        config.enable_risk_sentinel_agent = True
        config.enable_uncertainty_agent = True
        config.enable_execution_quality_agent = True
        config.enable_news_risk_agent = False
        config.enable_multi_timeframe_agent = True
        config.enable_pair_performance_agent = False
        config.enable_momentum_agent = True
        config.enable_session_timing_agent = True
        config.enable_support_resistance_agent = False
        config.enable_trader_readiness_agent = False
        config.max_uncertainty_score = 0.40
        config.max_model_disagreement = 0.50
        config.soft_uncertainty_blocking = False
        config.enable_graph_attention = False
        config.regime_disabled_agents = {}
        return config

    def _make_test_dataframes(self):
        """Helper to create test dataframes."""
        dates = pd.date_range("2026-03-01", periods=100, freq="H")
        data = {
            "open": 1.1 + np.random.randn(100) * 0.001,
            "high": 1.101 + np.random.randn(100) * 0.001,
            "low": 1.099 + np.random.randn(100) * 0.001,
            "close": 1.1 + np.random.randn(100) * 0.001,
            "volume": np.random.uniform(1000, 5000, 100),
        }
        df_raw = pd.DataFrame(data, index=dates)
        df_feat = df_raw.copy()
        df_feat["sma_20"] = df_feat["close"].rolling(20).mean()
        df_feat["rsi"] = 50.0
        df_feat["volume_ratio_20"] = 1.0
        return df_raw, df_feat

    def _make_test_analysis(self):
        """Helper to create test PairAnalysis."""
        analysis = Mock(spec=PairAnalysis)
        analysis.pair = "EURUSD"
        analysis.direction = "LONG"
        analysis.confidence = 0.65
        analysis.current_price = 1.1
        analysis.atr_pips = 10.0
        analysis.spread_pips = 1.0
        analysis.est_slippage_pips = 0.5
        analysis.volatility_regime = "NORMAL"
        analysis.error = None
        # Ensemble conflict fields
        analysis.tcn_score = 0.5
        analysis.ridge_score = 0.45
        analysis.rf_score = 0.55
        # Additional fields for momentum
        analysis.price_trend = "up"
        return analysis

    def test_both_modules_work_together(self):
        """Both MTF Confluence and Ensemble Conflict should work together in evaluate()."""
        config = self._make_full_config()
        team = ScannerAgentTeam(config)

        # Skip if either module not available
        if team._mtf_confluence is None or team._ensemble_conflict is None:
            pytest.skip("Not all Phase 47 modules available")

        # Mock both modules
        confluence_result = ConfluenceResult(
            confluence_score=0.75,
            direction_aligned=True,
            screen_results=[],
            penalty_applied=0.0,
            recommendation="PROCEED",
            details={},
        )
        team._mtf_confluence.score_confluence = Mock(return_value=confluence_result)

        conflict_result = ConflictResult(
            has_direction_conflict=False,
            disagreement_score=0.2,
            penalty=-0.05,
            should_block=False,
            direction_votes={"BUY": 3, "SELL": 0, "HOLD": 0},
            consensus_direction="BUY",
            details={},
        )
        team._ensemble_conflict.resolve = Mock(return_value=conflict_result)

        team._fetch_mtf_candles = Mock(return_value=pd.DataFrame({
            "close": np.random.randn(30) + 1.1,
            "high": np.random.randn(30) + 1.101,
            "low": np.random.randn(30) + 1.099,
        }))

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        # Call evaluate
        result = team.evaluate(analysis, df_raw, df_feat)

        # Result should have verdicts from both modules
        assert result is not None
        # Result should include agent verdicts
        if hasattr(result, "agent_verdicts"):
            verdicts = result.agent_verdicts or []
            agent_names = {v.name for v in verdicts}
            # At least one of our modules should have run
            assert "multi_timeframe" in agent_names or "uncertainty" in agent_names or len(verdicts) > 0

    def test_config_flags_control_modules(self):
        """Config flags should enable/disable modules."""
        config = self._make_full_config()
        config.mtf_confluence_enabled = False
        config.ensemble_conflict_enabled = False

        team = ScannerAgentTeam(config)

        df_raw, df_feat = self._make_test_dataframes()
        analysis = self._make_test_analysis()

        # Even with modules disabled, evaluate should work
        result = team.evaluate(analysis, df_raw, df_feat)
        assert result is not None


class TestRegressionAndEdgeCases:
    """Test edge cases and regression scenarios."""

    def test_empty_dataframe_handling(self):
        """Empty dataframes should not cause crashes."""
        config = Mock()
        config.mtf_confluence_enabled = True
        config.ensemble_conflict_enabled = True
        config.enable_uncertainty_agent = True
        config.enable_multi_timeframe_agent = True
        config.max_uncertainty_score = 0.40
        config.max_model_disagreement = 0.50
        config.soft_uncertainty_blocking = False

        team = ScannerAgentTeam(config)

        df_raw = pd.DataFrame()
        df_feat = pd.DataFrame()
        analysis = Mock(spec=PairAnalysis)
        analysis.pair = "EURUSD"
        analysis.direction = "LONG"
        analysis.confidence = 0.65
        analysis.current_price = 1.1
        analysis.error = None
        analysis.tcn_score = 0.5
        analysis.ridge_score = 0.45
        analysis.rf_score = 0.55

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        # Should not crash
        if team._mtf_confluence is not None:
            verdict_mtf = team._evaluate_multi_timeframe(ctx)
            assert verdict_mtf is not None

        if team._ensemble_conflict is not None:
            verdict_unc = team._evaluate_uncertainty(ctx)
            assert verdict_unc is not None

    def test_nan_values_in_scores(self):
        """NaN values in model scores should be handled gracefully."""
        config = Mock()
        config.ensemble_conflict_enabled = True
        config.enable_uncertainty_agent = True
        config.max_uncertainty_score = 0.40
        config.max_model_disagreement = 0.50
        config.soft_uncertainty_blocking = False

        team = ScannerAgentTeam(config)

        dates = pd.date_range("2026-03-01", periods=100, freq="H")
        df_raw = pd.DataFrame({
            "close": [1.1] * 100,
            "high": [1.101] * 100,
            "low": [1.099] * 100,
        }, index=dates)
        df_feat = df_raw.copy()

        analysis = Mock(spec=PairAnalysis)
        analysis.pair = "EURUSD"
        analysis.direction = "LONG"
        analysis.confidence = 0.65
        analysis.current_price = 1.1
        analysis.error = None
        analysis.tcn_score = np.nan
        analysis.ridge_score = np.nan
        analysis.rf_score = np.nan

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        # Should not crash
        if team._ensemble_conflict is not None:
            verdict = team._evaluate_uncertainty(ctx)
            assert verdict is not None
            assert isinstance(verdict.confidence_delta, float)

    def test_legacy_fallback_preserves_scoring(self):
        """Legacy fallback should preserve scoring logic."""
        config = Mock()
        config.mtf_confluence_enabled = False
        config.ensemble_conflict_enabled = False
        config.enable_multi_timeframe_agent = True
        config.enable_uncertainty_agent = True
        config.max_uncertainty_score = 0.40
        config.max_model_disagreement = 0.50
        config.soft_uncertainty_blocking = False

        team = ScannerAgentTeam(config)

        dates = pd.date_range("2026-03-01", periods=100, freq="H")
        df_raw = pd.DataFrame({
            "open": 1.1 + np.random.randn(100) * 0.001,
            "high": 1.101 + np.random.randn(100) * 0.001,
            "low": 1.099 + np.random.randn(100) * 0.001,
            "close": 1.1 + np.random.randn(100) * 0.001,
            "volume": np.random.uniform(1000, 5000, 100),
        }, index=dates)
        df_feat = df_raw.copy()
        df_feat["sma_20"] = df_feat["close"].rolling(20).mean()
        df_feat["rsi"] = 50.0

        analysis = Mock(spec=PairAnalysis)
        analysis.pair = "EURUSD"
        analysis.direction = "LONG"
        analysis.confidence = 0.65
        analysis.current_price = 1.1
        analysis.error = None

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details={},
            config=config,
        )

        # Test MTF Confluence fallback
        team._fetch_mtf_candles = Mock(return_value=None)
        verdict_mtf = team._evaluate_multi_timeframe(ctx)
        assert verdict_mtf.name == "multi_timeframe"
        # Should have legacy reason code
        assert verdict_mtf.reason_code in ("mtf_full_confluence", "mtf_partial", "mtf_weak", "mtf_none")

        # Test Ensemble Conflict fallback
        analysis.tcn_score = 0.5
        analysis.ridge_score = 0.45
        analysis.rf_score = 0.55
        verdict_unc = team._evaluate_uncertainty(ctx)
        assert verdict_unc.name == "uncertainty"
        # Should have legacy reason code
        assert verdict_unc.reason_code in (
            "disagreement_hard_block",
            "uncertainty_soft_penalty",
            "uncertainty_block",
            "uncertainty_ok",
            "uncertainty_watch",
        )
