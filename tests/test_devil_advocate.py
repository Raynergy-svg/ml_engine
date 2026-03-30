"""Unit tests for the Devil's Advocate agent.

Tests all 8 bear case factors and the overall scoring logic.
"""

import unittest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.scanner.agents._team import AgentDecisionContext, _clip01
from src.scanner.agents.agent_da import DevilsAdvocateAgent
from src.scanner.results import PairAnalysis


class TestDevilsAdvocateAgent(unittest.TestCase):
    """Test suite for DevilsAdvocateAgent."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.agent = DevilsAdvocateAgent()

        # Create minimal mock dataframes
        import pandas as pd
        self.df_raw = pd.DataFrame({
            "close": [1.0800, 1.0805, 1.0810],
            "high": [1.0810, 1.0815, 1.0820],
            "low": [1.0790, 1.0795, 1.0800],
        })
        self.df_feat = pd.DataFrame({
            "sma_20": [1.0800, 1.0805, 1.0810],
            "sma_50": [1.0790, 1.0795, 1.0800],
            "rsi": [50.0, 55.0, 60.0],
            "adx": [25.0, 28.0, 30.0],
        })

    def _make_context(
        self,
        pair: str = "EUR_USD",
        direction: str = "LONG",
        confidence: float = 0.65,
        sl_pips: float = 15.0,
        tp_pips: float = 30.0,
        **extra_kwargs
    ) -> AgentDecisionContext:
        """Helper to create AgentDecisionContext with defaults."""
        # Extract special non-dataclass fields that need setattr
        news_risk_score = extra_kwargs.pop("news_risk_score", 0.30)
        spread_median_pips = extra_kwargs.pop("spread_median_pips", None)
        support_resistance_data = extra_kwargs.pop("support_resistance_data", None)

        # Extract dataclass fields with defaults
        current_price = extra_kwargs.pop("current_price", 1.0810)
        atr_pips = extra_kwargs.pop("atr_pips", 10.0)
        volatility_regime = extra_kwargs.pop("volatility_regime", "NORMAL")
        uncertainty_score = extra_kwargs.pop("uncertainty_score", 0.25)
        model_disagreement = extra_kwargs.pop("model_disagreement", 0.15)
        spread_pips = extra_kwargs.pop("spread_pips", 0.5)

        # Build PairAnalysis
        analysis = PairAnalysis(
            pair=pair,
            direction=direction,
            confidence=confidence,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            current_price=current_price,
            atr_pips=atr_pips,
            volatility_regime=volatility_regime,
            uncertainty_score=uncertainty_score,
            model_disagreement=model_disagreement,
            spread_pips=spread_pips,
            **extra_kwargs
        )

        # Set attributes that don't exist in PairAnalysis dataclass using setattr
        setattr(analysis, "news_risk_score", news_risk_score)
        if spread_median_pips is not None:
            setattr(analysis, "spread_median_pips", spread_median_pips)
        if support_resistance_data is not None:
            setattr(analysis, "support_resistance_data", support_resistance_data)

        config = Mock()
        config.min_atr_pips = 5.0

        return AgentDecisionContext(
            analysis=analysis,
            df_raw=self.df_raw,
            df_feat=self.df_feat,
            gate_details={},
            config=config,
        )

    def test_all_factors_below_threshold_pass(self) -> None:
        """Test: All bear factors below threshold → DA_PASS."""
        ctx = self._make_context(
            direction="LONG",
            confidence=0.70,
            sl_pips=15.0,
            tp_pips=30.0,
            uncertainty_score=0.20,
            model_disagreement=0.10,
            news_risk_score=0.20,
        )

        verdict = self.agent.evaluate(ctx)

        self.assertTrue(verdict.passed, "Should pass when all factors low")
        self.assertEqual(verdict.reason_code, "DA_PASS")
        self.assertFalse(verdict.block_trade, "Should not block trade")
        self.assertGreater(verdict.score, 0.75, "Agent score should be high (low bear score)")

    def test_multiple_factors_triggered_warn(self) -> None:
        """Test: Multiple factors triggered → DA_WARN."""
        ctx = self._make_context(
            direction="LONG",
            uncertainty_score=0.58,  # High uncertainty (factor ~0.9)
            model_disagreement=0.39,  # High disagreement (factor ~0.93)
            sl_pips=20.0,
            tp_pips=24.0,  # RR = 1.20 (factor ~1.0)
            volatility_regime="HIGH",  # LONG in HIGH vol (factor ~0.5)
        )
        ctx.gate_details["episodic_suppression"] = True  # Episodic (factor ~1.0)

        verdict = self.agent.evaluate(ctx)

        # With multiple factors, should soft-veto (bear_score ~0.496)
        self.assertFalse(verdict.passed, "Should fail with multiple bear factors")
        self.assertEqual(verdict.reason_code, "DA_WARN")
        self.assertFalse(verdict.block_trade, "Soft veto should not hard-block")
        self.assertGreater(verdict.metadata.get("bear_score", 0.0), 0.40, "Should exceed warn threshold")
        self.assertLess(verdict.score, 0.65, "Agent score reduced by bear factors")
        self.assertLess(verdict.confidence_delta, 0.0, "Confidence delta should be negative")

    def test_high_bear_score_blocks_trade(self) -> None:
        """Test: High bear score (>= 0.60) → DA_BLOCK, block_trade=True."""
        # Accumulate multiple strong bear factors to exceed 0.60:
        # uncertainty + ensemble + regime + episodic + spread + news
        ctx = self._make_context(
            direction="LONG",
            uncertainty_score=0.58,  # Strong: (0.58-0.4)/(0.6-0.4) = 0.9
            model_disagreement=0.39,  # Strong: (0.39-0.25)/(0.4-0.25) = 0.933
            volatility_regime="EXTREME",  # EXTREME vol → 0.25 factor
            sl_pips=20.0,
            tp_pips=24.0,  # RR = 1.20 (factor ~1.0)
            spread_pips=3.0,  # Add spread limit factor
            news_risk_score=0.72,  # High news risk (>0.60)
        )
        ctx.analysis.spread_median_pips = 1.8  # Spread > 1.5x median
        ctx.gate_details["episodic_suppression"] = True  # Episodic: 1.0 factor
        ctx.gate_details["news_event_countdown_minutes"] = 60.0  # Within 120 min

        verdict = self.agent.evaluate(ctx)

        self.assertFalse(verdict.passed, "Should fail")
        self.assertEqual(verdict.reason_code, "DA_BLOCK")
        self.assertTrue(verdict.block_trade, "Should hard-block trade")
        self.assertGreaterEqual(
            verdict.metadata.get("bear_score", 0.0), 0.60,
            "Bear score should exceed or equal block threshold"
        )

    def test_missing_data_no_crash(self) -> None:
        """Test: Missing data (None values) → no crash, DA_PASS."""
        ctx = self._make_context(
            uncertainty_score=None,
            model_disagreement=None,
            news_risk_score=None,
            volatility_regime=None,
        )
        ctx.gate_details = {}

        # Should not raise
        verdict = self.agent.evaluate(ctx)

        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.name, "devil_advocate")
        # With all None, should be mostly passing
        self.assertTrue(verdict.passed or verdict.reason_code == "DA_WARN")

    def test_episodic_suppression_flag(self) -> None:
        """Test: episodic_suppression=True in gate_details → factor fires."""
        ctx = self._make_context()
        ctx.gate_details["episodic_suppression"] = True

        verdict = self.agent.evaluate(ctx)

        # Episodic alone may not block, but should reduce score
        self.assertIn("episodic_pattern_warning", verdict.metadata.get("triggered_factors", []))
        self.assertLess(verdict.score, 1.0, "Score should be reduced")

    def test_news_risk_amplified_fires(self) -> None:
        """Test: news_risk_score > 0.60 AND event within 120 min → factor fires."""
        ctx = self._make_context(
            news_risk_score=0.70,
        )
        ctx.gate_details["news_event_countdown_minutes"] = 60.0

        verdict = self.agent.evaluate(ctx)

        triggered = verdict.metadata.get("triggered_factors", [])
        self.assertIn("news_risk_amplified", triggered, "Should detect amplified news risk")

    def test_regime_mismatch_fires(self) -> None:
        """Test: LONG in HIGH volatility regime → regime_mismatch factor fires."""
        ctx = self._make_context(
            direction="LONG",
            volatility_regime="HIGH",
        )

        verdict = self.agent.evaluate(ctx)

        triggered = verdict.metadata.get("triggered_factors", [])
        self.assertIn("regime_mismatch", triggered, "Should detect regime mismatch")
        self.assertGreater(verdict.metadata.get("regime_mismatch", 0.0), 0.0)

    def test_verdict_to_dict(self) -> None:
        """Test: verdict.to_dict() works correctly."""
        ctx = self._make_context()
        verdict = self.agent.evaluate(ctx)

        verdict_dict = verdict.to_dict()

        self.assertEqual(verdict_dict["name"], "devil_advocate")
        self.assertIn("score", verdict_dict)
        self.assertIn("passed", verdict_dict)
        self.assertIn("reason", verdict_dict)
        self.assertIn("reason_code", verdict_dict)
        self.assertIn("block_trade", verdict_dict)
        self.assertIn("metadata", verdict_dict)
        self.assertIsInstance(verdict_dict["metadata"], dict)

    def test_rr_barely_passes_factor(self) -> None:
        """Test: R:R < 1.35 (barely above 1.2 min) → rr_barely_passes fires."""
        ctx = self._make_context(
            sl_pips=15.0,
            tp_pips=20.0,  # RR = 1.33, barely passes
        )

        verdict = self.agent.evaluate(ctx)

        triggered = verdict.metadata.get("triggered_factors", [])
        self.assertIn("rr_barely_passes", triggered, "Should flag marginal R:R")

    def test_spread_at_limit_factor(self) -> None:
        """Test: Spread > 1.5x median spread → spread_at_limit fires."""
        ctx = self._make_context(
            spread_pips=2.5,
        )
        ctx.analysis.spread_median_pips = 1.5  # Current spread 2.5 > 1.5 * 1.5 = 2.25

        verdict = self.agent.evaluate(ctx)

        triggered = verdict.metadata.get("triggered_factors", [])
        self.assertIn("spread_at_limit", triggered, "Should detect spread limit")

    def test_support_resistance_conflict(self) -> None:
        """Test: LONG signal within 10 pips of resistance → sr_conflict fires."""
        ctx = self._make_context(
            direction="LONG",
            current_price=1.0810,
        )
        ctx.gate_details["support_resistance_data"] = {
            "resistance": [1.0815, 1.0825],
            "support": [1.0795, 1.0785],
        }

        verdict = self.agent.evaluate(ctx)

        triggered = verdict.metadata.get("triggered_factors", [])
        self.assertIn(
            "support_resistance_conflict", triggered,
            "Should detect LONG near resistance"
        )

    def test_bear_score_calculation(self) -> None:
        """Test: bear_score is normalized correctly (0-1 range)."""
        ctx = self._make_context(
            uncertainty_score=0.50,
            model_disagreement=0.30,
        )

        verdict = self.agent.evaluate(ctx)

        bear_score = verdict.metadata.get("bear_score", 0.0)
        self.assertGreaterEqual(bear_score, 0.0, "Bear score >= 0.0")
        self.assertLessEqual(bear_score, 1.0, "Bear score <= 1.0")

    def test_confidence_delta_scales_with_bear_score(self) -> None:
        """Test: confidence_delta is negative and scales with bear_score."""
        # Low bear factors (all defaults, minimal triggers)
        ctx_low = self._make_context(
            uncertainty_score=0.20,
            model_disagreement=0.10,
        )
        verdict_low = self.agent.evaluate(ctx_low)
        delta_low = verdict_low.confidence_delta

        # High bear factors (all strong triggers)
        ctx_high = self._make_context(
            uncertainty_score=0.58,
            model_disagreement=0.39,
            volatility_regime="EXTREME",
        )
        ctx_high.gate_details["episodic_suppression"] = True
        verdict_high = self.agent.evaluate(ctx_high)
        delta_high = verdict_high.confidence_delta

        # Higher bear score should have more negative delta
        self.assertGreater(delta_low, delta_high, "More bear = more negative delta")
        self.assertLess(delta_high, delta_low, "High bear delta more negative")

    def test_agent_score_inverted_from_bear_score(self) -> None:
        """Test: agent_score = 1.0 - bear_score."""
        ctx = self._make_context(
            uncertainty_score=0.40,
        )

        verdict = self.agent.evaluate(ctx)

        bear_score = verdict.metadata.get("bear_score", 0.0)
        expected_agent_score = 1.0 - bear_score
        self.assertAlmostEqual(
            verdict.score, expected_agent_score, places=2,
            msg="Agent score should be inverted from bear score"
        )


if __name__ == "__main__":
    unittest.main()
