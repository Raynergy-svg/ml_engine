"""Tests for US-503: Staleness-aware uncertainty threshold."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.scanner.agents._team import AgentDecisionContext, ScannerAgentTeam
from src.scanner.config import ScannerConfig
from src.scanner.results import PairAnalysis


def _make_team() -> ScannerAgentTeam:
    config = Mock()
    config.weighted_vote_threshold = 0.55
    config.regime_vote_thresholds = {}
    config.weight_boost_on_win = 0.10
    config.weight_penalty_on_loss = 0.15
    config.min_agent_weight = 0.1
    config.max_agent_weight = 2.0
    return ScannerAgentTeam(config)


def _make_ctx(
    uncertainty_score_target: float = 0.38,
    max_uncertainty_score: float = 0.40,
    staleness_uncertainty_threshold: float = 0.35,
    staleness_age_threshold_days: float = 7.0,
    soft_uncertainty_blocking: bool = False,
) -> AgentDecisionContext:
    """Build a context where uncertainty_score ≈ uncertainty_score_target.

    uncertainty_score = clip(confidence_uncertainty * 0.60 + model_disagreement * 0.40)
    confidence_uncertainty = clip((0.60 - confidence) / 0.10)

    To get uncertainty_score ≈ 0.38:
      confidence = 0.58 → confidence_uncertainty = clip(0.02/0.10) = 0.20
      model_disagreement = 0.50 → disagreement_part = 0.20
      → uncertainty_score = 0.12 + 0.20 = 0.32  (too low)

    Let's drive it higher:
      confidence = 0.55 → confidence_uncertainty = clip(0.05/0.10) = 0.50
      model_disagreement = 0.15 → disagreement_part = 0.06
      → uncertainty_score = 0.30 + 0.06 = 0.36

    Actually let's use a more direct approach:
      confidence = 0.535 → confidence_uncertainty = clip(0.065/0.10) = 0.65
      heuristics: 2 oppose out of 5 → model_disagreement = 0.40
      → uncertainty_score = 0.65*0.60 + 0.40*0.40 = 0.39 + 0.16 = 0.55 (too high)

    Easiest: confidence=0.56 → confidence_uncertainty=clip(0.04/0.10)=0.40
    heuristics: all agree (model_disagreement=0) → uncertainty_score=0.40*0.60=0.24 (too low)

    Target 0.38:
      Let x = confidence_uncertainty, y = model_disagreement
      0.38 = 0.60x + 0.40y
      Use x=0.50 (confidence=0.55), y=0.20 (1 of 5 oppose):
      0.30 + 0.08 = 0.38 ✓
    """
    analysis = Mock(spec=PairAnalysis)
    analysis.direction = "LONG"
    analysis.current_price = 1.1000
    analysis.tcn_score = 0.0
    analysis.ridge_score = 0.0
    analysis.rf_score = 0.0
    # confidence=0.55 → confidence_uncertainty = clip((0.60-0.55)/0.10) = 0.50
    analysis.confidence = 0.55
    analysis.ridge_confidence = 0.55

    close_price = 1.1000
    # Build df_feat: 1 heuristic opposes out of 5 total → model_disagreement=0.20
    # close > sma_20 → heuristic +1 (agrees)
    # rsi < 50 (abs(rsi-50)>=3) → heuristic +1 (agrees)
    # ret_5 > 0 → heuristic +1 (agrees)
    # close < sma_50 → heuristic -1 (opposes)
    # macd_hist > 0 → heuristic +1 (agrees)
    # 1 oppose / 5 total = 0.20
    df_raw = pd.DataFrame({"close": [close_price]})
    df_feat = pd.DataFrame({
        "close": [close_price],
        "rsi": [46.0],        # < 50, abs diff = 4 ≥ 3 → +1
        "ret_5": [0.001],     # > 0 → +1
        "sma_20": [close_price - 0.001],  # close > sma_20 → +1
        "sma_50": [close_price + 0.001],  # close < sma_50 → -1 (opposes)
        "macd_hist": [1e-5],  # > 0 → +1
    })

    config = Mock()
    config.max_uncertainty_score = max_uncertainty_score
    config.max_model_disagreement = 0.65
    config.disagreement_hard_floor = 0.50
    config.soft_uncertainty_blocking = soft_uncertainty_blocking
    config.ensemble_conflict_enabled = False
    config.staleness_uncertainty_threshold = staleness_uncertainty_threshold
    config.staleness_age_threshold_days = staleness_age_threshold_days

    return AgentDecisionContext(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details={},
        config=config,
    )


_FRESHNESS_STALE = {
    "oldest_age_days": 10.0,
    "stale_models": ["modular_ensemble (10d)"],
    "status": "STALE",
}

_FRESHNESS_FRESH = {
    "oldest_age_days": 5.0,
    "stale_models": [],
    "status": "FRESH",
}


class TestStalenessUncertaintyBlock:
    """AC verification for US-503 staleness-aware uncertainty threshold."""

    def test_stale_models_block_uncertainty_above_035(self):
        """AC-1: oldest_age_days=10, uncertainty≈0.38 (>0.35 but <0.40) → blocks."""
        team = _make_team()
        ctx = _make_ctx(
            max_uncertainty_score=0.40,
            staleness_uncertainty_threshold=0.35,
            staleness_age_threshold_days=7.0,
            soft_uncertainty_blocking=False,
        )
        target = "src.scanner.agents._team.get_model_freshness"
        with patch(target, return_value=_FRESHNESS_STALE):
            verdict = team._evaluate_uncertainty(ctx)

        # uncertainty_score ≈ 0.38 > effective threshold 0.35 → must block
        assert verdict.block_trade is True, (
            f"Expected block_trade=True when stale (oldest=10d), got {verdict}"
        )

    def test_fresh_models_do_not_tighten_threshold(self):
        """AC-2: oldest_age_days=5 (≤7) → uses 0.40 threshold → uncertainty≈0.38 does NOT block."""
        team = _make_team()
        ctx = _make_ctx(
            max_uncertainty_score=0.40,
            staleness_uncertainty_threshold=0.35,
            staleness_age_threshold_days=7.0,
            soft_uncertainty_blocking=False,
        )
        target = "src.scanner.agents._team.get_model_freshness"
        with patch(target, return_value=_FRESHNESS_FRESH):
            verdict = team._evaluate_uncertainty(ctx)

        # uncertainty_score ≈ 0.38 ≤ 0.40 → should NOT block
        assert verdict.block_trade is False, (
            f"Expected block_trade=False when fresh (oldest=5d), got {verdict}"
        )

    def test_freshness_exception_falls_back_to_max_uncertainty(self):
        """AC-3: get_model_freshness raises → falls back to max_uncertainty_score=0.40, no crash."""
        team = _make_team()
        ctx = _make_ctx(
            max_uncertainty_score=0.40,
            staleness_uncertainty_threshold=0.35,
            staleness_age_threshold_days=7.0,
            soft_uncertainty_blocking=False,
        )
        target = "src.scanner.agents._team.get_model_freshness"
        with patch(target, side_effect=RuntimeError("disk error")):
            # Must not raise
            verdict = team._evaluate_uncertainty(ctx)

        # With fallback to 0.40 threshold and uncertainty≈0.38 → no block
        assert verdict.block_trade is False, (
            f"Expected fallback to 0.40 threshold (no block), got {verdict}"
        )


class TestScannerConfigStalenessFields:
    """Verify new fields exist on ScannerConfig with correct defaults."""

    def test_staleness_fields_exist_with_defaults(self):
        cfg = ScannerConfig()
        assert hasattr(cfg, "staleness_uncertainty_threshold")
        assert hasattr(cfg, "staleness_age_threshold_days")
        assert cfg.staleness_uncertainty_threshold == 0.35
        assert cfg.staleness_age_threshold_days == 7.0

    def test_smart_profile_includes_staleness_fields(self):
        from src.scanner.config import SCAN_PROFILES
        smart = SCAN_PROFILES["smart"]
        assert smart.get("staleness_uncertainty_threshold") == 0.35
        assert smart.get("staleness_age_threshold_days") == 7.0
