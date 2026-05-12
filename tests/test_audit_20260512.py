"""Tests for the 2026-05-12 audit changes (Buckets A + B quick wins).

Covers:
- Bucket A: trader_readiness agent default flipped to False (dataclass + all four profile dicts).
- Bucket B2: execution_quality hard-veto at spread >= 2x max_spread (reason_code execution_spread_extreme).
- Bucket B3: volatility hard-veto reason_code split (volatility_dead vs volatility_block).
- Bucket B4: graph attention enabled by default + smart profile carries explicit flag.

(Bucket B1 mean_reversion floor 0.25 -> 0.15 is covered in test_phase93_mean_reversion_veto.py.)
"""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.scanner.agents._team import AgentDecisionContext, ScannerAgentTeam
from src.scanner.config import SCAN_PROFILES, ScannerConfig
from src.scanner.results import PairAnalysis


# ---------------------------------------------------------------------------
# Bucket A — trader_readiness default off
# ---------------------------------------------------------------------------

def test_audit_trader_readiness_dataclass_default_false():
    """Dataclass default must be False: Aura writer is not implemented yet."""
    assert ScannerConfig().enable_trader_readiness_agent is False


@pytest.mark.parametrize("profile_name", ["balanced", "conservative", "aggressive", "smart"])
def test_audit_trader_readiness_profile_dicts_all_false(profile_name: str):
    """All four profile dicts must carry False to survive apply_profile()."""
    profile = SCAN_PROFILES[profile_name]
    # Profile dicts that omit the key are fine (dataclass default = False covers them);
    # but a profile dict that sets True would silently re-arm the dormant agent.
    assert profile.get("enable_trader_readiness_agent", False) is False, (
        f"Profile '{profile_name}' has enable_trader_readiness_agent=True; "
        f"this re-arms a dormant agent (no Aura writer). Set to False until writer ships."
    )


# ---------------------------------------------------------------------------
# Bucket B2 — execution_quality hard-veto on extreme spread
# ---------------------------------------------------------------------------

def _make_exec_ctx(spread_pips: float, max_spread_pips: float = 3.0) -> AgentDecisionContext:
    """Build context for _evaluate_execution_quality. Slippage held normal."""
    analysis = Mock(spec=PairAnalysis)
    analysis.atr_pips = 12.0
    analysis.spread_pips = spread_pips
    analysis.est_slippage_pips = 0.5  # well below cap

    df_raw = pd.DataFrame({"close": [1.1000]})
    df_feat = pd.DataFrame({"volume_ratio": [1.0]})  # neutral liquidity

    config = Mock()
    config.max_spread_pips = max_spread_pips
    config.max_slippage_pips = 2.0
    config.min_liquidity_score = 0.3

    return AgentDecisionContext(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details={},
        config=config,
    )


def _make_team() -> ScannerAgentTeam:
    config = Mock()
    config.weighted_vote_threshold = 0.55
    config.regime_vote_thresholds = {}
    config.weight_boost_on_win = 0.10
    config.weight_penalty_on_loss = 0.15
    config.min_agent_weight = 0.1
    config.max_agent_weight = 2.0
    return ScannerAgentTeam(config)


def test_audit_execution_quality_extreme_spread_hard_veto():
    """spread_pips=6 with max_spread=3 → 2x → hard-block with execution_spread_extreme code."""
    team = _make_team()
    ctx = _make_exec_ctx(spread_pips=6.0, max_spread_pips=3.0)
    verdict = team._evaluate_execution_quality(ctx)

    assert verdict.block_trade is True
    assert verdict.reason_code == "execution_spread_extreme"
    assert "EXTREME" in verdict.reason


def test_audit_execution_quality_above_extreme_threshold_hard_veto():
    """spread_pips=7 (2.33x) → still execution_spread_extreme, not the old execution_block."""
    team = _make_team()
    ctx = _make_exec_ctx(spread_pips=7.0, max_spread_pips=3.0)
    verdict = team._evaluate_execution_quality(ctx)

    assert verdict.block_trade is True
    assert verdict.reason_code == "execution_spread_extreme"


def test_audit_execution_quality_soft_block_unchanged():
    """spread_pips=4.5 (1.5x) → existing soft block (1.25x rail), code stays execution_block."""
    team = _make_team()
    ctx = _make_exec_ctx(spread_pips=4.5, max_spread_pips=3.0)
    verdict = team._evaluate_execution_quality(ctx)

    assert verdict.block_trade is True
    assert verdict.reason_code == "execution_block"  # NOT the new extreme code


def test_audit_execution_quality_passes_on_normal_spread():
    """spread_pips=2 (0.66x) → passes, no block."""
    team = _make_team()
    ctx = _make_exec_ctx(spread_pips=2.0, max_spread_pips=3.0)
    verdict = team._evaluate_execution_quality(ctx)

    assert verdict.block_trade is False


# ---------------------------------------------------------------------------
# Bucket B3 — volatility hard-veto reason_code split
# ---------------------------------------------------------------------------

def _make_vol_ctx(atr_pips: float, min_atr_pips: float = 5.0) -> AgentDecisionContext:
    """Build context for _evaluate_volatility."""
    analysis = Mock(spec=PairAnalysis)
    analysis.atr_pips = atr_pips
    analysis.volatility_percentile = 0.5
    analysis.volatility_regime = "NORMAL"
    analysis.volatility_gate_passed = True  # gate already passed; block driven by atr math

    df_raw = pd.DataFrame({"close": [1.1000]})
    df_feat = pd.DataFrame({"volume_ratio": [1.0]})

    config = Mock()
    config.min_atr_pips = min_atr_pips

    return AgentDecisionContext(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details={},
        config=config,
    )


def test_audit_volatility_dead_reason_code_at_50pct():
    """atr_pips=2.5 with min_atr=5.0 → exactly 0.5x → volatility_dead (new code)."""
    team = _make_team()
    ctx = _make_vol_ctx(atr_pips=2.5, min_atr_pips=5.0)
    verdict = team._evaluate_volatility(ctx)

    assert verdict.block_trade is True
    assert verdict.reason_code == "volatility_dead"
    assert "DEAD" in verdict.reason


def test_audit_volatility_dead_reason_code_below_50pct():
    """atr_pips=2.0 (0.4x) → volatility_dead."""
    team = _make_team()
    ctx = _make_vol_ctx(atr_pips=2.0, min_atr_pips=5.0)
    verdict = team._evaluate_volatility(ctx)

    assert verdict.block_trade is True
    assert verdict.reason_code == "volatility_dead"


def test_audit_volatility_gate_failure_uses_legacy_code():
    """volatility_gate_passed=False with atr above 0.5x → legacy volatility_block code."""
    team = _make_team()
    ctx = _make_vol_ctx(atr_pips=4.0, min_atr_pips=5.0)  # 0.8x, above the 0.5x dead line
    ctx.analysis.volatility_gate_passed = False

    verdict = team._evaluate_volatility(ctx)

    assert verdict.block_trade is True
    assert verdict.reason_code == "volatility_block"  # gate-driven block, not dead-atr


def test_audit_volatility_passes_above_dead_threshold():
    """atr_pips=4.0 (0.8x) with gate passing → no block."""
    team = _make_team()
    ctx = _make_vol_ctx(atr_pips=4.0, min_atr_pips=5.0)
    verdict = team._evaluate_volatility(ctx)

    assert verdict.block_trade is False


# ---------------------------------------------------------------------------
# Bucket B4 — graph attention on by default + smart profile explicit
# ---------------------------------------------------------------------------

def test_audit_graph_attention_dataclass_default_true():
    """enable_graph_attention is True by default; reweighting runs unless explicitly disabled."""
    assert ScannerConfig().enable_graph_attention is True


def test_audit_graph_attention_smart_profile_explicit():
    """Smart profile carries explicit True (was previously implicit/inherited)."""
    assert SCAN_PROFILES["smart"].get("enable_graph_attention") is True
