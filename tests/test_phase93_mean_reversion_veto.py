"""Tests for H2 / Phase 93: mean_reversion composite veto.

Rule: when mean_reversion fails (score < 0.52) AND model_disagreement > floor on a
directional trade → AgentVerdict.block_trade = True.

Evidence from 04-15 catastrophic 14-loss streak: mean_reversion voted NO on every
losing trade but its weight (0.90, lowest of core agents) let WVS arithmetic drown
the signal. The composite rule gives MR hard-block authority only when paired with
model disagreement evidence — the "nobody knows what's happening" state.

Critical design choice: reads model_disagreement from ctx.gate_details, NOT
ctx.analysis.model_disagreement. The analysis attribute is populated AFTER
verdicts are collected (see _team.py line ~1748); gate_details carries the
upstream value.
"""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.scanner.agents._team import AgentDecisionContext, ScannerAgentTeam
from src.scanner.config import ScannerConfig
from src.scanner.results import PairAnalysis


def _make_team() -> ScannerAgentTeam:
    config = Mock()
    config.enable_mean_reversion_veto = True
    config.weighted_vote_threshold = 0.55
    config.regime_vote_thresholds = {}
    config.weight_boost_on_win = 0.10
    config.weight_penalty_on_loss = 0.15
    config.min_agent_weight = 0.1
    config.max_agent_weight = 2.0
    return ScannerAgentTeam(config)


def _make_ctx(
    direction: str,
    rsi: float = 50.0,
    model_disagreement: float = 0.0,
    veto_enabled: bool = True,
    disagree_floor: float = 0.25,
) -> AgentDecisionContext:
    """Build MR-agent context. Defaults: neutral RSI → score≈0.50 → passed=False."""
    analysis = Mock(spec=PairAnalysis)
    analysis.direction = direction
    analysis.current_price = 1.1000

    df_raw = pd.DataFrame({"close": [1.1000]})
    df_feat = pd.DataFrame({"rsi": [rsi]})

    config = Mock()
    config.enable_mean_reversion_veto = veto_enabled
    config.mean_reversion_veto_disagree_floor = disagree_floor

    return AgentDecisionContext(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details={"model_disagreement": model_disagreement},
        config=config,
    )


# ---------------------------------------------------------------------------
# Composite veto fires (MR fails + disagreement > floor)
# ---------------------------------------------------------------------------

def test_veto_fires_on_neutral_rsi_plus_high_disagreement_long():
    """04-15 pattern: neutral RSI (MR passed=False) + disagreement=0.30 > 0.25 → block."""
    team = _make_team()
    ctx = _make_ctx(direction="LONG", rsi=50.0, model_disagreement=0.30)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.passed is False  # neutral RSI score=0.50 < 0.52
    assert verdict.block_trade is True
    assert "hard veto" in verdict.reason


def test_veto_fires_on_neutral_rsi_plus_high_disagreement_short():
    """Same catastrophic pattern, SHORT direction."""
    team = _make_team()
    ctx = _make_ctx(direction="SHORT", rsi=50.0, model_disagreement=0.30)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is True


def test_veto_fires_on_stretched_rsi_plus_high_disagreement():
    """Also covers mean_reversion_contra case (RSI stretched against direction)."""
    team = _make_team()
    # LONG with RSI=70 → oppose=(70-58)/20=0.6 → score=0.50+0-0.18=0.32 → fail
    ctx = _make_ctx(direction="LONG", rsi=70.0, model_disagreement=0.30)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is True
    assert verdict.reason_code == "mean_reversion_contra"


# ---------------------------------------------------------------------------
# Composite veto does NOT fire (safety conditions)
# ---------------------------------------------------------------------------

def test_veto_not_fires_when_disagreement_below_floor():
    """MR fails but disagreement=0.20 <= 0.25 floor → no block (soft case).

    Note: the explicit 0.25 floor is passed in; the new default floor is 0.15
    (2026-05-12 audit). See test_audit_lower_floor_fires_in_018_023_band below.
    """
    team = _make_team()
    ctx = _make_ctx(direction="LONG", rsi=50.0, model_disagreement=0.20)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is False


def test_audit_lower_floor_fires_in_018_023_band():
    """2026-05-12 audit: with new 0.15 floor, MR-fails + disagreement=0.18 fires.

    The previous 0.25 floor missed this band; three audit losses sat at 0.18-0.23.
    Asserts the new default floor catches them.
    """
    team = _make_team()
    ctx = _make_ctx(direction="LONG", rsi=50.0, model_disagreement=0.18, disagree_floor=0.15)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is True


def test_audit_defaults_floor_at_015():
    """Config defaults: dataclass default + smart profile entry both at 0.15."""
    from src.scanner.config import SCAN_PROFILES, ScannerConfig

    assert ScannerConfig().mean_reversion_veto_disagree_floor == 0.15
    assert SCAN_PROFILES["smart"]["mean_reversion_veto_disagree_floor"] == 0.15


def test_veto_not_fires_when_mr_passes():
    """Supportive RSI → MR passes → block never evaluated."""
    team = _make_team()
    # LONG with RSI=35 → support=(50-35)/20=0.75 → score=0.50+0.1875=0.69 → passed=True
    ctx = _make_ctx(direction="LONG", rsi=35.0, model_disagreement=0.50)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.passed is True
    assert verdict.block_trade is False


def test_veto_not_fires_on_hold_direction():
    """HOLD direction → veto skipped regardless of disagreement."""
    team = _make_team()
    ctx = _make_ctx(direction="HOLD", rsi=50.0, model_disagreement=0.50)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.block_trade is False


def test_veto_disabled_by_config_flag():
    """enable_mean_reversion_veto=False → no block even on catastrophic pattern."""
    team = _make_team()
    ctx = _make_ctx(direction="LONG", rsi=50.0, model_disagreement=0.50, veto_enabled=False)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is False


def test_veto_respects_custom_floor():
    """Custom floor of 0.35 → disagreement=0.30 is below → no block."""
    team = _make_team()
    ctx = _make_ctx(direction="LONG", rsi=50.0, model_disagreement=0.30, disagree_floor=0.35)
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is False


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_missing_gate_details_falls_back_safely():
    """If gate_details is empty, model_disagreement defaults to 0.0 → no block."""
    team = _make_team()
    analysis = Mock(spec=PairAnalysis)
    analysis.direction = "LONG"
    analysis.current_price = 1.1000

    config = Mock()
    config.enable_mean_reversion_veto = True
    config.mean_reversion_veto_disagree_floor = 0.25

    ctx = AgentDecisionContext(
        analysis=analysis,
        df_raw=pd.DataFrame({"close": [1.1000]}),
        df_feat=pd.DataFrame({"rsi": [50.0]}),
        gate_details={},  # empty
        config=config,
    )
    verdict = team._evaluate_mean_reversion(ctx)

    assert verdict.block_trade is False


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

def test_config_has_mean_reversion_veto_fields():
    """ScannerConfig exposes both fields with correct defaults.

    Default floor lowered 0.25 -> 0.15 by the 2026-05-12 audit (3 losses sat in 0.18-0.23 band).
    """
    config = ScannerConfig()
    assert hasattr(config, "enable_mean_reversion_veto")
    assert config.enable_mean_reversion_veto is True
    assert hasattr(config, "mean_reversion_veto_disagree_floor")
    assert config.mean_reversion_veto_disagree_floor == pytest.approx(0.15)


def test_smart_profile_enables_mean_reversion_veto():
    """Smart profile enables the veto with floor=0.15 (audit-lowered 2026-05-12)."""
    from src.scanner.config import SCAN_PROFILES
    profile = SCAN_PROFILES["smart"]
    assert profile.get("enable_mean_reversion_veto") is True
    assert profile.get("mean_reversion_veto_disagree_floor") == pytest.approx(0.15)
