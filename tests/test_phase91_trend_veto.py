"""Tests for US-502: Trend agent hard veto for directional trades."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.scanner.agents._team import AgentDecisionContext, AgentVerdict, ScannerAgentTeam
from src.scanner.config import ScannerConfig
from src.scanner.results import PairAnalysis


def _make_team(veto_enabled: bool = True) -> ScannerAgentTeam:
    config = Mock()
    config.enable_trend_agent_hard_veto = veto_enabled
    config.weighted_vote_threshold = 0.55
    config.regime_vote_thresholds = {}
    config.weight_boost_on_win = 0.10
    config.weight_penalty_on_loss = 0.15
    config.min_agent_weight = 0.1
    config.max_agent_weight = 2.0
    return ScannerAgentTeam(config)


def _make_ctx(direction: str, adx: float = 1.0, veto_enabled: bool = True) -> AgentDecisionContext:
    """Build a minimal AgentDecisionContext where trend signal is absent (adx=1 → no trend)."""
    analysis = Mock(spec=PairAnalysis)
    analysis.direction = direction
    analysis.current_price = 1.1000
    analysis.ridge_confidence = adx  # adx falls back to ridge_confidence

    # Close == sma_20 == sma_50 → no trend signal → trend_signal=0.0 → score≈0.48
    # With adx=1: adx_support = clip((1-18)/22) = 0.0 → score=0.48 < 0.55 → passed=False
    close_price = 1.1000
    df_raw = pd.DataFrame({"close": [close_price]})
    df_feat = pd.DataFrame({
        "sma_20": [close_price],
        "sma_50": [close_price],
        "adx": [adx],
    })

    config = Mock()
    config.enable_trend_agent_hard_veto = veto_enabled

    return AgentDecisionContext(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details={},
        config=config,
    )


def test_trend_veto_blocks_long_direction():
    """AC-1a: ADX=1 + LONG direction + veto enabled → passed=False, block_trade=True."""
    team = _make_team(veto_enabled=True)
    ctx = _make_ctx(direction="LONG", adx=1.0, veto_enabled=True)
    verdict = team._evaluate_trend(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is True
    assert "(hard veto)" in verdict.reason


def test_trend_veto_blocks_short_direction():
    """AC-1b: ADX=1 + SHORT direction + veto enabled → passed=False, block_trade=True."""
    team = _make_team(veto_enabled=True)
    ctx = _make_ctx(direction="SHORT", adx=1.0, veto_enabled=True)
    verdict = team._evaluate_trend(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is True
    assert "(hard veto)" in verdict.reason


def test_trend_veto_disabled_returns_block_false():
    """AC-2: Veto disabled → block_trade=False even when trend fails on directional trade."""
    team = _make_team(veto_enabled=False)
    ctx = _make_ctx(direction="LONG", adx=1.0, veto_enabled=False)
    verdict = team._evaluate_trend(ctx)

    assert verdict.passed is False
    assert verdict.block_trade is False


def test_trend_veto_hold_direction_not_blocked():
    """AC-3: HOLD direction → veto does not apply → block_trade=False."""
    team = _make_team(veto_enabled=True)
    ctx = _make_ctx(direction="HOLD", adx=1.0, veto_enabled=True)
    verdict = team._evaluate_trend(ctx)

    assert verdict.block_trade is False


def test_config_has_enable_trend_agent_hard_veto_field():
    """ScannerConfig dataclass has enable_trend_agent_hard_veto defaulting to True."""
    config = ScannerConfig()
    assert hasattr(config, "enable_trend_agent_hard_veto")
    assert config.enable_trend_agent_hard_veto is True


def test_smart_profile_enables_trend_veto():
    """Smart profile dict includes enable_trend_agent_hard_veto=True."""
    from src.scanner.config import SCAN_PROFILES
    assert SCAN_PROFILES["smart"].get("enable_trend_agent_hard_veto") is True
