"""Tests for tiered-autonomy levels in self-heal (Agent 2, 2026-05-10).

Verifies that:
  * Level constants are integers in strict order (3 < 4 < 5).
  * Every dispatch verb has a level-map entry (no silent gaps).
  * ScannerConfig exposes self_heal_max_autonomy_level with default 3.
  * Level-3 (reversible) actions are allowed at the default ceiling.
  * Level-4 (guarded) actions are blocked at default, allowed at max=4.
  * Level-5 (manual) actions are blocked at default and at max=4.
  * Unknown action types fail-closed to level 5 (manual).

NO MOCKS per CLAUDE.md No-Mock Rule. Real SelfHeal, real ScannerConfig,
real disk via tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.scanner.config import ScannerConfig
from src.scanner.feedback.self_heal import SelfHeal


# --------------------------------------------------------------------- #
# Level constants + map structure                                       #
# --------------------------------------------------------------------- #
def test_default_max_level_is_3():
    """ScannerConfig.self_heal_max_autonomy_level defaults to 3."""
    config = ScannerConfig()
    assert hasattr(config, "self_heal_max_autonomy_level")
    assert config.self_heal_max_autonomy_level == 3


def test_level_constants_correct():
    """Levels are integers in strict order with the expected names."""
    assert SelfHeal.LEVEL_3_REVERSIBLE == 3
    assert SelfHeal.LEVEL_4_GUARDED == 4
    assert SelfHeal.LEVEL_5_MANUAL == 5
    # Strict ordering — used by the > comparison in _check_action_level.
    assert (
        SelfHeal.LEVEL_3_REVERSIBLE
        < SelfHeal.LEVEL_4_GUARDED
        < SelfHeal.LEVEL_5_MANUAL
    )


def test_handler_level_map_complete():
    """Every action verb in _dispatch must have a _HANDLER_LEVELS entry.

    Drift between the two — a new dispatch verb without a level entry —
    silently defaults the new action to LEVEL_5 (manual). That fail-closed
    default is intentional for unknowns at runtime, but a verb shipped
    via _dispatch SHOULD have an explicit level so the operator knows
    the intended autonomy tier without reading the source.
    """
    sh = SelfHeal()
    dispatch_verbs = set(sh._dispatch.keys())
    level_keys = set(SelfHeal._HANDLER_LEVELS.keys())
    missing = dispatch_verbs - level_keys
    assert not missing, (
        "verbs in _dispatch missing from _HANDLER_LEVELS: "
        + ", ".join(sorted(missing))
    )


# --------------------------------------------------------------------- #
# _check_action_level — gate semantics                                  #
# --------------------------------------------------------------------- #
def test_level_3_action_allowed_at_default():
    """Default max=3 allows reversible (level 3) actions."""
    config = ScannerConfig()
    assert config.self_heal_max_autonomy_level == 3
    sh = SelfHeal(config=config)

    allowed, reason, level = sh._check_action_level("soft_reset_agent_weight")
    assert allowed is True
    assert reason == ""
    assert level == 3


def test_level_3_reset_gate_allowed_at_default():
    """reset_gate_threshold_to_default is also level 3 — auto-apply OK."""
    config = ScannerConfig()
    sh = SelfHeal(config=config)
    allowed, _, level = sh._check_action_level("reset_gate_threshold_to_default")
    assert allowed is True
    assert level == 3


def test_level_4_action_blocked_at_default():
    """Default max=3 blocks guarded (level 4) actions like tighten."""
    config = ScannerConfig()
    sh = SelfHeal(config=config)

    allowed, reason, level = sh._check_action_level("tighten_gate_threshold")
    assert allowed is False
    assert level == 4
    assert "level_4_above_max_3" in reason


def test_level_4_reduce_risk_blocked_at_default():
    """reduce_risk_per_trade_pct is level 4 — blocked at default."""
    config = ScannerConfig()
    sh = SelfHeal(config=config)
    allowed, _, level = sh._check_action_level("reduce_risk_per_trade_pct")
    assert allowed is False
    assert level == 4


def test_level_4_action_allowed_at_4():
    """Elevating max to 4 allows guarded actions."""
    config = ScannerConfig()
    config.self_heal_max_autonomy_level = 4
    sh = SelfHeal(config=config)

    allowed, reason, level = sh._check_action_level("tighten_gate_threshold")
    assert allowed is True
    assert reason == ""
    assert level == 4


def test_level_5_action_blocked_at_4():
    """Even with max=4, level-5 (manual) actions are still blocked."""
    config = ScannerConfig()
    config.self_heal_max_autonomy_level = 4
    sh = SelfHeal(config=config)

    allowed, reason, level = sh._check_action_level("retrain_gates")
    assert allowed is False
    assert level == 5
    assert "level_5_above_max_4" in reason


def test_level_5_action_allowed_at_5():
    """Operator can elevate to 5 to allow retrain actions (NOT recommended)."""
    config = ScannerConfig()
    config.self_heal_max_autonomy_level = 5
    sh = SelfHeal(config=config)
    allowed, _, level = sh._check_action_level("retrain_rl_position_sizer")
    assert allowed is True
    assert level == 5


def test_unknown_action_defaults_to_manual():
    """Unknown action types fail-closed to level 5 (manual).

    Critical safety property: a typo'd action verb or a new handler
    shipped without a _HANDLER_LEVELS entry must NEVER auto-apply.
    """
    config = ScannerConfig()
    sh = SelfHeal(config=config)
    allowed, reason, level = sh._check_action_level("not_a_real_action_xyzzy")
    assert allowed is False
    assert level == 5
    assert "level_5_above_max_3" in reason


def test_unknown_action_blocked_even_at_4():
    """Unknown actions remain blocked even when max=4 (still level 5)."""
    config = ScannerConfig()
    config.self_heal_max_autonomy_level = 4
    sh = SelfHeal(config=config)
    allowed, _, level = sh._check_action_level("hypothetical_new_action")
    assert allowed is False
    assert level == 5


def test_check_works_without_config():
    """Without a config handle, _check_action_level falls back to max=3."""
    sh = SelfHeal()  # no config
    assert sh.config is None
    # level 3 still allowed
    allowed, _, _ = sh._check_action_level("soft_reset_agent_weight")
    assert allowed is True
    # level 4 still blocked
    allowed, _, _ = sh._check_action_level("tighten_gate_threshold")
    assert allowed is False


# --------------------------------------------------------------------- #
# End-to-end: apply() honors the ceiling                                #
# --------------------------------------------------------------------- #
def _make_self_heal_for_apply(tmp_path: Path, config: ScannerConfig) -> SelfHeal:
    """Build a SelfHeal with disk paths redirected to tmp_path.

    Real disk per the No-Mock Rule. We redirect the constants to keep
    the test process from touching shared project paths.
    """
    sh = SelfHeal(config=config)
    sh.JOURNAL_PATH = tmp_path / "trade_journal.json"
    sh.CONFIG_ADJUSTMENTS_PATH = tmp_path / "config_adjustments.json"
    sh.DEBOUNCE_STATE_PATH = tmp_path / "debounce.json"
    sh.DECISION_LOG_PATH = tmp_path / "decisions.jsonl"
    sh.ERROR_LOG_PATH = tmp_path / "unhandled_errors.jsonl"
    sh.DEGRADED_FLAG_PATH = tmp_path / "degraded.flag"
    sh.ACTION_BUDGET_PATH = tmp_path / "action_budget.json"
    sh.AGENT_WEIGHTS_PATH = tmp_path / "agent_weights.json"
    return sh


def test_apply_skips_level_4_at_default(tmp_path):
    """End-to-end: apply() with default config skips level-4 tighten action.

    Verifies the guard is wired into the dispatch flow, not just the
    helper. The action should appear in actions_taken with
    success=False and a level_gate_skipped detail.
    """
    config = ScannerConfig()
    sh = _make_self_heal_for_apply(tmp_path, config)

    diag_result = {
        "status": "DEGRADED",
        "issues": [
            {
                "check": "gate_firing_rate",
                "severity": "warning",
                "detail": "test issue",
            }
        ],
        "recommended_actions": [
            "tighten_gate_threshold:min_confidence",
        ],
    }
    result = sh.apply(diag_result)

    assert result["status"] in ("degraded", "no_action", "applied")
    actions = result["actions_taken"]
    assert len(actions) == 1
    assert actions[0]["action"] == "tighten_gate_threshold:min_confidence"
    assert actions[0]["success"] is False
    assert "level_gate_skipped" in actions[0]["detail"]
    assert "level=4" in actions[0]["detail"]


def test_apply_allows_level_3_at_default(tmp_path):
    """End-to-end: level-3 soft_reset_agent_weight is NOT level-gate-skipped.

    The action may still fail downstream (no agent_weights.json on disk
    in tmp_path), but the level gate must not be the rejection reason.
    """
    config = ScannerConfig()
    sh = _make_self_heal_for_apply(tmp_path, config)

    diag_result = {
        "status": "DEGRADED",
        "issues": [
            {
                "check": "agent_weight_boundary",
                "severity": "warning",
                "detail": "test issue",
            }
        ],
        "recommended_actions": [
            "soft_reset_agent_weight:trend",
        ],
    }
    result = sh.apply(diag_result)

    actions = result["actions_taken"]
    assert len(actions) == 1
    assert actions[0]["action"] == "soft_reset_agent_weight:trend"
    # The action should NOT be level-gate-skipped. It may have other
    # detail (e.g. agent_weights_missing) but level_gate_skipped must
    # NOT appear.
    assert "level_gate_skipped" not in actions[0]["detail"]
