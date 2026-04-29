"""TDD coverage for the gate-overtightening trap recovery handler.

Bug surfaced by Mythos audit 2026-04-28 (post-restart at 7:19PM):

  PostTradeDiagnostics._check_quiet_streak fires CRITICAL with action
  "reset_gate_threshold_to_default:min_confidence" (and 6 other ScannerConfig
  field names from _QUIET_STREAK_RECOVERY_ORDER), but
  SelfHeal._handle_reset_gate_threshold:

    1. Cannot resolve ScannerConfig field names because _find_gate_default's
       paths_by_gate dict only knows {"confidence","momentum","risk",...} aliases.
       Returns None → handler returns (False, "no_default_in_yaml", []) — silent
       dead-write on 5 of 7 trap-recovery gates.

    2. Even if a default were found, _merge_adjustment writes under
       data["self_heal"][gate_threshold_X] — a namespace ConfigAdjuster
       NEVER reads. ConfigAdjuster._load_approved_history reads data["history"]
       only.

  Net effect: trap fires every scan cycle, recommends reset, SelfHeal returns
  False, no config mutation, gates stay max-tightened, no trade flow. This is
  exactly the "200 cumulative tightenings, gates maxed, 12-day silence"
  pattern documented in commit 0772581 — the trap detector shipped without
  unit-test coverage of its application chain.

  Coverage here is end-to-end:
    A. handler resolves all 7 _QUIET_STREAK_RECOVERY_ORDER ScannerConfig
       field defaults
    B. handler appends a history-shaped entry that ConfigAdjuster will apply
    C. ConfigAdjuster.apply_adjustments() picks up the entry and mutates the
       live ScannerConfig back to the default
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.config import ScannerConfig
from src.scanner.feedback.self_heal import SelfHeal
from src.scanner.automation.config_adjuster import ConfigAdjuster


_TRAP_GATES = (
    "min_confidence",
    "weighted_vote_threshold",
    "max_model_disagreement",
    "max_uncertainty_score",
    "atr_sl_multiplier",
    "min_momentum",
    "min_risk_reward_ratio",
)


@pytest.fixture
def isolated_self_heal(tmp_path, monkeypatch):
    """Point CONFIG_ADJUSTMENTS_PATH and DEBOUNCE_STATE_PATH into tmp_path
    so production state is never touched by these tests."""
    cfg_adj = tmp_path / "config_adjustments.json"
    debounce = tmp_path / "self_heal_debounce.json"
    decision_log = tmp_path / "system_decisions.jsonl"
    error_log = tmp_path / "unhandled_errors.jsonl"
    monkeypatch.setattr(SelfHeal, "CONFIG_ADJUSTMENTS_PATH", cfg_adj)
    monkeypatch.setattr(SelfHeal, "DEBOUNCE_STATE_PATH", debounce)
    monkeypatch.setattr(SelfHeal, "DECISION_LOG_PATH", decision_log)
    monkeypatch.setattr(SelfHeal, "ERROR_LOG_PATH", error_log)
    return cfg_adj


def _diag(action: str) -> dict:
    """Minimal diag dict that triggers SelfHeal.apply for one action."""
    return {
        "status": "CRITICAL",
        "issues": [{"check": "quiet_streak", "severity": "CRITICAL"}],
        "recommended_actions": [action],
    }


@pytest.mark.parametrize("gate", _TRAP_GATES)
def test_handler_resolves_default_for_every_trap_recovery_gate(
    isolated_self_heal, gate
):
    """A. handler must succeed for every gate in _QUIET_STREAK_RECOVERY_ORDER.
    Currently fails for 5/7 because paths_by_gate is missing those keys."""
    heal = SelfHeal()
    expected_default = getattr(ScannerConfig(), gate)

    result = heal.apply(_diag(f"reset_gate_threshold_to_default:{gate}"))

    actions = result.get("actions_taken") or []
    assert actions, f"No action recorded for gate={gate}"
    rec = actions[0]
    assert rec.get("success") is True, (
        f"Handler failed for gate={gate}: detail={rec.get('detail')!r}. "
        f"Expected to resolve default {expected_default}."
    )


def test_handler_writes_history_entry_consumable_by_config_adjuster(
    isolated_self_heal,
):
    """B. The handler must append a history-shaped entry so the existing
    ConfigAdjuster.apply_adjustments() pipeline picks it up.

    Schema must include: key (matching a ScannerConfig field), new_value,
    source, approved_at, timestamp, proposal_id.
    """
    heal = SelfHeal()
    gate = "min_confidence"
    expected = ScannerConfig().min_confidence  # 42.0

    result = heal.apply(_diag(f"reset_gate_threshold_to_default:{gate}"))
    assert result["status"] == "applied"

    blob = json.loads(isolated_self_heal.read_text())
    history = blob.get("history") or []
    matching = [h for h in history if h.get("key") == gate and h.get("source") == "self_heal"]
    assert matching, (
        f"No history entry written for gate={gate} source=self_heal. "
        f"history len={len(history)} keys={[h.get('key') for h in history]}"
    )
    entry = matching[-1]
    assert entry.get("new_value") == pytest.approx(expected)
    assert entry.get("approved_at"), "history entry must have approved_at"
    assert entry.get("timestamp"), "history entry must have timestamp"
    assert entry.get("proposal_id"), "history entry must have proposal_id"


def test_config_adjuster_applies_self_heal_reset_to_scanner_config(
    isolated_self_heal, tmp_path
):
    """C. End-to-end: trap recommendation → SelfHeal applies → ConfigAdjuster
    picks it up → ScannerConfig.min_confidence resets to its dataclass default.

    Pre-condition: ScannerConfig in-memory has min_confidence=99.0 (simulating
    the max-tightened production state). Post: config.min_confidence=42.0.
    """
    heal = SelfHeal()
    heal.apply(_diag("reset_gate_threshold_to_default:min_confidence"))

    pending_path = tmp_path / "pending_adjustments.json"
    adjuster = ConfigAdjuster(
        persistence_path=isolated_self_heal,
        pending_path=pending_path,
    )

    cfg = ScannerConfig()
    cfg.min_confidence = 99.0  # simulate max-tightened state
    applied = adjuster.apply_adjustments(cfg, current_cycle=10000)

    assert any(a.get("key") == "min_confidence" for a in applied), (
        f"ConfigAdjuster did not apply min_confidence reset. applied={applied}"
    )
    assert cfg.min_confidence == pytest.approx(42.0)


def test_handler_rejects_unknown_gate_without_writing(isolated_self_heal):
    """Defensive: arbitrary gate names (not in ScannerConfig) must NOT
    produce an orphan-key write — that's the original $3,527 bug pattern."""
    heal = SelfHeal()
    result = heal.apply(
        _diag("reset_gate_threshold_to_default:not_a_real_field_xyz")
    )

    actions = result.get("actions_taken") or []
    assert actions
    assert actions[0].get("success") is False
    if isolated_self_heal.exists():
        blob = json.loads(isolated_self_heal.read_text())
        history = blob.get("history") or []
        assert not any(
            h.get("key") == "not_a_real_field_xyz" for h in history
        ), "Orphan key written to history despite unknown ScannerConfig field"
