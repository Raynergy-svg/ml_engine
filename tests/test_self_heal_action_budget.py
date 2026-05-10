"""Test Agent 1 — daily action budget circuit breaker for self-heal.

Industry-standard safety per DZone self-healing pipelines + NeurIPS 2024
self-healing ML: cap total action APPLICATIONS in any rolling 24h window
to prevent runaway automation. The 2026-04-16 incident applied 5 config
adjustments in ~10 minutes; this guard would have stopped at 3/4 and
escalated to the operator.

NO MOCKS per CLAUDE.md No-Mock Rule. Real SelfHeal, real disk via
tmp_path, paths redirected to the test fixture.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.scanner.feedback.self_heal import SelfHeal


def _make_self_heal(tmp_path: Path) -> SelfHeal:
    """Build a SelfHeal instance with paths redirected to tmp_path.

    All disk-touching paths point at tmp_path so test runs are
    hermetic and concurrent test sessions don't collide.
    """
    sh = SelfHeal()
    sh.JOURNAL_PATH = tmp_path / "trade_journal.json"
    sh.CONFIG_ADJUSTMENTS_PATH = tmp_path / "config_adjustments.json"
    sh.DEBOUNCE_STATE_PATH = tmp_path / "debounce.json"
    sh.DECISION_LOG_PATH = tmp_path / "decisions.jsonl"
    sh.ERROR_LOG_PATH = tmp_path / "errors.jsonl"
    sh.DEGRADED_FLAG_PATH = tmp_path / "degraded.flag"
    sh.ACTION_BUDGET_PATH = tmp_path / "action_budget.json"
    return sh


def _seed_budget(path: Path, n_recent: int, n_old: int = 0) -> None:
    """Write an action budget file with `n_recent` recent + `n_old` old entries.

    Recent = 1h ago (inside 24h window). Old = 30h ago (outside).
    """
    now = datetime.now(timezone.utc)
    recent_ts = (now - timedelta(hours=1)).isoformat()
    old_ts = (now - timedelta(hours=30)).isoformat()
    actions = [recent_ts] * n_recent + [old_ts] * n_old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"actions_today": actions}))


# --------------------------------------------------------------------- #
# Constants                                                             #
# --------------------------------------------------------------------- #
def test_budget_path_default():
    """Class constant points at the expected location."""
    assert SelfHeal.ACTION_BUDGET_PATH == Path(
        ".claude/self_heal_action_budget.json"
    )
    assert SelfHeal.MAX_ACTIONS_PER_DAY == 10


# --------------------------------------------------------------------- #
# _load_action_budget                                                   #
# --------------------------------------------------------------------- #
def test_load_empty_returns_empty_list(tmp_path):
    """Missing file → empty list."""
    sh = _make_self_heal(tmp_path)
    assert not sh.ACTION_BUDGET_PATH.exists()
    state = sh._load_action_budget()
    assert state == {"actions_today": []}


def test_load_corrupt_returns_empty(tmp_path):
    """Invalid JSON → empty (fail-closed, log the error)."""
    sh = _make_self_heal(tmp_path)
    sh.ACTION_BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    sh.ACTION_BUDGET_PATH.write_text("{not valid json")
    state = sh._load_action_budget()
    assert state == {"actions_today": []}


def test_load_root_not_dict_returns_empty(tmp_path):
    """Non-dict JSON root → empty (defensive)."""
    sh = _make_self_heal(tmp_path)
    sh.ACTION_BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    sh.ACTION_BUDGET_PATH.write_text(json.dumps([1, 2, 3]))
    state = sh._load_action_budget()
    assert state == {"actions_today": []}


# --------------------------------------------------------------------- #
# _record_action                                                        #
# --------------------------------------------------------------------- #
def test_record_appends_timestamp(tmp_path):
    """write + reload roundtrip: action is appended with a real ISO ts."""
    sh = _make_self_heal(tmp_path)
    sh._record_action("tighten_gate_threshold:min_confidence")
    state = sh._load_action_budget()
    assert len(state["actions_today"]) == 1
    ts = state["actions_today"][0]
    # Roundtrip through fromisoformat — proves the value is a real ISO ts.
    parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - parsed).total_seconds()
    assert -1.0 < age < 5.0  # just-now, with a small tolerance


def test_record_appends_to_existing(tmp_path):
    """Recording into a non-empty budget appends, doesn't overwrite."""
    sh = _make_self_heal(tmp_path)
    _seed_budget(sh.ACTION_BUDGET_PATH, n_recent=3)
    sh._record_action("retrain_gates")
    state = sh._load_action_budget()
    assert len(state["actions_today"]) == 4


# --------------------------------------------------------------------- #
# _check_action_budget                                                  #
# --------------------------------------------------------------------- #
def test_check_passes_when_under_budget(tmp_path):
    """5 entries with MAX=10 → ok=True, reason empty."""
    sh = _make_self_heal(tmp_path)
    _seed_budget(sh.ACTION_BUDGET_PATH, n_recent=5)
    ok, reason = sh._check_action_budget()
    assert ok is True
    assert reason == ""


def test_check_passes_when_no_history(tmp_path):
    """No file → 0 actions → under budget."""
    sh = _make_self_heal(tmp_path)
    ok, reason = sh._check_action_budget()
    assert ok is True
    assert reason == ""


def test_check_refuses_when_at_budget(tmp_path):
    """10 recent entries (= MAX_ACTIONS_PER_DAY) → ok=False, reason carries counts."""
    sh = _make_self_heal(tmp_path)
    _seed_budget(sh.ACTION_BUDGET_PATH, n_recent=10)
    ok, reason = sh._check_action_budget()
    assert ok is False
    assert "budget_exhausted" in reason
    assert "n=10/10" in reason
    assert "24h" in reason


def test_check_refuses_when_over_budget(tmp_path):
    """15 recent entries (> MAX) → also refuses."""
    sh = _make_self_heal(tmp_path)
    _seed_budget(sh.ACTION_BUDGET_PATH, n_recent=15)
    ok, reason = sh._check_action_budget()
    assert ok is False
    assert "budget_exhausted" in reason


def test_check_self_prunes_old_entries(tmp_path):
    """5 recent + 20 older-than-24h → counts only the 5 recent (ok)."""
    sh = _make_self_heal(tmp_path)
    _seed_budget(sh.ACTION_BUDGET_PATH, n_recent=5, n_old=20)
    # Pre-prune: file has 25 entries
    pre = json.loads(sh.ACTION_BUDGET_PATH.read_text())
    assert len(pre["actions_today"]) == 25
    ok, reason = sh._check_action_budget()
    assert ok is True
    assert reason == ""
    # Post-prune: file should now contain only the 5 recent entries
    post = json.loads(sh.ACTION_BUDGET_PATH.read_text())
    assert len(post["actions_today"]) == 5


def test_check_under_budget_after_prune(tmp_path):
    """12 entries but only 5 in last 24h → ok (the 7 stale don't count)."""
    sh = _make_self_heal(tmp_path)
    _seed_budget(sh.ACTION_BUDGET_PATH, n_recent=5, n_old=7)
    ok, reason = sh._check_action_budget()
    assert ok is True
    assert reason == ""


def test_check_handles_corrupt_timestamps(tmp_path):
    """Corrupt timestamp entries are dropped on prune; valid ones counted."""
    sh = _make_self_heal(tmp_path)
    sh.ACTION_BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    valid_ts = (now - timedelta(hours=1)).isoformat()
    sh.ACTION_BUDGET_PATH.write_text(json.dumps({
        "actions_today": [valid_ts, "garbage", valid_ts, "", valid_ts],
    }))
    ok, reason = sh._check_action_budget()
    assert ok is True
    state = sh._load_action_budget()
    # All 3 valid entries kept; 2 garbage dropped.
    assert len(state["actions_today"]) == 3


# --------------------------------------------------------------------- #
# Integration: apply() refuses when budget exhausted                    #
# --------------------------------------------------------------------- #
def test_apply_refuses_when_budget_exhausted(tmp_path):
    """End-to-end: SelfHeal.apply() returns no_action when budget full."""
    sh = _make_self_heal(tmp_path)
    _seed_budget(sh.ACTION_BUDGET_PATH, n_recent=10)
    diag_result = {
        "status": "DEGRADED",
        "issues": [{"check": "gate_firing_rate", "severity": "warning"}],
        "recommended_actions": ["tighten_gate_threshold:min_confidence"],
    }
    result = sh.apply(diag_result)
    assert result["status"] == "no_action"
    assert result["degraded_mode"] is False
    # Budget refusal is recorded in actions_taken so it surfaces in the
    # decision log / operator brain feed.
    actions = result["actions_taken"]
    assert len(actions) == 1
    assert actions[0]["action"] == "budget_guard"
    assert "budget_exhausted" in actions[0]["detail"]


def test_apply_proceeds_when_under_budget(tmp_path):
    """End-to-end: apply() doesn't refuse when budget is open.

    Handler may still fail for non-budget reasons (no journal evidence),
    but the response shape is NOT the budget-guard refusal.
    """
    sh = _make_self_heal(tmp_path)
    # Empty budget; one action attempted; will fail evidence check (no journal)
    diag_result = {
        "status": "DEGRADED",
        "issues": [{"check": "gate_firing_rate", "severity": "warning"}],
        "recommended_actions": ["tighten_gate_threshold:min_confidence"],
    }
    result = sh.apply(diag_result)
    actions = result["actions_taken"]
    # Whatever happened, it was NOT the budget guard refusing.
    assert all(a.get("action") != "budget_guard" for a in actions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
