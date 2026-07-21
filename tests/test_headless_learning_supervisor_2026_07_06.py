"""Headless Learning Supervisor — P1 wiring (docs/ENGINEERING_BRAIN.md, commit
b836075): ``offline_learning_cycle.py`` is now the single headless entrypoint
for BOTH the risk-calibration learner AND ``ExecutionManager.
apply_pending_rl_weight_updates`` (the RL agent-weight sync, commit 51b85bf).

Root cause reproduced here: before this change, ``apply_pending_rl_weight_
updates``'s ONLY production caller was ``src/tui/embedded_scanner.py``
(``_run_smart_loop``) — i.e. it was welded to the TUI process. With the TUI
dormant (confirmed live: ``.claude/heartbeat.json`` reads
``scanner_alive: false, supervisor_alive: true, writer: tier7_supervisor`` as
of 2026-07-06T20:30Z — no ``embedded_scanner`` process in `ps`), a resolved,
agent-scoreable trade closed by ANY path (OutcomeBackfill, a future
scanner run, etc.) would sit forever with ``rl_weights_applied`` unset. This
suite proves the fix: the SAME headless, already-scheduled, market-closed
batch job (``run_cycle``) now drains it — with zero TUI/embedded_scanner
import anywhere in this file or its call graph.

NO MOCKS per CLAUDE.md No-Mock Rule. Real ``run_cycle``, real
``ExecutionManager``/``ScannerAgentTeam``, real disk via ``tmp_path``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import scripts.offline_learning_cycle as olc

SATURDAY = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

_AGENT_REASONS = [
    {"name": "trend", "passed": True, "reason": "trend up", "score": 0.7},
    {"name": "risk_sentinel", "passed": False, "reason": "elevated dd", "score": 0.3},
]


def _patch_paths(monkeypatch, tmp_path: Path) -> Path:
    """Same sandboxing convention as test_offline_learning_cycle_2026_07_04.py
    — sandboxes the calibration-side module constants. Nested under
    tmp_path/"trained_data" so JOURNAL_PATH.parent.parent (what
    _run_rl_weight_sync chdirs into) matches this test's tmp_path root, and
    apply_pending_rl_weight_updates's hardcoded relative
    "trained_data/trade_journal_rl.json" resolves to the SAME file this test
    writes/reads.
    """
    data_root = tmp_path
    (data_root / "trained_data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(olc, "JOURNAL_PATH", data_root / "trained_data" / "trade_journal_rl.json")
    monkeypatch.setattr(olc, "RETRAIN_REQUESTS_DIR", data_root / "trained_data" / "retrain_requests")
    monkeypatch.setattr(olc, "PROCESSED_DIR", data_root / "trained_data" / "retrain_requests" / "_processed")
    monkeypatch.setattr(olc, "STATE_PATH", data_root / "trained_data" / "risk_calibration_state.json")
    monkeypatch.setattr(olc, "HISTORY_PATH", data_root / "trained_data" / "learning_loop" / "history.jsonl")
    monkeypatch.setattr(olc, "CURSOR_PATH", data_root / ".claude" / "learning_loop_cursor.json")
    monkeypatch.setattr(olc, "BRAIN_STATUS_PATH", data_root / ".claude" / "brain" / "status.json")
    monkeypatch.setattr(olc, "BRAIN_FEED_PATH", data_root / ".claude" / "brain" / "feed.jsonl")
    # A halt/arm state file this job must NEVER touch — matches the existing
    # convention in test_offline_learning_cycle_2026_07_04.py.
    state_path = data_root / ".claude" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"halted": True, "oanda_environment": "practice"}))
    return state_path


def _write_journal(tmp_path: Path, entries: list) -> Path:
    path = tmp_path / "trained_data" / "trade_journal_rl.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))
    return path


def _scoreable_scanner_entry(trade_id: str, ts: str, win: bool) -> dict:
    """A resolved, agent-having FX-scanner trade — exactly the shape
    apply_pending_rl_weight_updates looks for (agents.agent_reasons present,
    rl_weights_applied unset, outcome resolved)."""
    return {
        "trade_id": trade_id,
        "pair": "EUR_USD",
        "timestamp": ts,
        "outcome": "win" if win else "loss",
        "realized_pl": 50.0 if win else -30.0,
        "agents": {"agent_reasons": _AGENT_REASONS},
        "regime": {"volatility_regime": "NORMAL"},
    }


def test_closed_trade_reaches_agent_weights_headless_tui_absent(tmp_path, monkeypatch):
    """A single closed, resolved, agent-scored trade — never opened by the
    TUI, never touched by embedded_scanner — flows through run_cycle() alone
    into ScannerAgentTeam.update_weights_from_outcome. Reproduces the exact
    gap named in docs/ENGINEERING_BRAIN.md P1 ('self-heal produces
    adjustments only the dead TUI can consume') for the sibling RL-weight
    subsystem, then proves it resolved."""
    _patch_paths(monkeypatch, tmp_path)
    _write_journal(tmp_path, [_scoreable_scanner_entry("7001", "2026-06-01T00:00:00Z", win=True)])

    result = olc.run_cycle(now=SATURDAY)

    assert result["ran"] is True
    assert result["rl_weight_sync"]["applied"] == 1
    assert result["rl_weight_sync"]["weights_updated"] is True

    journal = json.loads((tmp_path / "trained_data" / "trade_journal_rl.json").read_text())
    assert journal[0]["rl_weights_applied"] is True

    weights_path = tmp_path / "trained_data" / "models" / "agent_weights.json"
    assert weights_path.exists(), "the headless path must persist agent_weights.json, same as the TUI path"
    weights = json.loads(weights_path.read_text())
    assert weights["_global"]["trend"] != 1.15  # base weight moved


def test_rl_weight_sync_is_idempotent_across_cycles(tmp_path, monkeypatch):
    """Running the batch job twice (e.g. two Saturday windows) must not
    double-apply the same trade's outcome to agent weights — the
    rl_weights_applied flag is the shared idempotency guard between this
    headless caller and the (dormant) TUI caller."""
    _patch_paths(monkeypatch, tmp_path)
    _write_journal(tmp_path, [_scoreable_scanner_entry("7002", "2026-06-01T00:00:00Z", win=False)])

    first = olc.run_cycle(now=SATURDAY)
    second = olc.run_cycle(now=datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc))

    assert first["rl_weight_sync"]["applied"] == 1
    assert second["rl_weight_sync"] == {
        "applied": 0,
        "weights_updated": False,
        "detail": "no pending backfilled outcomes",
    }


def test_rl_weight_sync_never_blocks_calibration_on_failure(tmp_path, monkeypatch):
    """A malformed journal that breaks the RL-weight-sync scan must not
    prevent the (independent) calibration cycle from running to completion —
    the two subsystems share one entrypoint but must fail independently."""
    _patch_paths(monkeypatch, tmp_path)
    # A dict-shaped "agents" field whose agent_reasons is itself malformed
    # (not a list of dicts) — exercises the try/except in
    # apply_pending_rl_weight_updates without crashing the whole cycle.
    bad_entry = _scoreable_scanner_entry("7003", "2026-06-01T00:00:00Z", win=True)
    bad_entry["agents"] = {"agent_reasons": "not-a-list"}
    _write_journal(tmp_path, [bad_entry])

    result = olc.run_cycle(now=SATURDAY)

    assert result["ran"] is True
    # Either cleanly skipped (no agent context usable) or reported as an
    # error — never raises, and the calibration decision key is always present.
    assert "decision" in result
    assert result["rl_weight_sync"]["weights_updated"] is False


def test_rl_weight_sync_survives_unreachable_data_root(tmp_path, monkeypatch):
    """os.chdir(data_root) must be INSIDE the guarded try/except -- a missing
    or inaccessible data_root (misconfigured OLC_DATA_ROOT, fresh deployment,
    permissions issue) must be caught and reported, never propagate out of
    _run_rl_weight_sync and crash the whole offline learning cycle."""
    data_root = tmp_path
    (data_root / "trained_data").mkdir(parents=True, exist_ok=True)
    # JOURNAL_PATH.parent.parent must point at a path that does NOT exist on
    # disk, so os.chdir() itself raises FileNotFoundError.
    monkeypatch.setattr(
        olc, "JOURNAL_PATH",
        data_root / "does_not_exist" / "trained_data" / "trade_journal_rl.json",
    )

    before_cwd = os.getcwd()
    result = olc._run_rl_weight_sync()

    assert result["applied"] == 0
    assert result["weights_updated"] is False
    assert "error" in result["detail"]
    # cwd must be restored even though chdir itself is what failed.
    assert os.getcwd() == before_cwd


def test_rl_weight_sync_never_touches_halt_or_arm_state(tmp_path, monkeypatch):
    """Reproduce-then-resolve for the safety rail: even with a real
    weight-update firing, state.json (halt/oanda_environment) must be
    byte-identical before and after."""
    state_path = _patch_paths(monkeypatch, tmp_path)
    before = state_path.read_text()
    _write_journal(tmp_path, [_scoreable_scanner_entry("7004", "2026-06-01T00:00:00Z", win=True)])

    result = olc.run_cycle(now=SATURDAY)

    assert result["rl_weight_sync"]["applied"] == 1  # sanity: the sync really ran
    assert state_path.read_text() == before, "halt/practice-pin state must be untouched"


def test_rl_weight_sync_absent_from_production_journal_paths(tmp_path, monkeypatch):
    """Sandbox-leak guard: prove _run_rl_weight_sync derives its chdir target
    from the (mockable) JOURNAL_PATH module constant, not the separate
    OLC_DATA_ROOT-bound _DATA_ROOT — the exact bug caught before shipping
    this change. If this regresses, patching JOURNAL_PATH alone would no
    longer sandbox the RL-sync call, and it would silently read/write the
    real repo's trained_data/trade_journal_rl.json."""
    _patch_paths(monkeypatch, tmp_path)
    _write_journal(tmp_path, [_scoreable_scanner_entry("7005", "2026-06-01T00:00:00Z", win=True)])

    result = olc.run_cycle(now=SATURDAY)

    assert result["rl_weight_sync"]["applied"] == 1
    # The real repo's production journal (this file's own REPO_ROOT) must be
    # completely unaffected by this in-process test run.
    prod_journal = olc.ROOT / "trained_data" / "trade_journal_rl.json"
    prod_entries = json.loads(prod_journal.read_text())
    assert not any(e.get("trade_id") == "7005" for e in prod_entries if isinstance(e, dict))
