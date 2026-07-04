"""Adversarial-review atomicity fixes for the RL backfill sync (2026-07-04).

Reproduce-then-resolve for review findings #4/#5:

- #4: the `rl_weights_applied` flag must be set ONLY after the agent-weight
  update for that entry actually succeeds — never before. If a weight update
  throws, the entry must stay retryable (unflagged), not be stranded as
  "handled" with no weights moved.
- #5: a mid-batch failure must leave the on-disk journal CONSISTENT — every
  flagged entry corresponds to a persisted weight update; a failed entry is
  left unflagged so the next cycle retries it.

Induced without mocks (No-Mock Rule) by feeding a real malformed
`agent_reasons` (`[42]` — a list whose element has no `.get`) that makes
`ScannerAgentTeam.update_weights_from_outcome` genuinely raise for that one
entry while the well-formed entry in the same batch succeeds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.scanner.execution import ExecutionConfig, ExecutionManager


def _write_journal(tmp_path: Path, entries: list) -> None:
    (tmp_path / "trained_data").mkdir(exist_ok=True)
    (tmp_path / "trained_data" / "trade_journal_rl.json").write_text(json.dumps(entries))


def _read_journal(tmp_path: Path) -> list:
    return json.loads((tmp_path / "trained_data" / "trade_journal_rl.json").read_text())


def _run(tmp_path: Path, mgr: ExecutionManager):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return mgr.apply_pending_rl_weight_updates()
    finally:
        os.chdir(original_cwd)


_GOOD_REASONS = [{"name": "trend", "passed": True, "reason": "up", "score": 0.7}]


def _entry(tid, agent_reasons, win=True):
    return {
        "trade_id": tid,
        "pair": "EUR_USD",
        "timestamp": "2026-06-01T00:00:00Z",
        "confidence": 0.6,
        "outcome": "win" if win else "loss",
        "realized_pl": 50.0 if win else -50.0,
        "agents": {"agent_reasons": agent_reasons},
        "regime": {"volatility_regime": "NORMAL"},
    }


def test_failed_weight_update_leaves_entry_retryable(tmp_path):
    """The entry whose weight update raises must NOT be flagged applied — it
    stays a candidate for the next cycle, rather than being stranded."""
    good = _entry("G1", _GOOD_REASONS)
    bad = _entry("B1", [42])  # 42.get(...) -> AttributeError inside the updater
    _write_journal(tmp_path, [good, bad])
    mgr = ExecutionManager(config=ExecutionConfig())

    result = _run(tmp_path, mgr)

    assert result["applied"] == 1  # only the good one scored
    entries = {e["trade_id"]: e for e in _read_journal(tmp_path)}
    assert entries["G1"].get("rl_weights_applied") is True
    assert entries["B1"].get("rl_weights_applied") is not True  # retryable


def test_second_run_retries_the_previously_failed_entry(tmp_path):
    """After the bad entry is repaired, a subsequent run picks it up — proof
    it was never falsely marked done."""
    good = _entry("G2", _GOOD_REASONS)
    bad = _entry("B2", [42])
    _write_journal(tmp_path, [good, bad])
    mgr = ExecutionManager(config=ExecutionConfig())
    _run(tmp_path, mgr)

    # repair the bad entry's agent_reasons on disk, rerun
    entries = _read_journal(tmp_path)
    for e in entries:
        if e["trade_id"] == "B2":
            e["agents"]["agent_reasons"] = _GOOD_REASONS
    _write_journal(tmp_path, entries)

    result = _run(tmp_path, mgr)

    assert result["applied"] == 1  # the repaired B2, now scored
    entries = {e["trade_id"]: e for e in _read_journal(tmp_path)}
    assert entries["B2"].get("rl_weights_applied") is True


def test_journal_consistent_after_partial_failure(tmp_path):
    """On-disk journal after a mixed batch: flagged iff scored. No entry is
    both unscored-and-flagged (the strand) nor scored-and-unflagged."""
    entries_in = [
        _entry("A", _GOOD_REASONS),
        _entry("B", [42]),
        _entry("C", _GOOD_REASONS, win=False),
    ]
    _write_journal(tmp_path, entries_in)
    mgr = ExecutionManager(config=ExecutionConfig())

    _run(tmp_path, mgr)

    on_disk = {e["trade_id"]: e for e in _read_journal(tmp_path)}
    assert on_disk["A"].get("rl_weights_applied") is True
    assert on_disk["C"].get("rl_weights_applied") is True
    assert on_disk["B"].get("rl_weights_applied") is not True
