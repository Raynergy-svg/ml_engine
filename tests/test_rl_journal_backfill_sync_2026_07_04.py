"""Reconnect backfilled journal outcomes to the RL agent-weight update path.

Root cause (2026-07-04, confirmed independently across 3 sessions — see
memory observations 1967 [2026-05-12], 14959/15342 [2026-07-02/03]):
``OutcomeBackfill._apply_closure()`` (src/scanner/automation/outcome_backfill.py)
and ``TrendJournalSync`` both stamp ``entry["outcome"]`` (as a plain string)
directly onto journal entries when a trade closes. ``sync_closed_trades_rl``'s
pending-detection guard is ``entry.get("outcome") is None`` — so once either
writer stamps ANY outcome value, ``sync_closed_trades_rl`` considers the
entry "already handled" and never calls
``ScannerAgentTeam.update_weights_from_outcome`` for it. The last live
trade to ever pass through the real weight-update path was trade_id 1261,
closed 2026-04-16 — every trade closed since then that got its outcome
filled in by ``OutcomeBackfill`` (the common case: scanner-lane trades that
close while the bot process isn't running) silently skipped RL scoring.

``ExecutionManager.apply_pending_rl_weight_updates`` is outcome-shape-agnostic
(handles both the string shape written by OutcomeBackfill/TrendJournalSync and
the dict shape written by sync_closed_trades_rl itself) and tracks its own
``rl_weights_applied`` flag so it is safe to call repeatedly — from the live
per-cycle loop and from an offline batch job — without double-scoring a trade.

NO MOCKS per CLAUDE.md No-Mock Rule. Real ExecutionManager, real
ScannerAgentTeam, real disk via tmp_path.
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


def _run_in(tmp_path: Path, mgr: ExecutionManager):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return mgr.apply_pending_rl_weight_updates()
    finally:
        os.chdir(original_cwd)


_AGENT_REASONS = [
    {"name": "trend", "passed": True, "reason": "trend up", "score": 0.7},
    {"name": "risk_sentinel", "passed": False, "reason": "elevated dd", "score": 0.3},
]

_BACKFILLED_STRING_ENTRY = {
    "trade_id": "9101",
    "pair": "EUR_USD",
    "direction": "LONG",
    "timestamp": "2026-06-01T00:00:00Z",
    "entry_price": 1.10,
    "outcome": "win",
    "realized_pl": 120.0,
    "backfill_source": "oanda_tx_stream",
    "agents": {"agent_reasons": _AGENT_REASONS},
    "regime": {"volatility_regime": "NORMAL"},
}


def test_backfilled_string_outcome_entry_triggers_weight_update(tmp_path):
    """A trade whose outcome was filled in by OutcomeBackfill (string shape,
    never scored) must still reach update_weights_from_outcome."""
    _write_journal(tmp_path, [dict(_BACKFILLED_STRING_ENTRY)])
    mgr = ExecutionManager(config=ExecutionConfig())

    result = _run_in(tmp_path, mgr)

    assert result["applied"] == 1
    assert result["weights_updated"] is True

    entries = _read_journal(tmp_path)
    assert entries[0]["rl_weights_applied"] is True

    weights_path = tmp_path / "trained_data" / "models" / "agent_weights.json"
    assert weights_path.exists(), "weight update must persist agent_weights.json"
    weights = json.loads(weights_path.read_text())
    assert weights["_global"]["trend"] != 1.15  # base weight must have moved


def test_second_call_is_idempotent_no_double_scoring(tmp_path):
    """Calling the sync twice must not apply the same trade's outcome to
    agent weights a second time."""
    _write_journal(tmp_path, [dict(_BACKFILLED_STRING_ENTRY)])
    mgr = ExecutionManager(config=ExecutionConfig())

    first = _run_in(tmp_path, mgr)
    second = _run_in(tmp_path, mgr)

    assert first["applied"] == 1
    assert second == {
        "applied": 0,
        "weights_updated": False,
        "detail": "no pending backfilled outcomes",
    }


def test_trend_lane_entry_stays_excluded_even_when_resolved(tmp_path):
    """rl_eligible=False (trend lane) must never be scored, matching the
    existing sync_closed_trades_rl guard (test_execution_rl_sync_trend_lane_guard)."""
    trend_entry = {
        "trade_id": "9102",
        "pair": "USD_JPY",
        "lane": "trend",
        "rl_eligible": False,
        "outcome": "loss",
        "realized_pl": -50.0,
        "agents": {"agent_votes": None},
        "regime": None,
    }
    _write_journal(tmp_path, [trend_entry])
    mgr = ExecutionManager(config=ExecutionConfig())

    result = _run_in(tmp_path, mgr)

    assert result == {
        "applied": 0,
        "weights_updated": False,
        "detail": "no pending backfilled outcomes",
    }
    entries = _read_journal(tmp_path)
    assert "rl_weights_applied" not in entries[0]


def test_dict_shaped_outcome_from_sync_closed_trades_rl_also_scores(tmp_path):
    """The dict shape sync_closed_trades_rl itself writes (trade_won key)
    must also be handled — this is the self-healing backstop case where a
    prior run wrote the outcome but crashed before the weight update
    completed."""
    entry = {
        "trade_id": "9103",
        "pair": "GBP_USD",
        "outcome": {"trade_won": True, "realized_pl": 80.0},
        "agents": {"agent_reasons": _AGENT_REASONS},
        "regime": {"volatility_regime": "HIGH"},
    }
    _write_journal(tmp_path, [entry])
    mgr = ExecutionManager(config=ExecutionConfig())

    result = _run_in(tmp_path, mgr)

    assert result["applied"] == 1
    entries = _read_journal(tmp_path)
    assert entries[0]["rl_weights_applied"] is True


def test_entry_without_agent_context_is_skipped_not_crashed(tmp_path):
    """Entries with a resolved outcome but no agent_reasons (e.g. malformed
    legacy rows) can't be scored — must be skipped cleanly, never raise."""
    entry = {
        "trade_id": "9104",
        "pair": "AUD_USD",
        "outcome": "win",
        "agents": {},
        "regime": {"volatility_regime": "NORMAL"},
    }
    _write_journal(tmp_path, [entry])
    mgr = ExecutionManager(config=ExecutionConfig())

    result = _run_in(tmp_path, mgr)

    assert result == {
        "applied": 0,
        "weights_updated": False,
        "detail": "no pending backfilled outcomes",
    }
