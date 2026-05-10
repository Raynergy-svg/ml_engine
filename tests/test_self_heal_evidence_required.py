"""Test Angle A1 — evidence-required gating for self-heal proposals.

NO MOCKS per CLAUDE.md No-Mock Rule. Real SelfHeal, real disk via tmp_path.
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


def _write_journal(path: Path, n_closed: int, hours_ago: float = 1.0) -> None:
    """Write a journal with `n_closed` entries closed `hours_ago` ago."""
    close_time = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    trades = []
    for i in range(n_closed):
        trades.append({
            "trade_id": f"t{i}",
            "pair": "EUR_USD",
            "direction": "LONG",
            "close_time": close_time.isoformat(),
            "outcome": {"trade_won": (i % 2 == 0)},
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trades))


def _make_self_heal(tmp_path: Path) -> SelfHeal:
    """Build a SelfHeal instance with paths redirected to tmp_path."""
    sh = SelfHeal()
    sh.JOURNAL_PATH = tmp_path / "trade_journal.json"
    sh.CONFIG_ADJUSTMENTS_PATH = tmp_path / "config_adjustments.json"
    sh.DEBOUNCE_STATE_PATH = tmp_path / "debounce.json"
    sh.DECISION_LOG_PATH = tmp_path / "decisions.jsonl"
    return sh


def test_evidence_required_blocks_tighten_with_empty_journal(tmp_path):
    """No journal file → tighten refuses with insufficient_evidence."""
    sh = _make_self_heal(tmp_path)
    # No journal written
    ok, detail, _ = sh._handle_tighten_gate_threshold("min_confidence")
    assert ok is False
    assert "insufficient_evidence" in detail
    assert "n=0" in detail


def test_evidence_required_blocks_tighten_with_too_few_trades(tmp_path):
    """Journal with < EVIDENCE_MIN_RECENT_TRADES → tighten refuses."""
    sh = _make_self_heal(tmp_path)
    _write_journal(sh.JOURNAL_PATH, n_closed=4, hours_ago=1.0)
    ok, detail, _ = sh._handle_tighten_gate_threshold("min_confidence")
    assert ok is False
    assert "insufficient_evidence" in detail
    assert "n=4" in detail


def test_evidence_required_blocks_reset_with_empty_journal(tmp_path):
    """Reset proposals also gated by evidence — symmetric with tighten."""
    sh = _make_self_heal(tmp_path)
    ok, detail, _ = sh._handle_reset_gate_threshold("min_confidence")
    assert ok is False
    assert "insufficient_evidence" in detail


def test_evidence_window_excludes_old_trades(tmp_path):
    """Trades older than EVIDENCE_WINDOW_HOURS don't count."""
    sh = _make_self_heal(tmp_path)
    # 10 trades closed 24h ago — outside 6h window
    _write_journal(sh.JOURNAL_PATH, n_closed=10, hours_ago=24.0)
    recent = sh._count_recent_closed_trades()
    assert recent == 0
    ok, detail, _ = sh._handle_tighten_gate_threshold("min_confidence")
    assert ok is False
    assert "insufficient_evidence" in detail


def test_evidence_threshold_passes_with_enough_recent_trades(tmp_path):
    """≥5 recent closed trades → evidence check passes (handler may still
    fail for other reasons like missing default, but NOT for evidence)."""
    sh = _make_self_heal(tmp_path)
    _write_journal(sh.JOURNAL_PATH, n_closed=5, hours_ago=1.0)
    recent = sh._count_recent_closed_trades()
    assert recent == 5
    # Tighten will still fail due to no_default_in_yaml (not insufficient_evidence)
    ok, detail, _ = sh._handle_tighten_gate_threshold("min_confidence")
    # Either succeeds (if a default is found) or fails with a NON-evidence reason
    assert "insufficient_evidence" not in detail


def test_count_handles_corrupt_journal(tmp_path):
    """Corrupt JSON → count returns 0 (fail-closed)."""
    sh = _make_self_heal(tmp_path)
    sh.JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    sh.JOURNAL_PATH.write_text("{not valid json")
    assert sh._count_recent_closed_trades() == 0


def test_count_handles_missing_close_time(tmp_path):
    """Trade entries without close_time field don't count."""
    sh = _make_self_heal(tmp_path)
    sh.JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    sh.JOURNAL_PATH.write_text(json.dumps([
        {"trade_id": "t0", "pair": "EUR_USD"},  # no close_time
        {"trade_id": "t1", "pair": "EUR_USD"},
    ]))
    assert sh._count_recent_closed_trades() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
