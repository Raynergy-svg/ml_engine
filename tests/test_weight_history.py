"""Tests for US-515: Agent Weight Inspector — RL weight history tracking.

Covers:
- ExecutionManager.sync_rl_weights() atomic append to weight_history.jsonl
- Row schema conformance
- WeightInspector reads and renders rows correctly
- Integration: mocked RL sync appends, inspector refresh sees new rows
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Unit: ExecutionManager.sync_rl_weights appends rows with correct schema
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def test_sync_rl_weights_appends_three_rl_syncs(tmp_path: Path) -> None:
    """Simulate 3 RL syncs and verify weight_history.jsonl contains all rows."""
    from src.scanner.execution import ExecutionManager

    em = ExecutionManager()
    hist = tmp_path / "weight_history.jsonl"

    # Sync 1: win — trend gains, mean_reversion loses
    appended1 = em.sync_rl_weights(
        weights_before={"trend": 1.10, "mean_reversion": 0.90},
        weights_after={"trend": 1.20, "mean_reversion": 0.85},
        trade_id="T-1001",
        pnl=42.50,
        reason="trade_closed_win",
        history_path=str(hist),
    )
    # Sync 2: loss — both drift down
    appended2 = em.sync_rl_weights(
        weights_before={"trend": 1.20, "mean_reversion": 0.85},
        weights_after={"trend": 1.05, "mean_reversion": 0.78},
        trade_id="T-1002",
        pnl=-35.10,
        reason="trade_closed_loss",
        history_path=str(hist),
    )
    # Sync 3: win — recovery
    appended3 = em.sync_rl_weights(
        weights_before={"trend": 1.05, "mean_reversion": 0.78},
        weights_after={"trend": 1.15, "mean_reversion": 0.80},
        trade_id="T-1003",
        pnl=18.75,
        reason="trade_closed_win",
        history_path=str(hist),
    )

    # Each sync touched 2 agents → 6 rows total
    assert appended1 == 2
    assert appended2 == 2
    assert appended3 == 2

    rows = _read_jsonl(hist)
    assert len(rows) == 6, f"expected 6 rows, got {len(rows)}"

    # Schema check: {ts, trade_id, agent, weight_before, weight_after, pnl, reason}
    required = {"ts", "trade_id", "agent", "weight_before", "weight_after", "pnl", "reason"}
    for row in rows:
        assert required.issubset(row.keys()), f"row missing fields: {row}"

    # Trade ids preserved per sync
    trade_ids = {r["trade_id"] for r in rows}
    assert trade_ids == {"T-1001", "T-1002", "T-1003"}

    # Per-agent ordering within a sync is sorted, so first two rows = T-1001
    first_two = [r for r in rows if r["trade_id"] == "T-1001"]
    assert len(first_two) == 2
    agent_names = sorted(r["agent"] for r in first_two)
    assert agent_names == ["mean_reversion", "trend"]

    # Weight values preserved
    trend_t1001 = next(r for r in first_two if r["agent"] == "trend")
    assert trend_t1001["weight_before"] == pytest.approx(1.10)
    assert trend_t1001["weight_after"] == pytest.approx(1.20)
    assert trend_t1001["pnl"] == pytest.approx(42.50)
    assert trend_t1001["reason"] == "trade_closed_win"


def test_sync_rl_weights_empty_dicts_noop(tmp_path: Path) -> None:
    """No agents → no rows written, returns 0."""
    from src.scanner.execution import ExecutionManager

    em = ExecutionManager()
    hist = tmp_path / "weight_history.jsonl"
    appended = em.sync_rl_weights(
        weights_before={},
        weights_after={},
        trade_id="T-empty",
        pnl=0.0,
        reason="noop",
        history_path=str(hist),
    )
    assert appended == 0
    assert not hist.exists() or hist.read_text() == ""


def test_sync_rl_weights_is_append_only(tmp_path: Path) -> None:
    """Subsequent syncs must append — not overwrite."""
    from src.scanner.execution import ExecutionManager

    em = ExecutionManager()
    hist = tmp_path / "weight_history.jsonl"
    em.sync_rl_weights(
        weights_before={"trend": 1.0}, weights_after={"trend": 1.1},
        trade_id="A", pnl=1.0, reason="r", history_path=str(hist),
    )
    em.sync_rl_weights(
        weights_before={"trend": 1.1}, weights_after={"trend": 1.2},
        trade_id="B", pnl=2.0, reason="r", history_path=str(hist),
    )
    rows = _read_jsonl(hist)
    assert [r["trade_id"] for r in rows] == ["A", "B"]


# ---------------------------------------------------------------------------
# WeightInspector reads rows correctly
# ---------------------------------------------------------------------------

def test_weight_inspector_reads_history(tmp_path: Path) -> None:
    """Inspector filters history by agent and computes delta over window."""
    from src.tui.screens.agents_screen import WeightInspector

    hist = tmp_path / "weight_history.jsonl"
    weights = tmp_path / "agent_weights.json"

    # Write 4 rows for "trend": 1.00 -> 1.05 -> 1.10 -> 1.08
    ts = "2026-04-16T00:00:00+00:00"
    seq = [(1.00, 1.05), (1.05, 1.10), (1.10, 1.08)]
    with hist.open("w") as f:
        for i, (wb, wa) in enumerate(seq):
            row = {
                "ts": ts, "trade_id": f"T-{i}", "agent": "trend",
                "weight_before": wb, "weight_after": wa,
                "pnl": 10.0, "reason": "trade_closed_win",
            }
            f.write(json.dumps(row) + "\n")
        # One row for mean_reversion
        f.write(json.dumps({
            "ts": ts, "trade_id": "T-9", "agent": "mean_reversion",
            "weight_before": 0.90, "weight_after": 0.85,
            "pnl": -5.0, "reason": "trade_closed_loss",
        }) + "\n")

    # Current weights snapshot
    weights.write_text(json.dumps({
        "_global": {"trend": 1.08, "mean_reversion": 0.85},
    }))

    inspector = WeightInspector()
    inspector.set_paths(history_path=hist, weights_path=weights)
    inspector.refresh_data(window=50)

    # Current weights read
    assert inspector._current["trend"] == pytest.approx(1.08)
    assert inspector._current["mean_reversion"] == pytest.approx(0.85)

    # Per-agent history captured
    assert inspector._per_agent_history["trend"] == [1.05, 1.10, 1.08]
    assert inspector._per_agent_history["mean_reversion"] == [0.85]

    # Gainers/losers summary: trend delta = 1.08 - 1.05 = +0.03
    gainer_names = [a for a, _ in inspector._gainers]
    assert "trend" in gainer_names


def test_weight_inspector_handles_corrupt_jsonl(tmp_path: Path) -> None:
    """JSON Safety Gate: corrupt lines skipped, no crash."""
    from src.tui.screens.agents_screen import WeightInspector

    hist = tmp_path / "weight_history.jsonl"
    weights = tmp_path / "agent_weights.json"
    hist.write_text(
        'not json\n'
        + json.dumps({
            "ts": "x", "trade_id": "T-1", "agent": "trend",
            "weight_before": 1.0, "weight_after": 1.1, "pnl": 0.0, "reason": "r",
        }) + "\n"
        + '{"broken": \n'
    )
    weights.write_text("{}")

    inspector = WeightInspector()
    inspector.set_paths(history_path=hist, weights_path=weights)
    inspector.refresh_data(window=50)
    # One valid row survives
    assert inspector._per_agent_history.get("trend") == [1.1]


def test_weight_inspector_handles_missing_files(tmp_path: Path) -> None:
    """Missing files → empty fallback, no exception."""
    from src.tui.screens.agents_screen import WeightInspector

    inspector = WeightInspector()
    inspector.set_paths(
        history_path=tmp_path / "does_not_exist.jsonl",
        weights_path=tmp_path / "missing.json",
    )
    inspector.refresh_data(window=50)
    assert inspector._current == {}
    assert inspector._per_agent_history == {}


# ---------------------------------------------------------------------------
# Integration: sync_rl_weights writes → inspector sees updates on refresh
# ---------------------------------------------------------------------------

def test_integration_sync_then_inspector_refresh(tmp_path: Path) -> None:
    """Mocked RL sync (sync_rl_weights) appends; inspector picks up on next refresh."""
    from src.scanner.execution import ExecutionManager
    from src.tui.screens.agents_screen import WeightInspector

    em = ExecutionManager()
    hist = tmp_path / "weight_history.jsonl"
    weights = tmp_path / "agent_weights.json"
    weights.write_text(json.dumps({"_global": {"trend": 1.0}}))

    inspector = WeightInspector()
    inspector.set_paths(history_path=hist, weights_path=weights)

    # Initial refresh — empty history
    inspector.refresh_data(window=50)
    assert inspector._per_agent_history == {}

    # Simulate a trade close triggering RL sync
    em.sync_rl_weights(
        weights_before={"trend": 1.00},
        weights_after={"trend": 1.08},
        trade_id="INT-1",
        pnl=20.0,
        reason="trade_closed_win",
        history_path=str(hist),
    )

    # Update weights snapshot (what agent_weights.json would show post-update)
    weights.write_text(json.dumps({"_global": {"trend": 1.08}}))

    # Inspector refresh — should see the new row
    inspector.refresh_data(window=50)
    assert inspector._current["trend"] == pytest.approx(1.08)
    assert inspector._per_agent_history["trend"] == [1.08]

    # Second sync
    em.sync_rl_weights(
        weights_before={"trend": 1.08},
        weights_after={"trend": 1.15},
        trade_id="INT-2",
        pnl=25.0,
        reason="trade_closed_win",
        history_path=str(hist),
    )
    inspector.refresh_data(window=50)
    assert inspector._per_agent_history["trend"] == [1.08, 1.15]
