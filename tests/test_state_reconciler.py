"""Tests for the US-602 OANDA state reconciliation watchdog."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest

from src.scanner.automation.state_reconciler import (
    FAILURE_THRESHOLD,
    ReconciliationResult,
    StateReconciler,
)


def _write_state(path: Path, snapshot: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"portfolio_snapshot": snapshot, "schema_version": "2"}),
        encoding="utf-8",
    )


def _make_reconciler(
    tmp_path: Path,
    *,
    snapshot: Dict[str, Any],
    oanda: Dict[str, Any] | Callable[[], Dict[str, Any]] | Exception,
    events: List[Tuple[str, Dict[str, Any]]] | None = None,
) -> StateReconciler:
    state_path = tmp_path / "state.json"
    drift_log = tmp_path / "state_drift_log.jsonl"
    _write_state(state_path, snapshot)

    if isinstance(oanda, Exception):
        def _fetch() -> Dict[str, Any]:
            raise oanda  # type: ignore[misc]
    elif callable(oanda):
        _fetch = oanda  # type: ignore[assignment]
    else:
        def _fetch() -> Dict[str, Any]:
            return oanda  # type: ignore[return-value]

    def _publish(event_type: str, payload: Dict[str, Any]) -> None:
        if events is not None:
            events.append((event_type, payload))

    return StateReconciler(
        state_path=state_path,
        drift_log_path=drift_log,
        oanda_fetcher=_fetch,
        event_publisher=_publish,
    )


# ── AC: detects NAV drift ─────────────────────────────────────────────

def test_detects_nav_drift_and_autocorrects(tmp_path: Path) -> None:
    snap = {"nav": 99_484.61, "open_trades": 0, "unrealized_pl": 0.0}
    live = {"nav": 100_574.61, "open_trades": 0, "unrealized_pl": 0.0}
    rec = _make_reconciler(tmp_path, snapshot=snap, oanda=live)

    result = rec.reconcile_once()

    assert result.drifted is True
    assert "nav" in result.fields
    assert result.fields["nav"]["delta"] == pytest.approx(1090.0)
    assert result.corrections_applied["nav"] == 100_574.61

    # state.json rewritten
    state_on_disk = json.loads((tmp_path / "state.json").read_text())
    assert state_on_disk["portfolio_snapshot"]["nav"] == 100_574.61
    assert state_on_disk["portfolio_snapshot"]["source"] == "state_reconciler"


# ── AC: detects trade-count mismatch ──────────────────────────────────

def test_detects_trade_count_mismatch(tmp_path: Path) -> None:
    snap = {"nav": 100_000.0, "open_trades": 0, "unrealized_pl": 0.0}
    live = {"nav": 100_000.0, "open_trades": 3, "unrealized_pl": 0.0}
    rec = _make_reconciler(tmp_path, snapshot=snap, oanda=live)

    result = rec.reconcile_once()

    assert result.drifted is True
    assert result.fields["open_trades"]["before"] == 0
    assert result.fields["open_trades"]["after"] == 3
    assert result.fields["open_trades"]["delta"] == 3
    assert "nav" not in result.fields  # nav was equal


# ── AC: drift log schema ──────────────────────────────────────────────

def test_drift_log_schema(tmp_path: Path) -> None:
    snap = {"nav": 99_000.0, "open_trades": 0, "unrealized_pl": 0.0}
    live = {"nav": 100_000.0, "open_trades": 2, "unrealized_pl": 25.0}
    rec = _make_reconciler(tmp_path, snapshot=snap, oanda=live)

    rec.reconcile_once()

    log_path = tmp_path / "state_drift_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert set(entry.keys()) == {"timestamp", "before", "after", "oanda", "drift"}
    assert entry["before"]["nav"] == 99_000.0
    assert entry["after"]["nav"] == 100_000.0
    assert entry["drift"]["nav"]["delta"] == pytest.approx(1000.0)
    assert entry["drift"]["open_trades"]["delta"] == 2
    assert entry["drift"]["unrealized_pl"]["delta"] == pytest.approx(25.0)


# ── AC: handles OANDA API errors gracefully ───────────────────────────

def test_handles_oanda_api_error_gracefully(tmp_path: Path) -> None:
    snap = {"nav": 100_000.0, "open_trades": 0, "unrealized_pl": 0.0}
    events: List[Tuple[str, Dict[str, Any]]] = []
    rec = _make_reconciler(
        tmp_path,
        snapshot=snap,
        oanda=RuntimeError("connection reset"),
        events=events,
    )

    result = rec.reconcile_once()

    assert isinstance(result, ReconciliationResult)
    assert result.drifted is False
    assert result.error is not None
    assert "connection reset" in result.error
    # state.json must remain untouched
    on_disk = json.loads((tmp_path / "state.json").read_text())
    assert on_disk["portfolio_snapshot"]["nav"] == 100_000.0
    # single failure does not yet emit the offline event
    assert events == []


# ── AC: emits control.reconciler_offline after 3 consecutive failures ──

def test_emits_reconciler_offline_on_repeated_failure(tmp_path: Path) -> None:
    snap = {"nav": 100_000.0, "open_trades": 0, "unrealized_pl": 0.0}
    events: List[Tuple[str, Dict[str, Any]]] = []
    rec = _make_reconciler(
        tmp_path,
        snapshot=snap,
        oanda=RuntimeError("oanda 503"),
        events=events,
    )

    for _ in range(FAILURE_THRESHOLD):
        rec.reconcile_once()

    assert len(events) == 1
    event_type, payload = events[0]
    assert event_type == "control.reconciler_offline"
    assert payload["consecutive_failures"] >= FAILURE_THRESHOLD
    assert "oanda 503" in payload["reason"]

    # Additional failures should NOT re-emit (flag is latched until success)
    rec.reconcile_once()
    assert len(events) == 1
