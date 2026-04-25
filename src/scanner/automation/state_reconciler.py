"""
State Reconciliation Watchdog — US-602.

Periodically fetches the live OANDA account snapshot (NAV, unrealized P/L,
open trade count) and compares it against ``.claude/state.json``'s
``portfolio_snapshot`` block. When drift exceeds the documented thresholds,
the reconciler rewrites the snapshot to match OANDA truth and appends a
structured event to ``.claude/state_drift_log.jsonl``.

The bug class this guards against caused the 2026-04-24 audit to surface a
$1,090 stale-NAV discrepancy because nothing was forcing the cached snapshot
to re-sync with OANDA after process restarts or skipped journal updates.

The reconciler NEVER crashes the scanner: all OANDA/JSON errors are caught,
logged, and counted. After three consecutive failures a
``control.reconciler_offline`` event is emitted on the trading event bus so
the operator console can surface the outage.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Drift detection thresholds (see AC in US-602)
NAV_DRIFT_USD = 0.50
UPL_DRIFT_USD = 1.00
FAILURE_THRESHOLD = 3
DEFAULT_INTERVAL_SECONDS = 60


@dataclass
class ReconciliationResult:
    """Return value from :meth:`StateReconciler.reconcile_once`.

    ``fields`` maps each drifted field to ``{"before": <state>, "after": <oanda>,
    "delta": <oanda - state>}``. ``corrections_applied`` is the portion of the
    snapshot that was actually rewritten on disk (empty when no drift).
    ``error`` is set when the OANDA fetch or state-file read failed.
    """

    drifted: bool = False
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    corrections_applied: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class StateReconciler:
    """Polls OANDA and auto-corrects ``state.json`` drift."""

    def __init__(
        self,
        *,
        state_path: Optional[Path] = None,
        drift_log_path: Optional[Path] = None,
        oanda_fetcher: Optional[Callable[[], Dict[str, Any]]] = None,
        event_publisher: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        project_root: Optional[Path] = None,
    ) -> None:
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self._state_path = Path(state_path) if state_path else root / ".claude" / "state.json"
        self._drift_log_path = (
            Path(drift_log_path) if drift_log_path else root / ".claude" / "state_drift_log.jsonl"
        )
        self._fetcher = oanda_fetcher or self._default_oanda_fetcher
        self._publisher = event_publisher
        self._interval = max(1.0, float(interval_seconds))

        self._failure_streak = 0
        self._offline_emitted = False
        self._last_result: Optional[ReconciliationResult] = None
        self._lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ public

    @property
    def last_result(self) -> Optional[ReconciliationResult]:
        return self._last_result

    def reconcile_once(self) -> ReconciliationResult:
        """Fetch OANDA truth, compare, and auto-correct on drift.

        Never raises — all errors are caught and reflected on the result.
        """
        try:
            oanda = self._fetcher()
        except Exception as exc:  # noqa: BLE001 — watchdog must never crash caller
            self._record_failure(f"oanda_fetch_failed: {exc}")
            result = ReconciliationResult(error=str(exc))
            self._last_result = result
            return result

        try:
            state = self._load_state()
        except Exception as exc:  # noqa: BLE001
            self._record_failure(f"state_load_failed: {exc}")
            result = ReconciliationResult(error=str(exc))
            self._last_result = result
            return result

        snapshot = dict(state.get("portfolio_snapshot") or {})
        drift = self._compute_drift(snapshot, oanda)
        result = ReconciliationResult(drifted=bool(drift), fields=drift)

        if drift:
            try:
                corrections = self._apply_corrections(state, snapshot, oanda)
                self._append_drift_log(snapshot, oanda, drift, corrections)
                result.corrections_applied = corrections
            except Exception as exc:  # noqa: BLE001
                self._record_failure(f"state_write_failed: {exc}")
                result.error = str(exc)
                self._last_result = result
                return result

        # Success path resets the failure counter + offline flag.
        with self._lock:
            self._failure_streak = 0
            self._offline_emitted = False
        self._last_result = result
        return result

    async def run_forever(self) -> None:
        """Async loop — calls :meth:`reconcile_once` every ``interval`` seconds."""
        while True:
            try:
                self.reconcile_once()
            except Exception as exc:  # defensive — reconcile_once already catches
                logger.error("StateReconciler: unexpected error in run_forever: %s", exc)
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                raise

    def attach_to_loop(self, loop: asyncio.AbstractEventLoop) -> asyncio.Task:
        """Schedule :meth:`run_forever` as a task on ``loop`` and return it."""
        self._task = loop.create_task(self.run_forever())
        return self._task

    # ---------------------------------------------------------------- internal

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return {}
        with self._state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh) or {}

    def _write_state(self, state: Dict[str, Any]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, self._state_path)

    @staticmethod
    def _compute_drift(snapshot: Dict[str, Any], oanda: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        drift: Dict[str, Dict[str, Any]] = {}

        def _num(block: Dict[str, Any], key: str) -> float:
            try:
                return float(block.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        nav_state = _num(snapshot, "nav")
        nav_oanda = _num(oanda, "nav")
        if abs(nav_oanda - nav_state) > NAV_DRIFT_USD:
            drift["nav"] = {
                "before": nav_state,
                "after": nav_oanda,
                "delta": round(nav_oanda - nav_state, 4),
            }

        trades_state = int(_num(snapshot, "open_trades"))
        trades_oanda = int(_num(oanda, "open_trades"))
        if trades_state != trades_oanda:
            drift["open_trades"] = {
                "before": trades_state,
                "after": trades_oanda,
                "delta": trades_oanda - trades_state,
            }

        upl_state = _num(snapshot, "unrealized_pl")
        upl_oanda = _num(oanda, "unrealized_pl")
        if abs(upl_oanda - upl_state) > UPL_DRIFT_USD:
            drift["unrealized_pl"] = {
                "before": upl_state,
                "after": upl_oanda,
                "delta": round(upl_oanda - upl_state, 4),
            }
        return drift

    def _apply_corrections(
        self,
        state: Dict[str, Any],
        snapshot: Dict[str, Any],
        oanda: Dict[str, Any],
    ) -> Dict[str, Any]:
        corrected = dict(snapshot)
        for key in ("nav", "open_trades", "unrealized_pl"):
            if key in oanda:
                corrected[key] = oanda[key]
        corrected["source"] = "state_reconciler"
        corrected["reconciled_at"] = datetime.now(timezone.utc).isoformat()
        state["portfolio_snapshot"] = corrected
        self._write_state(state)
        return corrected

    def _append_drift_log(
        self,
        snapshot_before: Dict[str, Any],
        oanda: Dict[str, Any],
        drift: Dict[str, Dict[str, Any]],
        corrections: Dict[str, Any],
    ) -> None:
        self._drift_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "before": snapshot_before,
            "after": corrections,
            "oanda": oanda,
            "drift": drift,
        }
        try:
            with self._drift_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError as exc:
            logger.error("StateReconciler: failed to append drift log: %s", exc)

    def _record_failure(self, reason: str) -> None:
        logger.warning("StateReconciler: %s", reason)
        with self._lock:
            self._failure_streak += 1
            should_emit = (
                self._failure_streak >= FAILURE_THRESHOLD and not self._offline_emitted
            )
            if should_emit:
                self._offline_emitted = True
        if should_emit:
            self._emit_offline(reason)

    def _emit_offline(self, reason: str) -> None:
        payload = {
            "reason": reason,
            "consecutive_failures": self._failure_streak,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._publisher is not None:
            try:
                self._publisher("control.reconciler_offline", payload)
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("StateReconciler: custom publisher failed: %s", exc)
        try:
            from src.scanner.automation.event_bus import get_event_bus
            get_event_bus().publish("control.reconciler_offline", payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("StateReconciler: failed to emit offline event: %s", exc)

    # ---------------------------------------------------------------- fetcher

    def _default_oanda_fetcher(self) -> Dict[str, Any]:
        """Default OANDA fetcher — hits GET /v3/accounts/{id} + /openTrades."""
        import requests  # lazy import to keep test surface small

        token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")
        if not token or not acct:
            raise RuntimeError("OANDA_API_TOKEN / OANDA_ACCOUNT_ID not set")

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        acc_resp = requests.get(
            f"{base}/v3/accounts/{acct}",
            headers=headers,
            timeout=(5, 30),
        )
        acc_resp.raise_for_status()
        acc = acc_resp.json().get("account", {}) or {}

        trades_resp = requests.get(
            f"{base}/v3/accounts/{acct}/openTrades",
            headers=headers,
            timeout=(5, 30),
        )
        trades_resp.raise_for_status()
        open_trades = trades_resp.json().get("trades", []) or []

        def _f(x: Any) -> float:
            try:
                return float(x)
            except (TypeError, ValueError):
                return 0.0

        return {
            "nav": _f(acc.get("NAV")),
            "balance": _f(acc.get("balance")),
            "unrealized_pl": _f(acc.get("unrealizedPL")),
            "open_trades": len(open_trades),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
