"""Maintenance/diagnostics report — the supervisor's honest per-tick statement of
what needs operator attention. READ-ONLY over disk artifacts; writes ONE snapshot
file (``.claude/maintenance_report.json``) atomically.

Root cause this closes (2026-07-03 autonomy audit): detection existed without
reporting — alerts sat unacknowledged in ``alert_state.json``, self-heal staged
config adjustments no live process consumed, and nothing surfaced any of it. This
module aggregates those signals into one snapshot that tier7_state (and therefore
AXIOM) embeds, so "maintenance/diagnostics" is a report the operator actually sees
instead of scattered JSON files.

Scope discipline (same contract as PostTradeDiagnostics): stdlib-only reads, no
scanner-subsystem imports, never raises out of ``build_maintenance_report`` — a
corrupt source file degrades that section to an ``{"error": ...}`` stub, it never
kills the supervisor tick. This module has NO control path: it cannot halt/unhalt,
trade, ack alerts, or consume adjustments. Reporting only.

Why it does NOT consume config adjustments (deliberate): ``ConfigAdjuster``
persists ``applied_ids`` in the shared ``config_adjustments.json``; a headless
process "applying" adjustments to a throwaway config and saving would mark them
consumed on disk and the real scanner would never receive them. Until the headless
learning supervisor (P1) owns a durable config overlay, the honest fix is to COUNT
and AGE the backlog loudly, not to eat it.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MAINTENANCE_REPORT_REL = ".claude/maintenance_report.json"

# A lane artifact older than this is reported stale (trend writes hourly; 2h grace
# matches the daily-monitor contract in NOTES).
_TREND_STALE_S = 2 * 3600
_EQUITY_STALE_S = 2 * 3600
_HEARTBEAT_FRESH_S = 90


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _age_s(path: Path, now: float) -> Optional[float]:
    try:
        return max(0.0, now - path.stat().st_mtime)
    except OSError:
        return None


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (OSError, ProcessLookupError, ValueError):
        return False
    return True


def _adjustments_section(root: Path) -> Dict[str, Any]:
    """Backlog of the config-adjustment pipeline, both stages.

    - approved_unconsumed: entries in config_adjustments.json ``history`` whose id is
      NOT in ``applied_ids`` — approved but no live process has applied them. Identity
      mirrors ConfigAdjuster.apply_adjustments exactly (proposal_id, else key@timestamp).
    - pending_proposals: rows in pending_adjustments.json still awaiting operator
      approval (status == "pending"); ``invalid`` counted separately.
    """
    try:
        approved = _read_json(root / ".claude" / "config_adjustments.json")
        history: List[Dict[str, Any]] = []
        applied_ids: set = set()
        if isinstance(approved, list):  # legacy config_tuner format
            history = [e for e in approved if isinstance(e, dict)]
        elif isinstance(approved, dict):
            history = [e for e in approved.get("history", []) if isinstance(e, dict)]
            applied_ids = set(approved.get("applied_ids", []))
        unconsumed = []
        for entry in history:
            entry_id = entry.get("proposal_id") or f"{entry.get('key')}@{entry.get('timestamp', '')}"
            if entry_id not in applied_ids:
                unconsumed.append(entry)

        pending_raw = _read_json(root / ".claude" / "pending_adjustments.json")
        proposals = pending_raw.get("proposals", []) if isinstance(pending_raw, dict) else []
        pending = [p for p in proposals if isinstance(p, dict) and p.get("status") == "pending"]
        invalid = [p for p in proposals if isinstance(p, dict) and p.get("status") == "invalid"]

        def _oldest_ts(rows: List[Dict[str, Any]]) -> Optional[str]:
            stamps = sorted(str(r.get("timestamp", "")) for r in rows if r.get("timestamp"))
            return stamps[0] if stamps else None

        return {
            "approved_unconsumed": len(unconsumed),
            "approved_unconsumed_keys": sorted({str(e.get("key")) for e in unconsumed})[:10],
            "oldest_unconsumed_ts": _oldest_ts(unconsumed),
            "pending_proposals": len(pending),
            "oldest_pending_ts": _oldest_ts(pending),
            "invalid_proposals": len(invalid),
        }
    except Exception as exc:  # noqa: BLE001 — degrade the section, never the tick
        return {"error": str(exc)}


# Alert routing map (2026-07-03, P0 "detection without action"): every alert type
# names its standing remediation so an active alert is never an unowned dead end.
# The supervisor does NOT mutate alert_state.json (AlertManager's owner process
# acks); routing here is the honest middle ground — visible owner + mechanism.
_ALERT_ROUTES: Dict[str, str] = {
    "consecutive_losses": "self_heal reduce_risk (evidence-anchored, auto) + auto-halt "
                          "circuit breaker at threshold (engine); operator ack via AXIOM",
    "drawdown": "drawdown guardian / NAV auto-halt (engine, auto); operator review",
    "win_rate_drop": "operator review (no safe automated remediation)",
    "weight_instability": "self_heal soft_reset_agent_weight (auto at L3+)",
}


def _alerts_section(root: Path) -> Dict[str, Any]:
    """Active (unacknowledged) alerts from AlertManager's state file, verbatim,
    each annotated with its standing route (owner + mechanism)."""
    try:
        raw = _read_json(root / ".claude" / "alert_state.json")
        if not isinstance(raw, dict):
            return {"unacknowledged": 0, "active": []}
        active = []
        for alert in raw.get("active_alerts", []) or []:
            if isinstance(alert, dict) and not alert.get("acknowledged", False):
                a_type = alert.get("alert_type") or alert.get("type")
                active.append({
                    "type": a_type,
                    "severity": alert.get("severity"),
                    "ts": alert.get("timestamp"),
                    "message": alert.get("message"),
                    "route": _ALERT_ROUTES.get(str(a_type),
                                               "UNROUTED — escalate to operator"),
                })
        return {"unacknowledged": len(active), "active": active[:10]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _lanes_section(root: Path, now: float) -> Dict[str, Any]:
    """Per-lane halt flags + liveness signals (artifact freshness, not self-report)."""
    try:
        state = _read_json(root / ".claude" / "state.json")
        halted = True
        lanes: Dict[str, Any] = {}
        if isinstance(state, dict):
            halted = bool(state.get("halted", True))
            raw_lanes = state.get("halted_lanes")
            if isinstance(raw_lanes, dict):
                lanes = {k: bool(v) for k, v in raw_lanes.items()}

        trend_age = _age_s(root / "trained_data" / "oanda" / "account_state.json", now)
        equity_age = _age_s(root / "trained_data" / "equity" / "cycle_ledger.jsonl", now)

        hb = _read_json(root / ".claude" / "heartbeat.json")
        hb_out: Dict[str, Any] = {"fresh": False, "pid_alive": False, "writer": None,
                                  "scanner_alive": None}
        if isinstance(hb, dict):
            try:
                ts = datetime.fromisoformat(str(hb.get("ts_iso", "")).replace("Z", "+00:00"))
                hb_out["fresh"] = (datetime.now(timezone.utc) - ts).total_seconds() <= _HEARTBEAT_FRESH_S
            except ValueError:
                pass
            hb_out["pid_alive"] = _pid_alive(hb.get("pid"))
            hb_out["writer"] = hb.get("writer")
            hb_out["scanner_alive"] = hb.get("scanner_alive")

        return {
            "global_halted": halted,
            "halted_lanes": lanes,
            "trend_account_state_age_s": trend_age,
            "trend_artifact_stale": (trend_age is None) or (trend_age > _TREND_STALE_S),
            "equity_ledger_age_s": equity_age,
            "equity_artifact_stale": (equity_age is None) or (equity_age > _EQUITY_STALE_S),
            "heartbeat": hb_out,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _self_heal_section(root: Path) -> Dict[str, Any]:
    try:
        budget = _read_json(root / ".claude" / "self_heal_action_budget.json")
        if not isinstance(budget, dict):
            return {"available": False}
        actions = budget.get("actions_today")
        return {
            "available": True,
            "actions_today": len(actions) if isinstance(actions, list) else None,
            "last_action_label": budget.get("last_action_label"),
            "last_action_at": budget.get("last_action_at"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _hedge_scorecard_section(root: Path, now: float) -> Dict[str, Any]:
    """Strategy evaluation readout (AXIOM Hedge Layer Phase 4, ANALYSIS-ONLY):
    raw-vs-hedged after-cost expectancy per strategy. Reads the persisted
    ``hedge_scorecard_report.json`` artifact directly via ``_read_json``
    (stdlib json) rather than importing ``src.hedge.hedge_scorecard`` and
    computing it here — keeps this module's contract exactly as documented
    above ("stdlib-only reads, no scanner-subsystem imports"; every other
    section here is a direct artifact read, not a live recompute). An absent
    artifact (nothing has produced it yet) is reported honestly as
    unavailable, never as an error — this is reporting-path wiring only, no
    execution/order path touch."""
    path = root / "trained_data" / "hedge" / "hedge_scorecard_report.json"
    raw = _read_json(path)
    if not isinstance(raw, dict):
        return {"available": False, "reason": "hedge_scorecard_report.json not yet produced"}
    scorecards = raw.get("scorecards")
    summary: Dict[str, Any] = {}
    if isinstance(scorecards, dict):
        for strategy, card in scorecards.items():
            if not isinstance(card, dict):
                continue
            decision = card.get("decision") if isinstance(card.get("decision"), dict) else {}
            summary[str(strategy)] = {
                "n_cycles": card.get("n_cycles"),
                "verdict": decision.get("verdict"),
                "cost_unmodeled_for_venue": card.get("cost_unmodeled_for_venue"),
            }
    return {
        "available": True,
        "report_age_s": _age_s(path, now),
        "strategies_with_history": sorted(summary.keys()),
        "summary": summary,
        "runtime_allowed": raw.get("runtime_allowed"),
    }


def _overlay_section(root: Path) -> Optional[Dict[str, Any]]:
    """Durable config-overlay summary (P1). None until an overlay exists."""
    try:
        from src.scanner.automation.config_overlay import overlay_status
        return overlay_status(root)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def build_maintenance_report(
    root: Path,
    *,
    supervisor_pid: Optional[int] = None,
    code: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the full snapshot. Never raises; sections degrade independently.

    ``recommendations``: diagnostics' human-readable advice for the operator
    (informational — deliberately NOT added to needs_attention, to keep that
    list scoped to actionable backlog and avoid alarm fatigue).
    """
    now = datetime.now(timezone.utc)
    now_epoch = now.timestamp()
    report: Dict[str, Any] = {
        "ts_iso": now.isoformat(),
        "writer": "tier7_supervisor",
        "supervisor_pid": int(supervisor_pid) if supervisor_pid else os.getpid(),
        "adjustments": _adjustments_section(root),
        "alerts": _alerts_section(root),
        "lanes": _lanes_section(root, now_epoch),
        "self_heal": _self_heal_section(root),
        "recommendations": [str(r) for r in (recommendations or []) if r][:10],
        "config_overlay": _overlay_section(root),
        "hedge_scorecard": _hedge_scorecard_section(root, now_epoch),
    }
    if code:
        report["code"] = code
    # One-line operator summary: what (if anything) needs a human.
    needs: List[str] = []
    adj = report["adjustments"]
    if isinstance(adj.get("approved_unconsumed"), int) and adj["approved_unconsumed"] > 0:
        needs.append(f"{adj['approved_unconsumed']} approved config adjustment(s) no live process has consumed")
    if isinstance(adj.get("pending_proposals"), int) and adj["pending_proposals"] > 0:
        needs.append(f"{adj['pending_proposals']} adjustment proposal(s) awaiting operator approval")
    al = report["alerts"]
    if isinstance(al.get("unacknowledged"), int) and al["unacknowledged"] > 0:
        needs.append(f"{al['unacknowledged']} unacknowledged alert(s)")
    report["needs_attention"] = needs
    return report


def write_maintenance_report(root: Path, report: Dict[str, Any]) -> Path:
    """Atomic write (tmp + os.replace) of the snapshot; returns the path."""
    target = root / MAINTENANCE_REPORT_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target
