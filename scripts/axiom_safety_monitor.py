#!/usr/bin/env python3
"""AXIOM safety/rules monitor.

Read-only by default. With ``--halt-on-critical`` it may only call the bounded
control halt endpoint after a high-severity safety invariant fails. It never
places orders, starts loops, changes leverage, or loosens gates.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "trained_data" / "axiom" / "safety_monitor.jsonl"
DEFAULT_API = "http://127.0.0.1:8888"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_json(api: str, path: str, *, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{api}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(api: str, path: str, body: dict[str, Any], confirm: str, *, timeout: float = 5.0) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{api}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "x-axiom-confirm": confirm},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_recent_brackets(limit: int = 120) -> dict[str, Any]:
    path = REPO_ROOT / "trained_data" / "oanda" / "transactions.jsonl"
    if not path.exists():
        return {"ok": False, "reason": "transactions.jsonl missing"}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue

    by_trade: dict[str, set[str]] = {}
    recent_opens: list[dict[str, Any]] = []
    for row in rows:
        typ = row.get("type")
        trade_id = str(row.get("tradeID") or "")
        if typ in {"STOP_LOSS_ORDER", "TAKE_PROFIT_ORDER"} and trade_id:
            by_trade.setdefault(trade_id, set()).add(str(typ))
        if typ == "ORDER_FILL" and isinstance(row.get("tradeOpened"), dict):
            opened_id = str(row["tradeOpened"].get("tradeID") or "")
            if opened_id:
                recent_opens.append({
                    "id": row.get("id"),
                    "trade_id": opened_id,
                    "instrument": row.get("instrument"),
                    "time": row.get("time"),
                })

    naked = []
    for opened in recent_opens[-8:]:
        legs = by_trade.get(opened["trade_id"], set())
        if not {"STOP_LOSS_ORDER", "TAKE_PROFIT_ORDER"}.issubset(legs):
            naked.append({**opened, "legs": sorted(legs)})
    return {"ok": not naked, "recent_open_count": len(recent_opens), "naked_recent_opens": naked}


def run_once(api: str, *, halt_on_critical: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    critical: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any, *, severity: str = "soft") -> None:
        item = {"name": name, "ok": bool(ok), "severity": severity, "detail": detail}
        checks.append(item)
        if not ok and severity == "critical":
            critical.append(item)

    try:
        health = _fetch_json(api, "/api/health")
        control = _fetch_json(api, "/api/control/state")
        tier7 = _fetch_json(api, "/api/tier7")
        status = _fetch_json(api, "/api/status")
        account = _fetch_json(api, "/api/account")
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        result = {
            "ts": _now(),
            "ok": False,
            "fatal": f"{type(exc).__name__}: {exc}",
            "halt_attempted": False,
        }
        return result

    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    loops = control.get("loops") or {}
    leverage = control.get("gross_leverage")
    leverage_cap = _safe_float(control.get("leverage_cap"), 15.0)
    _last_cycle = tier7.get("last_cycle")
    tier7_pid = _last_cycle.get("pid") if isinstance(_last_cycle, dict) else None
    tier7_control_pids = (loops.get("tier7") or {}).get("pids") or []

    add("api_up_practice", health.get("ok") and health.get("environment") == "practice", health, severity="critical")
    add("control_enabled", health.get("control_enabled") is True, {"control_enabled": health.get("control_enabled")}, severity="critical")
    add("control_env_practice", control.get("environment") == "practice", {"environment": control.get("environment")}, severity="critical")
    _leverage_f = None if leverage is None else _safe_float(leverage, float("nan"))
    add(
        "leverage_cap",
        leverage is None or (_leverage_f == _leverage_f and 0.0 <= _leverage_f <= min(leverage_cap, 15.0)),
        {"gross_leverage": leverage, "leverage_cap": leverage_cap},
        severity="critical",
    )
    add("tier7_loop_running", (loops.get("tier7") or {}).get("running") is True, loops.get("tier7"))
    add("trend_loop_running", (loops.get("trend") or {}).get("running") is True, loops.get("trend"))
    add(
        "tier7_fresh_live",
        tier7.get("connected") and tier7.get("running") and not tier7.get("snapshot_stale"),
        {k: tier7.get(k) for k in ("running", "running_reason", "snapshot_stale", "snapshot_age_s", "autonomy_level", "bounded")},
        severity="critical",
    )
    add(
        "tier7_pid_matches_control",
        tier7_pid in tier7_control_pids,
        {"tier7_pid": tier7_pid, "control_pids": tier7_control_pids},
        severity="critical",
    )
    add(
        "status_practice_running",
        status.get("environment") == "practice" and status.get("running") is True,
        {k: status.get(k) for k in ("environment", "running", "lane_running", "running_source", "halted")},
    )

    positions = account.get("positions") or []
    naked_positions = []
    for pos in positions:
        try:
            units = abs(float(pos.get("net_units") or 0.0))
        except (TypeError, ValueError):
            units = 0.0
        if units > 0 and (pos.get("stop_loss") is None or pos.get("take_profit") is None):
            naked_positions.append({
                "instrument": pos.get("instrument"),
                "net_units": pos.get("net_units"),
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit"),
            })
    add(
        "no_open_position_without_tp_sl",
        not naked_positions,
        {
            "open_trade_count": account.get("open_trade_count"),
            "naked": naked_positions,
            "positions": [
                {k: pos.get(k) for k in ("instrument", "net_units", "stop_loss", "take_profit", "unrealized_pl")}
                for pos in positions
            ],
        },
        severity="critical",
    )

    bracket_check = _check_recent_brackets()
    # Ledger tails are supporting evidence only. The authoritative live safety
    # invariant is the current account snapshot above; old closed fills may predate
    # repair or have dependent orders outside the inspected window.
    add("recent_fills_have_sl_tp", bracket_check.get("ok") is True, bracket_check, severity="soft")

    halt_result: dict[str, Any] | None = None
    if critical and halt_on_critical:
        try:
            halt_result = _post_json(api, "/api/control/halt", {"params": {}}, "halt")
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            halt_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    result = {
        "ts": _now(),
        "ok": not critical,
        "critical_count": len(critical),
        "checks": checks,
        "halt_attempted": bool(critical and halt_on_critical),
        "halt_result": halt_result,
    }
    return result


def _append_log(result: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, sort_keys=True) + "\n")
        fh.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--interval", type=float, default=0.0, help="seconds between checks; 0 means run once")
    ap.add_argument("--halt-on-critical", action="store_true")
    args = ap.parse_args(argv)

    while True:
        try:
            result = run_once(args.api, halt_on_critical=args.halt_on_critical)
        except Exception as exc:  # noqa: BLE001 - watchdog must never die silently
            result = {"ts": _now(), "ok": False, "fatal": f"{type(exc).__name__}: {exc}",
                      "halt_attempted": False}
        _append_log(result)
        print(json.dumps(result, indent=2), flush=True)
        if not args.interval:
            return 0 if result.get("ok") else 2
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
