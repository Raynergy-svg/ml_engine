"""AXIOM data layer — read the bot's state files + read-only OANDA v20 views.

Every reader is fail-soft: corrupt/missing files return an explicit empty shape
(never crash, never fabricate) per the repo's JSON-safety rules. File-backed
readers need no network and reflect exactly what the bot wrote. Live readers wrap
the read-only OANDA client and degrade to ``connected: False`` when the broker is
unreachable — the frontend renders that as an honest 'not connected' state.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("axiom.data")

# dashboard/server/data_sources.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
OANDA_DIR = REPO_ROOT / "trained_data" / "oanda"
EQUITY_DIR = REPO_ROOT / "trained_data" / "equity"
CLAUDE_DIR = REPO_ROOT / ".claude"

# The validated trend lane's FX-major universe (mirrors run_oanda_trend candidates).
FX_MAJORS: List[str] = [
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD",
    "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP",
]
HEARTBEAT_FRESH_S = 30.0   # TUI scanner heartbeat ticks ~10s; >30s old => not alive
LANE_FRESH_S = 7200.0      # trend lane rebalances hourly; account snapshot <2h => lane live

# Tier 7 autonomous-loop snapshot — contracted at docs/dashboard-data-contract.md
# (written by src/scanner/automation/tier7_state.py:write_tier7_state). Read-only
# display: the loop's incident->propose->gate->soak->promote->close + self-heal state.
# Absent => panel renders honest not-connected (never fabricates loop activity).
TIER7_STATE_PATH = CLAUDE_DIR / "tier7_state.json"
TIER7_FRESH_S = 900.0      # snapshot-write freshness; >15m old => snapshot considered stale


# --------------------------------------------------------------------------- #
# Generic fail-soft JSON / JSONL readers
# --------------------------------------------------------------------------- #
def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _iter_jsonl(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def _pip_size(instrument: str) -> float:
    # JPY-quoted pairs price to 0.01 per pip; all other FX majors to 0.0001.
    return 0.01 if instrument.endswith("_JPY") else 0.0001


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _fill_kind(t: Dict[str, Any]) -> Optional[str]:
    if t.get("type") != "ORDER_FILL":
        return None
    opened = bool(t.get("tradeOpened"))
    reduced = bool(t.get("tradeReduced"))
    closed = bool(t.get("tradesClosed"))
    if opened and (reduced or closed):
        return "OPEN+REDUCE"
    if opened:
        return "OPEN"
    if closed:
        return "CLOSE"
    if reduced:
        return "REDUCE"
    return "FILL"


def _trade_refs(t: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    if t.get("tradeID"):
        refs.append(str(t.get("tradeID")))
    opened = t.get("tradeOpened")
    if isinstance(opened, dict) and opened.get("tradeID"):
        refs.append(str(opened.get("tradeID")))
    reduced = t.get("tradeReduced")
    if isinstance(reduced, dict) and reduced.get("tradeID"):
        refs.append(str(reduced.get("tradeID")))
    closed = t.get("tradesClosed")
    if isinstance(closed, list):
        for item in closed:
            if isinstance(item, dict) and item.get("tradeID"):
                refs.append(str(item.get("tradeID")))
    return refs


def _transaction_status(t: Dict[str, Any], fills_by_order: Dict[str, Dict[str, Any]],
                        cancelled_order_ids: set[str]) -> str:
    tx_type = str(t.get("type") or "UNKNOWN")
    tx_id = str(t.get("id") or "")
    if t.get("rejectReason") or tx_type.endswith("_REJECT"):
        return "REJECTED"
    if tx_type == "ORDER_FILL":
        return "FILLED"
    if tx_type == "ORDER_CANCEL":
        return "CANCELLED"
    if tx_type == "DAILY_FINANCING":
        return "POSTED"
    if tx_id in fills_by_order:
        return "FILLED"
    if tx_id in cancelled_order_ids:
        return "CANCELLED"
    if tx_type.endswith("_ORDER"):
        return "ACTIVE"
    return "RECORDED"


# --------------------------------------------------------------------------- #
# File-backed readers (no network) — what the bot itself wrote
# --------------------------------------------------------------------------- #
def read_account() -> Dict[str, Any]:
    """Account summary + positions the live loop last snapshotted to disk."""
    data = _read_json(OANDA_DIR / "account_state.json", {})
    peak = _read_json(OANDA_DIR / "peak_nav.json", {}).get("peak_nav")
    nav = float(data.get("nav") or 0.0)
    out: Dict[str, Any] = {
        "connected": bool(data),
        "source": "account_state.json",
        "account_id": data.get("account_id"),
        "currency": data.get("currency"),
        "nav": nav,
        "unrealized_pl": float(data.get("unrealized_pl") or 0.0),
        "realized_pl": float(data.get("realized_pl") or 0.0),
        "margin_used": float(data.get("margin_used") or 0.0),
        "margin_available": float(data.get("margin_available") or 0.0),
        "open_trade_count": int(data.get("open_trade_count") or 0),
        "positions": data.get("positions") or [],
        "peak_nav": float(peak) if peak is not None else None,
    }
    if out["peak_nav"] and out["peak_nav"] > 0 and nav > 0:
        out["drawdown_pct"] = max(0.0, (out["peak_nav"] - nav) / out["peak_nav"])
    else:
        out["drawdown_pct"] = None
    # Freshness: the trend lane rewrites this snapshot each cycle. A stopped lane
    # leaves a stale-but-parseable file — surface its age + a stale flag so the UI
    # never presents an old snapshot as live truth (honesty: D-C1).
    try:
        out["snapshot_age_s"] = round(time.time() - (OANDA_DIR / "account_state.json").stat().st_mtime, 1)
    except OSError:
        out["snapshot_age_s"] = None
    out["stale"] = out["snapshot_age_s"] is None or out["snapshot_age_s"] > LANE_FRESH_S
    return out


def _last_fill_time() -> Optional[str]:
    """Cheap tail-read of the ledger for the most recent ORDER_FILL time.

    Scans only an ORDER_FILL (not bracket-order/financing/cancel rows, which also
    carry a ``time``) so the value matches the field name (D-H2). Reads a 64 KB tail
    — large enough to clear a run of non-fill rows; returns None (honest "no recent
    fill") if no fill is in the window rather than a misleading non-fill timestamp.
    """
    path = OANDA_DIR / "transactions.jsonl"
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(max(0, size - 65536))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("type") == "ORDER_FILL" and obj.get("time"):
            return obj.get("time")
    return None


def read_status(now_iso: Optional[str] = None) -> Dict[str, Any]:
    """Runtime status from state.json + the lane's own freshness signals.

    The OANDA trend lane and the TUI scanner are SEPARATE processes. The lane's
    liveness is the freshness of the account snapshot it rewrites each cycle (NOT
    .claude/heartbeat.json, which belongs to the TUI scanner). We surface both,
    honestly labeled, and base 'running' on the lane.
    """
    from datetime import datetime, timezone

    state = _read_json(CLAUDE_DIR / "state.json", {})
    hb = _read_json(CLAUDE_DIR / "heartbeat.json", {})
    now = datetime.now(timezone.utc)

    # TUI scanner heartbeat (separate process)
    hb_age = None
    hb_alive = False
    ts_iso = hb.get("ts_iso")
    if ts_iso:
        try:
            hb_age = (now - datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))).total_seconds()
            hb_alive = hb_age <= HEARTBEAT_FRESH_S
        except (ValueError, TypeError):
            pass

    # Account snapshot age — kept as an INFORMATIONAL signal only.
    acct_age = None
    try:
        acct_age = now.timestamp() - (OANDA_DIR / "account_state.json").stat().st_mtime
    except OSError:
        pass

    # Lane liveness: defer to the FIXED live-lane oracle (running_status.live_lane_running)
    # — the SAME source /api/system_health uses — so status / system_health / strategy
    # can never disagree on "is the bot live?" (D-H1). Snapshot-age is only a fallback
    # when the oracle module can't be loaded.
    running_source = "oracle"
    mod = _load_running_status()
    if mod is not None and hasattr(mod, "live_lane_running"):
        try:
            lane_running = bool(mod.live_lane_running().get("running"))
        except Exception as exc:  # noqa: BLE001 — oracle optional; degrade to snapshot age
            logger.warning("read_status oracle failed, falling back to snapshot age: %s", exc)
            lane_running = acct_age is not None and acct_age <= LANE_FRESH_S
            running_source = "snapshot_age_fallback"
    else:
        lane_running = acct_age is not None and acct_age <= LANE_FRESH_S
        running_source = "snapshot_age_fallback"

    return {
        "halted": bool(state.get("halted", True)),  # fail-closed default: assume halted
        "mode": state.get("mode"),
        "status": state.get("status"),
        "scan_cycle_count": state.get("scan_cycle_count"),
        "environment": "practice",  # immutable Hard NO
        "running": bool(lane_running),
        "lane_running": bool(lane_running),
        "running_source": running_source,
        "account_snapshot_age_s": round(acct_age, 1) if acct_age is not None else None,
        "last_fill_time": _last_fill_time(),
        "scanner_heartbeat_alive": bool(hb_alive),
        "scanner_heartbeat_age_s": round(hb_age, 1) if hb_age is not None else None,
        "scanner_pid": hb.get("pid"),
        "last_updated": state.get("last_updated"),
    }


def read_trades(limit: int = 100) -> Dict[str, Any]:
    """Parse the local OANDA transaction ledger, not just fills.

    The frontend route name remains ``/api/trades`` for compatibility, but this
    is a transaction ledger: order submissions, fills, bracket orders,
    cancellations, financing, and rejects all need to be visible so operators
    can distinguish "submitted but active/cancelled" from "filled".
    """
    txns = list(_iter_jsonl(OANDA_DIR / "transactions.jsonl"))
    fills_by_order: Dict[str, Dict[str, Any]] = {}
    cancelled_order_ids: set[str] = set()
    instrument_by_trade_id: Dict[str, str] = {}
    instrument_by_order_id: Dict[str, str] = {}
    for t in txns:
        tx_id = str(t.get("id") or "")
        instrument = t.get("instrument")
        if tx_id and instrument:
            instrument_by_order_id[tx_id] = str(instrument)
        if t.get("type") == "ORDER_FILL" and t.get("orderID"):
            fills_by_order[str(t.get("orderID"))] = t
            if instrument:
                for trade_id in _trade_refs(t):
                    instrument_by_trade_id[trade_id] = str(instrument)
        if t.get("type") == "ORDER_CANCEL" and t.get("orderID"):
            cancelled_order_ids.add(str(t.get("orderID")))

    rows: List[Dict[str, Any]] = []
    for t in txns:
        units = _to_float(t.get("units")) or 0.0
        tx_type = str(t.get("type") or "UNKNOWN")
        tx_id = str(t.get("id") or "")
        order_id = _str_or_none(t.get("orderID") if tx_type in {"ORDER_FILL", "ORDER_CANCEL"} else t.get("id"))
        linked_fill = fills_by_order.get(tx_id)
        instrument = t.get("instrument")
        if not instrument and t.get("tradeID"):
            instrument = instrument_by_trade_id.get(str(t.get("tradeID")))
        if not instrument and t.get("orderID"):
            instrument = instrument_by_order_id.get(str(t.get("orderID")))
        price = _to_float(t.get("price"))
        pl = _to_float(t.get("pl")) or 0.0
        client_extensions = t.get("clientExtensions") if isinstance(t.get("clientExtensions"), dict) else {}
        linked_price = _to_float(linked_fill.get("price")) if linked_fill else None
        linked_pl = _to_float(linked_fill.get("pl")) if linked_fill else None
        rows.append({
            "id": t.get("id"),
            "time": t.get("time"),
            "type": tx_type,
            "status": _transaction_status(t, fills_by_order, cancelled_order_ids),
            "instrument": instrument,
            "units": units,
            "side": "BUY" if units >= 0 else "SELL",
            "price": price,
            "pl": pl,
            "financing": _to_float(t.get("financing")) or 0.0,
            "half_spread_cost": _to_float(t.get("halfSpreadCost")) or 0.0,
            "reason": t.get("reason"),
            "balance": _to_float(t.get("accountBalance")),
            "tag": client_extensions.get("tag"),
            "order_id": order_id,
            "linked_fill_id": str(linked_fill.get("id")) if linked_fill else (tx_id if tx_type == "ORDER_FILL" else None),
            "linked_fill_price": linked_price,
            "linked_fill_pl": linked_pl,
            "fill_kind": _fill_kind(t),
            "trade_ids": _trade_refs(t),
            "reject_reason": t.get("rejectReason"),
        })
    rows.reverse()  # most recent first
    fill_count = sum(1 for t in txns if t.get("type") == "ORDER_FILL")
    order_count = sum(1 for t in txns if str(t.get("type") or "").endswith("_ORDER"))
    return {
        "connected": bool(rows),
        "source": "transactions.jsonl",
        "count": len(rows),
        "fill_count": fill_count,
        "order_count": order_count,
        "trades": rows[: int(limit)],
    }


def read_equity() -> Dict[str, Any]:
    """Equity curve from the per-fill accountBalance recorded in the ledger.

    ``ledger_realized_pl`` is the P&L summed over THIS local ledger only — it is
    NOT the account's since-inception realized P&L (the ledger backfills only
    recent history). The header's "Total Realized" uses the broker's own figure
    from account_state.json; these are deliberately different sources.
    """
    points: List[Dict[str, Any]] = []
    realized = 0.0
    for t in _iter_jsonl(OANDA_DIR / "transactions.jsonl"):  # single pass
        if t.get("type") == "ORDER_FILL":
            try:
                realized += float(t.get("pl") or 0)
            except (ValueError, TypeError):
                pass
        bal, ts = t.get("accountBalance"), t.get("time")
        if bal is None or ts is None:
            continue
        try:
            points.append({"time": ts, "balance": float(bal)})
        except (ValueError, TypeError):
            continue
    return {"connected": bool(points), "source": "transactions.jsonl",
            "points": points, "n": len(points), "ledger_realized_pl": round(realized, 4)}


def read_sentiment() -> Dict[str, Any]:
    """Order/position-book sentiment snapshot plus live order_flow wiring metadata."""
    data = _read_json(OANDA_DIR / "sentiment_snapshot.json", {})
    books = data.get("books", {})
    features: Dict[str, Any] = {}
    try:
        from src.data.order_book import extract_position_features
        for inst, buckets in books.items():
            if not isinstance(buckets, list):
                continue
            prices = []
            for bucket in buckets:
                try:
                    prices.append(float(bucket.get("price")))
                except (AttributeError, TypeError, ValueError):
                    continue
            current_price = prices[len(prices) // 2] if prices else 1.0
            features[inst] = extract_position_features(buckets, current_price)
    except Exception as exc:  # noqa: BLE001 — sentiment panel must degrade honestly
        logger.warning("sentiment feature extraction failed: %s", exc)
        features = {}

    order_flow_enabled = True
    order_flow_weight = 0.95
    try:
        from src.scanner.config import ScannerConfig
        from src.scanner.agents._team import ScannerAgentTeam
        order_flow_enabled = bool(getattr(ScannerConfig(), "enable_order_flow_agent", True))
        order_flow_weight = float(ScannerAgentTeam._BASE_WEIGHTS.get("order_flow", order_flow_weight))
    except Exception:
        pass

    return {
        "connected": bool(data.get("books")),
        "wired_into_strategy": order_flow_enabled,
        "strategy_agent": "order_flow",
        "agent_weight": order_flow_weight,
        "note": (
            "Position-book crowding feeds the scanner order_flow agent."
            if order_flow_enabled
            else data.get("note", "DATA-ONLY; order_flow agent disabled.")
        ),
        "books": books,
        "features": features,
        "source": "sentiment_snapshot.json",
    }


def read_equity_sleeve() -> Dict[str, Any]:
    """The (currently dormant) equity-harvester sleeve target weights, if present."""
    data = _read_json(EQUITY_DIR / "rebalance_state.json", {})
    plan = data.get("active_plan") or {}
    return {
        "connected": bool(plan),
        "dormant": True,
        "asof": plan.get("asof"),
        "rebalance_id": plan.get("rebalance_id"),
        "target_weights": plan.get("target_weights", {}),
        "actual_weights": data.get("current_actual_weights", {}),
        "source": "trained_data/equity/rebalance_state.json",
    }


def read_tier7() -> Dict[str, Any]:
    """Tier 7 autonomous-loop status — fail-soft passthrough of the bot's snapshot.

    Reads ``.claude/tier7_state.json`` (written by
    ``src/scanner/automation/tier7_state.py:write_tier7_state``; contract: docs/
    dashboard-data-contract.md). Honest by design: it trusts the bot's own
    ``running`` / ``running_reason`` (heartbeat-fresh AND pid-alive), adds a separate
    snapshot-freshness signal (is the file itself being refreshed?), and returns
    ``connected: False`` when the file is absent — it NEVER fabricates loop activity.
    """
    from datetime import datetime, timezone

    rel = str(TIER7_STATE_PATH.relative_to(REPO_ROOT))
    if not TIER7_STATE_PATH.exists():
        return {"connected": False, "source": rel, "pending_contract": True}
    data = _read_json(TIER7_STATE_PATH, None)
    if not isinstance(data, dict):
        return {"connected": False, "source": rel, "error": "unreadable"}

    # Snapshot freshness: is the bot still WRITING this file? (distinct from whether
    # the loop is running — a fresh snapshot can honestly report running:false).
    snap_age = None
    gen = data.get("generated_at")
    if gen:
        try:
            snap_age = (datetime.now(timezone.utc)
                        - datetime.fromisoformat(str(gen).replace("Z", "+00:00"))).total_seconds()
        except (ValueError, TypeError):
            pass
    if snap_age is None:  # fall back to file mtime
        try:
            snap_age = datetime.now(timezone.utc).timestamp() - TIER7_STATE_PATH.stat().st_mtime
        except OSError:
            pass

    return {
        "connected": True,
        "source": rel,
        "snapshot_age_s": round(snap_age, 1) if snap_age is not None else None,
        "snapshot_stale": (snap_age is not None and snap_age > TIER7_FRESH_S),
        # honest, bot-derived liveness (do NOT recompute — trust the bot's pid+heartbeat check)
        "running": bool(data.get("running")),
        "running_reason": data.get("running_reason"),
        "halted": data.get("halted"),
        "mode": data.get("mode"),
        "status": data.get("status"),
        "goal": data.get("goal"),
        "improvement_focus": data.get("improvement_focus"),
        "scan_cycle_count": data.get("scan_cycle_count"),
        "current_action": data.get("current_action"),
        "autonomy_level": data.get("autonomy_level"),
        "max_autonomy": data.get("max_autonomy"),
        "bounded": data.get("bounded"),
        "runtime": data.get("runtime"),
        "last_cycle": data.get("last_cycle") or {},
        "self_heal": data.get("self_heal") or {},
        "meta_last_event": data.get("meta_last_event") or {},
        "generated_at": gen,
        "note": data.get("note"),
    }


LOOP_DIR = CLAUDE_DIR / "loop"
_running_status_mod = None


def _load_running_status():
    """Import the bot's live-lane oracle (.claude/loop/running_status.py) by path."""
    global _running_status_mod
    if _running_status_mod is not None:
        return _running_status_mod
    import importlib.util
    path = LOOP_DIR / "running_status.py"
    try:
        spec = importlib.util.spec_from_file_location("axiom_running_status", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _running_status_mod = mod
        return mod
    except Exception as exc:  # noqa: BLE001 — oracle optional; health degrades honestly
        logger.warning("running_status oracle unavailable: %s", exc)
        return None


def read_health() -> Dict[str, Any]:
    """System health: live-lane oracle + verify-gate Hard-NO checks + active alerts.

    Mirrors the bot's own honest signals — the FIXED live-lane oracle
    (`running_status.live_lane_running`), the recorded `verify_gate` verdict, and the
    AlertManager state. Fail-soft: any missing source degrades to a labeled empty,
    never a fabricated 'all good'.
    """
    from datetime import datetime, timezone

    # 1) lanes — the authoritative running:yes/no oracle (live lane vs dormant harvester)
    lanes: Dict[str, Any] = {"available": False}
    mod = _load_running_status()
    if mod is not None and hasattr(mod, "live_lane_running"):
        try:
            lanes = {"available": True, **mod.live_lane_running()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("live_lane_running failed: %s", exc)
            lanes = {"available": False, "error": type(exc).__name__}

    # 2) gates — recorded verify_gate Hard-NO checks (verdict.json) + freshness
    verdict = _read_json(LOOP_DIR / "verdict.json", {})
    checks = verdict.get("checks") or []
    gate_age = None
    try:
        gate_age = datetime.now(timezone.utc).timestamp() - (LOOP_DIR / "verdict.json").stat().st_mtime
    except OSError:
        pass
    hard_no = [c for c in checks if c.get("hard_no")]
    all_ok = bool(checks) and all(c.get("ok") for c in checks)
    gates = {
        "available": bool(checks),
        "all_ok": all_ok,
        "status": "GREEN" if all_ok else ("RED" if checks else "UNKNOWN"),
        "checks": checks,
        "hard_no_count": len(hard_no),
        "verdict_age_s": round(gate_age, 1) if gate_age is not None else None,
    }

    # 3) alerts — AlertManager active alerts (consecutive losses, drawdown, win-rate)
    alert_state = _read_json(CLAUDE_DIR / "alert_state.json", {})
    active = alert_state.get("active_alerts") or []
    alerts = {
        "available": bool(alert_state),
        "active": active,
        "count": len(active),
        "max_severity": _max_severity(active),
        "last_updated": alert_state.get("last_updated"),
    }
    return {"lanes": lanes, "gates": gates, "alerts": alerts}


def _max_severity(active: List[Dict[str, Any]]) -> Optional[str]:
    order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2, "ALARM": 3}
    best = None
    for a in active:
        sev = a.get("severity")
        if sev in order and (best is None or order[sev] > order[best]):
            best = sev
    return best


# --------------------------------------------------------------------------- #
# Live read-only OANDA views (degrade to connected:False, never fabricate)
# --------------------------------------------------------------------------- #
def live_prices(client, instruments: List[str]) -> Dict[str, Any]:
    """Live bid/ask/mid + spread (pips) for the given instruments."""
    if client is None:
        return {"connected": False, "prices": {}}
    try:
        resp = client.get_pricing(instruments=",".join(instruments)) or {}
    except Exception as exc:  # noqa: BLE001 — broker unreachable / stale token
        logger.warning("live_prices failed: %s", exc)
        return {"connected": False, "prices": {}, "error": type(exc).__name__}
    prices: Dict[str, Any] = {}
    for p in resp.get("prices", []) or []:
        inst = p.get("instrument")
        if not inst:
            continue
        try:
            bid = float((p.get("bids") or [{}])[0].get("price"))
            ask = float((p.get("asks") or [{}])[0].get("price"))
        except (ValueError, TypeError, IndexError, AttributeError):
            continue
        mid = (bid + ask) / 2.0
        # NEVER fabricate tradeability: use the broker's explicit flag if present,
        # else derive from status, else None (honest unknown). Defaulting to True
        # would paint a halted/closed market as tradeable (D-C2).
        if "tradeable" in p:
            tradeable: Optional[bool] = bool(p["tradeable"])
        elif p.get("status"):
            tradeable = (p.get("status") == "tradeable")
        else:
            tradeable = None
        prices[inst] = {
            "bid": bid, "ask": ask, "mid": mid,
            "spread_pips": round((ask - bid) / _pip_size(inst), 2),
            "time": p.get("time"),
            "status": p.get("status"),
            "tradeable": tradeable,
        }
    return {"connected": bool(prices), "prices": prices}


def live_candles(client, instrument: str, granularity: str = "D",
                 count: int = 300, sma_window: int = 100) -> Dict[str, Any]:
    """OHLC candles + SMA overlay + current long-or-flat trend state for one instrument.

    The trend signal is the *validated* rule the bot actually trades:
    price[t-1] > SMA(window)[t-1] => long, else flat (strictly causal, shift(1)).
    """
    if client is None:
        return {"connected": False, "instrument": instrument, "candles": [],
                "sma": [], "signal": None}
    try:
        resp = client.get_candles(instrument, granularity=granularity, count=count, price="M") or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_candles failed for %s: %s", instrument, exc)
        return {"connected": False, "instrument": instrument, "candles": [],
                "sma": [], "signal": None, "error": type(exc).__name__}

    import pandas as pd
    from datetime import datetime, timezone

    candles, closes, times = [], [], []
    for c in resp.get("candles", []) or []:
        if not c.get("complete", False):
            continue
        mid = c.get("mid") or {}
        try:
            o, h, l, cl = float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])
            ts = int(datetime.fromisoformat(c["time"].replace("Z", "+00:00"))
                     .replace(tzinfo=timezone.utc).timestamp())
        except (KeyError, ValueError, TypeError):
            continue
        candles.append({"time": ts, "open": o, "high": h, "low": l, "close": cl})
        closes.append(cl)
        times.append(ts)

    sma_series: List[Dict[str, Any]] = []
    signal: Optional[Dict[str, Any]] = None
    if closes:
        s = pd.Series(closes)
        sma = s.rolling(window=sma_window, min_periods=sma_window).mean()
        for ts, v in zip(times, sma.tolist()):
            if v == v:  # not NaN
                sma_series.append({"time": ts, "value": round(v, 6)})
        # current state uses the last CLOSED bar vs its SMA (shift(1) causal at runtime).
        last_price = closes[-1]
        last_sma = sma.iloc[-1] if len(sma) else float("nan")
        if last_sma == last_sma:
            on = last_price > last_sma
            signal = {
                "on": bool(on),
                "state": "LONG" if on else "FLAT",
                "price": round(last_price, 6),
                "sma": round(float(last_sma), 6),
                "sma_window": sma_window,
                "distance_pct": round((last_price - float(last_sma)) / float(last_sma) * 100, 3),
            }
    return {"connected": True, "instrument": instrument, "granularity": granularity,
            "candles": candles, "sma": sma_series, "signal": signal}


def live_trend(client, instruments: Optional[List[str]] = None,
               sma_window: int = 100, granularity: str = "D") -> Dict[str, Any]:
    """Recompute the trend targets across the universe (the bot's actual signal).

    Reuses the pure helpers from ``src.equity.oanda_trend`` — same math the live
    cycle uses — but never calls the cycle (which would place orders).
    """
    universe = instruments or FX_MAJORS
    if client is None:
        return {"connected": False, "universe": universe, "targets": {}, "on": [], "flat": []}
    from src.equity.oanda_trend import candles_to_close_panel, trend_targets

    # Per-instrument fetch: a transient 401/rate-limit on one pair must not blank
    # the whole panel (the dashboard bursts many reads at load). Skip failures.
    candles: Dict[str, dict] = {}
    missing: List[str] = []
    for inst in universe:
        try:
            candles[inst] = client.get_candles(inst, granularity=granularity, count=300, price="M")
        except Exception as exc:  # noqa: BLE001
            logger.warning("live_trend candle fetch failed for %s: %s", inst, exc)
            missing.append(inst)
    if not candles:
        return {"connected": False, "universe": universe, "targets": {}, "on": [],
                "flat": [], "missing": missing}
    panel = candles_to_close_panel(candles)
    targets = trend_targets(panel, sma_window=sma_window) if not panel.empty else {}
    on = sorted([k for k, v in targets.items() if v > 0])
    flat = sorted([k for k in targets if k not in on])
    return {"connected": bool(targets), "universe": universe, "sma_window": sma_window,
            "granularity": granularity, "targets": targets, "on": on, "flat": flat,
            "missing": missing, "partial": bool(missing)}
