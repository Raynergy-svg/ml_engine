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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("axiom.data")

# dashboard/server/data_sources.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
OANDA_DIR = REPO_ROOT / "trained_data" / "oanda"
EQUITY_DIR = REPO_ROOT / "trained_data" / "equity"
CRYPTO_DIR = REPO_ROOT / "trained_data" / "crypto"
EVIDENCE_DIR = REPO_ROOT / "trained_data" / "evidence"
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

ACTIVITY_LOG_FEEDS = [
    {
        "id": "trend",
        "title": "OANDA trend loop",
        "path": Path("trained_data/axiom/trend_loop.out"),
    },
    {
        "id": "tier7",
        "title": "Tier 7 supervisor",
        "path": Path("trained_data/axiom/tier7_loop.out"),
    },
    {
        "id": "equity",
        "title": "Equity harvester",
        "path": Path("logs/equity_harvester_loop.out"),
    },
    {
        "id": "safety",
        "title": "Safety monitor",
        "path": Path("trained_data/axiom/safety_monitor.out"),
    },
]


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
    # JPY-quoted pairs price to 0.01 per pip; metals (XAU/XAG) are quoted directly
    # in USD so "spread" is the raw price difference, not a pip-scaled figure.
    if instrument.startswith(("XAU", "XAG")):
        return 1.0
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


def _utc_iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _tail_text_lines(path: Path, limit: int) -> List[str]:
    """Return a bounded tail from a fixed local log file.

    These logs are small enough for simple line reads today, but the byte cap keeps
    this read-only dashboard endpoint predictable even if a loop log grows large.
    """
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64_000))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()][-limit:]


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


def read_activity(line_limit: int = 8) -> Dict[str, Any]:
    """Concise live operations feed for the AXIOM Activity tab.

    Combines the structured BackgroundActivity registry with short tails from the
    fixed loop logs. No caller-supplied path is accepted here; this is read-only
    visibility into known bot processes, not a file browser.
    """
    try:
        from src.scanner.automation.background_activity import get_background_activity_tracker

        background = get_background_activity_tracker().get_snapshot(limit=12)
    except Exception as exc:  # noqa: BLE001 - dashboard visibility must degrade, not crash
        background = {
            "updated_at": None,
            "active_count": 0,
            "history_count": 0,
            "active": [],
            "history": [],
            "error": str(exc),
        }

    now = time.time()
    feeds: List[Dict[str, Any]] = []
    for feed in ACTIVITY_LOG_FEEDS:
        rel_path = feed["path"]
        abs_path = REPO_ROOT / rel_path
        lines = _tail_text_lines(abs_path, line_limit)
        try:
            stat = abs_path.stat()
            updated_at = _utc_iso_from_ts(stat.st_mtime)
            age_s: Optional[float] = round(now - stat.st_mtime, 1)
            size_bytes: Optional[int] = stat.st_size
        except OSError:
            updated_at = None
            age_s = None
            size_bytes = None

        feeds.append({
            "id": feed["id"],
            "title": feed["title"],
            "path": str(rel_path),
            "exists": abs_path.exists(),
            "updated_at": updated_at,
            "age_s": age_s,
            "size_bytes": size_bytes,
            "lines": lines,
        })

    return {
        "updated_at": _utc_iso_from_ts(now),
        "background": background,
        "log_feeds": feeds,
    }


# --------------------------------------------------------------------------- #
# File-backed readers (no network) — what the bot itself wrote
# --------------------------------------------------------------------------- #
def _position_entry_contexts() -> Dict[str, Dict[str, Any]]:
    """Weighted-average entry price + open time for each instrument's CURRENT lot.

    Replays ORDER_FILL rows chronologically per instrument. Only fills that ADD to
    the position in its current direction contribute to the cost basis (a reduction
    doesn't move the average entry price); a lot resets whenever net units cross
    through / start from flat. Pure derivation from the real ledger — never fabricated;
    an instrument with no fill history (or a lot we can't reconstruct) yields None/None
    and the frontend must render an honest "—", not a guess.
    """
    net: Dict[str, float] = {}
    lot_units: Dict[str, float] = {}
    lot_notional: Dict[str, float] = {}
    lot_opened_at: Dict[str, Optional[str]] = {}
    for t in _iter_jsonl(OANDA_DIR / "transactions.jsonl"):
        if t.get("type") != "ORDER_FILL":
            continue
        inst = t.get("instrument")
        if not inst:
            continue
        try:
            units = float(t.get("units") or 0)
            price = float(t.get("price") or 0)
        except (ValueError, TypeError):
            continue
        if units == 0 or price <= 0:
            continue
        prev = net.get(inst, 0.0)
        new_net = prev + units
        crossed_flat = prev == 0.0 or (prev > 0) != (new_net > 0)
        if crossed_flat:
            lot_units[inst], lot_notional[inst], lot_opened_at[inst] = 0.0, 0.0, None
        # a fill "adds" to the lot when it moves net in the SAME direction as the
        # resulting position (i.e. |new_net| > |prev| in that direction)
        adding = new_net != 0 and ((units > 0) == (new_net > 0)) and abs(new_net) >= abs(prev)
        if adding:
            lot_units[inst] = lot_units.get(inst, 0.0) + units
            lot_notional[inst] = lot_notional.get(inst, 0.0) + units * price
            if lot_opened_at.get(inst) is None:
                lot_opened_at[inst] = t.get("time")
        net[inst] = new_net
    out: Dict[str, Dict[str, Any]] = {}
    for inst, lu in lot_units.items():
        if lu == 0:
            continue
        out[inst] = {
            "entry_price": round(lot_notional[inst] / lu, 6),
            "opened_at": lot_opened_at.get(inst),
        }
    return out


def read_account() -> Dict[str, Any]:
    """Account summary + positions the live loop last snapshotted to disk."""
    data = _read_json(OANDA_DIR / "account_state.json", {})
    peak = _read_json(OANDA_DIR / "peak_nav.json", {}).get("peak_nav")
    nav = float(data.get("nav") or 0.0)
    positions = list(data.get("positions") or [])
    if positions:
        entries = _position_entry_contexts()
        for p in positions:
            ctx = entries.get(p.get("instrument")) or {}
            p["entry_price"] = ctx.get("entry_price")
            p["opened_at"] = ctx.get("opened_at")
            units = p.get("net_units") or 0
            p["side"] = "LONG" if units > 0 else ("SHORT" if units < 0 else "FLAT")
            # Real strategy identifier — the validated long-or-flat SMA(100) trend
            # rule (src/equity/oanda_trend.py), never the repo dirname or a fake label.
            p["strategy"] = "trend_sma100"
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
        "positions": positions,
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
            lane_running = bool(_cached_live_lane_running(mod).get("running"))
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


def read_lane_status() -> Dict[str, Any]:
    """Per-lane halted status (oanda_fx / equity / brain), read-only.

    Wraps ``StateEngine.get_lane_status()`` — the same source the (gated)
    ``/api/control/state`` endpoint uses for its ``lanes`` field — but exposes it
    unconditionally so lane status is visible even when ``AXIOM_CONTROL_ENABLED``
    is off. Never calls ``set_halted``. On any read error this reports every known
    lane as halted=True: for a passive display, defaulting to the scarier state on
    an unreadable file is more honest than defaulting to "all clear".
    """
    from src.scanner.automation.state_engine import KNOWN_LANES, StateEngine

    try:
        eng = StateEngine()
        lanes = {lane: bool(v) for lane, v in eng.get_lane_status().items()}
        global_halted = bool(eng.get_halted())
        readable = True
    except Exception as exc:  # noqa: BLE001 — display data; fail closed, never crash
        logger.warning("read_lane_status: StateEngine unreadable (%s) — reporting fail-closed", exc)
        lanes = {lane: True for lane in KNOWN_LANES}
        global_halted = True
        readable = False
    return {
        "readable": readable,
        "global_halted": global_halted,
        "lanes": lanes,
        "known_lanes": list(KNOWN_LANES),
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


_LIVE_GATE_STATE_PATH = REPO_ROOT / "trained_data" / "equity" / "live_gate_state.json"
_SHIP_GATE_PATH = REPO_ROOT / "trained_data" / "backtests" / "SHIP_GATE.json"
_CYCLE_LEDGER_PATH = EQUITY_DIR / "cycle_ledger.jsonl"


def _equity_ship_gate() -> Dict[str, Any]:
    data = _read_json(_SHIP_GATE_PATH, {})
    if not data:
        return {"available": False}
    return {
        "available": True,
        "gate_pass": bool(data.get("gate_pass")),
        "net_sharpe": data.get("net_sharpe"),
        "max_dd": data.get("max_dd"),
        "asof": data.get("asof"),
        "recommendation": data.get("recommendation"),
        "universe_hash": data.get("universe_hash"),
    }


def _equity_last_cycles(limit: int = 5) -> List[Dict[str, Any]]:
    """Tail of the hash-chained cycle ledger — most recent first. Best-effort:
    a 64KB tail read is plenty for `limit` small JSON lines; a truncated first
    partial line is simply skipped (still fail-soft, never crashes)."""
    try:
        size = _CYCLE_LEDGER_PATH.stat().st_size
        with open(_CYCLE_LEDGER_PATH, "rb") as fh:
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    rows = []
    for line in tail.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    rows.reverse()
    return rows[:limit]


def _equity_gate_decision() -> Dict[str, Any]:
    """Fresh, live per-cycle gate verdict (REFUSE/HALT/NO_ACT/ABSTAIN/CONTINUE),
    re-derived straight from disk via the harvester's own objective decision gate
    (`src.equity.decision_gate.decide_cycle`, lane="equity"). Read-only / side-effect
    free — same function the real harvester loop calls, so this can never drift
    from what the harvester itself would decide right now. Any construction/import
    failure degrades to an honest 'unavailable' rather than a fabricated verdict."""
    try:
        from src.equity.decision_gate import decide_cycle
        from src.scanner.config import ScannerConfig

        decision = decide_cycle(config=ScannerConfig(), project_root=REPO_ROOT, lane="equity")
        return {"available": True, **decision.to_dict()}
    except Exception as exc:  # noqa: BLE001 — display data; never blocks/crashes on failure
        logger.warning("equity decide_cycle() unavailable: %s", exc)
        return {"available": False, "error": type(exc).__name__}


def read_equity_sleeve() -> Dict[str, Any]:
    """Equity-harvester lane status: rebalance plan, LiveGate armed/live-vs-shadow
    state, ship-gate verdict, recent cycle-ledger decisions, and the live gate
    verdict this cycle would get. The harvester's runner (`src.equity.runner`) is
    ALWAYS shadow-execution (it never contacts a broker) — "live" here means the
    separate LiveGate is armed, which is what `scripts/run_equity_harvester.py`
    checks before routing to the real (still practice/paper) IBKR broker.
    """
    data = _read_json(EQUITY_DIR / "rebalance_state.json", {})
    plan = data.get("active_plan") or {}
    live_gate = _read_json(_LIVE_GATE_STATE_PATH, {})
    armed = bool(live_gate.get("armed", False))
    return {
        "connected": bool(plan),
        "dormant": not armed,
        "mode": "live" if armed else "shadow",
        "asof": plan.get("asof"),
        "rebalance_id": plan.get("rebalance_id"),
        "target_weights": plan.get("target_weights", {}),
        "actual_weights": data.get("current_actual_weights", {}),
        "source": "trained_data/equity/rebalance_state.json",
        "live_gate": {
            "available": bool(live_gate),
            "armed": armed,
            "universe_hash": live_gate.get("universe_hash"),
            "initial_nav_fraction": live_gate.get("initial_nav_fraction"),
            "max_portfolio_risk_fraction": live_gate.get("max_portfolio_risk_fraction"),
            "armed_at_ts": live_gate.get("armed_at_ts"),
            "disarmed_at_ts": live_gate.get("disarmed_at_ts"),
            "last_event": live_gate.get("last_event"),
            "last_event_reason": live_gate.get("last_event_reason"),
            "last_event_ts": live_gate.get("last_event_ts"),
        },
        "ship_gate": _equity_ship_gate(),
        "last_cycles": _equity_last_cycles(),
        "gate_decision": _equity_gate_decision(),
    }


_BRAIN_LOOP_DIR = REPO_ROOT / "trained_data" / "brain_loop"
_BRAIN_LOOP_LEDGER_PATH = _BRAIN_LOOP_DIR / "hypotheses.jsonl"
_BRAIN_LOOP_REQUESTS_DIR = _BRAIN_LOOP_DIR / "promotion_requests"
_BRAIN_LOOP_AUDIT_PATH = CLAUDE_DIR / "brain_loop_audit.jsonl"


def read_axiom_operator() -> Dict[str, Any]:
    """Subscription-backed AXIOM operator status.

    This is intentionally read-only dashboard hydration. It never starts Claude,
    never touches controls, and degrades to an honest idle snapshot before the
    first operator epoch has run.
    """
    try:
        from src.axiom_operator.session import read_operator_snapshot

        return read_operator_snapshot()
    except Exception as exc:  # noqa: BLE001 - dashboard visibility must not crash
        logger.warning("axiom operator snapshot unreadable: %s", exc)
        return {
            "has_run": False,
            "session": {
                "status": "unavailable",
                "provider": "claude_subscription",
                "epoch": 0,
                "handoff_summary": "AXIOM operator snapshot unavailable.",
                "open_incidents": [],
                "last_tools": [],
                "blocked_reasons": [str(exc)],
                "next_action": "wait_for_operator",
                "last_action": None,
                "last_reasoning": None,
                "last_error": str(exc),
                "api_key_refused": False,
                "cli_available": None,
            },
            "last_decision": None,
            "recent_decisions": [],
            "source": {
                "session": "trained_data/axiom/operator_session.json",
                "decisions": "trained_data/axiom/operator_decisions.jsonl",
            },
        }


def read_mind_window() -> Dict[str, Any]:
    """AXIOM resident reasoning loop ("mind-window") status: recent
    OBSERVE->DIAGNOSE->PROPOSE->ACT->VERIFY->LOG cycles, the current
    ``agent_autonomy_enabled`` flag state (SHADOW vs LIVE-OPERATIONAL), and
    any pending ESCALATION proposals awaiting the operator. Read-only; never
    triggers a cycle. Degrades to an honest not-yet-run snapshot before the
    resident loop has ever executed.
    """
    try:
        from src.agent_runtime.loop import read_mind_window as _read

        return _read()
    except Exception as exc:  # noqa: BLE001 - dashboard visibility must not crash
        logger.warning("agent runtime mind-window unreadable: %s", exc)
        return {
            "has_run": False,
            "autonomy_enabled": False,
            "recent_cycles": [],
            "last_cycle": None,
            "pending_operator_proposals": [],
            "error": str(exc),
            "source": {
                "cycles": "trained_data/axiom/loop_cycles.jsonl",
                "autonomy_flag": "trained_data/axiom/loop_autonomy.json",
            },
        }


def _brain_loop_ledger_tail(limit: int = 10) -> List[Dict[str, Any]]:
    rows = list(_iter_jsonl(_BRAIN_LOOP_LEDGER_PATH))
    return list(reversed(rows[-limit:]))


def _brain_loop_promotion_requests() -> List[Dict[str, Any]]:
    try:
        from src.brain_loop.promotion import list_requests
        return list_requests(_BRAIN_LOOP_REQUESTS_DIR)
    except Exception as exc:  # noqa: BLE001 — display data; never crash on failure
        logger.warning("brain_loop promotion requests unreadable: %s", exc)
        return []


def _brain_loop_audit_tail(limit: int = 10) -> List[Dict[str, Any]]:
    rows = list(_iter_jsonl(_BRAIN_LOOP_AUDIT_PATH))
    return list(reversed(rows[-limit:]))


def read_brain_loop() -> Dict[str, Any]:
    """Sonnet brain-loop status: recent hypothesis-ledger events, promotion
    requests (including any ``PENDING_OPERATOR`` live promotion awaiting the
    operator's own ARM + start_loop flow — this reader never arms or promotes
    anything, it only lists what's on disk), and de-risk/halt audit events.

    The brain loop (``src/brain_loop/``) has not been run in production as of
    this writing — every source here degrades to an honest empty list, never a
    fabricated cycle.
    """
    ledger = _brain_loop_ledger_tail()
    requests = _brain_loop_promotion_requests()
    audit = _brain_loop_audit_tail()
    pending_operator = [r for r in requests if r.get("status") == "PENDING_OPERATOR"]
    return {
        "has_run": bool(ledger) or bool(requests) or bool(audit),
        "last_event": ledger[0] if ledger else None,
        "recent_ledger": ledger,
        "promotion_requests": requests,
        "pending_operator_count": len(pending_operator),
        "derisk_audit": audit,
        "source": {
            "ledger": "trained_data/brain_loop/hypotheses.jsonl",
            "promotion_requests": "trained_data/brain_loop/promotion_requests/",
            "derisk_audit": ".claude/brain_loop_audit.jsonl",
        },
    }


_CRYPTO_LEDGER_PATH = CRYPTO_DIR / "shadow_momentum_ledger.jsonl"
_CRYPTO_LIVE_GATE_STATE_PATH = CRYPTO_DIR / "live_gate_state.json"


def read_crypto_momentum() -> Dict[str, Any]:
    """Crypto XS-momentum SHADOW lane: current would-be book, running shadow
    P&L, and the accumulating live-forward-OOS track record for the campaign's
    H2/H4 lead (docs/experiment-crypto-edge-hunt-round2-2026-06-29.md). This
    signal FAILED the ship gate on significance — it is NOT a verified edge;
    this panel exists to show whether the forward record eventually confirms
    or refutes the pre-registered +0.75 OOS Sharpe.

    Read-only, real data, no fabrication: an empty ledger (the lane has not
    run yet, or every cycle so far has been a halted no-op) renders as an
    honest zero-cycle state, never a fabricated book or P&L number. The
    live gate is read the same way the equity harvester's is — `armed` can
    only ever be True via an explicit operator `LiveGate.arm()` call, which
    nothing in this codebase currently makes (see
    `src/crypto/crypto_live_gate.py`).
    """
    rows = list(_iter_jsonl(_CRYPTO_LEDGER_PATH))
    last = rows[-1] if rows else None
    live_gate = _read_json(_CRYPTO_LIVE_GATE_STATE_PATH, {})
    armed = bool(live_gate.get("armed", False))

    # Reuse the module's own summary (single source of truth for the Sharpe
    # math) instead of reimplementing it here — a duplicated calculation is
    # exactly how the dashboard and the ledger writer could silently disagree.
    try:
        from src.crypto.momentum_shadow import forward_oos_summary
        summary = forward_oos_summary(ledger_path=_CRYPTO_LEDGER_PATH)
    except Exception as exc:  # noqa: BLE001 — display data; never crash on import/compute failure
        logger.warning("read_crypto_momentum: forward_oos_summary unavailable (%s)", exc)
        summary = {"n_cycles": len(rows), "forward_sharpe_annualized": None}
    n = summary["n_cycles"]
    # forward_oos_summary's zero-cycle state omits this key entirely (never a
    # fabricated Sharpe on n<2) — .get() so the empty-ledger case (lane built
    # but never run) doesn't KeyError the whole endpoint.
    forward_sharpe = summary.get("forward_sharpe_annualized")

    return {
        "has_run": bool(rows),
        "n_forward_cycles": n,
        "current_book": last.get("book") if last else None,
        "current_asof": last.get("asof_date") if last else None,
        "current_gross_leverage": last.get("gross_leverage") if last else None,
        "cumulative_shadow_return": last.get("cumulative_shadow_return", 0.0) if last else 0.0,
        "forward_sharpe_annualized": forward_sharpe,
        "first_asof_date": rows[0].get("asof_date") if rows else None,
        "last_asof_date": last.get("asof_date") if last else None,
        "recent_cycles": list(reversed(rows[-20:])),
        "construction": last.get("construction") if last else None,
        "live_gate": {
            "available": bool(live_gate),
            "armed": armed,
            "last_event": live_gate.get("last_event"),
            "last_event_reason": live_gate.get("last_event_reason"),
        },
        "mode": "live" if armed else "shadow",
        "source": {
            "ledger": "trained_data/crypto/shadow_momentum_ledger.jsonl",
            "live_gate": "trained_data/crypto/live_gate_state.json",
            "pre_registration": "docs/experiment-crypto-edge-hunt-round2-2026-06-29.md",
        },
    }


_TRACK_B_LEDGER_PATH = REPO_ROOT / "trained_data" / "research" / "track_b_shadow_ledger.jsonl"
_TRACK_B_LIVE_GATE_STATE_PATH = REPO_ROOT / "trained_data" / "equity" / "track_b" / "live_gate_state.json"


def read_track_b() -> Dict[str, Any]:
    """Track B (SEC filing-text research-alpha) SHADOW lane: current would-be
    Q5 book, running shadow P&L, and the accumulating live-forward-OOS track
    record for the campaign's agentic filing-text factor (docs/experiment-
    equity-research-alpha-prereg-2026-06-30.md). This signal is
    self-labeled `overall_verdict=INSUFFICIENT` — UNDERPOWERED, not a
    measured NO EDGE (N scored filings vs ~405 needed for 80% power on the
    +0.16 rank-IC point estimate; see docs/adversarial-review-no-edge-
    verdicts-2026-07-02.md). This panel exists to show whether the forward
    record eventually clears the gate, not because the signal is verified.

    Read-only, real data, no fabrication: an empty ledger (the lane has not
    run yet, or every cycle so far has been a halted no-op) renders as an
    honest zero-cycle state, never a fabricated book or P&L number. The live
    gate is read the same way the crypto momentum lane's is — `armed` can
    only ever be True via an explicit operator `LiveGate.arm()` call, which
    nothing in this codebase currently makes (see
    `src/equity/track_b_live_gate.py`).
    """
    rows = list(_iter_jsonl(_TRACK_B_LEDGER_PATH))
    last = rows[-1] if rows else None
    live_gate = _read_json(_TRACK_B_LIVE_GATE_STATE_PATH, {})
    armed = bool(live_gate.get("armed", False))

    # Reuse the module's own summary (single source of truth for the Sharpe
    # math) instead of reimplementing it here — a duplicated calculation is
    # exactly how the dashboard and the ledger writer could silently disagree.
    try:
        from src.equity.track_b_shadow import forward_oos_summary, construction_manifest
        summary = forward_oos_summary(ledger_path=_TRACK_B_LEDGER_PATH)
        # Honest coverage even before the first cycle has run — the scored-
        # filing count doesn't depend on a ledger row existing.
        try:
            from src.equity.track_b_shadow import load_frozen_scores
            n_scored_now = len(load_frozen_scores())
        except Exception:  # noqa: BLE001 — display data; fall back to last ledger row's count
            n_scored_now = last.get("n_scored_filings") if last else None
    except Exception as exc:  # noqa: BLE001 — display data; never crash on import/compute failure
        logger.warning("read_track_b: track_b_shadow module unavailable (%s)", exc)
        summary = {"n_cycles": len(rows), "forward_sharpe_annualized": None}
        n_scored_now = last.get("n_scored_filings") if last else None
        construction_manifest = None
    n = summary["n_cycles"]
    # forward_oos_summary's zero-cycle state omits this key entirely (never a
    # fabricated Sharpe on n<2) — .get() so the empty-ledger case (lane built
    # but never run) doesn't KeyError the whole endpoint.
    forward_sharpe = summary.get("forward_sharpe_annualized")
    construction = (
        last.get("construction") if last
        else (construction_manifest(n_scored_now) if construction_manifest and n_scored_now else None)
    )

    return {
        "has_run": bool(rows),
        "n_forward_cycles": n,
        "n_scored_filings": n_scored_now,
        "current_book": last.get("book") if last else None,
        "current_asof": last.get("asof_date") if last else None,
        "current_gross_leverage": last.get("gross_leverage") if last else None,
        "cumulative_shadow_return": last.get("cumulative_shadow_return", 0.0) if last else 0.0,
        "forward_sharpe_annualized": forward_sharpe,
        "first_asof_date": rows[0].get("asof_date") if rows else None,
        "last_asof_date": last.get("asof_date") if last else None,
        "recent_cycles": list(reversed(rows[-20:])),
        "construction": construction,
        "live_gate": {
            "available": bool(live_gate),
            "armed": armed,
            "last_event": live_gate.get("last_event"),
            "last_event_reason": live_gate.get("last_event_reason"),
        },
        "mode": "live" if armed else "shadow",
        "source": {
            "ledger": "trained_data/research/track_b_shadow_ledger.jsonl",
            "live_gate": "trained_data/equity/track_b/live_gate_state.json",
            "pre_registration": "docs/experiment-equity-research-alpha-prereg-2026-06-30.md",
            "scaleup_run": "docs/track-b-postcutoff-scaleup-2026-07-02.md",
            "adversarial_review": "docs/adversarial-review-no-edge-verdicts-2026-07-02.md",
        },
    }


_CRYPTO_CARRY_LEDGER_PATH = REPO_ROOT / "trained_data" / "crypto_carry" / "shadow_carry_ledger.jsonl"
_CRYPTO_CARRY_LIVE_GATE_STATE_PATH = REPO_ROOT / "trained_data" / "crypto_carry" / "live_gate_state.json"


def read_crypto_carry() -> Dict[str, Any]:
    """Crypto cash-and-carry SHADOW lane: current would-be book, running
    shadow P&L, and the accumulating live-forward-OOS track record for the
    frozen positive-funding-only cash-and-carry construction
    (docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md). This signal
    FAILED the ship gate on turnover cost — it is NOT a verified edge; this
    panel exists to show whether the forward record eventually confirms a
    real deployment fares differently. **This is a risk premium with a real
    exchange-solvency/liquidation tail, not free money** — see
    `src/crypto/carry_shadow.py`'s `RISK_PREMIUM_NOTE`.

    Read-only, real data, no fabrication: an empty ledger (the lane has not
    run yet, or every cycle so far has been a halted no-op) renders as an
    honest zero-cycle state, never a fabricated book or P&L number. The live
    gate is read the same way every other lane's is — `armed` can only ever
    be True via an explicit operator `LiveGate.arm()` call, which nothing in
    this codebase currently makes (see `src/crypto/crypto_carry_live_gate.py`).
    """
    rows = list(_iter_jsonl(_CRYPTO_CARRY_LEDGER_PATH))
    last = rows[-1] if rows else None
    live_gate = _read_json(_CRYPTO_CARRY_LIVE_GATE_STATE_PATH, {})
    armed = bool(live_gate.get("armed", False))

    # Reuse the module's own summary (single source of truth for the Sharpe
    # math) instead of reimplementing it here — a duplicated calculation is
    # exactly how the dashboard and the ledger writer could silently disagree.
    try:
        from src.crypto.carry_shadow import forward_oos_summary, RISK_PREMIUM_NOTE
        summary = forward_oos_summary(ledger_path=_CRYPTO_CARRY_LEDGER_PATH)
    except Exception as exc:  # noqa: BLE001 — display data; never crash on import/compute failure
        logger.warning("read_crypto_carry: carry_shadow unavailable (%s)", exc)
        summary = {"n_cycles": len(rows), "forward_sharpe_annualized": None}
        RISK_PREMIUM_NOTE = None
    n = summary.get("n_cycles", len(rows))
    # forward_oos_summary's zero-cycle state omits this key entirely (never a
    # fabricated Sharpe on n<2) — .get() so the empty-ledger case (lane built
    # but never run) doesn't KeyError the whole endpoint (this exact bug hit
    # read_crypto_momentum() before it was fixed 2026-07-04 — never repeat it).
    forward_sharpe = summary.get("forward_sharpe_annualized")

    return {
        "has_run": bool(rows),
        "n_forward_cycles": n,
        "current_book": last.get("book") if last else None,
        "current_asof": last.get("asof_date") if last else None,
        "current_gross_leverage": last.get("gross_leverage") if last else None,
        "cumulative_shadow_return": last.get("cumulative_shadow_return", 0.0) if last else 0.0,
        "forward_sharpe_annualized": forward_sharpe,
        "first_asof_date": rows[0].get("asof_date") if rows else None,
        "last_asof_date": last.get("asof_date") if last else None,
        "recent_cycles": list(reversed(rows[-20:])),
        "construction": last.get("construction") if last else None,
        "risk_premium_note": RISK_PREMIUM_NOTE,
        "live_gate": {
            "available": bool(live_gate),
            "armed": armed,
            "last_event": live_gate.get("last_event"),
            "last_event_reason": live_gate.get("last_event_reason"),
        },
        "mode": "live" if armed else "shadow",
        "source": {
            "ledger": "trained_data/crypto_carry/shadow_carry_ledger.jsonl",
            "live_gate": "trained_data/crypto_carry/live_gate_state.json",
            "pre_registration": "docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md",
        },
    }


_HEDGE_DIR = REPO_ROOT / "trained_data" / "hedge"
_RAW_VS_HEDGED_LEDGER_PATH = _HEDGE_DIR / "raw_vs_hedged_ledger.jsonl"
_HEDGE_SCORECARD_PATH = _HEDGE_DIR / "hedge_scorecard_report.json"
_EXPOSURE_HISTORY_PATH = _HEDGE_DIR / "exposure_history.jsonl"


def _live_fx_exposure() -> Optional[Dict[str, Any]]:
    """Compute the LIVE FX trend-lane exposure + hedge proposal right now,
    directly from the open OANDA book — no dependency on the capture loop
    having run. Returns None (honest empty) on a stale/flat book or any
    compute error. Read-only: run_cycle_for_strategy is called with
    persist=False so nothing is written and no order path is touched."""
    try:
        from src.hedge.hedged_shadow_lane import load_fx_trend_book, run_cycle_for_strategy
        book = load_fx_trend_book()
        if book is None:
            return None
        row = run_cycle_for_strategy("fx_trend", book=book, persist=False)
    except Exception as exc:  # noqa: BLE001 — display data; never crash the endpoint
        logger.warning("read_hedge: live FX exposure compute failed (%s)", exc)
        return None
    if not row:
        return None
    exposure = row.get("exposure", {})
    hedge = row.get("hedge", {})
    return {
        "asof_date": row.get("asof_date"),
        "nav": (row.get("meta") or {}).get("nav"),
        "net_currency_exposure": exposure.get("net_currency_exposure", {}),
        "net_correlation_bucket_exposure": exposure.get("net_correlation_bucket_exposure", {}),
        "concentration_warnings": exposure.get("concentration_warnings", []),
        "narrative": exposure.get("narrative", []),
        "position_count": exposure.get("position_count"),
        "resolved_position_count": exposure.get("resolved_position_count"),
        "raw_net_return": (row.get("raw") or {}).get("net_return"),
        "raw_return_basis": (row.get("meta") or {}).get("raw_return_basis"),
        "hedge_status": hedge.get("status"),
        "hedge_decision": hedge.get("decision"),
        "applied_hedge": hedge.get("applied_proposal"),
        "hedged_return_basis": (row.get("hedged") or {}).get("return_basis"),
    }


def read_hedge() -> Dict[str, Any]:
    """AXIOM Hedge Layer readout — SHADOW / ANALYSIS-ONLY.

    Three things the recent hedge-layer backend work now produces, none of
    which had a dashboard surface before:
      1. LIVE FX exposure netting — decomposes the open OANDA trend book into
         signed currency buckets so a hidden concentration ("one USD bet
         wearing three outfits") is visible, plus the Phase-3 hedge proposal.
         Computed live each request (persist=False), so it works even before
         the exposure-history capture loop is running.
      2. Raw-vs-hedged scorecard + recent ledger cycles across every covered
         strategy (equity_harvester / crypto_momentum / track_b / fx_trend).
      3. Exposure-history accumulation counter — the training bridge
         (trained_data/hedge/exposure_history.jsonl); shows how many
         snapshots have been captured toward the P2 risk-target feature round.

    Read-only, real data, honest empty states: no ledger / no history / a flat
    book each render as an explicit empty state, never a fabricated number.
    The hedge layer places no orders, unhalts nothing, mutates no broker —
    every row carries paper_only / runtime_allowed flags from its writer.
    """
    scorecard = _read_json(_HEDGE_SCORECARD_PATH, {})
    ledger_rows = list(_iter_jsonl(_RAW_VS_HEDGED_LEDGER_PATH))
    exposure_history = list(_iter_jsonl(_EXPOSURE_HISTORY_PATH))
    last_history = exposure_history[-1] if exposure_history else None

    return {
        "live_fx_exposure": _live_fx_exposure(),
        "scorecards": scorecard.get("scorecards", {}),
        "strategies_with_history": scorecard.get("strategies_with_history", []),
        "recent_cycles": list(reversed(ledger_rows[-20:])),
        "n_ledger_cycles": len(ledger_rows),
        "exposure_history": {
            "n_snapshots": len(exposure_history),
            "first_captured_at": exposure_history[0].get("captured_at_utc") if exposure_history else None,
            "last_captured_at": last_history.get("captured_at_utc") if last_history else None,
            "last_nav": last_history.get("nav") if last_history else None,
            "last_net_currency": last_history.get("net_currency_notional_home") if last_history else None,
            "note": "training bridge for the P2 risk-target exposure-feature round "
                    "(docs/experiment-risk-target-p2-exposure-features-2026-07-08.md)",
        },
        "paper_only": bool(scorecard.get("paper_only", True)),
        "runtime_allowed": bool(scorecard.get("runtime_allowed", False)),
        "human_review_required": bool(scorecard.get("human_review_required", True)),
        "source": {
            "ledger": "trained_data/hedge/raw_vs_hedged_ledger.jsonl",
            "scorecard": "trained_data/hedge/hedge_scorecard_report.json",
            "exposure_history": "trained_data/hedge/exposure_history.jsonl",
            "roadmap": "AXIOM Hedge Layer Roadmap (docs/)",
        },
    }


_LEARNING_LOOP_HISTORY_PATH = REPO_ROOT / "trained_data" / "learning_loop" / "history.jsonl"
_LEARNING_LOOP_STATUS_PATH = CLAUDE_DIR / "brain" / "learning_loop_status.json"
_LEARNING_LOOP_RETRAIN_REQUESTS_DIR = REPO_ROOT / "trained_data" / "retrain_requests"


def read_risk_target_evidence() -> Dict[str, Any]:
    """Read-only Phase-I risk-target evidence cockpit view.

    Surfaces the local evidence authority's committed disposition state for the
    risk-target vertical slice (per-lane package state + import verdict), read
    from ``trained_data/evidence/indexes/current.json``. Purely a display
    projection — it never signs, promotes, or mutates evidence, and imports
    only the dependency-free view helper (no training stack).
    """
    try:
        from src.evidence.risk_target.dashboard import risk_target_evidence_view

        view = risk_target_evidence_view(EVIDENCE_DIR)
        view["connected"] = bool(view.get("available"))
        return view
    except Exception as exc:  # noqa: BLE001 - dashboard reads must never 500
        return {
            "available": False,
            "connected": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "lanes": [],
            "champions": {},
        }


def read_learning_loop() -> Dict[str, Any]:
    """Market-closed continual-learning readout: last offline retrain cycle
    (accepted/rejected, calibration metric before/after), pending Tier-7
    retrain markers awaiting the next cycle, and recent history.

    Backing job: ``scripts/offline_learning_cycle.py`` — drains
    ``trained_data/retrain_requests/`` markers and walk-forward-gates a
    ``RiskCalibrationLearner`` update (risk/execution/calibration scope
    only, never directional-alpha; see risk_calibration_learner.py). This
    panel is read-only and cannot trigger a cycle — it only reports what the
    scheduled job already did.

    Read-only, real data, no fabrication: no history file yet (job has never
    run) renders as an honest zero-cycle state, not a fabricated result.
    """
    status = _read_json(_LEARNING_LOOP_STATUS_PATH, {})
    history_rows = list(_iter_jsonl(_LEARNING_LOOP_HISTORY_PATH))
    pending_markers = 0
    if _LEARNING_LOOP_RETRAIN_REQUESTS_DIR.exists():
        pending_markers = len(list(_LEARNING_LOOP_RETRAIN_REQUESTS_DIR.glob("*.json")))

    return {
        "has_run": bool(history_rows),
        "pending_retrain_markers": pending_markers,
        "last_cycle": status or (history_rows[-1] if history_rows else None),
        "recent_cycles": list(reversed(history_rows[-20:])),
        "scope": "risk/execution/calibration only (position sizing, confidence "
                 "calibration) — never directional alpha",
        # Honest disclosure (adversarial review 2026-07-04): the learned
        # size_multiplier is SHADOW/advisory — no live sizing path
        # (DynamicPositionSizer) consumes risk_calibration_state.json yet. A
        # promoted candidate changes this readout + the model file only, not
        # any real order size. Wiring is a separate operator-gated hot-path step.
        "consumed_by_live_sizing": False,
        "source": {
            "history": "trained_data/learning_loop/history.jsonl",
            "status": ".claude/brain/learning_loop_status.json",
            "retrain_requests": "trained_data/retrain_requests/",
            "batch_job": "scripts/offline_learning_cycle.py",
        },
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

    snapshot_stale = snap_age is not None and snap_age > TIER7_FRESH_S
    running = bool(data.get("running")) and not snapshot_stale
    running_reason = data.get("running_reason")
    if snapshot_stale:
        running_reason = f"snapshot stale ({int(snap_age)}s > {int(TIER7_FRESH_S)}s)" if snap_age is not None else "snapshot stale"

    return {
        "connected": True,
        "source": rel,
        "snapshot_age_s": round(snap_age, 1) if snap_age is not None else None,
        "snapshot_stale": snapshot_stale,
        # honest liveness: trust the bot's pid+heartbeat check only while the
        # exported snapshot itself is fresh; an old "fresh heartbeat" is stale truth.
        "running": running,
        "running_reason": running_reason,
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
_lane_cache: Dict[str, Any] = {"ts": 0.0, "data": None}
_LANE_CACHE_TTL_S = 2.0


def _cached_live_lane_running(mod) -> Dict[str, Any]:
    """TTL-cache the ps-based live-lane oracle call.

    ``live_lane_running()`` shells out to `ps axo command` (running_status.py).
    read_status()/read_health() are called on EVERY SSE tick for EVERY connected
    client (app.py's ``_event_stream``, every ~2s, no per-connection dedup) — with
    ~86 concurrent connections that's dozens of blocking subprocess spawns/sec,
    which stalled /api/control/state under load (shared event-loop/threadpool
    contention). A short TTL collapses concurrent callers onto one real subprocess
    call per window regardless of connection count.
    """
    now = time.time()
    if _lane_cache["data"] is not None and (now - _lane_cache["ts"]) < _LANE_CACHE_TTL_S:
        return _lane_cache["data"]
    data = mod.live_lane_running()
    _lane_cache["ts"], _lane_cache["data"] = now, data
    return data


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
            lanes = {"available": True, **_cached_live_lane_running(mod)}
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


_diag_cache: Dict[str, Any] = {"ts": 0.0, "network_latency_ms": None}


def _ps_stat(pid: Optional[int]) -> Optional[Dict[str, float]]:
    """Real per-process CPU%/RSS-MB for a pid via `ps` (no psutil dependency;
    reuses the same subprocess+ps pattern already used by control.py's loop-pid
    lookup). Returns None if the pid is absent/dead — never a fabricated number."""
    if not pid:
        return None
    import subprocess
    try:
        out = subprocess.run(["ps", "-o", "%cpu=,rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=3)
        line = out.stdout.strip()
        if not line:
            return None
        cpu_s, rss_s = line.split()
        return {"cpu_pct": float(cpu_s), "rss_mb": float(rss_s) / 1024.0}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def read_tier7_diagnostics(client=None) -> Dict[str, Any]:
    """Real per-subsystem checklist for the Tier 7 panel — every row measured from
    an actual signal, never a fabricated latency. Mirrors the operator's requested
    checklist (Market Data / Execution Layer / Risk Engine / Strategy Runtime /
    Position Reconciler / Memory-CPU / Disk-Network) but each metric is honestly
    computed from what we can actually observe:
      - Market Data: real OANDA pricing round-trip latency (measured here, cached
        30s so this endpoint doesn't hammer the broker on every poll).
      - Execution Layer / Strategy Runtime: the trend/tier7 loop pids from the
        live-lane oracle + their real heartbeat/snapshot ages.
      - Risk Engine: the recorded verify_gate verdict (GREEN/RED) + its real age.
      - Position Reconciler: freshness of account_state.json (the snapshot the
        trend loop rewrites each cycle after reconciling positions).
      - Memory/CPU: real `ps` cpu%/rss for the trend + tier7 processes.
      - Disk/Network: real free-disk% (shutil.disk_usage) + the measured broker
        round-trip above.
    """
    import shutil
    import time as _time

    health = read_health()
    lanes = health.get("lanes") or {}
    gates = health.get("gates") or {}
    tier7 = read_tier7()

    # Market data latency — cached 30s; a live client call only on cache miss.
    now = _time.time()
    net_ms = _diag_cache.get("network_latency_ms")
    if client is not None and (now - _diag_cache.get("ts", 0.0)) > 30.0:
        t0 = _time.monotonic()
        try:
            client.get_pricing(instruments="EUR_USD")
            net_ms = round((_time.monotonic() - t0) * 1000, 1)
        except Exception:  # noqa: BLE001 — best-effort diagnostic, never blocks
            net_ms = None
        _diag_cache["ts"], _diag_cache["network_latency_ms"] = now, net_ms

    trend_pid = None
    tier7_pid = (tier7.get("last_cycle") or {}).get("pid") if tier7.get("connected") else None
    try:
        import subprocess
        out = subprocess.run(["pgrep", "-f", "run_oanda_trend.py"], capture_output=True, text=True, timeout=3)
        pids = [int(p) for p in out.stdout.split() if p.isdigit()]
        trend_pid = pids[0] if pids else None
    except (OSError, subprocess.SubprocessError, ValueError):
        # A hung/slow pgrep (subprocess.TimeoutExpired, a SubprocessError subclass)
        # must degrade this row to unknown, not crash the worker — it did exactly
        # that at 00:44 on 2026-07-02 (uncaught TimeoutExpired killed the process;
        # launchd's KeepAlive restarted it). Mirrors _ps_stat()'s except clause above.
        pass

    trend_stat = _ps_stat(trend_pid)
    tier7_stat = _ps_stat(tier7_pid)
    disk_free_pct = None
    try:
        du = shutil.disk_usage(str(REPO_ROOT))
        disk_free_pct = round(du.free / du.total * 100, 1)
    except OSError:
        pass

    checks = [
        {"label": "Market Data", "ok": net_ms is not None,
         "metric": f"{net_ms}ms" if net_ms is not None else None},
        {"label": "Execution Layer", "ok": lanes.get("oanda_trend_proc"),
         "metric": f"{ago_seconds_str(lanes.get('account_state_age_s'))}" if lanes.get("account_state_age_s") is not None else None},
        {"label": "Risk Engine", "ok": gates.get("status") == "GREEN" if gates.get("available") else None,
         "metric": f"{ago_seconds_str(gates.get('verdict_age_s'))}" if gates.get("verdict_age_s") is not None else None},
        {"label": "Strategy Runtime", "ok": tier7.get("running") if tier7.get("connected") else None,
         "metric": f"{ago_seconds_str((tier7.get('last_cycle') or {}).get('age_seconds'))}"
                   if (tier7.get("last_cycle") or {}).get("age_seconds") is not None else None},
        {"label": "Position Reconciler", "ok": lanes.get("account_state_fresh"),
         "metric": f"{ago_seconds_str(lanes.get('account_state_age_s'))}" if lanes.get("account_state_age_s") is not None else None},
        {"label": "Memory / CPU", "ok": trend_stat is not None,
         "metric": f"{trend_stat['cpu_pct']:.0f}% / {trend_stat['rss_mb']:.0f}MB" if trend_stat else None},
        {"label": "Disk / Network", "ok": disk_free_pct is not None and disk_free_pct > 10,
         "metric": f"{disk_free_pct}% free" + (f" / {net_ms}ms" if net_ms is not None else "") if disk_free_pct is not None else None},
    ]
    known = [c for c in checks if c["ok"] is not None]
    score = round(sum(1 for c in known if c["ok"]) / len(known) * 100, 1) if known else None
    return {"checks": checks, "score": score,
            "trend_pid": trend_pid, "tier7_pid": tier7_pid,
            "tier7_process_stat": tier7_stat}


def ago_seconds_str(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 90:
        return f"{round(seconds)}s"
    if seconds < 5400:
        return f"{round(seconds / 60)}m"
    return f"{seconds / 3600:.1f}h"


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
        vol = c.get("volume")  # real OANDA tick-count volume when the broker reports it
        candles.append({"time": ts, "open": o, "high": h, "low": l, "close": cl,
                        "volume": int(vol) if isinstance(vol, (int, float)) else None})
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
