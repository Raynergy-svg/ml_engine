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
    return out


def _last_fill_time() -> Optional[str]:
    """Cheap tail-read of the transaction ledger for the most recent fill time."""
    path = OANDA_DIR / "transactions.jsonl"
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(max(0, size - 16384))
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
        if obj.get("time"):
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

    # Trend lane liveness = age of the account snapshot it rewrites each cycle.
    acct_age = None
    try:
        acct_age = now.timestamp() - (OANDA_DIR / "account_state.json").stat().st_mtime
    except OSError:
        pass
    lane_running = acct_age is not None and acct_age <= LANE_FRESH_S

    return {
        "halted": bool(state.get("halted", True)),  # fail-closed default: assume halted
        "mode": state.get("mode"),
        "status": state.get("status"),
        "scan_cycle_count": state.get("scan_cycle_count"),
        "environment": "practice",  # immutable Hard NO
        "running": bool(lane_running),
        "lane_running": bool(lane_running),
        "account_snapshot_age_s": round(acct_age, 1) if acct_age is not None else None,
        "last_fill_time": _last_fill_time(),
        "scanner_heartbeat_alive": bool(hb_alive),
        "scanner_heartbeat_age_s": round(hb_age, 1) if hb_age is not None else None,
        "scanner_pid": hb.get("pid"),
        "last_updated": state.get("last_updated"),
    }


def read_trades(limit: int = 100) -> Dict[str, Any]:
    """Parse ORDER_FILL transactions into a trade-history table (most recent first)."""
    fills: List[Dict[str, Any]] = []
    for t in _iter_jsonl(OANDA_DIR / "transactions.jsonl"):
        if t.get("type") != "ORDER_FILL":
            continue
        try:
            units = float(t.get("units") or 0)
        except (ValueError, TypeError):
            units = 0.0
        fills.append({
            "id": t.get("id"),
            "time": t.get("time"),
            "instrument": t.get("instrument"),
            "units": units,
            "side": "BUY" if units >= 0 else "SELL",
            "price": float(t.get("price") or 0) or None,
            "pl": float(t.get("pl") or 0),
            "financing": float(t.get("financing") or 0),
            "half_spread_cost": float(t.get("halfSpreadCost") or 0),
            "reason": t.get("reason"),
            "balance": float(t.get("accountBalance") or 0) or None,
            "tag": (t.get("clientExtensions") or {}).get("tag"),
        })
    fills.reverse()  # most recent first
    return {"connected": bool(fills), "source": "transactions.jsonl",
            "count": len(fills), "trades": fills[: int(limit)]}


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
    """Order/position-book sentiment snapshot. LABELED PLACEHOLDER — data-only."""
    data = _read_json(OANDA_DIR / "sentiment_snapshot.json", {})
    return {
        "connected": bool(data.get("books")),
        "wired_into_strategy": False,
        "note": data.get("note", "DATA-ONLY; not wired into strategy"),
        "books": data.get("books", {}),
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
        "last_cycle": data.get("last_cycle") or {},
        "self_heal": data.get("self_heal") or {},
        "meta_last_event": data.get("meta_last_event") or {},
        "generated_at": gen,
        "note": data.get("note"),
    }


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
        prices[inst] = {
            "bid": bid, "ask": ask, "mid": mid,
            "spread_pips": round((ask - bid) / _pip_size(inst), 2),
            "time": p.get("time"),
            "status": p.get("status"),
            "tradeable": p.get("tradeable", True),
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
