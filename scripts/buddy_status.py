#!/usr/bin/env python
"""US-036: Live performance metrics and system health summary.

Usage:
    python scripts/buddy_status.py
    python main.py status  (once wired into dispatch table)

Displays:
    - NAV and account info
    - Open trades with P/L
    - Win rate and last 5 trades
    - Model/ensemble health
    - Agent weights status
    - Risk scaler state
    - Blocked pairs
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── ANSI helpers ──────────────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    G = "\033[92m"   # green
    Y = "\033[93m"   # yellow
    R = "\033[91m"   # red
    B = "\033[1m"    # bold
    D = "\033[2m"    # dim
    CY = "\033[96m"  # cyan
    W = "\033[97m"   # white
    _  = "\033[0m"   # reset


def _hline(width: int = 70) -> str:
    return "─" * width


def _section(title: str) -> None:
    print(f"\n{C.B}{C.CY}┌{'─' * 68}┐{C._}")
    print(f"{C.B}{C.CY}│ {title:<66} │{C._}")
    print(f"{C.B}{C.CY}└{'─' * 68}┘{C._}")


# ── Data loaders ──────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    except FileNotFoundError:
        return []


def _load_state() -> Dict[str, Any]:
    return _load_json(_ROOT / ".claude" / "state.json") or {}


def _load_adaptive_sizer_state() -> Dict[str, Any]:
    return _load_json(_ROOT / "trained_data" / "adaptive_sizer_state.json") or {}


def _load_risk_scaler_state() -> Dict[str, Any]:
    return _load_json(_ROOT / "trained_data" / "risk_scaler_state.json") or {}


def _load_agent_weights() -> Dict[str, Any]:
    return _load_json(_ROOT / "trained_data" / "models" / "agent_weights.json") or {}


def _load_journal() -> List[Dict[str, Any]]:
    path = _ROOT / "trained_data" / "trade_journal_rl.json"
    data = _load_json(path)
    if isinstance(data, list):
        return data
    return []


def _load_blocked_pairs() -> List[str]:
    state = _load_state()
    return state.get("blocked_pairs", [])


def _load_scan_diagnostics_report() -> Dict[str, Any]:
    return _load_json(_ROOT / "trained_data" / "scan_diagnostics_report.json") or {}


def _load_scan_diagnostics_state() -> Dict[str, Any]:
    return _load_json(_ROOT / "trained_data" / "scan_diagnostics_state.json") or {}


def _load_signal_funnel_history() -> List[Dict[str, Any]]:
    return _load_jsonl(_ROOT / "trained_data" / "signal_funnel_state.jsonl")


def _load_execution_quality_state() -> Dict[str, Any]:
    return _load_json(_ROOT / "trained_data" / "execution_quality.json") or {}


def _load_pair_accuracy() -> Dict[str, Any]:
    return _load_json(_ROOT / "trained_data" / "pair_accuracy.json") or {}


def _load_confidence_calibration() -> Dict[str, Any]:
    return _load_json(_ROOT / "trained_data" / "confidence_calibration.json") or {}


def _derive_rr_ratio(trade: Dict[str, Any]) -> Optional[float]:
    """Derive R:R from explicit ratio, pip distances, or stored prices."""
    outcome = trade.get("outcome") or {}

    for source in (outcome, trade):
        if not isinstance(source, dict):
            continue
        for key in ("rr_ratio", "risk_reward"):
            value = source.get(key)
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if value_f > 0:
                return value_f

    try:
        sl_pips = float((outcome.get("sl_pips") if isinstance(outcome, dict) else None) or trade.get("sl_pips") or 0)
        tp_pips = float((outcome.get("tp_pips") if isinstance(outcome, dict) else None) or trade.get("tp_pips") or 0)
    except (TypeError, ValueError):
        sl_pips = 0.0
        tp_pips = 0.0
    if sl_pips > 0 and tp_pips > 0:
        return tp_pips / sl_pips

    pair = str(trade.get("pair", "") or "")
    entry_price = trade.get("entry_price")
    sl_price = trade.get("sl_price")
    tp_price = trade.get("tp_price")
    try:
        entry = float(entry_price or 0)
        sl = float(sl_price or 0)
        tp = float(tp_price or 0)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or sl <= 0 or tp <= 0:
        return None

    pip = 0.01 if "JPY" in pair else 0.0001
    sl_pips = abs(entry - sl) / pip if pip > 0 else 0.0
    tp_pips = abs(tp - entry) / pip if pip > 0 else 0.0
    if sl_pips > 0 and tp_pips > 0:
        return tp_pips / sl_pips
    return None


def _journal_trade_history() -> List[tuple[float, bool]]:
    """Closed-trade history from the RL journal sorted by close time."""
    journal = _load_journal()
    closed: List[tuple[str, float, bool]] = []
    for trade in journal:
        outcome = trade.get("outcome")
        if not isinstance(outcome, dict):
            continue
        try:
            pnl = float(outcome.get("realized_pl", 0) or 0)
            won = bool(outcome.get("trade_won", False))
        except (TypeError, ValueError):
            continue
        closed.append((
            str(outcome.get("close_time") or outcome.get("exit_time") or trade.get("timestamp") or ""),
            pnl,
            won,
        ))
    closed.sort(key=lambda item: item[0])
    return [(pnl, won) for _, pnl, won in closed]


def _build_scan_diagnostics_snapshot() -> Dict[str, Any]:
    report = _load_scan_diagnostics_report()
    state = _load_scan_diagnostics_state()
    if not report and not state:
        return {}
    return {
        "summary": str(report.get("diagnostic_summary", "No diagnostic summary")),
        "gate_pass_rate_pct": report.get("gate_pass_rate_pct"),
        "pairs_with_direction_pct": report.get("pairs_with_direction_pct"),
        "avg_confidence": report.get("avg_confidence"),
        "avg_top_score": report.get("avg_top_score"),
        "stalled": bool(report.get("stalled", False)),
        "scans_since_last_trade": state.get("scans_since_last_trade", report.get("scans_since_last_trade")),
        "total_scans": state.get("total_scans", report.get("total_scans")),
        "timestamp": report.get("timestamp") or state.get("last_updated"),
    }


def _build_signal_funnel_snapshot(window: int = 10) -> Dict[str, Any]:
    rows = _load_signal_funnel_history()
    if not rows:
        return {}
    recent = rows[-window:]
    count = len(recent)

    def _avg(key: str) -> float:
        vals = [float(r.get(key, 0) or 0) for r in recent]
        return round(sum(vals) / max(len(vals), 1), 2)

    stages = [str(r.get("dominant_kill_stage", "none")) for r in recent]
    stage_counts: Dict[str, int] = {}
    for stage in stages:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    dominant_stage = max(stage_counts.items(), key=lambda item: item[1])[0] if stage_counts else "none"

    return {
        "window": count,
        "tradeable_pct": _avg("tradeable_pct"),
        "passed_confidence_pct": _avg("passed_confidence_pct"),
        "passed_momentum_pct": _avg("passed_momentum_pct"),
        "passed_risk_pct": _avg("passed_risk_pct"),
        "dominant_kill_stage": dominant_stage,
        "latest": recent[-1],
    }


def _build_pair_accuracy_snapshot(limit: int = 4) -> Dict[str, Any]:
    data = _load_pair_accuracy()
    journal = _load_journal()

    journal_rows: Dict[str, Dict[str, Any]] = {}
    for trade in journal:
        outcome = trade.get("outcome")
        if not isinstance(outcome, dict):
            continue
        pair = str(trade.get("pair", "") or "")
        if not pair:
            continue
        row = journal_rows.setdefault(pair, {
            "pair": pair,
            "accuracy": 0.0,
            "rolling_total": 0,
            "rolling_wins": 0,
            "total": 0,
        })
        row["total"] += 1
        row["rolling_total"] += 1
        if bool(outcome.get("trade_won", False)):
            row["rolling_wins"] += 1

    for row in journal_rows.values():
        if row["rolling_total"] > 0:
            row["accuracy"] = row["rolling_wins"] / row["rolling_total"]

    use_journal = bool(journal_rows)
    file_total = 0
    journal_total = sum(int(r["rolling_total"]) for r in journal_rows.values())
    if isinstance(data, dict) and data:
        try:
            file_total = sum(int((info or {}).get("rolling_total", (info or {}).get("total", 0)) or 0) for info in data.values() if isinstance(info, dict))
        except Exception:
            file_total = 0
    if file_total > 0 and journal_total > 0 and file_total <= journal_total * 2:
        use_journal = False

    source_rows: List[Dict[str, Any]] = []
    if use_journal:
        source_rows = list(journal_rows.values())
    elif isinstance(data, dict):
        for pair, info in data.items():
            if not isinstance(info, dict):
                continue
            accuracy = info.get("accuracy")
            rolling_total = int(info.get("rolling_total", info.get("total", 0)) or 0)
            rolling_wins = int(info.get("rolling_wins", info.get("wins", 0)) or 0)
            if accuracy is None:
                continue
            source_rows.append({
                "pair": pair,
                "accuracy": float(accuracy),
                "rolling_total": rolling_total,
                "rolling_wins": rolling_wins,
                "total": int(info.get("total", 0) or 0),
            })

    blocked = [row for row in source_rows if row["rolling_total"] >= 5 and row["accuracy"] < 0.50]
    source_rows.sort(key=lambda r: (r["accuracy"], -r["rolling_total"], r["pair"]))
    return {
        "worst": source_rows[:limit],
        "blocked_like": blocked[:limit],
        "source": "journal" if use_journal else "pair_accuracy.json",
    }


def _build_execution_quality_snapshot() -> Dict[str, Any]:
    data = _load_execution_quality_state()
    if not isinstance(data, dict) or not data:
        return {}
    pair_rows = []
    for pair, fills in (data.get("pairs") or {}).items():
        if not isinstance(fills, list) or not fills:
            continue
        slippages = [abs(float(f.get("slippage_pips", 0) or 0)) for f in fills]
        fill_count = sum(1 for f in fills if str(f.get("fill_status", "")).lower() == "filled")
        pair_rows.append({
            "pair": pair,
            "samples": len(fills),
            "avg_slippage": round(sum(slippages) / max(len(slippages), 1), 2),
            "fill_rate": round(fill_count / max(len(fills), 1), 2),
        })
    pair_rows.sort(key=lambda r: (-r["avg_slippage"], r["fill_rate"], r["pair"]))
    return {
        "total_fills": int(data.get("total_fills", 0) or 0),
        "total_rejections": int(data.get("total_rejections", 0) or 0),
        "worst": pair_rows[:3],
        "last_updated": data.get("last_updated"),
    }


def _build_calibration_snapshot() -> Dict[str, Any]:
    data = _load_confidence_calibration()
    if not isinstance(data, dict) or not data:
        return {}
    trades = data.get("trade_history") or []
    n_trades = len(trades) if isinstance(trades, list) else 0
    return {
        "n_trades": n_trades,
        "timestamp": data.get("timestamp"),
        "regimes": int((data.get("metadata") or {}).get("n_regimes", 0) or 0),
        "global_platt": bool(((data.get("platt_params") or {}).get("_global"))),
    }


def build_status_diagnostics() -> Dict[str, Any]:
    """Shared diagnostic snapshot for both status renderers."""
    return {
        "scan": _build_scan_diagnostics_snapshot(),
        "funnel": _build_signal_funnel_snapshot(),
        "pair_accuracy": _build_pair_accuracy_snapshot(),
        "execution_quality": _build_execution_quality_snapshot(),
        "calibration": _build_calibration_snapshot(),
        "blocked_pairs": _load_blocked_pairs(),
    }


# ── OANDA account info (best-effort) ─────────────────────────────────────

def _get_oanda_account() -> Optional[Dict[str, Any]]:
    """Fetch live OANDA account summary if credentials available."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env.local")
    except ImportError:
        pass

    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    api_url = os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")

    if not api_key or not account_id:
        return None

    try:
        import urllib.request
        import ssl

        url = f"{api_url}/v3/accounts/{account_id}/summary"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")

        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode())
            return data.get("account", {})
    except Exception:
        return None


def _get_open_trades() -> List[Dict[str, Any]]:
    """Fetch open trades from OANDA."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env.local")
    except ImportError:
        pass

    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    api_url = os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")

    if not api_key or not account_id:
        return []

    try:
        import urllib.request
        import ssl

        url = f"{api_url}/v3/accounts/{account_id}/openTrades"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")

        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode())
            return data.get("trades", [])
    except Exception:
        return []


# ── Display sections ──────────────────────────────────────────────────────

def show_account(account: Optional[Dict[str, Any]]) -> None:
    _section("Account & NAV")
    if account is None:
        print(f"  {C.Y}⚠ OANDA credentials not configured{C._}")
        state = _load_state()
        nav = state.get("nav", state.get("account_nav"))
        if nav:
            print(f"  NAV (cached):  {C.B}${nav:,.2f}{C._}")
        return

    nav = float(account.get("NAV", 0))
    balance = float(account.get("balance", 0))
    unrealized = float(account.get("unrealizedPL", 0))
    margin_used = float(account.get("marginUsed", 0))
    open_count = int(account.get("openTradeCount", 0))

    print(f"  NAV:           {C.B}${nav:,.2f}{C._}")
    print(f"  Balance:       ${balance:,.2f}")
    color = C.G if unrealized >= 0 else C.R
    print(f"  Unrealized PL: {color}${unrealized:,.2f}{C._}")
    print(f"  Margin used:   ${margin_used:,.2f}")
    print(f"  Open trades:   {open_count}")


def show_open_trades(trades: List[Dict[str, Any]]) -> None:
    _section("Open Trades")
    if not trades:
        print(f"  {C.D}No open trades{C._}")
        return

    print(f"  {'Pair':<12} {'Dir':>5} {'Units':>10} {'Price':>10} {'P/L':>10}")
    print(f"  {_hline(49)}")
    for t in trades:
        instrument = t.get("instrument", "?")
        units = float(t.get("currentUnits", 0))
        direction = "LONG" if units > 0 else "SHORT"
        price = float(t.get("price", 0))
        pl = float(t.get("unrealizedPL", 0))
        color = C.G if pl >= 0 else C.R
        print(f"  {instrument:<12} {direction:>5} {abs(units):>10.0f} {price:>10.5f} {color}{pl:>+10.2f}{C._}")


def show_recent_trades(journal: List[Dict[str, Any]]) -> None:
    _section("Last 5 Trades & Win Rate")
    if not journal:
        print(f"  {C.D}No trades in journal{C._}")
        return

    completed = [
        t for t in journal
        if isinstance(t.get("outcome"), dict)
    ]
    wins = sum(
        1 for t in completed
        if bool((t.get("outcome") or {}).get("trade_won", False))
    )
    total = len(completed)
    win_rate = (wins / total * 100) if total > 0 else 0.0

    color = C.G if win_rate >= 50 else C.Y if win_rate >= 35 else C.R
    print(f"  Win rate: {color}{win_rate:.1f}%{C._} ({wins}/{total} trades)")
    print()

    # Last 5 closed trades
    recent = completed[-5:]
    print(f"  {'Pair':<12} {'Dir':>5} {'P/L':>10} {'R:R':>6} {'Date':<12}")
    print(f"  {_hline(47)}")
    for t in reversed(recent):
        outcome = t.get("outcome") or {}
        pair = t.get("pair", "?")
        direction = t.get("direction", "?")
        pnl = float(outcome.get("realized_pl", t.get("pnl", t.get("profit", 0)) or 0))
        rr_ratio = _derive_rr_ratio(t)
        rr = f"{rr_ratio:.1f}:1" if rr_ratio and rr_ratio > 0 else "N/A"
        date = str(
            outcome.get("close_time")
            or outcome.get("exit_time")
            or t.get("exit_time")
            or t.get("entry_time")
            or t.get("timestamp", "")
        )[:10]
        color = C.G if pnl > 0 else C.R if pnl < 0 else C.D
        print(f"  {pair:<12} {direction:>5} {color}{pnl:>+10.2f}{C._} {rr:>6} {date:<12}")


def show_ensemble_health() -> None:
    _section("Model / Ensemble Health")

    # Try to instantiate scanner to check ensemble
    try:
        from src.scanner.config import ScannerConfig
        from src.scanner.engine import Scanner

        config = ScannerConfig()
        config.non_interactive = True
        scanner = Scanner(config=config)

        # Load models to assess health
        scanner._init_modular_ensemble()
        health = scanner._ensemble_health

        available = health.get("available", [])
        missing = health.get("missing", [])
        degraded = health.get("degraded", False)
        penalty = health.get("penalty", 1.0)

        if scanner._ensemble_type == "MultiPairInference":
            status_color = C.Y
            status_label = "FALLBACK (MultiPairInference)"
        elif degraded:
            status_color = C.R
            status_label = "DEGRADED"
        else:
            status_color = C.G
            status_label = "HEALTHY"

        print(f"  Status:    {status_color}{C.B}{status_label}{C._}")
        print(f"  Available: {C.G}{', '.join(available) if available else 'fallback only'}{C._}")
        if missing:
            print(f"  Missing:   {C.Y}{', '.join(missing)}{C._}")
        print(f"  Penalty:   {penalty:.0%} confidence multiplier")
        print(f"  Type:      {scanner._ensemble_type or 'none loaded'}")
    except Exception as e:
        print(f"  {C.Y}⚠ Could not assess ensemble: {e}{C._}")


def show_agent_weights() -> None:
    _section("Agent Weights Status")
    weights = _load_agent_weights()
    if not weights:
        print(f"  {C.Y}⚠ No agent_weights.json found{C._}")
        return

    # Show _global weights
    global_w = weights.get("_global", {})
    if not global_w:
        print(f"  {C.Y}⚠ No _global regime weights{C._}")
        return

    meta = weights.get("_meta", {})
    updated = meta.get("last_updated", "unknown")
    source = meta.get("initialized_from", "unknown")

    print(f"  Source: {source}  |  Updated: {updated[:19]}")
    print()

    # Two-column display
    agents = sorted(global_w.items(), key=lambda x: -x[1])
    mid = (len(agents) + 1) // 2
    col1 = agents[:mid]
    col2 = agents[mid:]

    for i in range(mid):
        left = col1[i] if i < len(col1) else None
        right = col2[i] if i < len(col2) else None

        def _fmt(item):
            if item is None:
                return ""
            name, w = item
            bar_len = int(w * 8)
            bar = "█" * min(bar_len, 15)
            color = C.G if 0.9 <= w <= 1.3 else C.Y
            return f"  {name:<22} {color}{w:.2f}{C._} {C.D}{bar}{C._}"

        line = _fmt(left)
        if right:
            line = f"{line:<55}{_fmt(right)}"
        print(line)


def _infer_streak_from_history(history: List[Any]) -> tuple[int, Optional[bool]]:
    """Infer the current win/loss streak from persisted trade history."""
    last_outcome: Optional[bool] = None
    streak = 0

    for item in reversed(history):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        won = bool(item[1])
        if last_outcome is None:
            last_outcome = won
            streak = 1
            continue
        if won != last_outcome:
            break
        streak += 1

    return streak, last_outcome


def _adaptive_streak_multiplier(
    streak: int,
    won: Optional[bool],
    streak_state: Dict[str, Any],
) -> float:
    """Compute streak multiplier for adaptive position sizing."""
    if streak <= 0 or won is None:
        return 1.0

    cfg = streak_state.get("config", {}) if isinstance(streak_state, dict) else {}
    win_mult = float(cfg.get("win_multiplier", 1.3) or 1.3)
    loss_div = float(cfg.get("loss_divisor", 1.8) or 1.8)
    max_boost = float(cfg.get("max_boost", 2.0) or 2.0)
    min_floor = float(cfg.get("min_floor", 0.3) or 0.3)

    if won:
        return min(win_mult ** streak, max_boost)
    return max(1.0 / (loss_div ** streak), min_floor)


def show_risk_scaler() -> None:
    _section("Risk Scaler State")
    adaptive = _load_adaptive_sizer_state()
    if adaptive:
        history = adaptive.get("trade_history", [])
        if not isinstance(history, list):
            history = []
        streak_state = adaptive.get("streak_tracker", {})
        if not isinstance(streak_state, dict):
            streak_state = {}

        journal_history = _journal_trade_history()
        config = adaptive.get("config", {})
        kelly_window = int(config.get("kelly_window", 50) or 50)
        expected_history = journal_history[-kelly_window:] if journal_history else []
        using_journal = len(expected_history) > len(history)
        if using_journal:
            history = expected_history
            streak_state = {}

        wins = sum(1 for item in history if isinstance(item, (list, tuple)) and len(item) == 2 and bool(item[1]))
        total = sum(1 for item in history if isinstance(item, (list, tuple)) and len(item) == 2)

        consecutive_wins = int(streak_state.get("consecutive_wins", 0) or 0)
        consecutive_losses = int(streak_state.get("consecutive_losses", 0) or 0)
        if consecutive_wins > 0:
            streak = consecutive_wins
            streak_won = True
        elif consecutive_losses > 0:
            streak = consecutive_losses
            streak_won = False
        else:
            streak, streak_won = _infer_streak_from_history(history)

        multiplier = (
            float(streak_state.get("multiplier"))
            if isinstance(streak_state.get("multiplier"), (int, float))
            else _adaptive_streak_multiplier(streak, streak_won, streak_state)
        )

        drawdown_floor = float(config.get("drawdown_floor", 0.0) or 0.0)
        max_drawdown = float(config.get("max_acceptable_drawdown", 0.0) or 0.0)

        if multiplier > 1.0:
            color = C.G
            level = "BOOSTED"
        elif multiplier < 1.0:
            color = C.Y
            level = "REDUCED"
        else:
            color = C.G
            level = "NORMAL"

        streak_label = "flat"
        if streak > 0 and streak_won is not None:
            streak_label = f"{streak}{'W' if streak_won else 'L'}"

        source = "adaptive position sizer"
        if using_journal:
            source = "adaptive position sizer (rehydrated from journal)"
        print(f"  Source: {C.B}{source}{C._}")
        print(f"  Level:  {color}{level}{C._}")
        print(f"  Factor: {multiplier:.2f}x position sizing")
        print(f"  Streak: {streak_label}  |  Recorded trades: {wins}/{total} wins")
        if max_drawdown > 0:
            print(f"  Drawdown: floor={drawdown_floor:.2f}x  |  max acceptable={max_drawdown:.1%}")
        return

    scaler = _load_risk_scaler_state()
    if scaler:
        factor = float(scaler.get("scale_factor", 1.0) or 1.0)
        wins = int(scaler.get("consecutive_wins", 0) or 0)
        losses = int(scaler.get("consecutive_losses", 0) or 0)
        total = int(scaler.get("total_trades", 0) or 0)

        if factor > 1.0:
            color = C.G
            level = "BOOSTED"
        elif factor < 1.0:
            color = C.Y
            level = "REDUCED"
        else:
            color = C.G
            level = "NORMAL"

        print(f"  Source: {C.B}legacy adaptive risk scaler{C._}")
        print(f"  Level:  {color}{level}{C._}")
        print(f"  Factor: {factor:.2f}x position sizing")
        print(f"  Streak: {wins}W/{losses}L  |  Total trades: {total}")
        return

    state = _load_state()
    drawdown = state.get("max_drawdown_pct", state.get("current_drawdown"))
    if drawdown is not None:
        print(f"  Source: {C.D}.claude/state.json fallback{C._}")
        print(f"  Current drawdown: {C.Y}{drawdown}{C._}")
    else:
        print(f"  {C.D}Normal (no scaler state recorded){C._}")


def build_status_risk_snapshot() -> Dict[str, Any]:
    """Compact machine-readable risk scaler summary for rich status rendering."""
    adaptive = _load_adaptive_sizer_state()
    if adaptive:
        history = adaptive.get("trade_history", [])
        if not isinstance(history, list):
            history = []
        streak_state = adaptive.get("streak_tracker", {})
        if not isinstance(streak_state, dict):
            streak_state = {}
        journal_history = _journal_trade_history()
        config = adaptive.get("config", {})
        kelly_window = int(config.get("kelly_window", 50) or 50)
        expected_history = journal_history[-kelly_window:] if journal_history else []
        using_journal = len(expected_history) > len(history)
        if using_journal:
            history = expected_history
            streak_state = {}

        consecutive_wins = int(streak_state.get("consecutive_wins", 0) or 0)
        consecutive_losses = int(streak_state.get("consecutive_losses", 0) or 0)
        if consecutive_wins > 0:
            streak = consecutive_wins
            streak_won = True
        elif consecutive_losses > 0:
            streak = consecutive_losses
            streak_won = False
        else:
            streak, streak_won = _infer_streak_from_history(history)
        multiplier = (
            float(streak_state.get("multiplier"))
            if isinstance(streak_state.get("multiplier"), (int, float))
            else _adaptive_streak_multiplier(streak, streak_won, streak_state)
        )
        level = "NORMAL"
        if multiplier > 1.0:
            level = "BOOSTED"
        elif multiplier < 1.0:
            level = "REDUCED"
        streak_label = "flat"
        if streak > 0 and streak_won is not None:
            streak_label = f"{streak}{'W' if streak_won else 'L'}"
        source = "adaptive position sizer (rehydrated from journal)" if using_journal else "adaptive position sizer"
        return {
            "source": source,
            "level": level,
            "factor": multiplier,
            "streak": streak_label,
        }

    scaler = _load_risk_scaler_state()
    if scaler:
        factor = float(scaler.get("scale_factor", 1.0) or 1.0)
        level = "NORMAL"
        if factor > 1.0:
            level = "BOOSTED"
        elif factor < 1.0:
            level = "REDUCED"
        return {
            "source": "legacy adaptive risk scaler",
            "level": level,
            "factor": factor,
            "streak": f"{int(scaler.get('consecutive_wins', 0) or 0)}W/{int(scaler.get('consecutive_losses', 0) or 0)}L",
        }

    return {}


def show_blocked_pairs() -> None:
    _section("Blocked Pairs")
    blocked = _load_blocked_pairs()
    state = _load_state()

    # Also check accuracy gate blocks
    accuracy_blocks = state.get("accuracy_blocked_pairs", {})

    if not blocked and not accuracy_blocks:
        print(f"  {C.G}No pairs blocked{C._}")
        return

    if blocked:
        for p in blocked:
            print(f"  {C.R}✗ {p}{C._}")

    if accuracy_blocks:
        for pair, info in accuracy_blocks.items():
            reason = info if isinstance(info, str) else str(info)
            print(f"  {C.Y}⚠ {pair} — {reason}{C._}")


def show_diagnostics_dashboard() -> None:
    _section("Diagnostics Dashboard")
    diag = build_status_diagnostics()
    scan = diag.get("scan", {})
    funnel = diag.get("funnel", {})
    pair_accuracy = diag.get("pair_accuracy", {})
    execution_quality = diag.get("execution_quality", {})
    calibration = diag.get("calibration", {})

    if scan:
        state_color = C.R if scan.get("stalled") else C.G
        state_label = "STALLED" if scan.get("stalled") else "ACTIVE"
        print(f"  Scan health: {state_color}{state_label}{C._}")
        if scan.get("summary"):
            print(f"  Summary: {scan['summary']}")
        gp = scan.get("gate_pass_rate_pct")
        dc = scan.get("pairs_with_direction_pct")
        ac = scan.get("avg_confidence")
        if gp is not None or dc is not None or ac is not None:
            signal_line = (
                f"  Signals: gate-pass={gp if gp is not None else '-'}%  |  "
                f"direction={dc if dc is not None else '-'}%"
            )
            if isinstance(ac, (int, float)):
                signal_line += f"  |  avg_conf={ac:.3f}"
            print(signal_line)
        if scan.get("scans_since_last_trade") is not None:
            print(
                f"  Activity: scans since last tradeable = "
                f"{scan.get('scans_since_last_trade')}  |  total scans = {scan.get('total_scans', '-')}"
            )
        print()

    if funnel:
        print("  Gate funnel:")
        print(
            f"  confidence={funnel.get('passed_confidence_pct', 0):.1f}%  |  "
            f"momentum={funnel.get('passed_momentum_pct', 0):.1f}%  |  "
            f"risk={funnel.get('passed_risk_pct', 0):.1f}%  |  "
            f"tradeable={funnel.get('tradeable_pct', 0):.1f}%"
        )
        print(
            f"  Dominant kill stage: {str(funnel.get('dominant_kill_stage', 'none')).upper()} "
            f"(last {funnel.get('window', 0)} scans)"
        )
        print()

    worst_pairs = pair_accuracy.get("worst", [])
    if worst_pairs:
        print("  Pair accuracy watchlist:")
        for row in worst_pairs:
            color = C.G if row["accuracy"] >= 0.55 else C.Y if row["accuracy"] >= 0.50 else C.R
            print(
                f"  {row['pair']:<10} {color}{row['accuracy'] * 100:>5.1f}%{C._}  "
                f"({row['rolling_wins']}/{row['rolling_total']} rolling)"
            )
        print()

    worst_exec = execution_quality.get("worst", [])
    if worst_exec or calibration:
        print("  Model / execution diagnostics:")
        if worst_exec:
            top = worst_exec[0]
            print(
                f"  Worst execution pair: {top['pair']}  |  "
                f"slip={top['avg_slippage']:.2f} pips  |  fill={top['fill_rate'] * 100:.0f}%  "
                f"({top['samples']} samples)"
            )
        if calibration:
            fitted = "yes" if calibration.get("global_platt") else "no"
            print(
                f"  Calibration: trades={calibration.get('n_trades', 0)}  |  "
                f"global fit={fitted}  |  regimes={calibration.get('regimes', 0)}"
            )
        blocked = diag.get("blocked_pairs", [])
        if blocked:
            print(f"  Blocked pairs: {', '.join(blocked[:6])}")
    if not any([scan, funnel, worst_pairs, worst_exec, calibration]):
        print(f"  {C.D}No diagnostic state recorded yet{C._}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{C.B}{'═' * 70}{C._}")
    print(f"{C.B}  🤖 BUDDY ML ENGINE — System Health Dashboard{C._}")
    print(f"{C.B}  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}{C._}")
    print(f"{C.B}{'═' * 70}{C._}")

    account = _get_oanda_account()
    trades = _get_open_trades()

    show_account(account)
    show_open_trades(trades)
    show_recent_trades(_load_journal())
    show_diagnostics_dashboard()
    show_ensemble_health()
    show_agent_weights()
    show_risk_scaler()
    show_blocked_pairs()

    print(f"\n{C.D}{'─' * 70}{C._}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
