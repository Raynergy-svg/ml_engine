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


def _load_state() -> Dict[str, Any]:
    return _load_json(_ROOT / ".claude" / "state.json") or {}


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

    # Win rate
    completed = [t for t in journal if "pnl" in t or "profit" in t or "result" in t]
    wins = sum(1 for t in completed if float(t.get("pnl", t.get("profit", 0))) > 0)
    total = len(completed)
    win_rate = (wins / total * 100) if total > 0 else 0.0

    color = C.G if win_rate >= 50 else C.Y if win_rate >= 35 else C.R
    print(f"  Win rate: {color}{win_rate:.1f}%{C._} ({wins}/{total} trades)")
    print()

    # Last 5
    recent = journal[-5:]
    print(f"  {'Pair':<12} {'Dir':>5} {'P/L':>10} {'R:R':>6} {'Date':<12}")
    print(f"  {_hline(47)}")
    for t in reversed(recent):
        pair = t.get("pair", "?")
        direction = t.get("direction", "?")
        pnl = float(t.get("pnl", t.get("profit", 0)))
        sl = float(t.get("sl_pips", 1))
        tp = float(t.get("tp_pips", 0))
        rr = f"{tp / sl:.1f}:1" if sl > 0 else "N/A"
        date = str(t.get("exit_time", t.get("entry_time", "")))[:10]
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


def show_risk_scaler() -> None:
    _section("Risk Scaler State")
    state = _load_state()

    scaler = state.get("risk_scaler", {})
    if not scaler:
        # Try to infer from other state fields
        drawdown = state.get("max_drawdown_pct", state.get("current_drawdown"))
        if drawdown:
            print(f"  Current drawdown: {C.Y}{drawdown}%{C._}")
        else:
            print(f"  {C.D}Normal (no scaler state recorded){C._}")
        return

    level = scaler.get("level", "normal")
    factor = scaler.get("factor", 1.0)
    reason = scaler.get("reason", "")

    color = C.G if level == "normal" else C.Y if level == "reduced" else C.R
    print(f"  Level:  {color}{level.upper()}{C._}")
    print(f"  Factor: {factor:.2f}x position sizing")
    if reason:
        print(f"  Reason: {reason}")


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
    show_ensemble_health()
    show_agent_weights()
    show_risk_scaler()
    show_blocked_pairs()

    print(f"\n{C.D}{'─' * 70}{C._}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
