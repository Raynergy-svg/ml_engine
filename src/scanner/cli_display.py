"""
Unified CLI Display Module for Buddy ML Engine.

Replaces fragmented print statements with a cohesive Rich-based
terminal experience. All functions print directly to the console.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val: Any, fmt: str = "", fallback: str = "-") -> str:
    """Safely format a value, returning fallback for None/zero where appropriate."""
    if val is None:
        return fallback
    try:
        if fmt:
            return format(val, fmt)
        return str(val)
    except (ValueError, TypeError):
        return fallback


def _pct(val: Optional[float], fallback: str = "-") -> str:
    if val is None:
        return fallback
    return f"{val * 100:.0f}%"


def _dollars(val: Optional[float], signed: bool = False) -> str:
    if val is None:
        return "-"
    if signed:
        if val > 0:
            return f"+${val:,.2f}"
        elif val < 0:
            return f"-${abs(val):,.2f}"
    return f"${val:,.2f}"


def _pnl_color(val: float) -> str:
    if val > 0:
        return "green"
    if val < 0:
        return "red"
    return "dim"


def _gate_char(passed: Optional[bool]) -> str:
    if passed is None:
        return "?"
    return "[green]Y[/green]" if passed else "[red]X[/red]"


def _direction_arrow(direction: str) -> Text:
    d = (direction or "").upper()
    if d == "LONG":
        t = Text("\u25b2 LONG", style="green bold")
    elif d == "SHORT":
        t = Text("\u25bc SHORT", style="red bold")
    else:
        t = Text("- HOLD", style="dim")
    return t


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h = s // 3600
    m = (s % 3600) // 60
    if h < 24:
        return f"{h}h {m:02d}m"
    d = h // 24
    h = h % 24
    return f"{d}d {h}h"


def _format_pair(pair: str) -> str:
    """EUR_USD -> EUR/USD"""
    return pair.replace("_", "/") if pair else "-"


def _rr_ratio(sl: Optional[float], tp: Optional[float]) -> str:
    if not sl or not tp or sl <= 0:
        return "-"
    return f"{tp / sl:.2f}"


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Function 1: render_scan_result
# ---------------------------------------------------------------------------

def render_scan_result(
    scan_result: Any,
    config: Any = None,
    cycle_num: int = 0,
    elapsed_s: float = 0.0,
) -> None:
    """Render a full scan cycle result table inside a bordered panel.

    Args:
        scan_result: ScanResult dataclass (or any object with .analyses, .tradeable, etc.)
        config: ScannerConfig (optional, for context)
        cycle_num: Current scan cycle number
        elapsed_s: Time taken for this scan in seconds
    """
    analyses: list = getattr(scan_result, "analyses", []) or []
    tradeable: list = getattr(scan_result, "tradeable", []) or []
    nav = getattr(scan_result, "account_equity", None)

    # Build table
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Pair", style="bold white", min_width=9)
    table.add_column("Signal", min_width=8)
    table.add_column("Conf", justify="right", min_width=4)
    table.add_column("Gates", min_width=9)
    table.add_column("Agents", justify="center", min_width=6)
    table.add_column("SL", justify="right", min_width=4)
    table.add_column("TP", justify="right", min_width=4)
    table.add_column("R:R", justify="right", min_width=5)
    table.add_column("Status", min_width=8)

    # Sort: tradeable first (by score desc), then non-tradeable
    sorted_analyses = sorted(
        analyses,
        key=lambda a: (a.is_tradeable, getattr(a, "overall_score", 0)),
        reverse=True,
    )

    # Kill reason counters
    kill_counts: Dict[str, int] = {}

    for a in sorted_analyses:
        is_t = getattr(a, "is_tradeable", False)
        direction = getattr(a, "direction", "HOLD") or "HOLD"
        conf = getattr(a, "confidence", 0) or 0
        err = getattr(a, "error", None)

        # Gate flags
        c_pass = getattr(a, "confidence_passed", False) or getattr(a, "confidence_gate_passed", False)
        m_pass = getattr(a, "momentum_passed", False) or getattr(a, "momentum_gate_passed", False)
        r_pass = getattr(a, "risk_passed", False) or getattr(a, "risk_gate_passed", False)

        gate_str = f"C{_gate_char(c_pass)} M{_gate_char(m_pass)} R{_gate_char(r_pass)}"

        # Agents
        av = getattr(a, "agent_votes", 0) or 0
        at_ = getattr(a, "agent_total", 0) or 0
        agent_str = f"{av}/{at_}" if at_ > 0 else "-"

        sl = getattr(a, "sl_pips", None)
        tp = getattr(a, "tp_pips", None)

        if is_t:
            sig = _direction_arrow(direction)
            status = Text("READY", style="green bold")
            sl_str = f"{sl:.0f}" if sl else "-"
            tp_str = f"{tp:.0f}" if tp else "-"
            rr_str = _rr_ratio(sl, tp)
        elif err:
            sig = Text("ERROR", style="red")
            status = Text("ERROR", style="red")
            sl_str = tp_str = rr_str = "-"
        elif direction in ("LONG", "SHORT"):
            sig = _direction_arrow(direction)
            status = Text("WATCH", style="yellow")
            sl_str = tp_str = rr_str = "-"
            # Count kill reasons
            kr = getattr(a, "kill_reason", None)
            if kr:
                kill_counts[kr] = kill_counts.get(kr, 0) + 1
        else:
            sig = Text("- HOLD", style="dim")
            status = Text("HOLD", style="dim")
            sl_str = tp_str = rr_str = "-"

        conf_str = f"{conf * 100:.0f}%" if conf else "-"

        table.add_row(
            _format_pair(getattr(a, "pair", "")),
            sig,
            conf_str,
            Text.from_markup(gate_str),
            agent_str,
            sl_str,
            tp_str,
            rr_str,
            status,
        )

    # Footer line
    total = len(analyses)
    t_count = len(tradeable)
    kills_parts = []
    for k, v in sorted(kill_counts.items(), key=lambda x: -x[1]):
        label = k.replace("_gate", "").replace("_", " ")
        kills_parts.append(f"{label}:{v}")
    kills_str = "  ".join(kills_parts) if kills_parts else "none"
    nav_str = _dollars(nav) if nav else "-"
    elapsed_str = f"{elapsed_s:.1f}s" if elapsed_s else "-"

    footer = Text()
    footer.append(f"  Tradeable: {t_count}/{total}", style="bold white")
    footer.append(f"  Kills: {kills_str}", style="dim")
    footer.append(f"  NAV: {nav_str}", style="bold white")
    footer.append(f"  {elapsed_str}", style="dim")

    # Panel title
    title = Text()
    title.append(f" SCAN #{cycle_num} ", style="bold cyan")
    subtitle = Text(_now_utc_str(), style="dim")

    panel = Panel(
        table,
        title=title,
        subtitle=footer,
        border_style="cyan",
        expand=True,
        padding=(0, 1),
    )
    _console.print(panel)


# ---------------------------------------------------------------------------
# Function 2: render_trade_execution
# ---------------------------------------------------------------------------

def render_trade_execution(
    pair: str,
    direction: str,
    confidence: float,
    lots: float,
    sl_pips: float,
    tp_pips: float,
    rr_ratio: float,
    regime: str = "NORMAL",
    risk_pct: float = 0.0,
    entry_price: float = 0.0,
    trade_id: Optional[str] = None,
) -> None:
    """Render a compact 3-line trade execution ticket."""
    d = (direction or "").upper()
    arrow = "\u25b2" if d == "LONG" else "\u25bc"
    border_color = "green" if d == "LONG" else "red"
    dir_style = "green bold" if d == "LONG" else "red bold"

    line1 = Text()
    line1.append(f"  {arrow} {d} ", style=dir_style)
    line1.append(_format_pair(pair), style="bold white")
    line1.append(f" @ {entry_price}", style="bold white")

    line2 = Text()
    line2.append(f"  Lots: {lots:.2f}", style="white")
    line2.append(f"  SL: {sl_pips:.0f}p", style="white")
    line2.append(f"  TP: {tp_pips:.0f}p", style="white")
    line2.append(f"  R:R: {rr_ratio:.2f}", style="white")
    line2.append(f"  Risk: {risk_pct * 100:.1f}%", style="white")
    line2.append(f"  Conf: {confidence * 100:.0f}%", style="white")

    line3 = Text()
    line3.append(f"  Regime: {regime or 'N/A'}", style="dim")
    if trade_id:
        line3.append(f"  Order: #{trade_id}", style="dim")

    body = Text()
    body.append_text(line1)
    body.append("\n")
    body.append_text(line2)
    body.append("\n")
    body.append_text(line3)

    panel = Panel(
        body,
        title=Text(" TRADE ", style=f"bold {border_color}"),
        border_style=border_color,
        expand=True,
        padding=(0, 0),
    )
    _console.print(panel)


# ---------------------------------------------------------------------------
# Function 3: render_watch_header
# ---------------------------------------------------------------------------

def render_watch_header(
    nav: Optional[float] = None,
    open_count: Optional[int] = None,
    unrealized_pnl: Optional[float] = None,
    session: str = "",
    cycle_num: int = 0,
    uptime_s: float = 0.0,
    next_scan_s: float = 0.0,
    account_error: str = "",
    mode: str = "WATCH",
) -> None:
    """Render a compact watch-mode status bar.

    Args:
        mode: Display mode label — "WATCH" (continuous scan) or "SUPERVISE" (supervision mode)
    """
    uptime_str = _format_duration(uptime_s)

    pnl_value = float(unrealized_pnl) if unrealized_pnl is not None else None
    pnl_color = _pnl_color(pnl_value or 0.0) if pnl_value is not None else "dim"
    pnl_sign = "+" if pnl_value is not None and pnl_value >= 0 else ""
    nav_pct = ""
    if nav is not None and nav > 0 and pnl_value is not None and pnl_value != 0:
        pct = (pnl_value / nav) * 100
        nav_pct = f" ({pnl_sign}{pct:.1f}%)"

    next_str = _format_duration(next_scan_s)

    top = Text()
    top.append("\u2501\u2501 ", style="cyan bold")
    top.append(f"BUDDY \u00b7 {mode} ", style="cyan bold")
    # Pad to fill
    top.append("\u2501" * 40, style="cyan")
    top.append(f" #{cycle_num}", style="dim")
    top.append(f" \u00b7 {uptime_str} ", style="dim")
    top.append("\u2501\u2501", style="cyan bold")

    mid = Text()
    mid.append(f"  NAV {_dollars(nav)}", style="bold white")
    mid.append(nav_pct, style=pnl_color)
    mid.append(f"  Open {_safe(open_count)}", style="white")
    pnl_display = _dollars(abs(pnl_value)) if pnl_value is not None else "-"
    mid.append(f"  P/L {pnl_sign}{pnl_display}", style=pnl_color)
    if session:
        mid.append(f"  Session {session}", style="white")
    if account_error:
        mid.append(f"  Account {account_error}", style="yellow")
    mid.append(f"  Next {next_str}", style="dim")

    bottom = Text("\u2501" * 76, style="cyan")

    _console.print(top)
    _console.print(mid)
    _console.print(bottom)


# ---------------------------------------------------------------------------
# Function 4: render_trade_closed
# ---------------------------------------------------------------------------

def render_trade_closed(
    pair: str,
    direction: str,
    won: bool,
    pnl_pips: float = 0.0,
    pnl_dollars: float = 0.0,
    exit_reason: str = "",
    duration_s: float = 0.0,
    weight_changes: Optional[Dict[str, float]] = None,
    session_wins: int = 0,
    session_losses: int = 0,
    session_avg_pnl: float = 0.0,
) -> None:
    """Render a trade closure summary with RL weight changes."""
    border_color = "green" if won else "red"
    result_str = "WIN" if won else "LOSS"
    result_style = "green bold" if won else "red bold"
    d = (direction or "").upper()
    arrow = "\u25b2" if d == "LONG" else "\u25bc"

    pnl_sign = "+" if pnl_pips >= 0 else ""
    dur_str = _format_duration(duration_s)

    # Line 1: pair, direction, result, P/L, exit reason, duration
    line1 = Text()
    line1.append(f"  {_format_pair(pair)} ", style="bold white")
    line1.append(f"{arrow} {d}  ", style="green bold" if d == "LONG" else "red bold")
    line1.append(result_str, style=result_style)
    line1.append(f"  {pnl_sign}{pnl_pips:.1f}p", style=border_color)
    line1.append(f"  {_dollars(pnl_dollars, signed=True)}", style=border_color)
    if exit_reason:
        line1.append(f"  {exit_reason}", style="dim")
    line1.append(f" \u00b7 {dur_str}", style="dim")

    # Line 2: Weight changes
    line2 = Text("  Weights: ", style="dim")
    if weight_changes:
        parts = []
        for agent, delta in sorted(weight_changes.items(), key=lambda x: -abs(x[1])):
            sign = "+" if delta >= 0 else ""
            color = "green" if delta > 0 else "red"
            parts.append(f"[{color}]{agent} {sign}{delta:.3f}[/{color}]")
        line2 = Text.from_markup("  Weights: " + "  ".join(parts[:5]))
    else:
        line2.append("none", style="dim")

    # Line 3: Session stats
    total = session_wins + session_losses
    wr = (session_wins / total * 100) if total > 0 else 0
    line3 = Text()
    line3.append(f"  Session: {session_wins}W {session_losses}L", style="white")
    line3.append(f" \u00b7 {wr:.0f}%", style="white")
    line3.append(f" \u00b7 {_dollars(session_avg_pnl, signed=True)}/trade", style="white")

    body = Text()
    body.append_text(line1)
    body.append("\n")
    body.append_text(line2)
    body.append("\n")
    body.append_text(line3)

    panel = Panel(
        body,
        title=Text(f" CLOSED ", style=f"bold {border_color}"),
        border_style=border_color,
        expand=True,
        padding=(0, 0),
    )
    _console.print(panel)


# ---------------------------------------------------------------------------
# Function 5: render_health_summary
# ---------------------------------------------------------------------------

def render_health_summary(
    health_score: float = 0.0,
    gate_kills: Optional[Dict[str, float]] = None,
    module_statuses: Optional[Dict[str, bool]] = None,
    win_rate: float = 0.0,
    trades_count: int = 0,
    rss_mb: float = 0.0,
    cycle_avg_s: float = 0.0,
    expectancy: float = 0.0,
    sharpe: float = 0.0,
    cycle_num: int = 0,
) -> None:
    """Render system health dashboard."""
    # Score label
    if health_score >= 80:
        score_label = "GOOD"
        score_style = "green bold"
    elif health_score >= 50:
        score_label = "FAIR"
        score_style = "yellow bold"
    else:
        score_label = "POOR"
        score_style = "red bold"

    # Line 1: Score, RSS, Cycle
    line1 = Text()
    line1.append(f"  Score: {health_score:.0f}/100 ", style="bold white")
    line1.append(score_label, style=score_style)
    line1.append(f"  RSS: {rss_mb:.0f}MB", style="dim")
    line1.append(f"  Cycle: {cycle_avg_s:.2f}s", style="dim")

    # Line 2: Gate kill distribution
    line2 = Text("  Gates: ", style="white")
    if gate_kills:
        parts = []
        for gate, pct in sorted(gate_kills.items(), key=lambda x: -x[1]):
            label = gate.replace("_gate", "").replace("_", " ")
            parts.append(f"{label} {pct:.0f}%")
        line2.append(" \u00b7 ".join(parts), style="dim")
    else:
        line2.append("N/A", style="dim")

    # Line 3: Win stats
    wins = int(trades_count * win_rate) if trades_count else 0
    line3 = Text()
    line3.append(f"  Win: {win_rate * 100:.0f}%", style="bold white")
    line3.append(f" ({wins}/{trades_count})", style="dim")
    line3.append(f"  Expectancy {_dollars(expectancy, signed=True)}", style="white")
    line3.append(f"  Sharpe {sharpe:.2f}", style="white")

    # Line 4: Module statuses
    line4 = Text("  ", style="white")
    if module_statuses:
        for name, ok in module_statuses.items():
            if ok:
                line4.append(f"\u25cf {name}  ", style="green")
            else:
                line4.append(f"\u25cb {name}  ", style="yellow")
    else:
        line4.append("No module data", style="dim")

    body = Text()
    body.append_text(line1)
    body.append("\n")
    body.append_text(line2)
    body.append("\n")
    body.append_text(line3)
    body.append("\n")
    body.append_text(line4)

    title_text = f" HEALTH #{cycle_num} " if cycle_num else " HEALTH "
    panel = Panel(
        body,
        title=Text(title_text, style="bold cyan"),
        border_style="cyan",
        expand=True,
        padding=(0, 0),
    )
    _console.print(panel)


# ---------------------------------------------------------------------------
# Function 6: render_status
# ---------------------------------------------------------------------------

def render_status(
    nav: float = 0.0,
    balance: float = 0.0,
    unrealized: float = 0.0,
    positions: Optional[List[Dict[str, Any]]] = None,
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    agent_weights: Optional[Dict[str, float]] = None,
    model_health: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    risk_scaler: Optional[Dict[str, Any]] = None,
    blocked_pairs: Optional[List[str]] = None,
    background_activity: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a full status screen with stacked panels."""

    # --- Panel 1: Account ---
    acct = Text()
    acct.append(f"  NAV: {_dollars(nav)}", style="bold white")
    acct.append(f"  Balance: {_dollars(balance)}", style="white")
    pnl_c = _pnl_color(unrealized)
    acct.append(f"  Unrealized: {_dollars(unrealized, signed=True)}", style=pnl_c)

    _console.print(Panel(
        acct,
        title=Text(" ACCOUNT ", style="bold cyan"),
        border_style="cyan",
        padding=(0, 0),
    ))

    # --- Panel 2: Open Positions ---
    if positions:
        pos_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        pos_table.add_column("Pair", style="bold white")
        pos_table.add_column("Dir", min_width=5)
        pos_table.add_column("Lots", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("P/L", justify="right")
        pos_table.add_column("SL", justify="right")
        pos_table.add_column("TP", justify="right")

        for p in positions:
            d = (p.get("direction", "") or "").upper()
            d_text = Text(d, style="green" if d == "LONG" else "red")
            pnl = p.get("pnl", 0) or 0
            pnl_text = Text(_dollars(pnl, signed=True), style=_pnl_color(pnl))
            pos_table.add_row(
                _format_pair(p.get("pair", "")),
                d_text,
                _safe(p.get("lots"), ".2f"),
                _safe(p.get("entry_price"), ".5f"),
                pnl_text,
                _safe(p.get("sl_pips"), ".0f"),
                _safe(p.get("tp_pips"), ".0f"),
            )

        _console.print(Panel(
            pos_table,
            title=Text(f" POSITIONS ({len(positions)}) ", style="bold cyan"),
            border_style="cyan",
            padding=(0, 0),
        ))
    else:
        _console.print(Panel(
            Text("  No open positions", style="dim"),
            title=Text(" POSITIONS (0) ", style="bold cyan"),
            border_style="cyan",
            padding=(0, 0),
        ))

    # --- Panel 3: Recent Trades ---
    if recent_trades:
        trades_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        trades_table.add_column("Pair", style="bold white")
        trades_table.add_column("Result")
        trades_table.add_column("P/L", justify="right")
        trades_table.add_column("Pips", justify="right")
        trades_table.add_column("Exit")
        trades_table.add_column("Time", style="dim")

        for t in recent_trades[:10]:
            won = t.get("won", False)
            result_text = Text("WIN" if won else "LOSS", style="green bold" if won else "red bold")
            pnl = t.get("pnl_dollars", 0) or 0
            pnl_text = Text(_dollars(pnl, signed=True), style=_pnl_color(pnl))
            pips = t.get("pnl_pips", 0) or 0
            pip_sign = "+" if pips >= 0 else ""
            trades_table.add_row(
                _format_pair(t.get("pair", "")),
                result_text,
                pnl_text,
                f"{pip_sign}{pips:.1f}",
                t.get("exit_reason", "-"),
                t.get("closed_at", "-"),
            )

        _console.print(Panel(
            trades_table,
            title=Text(f" RECENT TRADES ({len(recent_trades)}) ", style="bold cyan"),
            border_style="cyan",
            padding=(0, 0),
        ))

    # --- Panel 4: Diagnostics ---
    diag = diagnostics or {}
    scan = diag.get("scan", {}) if isinstance(diag, dict) else {}
    funnel = diag.get("funnel", {}) if isinstance(diag, dict) else {}
    pair_accuracy = diag.get("pair_accuracy", {}) if isinstance(diag, dict) else {}
    execution_quality = diag.get("execution_quality", {}) if isinstance(diag, dict) else {}
    calibration = diag.get("calibration", {}) if isinstance(diag, dict) else {}

    if any([scan, funnel, pair_accuracy, execution_quality, calibration]):
        body = Text()
        if scan:
            stalled = bool(scan.get("stalled", False))
            body.append("  Scan: ", style="white")
            body.append("STALLED", style="red bold" if stalled else "green bold")
            gp = scan.get("gate_pass_rate_pct")
            dc = scan.get("pairs_with_direction_pct")
            ac = scan.get("avg_confidence")
            scan_line = (
                f"  Gate-pass {gp if gp is not None else '-'}%  "
                f"Direction {dc if dc is not None else '-'}%"
            )
            if isinstance(ac, (int, float)):
                scan_line += f"  Avg conf {ac:.3f}"
            body.append(scan_line, style="white")
            body.append("\n")
            summary = str(scan.get("summary", "") or "").strip()
            if summary:
                body.append(f"  {summary}", style="dim")
                body.append("\n")
        if funnel:
            body.append(
                f"  Funnel: conf {funnel.get('passed_confidence_pct', 0):.1f}%  "
                f"mom {funnel.get('passed_momentum_pct', 0):.1f}%  "
                f"risk {funnel.get('passed_risk_pct', 0):.1f}%  "
                f"tradeable {funnel.get('tradeable_pct', 0):.1f}%  ",
                style="white",
            )
            body.append(
                f"kill={str(funnel.get('dominant_kill_stage', 'none')).upper()}",
                style="yellow" if str(funnel.get("dominant_kill_stage", "none")) != "none" else "dim",
            )
            body.append("\n")
        worst_pairs = pair_accuracy.get("worst", []) if isinstance(pair_accuracy, dict) else []
        if worst_pairs:
            body.append("  Accuracy: ", style="white")
            for idx, row in enumerate(worst_pairs[:3]):
                acc = float(row.get("accuracy", 0) or 0)
                style = "green" if acc >= 0.55 else "yellow" if acc >= 0.50 else "red"
                body.append(
                    f"{row.get('pair', '?')} {acc * 100:.1f}% ({row.get('rolling_wins', 0)}/{row.get('rolling_total', 0)})",
                    style=style,
                )
                if idx < min(len(worst_pairs), 3) - 1:
                    body.append("  ", style="dim")
            body.append("\n")
        worst_exec = execution_quality.get("worst", []) if isinstance(execution_quality, dict) else []
        if worst_exec or calibration:
            body.append("  Quality: ", style="white")
            if worst_exec:
                top = worst_exec[0]
                body.append(
                    f"{top.get('pair', '?')} slip {float(top.get('avg_slippage', 0) or 0):.2f}p "
                    f"fill {float(top.get('fill_rate', 0) or 0) * 100:.0f}%",
                    style="white",
                )
            if calibration:
                if worst_exec:
                    body.append("  |  ", style="dim")
                body.append(
                    f"calibration trades {int(calibration.get('n_trades', 0) or 0)}",
                    style="white",
                )

        _console.print(Panel(
            body,
            title=Text(" DIAGNOSTICS ", style="bold cyan"),
            border_style="cyan",
            padding=(0, 0),
        ))

    # --- Panel 5: Agent Weights ---
    if agent_weights:
        weights_text = Text("  ", style="white")
        sorted_agents = sorted(agent_weights.items(), key=lambda x: -x[1])
        for i, (agent, w) in enumerate(sorted_agents):
            if w >= 0.10:
                style = "green"
            elif w >= 0.05:
                style = "white"
            else:
                style = "dim"
            weights_text.append(f"{agent}: {w:.3f}", style=style)
            if i < len(sorted_agents) - 1:
                weights_text.append("  ", style="dim")

        _console.print(Panel(
            weights_text,
            title=Text(" AGENT WEIGHTS ", style="bold cyan"),
            border_style="cyan",
            padding=(0, 0),
        ))

    # --- Panel 6: Risk / Blocks ---
    if risk_scaler or blocked_pairs:
        body = Text()
        if risk_scaler:
            body.append("  Risk scaler: ", style="white")
            body.append(str(risk_scaler.get("source", "unknown")), style="dim")
            body.append("  ", style="dim")
            body.append(str(risk_scaler.get("level", "NORMAL")), style="green bold" if str(risk_scaler.get("level", "NORMAL")) == "NORMAL" else "yellow bold")
            factor = risk_scaler.get("factor")
            if factor is not None:
                body.append(f"  {float(factor):.2f}x", style="white")
            streak = risk_scaler.get("streak")
            if streak:
                body.append(f"  {streak}", style="dim")
            body.append("\n")
        if blocked_pairs:
            body.append("  Blocked: ", style="white")
            body.append(", ".join(blocked_pairs[:6]), style="yellow" if blocked_pairs else "dim")
        _console.print(Panel(
            body,
            title=Text(" RISK / BLOCKS ", style="bold cyan"),
            border_style="cyan",
            padding=(0, 0),
        ))

    # --- Panel 7: Model Health ---
    if model_health:
        health_text = Text("  ", style="white")
        for name, info in model_health.items():
            if isinstance(info, dict):
                ok = info.get("healthy", info.get("ok", True))
                dot = "\u25cf" if ok else "\u25cb"
                color = "green" if ok else "yellow"
                health_text.append(f"{dot} {name}", style=color)
                acc = info.get("accuracy")
                if acc is not None:
                    health_text.append(f" ({acc:.0%})", style="dim")
                health_text.append("  ", style="dim")
            else:
                ok = bool(info)
                dot = "\u25cf" if ok else "\u25cb"
                color = "green" if ok else "yellow"
                health_text.append(f"{dot} {name}  ", style=color)

        _console.print(Panel(
            health_text,
            title=Text(" MODEL HEALTH ", style="bold cyan"),
            border_style="cyan",
            padding=(0, 0),
        ))

    # --- Panel 8: Background Activity ---
    if background_activity:
        active = background_activity.get("active", []) if isinstance(background_activity, dict) else []
        history = background_activity.get("history", []) if isinstance(background_activity, dict) else []
        lines = Text()
        if active:
            for item in active[:4]:
                lines.append(f"  RUNNING  {item.get('title', 'Background task')}", style="green bold")
                summary = item.get("summary") or item.get("source") or "-"
                lines.append(f"  {summary}", style="dim")
                meta = item.get("metadata") or {}
                if meta.get("pid"):
                    lines.append(f"  pid={meta.get('pid')}", style="dim")
                lines.append("\n")
        elif history:
            item = history[0]
            status = str(item.get("status", "completed")).upper()
            style = "green bold" if status == "COMPLETED" else "yellow bold" if status == "CANCELED" else "red bold"
            lines.append(f"  {status}  {item.get('title', 'Background task')}", style=style)
            summary = item.get("summary") or item.get("error") or "-"
            lines.append(f"  {summary}", style="dim")
        else:
            lines.append("  No background activity tracked yet", style="dim")
        _console.print(Panel(
            lines,
            title=Text(" BACKGROUND ACTIVITY ", style="bold cyan"),
            border_style="cyan",
            padding=(0, 0),
        ))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataclasses import dataclass, field as dc_field

    # Minimal mock of PairAnalysis for demo
    @dataclass
    class _MockPair:
        pair: str = ""
        direction: str = "HOLD"
        confidence: float = 0.0
        gates_passed: bool = False
        confidence_passed: bool = False
        momentum_passed: bool = False
        risk_passed: bool = False
        agent_votes: int = 0
        agent_total: int = 0
        sl_pips: float = 0.0
        tp_pips: float = 0.0
        overall_score: float = 0.0
        error: Optional[str] = None
        kill_reason: Optional[str] = None
        is_tradeable: bool = False
        agent_passed: bool = False
        execution_quality_passed: bool = True
        model_disagreement: float = 0.0
        volatility_regime: str = "NORMAL"
        blocked_by_circuit_breaker: bool = False

    @dataclass
    class _MockScan:
        analyses: list = dc_field(default_factory=list)
        account_equity: float = 0.0

        @property
        def tradeable(self):
            return [a for a in self.analyses if a.is_tradeable]

    _console.print()
    _console.rule("[bold cyan]CLI Display Smoke Test[/bold cyan]")
    _console.print()

    # --- 1. Scan Result ---
    scan = _MockScan(
        analyses=[
            _MockPair(
                pair="EUR_USD", direction="SHORT", confidence=0.58,
                gates_passed=True, confidence_passed=True, momentum_passed=True,
                risk_passed=True, agent_votes=3, agent_total=3,
                sl_pips=20, tp_pips=30, overall_score=0.72,
                is_tradeable=True, agent_passed=True, volatility_regime="NORMAL",
            ),
            _MockPair(
                pair="USD_JPY", direction="LONG", confidence=0.74,
                gates_passed=True, confidence_passed=True, momentum_passed=True,
                risk_passed=True, agent_votes=3, agent_total=3,
                sl_pips=19, tp_pips=28, overall_score=0.81,
                is_tradeable=True, agent_passed=True, volatility_regime="NORMAL",
            ),
            _MockPair(
                pair="GBP_USD", direction="SHORT", confidence=0.27,
                gates_passed=False, confidence_passed=False, momentum_passed=True,
                risk_passed=True, agent_votes=1, agent_total=3,
                kill_reason="confidence_gate",
            ),
            _MockPair(
                pair="AUD_USD", direction="HOLD", confidence=0.12,
                gates_passed=False, confidence_passed=False, momentum_passed=False,
                risk_passed=True,
            ),
            _MockPair(
                pair="EUR_GBP", direction="LONG", confidence=0.41,
                gates_passed=False, confidence_passed=True, momentum_passed=False,
                risk_passed=True, agent_votes=2, agent_total=3,
                kill_reason="momentum_gate",
            ),
        ],
        account_equity=101251.0,
    )
    render_scan_result(scan, cycle_num=47, elapsed_s=0.82)

    _console.print()

    # --- 2. Trade Execution ---
    render_trade_execution(
        pair="USD_JPY", direction="LONG", confidence=0.68,
        lots=0.79, sl_pips=19, tp_pips=28, rr_ratio=1.47,
        regime="NORMAL", risk_pct=0.021, entry_price=158.730,
        trade_id="48271",
    )

    _console.print()

    # --- 3. Watch Header ---
    render_watch_header(
        nav=101251, open_count=3, unrealized_pnl=147.0,
        session="London", cycle_num=47, uptime_s=8040,
        next_scan_s=282,
    )

    _console.print()

    # --- 4. Trade Closed ---
    render_trade_closed(
        pair="EUR_USD", direction="LONG", won=True,
        pnl_pips=22.4, pnl_dollars=336.0,
        exit_reason="TP hit", duration_s=11520,
        weight_changes={
            "trend": 0.006, "momentum": 0.004, "mr": -0.003,
            "volatility": 0.001, "risk": -0.001,
        },
        session_wins=7, session_losses=3, session_avg_pnl=84.0,
    )

    _console.print()

    # --- 4b. Trade Closed (LOSS) ---
    render_trade_closed(
        pair="GBP_USD", direction="SHORT", won=False,
        pnl_pips=-15.2, pnl_dollars=-228.0,
        exit_reason="SL hit", duration_s=5400,
        weight_changes={"trend": -0.004, "momentum": -0.002, "mr": 0.001},
        session_wins=7, session_losses=4, session_avg_pnl=62.5,
    )

    _console.print()

    # --- 5. Health Summary ---
    render_health_summary(
        health_score=92, cycle_num=50,
        gate_kills={"confidence_gate": 42, "rr_gate": 24, "uncertainty_gate": 16, "momentum_gate": 12},
        module_statuses={
            "OANDA": True, "Models": True, "RL sync": True,
            "Guards": True, "News": False,
        },
        win_rate=0.68, trades_count=38,
        rss_mb=284, cycle_avg_s=0.82,
        expectancy=71.0, sharpe=1.42,
    )

    _console.print()

    # --- 6. Full Status ---
    render_status(
        nav=101251, balance=100800, unrealized=451.0,
        positions=[
            {"pair": "USD_JPY", "direction": "LONG", "lots": 0.79,
             "entry_price": 158.730, "pnl": 187.0, "sl_pips": 19, "tp_pips": 28},
            {"pair": "EUR_USD", "direction": "SHORT", "lots": 0.50,
             "entry_price": 1.08420, "pnl": -32.0, "sl_pips": 22, "tp_pips": 33},
            {"pair": "GBP_JPY", "direction": "LONG", "lots": 0.30,
             "entry_price": 193.450, "pnl": 296.0, "sl_pips": 35, "tp_pips": 52},
        ],
        recent_trades=[
            {"pair": "EUR_USD", "won": True, "pnl_dollars": 336.0, "pnl_pips": 22.4,
             "exit_reason": "TP hit", "closed_at": "14:12"},
            {"pair": "AUD_USD", "won": False, "pnl_dollars": -142.0, "pnl_pips": -11.3,
             "exit_reason": "SL hit", "closed_at": "13:48"},
            {"pair": "USD_CAD", "won": True, "pnl_dollars": 89.0, "pnl_pips": 8.7,
             "exit_reason": "trailing SL", "closed_at": "12:30"},
        ],
        agent_weights={
            "trend": 0.142, "momentum": 0.128, "mean_reversion": 0.095,
            "volatility": 0.088, "risk_sentinel": 0.082, "uncertainty": 0.078,
            "execution_quality": 0.071, "session_timing": 0.068,
            "multi_timeframe": 0.065, "support_resistance": 0.062,
            "pair_performance": 0.061, "news_risk": 0.060,
        },
        model_health={
            "TCN": {"healthy": True, "accuracy": 0.67},
            "Ridge": {"healthy": True, "accuracy": 0.72},
            "RF": {"healthy": True, "accuracy": 0.64},
            "RL": {"healthy": True},
            "News": {"healthy": False},
        },
    )

    _console.print()
    _console.rule("[bold green]All renders complete[/bold green]")
