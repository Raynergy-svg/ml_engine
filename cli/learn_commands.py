"""CLI handler for the 'learn' command — manual learning loop triggers.

US-011: Create buddy_learn CLI command for manual learning triggers.

Usage:
    buddy learn                     # Default: analyze + promote + report
    buddy learn --analyze           # Analyze unanalyzed trades
    buddy learn --promote           # Promote qualifying patterns to rules
    buddy learn --consolidate       # Consolidate learnings
    buddy learn --report            # Print improvement report
    buddy learn --status            # Print current state and learnings
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def handle_learn(args: Any) -> None:
    """Dispatch the learn subcommand."""
    from rich.console import Console
    console = Console()

    analyze = getattr(args, "analyze", False)
    promote = getattr(args, "promote", False)
    consolidate = getattr(args, "consolidate", False)
    report = getattr(args, "report", False)
    status = getattr(args, "status", False)

    # Default: analyze + promote + report
    if not any([analyze, promote, consolidate, report, status]):
        analyze = True
        promote = True
        report = True

    _maybe_sync_closed_trades(console)

    if status:
        _show_status(console)

    if analyze:
        _run_analyze(console)

    if promote:
        _run_promote(console)

    if consolidate:
        _run_consolidate(console)

    if report:
        _run_report(console)


def _maybe_sync_closed_trades(console: Any) -> None:
    """Pull recently closed OANDA trades into the RL journal when possible."""
    try:
        from src.utils.oanda_practice import _load_project_dotenv

        _load_project_dotenv()
    except Exception:
        pass

    token = (os.getenv("OANDA_API_TOKEN", "") or os.getenv("OANDA_API_KEY", "")).strip()
    account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
    if not token or not account_id:
        console.print("[dim]Trade sync note: OANDA credentials not loaded for learn sync.[/dim]")
        return

    try:
        from src.scanner.execution import ExecutionManager

        result = ExecutionManager().sync_closed_trades_rl()
    except Exception as exc:
        logger.debug("Learn sync skipped: %s", exc)
        return

    if not isinstance(result, dict):
        return

    trades_synced = int(result.get("trades_synced", 0) or 0)
    detail = str(result.get("detail", "") or "").strip()
    if trades_synced > 0:
        console.print(f"[green]Synced {trades_synced} closed trade outcome(s) from OANDA.[/green]")
    elif detail and (
        "error" in detail.lower()
        or "missing" in detail.lower()
        or detail in {"no pending trades", "no journal"}
    ):
        console.print(f"[dim]Trade sync note: {detail}[/dim]")


def _show_status(console: Any) -> None:
    """Print current state and recent learnings."""
    from src.scanner.automation.state_engine import StateEngine

    console.print("\n[bold cyan]=== Current State ===[/bold cyan]")
    se = StateEngine()
    state = se.load_state()

    console.print(f"Goal: {state.get('goal', 'N/A')}")
    console.print(f"Status: {state.get('status', 'N/A')}")
    console.print(f"Last updated: {state.get('last_updated', 'N/A')}")
    console.print(f"Improvement focus: {state.get('improvement_focus', 'N/A')}")

    snap = state.get("portfolio_snapshot", {})
    console.print(f"\n[bold]Portfolio:[/bold] NAV=${snap.get('nav', 0):,.2f}, "
                  f"Open={snap.get('open_trades', 0)}, "
                  f"P/L=${snap.get('total_realized_pnl', 0):,.2f}, "
                  f"Win rate={snap.get('win_rate', 0):.0%}")

    # Recent learnings
    learnings_path = Path(".claude/learnings.md")
    if learnings_path.exists():
        lines = learnings_path.read_text().strip().split("\n")
        recent = [l for l in lines[-10:] if l.strip().startswith("-")]
        if recent:
            console.print("\n[bold]Recent Learnings:[/bold]")
            for line in recent:
                console.print(f"  {line}")
    console.print()


def _run_analyze(console: Any) -> None:
    """Run trade outcome analysis on unanalyzed journal entries."""
    from src.scanner.automation.learning_engine import LearningEngine, LearningEntry

    console.print("[cyan]Analyzing trade outcomes...[/cyan]")
    le = LearningEngine()

    journal_path = Path("trained_data/trade_journal_rl.json")
    if not journal_path.exists():
        console.print("[yellow]No trade journal found.[/yellow]")
        return

    entries = json.loads(journal_path.read_text())
    closed = [e for e in entries if e.get("outcome") is not None]
    pending = [e for e in entries if e.get("outcome") is None]

    if not closed and pending:
        console.print(
            f"[yellow]No closed trade outcomes in the RL journal yet.[/yellow] "
            f"[dim]{len(pending)} trade(s) are still waiting for outcome sync from OANDA.[/dim]"
        )
        return

    total_learnings = 0
    all_learnings: list[LearningEntry] = []
    for entry in closed:
        learnings = le.analyze_trade(entry)
        if learnings:
            count = le.append_to_learnings(learnings)
            total_learnings += count
            all_learnings.extend(learnings)

    console.print()
    if all_learnings:
        _print_learning_summary(console, closed, all_learnings)
    else:
        console.print(f"[yellow]No new learnings extracted from {len(closed)} closed trades.[/yellow]")


def _print_learning_summary(console: Any, closed_entries: list[dict[str, Any]], learnings: list[Any]) -> None:
    """Print a cleaner result view for analyzed trades."""
    from rich.panel import Panel
    from rich.table import Table

    pair_counter = Counter(str(entry.get("pair", "UNKNOWN")) for entry in closed_entries)
    category_counter = Counter(str(getattr(item, "category", "unknown")) for item in learnings)

    overview = Table(show_header=False, box=None, padding=(0, 2))
    overview.add_column("Label", style="dim")
    overview.add_column("Value")
    overview.add_row("Closed Trades", str(len(closed_entries)))
    overview.add_row("Learning Entries", str(len(learnings)))
    overview.add_row(
        "Pairs",
        ", ".join(f"{pair} ({count})" for pair, count in pair_counter.most_common(5)) or "--",
    )
    overview.add_row(
        "Categories",
        ", ".join(f"{category} ({count})" for category, count in category_counter.most_common(5)) or "--",
    )

    console.print(Panel(overview, title="[bold]Trade Analysis[/bold]", border_style="green"))

    learnings_table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    learnings_table.add_column("Category", style="dim", width=18)
    learnings_table.add_column("Insight", width=58)
    learnings_table.add_column("Action", width=42)

    for item in learnings[:6]:
        insight = str(getattr(item, "insight", "")).strip()
        action = str(getattr(item, "action", "")).strip()
        if not insight:
            continue
        learnings_table.add_row(
            str(getattr(item, "category", "unknown")),
            insight,
            action or "--",
        )

    console.print(Panel(learnings_table, title="[bold]Key Learnings[/bold]", border_style="blue"))


def _run_promote(console: Any) -> None:
    """Check for and promote qualifying patterns."""
    from src.scanner.automation.learning_engine import LearningEngine

    console.print("[cyan]Checking for rule promotions...[/cyan]")
    le = LearningEngine()
    promoted = le.check_promotions()

    if promoted:
        console.print(f"[green]Promoted {len(promoted)} patterns to rules:[/green]")
        for rule in promoted:
            console.print(f"  {rule}")
    else:
        console.print("[dim]No patterns qualify for promotion yet (need 3+ observations).[/dim]")


def _run_consolidate(console: Any) -> None:
    """Run learnings consolidation."""
    from src.scanner.automation.learning_engine import LearningEngine

    console.print("[cyan]Consolidating learnings...[/cyan]")
    le = LearningEngine()
    audit_result = le.audit()

    actions = audit_result.get("actions", [])
    if actions:
        for action in actions:
            console.print(f"  [green]{action}[/green]")
    else:
        console.print("[dim]No consolidation needed.[/dim]")


def _run_report(console: Any) -> None:
    """Print improvement tracker report."""
    from src.scanner.automation.improvement_tracker import ImprovementTracker

    console.print("\n[bold cyan]=== Improvement Report ===[/bold cyan]")
    tracker = ImprovementTracker()
    report = tracker.generate_report()
    console.print(report)
    console.print()
