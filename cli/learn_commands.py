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
    from src.scanner.automation.learning_engine import LearningEngine

    console.print("[cyan]Analyzing trade outcomes...[/cyan]")
    le = LearningEngine()

    journal_path = Path("trained_data/trade_journal_rl.json")
    if not journal_path.exists():
        console.print("[yellow]No trade journal found.[/yellow]")
        return

    entries = json.loads(journal_path.read_text())
    closed = [e for e in entries if e.get("outcome") is not None]

    total_learnings = 0
    for entry in closed:
        learnings = le.analyze_trade(entry)
        if learnings:
            count = le.append_to_learnings(learnings)
            total_learnings += count

    console.print(f"[green]Extracted {total_learnings} learning entries from {len(closed)} closed trades.[/green]")


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
