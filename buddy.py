#!/usr/bin/env python3
"""Buddy ML Engine — Interactive Trading CLI"""

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
import sys
import os

# Neural Intelligence brand colors for questionary
BUDDY_STYLE = Style([
    ("qmark", "fg:#00BCD4 bold"),        # Cyan diamond
    ("question", "fg:#FFFFFF bold"),       # White question text
    ("answer", "fg:#00E676 bold"),        # Green answer
    ("pointer", "fg:#B388FF bold"),       # Violet pointer
    ("highlighted", "fg:#B388FF bold"),   # Violet highlight
    ("selected", "fg:#00E676"),           # Green selected
    ("separator", "fg:#757575"),          # Dim separator
    ("instruction", "fg:#757575"),        # Dim instructions
])

console = Console()

def show_header():
    console.print()
    console.print(Panel(
        "[bold cyan]◆ BUDDY[/bold cyan]  [dim]ML-Powered FX Trading Engine[/dim]",
        border_style="cyan",
        padding=(0, 2)
    ))
    console.print()

def main_menu():
    return questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice("Start Trading", value="start"),
            questionary.Choice("Supervision Mode", value="supervise"),
            questionary.Choice("Quick Scan", value="scan"),
            questionary.Choice("Status", value="status"),
            questionary.Choice("Configure", value="config"),
            questionary.Choice("Trade Journal", value="journal"),
            questionary.Choice("Health Check", value="health"),
            questionary.Choice("Learn & Improve", value="learn"),
            questionary.Choice("Exit", value="exit"),
        ],
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

def start_trading():
    """Interactive setup for watch mode with auto-execute."""
    profile = questionary.select(
        "Trading profile?",
        choices=[
            questionary.Choice("smart (recommended)", value="smart"),
            questionary.Choice("aggressive", value="aggressive"),
            questionary.Choice("balanced", value="balanced"),
            questionary.Choice("conservative", value="conservative"),
        ],
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()
    if profile is None: return  # User cancelled

    # Pair selection
    ALL_PAIRS = [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD",
        "USD_CAD", "USD_CHF", "EUR_GBP", "EUR_JPY", "GBP_JPY",
        "AUD_JPY", "EUR_AUD", "GBP_AUD", "EUR_CHF", "GBP_CHF",
    ]

    use_all = questionary.confirm(
        "Scan all 15 pairs?",
        default=True,
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

    if use_all:
        pairs = ALL_PAIRS
    else:
        pairs = questionary.checkbox(
            "Select pairs:",
            choices=ALL_PAIRS,
            style=BUDDY_STYLE,
            qmark="◆",
            validate=lambda x: len(x) > 0 or "Select at least one pair",
        ).ask()
    if pairs is None: return

    force = questionary.confirm(
        "Force past session filter? (for off-hours trading)",
        default=False,
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

    dry_run = questionary.confirm(
        "Dry run? (no real trades)",
        default=False,
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

    # Tier 7.5: Supervision mode
    supervision = questionary.confirm(
        "Use supervision mode? (phase-based runtime — recommended)",
        default=False,
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

    enforce_level = 0
    if supervision:
        enforce_choice = questionary.select(
            "Policy enforcement level?",
            choices=[
                questionary.Choice("0 — logging only (safe, recommended to start)", value="0"),
                questionary.Choice("1 — enforce safe actions (tighten SL, rescan)", value="1"),
                questionary.Choice("2 — enforce config changes (model switch, retrain)", value="2"),
                questionary.Choice("3 — full enforcement (including trade gating)", value="3"),
            ],
            style=BUDDY_STYLE,
            qmark="◆",
        ).ask()
        enforce_level = int(enforce_choice) if enforce_choice else 0

    # Build and run the command
    pairs_str = ",".join(pairs)
    cmd = f"python main.py scan --watch --auto-execute --profile {profile} --pairs {pairs_str}"
    if force:
        cmd += " --force"
    if dry_run:
        cmd += " --dry-run"
    if supervision:
        cmd += " --supervision"
    if enforce_level > 0:
        cmd += f" --enforce-policy {enforce_level}"

    console.print()
    console.print(f"[dim]Running: {cmd}[/dim]")
    console.print()
    os.execvp("python", cmd.split())

def start_supervision():
    """Interactive setup for supervision mode — phase-based runtime."""
    console.print("[bold cyan]◆ Supervision Mode[/bold cyan] — supervise-first, scan on-demand")
    console.print("[dim]Trades are supervised at 30s ticks. Scanner only triggers when capacity is available.[/dim]\n")

    profile = questionary.select(
        "Trading profile?",
        choices=[
            questionary.Choice("smart (recommended)", value="smart"),
            questionary.Choice("aggressive", value="aggressive"),
            questionary.Choice("balanced", value="balanced"),
            questionary.Choice("conservative", value="conservative"),
        ],
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()
    if profile is None: return

    ALL_PAIRS = [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD",
        "USD_CAD", "USD_CHF", "EUR_GBP", "EUR_JPY", "GBP_JPY",
        "AUD_JPY", "EUR_AUD", "GBP_AUD", "EUR_CHF", "GBP_CHF",
    ]

    use_all = questionary.confirm(
        "Scan all 15 pairs?",
        default=True,
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()
    pairs = ALL_PAIRS if use_all else questionary.checkbox(
        "Select pairs:",
        choices=ALL_PAIRS,
        style=BUDDY_STYLE,
        qmark="◆",
        validate=lambda x: len(x) > 0 or "Select at least one pair",
    ).ask()
    if pairs is None: return

    enforce_choice = questionary.select(
        "Policy enforcement level?",
        choices=[
            questionary.Choice("0 — logging only (recommended to start)", value="0"),
            questionary.Choice("1 — safe actions enforced", value="1"),
            questionary.Choice("2 — + config/model changes", value="2"),
            questionary.Choice("3 — full enforcement", value="3"),
        ],
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()
    enforce_level = int(enforce_choice) if enforce_choice else 0

    force = questionary.confirm(
        "Force past session filter?",
        default=False,
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

    dry_run = questionary.confirm(
        "Dry run? (no real trades)",
        default=False,
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

    pairs_str = ",".join(pairs)
    cmd = f"python main.py scan --watch --auto-execute --supervision --profile {profile} --pairs {pairs_str}"
    if enforce_level > 0:
        cmd += f" --enforce-policy {enforce_level}"
    if force:
        cmd += " --force"
    if dry_run:
        cmd += " --dry-run"

    console.print()
    console.print(f"[dim]Running: {cmd}[/dim]")
    console.print()
    os.execvp("python", cmd.split())


def quick_scan():
    """One-shot scan with interactive pair/profile selection."""
    profile = questionary.select(
        "Profile?",
        choices=["smart", "aggressive", "balanced", "conservative"],
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()
    if profile is None: return

    pairs = questionary.checkbox(
        "Select pairs to scan:",
        choices=[
            "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD",
            "USD_CAD", "USD_CHF", "EUR_GBP", "EUR_JPY", "GBP_JPY",
        ],
        style=BUDDY_STYLE,
        qmark="◆",
        validate=lambda x: len(x) > 0 or "Select at least one",
    ).ask()
    if pairs is None: return

    force = questionary.confirm("Force past session filter?", default=True, style=BUDDY_STYLE, qmark="◆").ask()

    pairs_str = ",".join(pairs)
    cmd = f"python main.py scan --profile {profile} --pairs {pairs_str} --dry-run"
    if force:
        cmd += " --force"

    console.print(f"\n[dim]Running: {cmd}[/dim]\n")
    os.execvp("python", cmd.split())

def show_status():
    """Show status dashboard."""
    os.system("python main.py status")

def show_config():
    """Interactive config viewer/editor."""
    action = questionary.select(
        "Configuration:",
        choices=[
            questionary.Choice("View current profile settings", value="view"),
            questionary.Choice("Switch profile", value="switch"),
            questionary.Choice("Edit thresholds", value="edit"),
            questionary.Choice("Back", value="back"),
        ],
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

    if action == "view":
        os.system("python -c \"from src.scanner.config import ScannerConfig; c = ScannerConfig(); c.apply_profile('smart'); print(f'min_confidence: {c.min_confidence}'); print(f'min_momentum: {c.min_momentum}'); print(f'max_drawdown_pct: {c.max_drawdown_pct}'); print(f'risk_per_trade_pct: {c.risk_per_trade_pct}'); print(f'atr_sl_multiplier: {c.atr_sl_multiplier}'); print(f'atr_tp_multiplier: {c.atr_tp_multiplier}'); print(f'min_rr_ratio: {c.min_rr_ratio}')\"")
    elif action == "switch":
        profile = questionary.select("Profile?", choices=["smart", "aggressive", "balanced", "conservative"], style=BUDDY_STYLE, qmark="◆").ask()
        if profile:
            console.print(f"[green]Profile set to: {profile}[/green]")
    elif action == "edit":
        console.print("[dim]Threshold editor coming soon — edit src/scanner/config.py directly for now[/dim]")

def show_journal():
    """Show trade journal."""
    os.system("python main.py journal --last 10")

def show_health():
    """Run health/validation check."""
    os.system("python scripts/validate_deployment.py")

def show_learn():
    """Learning and improvement commands."""
    action = questionary.select(
        "Learning:",
        choices=[
            questionary.Choice("Analyze recent trades", value="analyze"),
            questionary.Choice("Promote patterns to rules", value="promote"),
            questionary.Choice("Consolidate learnings", value="consolidate"),
            questionary.Choice("Full report", value="report"),
            questionary.Choice("Back", value="back"),
        ],
        style=BUDDY_STYLE,
        qmark="◆",
    ).ask()

    if action and action != "back":
        os.system(f"python main.py learn --{action}")

def main():
    show_header()

    while True:
        action = main_menu()

        if action is None or action == "exit":
            console.print("\n[dim]Goodbye.[/dim]\n")
            sys.exit(0)

        handlers = {
            "start": start_trading,
            "supervise": start_supervision,
            "scan": quick_scan,
            "status": show_status,
            "config": show_config,
            "journal": show_journal,
            "health": show_health,
            "learn": show_learn,
        }

        handler = handlers.get(action)
        if handler:
            try:
                handler()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Cancelled.[/dim]")
            console.print()  # Breathing room before next menu

if __name__ == "__main__":
    main()
