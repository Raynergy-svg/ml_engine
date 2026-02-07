#!/usr/bin/env python3
"""Analytics and monitoring CLI commands for ML Engine Trading Bot.

This module contains commands for:
- Feature importance analysis (SHAP-based)
- Trade journal and performance tracking
- Model improvement suggestions via LLM
- System monitoring and drift detection
"""
from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

from cli.io_utils import console, DEFAULT_CONFIG_PATH, BUDDY_META_FILENAME
from cli.tf_config import _configure_tf_metal
from src.utils import load_config


def suggest_improvements(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str | None = None,
    auto_apply: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """Meta-learning: Generate and optionally auto-apply model improvement suggestions.
    
    Uses LLM to analyze current model architecture and performance, then suggests
    improvements. Hyperparam suggestions that pass gates are auto-applied if requested.
    
    When a specific instrument is provided, only that pair's config is modified.
    
    Gates for auto-approval:
    1. confidence > 70%
    2. expected Sharpe improvement > 5%
    3. implementation_ease == "easy"
    4. category == "hyperparam" (architecture/feature changes require manual review)
    
    Args:
        config_path: Path to config file
        instrument: Specific pair to analyze (e.g., USD_JPY). If None, uses global model.
        auto_apply: If True, automatically apply approved hyperparam changes
        verbose: Show detailed output
    """
    from pathlib import Path
    import json
    
    # Normalize instrument
    if instrument:
        instrument = instrument.upper().replace("/", "_")
        console.print(f"\n[bold cyan]🧠 META-LEARNING: Improvements for {instrument}[/bold cyan]\n")
    else:
        console.print("\n[bold cyan]🧠 META-LEARNING: Model Improvement Suggestions[/bold cyan]\n")
    
    # Initialize LLM
    try:
        from llm_providers import initialize_buddy_llm, check_provider_status
        provider = initialize_buddy_llm()
        
        if not provider.is_available:
            console.print("[red]✗ No LLM provider available[/red]")
            console.print("[dim]Install Ollama (brew install ollama && ollama pull llama3.1:7b) or set ANTHROPIC_API_KEY[/dim]")
            return
        
        console.print(f"[green]✓ LLM Provider: {provider.name.upper()}[/green]")
    except ImportError as e:
        console.print(f"[red]✗ LLM providers not available: {e}[/red]")
        return
    
    # Load current architecture info from model metadata
    models_dir = Path("trained_data/models")
    
    # Check for pair-specific model first
    if instrument:
        pair_meta_path = models_dir / instrument / "modular_ensemble.meta.json"
        if pair_meta_path.exists():
            meta_path = pair_meta_path
            console.print(f"[dim]Using {instrument}-specific model metadata[/dim]")
        else:
            # Fallback to global model
            meta_path = models_dir / "modular_ensemble.meta.json"
            console.print(f"[yellow]No {instrument}-specific model found, using global model[/yellow]")
    else:
        meta_path = models_dir / "modular_ensemble.meta.json"
    
    if not meta_path.exists():
        meta_path = models_dir / "buddy_tf.meta.json"
    
    if not meta_path.exists():
        console.print("[red]✗ No model metadata found. Train a model first.[/red]")
        return
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Build architecture description
    from buddy_intelligent_mode import (
        BuddyArchitectureDescription,
        PerformanceStats,
        get_intelligent_mode,
        gated_auto_approve_suggestions,
    )
    
    # Format feature set and model type for description
    model_type = meta.get('model_type', 'tcn')
    models_list = list(meta.get('models', {}).keys()) if 'models' in meta else [model_type]
    model_type_str = f"Ensemble ({', '.join(models_list)})" if len(models_list) > 1 else model_type.upper()
    
    features_list = meta.get('feature_columns', meta.get('feature_names', []))[:10]
    feature_str = ", ".join(features_list) if isinstance(features_list, list) else str(features_list)
    
    architecture = BuddyArchitectureDescription(
        model_type=model_type_str,
        feature_set=feature_str,
        lookback_window=meta.get('seq_len', 60),
        hidden_size=meta.get('hidden_size', 64),
        num_layers=meta.get('num_layers', 2),
        dropout=meta.get('dropout', 0.2),
    )
    
    console.print(f"[dim]Architecture: {architecture.to_description()}[/dim]\n")
    
    # Load performance stats from journal if available (filter by instrument if specified)
    try:
        from trade_journal import TradeJournal
        from oanda_practice import OandaPracticeClient
        
        journal = TradeJournal()
        client = OandaPracticeClient.from_env()
        perf_data = journal.get_full_performance_stats(client, days=30)
        stats = perf_data.get('stats', {})
        
        # If instrument specified, try to get pair-specific stats
        if instrument:
            pair_stats = perf_data.get('pair_stats', {}).get(instrument, stats)
            if pair_stats != stats:
                console.print(f"[dim]Using {instrument}-specific performance stats[/dim]")
                stats = pair_stats
        
        performance = PerformanceStats(
            win_rate=stats.get('win_rate', 0.5),
            profit_factor=stats.get('profit_factor', 1.0),
            sharpe_ratio=stats.get('sharpe_ratio', 0.0),
            max_drawdown=stats.get('max_drawdown', 0.0),
            avg_trade_pnl=stats.get('avg_pnl', 0.0),
            total_trades=stats.get('total_trades', 0),
        )
    except Exception:
        # Use defaults if journal not available
        performance = PerformanceStats()
    
    console.print(f"[dim]Performance: {performance.to_summary()}[/dim]\n")
    
    # Get improvement suggestions
    console.print("[cyan]Analyzing model for improvements...[/cyan]")
    
    # Add instrument context to known weaknesses
    weaknesses = ["regime shifts", "news events", "low volatility periods"]
    if instrument:
        # Add pair-specific context
        if "JPY" in instrument:
            weaknesses.append("BOJ intervention risk")
        if "GBP" in instrument:
            weaknesses.append("Brexit-related volatility")
        if "AUD" in instrument or "NZD" in instrument:
            weaknesses.append("China trade sensitivity")
    
    intel_mode = get_intelligent_mode()
    result = intel_mode.get_improvement_suggestions(
        architecture=architecture,
        performance=performance,
        known_weaknesses=weaknesses,
    )
    
    if result.get('error'):
        console.print(f"[red]✗ Failed to get suggestions: {result['error']}[/red]")
        return
    
    improvements = result.get('improvements', [])
    
    if not improvements:
        console.print("[yellow]No improvements suggested (or LLM response couldn't be parsed)[/yellow]")
        if result.get('raw_response'):
            console.print(f"\n[dim]Raw response:[/dim]\n{result['raw_response'][:500]}...")
        return
    
    # Display all suggestions
    console.print(f"\n[bold]📋 {len(improvements)} Suggestions{' for ' + instrument if instrument else ''}:[/bold]\n")
    
    for i, imp in enumerate(improvements, 1):
        category_color = {
            'hyperparam': 'green',
            'hyperparameter': 'green',
            'parameter': 'green',
            'feature': 'blue',
            'architecture': 'magenta',
            'training': 'yellow',
        }.get(imp.category.lower(), 'white')
        
        ease_emoji = {'easy': '🟢', 'medium': '🟡', 'hard': '🔴'}.get(imp.implementation_ease.lower(), '⚪')
        
        console.print(f"  {i}. [{category_color}][{imp.category.upper()}][/{category_color}] {ease_emoji}")
        console.print(f"     {imp.suggestion}")
        console.print(f"     [dim]Impact: {imp.expected_impact}[/dim]")
        console.print(f"     [dim]Test: {imp.test_plan}[/dim]")
        console.print()
    
    # Determine config path for auto-apply
    # If instrument specified, create/use pair-specific config override
    if auto_apply and instrument:
        pair_config_dir = Path("trained_data/configs")
        pair_config_dir.mkdir(parents=True, exist_ok=True)
        pair_config_path = pair_config_dir / f"{instrument}_overrides.yaml"
        config_file = pair_config_path
        console.print(f"[dim]Changes will be saved to: {pair_config_path}[/dim]")
    else:
        config_file = Path(config_path)
    
    # Process through gated auto-approval
    approval_result = gated_auto_approve_suggestions(
        improvements=improvements,
        config_path=config_file if auto_apply else None,
        auto_apply=auto_apply,
    )
    
    # Display approval results
    console.print("\n[bold]🚦 Gated Auto-Approval Results:[/bold]\n")
    
    for line in approval_result.changelog:
        if line.startswith("✓"):
            console.print(f"  [green]{line}[/green]")
        elif line.startswith("✗"):
            console.print(f"  [red]{line}[/red]")
        elif line.startswith("⏸"):
            console.print(f"  [yellow]{line}[/yellow]")
        elif line.startswith("  →"):
            console.print(f"  [cyan]{line}[/cyan]")
        else:
            console.print(f"  {line}")
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  ✓ Approved: {len(approval_result.approved)} (hyperparams passing all gates)")
    console.print(f"  ⏸ Pending:  {len(approval_result.pending_review)} (require manual review)")
    console.print(f"  ✗ Rejected: {len(approval_result.rejected)} (failed gates)")
    
    if auto_apply and approval_result.applied_changes:
        console.print(f"\n[green]✓ Applied {len(approval_result.applied_changes)} changes to {config_file}[/green]")
        if instrument:
            console.print(f"[dim]These changes only affect {instrument} training/inference[/dim]")
    elif not auto_apply and approval_result.approved:
        console.print(f"\n[dim]Run with --auto-apply to apply approved changes[/dim]")


def buddy_journal(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    update: bool = False,
    days: int = 30,
    verbose: bool = False,
    import_trades: bool = False,
    **kwargs: Any,
) -> None:
    """Display comprehensive trade journal with full OANDA account performance.
    
    Args:
        config_path: Path to config file
        update: If True, sync local journal with OANDA
        days: Number of days of history to show
        verbose: Show detailed trade list
        import_trades: If True, import untracked open trades into journal
    """
    from datetime import datetime
    from src.utils.trade_journal import TradeJournal
    from src.utils.oanda_practice import OandaPracticeClient
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    
    journal = TradeJournal()
    client = OandaPracticeClient.from_env()
    
    if update:
        console.print("\n[dim]Syncing local journal with OANDA...[/dim]")
        try:
            sync_results = journal.sync_open_trades(client)
            closed_updated = sync_results.get('closed_updated', 0)
            open_updated = sync_results.get('open_updated', 0)
            if closed_updated > 0 or open_updated > 0:
                console.print(f"[green]✓ Local journal synced ({closed_updated} closed, {open_updated} open)[/green]")
            else:
                console.print("[dim]No journal updates from OANDA[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to sync journal from OANDA: {e}[/red]")
    
    if import_trades:
        console.print("\n[dim]Importing untracked open trades from OANDA...[/dim]")
        try:
            imported = journal.import_untracked_trades(client)
            if imported > 0:
                console.print(f"[green]✓ Imported {imported} trades into local journal[/green]")
            else:
                console.print("[dim]No untracked open trades found[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to import open trades: {e}[/red]")
    
    # Get statistics
    stats = journal.get_statistics(days=days)
    
    if stats["total_trades"] == 0:
        console.print("\n[red]No trades found in journal.[/red]")
        console.print("[dim]Journal entries are created when the predictions command runs with --execute.[/dim]")
        console.print("=" * 70)
        sys.exit(1)
    
    # Fetch comprehensive performance data from OANDA
    console.print("\n[dim]Fetching account data from OANDA...[/dim]")
    perf_data = journal.get_full_performance_stats(client, days=days)
    
    account = perf_data.get('account')
    open_trades = perf_data.get('open_trades', [])
    closed_trades = perf_data.get('closed_trades', [])
    stats = perf_data.get('stats', {})
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════════
    console.print()
    console.print("╔" + "═" * 78 + "╗")
    console.print("║" + " " * 25 + "[bold cyan]📊 TRADING PERFORMANCE DASHBOARD[/bold cyan]" + " " * 22 + "║")
    console.print("╚" + "═" * 78 + "╝")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ACCOUNT SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    if account:
        console.print()
        
        # Calculate daily P/L (approximation)
        daily_pnl = account.realized_pnl / max(days, 1) if days else account.realized_pnl / 30
        
        # Account status panel
        balance_color = "green" if account.balance >= 100000 else "white"
        nav_color = "green" if account.nav >= account.balance else "red"
        unrealized_color = "green" if account.unrealized_pnl >= 0 else "red"
        realized_color = "green" if account.realized_pnl >= 0 else "red"
        
        # Main metrics
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column("Label", style="dim")
        metrics_table.add_column("Value", justify="right")
        metrics_table.add_column("Label2", style="dim")
        metrics_table.add_column("Value2", justify="right")
        
        metrics_table.add_row(
            "Balance:",
            f"[{balance_color}]${account.balance:,.2f}[/{balance_color}]",
            "NAV:",
            f"[{nav_color}]${account.nav:,.2f}[/{nav_color}]",
        )
        metrics_table.add_row(
            "Unrealized P/L:",
            f"[{unrealized_color}]${account.unrealized_pnl:+,.2f}[/{unrealized_color}]",
            "Realized P/L:",
            f"[{realized_color}]${account.realized_pnl:+,.2f}[/{realized_color}]",
        )
        metrics_table.add_row(
            "Margin Used:",
            f"${account.margin_used:,.2f}",
            "Margin Available:",
            f"${account.margin_available:,.2f}",
        )
        
        margin_pct = (account.margin_used / account.nav * 100) if account.nav > 0 else 0
        margin_color = "green" if margin_pct < 30 else ("yellow" if margin_pct < 50 else "red")
        metrics_table.add_row(
            "Margin Usage:",
            f"[{margin_color}]{margin_pct:.1f}%[/{margin_color}]",
            "Open Positions:",
            f"{account.open_position_count}",
        )
        
        console.print(Panel(
            metrics_table,
            title="[bold white]💰 ACCOUNT SUMMARY[/bold white]",
            border_style="blue",
            padding=(1, 2),
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPEN POSITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    if open_trades:
        total_unrealized = sum(t.unrealized_pnl for t in open_trades)
        pnl_color = "green" if total_unrealized >= 0 else "red"
        
        open_table = Table(show_header=True, header_style="bold white", box=None)
        open_table.add_column("Trade ID", style="dim")
        open_table.add_column("Pair", width=9)
        open_table.add_column("Direction", justify="center", width=7)
        open_table.add_column("Units", justify="right", width=10)
        open_table.add_column("Entry Price", justify="right", width=12)
        open_table.add_column("Stop Loss", justify="right", width=12)
        open_table.add_column("Take Profit", justify="right", width=12)
        open_table.add_column("Unrealized P/L", justify="right", width=14)
        
        for trade in open_trades:
            direction_style = "green" if trade.direction == "long" else "red"
            direction_text = "LONG" if trade.direction == "long" else "SHORT"
            
            pnl = trade.unrealized_pnl
            trade_pnl_color = "green" if pnl >= 0 else "red"
            
            is_jpy = trade.instrument.endswith('_JPY')
            price_fmt = ".3f" if is_jpy else ".5f"
            
            sl_str = f"{trade.stop_loss:{price_fmt}}" if trade.stop_loss else "—"
            tp_str = f"{trade.take_profit:{price_fmt}}" if trade.take_profit else "—"
            
            open_table.add_row(
                str(trade.trade_id),
                trade.instrument.replace("_", "/"),
                f"[{direction_style}]{direction_text}[/{direction_style}]",
                f"{trade.units:,}",
                f"{trade.entry_price:{price_fmt}}",
                sl_str,
                tp_str,
                f"[{trade_pnl_color}]${pnl:+,.2f}[/{trade_pnl_color}]",
            )
        
        header = f"[bold white]🔴 OPEN POSITIONS ({len(open_trades)})[/bold white]  │  Unrealized: [{pnl_color}]${total_unrealized:+,.2f}[/{pnl_color}]"
        console.print(Panel(
            open_table,
            title=header,
            border_style="cyan",
            padding=(0, 1),
        ))
    else:
        console.print(Panel(
            "[dim]No open positions[/dim]",
            title="[bold white]🔴 OPEN POSITIONS[/bold white]",
            border_style="dim",
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    if stats.get('total_trades', 0) > 0:
        # Performance metrics
        win_rate = stats.get('win_rate', 0) * 100
        wr_color = "green" if win_rate >= 50 else ("yellow" if win_rate >= 40 else "red")
        
        pf = stats.get('profit_factor', 0)
        pf_color = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        
        total_pnl = stats.get('total_pnl', 0)
        pnl_color = "green" if total_pnl >= 0 else "red"
        
        # Build performance panel
        perf_lines = []
        perf_lines.append(f"[bold]📈 Performance Metrics[/bold]")
        perf_lines.append("")
        perf_lines.append(f"  Total Trades:     [white]{stats.get('total_trades', 0)}[/white]")
        perf_lines.append(f"  Win Rate:         [{wr_color}]{win_rate:.1f}%[/{wr_color}]  ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)")
        perf_lines.append(f"  Profit Factor:    [{pf_color}]{pf_str}[/{pf_color}]")
        perf_lines.append(f"  Total P/L:        [{pnl_color}]${total_pnl:+,.2f}[/{pnl_color}]")
        perf_lines.append(f"  Total Pips:       {stats.get('total_pips', 0):+.1f}")
        
        if stats.get('total_financing', 0) != 0:
            fin_color = "red" if stats.get('total_financing', 0) < 0 else "green"
            perf_lines.append(f"  Financing:        [{fin_color}]${stats.get('total_financing', 0):+.2f}[/{fin_color}]")
        
        perf_lines.append("")
        perf_lines.append(f"[bold]📊 Risk Metrics[/bold]")
        perf_lines.append("")
        avg_win = stats.get('avg_win', 0)
        avg_loss = stats.get('avg_loss', 0)
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        rr_text = f"{rr_ratio:.2f}" if rr_ratio != float('inf') else "∞"
        
        perf_lines.append(f"  Avg Win:          [green]${avg_win:,.2f}[/green]")
        perf_lines.append(f"  Avg Loss:         [red]${avg_loss:,.2f}[/red]")
        perf_lines.append(f"  Risk/Reward:      {rr_text}")
        perf_lines.append(f"  Max Drawdown:     [red]${stats.get('max_drawdown', 0):,.2f}[/red]")
        
        # Streak info
        streak = stats.get('current_streak', 0)
        streak_color = "green" if streak > 0 else ("red" if streak < 0 else "white")
        streak_text = f"+{streak}" if streak > 0 else str(streak)
        
        perf_lines.append("")
        perf_lines.append(f"[bold]🔥 Streaks[/bold]")
        perf_lines.append("")
        perf_lines.append(f"  Current Streak:   [{streak_color}]{streak_text}[/{streak_color}]")
        perf_lines.append(f"  Best Win Streak:  [green]+{stats.get('max_win_streak', 0)}[/green]")
        perf_lines.append(f"  Worst Loss Streak:[red]-{stats.get('max_loss_streak', 0)}[/red]")
        
        # By instrument breakdown
        by_instrument = stats.get('by_instrument', {})
        if by_instrument:
            perf_lines.append("")
            perf_lines.append(f"[bold]🌍 By Instrument[/bold]")
            perf_lines.append("")
            
            # Sort by P/L
            sorted_instruments = sorted(by_instrument.items(), key=lambda x: x[1]['pnl'], reverse=True)
            for instr, data in sorted_instruments:
                total = data['wins'] + data['losses']
                wr = data['wins'] / total * 100 if total > 0 else 0
                instr_pnl_color = "green" if data['pnl'] >= 0 else "red"
                perf_lines.append(
                    f"  {instr.replace('_', '/'):<10} {data['wins']}W/{data['losses']}L "
                    f"({wr:.0f}%) [{instr_pnl_color}]${data['pnl']:+,.2f}[/{instr_pnl_color}]"
                )
        
        console.print(Panel(
            "\n".join(perf_lines),
            title=f"[bold white]📈 PERFORMANCE SUMMARY ({stats.get('total_trades', 0)} trades)[/bold white]",
            border_style="green" if total_pnl >= 0 else "red",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            "[dim]No closed trades to analyze[/dim]",
            title="[bold white]📈 PERFORMANCE SUMMARY[/bold white]",
            border_style="dim",
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECENT TRADE HISTORY
    # ═══════════════════════════════════════════════════════════════════════════
    if verbose and closed_trades:
        console.print()
        
        history_table = Table(show_header=True, header_style="bold white", box=None)
        history_table.add_column("Date", style="dim", width=12)
        history_table.add_column("Pair", width=9)
        history_table.add_column("Dir", justify="center", width=4)
        history_table.add_column("Units", justify="right", width=10)
        history_table.add_column("Entry", justify="right", width=10)
        history_table.add_column("Exit", justify="right", width=10)
        history_table.add_column("P/L", justify="right", width=12)
        
        for trade in closed_trades:
            direction_style = "green" if trade.direction == "long" else "red"
            direction_symbol = "⬆" if trade.direction == "long" else "⬇"
            
            pnl = trade.realized_pnl
            trade_pnl_color = "green" if pnl >= 0 else "red"
            
            is_jpy = trade.instrument.endswith('_JPY')
            price_fmt = ".3f" if is_jpy else ".5f"
            
            try:
                dt = datetime.fromisoformat(trade.close_time.replace('Z', '+00:00'))
                date_str = dt.strftime("%m/%d %H:%M")
            except Exception:
                date_str = "—"
            
            history_table.add_row(
                date_str,
                trade.instrument.replace("_", "/"),
                f"[{direction_style}]{direction_symbol}[/{direction_style}]",
                f"{trade.units:,}",
                f"{trade.entry_price:{price_fmt}}",
                f"{trade.exit_price:{price_fmt}}",
                f"[{trade_pnl_color}]${pnl:+,.2f}[/{trade_pnl_color}]",
            )
        
        console.print(Panel(
            history_table,
            title=f"[bold white]📋 RECENT TRADE HISTORY ({len(closed_trades)} trades)[/bold white]",
            border_style="white",
            padding=(0, 1),
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════════════════
    console.print()
    console.print("─" * 80)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[dim]Last updated: {timestamp}  │  Use --verbose for trade history  │  Use --update to sync journal[/dim]")


def buddy_analyze(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    top_n: int = 30,
    save_plots: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    Analyze feature importance using SHAP values.
    
    Shows which features contribute most to predictions, helping to:
    - Understand what the model learned
    - Remove noisy features
    - Improve model performance
    """
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from rich.table import Table
    
    console.print("\n" + "=" * 70)
    console.print(f"[bold magenta]🔬 FEATURE IMPORTANCE ANALYSIS[/bold magenta]")
    console.print("=" * 70)
    
    cfg = load_config(config_path)
    
    # Load model metadata
    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    candidate_meta_path = Path("trained_data") / "models" / "buddy_tf_candidate.meta.json"
    ensemble_meta_path = Path("trained_data") / "models" / "buddy_ensemble.meta.json"
    
    meta = None
    for mp in [ensemble_meta_path, candidate_meta_path, meta_path]:
        if mp.exists():
            meta = json.loads(mp.read_text())
            break
    
    if meta is None:
        console.print("[red]No trained model found. Run: buddy train --oanda-live[/red]")
        return
    
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        console.print("[red]No feature columns in model metadata[/red]")
        return
    
    console.print(f"[dim]Analyzing {len(feature_columns)} features...[/dim]")
    
    # Try to use SHAP if available, otherwise fallback to simpler methods
    try:
        from feature_importance import FeatureImportanceAnalyzer, SHAP_AVAILABLE
        use_shap = SHAP_AVAILABLE
    except ImportError:
        use_shap = False
    
    # Load recent data for analysis
    console.print("[dim]Fetching recent data for analysis...[/dim]")
    
    from src.utils.oanda_practice import OandaPracticeClient
    from feature_engineering import FeatureEngineering
    from fx_paper import candles_to_ohlcv_df
    
    client = OandaPracticeClient.from_env()
    fe = FeatureEngineering()
    
    resp = client.get_candles("USD_JPY", granularity="H1", count=2000, price="MBA")
    df_raw = candles_to_ohlcv_df(resp)
    df = fe.create_features(df_raw.copy(), include_all=True)
    
    # Ensure required columns
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    
    X = df[feature_columns].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create direction labels for correlation analysis
    df["future_return"] = df["close"].pct_change(12).shift(-12)
    df["direction"] = (df["future_return"] > 0).astype(float)
    y = df["direction"].values
    
    # Remove NaN
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    if use_shap:
        console.print("[dim]Using SHAP for feature importance (this may take a minute)...[/dim]")
        
        # Load model for SHAP
        _configure_tf_metal(verbose=verbose)
        import tensorflow as tf
        
        model_path = meta.get("model_path")
        model_type = meta.get("model_type", "keras")
        
        if model_type == "ensemble":
            # For ensemble, use XGBoost component if available
            from ensemble_model import EnsembleModel
            ensemble_path = meta.get("ensemble_path")
            if ensemble_path and Path(ensemble_path).exists():
                ensemble = EnsembleModel.load(ensemble_path)
                # Use XGBoost or RF from ensemble
                if "xgboost" in ensemble.base_models:
                    model = ensemble.base_models["xgboost"].model
                elif "random_forest" in ensemble.base_models:
                    model = ensemble.base_models["random_forest"].model
                else:
                    model = None
            else:
                model = None
        else:
            model = None  # SHAP with Keras is slow, use correlation instead
        
        if model is not None:
            try:
                analyzer = FeatureImportanceAnalyzer(model, feature_columns)
                analyzer.fit(X)
                results = analyzer.analyze(X)
                importance_df = results["importance_df"]
                
                # Save plots if requested
                if save_plots:
                    viz_dir = Path("trained_data") / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    analyzer.plot_importance(max_display=top_n, save_path=str(viz_dir / "shap_importance.png"))
                    console.print(f"[green]Saved plot to {viz_dir / 'shap_importance.png'}[/green]")
                
            except Exception as e:
                console.print(f"[yellow]SHAP analysis failed: {e}[/yellow]")
                use_shap = False
    
    if not use_shap:
        console.print("[dim]Using correlation-based feature importance...[/dim]")
        
        # Calculate correlation with target
        correlations = []
        for i, col in enumerate(feature_columns):
            if i < X.shape[1]:
                try:
                    # Use pandas corr for robustness
                    x_col = pd.Series(X[:, i])
                    y_series = pd.Series(y)
                    corr = x_col.corr(y_series)
                    correlations.append(abs(corr) if not pd.isna(corr) else 0)
                except Exception:
                    correlations.append(0)
            else:
                correlations.append(0)
        
        importance_df = pd.DataFrame({
            "feature": feature_columns,
            "importance": correlations,
            "rank": range(1, len(feature_columns) + 1),
        })
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
    
    # Display results
    console.print("\n" + "=" * 70)
    console.print(f"[bold green]📊 TOP {top_n} MOST IMPORTANT FEATURES[/bold green]")
    console.print("=" * 70)
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Feature", style="bold")
    table.add_column("Importance", justify="right")
    table.add_column("Category", justify="center")
    
    # Categorize features
    def get_category(name: str) -> str:
        name_lower = name.lower()
        if any(x in name_lower for x in ["rsi", "macd", "stoch", "cci", "williams", "mfi"]):
            return "Momentum"
        elif any(x in name_lower for x in ["sma", "ema", "bb_"]):
            return "Trend"
        elif any(x in name_lower for x in ["atr", "volatility", "bb_width"]):
            return "Volatility"
        elif any(x in name_lower for x in ["volume", "obv"]):
            return "Volume"
        elif any(x in name_lower for x in ["hour", "day", "session"]):
            return "Time"
        elif any(x in name_lower for x in ["adx", "trend", "regime"]):
            return "Regime"
        else:
            return "Other"
    
    for _, row in importance_df.head(top_n).iterrows():
        category = get_category(row["feature"])
        cat_color = {
            "Momentum": "yellow",
            "Trend": "blue",
            "Volatility": "red",
            "Volume": "green",
            "Time": "cyan",
            "Regime": "magenta",
            "Other": "white",
        }.get(category, "white")
        
        table.add_row(
            str(int(row["rank"])),
            row["feature"],
            f"{row['importance']:.4f}",
            f"[{cat_color}]{category}[/{cat_color}]",
        )
    
    console.print(table)
    
    # Summary by category
    console.print("\n[bold]📈 IMPORTANCE BY CATEGORY:[/bold]")
    importance_df["category"] = importance_df["feature"].apply(get_category)
    category_importance = importance_df.groupby("category")["importance"].sum().sort_values(ascending=False)
    
    total_importance = category_importance.sum()
    for cat, imp in category_importance.items():
        pct = imp / total_importance * 100 if total_importance > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        console.print(f"  {cat:12} {bar} {pct:.1f}%")
    
    # Recommendations
    console.print("\n[bold]💡 RECOMMENDATIONS:[/bold]")
    
    top_features = importance_df.head(20)["feature"].tolist()
    bottom_features = importance_df.tail(50)["feature"].tolist()
    
    console.print(f"  • Top 20 features explain most of the signal")
    console.print(f"  • Consider removing {len(bottom_features)} low-importance features")
    
    # Feature groups that matter
    top_categories = importance_df.head(30).groupby("category").size().sort_values(ascending=False)
    console.print(f"  • Most important category: {top_categories.index[0]} ({top_categories.iloc[0]} features in top 30)")
    
    console.print("\n" + "=" * 70)


def buddy_monitor(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    show_alerts: bool = False,
    show_drift: bool = False,
    generate_report: bool = False,
    drift_limit: int = 20,
    **kwargs: Any,
) -> None:
    """Display the monitoring dashboard using the Rich-based CLI renderer."""
    from cli.monitoring_cli import run_monitoring_dashboard

    run_monitoring_dashboard(
        config_path,
        show_alerts=show_alerts,
        show_drift=show_drift,
        generate_report=generate_report,
        drift_limit=drift_limit,
    )
