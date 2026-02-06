#!/usr/bin/env python3
"""
Enhanced ML Engine Trading Bot CLI - Entry Point

Modularized CLI for the ML trading engine.
See cli/ package for implementations.
"""
from __future__ import annotations

# Suppress TensorFlow CPU instruction warnings (irrelevant on Apple Silicon/Metal)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import logging

# Import from CLI package
from cli.io_utils import console, DEFAULT_CONFIG_PATH
from cli.argparser import create_argument_parser
from cli.training import train_buddy
from cli.commands import (
    buddy, buddy_loop, buddy_scan, buddy_monitor, buddy_journal,
    buddy_analyze, buddy_validate, buddy_test, model_status, promote_model,
    train_rl_sizer, retrain_gates, suggest_improvements,
    _dispatch_buddy, _dispatch_train_buddy, _normalize_command_args,
    _maybe_run_buddy_interactive_wizard, _maybe_launch_buddy_repl,
)


def main() -> None:
    """Main CLI entry point."""
    parser = create_argument_parser()

    # (The installed `buddy` launcher calls: python main.py buddy ...)
    if _maybe_run_buddy_interactive_wizard(default_config=DEFAULT_CONFIG_PATH):
        return

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    _normalize_command_args(args)

    # If user requested direct REPL, handle it immediately (before heavy dispatch).
    if _maybe_launch_buddy_repl(args):
        return

    command_map = {
        "train": train_buddy,  # Short alias
        "train-buddy": train_buddy,
        "retrain-gates": retrain_gates,
        "train-rl-sizer": train_rl_sizer,
        "buddy": buddy,
        "Buddy": buddy,
        "promote-model": promote_model,
        "model-status": model_status,
        "test": buddy_test,  # Legacy - redirects to validate
        "validate": buddy_validate,  # Professional validation
        "scan": buddy_scan,
        "analyze": buddy_analyze,
        "journal": buddy_journal,
        "monitor": buddy_monitor,
        "suggest-improvements": suggest_improvements,
    }

    try:
        if args.command in ("train", "train-buddy"):
            _dispatch_train_buddy(args, command_map)
        elif args.command == "retrain-gates":
            # Retrain sklearn gates only (XGBoost, RF, Ridge)
            # Keeps Transformer direction model unchanged
            retrain_gates(
                config_path=args.config,
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 5000)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command == "train-rl-sizer":
            # Train RL position sizing agent
            train_rl_sizer(
                config_path=args.config,
                timesteps=int(getattr(args, "timesteps", 500_000)),
                episodes=getattr(args, "rl_episodes", None),
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 5000)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command in {"buddy", "Buddy"}:
            _dispatch_buddy(args, command_map)
        elif args.command == "promote-model":
            promote_model(args.config)
        elif args.command == "model-status":
            model_status(args.config)
        elif args.command == "test":
            buddy_test(
                args.config,
                instrument=str(getattr(args, "instrument", "USD_JPY")),
                granularity=str(getattr(args, "granularity", "H1")),
                test_candles=int(getattr(args, "candles", 50)),
                min_confidence=float(getattr(args, "min_confidence", 0.0)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command == "validate":
            buddy_validate(
                args.config,
                instrument=str(getattr(args, "instrument", "EUR_USD")),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 800)),
                lookahead=int(getattr(args, "lookahead", 24)),
                all_pairs=bool(getattr(args, "all_pairs", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command == "scan":
            watch_mode = bool(getattr(args, "watch", False))
            
            if watch_mode:
                # Continuous watch mode
                from buddy_scanner import BuddyScanner
                
                scanner = BuddyScanner(
                    config_path=args.config,
                    use_rl_sizer=bool(getattr(args, "use_rl_sizer", False)),
                )
                
                pairs_str = getattr(args, "pairs", None)
                pair_list = None
                if pairs_str:
                    pair_list = [p.strip().upper().replace("/", "_") for p in pairs_str.split(",")]
                
                scanner.scan_continuous(
                    pairs=pair_list,
                    granularity=str(getattr(args, "granularity", "H1")),
                    interval_minutes=int(getattr(args, "interval", 5)),
                    auto_execute=bool(getattr(args, "auto_execute", False)),
                    top_n=int(getattr(args, "top", 5)),
                )
            else:
                # Single scan
                buddy_scan(
                    args.config,
                    pairs=str(getattr(args, "pairs", "")) if getattr(args, "pairs", None) else None,
                    granularity=str(getattr(args, "granularity", "H1")),
                    top_n=int(getattr(args, "top", 5)),
                    verbose=bool(getattr(args, "verbose", False)),
                    prompt_train=not bool(getattr(args, "no_train", False)),
                    no_execute=bool(getattr(args, "skip_execute", False)),
                    use_rl_sizer=bool(getattr(args, "use_rl_sizer", False)),
                    diversified=bool(getattr(args, "diversified", False)),
                )
        elif args.command == "trade-analysis":
            # Trade analysis command
            from trade_analyzer import TradeAnalyzer
            from pathlib import Path
            
            analyzer = TradeAnalyzer()
            report = analyzer.analyze_recent(n=int(getattr(args, "last", 50)))
            
            # Print markdown report
            console.print(analyzer.format_markdown(report))
            
            # Export if requested
            export_path = getattr(args, "export", None)
            if export_path:
                import json
                with open(export_path, 'w') as f:
                    json.dump(analyzer.to_dict(report), f, indent=2)
                console.print(f"\n[green]✅ Report exported to {export_path}[/green]")
        elif args.command == "analyze":
            buddy_analyze(
                args.config,
                top_n=int(getattr(args, "top", 30)),
                save_plots=bool(getattr(args, "save", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command == "journal":
            buddy_journal(
                args.config,
                update=bool(getattr(args, "update", False)),
                days=int(getattr(args, "days", 30)),
                verbose=bool(getattr(args, "verbose", False)),
                import_trades=bool(getattr(args, "import_trades", False)),
            )
        elif args.command == "monitor":
            buddy_monitor(
                args.config,
                show_alerts=bool(getattr(args, "monitor_alerts", False)),
                show_drift=bool(getattr(args, "monitor_drift", False)),
                generate_report=bool(getattr(args, "monitor_report", False)),
                drift_limit=int(getattr(args, "monitor_limit", 20)),
            )
        elif args.command == "suggest-improvements":
            suggest_improvements(
                args.config,
                instrument=getattr(args, "instrument", None),
                auto_apply=bool(getattr(args, "auto_apply", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        else:
            command_map[args.command](args.config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logging.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
