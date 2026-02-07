#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa: E402
"""
ML Engine Trading Bot CLI — thin entry point.

All logic lives in the ``cli/`` package; this file merely wires up
argparse, dispatches the chosen sub-command, and exits.
"""
from __future__ import annotations

# ── TensorFlow noise suppression (must precede any TF import) ──────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_AFFINITY"] = "noverbose"

import logging
import sys
from typing import Any

# ── CLI package ────────────────────────────────────────────────────────────
from cli import (
    console,
    DEFAULT_CONFIG_PATH,
    _normalize_command_args,
    _maybe_run_buddy_interactive_wizard,
    _maybe_launch_buddy_repl,
    _dispatch_buddy,
    _dispatch_train_buddy,
)
from cli.argparser import create_argument_parser
from cli.training import train_buddy
from cli.commands import (
    buddy,
    buddy_loop,
    buddy_validate,
    buddy_test,
)
from cli.buddy_scanning import buddy_scan
from cli.analytics_commands import (
    suggest_improvements,
    buddy_journal,
    buddy_analyze,
    buddy_monitor,
)
from cli.model_management import model_status, promote_model
from cli.training_ops import (
    train_rl_sizer,
    retrain_gates,
    train_rl_gates,
    train_rl_exits,
    retrain_all,
)
from cli.candle_optimizer import find_optimal_candles

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_VERBOSE_MODE: bool = False


# ---------------------------------------------------------------------------
# Command map
# ---------------------------------------------------------------------------
COMMAND_MAP: dict[str, Any] = {
    "train": train_buddy,
    "train-buddy": train_buddy,
    "retrain-gates": retrain_gates,
    "retrain-all": retrain_all,
    "train-rl-sizer": train_rl_sizer,
    "train-rl-gates": train_rl_gates,
    "train-rl-exits": train_rl_exits,
    "buddy": buddy,
    "Buddy": buddy,
    "promote-model": promote_model,
    "model-status": model_status,
    "test": buddy_test,
    "validate": buddy_validate,
    "scan": buddy_scan,
    "analyze": buddy_analyze,
    "journal": buddy_journal,
    "suggest-improvements": suggest_improvements,
    "find-candles": find_optimal_candles,
}


# ---------------------------------------------------------------------------
# Dispatch helpers for specialised sub-commands
# ---------------------------------------------------------------------------

def _dispatch_scan(args: Any) -> None:
    """Handle the scan command (with optional watch mode)."""
    import logging as _logging
    root_logger = _logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(_logging.CRITICAL)
    try:
        watch_mode = bool(getattr(args, "watch", False))
        if watch_mode:
            from src.scanner import Scanner, ScannerConfig, ContinuousScanner

            pairs_str = getattr(args, "pairs", None)
            pair_list = (
                [p.strip().upper().replace("/", "_") for p in pairs_str.split(",")]
                if pairs_str else None
            )
            config = ScannerConfig.from_cli_args(
                config_path=args.config,
                pairs=pair_list,
                granularity=str(getattr(args, "granularity", "H1")),
                top_n=int(getattr(args, "top", 5)),
            )
            scanner = Scanner(config=config)
            interval_minutes = int(getattr(args, "interval", 5))
            auto_execute = bool(getattr(args, "auto_execute", False))
            console.print(f"[cyan]Starting watch mode (interval: {interval_minutes}m)[/cyan]")
            console.print("[dim]Press Ctrl+C to stop[/dim]\n")
            ContinuousScanner(scanner).run(
                pairs=pair_list,
                interval_minutes=interval_minutes,
                auto_execute=auto_execute,
                top_n=int(getattr(args, "top", 5)),
            )
        else:
            buddy_scan(
                args.config,
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                top_n=int(getattr(args, "top", 5)),
                verbose=bool(getattr(args, "verbose", False)),
                prompt_train=not bool(getattr(args, "no_train", False)),
                no_execute=bool(getattr(args, "skip_execute", False)),
                use_rl_sizer=bool(getattr(args, "use_rl_sizer", True)),
                diversified=bool(getattr(args, "diversified", False)),
                force=bool(getattr(args, "force", False)),
            )
    finally:
        root_logger.setLevel(prev_level)


def _dispatch_train_joint(args: Any) -> None:
    """Handle the train-joint command."""
    from src.training.buddy_training_helpers import train_joint_multi_pair_ensemble

    instruments_str = getattr(args, "instruments", "EUR_USD,GBP_USD,USD_JPY")
    instruments = [p.strip().upper().replace("/", "_") for p in instruments_str.split(",")]
    train_joint_multi_pair_ensemble(
        instruments=instruments,
        granularity=str(getattr(args, "granularity", "H1")),
        candles=int(getattr(args, "candles", 15000)),
        fine_tune=bool(getattr(args, "fine_tune", True)),
        fine_tune_threshold=float(getattr(args, "fine_tune_threshold", 0.05)),
        console=console,
    )


def _dispatch_train_tcn_volatility(args: Any) -> None:
    """Handle the train-tcn-volatility command."""
    import numpy as np
    import pandas as pd
    import yaml
    from pathlib import Path
    from src.training.modular_trainers import TCNTrainer, TrainerConfig
    from src.core.modular_data_loaders import load_volatility_regime_data
    from src.utils.oanda_practice import OandaPracticeClient

    MODEL_DIR_PATH = "trained_data/models"
    TCN_VOLATILITY_REGIME_FILENAME = "tcn_volatility_regime.keras"

    console.print("\n[bold cyan]" + "=" * 63 + "[/bold cyan]")
    console.print("[bold cyan]TCN VOLATILITY REGIME TRAINING[/bold cyan]")
    console.print("[bold cyan]" + "=" * 63 + "[/bold cyan]\n")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    instruments_str = getattr(args, "instruments", "EUR_USD,GBP_USD,USD_JPY")
    instruments = [p.strip().upper().replace("/", "_") for p in instruments_str.split(",")]
    candles = int(getattr(args, "candles", 12000))
    granularity = str(getattr(args, "granularity", "H1"))

    console.print(f"  Instruments: {', '.join(instruments)}")
    console.print(f"  Candles: {candles}")
    console.print(f"  Granularity: {granularity}\n")

    client = OandaPracticeClient.from_env()
    dfs: list[pd.DataFrame] = []

    for inst in instruments:
        console.print(f"  Fetching {inst}...")
        inst_rows: list[dict[str, Any]] = []
        remaining = candles
        to_time = None

        while remaining > 0:
            batch_size = min(remaining, 5000)
            kw: dict[str, Any] = {"granularity": granularity, "count": batch_size}
            if to_time:
                kw["to_time"] = to_time
            response = client.get_candles(inst, **kw)
            candles_list = response.get("candles", [])
            if not candles_list:
                break
            for c in candles_list:
                mid = c.get("mid", {})
                inst_rows.append({
                    "time": c.get("time"),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })
            console.print(f"    ... fetched {len(candles_list)} candles")
            remaining -= len(candles_list)
            if candles_list:
                to_time = candles_list[0].get("time")
            if len(candles_list) < batch_size:
                break

        if inst_rows:
            inst_df = pd.DataFrame(inst_rows)
            inst_df["time"] = pd.to_datetime(inst_df["time"])
            inst_df = inst_df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
            dfs.append(inst_df)
            console.print(f"    ✓ {len(inst_df)} total rows")

    if not dfs:
        console.print("[red]No data fetched![/red]")
        sys.exit(1)

    combined_df = pd.concat(dfs, ignore_index=True)
    console.print(f"\n  Combined: {len(combined_df)} rows\n")

    console.print("  Preparing volatility regime labels...")
    vol_cfg = cfg.get("volatility_regime", {})
    result = load_volatility_regime_data(
        combined_df,
        lookback_bars=vol_cfg.get("lookback_bars", 100),
        thresholds=vol_cfg.get("thresholds", [0.25, 0.60, 0.85]),
    )
    X_train, y_train = result["X_train"], result["y_train"]
    X_val, y_val = result["X_val"], result["y_val"]
    console.print(f"    ✓ Train: {X_train.shape}, Val: {X_val.shape}")

    unique, counts = np.unique(y_train, return_counts=True)
    console.print("    Class distribution (train):")
    regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
    for u, c_ in zip(unique, counts):
        console.print(f"      {regime_names[int(u)]}: {c_} ({100 * c_ / len(y_train):.1f}%)")

    console.print("\n[bold]Training TCN Volatility Regime...[/bold]\n")
    trainer_config = TrainerConfig(
        seq_len=cfg.get("model", {}).get("seq_len", 60),
        tcn_hidden_size=cfg.get("model", {}).get("tcn_filters", 32),
        tcn_kernel_size=cfg.get("model", {}).get("tcn_kernel_size", 3),
        tcn_dropout=cfg.get("model", {}).get("dropout", 0.4),
        learning_rate=cfg.get("training", {}).get("learning_rate", 0.0003),
        epochs=cfg.get("training", {}).get("epochs", 100),
        batch_size=cfg.get("training", {}).get("batch_size", 64),
        patience=cfg.get("training", {}).get("early_stopping_patience", 20),
    )
    tcn_trainer = TCNTrainer(trainer_config)
    feature_names = result.get("feature_names", [])

    model_dir = Path(MODEL_DIR_PATH)
    tcn_save_path = model_dir / TCN_VOLATILITY_REGIME_FILENAME
    warm_start_path = str(tcn_save_path) if tcn_save_path.exists() else None

    metrics = tcn_trainer.train(
        X_train, y_train, X_val, y_val,
        feature_names=feature_names,
        warm_start_path=warm_start_path,
        instrument=getattr(args, "instrument", "MULTI"),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    tcn_trainer.save(str(tcn_save_path))

    console.print("\n[green]✓ TCN Volatility Regime model saved![/green]")
    console.print(f"  Path: {tcn_save_path}")
    console.print(f"  Val Accuracy: {metrics.get('val_accuracy', 0):.1%}")
    console.print("\n[dim]Run 'buddy scan' to use the new volatility filter[/dim]\n")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    """Main CLI entry point."""
    global _VERBOSE_MODE

    # Interactive wizard shortcut (no args → wizard)
    if _maybe_run_buddy_interactive_wizard(default_config=DEFAULT_CONFIG_PATH):
        return

    if len(sys.argv) == 1:
        create_argument_parser().print_help()
        sys.exit(0)

    parser = create_argument_parser()
    args = parser.parse_args()

    # Global verbose flag
    _VERBOSE_MODE = bool(getattr(args, "verbose", False))
    if _VERBOSE_MODE:
        import logging as _logging
        for handler in logging.getLogger().handlers:
            if isinstance(handler, _logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(_logging.INFO)

    _normalize_command_args(args)

    # REPL shortcut
    if _maybe_launch_buddy_repl(args):
        return

    command = getattr(args, "command", None)
    if not command:
        parser.print_help()
        sys.exit(0)

    try:
        # ── Training commands ──────────────────────────────────────────
        if command in ("train", "train-buddy"):
            console.print(
                "\n[yellow]⚠ WARNING: train-buddy trains models for a SINGLE pair only.[/yellow]"
            )
            console.print(
                "[yellow]  Scanner requires JOINT-trained models. For scanner usage, run:[/yellow]"
            )
            console.print(
                "[yellow]    python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY[/yellow]\n"
            )
            _dispatch_train_buddy(args, COMMAND_MAP)

        elif command == "retrain-gates":
            retrain_gates(
                config_path=args.config,
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 5000)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        elif command == "retrain-all":
            retrain_all(
                config_path=args.config,
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 15000)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        elif command == "train-rl-sizer":
            train_rl_sizer(
                config_path=args.config,
                timesteps=int(getattr(args, "timesteps", 500_000)),
                episodes=getattr(args, "rl_episodes", None),
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 5000)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        elif command == "train-rl-gates":
            train_rl_gates(
                config_path=args.config,
                timesteps=int(getattr(args, "timesteps", 100_000)),
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 5000)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        elif command == "train-rl-exits":
            train_rl_exits(
                config_path=args.config,
                timesteps=int(getattr(args, "timesteps", 100_000)),
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 5000)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        elif command == "train-joint":
            _dispatch_train_joint(args)

        elif command == "train-tcn-volatility":
            _dispatch_train_tcn_volatility(args)

        # ── Inference / execution ──────────────────────────────────────
        elif command in {"buddy", "Buddy"}:
            _dispatch_buddy(args, COMMAND_MAP)

        # ── Scanning ───────────────────────────────────────────────────
        elif command == "scan":
            _dispatch_scan(args)

        # ── Validation / testing ───────────────────────────────────────
        elif command == "validate":
            buddy_validate(
                args.config,
                instrument=str(getattr(args, "instrument", "EUR_USD")),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 800)),
                lookahead=int(getattr(args, "lookahead", 24)),
                all_pairs=bool(getattr(args, "all_pairs", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        elif command == "test":
            buddy_test(
                args.config,
                instrument=str(getattr(args, "instrument", "USD_JPY")),
                granularity=str(getattr(args, "granularity", "H1")),
                test_candles=int(getattr(args, "candles", 50)),
                min_confidence=float(getattr(args, "min_confidence", 0.0)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        # ── Analytics ──────────────────────────────────────────────────
        elif command == "analyze":
            buddy_analyze(
                args.config,
                top_n=int(getattr(args, "top", 30)),
                save_plots=bool(getattr(args, "save", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        elif command == "journal":
            buddy_journal(
                args.config,
                update=bool(getattr(args, "update", False)),
                days=int(getattr(args, "days", 30)),
                verbose=bool(getattr(args, "verbose", False)),
                import_trades=bool(getattr(args, "import_trades", False)),
            )

        elif command == "suggest-improvements":
            suggest_improvements(
                args.config,
                instrument=getattr(args, "instrument", None),
                auto_apply=bool(getattr(args, "auto_apply", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        elif command == "monitor":
            buddy_monitor(
                args.config,
                show_alerts=bool(getattr(args, "monitor_alerts", False)),
                show_drift=bool(getattr(args, "monitor_drift", False)),
                generate_report=bool(getattr(args, "monitor_report", False)),
                drift_limit=int(getattr(args, "monitor_limit", 20)),
            )

        # ── Model management ──────────────────────────────────────────
        elif command == "promote-model":
            promote_model(args.config)

        elif command == "model-status":
            model_status(args.config)

        # ── Candle optimizer ───────────────────────────────────────────
        elif command == "find-candles":
            find_optimal_candles(
                args.config,
                instrument=str(getattr(args, "instrument", "EUR_USD")),
                granularity=str(getattr(args, "granularity", "H1")),
                min_candles=int(getattr(args, "min_candles", 3000)),
                max_candles=int(getattr(args, "max_candles", 30000)),
                step=int(getattr(args, "step", 3000)),
                n_folds=int(getattr(args, "cv_folds", 3)),
                auto_train=not bool(getattr(args, "no_auto_train", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )

        # ── Fallback ──────────────────────────────────────────────────
        else:
            handler = COMMAND_MAP.get(command)
            if handler:
                handler(args.config)
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                parser.print_help()
                sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logging.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
