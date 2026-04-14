#!/usr/bin/env python3
# flake8: noqa: E402
"""
ML Engine Trading Bot CLI — thin entry point.

All logic lives in the ``cli/`` package; this file merely wires up
argparse, dispatches the chosen sub-command, and exits.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: macOS OpenMP/MKL/LLVM Conflict Resolution
# ═══════════════════════════════════════════════════════════════════════════════
# MUST be set BEFORE any imports (especially numpy, tensorflow, scipy).
# These prevent Segmentation Fault 11 on macOS when using Intel-optimized
# libraries with multiprocessing enabled.
#
# Issue: Multiple OpenMP runtimes (conda's libomp vs system LLVM vs MKL)
# conflict when spawning threads, causing SIGSEGV.
#
# References:
# - https://github.com/numpy/numpy/issues/22918
# - https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib
# ═══════════════════════════════════════════════════════════════════════════════
import os
import platform

# Set these BEFORE any other imports
if platform.system() == "Darwin":  # macOS
    # Allow multiple OpenMP libraries (prevents "libiomp5.dylib already initialized")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Use GNU OpenMP threading layer (compatible with conda's libomp)
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    # Disable MKL affinity (prevents conflicts with macOS scheduler)
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    # Reduced from 4 → 2: fewer threads = less memory pressure per scan
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

    # ── TensorFlow Metal memory guard (CRITICAL for 8GB M1) ─────────────────
    # TF Metal uses UNIFIED memory — same pool as CPU, OS, WindowServer.
    # Without limits, TF grows until macOS compressor hits 100%, causing
    # kernel panic (userspace watchdog timeout, WindowServer crash).
    #
    # Strategy: disable Metal GPU for INFERENCE entirely.
    # Scan inference models (Transformer, TCN) are small; CPU is fast enough
    # for a 5-minute cycle. Eliminates all Metal/unified-memory contention.
    #
    # Training (--train) still re-enables Metal via configure_metal_runtime().
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    # Disable Metal device for inference processes (scan/watch/execute)
    # Training explicitly overrides this before model construction.
    os.environ.setdefault("ML_ENGINE_DISABLE_METAL", "1")

# ── GC tuning for long-running scan loop ───────────────────────────────────
# Default (700, 10, 10) is too aggressive; 50000 was too lazy (OOM on 8GB M1).
# 5000 balances: lets short-lived scan objects batch up slightly, but fires
# before transient DataFrames/numpy arrays accumulate into memory pressure.
import gc as _gc
_gc.set_threshold(5000, 10, 10)

# ── Process memory ceiling (CRITICAL for 8GB M1) ────────────────────────────
# Hard RSS cap at 3.5 GB. Python raises MemoryError cleanly rather than
# letting the OS compressor fill up and watchdog-kill WindowServer.
import resource as _resource
_GB = 1024 ** 3
try:
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_RSS)
    # Only set if no stricter limit already in place
    _target = int(3.5 * _GB)
    if _hard == _resource.RLIM_INFINITY or _hard > _target:
        _resource.setrlimit(_resource.RLIMIT_RSS, (_target, _target))
except Exception:
    pass  # Non-fatal: RLIMIT_RSS is advisory on macOS but still helps

# ── Suppress noisy third-party warnings (must precede library imports) ─────
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# ── TensorFlow noise suppression (must precede any TF import) ──────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import sys
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── CLI package ────────────────────────────────────────────────────────────
from cli import (
    console,
    DEFAULT_CONFIG_PATH,
    _normalize_command_args,
    _maybe_run_buddy_interactive_wizard,
    _maybe_launch_buddy_repl,
    _maybe_launch_buddy_chat,
    _dispatch_buddy,
    _dispatch_train_buddy,
)
from cli.argparser import create_argument_parser
from cli.training import train_buddy
from cli.commands import (
    buddy,
    buddy_validate,
    buddy_test,
    handle_transfer_command,
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
from cli.learn_commands import handle_learn

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
    "buddy-chat": buddy,
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

def _process_futures_and_instruments_flags(args: Any) -> None:
    """Process --futures flag and --instruments argument.

    Sets broker, profile, and pairs based on --futures and --instruments flags.
    This is called after argument parsing but before dispatch.
    """
    # Handle --futures shorthand: sets broker=ibkr, profile=futures_paper
    if getattr(args, "futures", False):
        args.broker = "ibkr"
        if str(getattr(args, "profile", "balanced")) == "balanced":
            args.profile = "futures_paper"
        # Log the transformation
        logger.info("[yellow]--futures flag detected: setting broker=ibkr, profile=futures_paper[/yellow]")

    # Handle --instruments: override pairs with comma-separated list
    instruments_str = getattr(args, "instruments", None)
    if instruments_str:
        args.pairs = instruments_str
        logger.info(f"[yellow]--instruments flag: overriding pairs to {instruments_str}[/yellow]")


def _dispatch_scan(args: Any) -> None:
    """Handle the scan command (with optional watch mode)."""
    import logging as _logging
    root_logger = _logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(_logging.ERROR)
    try:
        watch_mode = bool(getattr(args, "watch", False))
        if watch_mode:
            from src.scanner import Scanner, ScannerConfig, ContinuousScanner

            pairs_str = getattr(args, "pairs", None)
            pair_list = (
                [p.strip().upper().replace("/", "_") for p in pairs_str.split(",")]
                if pairs_str else None
            )
            profile = str(getattr(args, "profile", "balanced"))
            config = ScannerConfig.from_cli_args(
                config_path=args.config,
                pairs=pair_list,
                granularity=str(getattr(args, "granularity", "H1")),
                top_n=int(getattr(args, "top", 5)),
                profile=profile,
                force=bool(getattr(args, "force", False)),
            )
            # Note: apply_profile is already called inside from_cli_args.
            # Do NOT call it again here — it would re-enable session_filter,
            # overwriting the --force flag.
            scanner = Scanner(config=config)
            interval_minutes = int(getattr(args, "interval", 5))
            auto_execute = bool(getattr(args, "auto_execute", False))
            enforce_level = int(getattr(args, "enforce_policy", 0))
            supervision_mode = bool(getattr(args, "supervision", False))
            if supervision_mode:
                console.print(f"[bold cyan]Starting supervision mode (30s ticks)[/bold cyan]")
                config.enable_supervision_mode = True
            else:
                console.print(f"[cyan]Starting watch mode (interval: {interval_minutes}m)[/cyan]")
            if enforce_level > 0:
                console.print(f"[yellow]Policy enforcement level: {enforce_level}[/yellow]")
            console.print("[dim]Press Ctrl+C to stop[/dim]\n")
            cs = ContinuousScanner(scanner)
            # Tier 7: Set enforcement level if specified
            if enforce_level > 0 and cs._policy_engine is not None:
                cs._policy_engine.set_enforcement_level(enforce_level)
            cs.run(
                pairs=pair_list,
                interval_minutes=interval_minutes,
                auto_execute=auto_execute,
                top_n=int(getattr(args, "top", 5)),
            )
        else:
            profile = str(getattr(args, "profile", "balanced"))
            buddy_scan(
                args.config,
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                top_n=int(getattr(args, "top", 5)),
                verbose=bool(getattr(args, "verbose", False)),
                prompt_train=not bool(getattr(args, "no_train", False)),
                no_execute=bool(getattr(args, "skip_execute", False)),
                auto_execute=bool(getattr(args, "auto_execute", False)),
                clean_output=bool(getattr(args, "clean_output", False)),
                use_rl_sizer=bool(getattr(args, "use_rl_sizer", True)),
                diversified=bool(getattr(args, "diversified", False)),
                force=bool(getattr(args, "force", False)),
                profile=profile,
                run_agent_confirmation=(profile == "smart"),
            )
    finally:
        root_logger.setLevel(prev_level)


def _dispatch_train_joint(args: Any) -> None:
    """Handle the train-joint command."""
    from src.training.buddy_training_helpers import train_joint_multi_pair_ensemble

    instruments_str = getattr(args, "instruments", "EUR_USD,GBP_USD,USD_JPY")
    instruments = [p.strip().upper().replace("/", "_") for p in instruments_str.split(",")]

    # Expand "all" to the full list of default pairs
    if len(instruments) == 1 and instruments[0] == "ALL":
        from src.scanner.config import DEFAULT_PAIRS
        instruments = DEFAULT_PAIRS.copy()
        logger.info(f"Expanded 'all' to {len(instruments)} pairs: {', '.join(instruments)}")

    train_joint_multi_pair_ensemble(
        instruments=instruments,
        granularity=str(getattr(args, "granularity", "H1")),
        candles=int(getattr(args, "candles", 25000)),
        fine_tune=bool(getattr(args, "fine_tune", True)),
        fine_tune_threshold=float(getattr(args, "fine_tune_threshold", 0.05)),
        console=console,
    )


def _fetch_instrument_candles(
    client: Any,
    inst: str,
    candles: int,
    granularity: str,
) -> list[dict[str, Any]]:
    """Fetch candles for a single instrument in paginated batches."""
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
        to_time = candles_list[0].get("time")
        if len(candles_list) < batch_size:
            break

    return inst_rows


def _fetch_all_instruments(
    instruments: list[str],
    candles: int,
    granularity: str,
) -> "pd.DataFrame":
    """Fetch and combine candle data for multiple instruments."""
    import pandas as pd
    from src.utils.oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    dfs: list[pd.DataFrame] = []

    for inst in instruments:
        console.print(f"  Fetching {inst}...")
        inst_rows = _fetch_instrument_candles(client, inst, candles, granularity)
        if not inst_rows:
            continue
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
    return combined_df


def _build_tcn_trainer_config(cfg: dict[str, Any]) -> Any:
    """Build a TrainerConfig from the YAML configuration dict."""
    from src.training.modular_trainers import TrainerConfig

    return TrainerConfig(
        seq_len=cfg.get("model", {}).get("seq_len", 60),
        tcn_hidden_size=cfg.get("model", {}).get("tcn_filters", 32),
        tcn_kernel_size=cfg.get("model", {}).get("tcn_kernel_size", 3),
        tcn_dropout=cfg.get("model", {}).get("dropout", 0.4),
        learning_rate=cfg.get("training", {}).get("learning_rate", 0.0003),
        epochs=cfg.get("training", {}).get("epochs", 100),
        batch_size=cfg.get("training", {}).get("batch_size", 64),
        patience=cfg.get("training", {}).get("early_stopping_patience", 20),
    )


def _dispatch_train_tcn_volatility(args: Any) -> None:
    """Handle the train-tcn-volatility command."""
    import numpy as np
    import yaml
    from pathlib import Path
    from src.training.modular_trainers import TCNTrainer
    from src.core.modular_data_loaders import load_volatility_regime_data

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

    combined_df = _fetch_all_instruments(instruments, candles, granularity)

    console.print("  Preparing volatility regime labels...")
    vol_cfg = cfg.get("volatility_regime", {})
    result = load_volatility_regime_data(
        combined_df,
        lookback_bars=vol_cfg.get("lookback_bars", 100),
        thresholds=vol_cfg.get("thresholds", [0.25, 0.60, 0.85]),
    )
    x_train, y_train = result["X_train"], result["y_train"]
    x_val, y_val = result["X_val"], result["y_val"]
    console.print(f"    ✓ Train: {x_train.shape}, Val: {x_val.shape}")

    unique, counts = np.unique(y_train, return_counts=True)
    console.print("    Class distribution (train):")
    regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
    for u, c_ in zip(unique, counts):
        console.print(f"      {regime_names[int(u)]}: {c_} ({100 * c_ / len(y_train):.1f}%)")

    console.print("\n[bold]Training TCN Volatility Regime...[/bold]\n")
    trainer_config = _build_tcn_trainer_config(cfg)
    tcn_trainer = TCNTrainer(trainer_config)
    feature_names = result.get("feature_names", [])

    model_dir = Path(MODEL_DIR_PATH)
    tcn_save_path = model_dir / TCN_VOLATILITY_REGIME_FILENAME
    warm_start_path = str(tcn_save_path) if tcn_save_path.exists() else None

    metrics = tcn_trainer.train(
        x_train, y_train, x_val, y_val,
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
# Per-command dispatch handlers
# ---------------------------------------------------------------------------

def _handle_train(args: Any) -> None:
    """Handle train / train-buddy command."""
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


def _handle_retrain_gates(args: Any) -> None:
    retrain_gates(
        config_path=args.config,
        pairs=getattr(args, "pairs", None),
        granularity=str(getattr(args, "granularity", "H1")),
        candles=int(getattr(args, "candles", 5000)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_retrain_all(args: Any) -> None:
    retrain_all(
        config_path=args.config,
        granularity=str(getattr(args, "granularity", "H1")),
        candles=int(getattr(args, "candles", 25000)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_train_rl_sizer(args: Any) -> None:
    timesteps = getattr(args, "timesteps", None)
    train_rl_sizer(
        config_path=args.config,
        timesteps=(int(timesteps) if timesteps is not None else None),
        episodes=getattr(args, "rl_episodes", None),
        pairs=getattr(args, "pairs", None),
        granularity=str(getattr(args, "granularity", "H1")),
        candles=int(getattr(args, "candles", 5000)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_train_rl_gates(args: Any) -> None:
    timesteps = getattr(args, "timesteps", None)
    train_rl_gates(
        config_path=args.config,
        timesteps=(int(timesteps) if timesteps is not None else None),
        pairs=getattr(args, "pairs", None),
        granularity=str(getattr(args, "granularity", "H1")),
        candles=int(getattr(args, "candles", 5000)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_train_rl_exits(args: Any) -> None:
    timesteps = getattr(args, "timesteps", None)
    train_rl_exits(
        config_path=args.config,
        timesteps=(int(timesteps) if timesteps is not None else None),
        pairs=getattr(args, "pairs", None),
        granularity=str(getattr(args, "granularity", "H1")),
        candles=int(getattr(args, "candles", 5000)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_buddy(args: Any) -> None:
    _dispatch_buddy(args, COMMAND_MAP)


def _handle_validate(args: Any) -> None:
    buddy_validate(
        args.config,
        instrument=str(getattr(args, "instrument", "EUR_USD")),
        granularity=str(getattr(args, "granularity", "H1")),
        candles=int(getattr(args, "candles", 800)),
        lookahead=int(getattr(args, "lookahead", 24)),
        all_pairs=bool(getattr(args, "all_pairs", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_test(args: Any) -> None:
    buddy_test(
        args.config,
        instrument=str(getattr(args, "instrument", "USD_JPY")),
        granularity=str(getattr(args, "granularity", "H1")),
        test_candles=int(getattr(args, "candles", 50)),
        min_confidence=float(getattr(args, "min_confidence", 0.0)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_analyze(args: Any) -> None:
    buddy_analyze(
        args.config,
        top_n=int(getattr(args, "top", 30) or 30),
        save_plots=bool(getattr(args, "save", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_journal(args: Any) -> None:
    buddy_journal(
        args.config,
        update=bool(getattr(args, "update", False)),
        days=int(getattr(args, "days", 30)),
        verbose=bool(getattr(args, "verbose", False)),
        import_trades=bool(getattr(args, "import_trades", False)),
    )


def _handle_suggest_improvements(args: Any) -> None:
    suggest_improvements(
        args.config,
        instrument=getattr(args, "instrument", None),
        auto_apply=bool(getattr(args, "auto_apply", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _handle_monitor(args: Any) -> None:
    buddy_monitor(
        args.config,
        show_alerts=bool(getattr(args, "monitor_alerts", False)),
        show_drift=bool(getattr(args, "monitor_drift", False)),
        generate_report=bool(getattr(args, "monitor_report", False)),
        drift_limit=int(getattr(args, "monitor_limit", 20)),
        training_mode=bool(getattr(args, "training", False)),
        show_model=getattr(args, "model", None),
        replay_file=getattr(args, "replay", None),
    )


def _handle_promote_model(args: Any) -> None:
    promote_model(args.config)


def _handle_model_status(args: Any) -> None:
    model_status(args.config, decisions=bool(getattr(args, "decisions", False)))


def _handle_find_candles(args: Any) -> None:
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


def _handle_transfer(args: Any) -> None:
    """Handle the transfer subcommand."""
    handle_transfer_command(args)


def _handle_status(args: Any) -> None:
    """Handle the status command with optional cli_display rendering."""
    try:
        from src.scanner.cli_display import render_status as _render_status
        from scripts.buddy_status import (
            build_status_diagnostics as _build_status_diagnostics,
            build_status_risk_snapshot as _build_status_risk_snapshot,
        )

        # Gather data for the rich display
        _acct: dict = {}
        _trades: list = []
        _journal: list = []
        _weights: dict = {}
        _diagnostics: dict = {}
        _risk_scaler: dict = {}
        try:
            from src.utils.oanda_practice import OandaPracticeClient
            _client = OandaPracticeClient.from_env()
            _acct = _client.get_account_summary() or {}
        except Exception:
            pass
        try:
            from src.utils.oanda_practice import OandaPracticeClient
            _client = OandaPracticeClient.from_env()
            _trades = _client.get_open_trades() or []
        except Exception:
            pass
        try:
            import json as _json_s
            from pathlib import Path as _Path_s
            _jp = _Path_s("trained_data/trade_journal_rl.json")
            if _jp.exists():
                _jraw = _json_s.loads(_jp.read_text())
                _journal = _jraw if isinstance(_jraw, list) else []
        except Exception:
            pass
        try:
            import json as _json_w
            from pathlib import Path as _Path_w
            _wp = _Path_w("trained_data/models/agent_weights.json")
            if _wp.exists():
                _wraw = _json_w.loads(_wp.read_text())
                _weights = _wraw if isinstance(_wraw, dict) else {}
        except Exception:
            pass
        try:
            _diagnostics = _build_status_diagnostics()
        except Exception:
            pass
        try:
            _risk_scaler = _build_status_risk_snapshot()
        except Exception:
            pass
        try:
            from src.scanner.automation.background_activity import get_background_activity_tracker
            _background_activity = get_background_activity_tracker().get_snapshot(limit=10)
        except Exception:
            _background_activity = {}

        # Map open trades to positions format
        _positions = []
        for _t in _trades:
            _positions.append({
                "pair": _t.get("instrument", ""),
                "direction": "LONG" if int(_t.get("currentUnits", 0)) > 0 else "SHORT",
                "lots": abs(int(_t.get("currentUnits", 0))) / 100000.0,
                "entry_price": float(_t.get("price", 0)),
                "pnl": float(_t.get("unrealizedPL", 0)),
            })

        # Map recent closed journal entries to display format.
        _recent = []
        _closed_entries = [
            _rt for _rt in _journal
            if isinstance(_rt.get("outcome"), dict)
        ]
        for _rt in reversed(_closed_entries[-10:]):
            _outcome = _rt.get("outcome") or {}
            _recent.append({
                "pair": _rt.get("pair", ""),
                "won": bool(_outcome.get("trade_won", False)),
                "pnl_dollars": float(_outcome.get("realized_pl", _rt.get("pnl", 0) or 0) or 0),
                "pnl_pips": float(_outcome.get("pnl_pips", _rt.get("pnl_pips", 0) or 0) or 0),
                "exit_reason": str(_outcome.get("exit_reason", _rt.get("exit_reason", "")) or ""),
                "closed_at": str(
                    _outcome.get("close_time")
                    or _outcome.get("exit_time")
                    or _rt.get("closed_at")
                    or ""
                )[:19],
            })

        _render_status(
            nav=float(_acct.get("NAV", _acct.get("nav", 0))),
            balance=float(_acct.get("balance", 0)),
            unrealized=float(_acct.get("unrealizedPL", 0)),
            positions=_positions if _positions else None,
            recent_trades=_recent if _recent else None,
            agent_weights=_weights if _weights else None,
            diagnostics=_diagnostics if _diagnostics else None,
            risk_scaler=_risk_scaler if _risk_scaler else None,
            blocked_pairs=(_diagnostics.get("blocked_pairs") if isinstance(_diagnostics, dict) else None),
            background_activity=_background_activity if _background_activity else None,
        )
    except ImportError:
        # Fallback: use the legacy status script
        __import__("scripts.buddy_status", fromlist=["main"]).main()
    except Exception as _st_err:
        logger.debug("cli_display status error: %s", _st_err)
        # Fallback to legacy
        __import__("scripts.buddy_status", fromlist=["main"]).main()


def _handle_feedback_status(args: Any) -> None:
    """Show feedback loop health status."""
    import json as _json
    from src.scanner.feedback.diagnostics import PostTradeDiagnostics
    diag = PostTradeDiagnostics().run()
    # Aggregate additional context
    result = {
        "status": diag.get("status", "UNKNOWN"),
        "issues": diag.get("issues", []),
        "recommended_actions": diag.get("recommended_actions", []),
        "trades_analyzed": diag.get("metadata", {}).get("trades_analyzed", 0),
    }
    # RL model age
    import os as _os
    from pathlib import Path as _Path
    from datetime import datetime as _dt, timezone as _tz
    rl_path = _Path("trained_data/models/rl_position_sizer.zip")
    if rl_path.exists():
        age_s = _dt.now(_tz.utc).timestamp() - _os.path.getmtime(rl_path)
        result["rl_model_age_hours"] = round(max(0.0, age_s) / 3600.0, 1)
    else:
        result["rl_model_age_hours"] = None
    # Agent weights summary
    weights_path = _Path("trained_data/models/agent_weights.json")
    if weights_path.exists():
        try:
            data = _json.loads(weights_path.read_text())
            count = sum(len(b) for k, b in data.items() if not k.startswith("_") and isinstance(b, dict))
            result["agent_weights_count"] = count
        except Exception:
            result["agent_weights_count"] = 0
    if getattr(args, "json_output", False):
        print(_json.dumps(result, indent=2, default=str))
    else:
        print("Feedback Loop Status: %s" % result["status"])
        print("Trades analyzed: %d" % result["trades_analyzed"])
        print("RL model age: %s hours" % result["rl_model_age_hours"])
        print("Agent weight entries: %s" % result.get("agent_weights_count", "?"))
        if result["issues"]:
            print("Issues:")
            for i in result["issues"]:
                print("  - [%s] %s: %s" % (i.get("severity", "?"), i.get("check", "?"), i.get("detail", "")))
        if result["recommended_actions"]:
            print("Recommended actions: %s" % ", ".join(result["recommended_actions"]))


def _handle_train_offline_sizer(args: Any) -> None:
    """Train the offline IQL position sizer."""
    from src.training.rl.offline_sizer import train_offline_sizer
    journal = getattr(args, "journal", None)
    epochs = int(getattr(args, "iql_epochs", 100))
    dry_run = bool(getattr(args, "iql_dry_run", False))
    try:
        result = train_offline_sizer(
            journal_path=journal,
            n_epochs=epochs,
            dry_run=dry_run,
        )
        print("Offline IQL sizer: %s" % ("dry-run complete" if dry_run else "saved to %s" % result))
    except Exception as e:
        print("Error: %s" % e)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Dispatch table  (command string → handler)
# ---------------------------------------------------------------------------
_DISPATCH_TABLE: dict[str, Any] = {
    "train": _handle_train,
    "train-buddy": _handle_train,
    "retrain-gates": _handle_retrain_gates,
    "retrain-all": _handle_retrain_all,
    "train-rl-sizer": _handle_train_rl_sizer,
    "train-rl-gates": _handle_train_rl_gates,
    "train-rl-exits": _handle_train_rl_exits,
    "train-joint": _dispatch_train_joint,
    "train-tcn-volatility": _dispatch_train_tcn_volatility,
    "buddy": _handle_buddy,
    "Buddy": _handle_buddy,
    "buddy-chat": _handle_buddy,
    "scan": _dispatch_scan,
    "validate": _handle_validate,
    "test": _handle_test,
    "analyze": _handle_analyze,
    "journal": _handle_journal,
    "suggest-improvements": _handle_suggest_improvements,
    "monitor": _handle_monitor,
    "promote-model": _handle_promote_model,
    "model-status": _handle_model_status,
    "find-candles": _handle_find_candles,
    "transfer": _handle_transfer,
    "learn": handle_learn,
    "status": lambda args: _handle_status(args),
    "feedback-status": _handle_feedback_status,
    "train-offline-sizer": _handle_train_offline_sizer,
}


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _configure_verbose_logging() -> None:
    """Enable verbose logging on stdout handlers."""
    import logging as _logging
    for handler in logging.getLogger().handlers:
        if isinstance(handler, _logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(_logging.INFO)


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
        _configure_verbose_logging()

    # Process --futures and --instruments flags (before _normalize_command_args)
    _process_futures_and_instruments_flags(args)

    _normalize_command_args(args)

    # REPL shortcut
    if _maybe_launch_buddy_repl(args) or _maybe_launch_buddy_chat(args):
        return

    command = getattr(args, "command", None)
    if not command:
        parser.print_help()
        sys.exit(0)

    try:
        handler = _DISPATCH_TABLE.get(command) or COMMAND_MAP.get(command)
        if handler:
            handler(args)
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
