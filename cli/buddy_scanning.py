#!/usr/bin/env python3
"""Buddy scanning functionality for ML Engine Trading Bot.

This module contains the multi-pair scanning and prediction functions:
- buddy_scan: Enhanced scanner with BuddyScanner integration
- buddy_predict_78: Single-pair prediction using the 78% model
- _buddy_scan_legacy: Legacy fallback scanner using technical indicators
"""
from __future__ import annotations

import os
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from rich.table import Table

from cli.io_utils import (
    console,
    DEFAULT_CONFIG_PATH,
    _parse_bool_answer,
)
from cli.wizard import _buddy_wizard_ask
from cli.config import BuddyTrainingOptions, OandaFetchOptions
from src.utils import load_config


def buddy_scan(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    pairs: Optional[str] = None,
    granularity: str = "H1",
    top_n: int = 5,
    verbose: bool = False,
    prompt_train: bool = False,
    diversified: bool = False,
    force: bool = False,
    profile: str = "balanced",
    no_execute: bool = False,
    auto_execute: bool = False,
    clean_output: bool = False,
    run_agent_confirmation: bool = False,
    session_filter_enabled_override: Optional[bool] = None,
    session_window_utc_override: Optional[tuple[int, int]] = None,
    **kwargs: Any,
) -> list:
    """
    Scan multiple FX pairs to find the best trading opportunities.

    UPGRADED v2.0: Now uses BuddyScanner with:
    - Parallel pair scanning (4x faster)
    - Integrated position sizing (lots, SL/TP)
    - 50-candle quick backtesting
    - Drift detection with retraining prompts
    - Correlation analysis for diversification
    - Historical accuracy tracking
    - Shows tradeable pairs vs needs training

    Uses pair-specific models for predictions:
    - Transformer direction model for signal
    - XGBoost momentum for confirmation
    - Ridge confidence scoring
    - RF risk assessment

    Falls back to technical indicators if model unavailable.

    NOTE: By default scanner displays results only. Use '--auto-execute' to
    place trades for passing/agent-promoted setups.

    Args:
        diversified: If True, auto-filter to show only best pair per correlation cluster
    """
    try:
        # Use new enhanced BuddyScanner
        from buddy_scanner import BuddyScanner, suppress_logging

        # Don't override account_equity unless explicitly passed
        # Let config file value be used by default
        explicit_equity = kwargs.get("account_equity")
        use_rl_sizer = kwargs.get("use_rl_sizer", True)  # RL sizer enabled by default

        # Suppress all verbose logging — only scanner's own output should show
        with suppress_logging():
            scanner = BuddyScanner(
                config_path=config_path,
                account_equity=explicit_equity,  # None = use config
                use_rl_sizer=use_rl_sizer,
            )

        # Parse pairs
        pair_list = None
        if pairs:
            pair_list = [p.strip().upper().replace("/", "_") for p in pairs.split(",")]

        # Run enhanced scan (display only, no execution prompt)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Argument `input_shape` is deprecated\. Use `shape` instead\.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*structure of `inputs` doesn't match the expected structure.*",
                category=UserWarning,
            )
            with suppress_logging(level=logging.ERROR):
                results = scanner.scan(
                    pairs=pair_list,
                    granularity=granularity,
                    top_n=top_n,
                    verbose=True,  # Always verbose for CLI
                    clean_output=clean_output,
                    prompt_train=prompt_train,
                    diversified=diversified,
                    force=force,
                    profile=profile,
                    run_agent_confirmation=run_agent_confirmation,
                    session_filter_enabled_override=session_filter_enabled_override,
                    session_window_utc_override=session_window_utc_override,
                )

        if auto_execute and not no_execute:
            try:
                # Enable execution path for this scan call.
                scanner._scanner.config.enable_execution = True
                tradeable_candidates = [
                    r for r in results
                    if r.direction in {"LONG", "SHORT"} and r.gates_passed and r.error is None
                ]
                confirmed_candidates = [
                    r for r in tradeable_candidates
                    if r.agent_total > 0 and r.agent_passed
                ]
                execution_candidates = confirmed_candidates or tradeable_candidates

                if execution_candidates:
                    execution_results = scanner._scanner.execute_trades(
                        execution_candidates,
                        max_trades=min(len(execution_candidates), 3),
                    )
                    filled = [e for e in execution_results if e.success]
                    failed = [e for e in execution_results if not e.success]
                    console.print(
                        f"[cyan]Auto-execute:[/cyan] {len(filled)} filled, {len(failed)} failed "
                        f"(candidates={len(execution_candidates)})"
                    )
                    for f in filled:
                        console.print(
                            f"[green]  ✓ Trade {f.trade_id or 'N/A'} @ {f.fill_price:.5f} "
                            f"({f.lots:.2f} lots)[/green]"
                        )
                    for f in failed[:3]:
                        console.print(f"[yellow]  ✗ {f.error or 'execution failed'}[/yellow]")
                else:
                    console.print("[dim]Auto-execute: no executable agent-confirmed setups[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠ Auto-execute failed: {e}[/yellow]")

        # Return results as (PairAnalysis, gates_passed) tuples
        return [(r, r.gates_passed) for r in results]

    except ImportError as e:
        # Fallback to legacy implementation if buddy_scanner not available
        console.print(f"[yellow]⚠ BuddyScanner not available ({e}), using legacy scan[/yellow]")
        return _buddy_scan_legacy(
            config_path=config_path,
            pairs=pairs,
            granularity=granularity,
            top_n=top_n,
            verbose=verbose,
            prompt_train=prompt_train,
            **kwargs,
        )


def buddy_predict_78(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str = "EUR_USD",
    granularity: str = "H1",
    execute: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Run prediction using the verified 78% model with aggressive position sizing.

    This is the recommended predict command for the $100k → $1M compounding strategy.
    Uses BuddyScanner for consistent position sizing with scan results.

    Args:
        config_path: Path to config file
        instrument: Currency pair (e.g., "EUR_USD", "USD_JPY")
        granularity: Timeframe (default H1)
        execute: If True, execute the trade on OANDA practice
        verbose: Show extra details

    Returns:
        Dict with prediction details or None on failure
    """

    # Normalize instrument
    instrument = instrument.upper().replace("/", "_")

    console.print("\n" + "=" * 60)
    console.print("[bold cyan]📊 BUDDY PREDICT (78% Model)[/bold cyan]")
    console.print("=" * 60)

    # Use BuddyScanner for consistent results with scan
    try:
        from buddy_scanner import BuddyScanner

        scanner = BuddyScanner(config_path=config_path)

        # Run scan for just this one pair
        results = scanner.scan(
            pairs=[instrument],
            granularity=granularity,
            top_n=1,
            verbose=False,
            prompt_train=False,
        )

        if not results:
            console.print(f"[red]✗ No prediction for {instrument}[/red]")
            return None

        r = results[0]

        # Get live account equity from scanner (uses compounding!)
        account_equity = scanner._scan_config.account_equity

    except Exception as e:
        console.print(f"[red]✗ Scanner error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None

    # Display prediction (matches scan format)
    console.print(f"[green]✓ Pair Model[/green] ({instrument})")
    console.print(f"[dim]Data: {granularity} timeframe | Compounding Equity: ${account_equity:,.2f}[/dim]")
    console.print("-" * 60)

    signal_color = "green" if r.direction == "LONG" else "red"
    signal_emoji = "🟢" if r.direction == "LONG" else "🔴"

    console.print(f"\n{signal_emoji} [bold {signal_color}]{r.direction}[/bold {signal_color}] {instrument.replace('_', '/')}")
    console.print(f"   Confidence: [{signal_color}]{r.confidence:.1%}[/{signal_color}]")
    console.print(f"   Entry:      {r.current_price:.5f}")

    # Calculate actual SL/TP prices
    from src.utils.fx_paper import pip_size
    pip = pip_size(instrument)

    if r.direction == "LONG":
        sl_price = r.current_price - (r.sl_pips * pip)
        tp_price = r.current_price + (r.tp_pips * pip)
    else:
        sl_price = r.current_price + (r.sl_pips * pip)
        tp_price = r.current_price - (r.tp_pips * pip)

    console.print(f"   SL:         {sl_price:.5f} ({r.sl_pips:.0f} pips)")
    console.print(f"   TP:         {tp_price:.5f} ({r.tp_pips:.0f} pips)")
    console.print(f"   R:R:        1:{r.tp_pips/r.sl_pips:.1f}" if r.sl_pips and r.sl_pips > 0 else "   R:R:        N/A")
    console.print()

    # Position sizing
    units = int(r.recommended_lots * 100000)
    pip_value = 10.0 if not instrument.endswith("JPY") else 7.5
    est_profit = r.recommended_lots * r.tp_pips * pip_value
    est_loss = r.recommended_lots * r.sl_pips * pip_value

    console.print(f"   [bold]Position: {r.recommended_lots:.2f} lots[/bold] ({units:,} units)")
    console.print(f"   Risk:       ${est_loss:,.0f} ({r.risk_pct:.0%} of ${account_equity:,.0f})")
    console.print(f"   Target:     [green]+${est_profit:,.0f}[/green]")
    console.print("-" * 60)

    # Gate check
    if r.gates_passed:
        console.print(f"[green]✓ GATES PASSED[/green] (conf {r.confidence:.0%})")
    else:
        console.print(f"[yellow]⚠ GATES NOT PASSED[/yellow] (conf {r.confidence:.0%})")

    # Execute trade if requested
    if execute and r.gates_passed and r.recommended_lots > 0:
        console.print("\n[bold]Executing trade...[/bold]")
        try:
            from oanda_practice import OandaPracticeClient
            client = OandaPracticeClient.from_env()

            result = client.create_market_order(
                instrument=instrument,
                units=units if r.direction == "LONG" else -units,
                take_profit_price=round(tp_price, 5),
                stop_loss_price=round(sl_price, 5),
            )

            if result and "orderFillTransaction" in result:
                fill = result["orderFillTransaction"]
                fill_price = float(fill.get("price", r.current_price))
                trade_id = fill.get("tradeOpened", {}).get("tradeID", "N/A")
                console.print(f"[green]✓ FILLED[/green] @ {fill_price:.5f} (Trade #{trade_id})")

                # Log to journal
                try:
                    from memory_client import MLEngineMemory
                    mem = MLEngineMemory()
                    mem.log_trade({
                        "timestamp": datetime.now().isoformat(),
                        "instrument": instrument,
                        "direction": r.direction,
                        "confidence": r.confidence,
                        "lots": r.recommended_lots,
                        "entry": fill_price,
                        "sl": sl_price,
                        "tp": tp_price,
                        "trade_id": trade_id,
                        "model": "pair_specific",
                    })
                    console.print("[dim]📓 Trade logged[/dim]")
                except Exception:
                    pass
            else:
                console.print(f"[yellow]Order result: {result}[/yellow]")

        except Exception as e:
            console.print(f"[red]✗ Execution failed: {e}[/red]")
    elif execute and not r.gates_passed:
        console.print("[yellow]⚠ Trade not executed - gates not passed[/yellow]")
    elif execute and r.recommended_lots <= 0:
        console.print("[yellow]⚠ Trade not executed - no position size calculated[/yellow]")

    console.print("=" * 60 + "\n")

    return {
        "instrument": instrument,
        "direction": r.direction,
        "confidence": r.confidence,
        "lots": r.recommended_lots,
        "entry": r.current_price,
        "sl": sl_price,
        "tp": tp_price,
        "sl_pips": r.sl_pips,
        "tp_pips": r.tp_pips,
        "gates_passed": r.gates_passed,
        "score": r.overall_score,
    }


def _buddy_scan_legacy(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    pairs: Optional[str] = None,
    granularity: str = "H1",
    top_n: int = 5,
    verbose: bool = False,
    prompt_train: bool = True,
    **kwargs: Any,
) -> list:
    """Legacy buddy_scan implementation (fallback)."""
    from datetime import datetime

    # Import train_buddy here to avoid circular imports
    from cli.training import train_buddy

    console.print("\n" + "=" * 70)
    console.print("[bold cyan]📡 MULTI-PAIR SCANNER[/bold cyan]")
    console.print("=" * 70)

    # Suppress config loading message
    import logging as _logging
    _utils_logger = _logging.getLogger('utils')
    _utils_prev = _utils_logger.level
    _utils_logger.setLevel(_logging.WARNING)

    cfg = load_config(config_path)

    _utils_logger.setLevel(_utils_prev)

    # Parse pairs
    from pair_scanner import MAJOR_PAIRS, PairAnalysis

    if pairs:
        pair_list = [p.strip().upper().replace("/", "_") for p in pairs.split(",")]
    else:
        pair_list = MAJOR_PAIRS  # Default to majors

    console.print(f"[dim]Scanning {len(pair_list)} pairs: {', '.join(pair_list)}[/dim]")
    console.print(f"[dim]Timeframe: {granularity} | Top {top_n} results[/dim]")
    console.print("-" * 70)

    # Suppress verbose logging during model loading
    import logging as _logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO/WARNING

    # Suppress version warnings for XGBoost and sklearn (comprehensive)
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
    warnings.filterwarnings('ignore', message='.*serialized model.*')
    warnings.filterwarnings('ignore', message='.*older version of XGBoost.*')
    warnings.filterwarnings('ignore', message='.*Booster.save_model.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Suppress all model-related loggers
    _loggers_to_quiet = ['modular_trainers', 'modular_inference', 'feature_engineering',
                         'modular_data_loaders', 'tensorflow', 'absl', 'xgboost']
    _prev_levels = {}
    for name in _loggers_to_quiet:
        logger = _logging.getLogger(name)
        _prev_levels[name] = logger.level
        logger.setLevel(_logging.ERROR)  # Only show errors

    # Initialize scanner
    from src.utils.oanda_practice import OandaPracticeClient
    from src.data.feature_engineering import FeatureEngineering
    from src.utils.fx_paper import candles_to_ohlcv_df

    client = OandaPracticeClient.from_env()
    fe = FeatureEngineering(cfg.get("feature_engineering", {}))

    # Try to load modular ensemble (legacy fallback)
    modular_ensemble = None
    modular_ensemble_meta_path = Path("trained_data") / "models" / "modular_ensemble.meta.json"

    if modular_ensemble_meta_path.exists():
        try:
            from src.core.modular_inference import ModularEnsembleInference
            modular_ensemble = ModularEnsembleInference()
            modular_ensemble.load_models()

            # Load meta for model info
            meta = json.loads(modular_ensemble_meta_path.read_text())

            # Get val_accuracy - handle multiple possible locations
            val_acc = 0
            if "results" in meta:
                val_acc = meta.get("results", {}).get("transformer", {}).get("val_accuracy", 0)
            if val_acc == 0 and "models" in meta:
                val_acc = meta.get("models", {}).get("direction", {}).get("metrics", {}).get("val_accuracy", 0)

            # Get training info - check multiple possible locations
            trained_pairs = meta.get("selected_pairs", [])
            if not trained_pairs:
                trained_pairs = meta.get("trained_pairs", [])
            if not trained_pairs:
                trained_at = meta.get("trained_at", "unknown")
                trained_pairs = [f"EUR_USD (trained {trained_at[:10]})"] if trained_at != "unknown" else ["EUR_USD"]

            console.print(f"[bold green]✓ Legacy ensemble loaded[/bold green] (val_accuracy: {val_acc:.1%})")
            console.print(f"[dim]  Trained on: {', '.join(trained_pairs) if trained_pairs else 'EUR_USD'}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not load modular ensemble: {e}[/yellow]")
            console.print("[dim]Falling back to technical indicators[/dim]")
            modular_ensemble = None
    else:
        console.print("[dim]No modular ensemble found, using technical indicators[/dim]")

    console.print("-" * 70)

    # Scan each pair
    analyses = []

    for pair in pair_list:
        try:
            console.print(f"[dim]Scanning {pair}...[/dim]", end=" ")

            # Fetch data
            fetch_count = 400
            resp = client.get_candles(pair, granularity=granularity, count=fetch_count, price="MBA")
            df_raw = candles_to_ohlcv_df(resp)

            if len(df_raw) < 100:
                console.print("[yellow]insufficient data[/yellow]")
                continue

            # Feature engineering
            df = fe.create_features(df_raw.copy(), include_all=True)

            # Clean up data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0.0)

            # Calculate metrics from technical indicators
            vol_pct = 0.5
            trend_str = 0.5
            entry_score = 0.5
            current_price = df["close"].iloc[-1]
            atr = df["atr"].iloc[-1] if "atr" in df.columns else 0.0

            # Volatility percentile
            if "atr" in df.columns:
                atr_series = df["atr"].dropna()
                if len(atr_series) > 20:
                    vol_pct = float((atr_series < atr_series.iloc[-1]).mean())

            # Trend strength from ADX
            if "adx" in df.columns:
                adx_val = df["adx"].iloc[-1]
                trend_str = min(float(adx_val) / 60, 1.0)

            # === USE MODULAR ENSEMBLE IF AVAILABLE ===
            direction = "LONG"
            confidence = 0.5
            model_used = False
            gates_passed = False
            ridge_conf = 0.0
            tcn_conf = 0.0

            if modular_ensemble is not None:
                try:
                    signal = modular_ensemble.predict(df)

                    # Use TCN direction even if gates don't pass (for scanning)
                    if signal.tcn_direction is not None:
                        direction = "LONG" if signal.tcn_direction == 1 else "SHORT"
                        # Calculate TCN confidence (distance from 0.5 = uncertainty)
                        # 0.5 -> 0%, 0.6 -> 20%, 0.7 -> 40%, 0.8 -> 60%, 0.9 -> 80%, 1.0 -> 100%
                        tcn_conf = abs(signal.tcn_probability - 0.5) * 200  # 0-100 scale to match ridge
                        # Use the higher of TCN or Ridge confidence
                        ridge_conf = signal.ridge_confidence
                        confidence = max(tcn_conf, ridge_conf) / 100  # Convert to 0-1 for internal use
                        model_used = True
                        gates_passed = signal.trade
                    elif signal.direction is not None:
                        direction = signal.direction.upper()
                        confidence = signal.confidence if signal.confidence > 0 else 0.5
                        model_used = True
                        gates_passed = signal.trade
                        ridge_conf = signal.ridge_confidence
                        tcn_conf = confidence * 100

                except Exception as e:
                    if verbose:
                        console.print(f"[yellow]Model prediction failed: {e}[/yellow]")
                    model_used = False

            # === FALLBACK TO TECHNICAL INDICATORS ===
            if not model_used:
                # Use RSI for direction
                if "rsi" in df.columns:
                    rsi = df["rsi"].iloc[-1]
                    if rsi < 30:
                        direction = "LONG"
                        confidence = 0.5 + (30 - rsi) / 60
                    elif rsi > 70:
                        direction = "SHORT"
                        confidence = 0.5 + (rsi - 70) / 60
                    else:
                        if "macd" in df.columns:
                            macd = df["macd"].iloc[-1]
                            direction = "LONG" if macd > 0 else "SHORT"
                            confidence = 0.5 + min(abs(macd) * 10, 0.2)

            # Entry score based on MA alignment and RSI
            if "rsi" in df.columns:
                rsi = df["rsi"].iloc[-1]
                if 30 < rsi < 70:
                    entry_score += 0.15

            if "sma_20" in df.columns and "sma_50" in df.columns:
                sma_20 = df["sma_20"].iloc[-1]
                sma_50 = df["sma_50"].iloc[-1]
                close = df["close"].iloc[-1]

                # Aligned trend
                if (close > sma_20 > sma_50) or (close < sma_20 < sma_50):
                    entry_score += 0.15
                    if direction == "LONG" and close > sma_20 or direction == "SHORT" and close < sma_20:
                        confidence += 0.05

            # Cap confidence
            confidence = min(confidence, 0.95)

            analysis = PairAnalysis(
                pair=pair,
                direction=direction,
                confidence=confidence,
                volatility_percentile=vol_pct,
                trend_strength=trend_str,
                optimal_entry_score=entry_score,
                current_price=current_price,
                atr=atr,
                timestamp=datetime.now(),
            )
            analyses.append((analysis, gates_passed))  # Include gate status

            # Show result with model info
            signal_marker = "🤖" if model_used else "📊"
            gate_marker = "✅" if gates_passed else "⚠️" if model_used else ""
            # Display the actual confidence used (max of tcn_conf and ridge_conf)
            effective_conf = max(tcn_conf, ridge_conf) if model_used else confidence * 100
            conf_display = f"{effective_conf:.0f}%"
            console.print(f"{signal_marker} [{('green' if direction == 'LONG' else 'red')}]{direction}[/] (conf: {conf_display}) {gate_marker}")

        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()
            continue

    # Sort and display results
    analyses.sort(key=lambda x: x[0].overall_score, reverse=True)

    console.print("\n" + "=" * 70)
    model_label = "Ensemble" if modular_ensemble else "Technical"
    console.print(f"[bold green]📊 SCAN RESULTS ({model_label}) - Top {min(top_n, len(analyses))}[/bold green]")
    console.print("=" * 70)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Pair", style="bold")
    table.add_column("Signal", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Gates", justify="center")
    table.add_column("Trend", justify="right")
    table.add_column("Volatility", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Price", justify="right")

    for i, (a, gates_ok) in enumerate(analyses[:top_n], 1):
        signal_color = "green" if a.direction == "LONG" else "red"
        gate_display = "[green]✓[/green]" if gates_ok else "[yellow]⚠[/yellow]"
        table.add_row(
            str(i),
            a.pair.replace("_", "/"),
            f"[{signal_color}]{a.direction}[/{signal_color}]",
            f"{a.confidence:.0%}",
            gate_display,
            f"{a.trend_strength:.2f}",
            f"{a.volatility_percentile:.0%}",
            f"{a.overall_score:.2f}",
            f"{a.current_price:.5f}" if a.current_price else "-",
        )

    console.print(table)

    # Best opportunities
    if analyses:
        analyses_only = [a for a, _ in analyses]
        best_long = max([a for a in analyses_only if a.direction == "LONG"], key=lambda x: x.overall_score, default=None)
        best_short = max([a for a in analyses_only if a.direction == "SHORT"], key=lambda x: x.overall_score, default=None)

        # Count tradeable (gates passed)
        tradeable_count = sum(1 for _, gates_ok in analyses if gates_ok)

        console.print(f"\n[bold]🎯 TOP OPPORTUNITIES:[/bold] ({tradeable_count}/{len(analyses)} tradeable)")
        if best_long:
            console.print(f"  [green]Best LONG:[/green]  {best_long.pair.replace('_', '/')} (conf: {best_long.confidence:.0%}, score: {best_long.overall_score:.2f})")
        if best_short:
            console.print(f"  [red]Best SHORT:[/red] {best_short.pair.replace('_', '/')} (conf: {best_short.confidence:.0%}, score: {best_short.overall_score:.2f})")

    console.print("\n" + "=" * 70)

    # Restore logging levels
    for name, level in _prev_levels.items():
        _logging.getLogger(name).setLevel(level)

    # === SCAN → TRAIN INTERACTIVE PROMPT ===
    if prompt_train and analyses:
        # Find best tradeable pair (gates passed)
        tradeable = [(a, g) for a, g in analyses if g]
        if tradeable:
            best_pair = tradeable[0][0].pair
        else:
            # Fall back to best overall if none tradeable
            best_pair = analyses[0][0].pair

        # Check for existing model (warm-start available)
        model_path = Path("trained_data/models/transformer_direction.keras")
        warm_start_available = model_path.exists()
        warm_label = " (warm-start)" if warm_start_available else ""

        console.print(f"\n[bold yellow]🎯 Train {best_pair.replace('_', '/')}?{warm_label}[/bold yellow]")
        train_answer = _buddy_wizard_ask("  [y/n]: ")

        if _parse_bool_answer(train_answer, default=False):
            console.print(f"\n[bold green]Starting training for {best_pair}...[/bold green]")

            # Import memory client for tracking (if available)
            try:
                from memory_client import MLEngineMemory
                memory = MLEngineMemory()
                if memory.should_skip_pair(best_pair, max_age_hours=24):
                    console.print(f"[yellow]⚠ {best_pair} was trained < 24h ago. Training anyway...[/yellow]")
            except ImportError:
                pass  # Memory client not available

            # Call train_buddy with full ensemble
            train_opts = BuddyTrainingOptions(
                oanda_fetch=OandaFetchOptions(
                    instrument=best_pair,
                    granularity=granularity,
                    candles=12000,  # Full dataset for quality training
                ),
                epochs=200,
                batch_size=128,
                enterprise=True,  # Enable MLflow + CV
            )
            train_buddy(config_path, options=train_opts)

            # Log training to memory (if available)
            try:
                from memory_client import MLEngineMemory
                memory = MLEngineMemory()
                memory.log_training(best_pair, {"granularity": granularity})
            except ImportError:
                pass

    return analyses
