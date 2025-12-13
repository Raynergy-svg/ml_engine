#!/usr/bin/env python3
"""
Enhanced ML Engine Trading Bot CLI

A command-line interface for the ML trading engine that handles:
- Command processing and orchestration
- Interactive dashboard visualization
- Configuration management
- Training/evaluation progress display
- Real-time monitoring
"""

from __future__ import annotations

import sys
import time
import logging
import argparse
from typing import Dict, Any, TYPE_CHECKING

from pathlib import Path

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _TorchStub:  # minimal surface for type hints
        _IS_STUB = True

        class Tensor:  # noqa: D401
            """Stub type for torch.Tensor when torch isn't installed."""

        float32 = "float32"

    torch = _TorchStub()  # type: ignore
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from utils import setup_logging, load_config

if TYPE_CHECKING:
    from neural_network_integrator_enhanced import NeuralNetworkIntegrator

# Constants
DEFAULT_MESSAGE_FORMAT = "Epoch {epoch} completed"
TIMESTAMP_FORMAT = "%H:%M:%S"
TABLE_HEADER_STYLE = "bold magenta"

# Initialize console and logging
console = Console()
logger = setup_logging(log_file="cli.log")


def _configure_predict_output(verbose: bool) -> None:
    """Reduce log/warning noise for interactive predict runs."""
    if verbose:
        return

    import warnings

    # Silence torch GradScaler deprecation warning during prediction.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*GradScaler\(args\.\.\.\) is deprecated\..*",
    )

    # Keep CLI output clean by muting INFO logs from internal modules.
    logging.getLogger().setLevel(logging.WARNING)
    for name in (
        "utils",
        "neural_network_integrator_enhanced",
        "ml_engine_enhanced",
        "memory_manager_enhanced",
        "reasoning_enhanced",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _fx_confirm(prompt: str) -> bool:
    ans = (console.input(prompt) or "").strip().lower()
    return ans == "y"


def _fx_open_position_instruments(payload: Any) -> list[str]:
    pos = (payload or {}).get("positions") or []
    return [str(p.get("instrument")) for p in pos if p.get("instrument")]


def _fx_enforce_fx_policy(cfg: Dict[str, Any], *, instrument: str, granularity: str):
    from fx_guardrails import load_fx_policy

    policy = load_fx_policy(cfg)
    if instrument not in set(policy.instruments):
        console.print(f"[bold red]Blocked[/bold red]: instrument not allowed ({instrument}).")
        console.print(f"Allowed: {', '.join(policy.instruments)}")
        return None
    if granularity != policy.granularity:
        console.print(f"[bold red]Blocked[/bold red]: granularity must be {policy.granularity} (got {granularity}).")
        return None
    return policy


def _fx_refresh_fx_state(cfg: Dict[str, Any], policy: Any, state: Any, client: Any) -> tuple[dict[str, Any], str]:
    import fx_guardrails as fxg

    pnl = fxg.update_state_from_account_summary(policy, state, client.get_account_summary())

    # Only evaluate the *loss* stop here. Profit stop depends on confidence band
    # which is computed later from the current setup/model output.
    stop_hit, stop_reason, stop_kind = fxg.check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=None,
        confidence_band="medium",
    )
    if stop_hit and stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        fxg.save_state(cfg, policy, state)
        console.print(f"[bold red]Trading disabled[/bold red]: {stop_reason}")

    # Band is computed later; keep placeholder here for display.
    return pnl, "unknown"


def _fx_maybe_force_flat(policy: Any, state: Any, client: Any, *, execute: bool) -> bool:
    from fx_guardrails import should_force_flat

    # Only force-flat automatically for the time cutoff or loss-stop.
    force_flat_due_to_stop = bool(state.disabled_reason) and (getattr(state, "disabled_kind", None) == "loss")
    if not should_force_flat(policy) and not force_flat_due_to_stop:
        return False

    insts = _fx_open_position_instruments(client.get_open_positions())
    if not insts:
        console.print("[dim]No open positions to close.[/dim]")
        return True

    console.print("[yellow]Force-flat: open positions detected.[/yellow]")
    if not execute:
        console.print("[dim]Dry-run: would close all open positions.[/dim]")
        return True

    if policy.require_confirmation and not _fx_confirm("Confirm CLOSE ALL positions? (y/N): "):
        console.print("[dim]Close cancelled.[/dim]")
        return True

    for inst in insts:
        try:
            client.close_position(instrument=inst)
            console.print(f"Closed position: {inst}")
        except Exception as e:
            console.print(f"[bold red]Failed[/bold red] closing {inst}: {e}")

    return True


def _fx_gate_fx_entry(policy: Any, state: Any, client: Any) -> bool:
    from fx_guardrails import can_open_new_trade

    ok, why = can_open_new_trade(policy, state)
    if not ok:
        console.print(f"[bold red]Blocked[/bold red]: {why}")
        return False

    insts = _fx_open_position_instruments(client.get_open_positions())
    if len(insts) >= policy.limits.max_open_positions:
        console.print("[bold red]Blocked[/bold red]: max open positions reached.")
        return False

    return True


def _fx_load_fx_df(client: Any, *, instrument: str, granularity: str, candles: int):
    from fx_paper import candles_to_ohlcv_df

    resp = client.get_candles(instrument, granularity=granularity, count=candles, price="MBA")
    return candles_to_ohlcv_df(resp)


def _fx_spread_and_slippage(policy: Any, df: Any, *, instrument: str) -> tuple[bool, float, float, float]:
    from fx_paper import conservative_slippage_pips, pip_size, spread_pips_from_df

    spread_pips = spread_pips_from_df(df, instrument)
    if spread_pips is None:
        spread_pips = float(policy.costs.spread_fallback_pips.get(instrument, 0.0) or 0.0)

    max_spread = float(policy.costs.max_spread_pips.get(instrument, 0.0) or 0.0)
    if max_spread > 0 and spread_pips > max_spread:
        console.print(
            f"[bold red]Blocked[/bold red]: spread too wide ({spread_pips:.2f} pips > {max_spread:.2f} pips)."
        )
        return False, spread_pips, 0.0, 0.0

    slippage_pips = conservative_slippage_pips(
        spread_pips=spread_pips,
        min_pips=policy.costs.slippage_pips_min,
        spread_mult=policy.costs.slippage_pips_spread_mult,
    )
    slippage_price = slippage_pips * pip_size(instrument)
    return True, float(spread_pips), float(slippage_pips), float(slippage_price)


def _fx_require_account_metrics(pnl: Dict[str, Any]) -> bool:
    """Fail-closed if we can't read basic account metrics."""
    nav = pnl.get("nav")
    balance = pnl.get("balance")
    if nav is None or balance is None:
        console.print("[bold red]Blocked[/bold red]: missing account NAV/balance from broker.")
        console.print(f"[dim]pnl payload keys: {sorted((pnl or {}).keys())}[/dim]")
        return False
    return True


def _fx_get_signal_context(df: Any) -> tuple[str, float, float]:
    from fx_paper import atr as fx_atr
    from fx_paper import setup_signal

    signal = setup_signal(df)
    last_close = float(df["close"].iloc[-1])
    atr_value = float(fx_atr(df, period=14))
    return str(signal), float(last_close), float(atr_value)


def _fx_build_risk_rules(
    policy: Any,
    pnl: Dict[str, Any],
    *,
    equity: float,
    risk_per_trade_pct: float,
):
    from fx_paper import RiskRules

    # Prefer live NAV when available.
    nav = pnl.get("nav")
    base_equity = float(nav) if nav is not None else float(equity)

    # Use the explicit CLI flag if provided, else fallback to policy.
    rpt = float(risk_per_trade_pct) if risk_per_trade_pct is not None else float(policy.risk.risk_per_trade_pct)

    return RiskRules(
        equity=base_equity,
        risk_per_trade_pct=rpt,
        max_daily_loss_pct=float(getattr(policy.risk, "daily_loss_stop_pct", 0.02)),
        max_open_positions=int(getattr(policy.limits, "max_open_positions", 1)),
        atr_stop_mult=float(getattr(policy.risk, "atr_stop_mult", 1.5)),
        rr_take_profit=float(getattr(policy.risk, "rr_take_profit", 1.5)),
    )


def _fx_compute_confidence_and_band(
    policy: Any,
    *,
    instrument: str,
    signal: str,
    _price: float,
    atr_value: float,
    spread_pips: float,
) -> tuple[float, str, list[str]]:
    """Heuristic confidence score for guardrails (not ML-based)."""
    import fx_guardrails as fxg

    reasons: list[str] = []
    confidence = 0.70

    if signal not in {"buy", "sell"}:
        confidence = 0.0
        reasons.append("no actionable signal")
        return confidence, "low", reasons

    max_spread = float(getattr(policy.costs, "max_spread_pips", {}).get(instrument, 0.0) or 0.0)
    if max_spread > 0:
        ratio = float(spread_pips) / max_spread
        if ratio >= 0.9:
            confidence -= 0.15
            reasons.append("spread near max")
        elif ratio >= 0.75:
            confidence -= 0.08
            reasons.append("spread elevated")

    import math

    if float(atr_value) <= 0 or not math.isfinite(float(atr_value)):
        confidence -= 0.20
        reasons.append("invalid ATR")

    confidence = float(max(0.0, min(1.0, confidence)))
    band = fxg.confidence_band(policy, confidence)
    if not reasons:
        reasons.append("baseline")
    return confidence, str(band), reasons


def _fx_apply_daily_stops(cfg: Dict[str, Any], policy: Any, state: Any, pnl: Dict[str, Any], *, band: str) -> bool:
    import fx_guardrails as fxg

    stop_hit, stop_reason, stop_kind = fxg.check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=pnl.get("realized_pct"),
        confidence_band=str(band),
    )
    if not stop_hit:
        return False

    if stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        fxg.save_state(cfg, policy, state)
        console.print(f"[bold red]Trading disabled[/bold red]: {stop_reason}")
    return True


def _fx_build_order_units_and_prices(
    policy: Any,
    rules: Any,
    *,
    instrument: str,
    signal: str,
    price: float,
    atr_value: float,
    slippage_price: float,
) -> tuple[int, float, float, float, float]:
    from fx_paper import position_size_units

    stop_distance = float(atr_value) * float(getattr(policy.risk, "atr_stop_mult", 1.5))
    stop_distance = float(max(stop_distance, 0.0)) + float(max(slippage_price, 0.0))
    if stop_distance <= 0:
        raise ValueError("Computed stop distance is not positive")

    tp_distance = stop_distance * float(getattr(policy.risk, "rr_take_profit", 1.5))

    if signal == "buy":
        stop_price = float(price) - stop_distance
        tp_price = float(price) + tp_distance
        side = 1
    else:
        stop_price = float(price) + stop_distance
        tp_price = float(price) - tp_distance
        side = -1

    units_abs = position_size_units(
        instrument=instrument,
        equity=float(getattr(rules, "equity", 0.0)),
        risk_per_trade_pct=float(getattr(rules, "risk_per_trade_pct", 0.005)),
        stop_distance_price=stop_distance,
        price=float(price),
        pip_value_per_unit=None,
    )
    return int(side * units_abs), float(stop_price), float(tp_price), float(stop_distance), float(tp_distance)


def _fx_execution_guard_price_bound(_policy: Any, client: Any, *, instrument: str, units: int) -> float | None:
    from fx_paper import pip_size

    try:
        q = client.get_price_quote(instrument=instrument)
    except Exception as e:
        console.print(f"[bold red]Blocked[/bold red]: could not fetch live quote: {e}")
        return None

    bid = float(q["bid"])
    ask = float(q["ask"])
    mid = (bid + ask) / 2.0
    half_spread = max(0.0, ask - mid)

    # Add a small, conservative buffer on top of half-spread.
    buffer_price = 2.0 * pip_size(instrument)

    if int(units) > 0:
        return float(ask + half_spread + buffer_price)
    return float(bid - half_spread - buffer_price)


def _build_integrated_engines(config: Dict[str, Any]) -> NeuralNetworkIntegrator:
    """Create and wire the unified neural engine into the integrator."""
    from neural_network_integrator_enhanced import NeuralNetworkIntegrator
    from neural_engine_unified import UnifiedNeuralEngine
    from reasoning_enhanced import ReasoningEngine

    integrator_config = {
        "device": config.get("device", "cpu"),
        "use_attention": False,
        "use_dynamic_weights": True,
    }
    integrator = NeuralNetworkIntegrator(integrator_config)
    reasoning_engine = ReasoningEngine(config.get("reasoning", {}))

    base_input = int(config.get("model", {}).get("input_size", 7))
    unified_input = base_input + 11 + 11
    unified_engine = UnifiedNeuralEngine(
        {
            "device": integrator.device,
            "model": {
                "input_size": unified_input,
                "hidden_size": int(config.get("model", {}).get("hidden_size", 64)),
                "num_layers": int(config.get("model", {}).get("num_layers", 2)),
                "dropout": float(config.get("model", {}).get("dropout", 0.1)),
                "bidirectional": bool(config.get("model", {}).get("bidirectional", False)),
            },
        }
    )

    integrator.set_unified_engine(unified_engine=unified_engine, reasoning_engine=reasoning_engine)
    return integrator


def integrated_predict_once(config: Dict[str, Any]) -> Dict[str, Any]:
    """Smoke: run one integrated prediction across ML/MT/MR."""
    integrator = _build_integrated_engines(config)

    ml_features = torch.zeros((2, 5, int(config.get("model", {}).get("input_size", 7))), dtype=torch.float32)
    mt_features = torch.zeros((2, 5, 11), dtype=torch.float32)
    mr_features = torch.zeros((2, 5, 11), dtype=torch.float32)

    return integrator.predict(
        {"ml_features": ml_features, "mt_features": mt_features, "mr_features": mr_features}
    )


def compute_slm_indicators(market_data_df):
    """Compute a small subset of SLM technical indicators without network calls."""
    try:
        from SLM.stock_slm import (
            calculate_sma,
            calculate_ema,
            calculate_rsi,
            calculate_macd,
            calculate_bollinger_bands,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Failed to import SLM indicators: {e}") from e

    df_for_slm = market_data_df
    # SLM indicator functions expect title-case columns like 'Close'.
    if hasattr(market_data_df, "columns") and "Close" not in market_data_df.columns:
        rename_map = {}
        for col in market_data_df.columns:
            key = str(col).strip().lower()
            if key == "close":
                rename_map[col] = "Close"
            elif key == "open":
                rename_map[col] = "Open"
            elif key == "high":
                rename_map[col] = "High"
            elif key == "low":
                rename_map[col] = "Low"
            elif key == "volume":
                rename_map[col] = "Volume"
        if rename_map:
            df_for_slm = market_data_df.rename(columns=rename_map)

    sma_20 = calculate_sma(df_for_slm, 20)
    ema_20 = calculate_ema(df_for_slm, 20)
    rsi_14 = calculate_rsi(df_for_slm, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(df_for_slm)
    bb_upper, bb_lower = calculate_bollinger_bands(df_for_slm, 20, 2)

    return {
        "sma_20": sma_20,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }


def _pick_default_market_csv(config: Dict[str, Any]) -> str:
    data_dir = (
        config.get("data", {}).get("data_dir")
        or config.get("paths", {}).get("data_dir")
        or config.get("DATA_DIR")
        or config.get("data_dir")
        or "market_data"
    )
    candidates = sorted(Path(data_dir).glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No .csv files found under: {data_dir}")
    return str(candidates[0])


def _normalize_market_dataframe(df):
    """Return a copy with standardized OHLCV column names: open/high/low/close/volume."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("market_data_df must be a pandas DataFrame")

    out = df.copy()
    rename_map = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in {"open", "o"}:
            rename_map[col] = "open"
        elif key in {"high", "h"}:
            rename_map[col] = "high"
        elif key in {"low", "l"}:
            rename_map[col] = "low"
        elif key in {"close", "c", "adj close", "adj_close", "adjclose"}:
            rename_map[col] = "close"
        elif key in {"volume", "vol", "v"}:
            rename_map[col] = "volume"

    out = out.rename(columns=rename_map)
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Market DataFrame missing required columns: {missing}")
    return out


def build_integrated_feature_tensors(
    market_data_df, config: Dict[str, Any], *, allow_pad: bool = False
) -> Dict[str, Any]:
    """Build feature tensors for ML/MT/MR using OHLCV + SLM indicators."""
    import pandas as pd
    import numpy as np

    df = _normalize_market_dataframe(market_data_df)

    seq_len = int(
        config.get("data", {}).get("sequence_length")
        or config.get("sequence_length")
        or 60
    )
    if len(df) < seq_len:
        if not allow_pad:
            raise ValueError(f"Need at least {seq_len} rows; got {len(df)}")
        pad_rows = seq_len - len(df)
        first_row = df.iloc[[0]].copy()
        df = pd.concat([pd.concat([first_row] * pad_rows, ignore_index=True), df], ignore_index=True)

    indicators = compute_slm_indicators(df)
    feature_frame = df[["open", "high", "low", "close", "volume"]].copy()
    feature_frame["sma_20"] = indicators["sma_20"]
    feature_frame["ema_20"] = indicators["ema_20"]
    feature_frame["rsi_14"] = indicators["rsi_14"]
    feature_frame["macd_line"] = indicators["macd_line"]
    feature_frame["bb_upper"] = indicators["bb_upper"]
    feature_frame["bb_lower"] = indicators["bb_lower"]

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.ffill().bfill().fillna(0.0)
    window = feature_frame.tail(seq_len)

    ml_input_size = int(
        config.get("model", {}).get("input_size")
        or config.get("input_features")
        or config.get("input_size")
        or 7
    )

    base_cols = ["open", "high", "low", "close", "volume"]
    extra_cols = ["rsi_14", "sma_20", "ema_20", "macd_line", "bb_upper", "bb_lower"]
    ml_cols = (base_cols + extra_cols)[:ml_input_size]
    if len(ml_cols) < ml_input_size:
        for i in range(ml_input_size - len(ml_cols)):
            pad_name = f"ml_pad_{i}"
            window[pad_name] = 0.0
            ml_cols.append(pad_name)

    mt_mr_cols = base_cols + ["sma_20", "ema_20", "rsi_14", "macd_line", "bb_upper", "bb_lower"]

    ml_arr = window[ml_cols].to_numpy(dtype=np.float32, copy=True)
    mt_mr_arr = window[mt_mr_cols].to_numpy(dtype=np.float32, copy=True)

    def _as_tensor(arr):
        # Fall back to NumPy arrays if torch isn't available.
        if getattr(torch, "_IS_STUB", False):
            return arr[None, :, :]
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

    ml_tensor = _as_tensor(ml_arr)
    mt_tensor = _as_tensor(mt_mr_arr)
    mr_tensor = _as_tensor(mt_mr_arr)

    return {"ml_features": ml_tensor, "mt_features": mt_tensor, "mr_features": mr_tensor}


def integrated_predict(config_path: str, csv_path: str | None = None) -> None:
    """Run a unified prediction: CSV -> SLM indicators -> ML/MT/MR -> integrator -> reasoning."""
    import pandas as pd

    config = load_config(config_path)
    if csv_path is None:
        csv_path = _pick_default_market_csv(config)

    df = pd.read_csv(csv_path)
    features = build_integrated_feature_tensors(df, config)
    integrator = _build_integrated_engines(config)
    result = integrator.predict(features)

    pred = result.get("prediction")
    uncertainty = result.get("uncertainty")
    weights = result.get("weights")
    reasoning = result.get("reasoning")

    console.print(f"[bold blue]Integrated predict[/bold blue] CSV={csv_path}")
    console.print(f"[bold yellow]Prediction:[/bold yellow] {pred}")
    console.print(f"[bold yellow]Uncertainty:[/bold yellow] {uncertainty}")
    if weights is not None:
        console.print(f"[cyan]Weights (ML/MT/MR):[/cyan] {weights}")
    if isinstance(reasoning, dict) and reasoning.get("insights"):
        console.print("[bold green]Reasoning insights:[/bold green]")
        for insight in reasoning["insights"]:
            console.print(f"- {insight}")


def _fetch_live_market_data(ticker: str, period: str, interval: str):
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Live data fetch requires `yfinance`.") from e

    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data is None or data.empty:
        raise RuntimeError(f"No live data returned for {ticker} ({period}, {interval})")
    # Keep the index as a column (Date/Datetime) for display/debugging.
    return data.reset_index()


def _df_last_value(df, candidates: tuple[str, ...]):
    for col in candidates:
        if col in df.columns:
            return df[col].iloc[-1]
    return None


def _print_last_bar(df) -> None:
    last_close = _df_last_value(df, ("Close", "close"))
    last_ts = _df_last_value(df, ("Datetime", "Date", "date", "datetime"))
    if last_ts is None and last_close is None:
        return
    console.print(f"[dim]Last bar:[/dim] {last_ts}  close={last_close}")


def _print_reasoning_insights(reasoning) -> None:
    if not isinstance(reasoning, dict):
        return
    insights = reasoning.get("insights")
    if not insights:
        return
    console.print("[bold green]Reasoning insights:[/bold green]")
    for insight in insights:
        console.print(f"- {insight}")


def _integrated_predict_live_once(
    *,
    ticker: str,
    period: str,
    interval: str,
    config: Dict[str, Any],
    integrator: NeuralNetworkIntegrator,
) -> None:
    df = _fetch_live_market_data(ticker, period=period, interval=interval)
    features = build_integrated_feature_tensors(df, config, allow_pad=True)
    result = integrator.predict(features)

    console.print(f"[bold blue]Live predict[/bold blue] {ticker} ({period}, {interval})")
    _print_last_bar(df)
    console.print(f"[bold yellow]Prediction:[/bold yellow] {result.get('prediction')}")
    console.print(f"[bold yellow]Uncertainty:[/bold yellow] {result.get('uncertainty')}")

    weights = result.get("weights")
    if weights is not None:
        console.print(f"[cyan]Weights (ML/MT/MR):[/cyan] {weights}")

    _print_reasoning_insights(result.get("reasoning"))


def integrated_predict_live(
    config_path: str,
    ticker: str,
    period: str = "5d",
    interval: str = "1h",
    watch_seconds: int | None = None,
) -> None:
    """Run the unified prediction using live-ish OHLCV from yfinance."""
    config = load_config(config_path)
    integrator = _build_integrated_engines(config)

    _integrated_predict_live_once(
        ticker=ticker,
        period=period,
        interval=interval,
        config=config,
        integrator=integrator,
    )

    if watch_seconds is None:
        return

    interval_seconds = max(1, int(watch_seconds))
    while True:  # pragma: no cover
        time.sleep(interval_seconds)
        _integrated_predict_live_once(
            ticker=ticker,
            period=period,
            interval=interval,
            config=config,
            integrator=integrator,
        )


def fx_paper_trade(
    config_path: str,
    instrument: str,
    granularity: str = "M5",
    candles: int = 300,
    execute: bool = False,
    equity: float = 10_000.0,
    risk_per_trade_pct: float = 0.005,
) -> None:
    """Paper trade forex on OANDA PRACTICE using a simple setup rule.

    This is intentionally conservative and defaults to DRY-RUN (no order placed).
    """
    cfg = load_config(config_path)

    from oanda_practice import OandaPracticeClient
    import fx_guardrails as fxg

    policy = _fx_enforce_fx_policy(cfg, instrument=instrument, granularity=granularity)
    if policy is None:
        return

    client = OandaPracticeClient.from_env()
    state = fxg.load_state(cfg, policy)

    pnl, _ = _fx_refresh_fx_state(cfg, policy, state, client)
    if _fx_maybe_force_flat(policy, state, client, execute=execute):
        return

    if not _fx_require_account_metrics(pnl):
        return

    if not _fx_gate_fx_entry(policy, state, client):
        return

    df = _fx_load_fx_df(client, instrument=instrument, granularity=granularity, candles=candles)
    ok_spread, spread_pips, slippage_pips, slippage_price = _fx_spread_and_slippage(policy, df, instrument=instrument)
    if not ok_spread:
        return

    signal, last_close, atr_value = _fx_get_signal_context(df)
    if signal == "hold":
        console.print(f"FX setup: HOLD  {instrument} {granularity}  close={last_close:.5f}  ATR14={atr_value:.5f}")
        return

    rules = _fx_build_risk_rules(policy, pnl, equity=equity, risk_per_trade_pct=risk_per_trade_pct)

    confidence, band, conf_reasons = _fx_compute_confidence_and_band(
        policy,
        instrument=instrument,
        signal=signal,
        _price=last_close,
        atr_value=atr_value,
        spread_pips=float(spread_pips),
    )
    if band == "low":
        console.print(f"[bold red]Blocked[/bold red]: low confidence ({confidence:.2f}) => no trades.")
        console.print(f"[dim]Reasons: {', '.join(conf_reasons)}[/dim]")
        return

    if _fx_apply_daily_stops(cfg, policy, state, pnl, band=band):
        return

    units, stop_price, tp_price, _, _ = _fx_build_order_units_and_prices(
        policy,
        rules,
        instrument=instrument,
        signal=signal,
        price=last_close,
        atr_value=atr_value,
        slippage_price=float(slippage_price),
    )

    console.print(
        f"FX setup: {signal.upper()}  {instrument} {granularity}  close={last_close:.5f}  units={units}  SL={stop_price:.5f}  TP={tp_price:.5f}"
        f"  spread={spread_pips:.2f}p  slip={slippage_pips:.2f}p  conf={confidence:.2f}  band={band}"
    )

    if not execute:
        console.print("[dim]Dry-run (no order). Pass --execute to place a PRACTICE order.[/dim]")
        return

    if policy.require_confirmation and not _fx_confirm("Confirm PLACE ORDER? (y/N): "):
        console.print("[dim]Order cancelled.[/dim]")
        return

    price_bound = _fx_execution_guard_price_bound(policy, client, instrument=instrument, units=units)
    if price_bound is None:
        return

    result = client.create_market_order(
        instrument=instrument,
        units=units,
        stop_loss_price=stop_price,
        take_profit_price=tp_price,
        price_bound=price_bound,
    )
    state.entries_today += 1
    fxg.save_state(cfg, policy, state)
    console.print("[bold green]Order submitted (PRACTICE).[/bold green]")
    console.print(result)


def generate_dashboard(
    epoch: int,
    total_epochs: int,
    latency: int,
    train_loss: float,
    val_loss: float,
    lr_delta: float,
    api_logs: list,
    config: Dict[str, Any],
) -> Layout:
    """Generate the rich dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", size=8),
        Layout(name="footer", size=8),
    )
    metrics_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    metrics_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    metrics_table.add_column("Value", style="green")
    metrics_table.add_row("Epoch", f"{epoch}/{total_epochs}")
    metrics_table.add_row("Training Loss", f"{train_loss:.6f}")
    metrics_table.add_row("Validation Loss", f"{val_loss:.6f}")
    metrics_table.add_row("API Latency", f"{latency}ms")
    lr_color = "green" if lr_delta >= 0 else "red"
    metrics_table.add_row("LR Change", f"[{lr_color}]{lr_delta:+.2f}%[/{lr_color}]")
    metrics_table.add_row(
        "Device", f"{config.get('hardware', {}).get('device', 'cpu')}"
    )
    status = (
        "[bold green]Active[/bold green]"
        if epoch < total_epochs
        else "[bold yellow]Complete[/bold yellow]"
    )
    metrics_table.add_row("Status", status)
    layout["header"].update(
        Panel(metrics_table, title="[bold]Training Metrics[/bold]", border_style="blue")
    )
    body_layout = Layout()
    body_layout.split_row(
        Layout(name="progress", ratio=1), Layout(name="logs", ratio=2)
    )
    progress_pct = (epoch / total_epochs * 100) if total_epochs > 0 else 0
    bar = "█" * int(progress_pct / 2) + "░" * (50 - int(progress_pct / 2))
    eta = (total_epochs - epoch) * latency / 1000
    progress_text = f"{progress_pct:.1f}%\n{bar}\nETA: {eta:.1f}s"
    body_layout["progress"].update(
        Panel(progress_text, title="[bold]Progress[/bold]", border_style="green")
    )
    current_time = time.strftime("%H:%M:%S")
    if not api_logs:
        api_logs.append(f"[grey]{current_time}[/grey] Starting training...")
    formatted_logs = [
        log if log.startswith("[") else f"[grey]{current_time}[/grey] {log}"
        for log in api_logs[-5:]
    ]
    logs_content = "\n".join(formatted_logs)
    body_layout["logs"].update(
        Panel(logs_content, title="[bold]Recent Updates[/bold]", border_style="yellow")
    )
    layout["body"].update(body_layout)
    config_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    model_config = config.get("model", {})
    config_table.add_row("Learning Rate", f"{model_config.get('learning_rate', 'N/A')}")
    config_table.add_row("Batch Size", f"{model_config.get('batch_size', 'N/A')}")
    config_table.add_row("Architecture", f"{model_config.get('architecture', 'N/A')}")
    config_table.add_row("Hidden Size", f"{model_config.get('hidden_size', 'N/A')}")
    config_table.add_row("Num Layers", f"{model_config.get('num_layers', 'N/A')}")
    config_table.add_row("Dropout", f"{model_config.get('dropout', 'N/A')}")
    optimizer_config = model_config.get("optimizer", {})
    config_table.add_row("Optimizer", f"{optimizer_config.get('type', 'Adam')}")
    logs_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    logs_table.add_column("Latest Log", style="green")
    latest_log = api_logs[-1] if api_logs else "No logs available"
    logs_table.add_row(latest_log)
    footer_layout = Layout()
    footer_layout.split_row(
        Panel(config_table, title="[bold]Configuration[/bold]", border_style="blue"),
        Panel(logs_table, title="[bold]Log Info[/bold]", border_style="blue"),
    )
    layout["footer"].update(footer_layout)
    return layout


# CLI Command Implementations
def train_model(config_path: str, choose_csv: bool = False) -> None:
    """Run model training with live progress dashboard.
    
    Args:
        config_path: Path to configuration file
        choose_csv: Whether to choose CSV file interactively
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        
        # Validate training configuration
        if "training" not in config:
            raise ValueError("Configuration missing 'training' section")
        if "epochs" not in config["training"]:
            raise ValueError("Configuration missing 'training.epochs'")
        
        console.print("[bold blue]Initializing ML Engine...[/bold blue]")
        
        # Load training data
        from data_loader import MarketDataLoader
        data_loader = MarketDataLoader(config)
        
        console.print("[bold blue]Loading market data...[/bold blue]")
        
        # Get data directory from config
        data_dir = config.get("data", {}).get("data_dir", "market_data/")
        
        # Load multiple tickers from the data directory
        data_dict = data_loader.load_multiple_tickers(data_dir)
        
        if not data_dict:
            raise ValueError(f"No data files found in {data_dir}")
        
        # Combine all ticker data
        df = data_loader.combine_ticker_data(data_dict, method="concat")
        console.print(f"[cyan]Loaded {len(data_dict)} tickers, {len(df)} total rows[/cyan]")
        
        # Preprocess the data
        console.print("[bold blue]Preprocessing data...[/bold blue]")
        x_train, y_train, x_val, y_val, _, _ = data_loader.preprocess(
            df,
            add_features=True,
            scaler_type="standard",
            sequence_length=config.get("data", {}).get("sequence_length", 60),
            test_size=0.2,
        )
        
        console.print(f"[bold green]Data prepared: {len(x_train)} training samples, {len(x_val)} validation samples[/bold green]")
        
        # Update config with correct input size based on actual features
        input_size = x_train.shape[-1]  # Get feature dimension
        if "model" not in config:
            config["model"] = {}
        config["model"]["input_size"] = input_size
        console.print(f"[cyan]Model input size set to: {input_size}[/cyan]")

        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)
        
        # Get epochs from config (not nested in 'training')
        epochs = config.get("epochs", 100)
        console.print(f"[bold green]Training for {epochs} epochs[/bold green]")
        
        # Train the model and get results
        console.print("[bold green]Starting training...[/bold green]")
        result = engine.train(x_train, y_train, x_val, y_val, epochs=epochs)
        
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print(f"[cyan]Total epochs trained: {result.get('total_epochs', epochs)}[/cyan]")
        console.print(f"[cyan]Best validation loss: {result['best_val_loss']:.6f}[/cyan]")
        console.print(f"[cyan]Resumed from checkpoint: {result.get('resumed', False)}[/cyan]")
                
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logging.error(f"Unexpected error in train_model: {e}", exc_info=True)
        raise


def evaluate_model(config_path: str) -> None:
    """Run model evaluation with results dashboard.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        RuntimeError: If evaluation fails
    """
    try:
        console.print("[bold blue]Loading configuration and model...[/bold blue]")
        config = load_config(config_path)
        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)

        api_logs = [f"[blue]{time.strftime('%H:%M:%S')}[/blue] Starting evaluation"]
        
        with Live(generate_dashboard(0, 10, 200, 0.0, 0.0, 0.0, api_logs, config)) as live:
            try:
                console.print("[bold green]Evaluating model...[/bold green]")
                metrics = engine.evaluate_model()
                
                # Display evaluation results
                api_logs.append("[green]Evaluation completed[/green]")
                if metrics:
                    for key, value in metrics.items():
                        api_logs.append(f"[cyan]{key}: {value:.4f}[/cyan]")
                
                live.update(generate_dashboard(10, 10, 0, 0.0, 0.0, 0.0, api_logs, config))
                console.print("[bold green]Evaluation completed successfully![/bold green]")
                
            except Exception as e:
                console.print(f"[red]Evaluation error: {e}[/red]")
                logging.error(f"Evaluation failed: {e}", exc_info=True)
                raise RuntimeError(f"Evaluation failed: {e}")
                
    except Exception as e:
        console.print(f"[red]Failed to evaluate model: {e}[/red]")
        logging.error(f"Model evaluation error: {e}", exc_info=True)
        raise


# New CLI command implementations
def visualize_dashboard(config_path: str) -> None:
    """Display dashboard visualizations using the visualizer module."""
    import visualizer

    config = load_config(config_path)
    console.print("[bold blue]Launching visualizations...[/bold blue]")
    visualizer.display_dashboard(
        config
    )  # Assumes implementation exists in visualizer.py


def openai_tune(config_path: str) -> None:
    """Execute OpenAI-based auto-tuning using the openai_integration module."""
    try:
        import openai_integration
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "openai_integration requires the `openai` package; install it to use openai-tune."
        ) from e

    current_config = load_config(config_path)

    # Only set credentials when we actually use OpenAI integration.
    openai_integration.set_openai_credentials(current_config)

    # Gather current metrics from the engine
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(current_config)
    train_loss = (
        engine.train_losses[-1]
        if hasattr(engine, "train_losses") and engine.train_losses
        else None
    )
    val_loss = (
        engine.val_losses[-1]
        if hasattr(engine, "val_losses") and engine.val_losses
        else None
    )
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_config.get("model", {}).get("learning_rate", 0.001),
        "epochs_completed": len(engine.train_losses)
        if hasattr(engine, "train_losses")
        else 0,
        "device": current_config.get("hardware", {}).get("device", "cpu"),
        "batch_size": current_config.get("model", {}).get("batch_size", 32),
    }

    # Call the integration function with metrics and config in proper order
    result = openai_integration.query_for_auto_configuration(metrics, current_config)

    if result:
        _ = openai_integration.report_config_changes(current_config, result)
        console.print(
            "[bold green]Configuration updated successfully and written to config.yaml[/bold green]"
        )
        console.print(f"[cyan]Updates Applied:[/cyan] {result}")
    else:
        console.print(
            "[bold red]Failed to get configuration updates from OpenAI[/bold red]"
        )


def predict_price(config_path: str) -> None:
    """Use ML engine to predict prices given a dataset.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        console.print("[bold blue]Loading model for prediction...[/bold blue]")
        config = load_config(config_path)
        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)
        
        console.print("[bold green]Generating prediction...[/bold green]")
        prediction = engine.predict_price()
        
        console.print(f"[bold yellow]Predicted Price: {prediction}[/bold yellow]")
        logging.info(f"Price prediction completed: {prediction}")
        
    except AttributeError as e:
        console.print(f"[red]Method not available: {e}[/red]")
        logging.error(f"predict_price method error: {e}")
        raise RuntimeError(f"Prediction method not available: {e}")
    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        logging.error(f"Price prediction error: {e}", exc_info=True)
        raise


def realtime_loop(config_path: str) -> None:
    """
    Run the ML engine in a real-time loop for continuous inference.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.run_realtime_loop()
    console.print("[bold yellow]Realtime Loop command executed[/bold yellow]")


def tune_model(config_path: str) -> None:
    """
    Perform advanced hyperparameter tuning using ML engine methods.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.tune_hyperparameters()
    console.print("[bold yellow]Tune Model command executed[/bold yellow]")


def profile_pipeline(config_path: str) -> None:
    """
    Profile the ML pipeline for bottlenecks.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.profile_pipeline()
    console.print("[bold yellow]Profile Pipeline command executed[/bold yellow]")


def run_ai_assistant(config_path: str) -> None:
    config = load_config(config_path)  # removed Path() wrapper for proper string input
    # Instantiate the AI assistant using the class from ml_engine
    from ai_assistant import ai_assistant

    assistant_instance = ai_assistant(config)
    console.print("[bold blue]Launching AI Assistant...[/bold blue]")
    query = input("Enter your query: ")
    response = assistant_instance.process_query(query)
    console.print(f"[bold green]Response:[/bold green] {response}")

    # Optionally use ML Engine's AI assistant for simulation tasks
    from ml_engine_enhanced import EnhancedMLEngine

    engine_instance = EnhancedMLEngine(config)
    simulation_output = engine_instance.ai_assistant.process_query("simulate")
    console.print(
        f"[bold blue]AI Assistant Simulation Output:[/bold blue]\n{simulation_output}"
    )


def train_unified(config_path: str, csv_path: str | None = None) -> None:
    """Train the unified multi-head model and print head-health insights."""
    from unified_multitask_training import train_unified_multitask

    result = train_unified_multitask(config_path, csv_path=csv_path)
    console.print(
        f"[bold green]Unified training complete[/bold green] model={result.get('model_path')} metrics={result.get('metrics_path')}"
    )


def chat_unified(config_path: str, metrics_path: str | None = None) -> None:
    """Interactive chat over the latest unified head metrics."""
    from unified_chat import run_unified_chat

    run_unified_chat(config_path, metrics_path=metrics_path)


def talk_unified(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
    ticker: str | None = None,
    period: str = "5d",
    interval: str = "1h",
    verbose: bool = False,
) -> None:
    """Interactive REPL that runs the unified neural engine on-demand."""
    _configure_predict_output(verbose)
    from unified_talk import run_unified_talk

    run_unified_talk(
        config_path,
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
        ticker=ticker,
        period=period,
        interval=interval,
        verbose=verbose,
    )


def buddy(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "EUR_USD",
    granularity: str = "M5",
    candles: int = 300,
    execute: bool = False,
    verbose: bool = False,
) -> None:
    """Buddy: interactive offline REPL + OANDA demo/practice source."""
    _configure_predict_output(verbose)
    from unified_talk import run_unified_talk

    run_unified_talk(
        config_path,
        checkpoint_path=checkpoint_path,
        oanda=True,
        oanda_instrument=instrument,
        oanda_granularity=granularity,
        oanda_candles=candles,
        oanda_execute=execute,
        verbose=verbose,
        assistant_name="Buddy",
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ML Engine Trading Bot CLI")
    parser.add_argument(
        "command",
        nargs="?",
        default="predict",
        choices=[
            "train-model",
            "evaluate-model",
            "predict-price",
            "realtime-loop",
            "tune-model",
            "profile-pipeline",
            "visualize",  # New command
            "openai-tune",  # New command
            "ai-assistant",  # New sub-command
            "train-unified",
            "chat-unified",
            "talk-unified",
            "buddy",
            "Buddy",
            "integrated-predict",
            "predict",
            "fx",
            "fx-paper",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="./config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--csv",
        "-f",
        default=None,
        help="Path to market CSV file (used by integrated-predict)",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Path to unified training metrics JSON (used by chat-unified)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to unified checkpoint .pth (used by talk-unified)",
    )
    parser.add_argument(
        "--ticker",
        "-t",
        default=None,
        help="Ticker symbol to fetch live-ish data via yfinance (used by predict/integrated-predict)",
    )
    parser.add_argument(
        "--period",
        "-p",
        default="5d",
        help="yfinance period (used with --ticker), e.g. 1d,5d,1mo",
    )
    parser.add_argument(
        "--interval",
        "-i",
        default="1h",
        help="yfinance interval (used with --ticker), e.g. 1m,5m,1h,1d",
    )
    parser.add_argument(
        "--watch-seconds",
        "-w",
        default=None,
        type=int,
        help="If set with --ticker, rerun live predict every N seconds",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logs (default: quiet output)",
    )

    parser.add_argument(
        "--instrument",
        "-I",
        default="EUR_USD",
        help="OANDA instrument, e.g. EUR_USD (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--granularity",
        "-g",
        default="M5",
        help="OANDA candle granularity, e.g. M5, M15, H1 (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--candles",
        "-n",
        type=int,
        default=300,
        help="How many candles to fetch (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=10_000.0,
        help="Paper equity for sizing (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--risk",
        "-r",
        type=float,
        default=0.005,
        help="Risk per trade fraction (e.g. 0.005 = 0.5%%) (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--execute",
        "-x",
        action="store_true",
        help="Actually place a PRACTICE order on OANDA (default: dry-run)",
    )
    # ... add more CLI arguments as needed ...

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    command_map = {
        "train-model": train_model,
        "evaluate-model": evaluate_model,
        "predict-price": predict_price,  # New
        "realtime-loop": realtime_loop,  # New
        "tune-model": tune_model,  # New
        "profile-pipeline": profile_pipeline,  # New
        "visualize": visualize_dashboard,  # New mapping
        "openai-tune": openai_tune,  # New mapping
        "ai-assistant": run_ai_assistant,  # updated mapping
        "train-unified": train_unified,
        "chat-unified": chat_unified,
        "talk-unified": talk_unified,
        "buddy": buddy,
        "Buddy": buddy,
    }

    try:
        if args.command in {"integrated-predict", "predict"}:
            _configure_predict_output(args.verbose)
            if args.ticker:
                integrated_predict_live(
                    args.config,
                    ticker=args.ticker,
                    period=args.period,
                    interval=args.interval,
                    watch_seconds=args.watch_seconds,
                )
            else:
                integrated_predict(args.config, args.csv)
            return

        if args.command in {"fx", "fx-paper"}:
            fx_paper_trade(
                args.config,
                instrument=args.instrument,
                granularity=args.granularity,
                candles=args.candles,
                execute=args.execute,
                equity=args.equity,
                risk_per_trade_pct=args.risk,
            )
            return
        if args.command == "train-unified":
            command_map[args.command](args.config, args.csv)
        elif args.command == "chat-unified":
            command_map[args.command](args.config, args.metrics)
        elif args.command == "talk-unified":
            command_map[args.command](
                args.config,
                checkpoint_path=args.model_path,
                csv_path=args.csv,
                ticker=args.ticker,
                period=args.period,
                interval=args.interval,
                verbose=args.verbose,
            )
        elif args.command in {"buddy", "Buddy"}:
            command_map[args.command](
                args.config,
                checkpoint_path=args.model_path,
                instrument=args.instrument,
                granularity=args.granularity,
                candles=args.candles,
                execute=args.execute,
                verbose=args.verbose,
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
