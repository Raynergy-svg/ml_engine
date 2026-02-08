#!/usr/bin/env python3
"""FX trading execution and paper trading utilities."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel

from cli.io_utils import (
    console,
    load_config,
    BUDDY_META_FILENAME,
    TABLE_HEADER_STYLE,
)

if TYPE_CHECKING:
    # NeuralNetworkIntegrator is a legacy class; avoid import at runtime.
    pass


def _buddy_live_enabled_from_meta() -> tuple[bool, float]:
    """Return (live_enabled, confidence_threshold) from last Buddy training."""
    import json

    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    if not meta_path.exists():
        return False, 0.65
    try:
        meta = json.loads(meta_path.read_text())
        return bool(meta.get("live_enabled", False)), float(meta.get("live_confidence_threshold", 0.65))
    except Exception:
        return False, 0.65


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

    # Evaluate daily circuit breakers early so we can stop even if we later end up
    # in a HOLD / no-setup path. Profit stop may be configured per-band, but the
    # repo's Tier-1 defaults use a fixed 30% target for both medium/high.
    stop_hit, stop_reason, stop_kind = fxg.check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=pnl.get("realized_pct"),
        confidence_band="medium",
    )
    if stop_hit and stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        fxg.save_state(cfg, state)
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
    cfg: Dict[str, Any],
    policy: Any,
    *,
    instrument: str,
    signal: str,
    _price: float,
    atr_value: float,
    spread_pips: float,
) -> tuple[float, str, list[str]]:
    """Confidence score for Tier-1 gating (current source: fx_confidence_v1)."""
    import fx_guardrails as fxg
    from reasoning_enhanced import fx_confidence_v1

    max_spread = float(getattr(policy.costs, "max_spread_pips", {}).get(instrument, 0.0) or 0.0)
    msp = max_spread if max_spread > 0 else max(1e-9, float(spread_pips))
    params = dict((cfg.get("fx", {}) or {}).get("confidence_model") or {})

    payload = fx_confidence_v1(
        signal=str(signal),
        price=float(_price),
        atr=float(atr_value),
        spread_pips=float(spread_pips),
        max_spread_pips=float(msp),
        params=params,
    )
    confidence = float(payload.get("confidence") or 0.0)
    reasons = [str(x) for x in (payload.get("reasons") or [])]
    band = fxg.confidence_band(policy, confidence)
    return float(confidence), str(band), reasons


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
        fxg.save_state(cfg, state)
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
    # Determine buffer in pips. Prefer policy-level override `costs.price_bound_buffer_pips` if available.
    buffer_pips = None
    try:
        buffer_pips = float(_policy.costs.price_bound_buffer_pips)
    except Exception:
        try:
            # support dict-like policy.costs
            buffer_pips = float((_policy.get("costs") or {}).get("price_bound_buffer_pips", None))
        except Exception:
            buffer_pips = None

    # Default to a small buffer (1 pip) suitable for scalping; make configurable via policy.
    if buffer_pips is None:
        buffer_pips = 1.0

    buffer_price = float(buffer_pips) * pip_size(instrument)

    # Use the ask/bid with a modest buffer (avoid adding half-spread again which made bounds overly conservative).
    if int(units) > 0:
        return float(ask + buffer_price)
    return float(bid - buffer_price)


def _schedule_auto_close(client: Any, instrument: str, delay_s: float, *, verbose: bool = False) -> None:
    """Spawned in a daemon thread to close the instrument position after delay_s seconds.

    This is a best-effort helper for PRACTICE mode to ensure scalping-style trades
    are not left open beyond the desired timeframe.
    """
    import threading

    def _worker():
        try:
            if verbose:
                console.print(f"[dim]Auto-close thread[/dim]: sleeping {delay_s:.1f}s before closing {instrument}")
            time.sleep(max(0.0, float(delay_s)))
            try:
                # Prefer closing the specific trade if the create_market_order returned a trade id
                # The client may expose `close_trade(trade_id=...)` or fall back to close_position.
                if hasattr(client, "close_trade") and hasattr(client, "_last_trade_id") and client._last_trade_id:
                    tid = client._last_trade_id
                    try:
                        res = client.close_trade(trade_id=tid)
                        console.print(f"[dim]Auto-close[/dim]: closed trade {tid} for {instrument}: {res}")
                    except Exception:
                        res = client.close_position(instrument=instrument)
                        console.print(f"[dim]Auto-close[/dim]: fallback closed position for {instrument}: {res}")
                else:
                    res = client.close_position(instrument=instrument)
                    console.print(f"[dim]Auto-close[/dim]: closed position for {instrument}: {res}")
            except Exception as e:
                console.print(f"[yellow]Auto-close failed[/yellow]: could not close {instrument}: {e}")
        except Exception:
            # Never allow the worker to raise into the main thread
            return

    t = threading.Thread(target=_worker, daemon=True, name=f"auto-close-{instrument}")
    t.start()


def _build_integrated_engines(config: Dict[str, Any]):
    """Retired.

    Legacy integrated ML/MT/MR pipeline has been retired as part of the
    TensorFlow-only Buddy migration.
    """

    raise RuntimeError("Retired: integrated engines are no longer supported. Use `train-buddy`/`buddy`.")


def integrated_predict_once(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility shim for legacy integrated prediction.

    The production path is Buddy (`train-buddy`/`buddy`). Some unit tests still
    exercise the historical integrated API. Keep this no-network and lightweight.
    """

    import numpy as np

    batch_size = int((config or {}).get("batch_size") or 1)
    # A deterministic placeholder prediction tensor.
    prediction = np.zeros((batch_size, 1), dtype=np.float32)
    reasoning = {"insights": ["Integrated pipeline is deprecated; returned stub prediction."]}
    return {"prediction": prediction, "reasoning": reasoning}


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
    """Compatibility shim for legacy integrated feature tensors.

    Tests expect 3 tensors:
    - ml_features: (1, seq_len, 7)
    - mt_features: (1, seq_len, 11)
    - mr_features: (1, seq_len, 11)

    This implementation is intentionally simple and deterministic.
    """

    import numpy as np

    df = _normalize_market_dataframe(market_data_df)
    seq_len = int((config or {}).get("data", {}).get("sequence_length") or 60)

    # Base numeric series.
    o = df["open"].to_numpy(dtype=np.float32)
    h = df["high"].to_numpy(dtype=np.float32)
    low = df["low"].to_numpy(dtype=np.float32)
    c = df["close"].to_numpy(dtype=np.float32)
    v = df["volume"].to_numpy(dtype=np.float32)

    n = int(len(df))
    if n <= 0:
        raise ValueError("market_data_df must have at least 1 row")

    if n < seq_len:
        if not allow_pad:
            raise ValueError(f"Not enough rows ({n}) for sequence_length={seq_len}")
        pad = seq_len - n
        # Left-pad using the first row (conservative, stable).
        o = np.concatenate([np.repeat(o[:1], pad), o])
        h = np.concatenate([np.repeat(h[:1], pad), h])
        low = np.concatenate([np.repeat(low[:1], pad), low])
        c = np.concatenate([np.repeat(c[:1], pad), c])
        v = np.concatenate([np.repeat(v[:1], pad), v])
    else:
        o = o[-seq_len:]
        h = h[-seq_len:]
        low = low[-seq_len:]
        c = c[-seq_len:]
        v = v[-seq_len:]

    # Derived features.
    prev_c = np.concatenate([c[:1], c[:-1]])
    ret = (c - prev_c) / np.maximum(prev_c, 1e-8)
    rng = (h - low) / np.maximum(c, 1e-8)

    ml = np.stack([o, h, low, c, v, ret, rng], axis=-1).astype(np.float32)

    # MT/MR historical shapes include extra engineered slots; fill deterministically.
    zeros4 = np.zeros((seq_len, 4), dtype=np.float32)
    zeros4b = np.zeros((seq_len, 4), dtype=np.float32)
    mt = np.concatenate([ml, zeros4], axis=-1)
    mr = np.concatenate([ml, zeros4b], axis=-1)

    return {
        "ml_features": ml[None, :, :],
        "mt_features": mt[None, :, :],
        "mr_features": mr[None, :, :],
    }


def integrated_predict(config_path: str, csv_path: str | None = None) -> None:
    raise RuntimeError("Retired: integrated predict is no longer supported. Use `train-buddy`/`buddy`.")


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
    integrator: Any,
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


@dataclass(frozen=True)
class _FxPaperTradePlan:
    instrument: str
    granularity: str
    signal: str
    last_close: float
    atr_value: float
    units: int
    stop_price: float
    tp_price: float
    spread_pips: float
    slippage_pips: float
    slippage_price: float
    confidence: float
    band: str
    conf_reasons: list[str]


def _fx_setup_paper_trade(cfg: Dict[str, Any], *, instrument: str, granularity: str, execute: bool):
    from src.utils.oanda_practice import OandaPracticeClient
    import fx_guardrails as fxg

    policy = _fx_enforce_fx_policy(cfg, instrument=instrument, granularity=granularity)
    if policy is None:
        return None

    client = OandaPracticeClient.from_env()
    state = fxg.load_state(cfg, policy)

    pnl, _ = _fx_refresh_fx_state(cfg, policy, state, client)
    if _fx_maybe_force_flat(policy, state, client, execute=execute):
        return None

    if not _fx_require_account_metrics(pnl):
        return None

    if not _fx_gate_fx_entry(policy, state, client):
        return None

    return policy, client, state, pnl


def _fx_build_paper_trade_plan(
    cfg: Dict[str, Any],
    policy: Any,
    client: Any,
    state: Any,
    pnl: Dict[str, Any],
    *,
    instrument: str,
    granularity: str,
    candles: int,
    equity: float,
    risk_per_trade_pct: float,
) -> tuple["_FxPaperTradePlan", Any] | None:
    import fx_guardrails as fxg  # noqa: F401  (kept for symmetry; state saved elsewhere)

    df = _fx_load_fx_df(client, instrument=instrument, granularity=granularity, candles=candles)
    ok_spread, spread_pips, slippage_pips, slippage_price = _fx_spread_and_slippage(policy, df, instrument=instrument)
    if not ok_spread:
        return None

    signal, last_close, atr_value = _fx_get_signal_context(df)
    if signal == "hold":
        console.print(f"FX setup: HOLD  {instrument} {granularity}  close={last_close:.5f}  ATR14={atr_value:.5f}")
        return None

    rules = _fx_build_risk_rules(policy, pnl, equity=equity, risk_per_trade_pct=risk_per_trade_pct)

    confidence, band, conf_reasons = _fx_compute_confidence_and_band(
        cfg,
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
        return None

    if _fx_apply_daily_stops(cfg, policy, state, pnl, band=band):
        return None

    units, stop_price, tp_price, _, _ = _fx_build_order_units_and_prices(
        policy,
        rules,
        instrument=instrument,
        signal=signal,
        price=last_close,
        atr_value=atr_value,
        slippage_price=float(slippage_price),
    )

    plan = _FxPaperTradePlan(
        instrument=str(instrument),
        granularity=str(granularity),
        signal=str(signal),
        last_close=float(last_close),
        atr_value=float(atr_value),
        units=int(units),
        stop_price=float(stop_price),
        tp_price=float(tp_price),
        spread_pips=float(spread_pips),
        slippage_pips=float(slippage_pips),
        slippage_price=float(slippage_price),
        confidence=float(confidence),
        band=str(band),
        conf_reasons=[str(x) for x in (conf_reasons or [])],
    )
    return plan, rules


def _fx_execute_paper_trade_plan(
    cfg: Dict[str, Any],
    policy: Any,
    client: Any,
    state: Any,
    plan: _FxPaperTradePlan,
    rules: Any,
    *,
    execute: bool,
    verbose: bool,
) -> None:
    import fx_guardrails as fxg

    if not execute:
        console.print("[dim]Dry-run (no order). Pass --execute to place a PRACTICE order.[/dim]")
        return

    if policy.require_confirmation and not _fx_confirm("Confirm PLACE ORDER? (y/N): "):
        console.print("[dim]Order cancelled.[/dim]")
        return

    units = int(plan.units)

    # Apply any programmatic override (rules.force_units) before guard and submission.
    try:
        if hasattr(rules, "force_units") and rules.force_units is not None:
            fu = int(rules.force_units)
            units = -abs(int(fu)) if units < 0 else abs(int(fu))
            console.print(f"[dim]Force-units override (rules.force_units)[/dim]: using units={units}")
    except Exception:
        pass

    # Finalize units as integer (preserve sign) and submit exactly that value.
    try:
        units_final = int(units)
    except Exception:
        units_final = int(float(units))

    if verbose:
        console.print(f"[dim]Order sizing[/dim]: computed_units={units} final_submitted_units={units_final}")

    price_bound = _fx_execution_guard_price_bound(policy, client, instrument=plan.instrument, units=units_final)
    if price_bound is None:
        return

    result = client.create_market_order(
        instrument=plan.instrument,
        units=units_final,
        stop_loss_price=float(plan.stop_price),
        take_profit_price=float(plan.tp_price),
        price_bound=price_bound,
    )
    state.entries_today += 1
    fxg.save_state(cfg, state)
    console.print("[bold green]Order submitted (PRACTICE).[/bold green]")
    console.print(result)

    # Parse returned transaction to capture trade id(s) for precise auto-close.
    try:
        tx = (result or {}).get("orderFillTransaction") or (result or {}).get("orderCreateTransaction") or {}
        trade_id = None
        if isinstance(tx, dict):
            # Typical OANDA response may include 'tradeOpened' or 'tradesOpened' structures.
            to = tx.get("tradeOpened")
            if isinstance(to, dict):
                trade_id = to.get("tradeID") or to.get("id")
            tro = tx.get("tradesOpened")
            if trade_id is None and isinstance(tro, list) and len(tro) > 0:
                trade_id = tro[0].get("tradeID") or tro[0].get("id")
        if trade_id:
            try:
                # attach to client for auto-close worker to use
                client._last_trade_id = str(trade_id)
            except Exception:
                pass
    except Exception:
        pass

    # Schedule auto-close if `buddy.max_hold_minutes` is set in config (PRACTICE only).
    try:
        m_cfg = cfg.get("buddy", {}).get("max_hold_minutes", None)
        if m_cfg is not None:
            m = float(m_cfg)
            if m > 0:
                _schedule_auto_close(client, plan.instrument, delay_s=m * 60.0, verbose=bool(verbose))
    except Exception:
        pass


def fx_paper_trade(
    config_path: str,
    instrument: str,
    granularity: str = "M5",
    candles: int = 300,
    execute: bool = False,
    verbose: bool = False,
    equity: float = 10_000.0,
    risk_per_trade_pct: float = 0.005,
) -> None:
    """Paper trade forex on OANDA PRACTICE using a simple setup rule.

    This is intentionally conservative and defaults to DRY-RUN (no order placed).
    """
    cfg = load_config(config_path)

    setup = _fx_setup_paper_trade(cfg, instrument=instrument, granularity=granularity, execute=execute)
    if setup is None:
        return
    policy, client, state, pnl = setup

    planned = _fx_build_paper_trade_plan(
        cfg,
        policy,
        client,
        state,
        pnl,
        instrument=instrument,
        granularity=granularity,
        candles=candles,
        equity=equity,
        risk_per_trade_pct=risk_per_trade_pct,
    )
    if planned is None:
        return
    plan, rules = planned

    console.print(
        f"FX setup: {plan.signal.upper()}  {plan.instrument} {plan.granularity}  close={plan.last_close:.5f}  "
        f"units={plan.units}  SL={plan.stop_price:.5f}  TP={plan.tp_price:.5f}"
        f"  spread={plan.spread_pips:.2f}p  slip={plan.slippage_pips:.2f}p  conf={plan.confidence:.2f}  band={plan.band}"
    )

    _fx_execute_paper_trade_plan(
        cfg,
        policy,
        client,
        state,
        plan,
        rules,
        execute=execute,
        verbose=verbose,
    )


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
