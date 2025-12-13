"""Talk to the unified neural engine.

This is a minimal REPL that:
- loads the trained unified checkpoint (model + preprocessing metadata)
- runs predictions on a CSV or yfinance ticker
- answers in a chat-like format using the reasoning engine + head outputs

No LLM is required; the "AI" here is the trained neural network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from neural_engine_unified import UnifiedNeuralEngine, safe_torch_load
from reasoning_enhanced import ReasoningEngine
from utils import load_config

try:  # Optional dependency (listed in requirements.txt)
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


DEFAULT_ASSISTANT_NAME = "Assistant"


def _assistant_prefix(ctx: "TalkContext") -> str:
    return f"{ctx.assistant_name}: "


def _load_project_dotenv() -> None:
    """Load env vars from common local files.

    We support both `.env.local` and `.env` so Buddy can run with either.
    """
    if load_dotenv is None:
        return

    try:
        cwd = Path.cwd()
        for name in (".env.local", ".env"):
            p = cwd / name
            if p.exists() and p.is_file():
                load_dotenv(dotenv_path=str(p), override=False)
    except Exception:
        # Never block REPL startup due to dotenv issues.
        return


@dataclass
class TalkContext:
    config_path: str
    checkpoint_path: str
    feature_columns: list[str]
    sequence_length: int
    target_shift: int
    engine: UnifiedNeuralEngine
    reasoning: ReasoningEngine
    feature_scaler: Any = None
    target_scaler: Any = None
    active_source: Optional[str] = None
    active_df: Optional[pd.DataFrame] = None
    last_result: Optional[Dict[str, Any]] = None

    # OANDA demo/practice integration (initialized lazily)
    oanda_client: Any = None
    oanda_instrument: str = "EUR_USD"
    oanda_granularity: str = "M5"
    oanda_candles: int = 300
    oanda_price: str = "MBA"
    oanda_execute: bool = False

    # UI/verbosity
    assistant_name: str = DEFAULT_ASSISTANT_NAME
    verbose: bool = False


def select_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """Pick the newest unified checkpoint in a directory.

    Prefer filenames containing 'unified' to avoid accidentally loading unrelated models.
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    candidates = [p for p in model_dir.glob("*.pth") if p.is_file()]
    if not candidates:
        return None

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0

    unified = [p for p in candidates if "unified" in p.name.lower()]
    preferred = unified if unified else candidates
    preferred.sort(key=_mtime, reverse=True)
    return preferred[0] if preferred else None


def _default_checkpoint_path(cfg: Dict[str, Any]) -> str:
    model_dir = Path(cfg.get("MODEL_DIR") or cfg.get("paths", {}).get("model_dir") or "trained_data/models")
    latest = select_latest_checkpoint(model_dir)
    if latest is not None:
        return str(latest)
    return str(model_dir / "unified_market_net.pth")


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    # yfinance uses Title Case.
    rename_map = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key == "adj close":
            rename_map[col] = "close"
    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def _load_df_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_ohlcv_columns(df)


def _load_df_from_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Ticker mode requires `yfinance`.") from e

    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({period}, {interval})")
    df = df.reset_index()
    return _normalize_ohlcv_columns(df)


def _load_df_from_oanda(
    ctx: TalkContext,
    *,
    instrument: str,
    granularity: str,
    candles: int,
    price: str,
) -> pd.DataFrame:
    _load_project_dotenv()

    if ctx.oanda_client is None:
        from oanda_practice import OandaPracticeClient

        ctx.oanda_client = OandaPracticeClient.from_env()

    from fx_paper import candles_to_ohlcv_df

    resp = ctx.oanda_client.get_candles(
        instrument,
        granularity=granularity,
        count=int(candles),
        price=price,
    )
    df = candles_to_ohlcv_df(resp)
    return _normalize_ohlcv_columns(df)


def _coerce_sequence(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    sequence_length: int,
    feature_scaler: Any,
) -> np.ndarray:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    window = df[feature_columns].tail(int(sequence_length)).copy()
    if len(window) < int(sequence_length):
        raise ValueError(f"Need at least {sequence_length} rows; got {len(window)}")

    arr = window.to_numpy(dtype=np.float32, copy=True)

    # Use saved scaler when available.
    if feature_scaler is not None:
        arr = feature_scaler.transform(arr)

    return arr


def _inverse_target(y_scaled: np.ndarray, target_scaler: Any) -> np.ndarray:
    if target_scaler is None:
        return y_scaled
    y2 = y_scaled.reshape(-1, 1)
    return target_scaler.inverse_transform(y2).reshape(-1)


def _predict_one(ctx: TalkContext, df: pd.DataFrame) -> Dict[str, Any]:
    seq = _coerce_sequence(
        df,
        feature_columns=ctx.feature_columns,
        sequence_length=ctx.sequence_length,
        feature_scaler=ctx.feature_scaler,
    )

    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)
    out = ctx.engine.predict(x)

    pred_scaled = np.asarray(out.get("prediction"), dtype=float).reshape(-1)
    pred = _inverse_target(pred_scaled, ctx.target_scaler)

    last_close = None
    if "close" in df.columns:
        try:
            last_close = float(df["close"].iloc[-1])
        except Exception:
            last_close = None

    state_probs = out.get("state_probs")
    if state_probs is not None:
        state_probs = np.asarray(state_probs, dtype=float).reshape(-1)

    return {
        "prediction": float(pred[-1]) if len(pred) else None,
        "prediction_scaled": float(pred_scaled[-1]) if len(pred_scaled) else None,
        "uncertainty": float(out.get("uncertainty")) if out.get("uncertainty") is not None else None,
        "trend": float(np.asarray(out.get("trend"), dtype=float).reshape(-1)[-1]) if out.get("trend") is not None else None,
        "risk": float(np.asarray(out.get("risk"), dtype=float).reshape(-1)[-1]) if out.get("risk") is not None else None,
        "state_probs": state_probs.tolist() if state_probs is not None else None,
        "last_close": last_close,
    }


def _format_prediction_lines(pred: Any, last_close: Any) -> list[str]:
    if pred is None:
        return []

    pred_f = float(pred)
    if last_close is None:
        return [f"Next-close estimate: {pred_f:.4f}"]

    last_close_f = float(last_close)
    delta = pred_f - last_close_f
    pct = (delta / last_close_f) if last_close_f != 0 else 0.0
    return [f"Next-close estimate: {pred_f:.4f} (last close {last_close_f:.4f}, Δ {delta:+.4f}, {pct:+.2%})"]


def _format_head_lines(trend: Any, risk: Any) -> list[str]:
    lines: list[str] = []
    if trend is not None:
        lines.append(f"Trend head: {float(trend):+.6f} (higher = more upward)")
    if risk is not None:
        lines.append(f"Risk/uncertainty head: {float(risk):.3f} (0 low → 1 high)")
    return lines


def _format_state_probs_line(state_probs: Any) -> Optional[str]:
    if state_probs is None or not isinstance(state_probs, list) or len(state_probs) == 0:
        return None
    best = int(np.argmax(np.asarray(state_probs, dtype=float)))
    return f"State probs: {state_probs} (most likely state={best})"


def _format_trade_note(q: str, trend: Any, risk: Any) -> Optional[str]:
    ql = q.lower()
    if not any(k in ql for k in ("should", "buy", "sell", "trade")):
        return None

    if risk is not None and float(risk) > 0.7:
        return "Trade note: risk looks high; consider smaller size or waiting."
    if trend is not None and float(trend) > 0 and (risk is None or float(risk) < 0.6):
        return "Trade note: trend is positive with moderate risk; consider long bias (not financial advice)."
    if trend is not None and float(trend) < 0 and (risk is None or float(risk) < 0.6):
        return "Trade note: trend is negative with moderate risk; consider caution/short bias (not financial advice)."
    return None


def _direction_label(delta: float) -> str:
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def _risk_bucket(risk: float) -> str:
    if risk < 0.35:
        return "low"
    if risk < 0.7:
        return "medium"
    return "high"


def _summary_line(pred: Any, last_close: Any) -> Optional[str]:
    if pred is None or last_close is None:
        return None
    try:
        pred_f = float(pred)
        last_f = float(last_close)
        delta = pred_f - last_f
        pct = (delta / last_f) if last_f != 0 else 0.0
        direction = _direction_label(delta)
        return f"I'm leaning {direction}. Estimated next close {pred_f:.5f} vs {last_f:.5f} ({pct:+.2%})."
    except Exception:
        return None


def _render_answer(q: str, result: Dict[str, Any]) -> str:
    pred = result.get("prediction")
    last_close = result.get("last_close")
    risk = result.get("risk")
    trend = result.get("trend")

    lines: list[str] = []

    # Keep output concise by default.
    summary = _summary_line(pred, last_close)
    if summary is not None:
        lines.append(summary)
    else:
        lines.extend(_format_prediction_lines(pred, last_close))

    # Risk/trend summary in plain language.
    if risk is not None:
        try:
            r = float(risk)
            lines.append(f"Risk looks {_risk_bucket(r)}.")
        except Exception:
            pass

    # Only show raw heads/probabilities when explicitly requested via verbose mode.
    # (Verbose output is added by the caller when needed.)

    trade_note = _format_trade_note(q, trend, risk)
    if trade_note:
        lines.append(trade_note)

    if not lines:
        return "I ran the model but didn't get a usable prediction."

    return "\n".join(lines)


def _render_answer_verbose(q: str, result: Dict[str, Any]) -> str:
    pred = result.get("prediction")
    last_close = result.get("last_close")
    risk = result.get("risk")
    trend = result.get("trend")
    state_probs = result.get("state_probs")

    lines: list[str] = []
    lines.extend(_format_prediction_lines(pred, last_close))
    lines.extend(_format_head_lines(trend, risk))

    state_line = _format_state_probs_line(state_probs)
    if state_line:
        lines.append(state_line)

    trade_note = _format_trade_note(q, trend, risk)
    if trade_note:
        lines.append(trade_note)

    if not lines:
        return "I ran the model but didn't get a usable prediction."

    return "\n".join(lines)


def _load_context(config_path: str, *, checkpoint_path: Optional[str] = None) -> TalkContext:
    cfg = load_config(config_path)
    ckpt_path = checkpoint_path or _default_checkpoint_path(cfg)

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Unified checkpoint not found: {ckpt_path}. Run: python main.py train-unified --config {config_path} --csv <file.csv>"
        )

    engine = UnifiedNeuralEngine.from_checkpoint(ckpt_path, device=cfg.get("device"))
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    data_meta = dict(ckpt.get("data_meta") or {})

    feature_columns = list(data_meta.get("feature_columns") or ["open", "high", "low", "close", "volume"])
    sequence_length = int(data_meta.get("sequence_length") or cfg.get("data", {}).get("sequence_length") or cfg.get("sequence_length") or 60)
    target_shift = int(data_meta.get("target_shift") or cfg.get("data", {}).get("target_shift") or cfg.get("target_shift") or 1)

    reasoning = ReasoningEngine(cfg.get("reasoning", {}))

    return TalkContext(
        config_path=config_path,
        checkpoint_path=ckpt_path,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_shift=target_shift,
        engine=engine,
        reasoning=reasoning,
        feature_scaler=data_meta.get("feature_scaler"),
        target_scaler=data_meta.get("target_scaler"),
    )


def _print_talk_banner(ctx: TalkContext) -> None:
    if ctx.verbose:
        print("Unified engine talk mode.")
        print(f"Model: {ctx.checkpoint_path}")
        print(
            f"Features: {ctx.feature_columns} | seq_len={ctx.sequence_length} | shift={ctx.target_shift}"
        )
        print(
            "Commands: 'use csv <path>' | 'use ticker <SYM>' | 'use oanda [INSTR GRAN COUNT]' | 'predict' | 'help' | 'exit'"
        )
        print(
            "Tip: after you run 'predict' once, you can ask natural questions like 'should I buy?' or 'what's the trend?'."
        )
        return

    print(_assistant_prefix(ctx) + "Ready. Ask a question, or type 'help'.")


def _maybe_run_initial_prediction(
    ctx: TalkContext,
    *,
    csv_path: Optional[str],
    ticker: Optional[str],
    period: str,
    interval: str,
    oanda: bool,
) -> None:
    if csv_path:
        df = _load_df_from_csv(csv_path)
        ctx.active_source = f"csv:{csv_path}"
        ctx.active_df = df
        result = _predict_one(ctx, df)
        ctx.last_result = result
        ans = _render_answer_verbose("predict", result) if ctx.verbose else _render_answer("predict", result)
        print(_assistant_prefix(ctx) + ans)
        return

    if ticker:
        df = _load_df_from_ticker(ticker, period=period, interval=interval)
        ctx.active_source = f"ticker:{ticker}"
        ctx.active_df = df
        result = _predict_one(ctx, df)
        ctx.last_result = result
        ans = _render_answer_verbose("predict", result) if ctx.verbose else _render_answer("predict", result)
        print(_assistant_prefix(ctx) + ans)

    if oanda and not csv_path and not ticker:
        try:
            df = _load_df_from_oanda(
                ctx,
                instrument=ctx.oanda_instrument,
                granularity=ctx.oanda_granularity,
                candles=ctx.oanda_candles,
                price=ctx.oanda_price,
            )
            ctx.active_source = f"oanda:{ctx.oanda_instrument}:{ctx.oanda_granularity}:{ctx.oanda_candles}"
            ctx.active_df = df
            # Claude-like feel: don't print a prediction on startup unless verbose.
            if ctx.verbose:
                result = _predict_one(ctx, df)
                ctx.last_result = result
                print(_assistant_prefix(ctx) + _render_answer_verbose("predict", result))
        except Exception as e:
            print(_assistant_prefix(ctx) + f"OANDA not ready ({e}). You can still use 'use csv ...' or 'use ticker ...'.")


def _extract_command_arg(q: str, prefix: str) -> Optional[str]:
    ql = q.lower()
    if not ql.startswith(prefix):
        return None
    parts = q.split(" ", 2)
    if len(parts) < 3:
        return None
    return parts[2].strip()


def _answer_from_last(ctx: TalkContext, q: str) -> str:
    if ctx.last_result is None:
        return "I don't have a recent prediction yet. Type 'predict' first, or start with 'use oanda ...'."

    header = f"Based on latest run ({ctx.active_source or 'unknown source'}):"
    return header + "\n" + _render_answer(q, ctx.last_result)


def _ensure_oanda_client(ctx: TalkContext) -> None:
    _load_project_dotenv()
    if ctx.oanda_client is None:
        from oanda_practice import OandaPracticeClient

        ctx.oanda_client = OandaPracticeClient.from_env()


def _handle_basic_commands(ctx: TalkContext, ql: str) -> Optional[bool]:
    if ql in {"exit", "quit", "q"}:
        print("Bye.")
        return False

    if ql in {"help", "?"}:
        print(
            f"{ctx.assistant_name}: Try 'use csv market_data/MSFT_data.csv' then 'predict'. Or 'use ticker MSFT' then 'predict'.\n"
            f"{ctx.assistant_name}: OANDA demo/practice: 'use oanda EUR_USD M5 300' then 'predict'.\n"
            f"{ctx.assistant_name}: Trading (practice): 'trade buy 1000 EUR_USD' (dry-run unless 'execute on').\n"
            f"{ctx.assistant_name}: After a prediction, ask 'should I buy?', 'what is the risk?', 'what's the trend?'."
        )
        return True

    if ql in {"execute on", "execute true"}:
        ctx.oanda_execute = True
        print(_assistant_prefix(ctx) + "execute mode ON (practice orders will be sent).")
        return True
    if ql in {"execute off", "execute false"}:
        ctx.oanda_execute = False
        print(_assistant_prefix(ctx) + "execute mode OFF (dry-run).")
        return True

    return None


def _handle_use_commands(ctx: TalkContext, q: str, *, period: str, interval: str) -> Optional[bool]:
    use_csv_arg = _extract_command_arg(q, "use csv ")
    if use_csv_arg is not None:
        df = _load_df_from_csv(use_csv_arg)
        ctx.active_source = f"csv:{use_csv_arg}"
        ctx.active_df = df
        ctx.last_result = None
        print(_assistant_prefix(ctx) + f"loaded {ctx.active_source}.")
        return True

    use_ticker_arg = _extract_command_arg(q, "use ticker ")
    if use_ticker_arg is not None:
        df = _load_df_from_ticker(use_ticker_arg, period=period, interval=interval)
        ctx.active_source = f"ticker:{use_ticker_arg}"
        ctx.active_df = df
        ctx.last_result = None
        print(_assistant_prefix(ctx) + f"loaded {ctx.active_source}.")
        return True

    use_oanda_arg = _extract_command_arg(q, "use oanda ")
    if use_oanda_arg is not None:
        parts = [p for p in use_oanda_arg.split() if p]
        if len(parts) >= 1:
            ctx.oanda_instrument = parts[0]
        if len(parts) >= 2:
            ctx.oanda_granularity = parts[1]
        if len(parts) >= 3:
            try:
                ctx.oanda_candles = int(parts[2])
            except Exception:
                raise ValueError("COUNT must be an integer")

        df = _load_df_from_oanda(
            ctx,
            instrument=ctx.oanda_instrument,
            granularity=ctx.oanda_granularity,
            candles=ctx.oanda_candles,
            price=ctx.oanda_price,
        )
        ctx.active_source = f"oanda:{ctx.oanda_instrument}:{ctx.oanda_granularity}:{ctx.oanda_candles}"
        ctx.active_df = df
        ctx.last_result = None
        print(_assistant_prefix(ctx) + f"loaded {ctx.active_source}.")
        return True

    return None


def _handle_predict_commands(ctx: TalkContext, q: str, ql: str, *, period: str, interval: str) -> Optional[bool]:
    csv_arg = _extract_command_arg(q, "predict csv ")
    if csv_arg is not None:
        df = _load_df_from_csv(csv_arg)
        result = _predict_one(ctx, df)
        ctx.active_source = f"csv:{csv_arg}"
        ctx.active_df = df
        ctx.last_result = result
        ans = _render_answer_verbose(q, result) if ctx.verbose else _render_answer(q, result)
        print(_assistant_prefix(ctx) + ans)
        return True

    ticker_arg = _extract_command_arg(q, "predict ticker ")
    if ticker_arg is not None:
        df = _load_df_from_ticker(ticker_arg, period=period, interval=interval)
        result = _predict_one(ctx, df)
        ctx.active_source = f"ticker:{ticker_arg}"
        ctx.active_df = df
        ctx.last_result = result
        ans = _render_answer_verbose(q, result) if ctx.verbose else _render_answer(q, result)
        print(_assistant_prefix(ctx) + ans)
        return True

    if ql == "predict":
        if ctx.active_df is None:
            print(_assistant_prefix(ctx) + "pick a source first: 'use oanda ...' or 'use csv ...' or 'use ticker ...'.")
            return True
        if ctx.active_source and ctx.active_source.startswith("oanda:"):
            ctx.active_df = _load_df_from_oanda(
                ctx,
                instrument=ctx.oanda_instrument,
                granularity=ctx.oanda_granularity,
                candles=ctx.oanda_candles,
                price=ctx.oanda_price,
            )
        result = _predict_one(ctx, ctx.active_df)
        ctx.last_result = result
        ans = _render_answer_verbose("predict", result) if ctx.verbose else _render_answer("predict", result)
        print(_assistant_prefix(ctx) + ans)
        return True

    return None


def _handle_oanda_info_commands(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    if ql.startswith("oanda quote"):
        _ensure_oanda_client(ctx)
        parts = q.split()
        instrument = parts[-1] if len(parts) >= 3 else ctx.oanda_instrument
        quote = ctx.oanda_client.get_price_quote(instrument=instrument)
        print(_assistant_prefix(ctx) + f"{instrument} bid={quote['bid']:.6f} ask={quote['ask']:.6f}")
        return True

    if ql in {"oanda account", "oanda summary"}:
        _ensure_oanda_client(ctx)
        summary = ctx.oanda_client.get_account_summary()
        acct = (summary or {}).get("account") or {}
        nav = acct.get("NAV")
        bal = acct.get("balance")
        margin = acct.get("marginAvailable")
        print(_assistant_prefix(ctx) + f"account NAV={nav} balance={bal} marginAvailable={margin}")
        return True

    return None


def _trade_instrument(parts: list[str], *, default: str, idx: int) -> str:
    try:
        val = parts[idx]
    except Exception:
        return default
    return val or default


def _handle_trade_close(ctx: TalkContext, parts: list[str]) -> None:
    instrument = _trade_instrument(parts, default=ctx.oanda_instrument, idx=2)
    if not ctx.oanda_execute:
        print(_assistant_prefix(ctx) + f"dry-run: would close position for {instrument}. (run 'execute on' to send)")
        return
    ctx.oanda_client.close_position(instrument=instrument)
    print(_assistant_prefix(ctx) + f"close_position sent for {instrument}.")


def _parse_trade_units(parts: list[str]) -> int:
    if len(parts) < 3:
        raise ValueError("Usage: trade buy <units> [instrument]")
    try:
        return int(parts[2])
    except Exception:
        raise ValueError("units must be an integer")


def _handle_trade_market(ctx: TalkContext, *, action: str, parts: list[str]) -> None:
    units = _parse_trade_units(parts)
    instrument = _trade_instrument(parts, default=ctx.oanda_instrument, idx=3)
    signed_units = abs(int(units)) * (1 if action == "buy" else -1)

    if not ctx.oanda_execute:
        print(
            _assistant_prefix(ctx)
            + f"dry-run: would place MARKET order {action.upper()} {abs(signed_units)} {instrument}. "
            + "Run 'execute on' to send."
        )
        return

    resp = ctx.oanda_client.create_market_order(instrument=instrument, units=signed_units)
    tx = (resp or {}).get("orderFillTransaction") or (resp or {}).get("orderCreateTransaction") or {}
    print(_assistant_prefix(ctx) + f"order submitted (id={tx.get('id')}).")


def _handle_trade_commands(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    if not ql.startswith("trade "):
        return None

    parts = q.split()
    if len(parts) < 2:
        raise ValueError("Usage: trade buy|sell|close ...")

    action = parts[1].lower().strip()
    if action not in {"buy", "sell", "close"}:
        raise ValueError(
            "Usage: trade buy <units> [instrument] | trade sell <units> [instrument] | trade close [instrument]"
        )

    _ensure_oanda_client(ctx)
    if action == "close":
        _handle_trade_close(ctx, parts)
        return True
    _handle_trade_market(ctx, action=action, parts=parts)
    return True


def _handle_talk_command(ctx: TalkContext, q: str, *, period: str, interval: str) -> bool:
    ql = q.lower().strip()

    for handler in (
        lambda: _handle_basic_commands(ctx, ql),
        lambda: _handle_use_commands(ctx, q, period=period, interval=interval),
        lambda: _handle_predict_commands(ctx, q, ql, period=period, interval=interval),
        lambda: _handle_oanda_info_commands(ctx, q, ql),
        lambda: _handle_trade_commands(ctx, q, ql),
    ):
        out = handler()
        if out is not None:
            return out

    if ctx.active_df is not None and ctx.last_result is None:
        ctx.last_result = _predict_one(ctx, ctx.active_df)

    ans = _render_answer_verbose(q, ctx.last_result) if ctx.verbose else _render_answer(q, ctx.last_result)
    print(_assistant_prefix(ctx) + ans)
    return True


def _run_repl(ctx: TalkContext, *, period: str, interval: str) -> None:
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not q:
            continue

        try:
            if not _handle_talk_command(ctx, q, period=period, interval=interval):
                return
        except Exception as e:
            print(_assistant_prefix(ctx) + f"error: {e}")


def run_unified_talk(
    config_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    ticker: Optional[str] = None,
    period: str = "5d",
    interval: str = "1h",
    oanda: bool = False,
    oanda_instrument: str = "EUR_USD",
    oanda_granularity: str = "M5",
    oanda_candles: int = 300,
    oanda_execute: bool = False,
    verbose: bool = False,
    assistant_name: str = DEFAULT_ASSISTANT_NAME,
) -> None:
    ctx = _load_context(config_path, checkpoint_path=checkpoint_path)

    ctx.verbose = bool(verbose)
    ctx.assistant_name = str(assistant_name or DEFAULT_ASSISTANT_NAME)

    ctx.oanda_instrument = oanda_instrument
    ctx.oanda_granularity = oanda_granularity
    ctx.oanda_candles = int(oanda_candles)
    ctx.oanda_execute = bool(oanda_execute)

    _print_talk_banner(ctx)
    _maybe_run_initial_prediction(
        ctx,
        csv_path=csv_path,
        ticker=ticker,
        period=period,
        interval=interval,
        oanda=oanda,
    )
    _run_repl(ctx, period=period, interval=interval)
