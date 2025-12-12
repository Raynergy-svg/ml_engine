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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from neural_engine_unified import UnifiedNeuralEngine
from reasoning_enhanced import ReasoningEngine
from utils import load_config


ASSISTANT_PREFIX = "Assistant:\n"


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


def _default_checkpoint_path(cfg: Dict[str, Any]) -> str:
    model_dir = Path(cfg.get("MODEL_DIR") or cfg.get("paths", {}).get("model_dir") or "trained_data/models")
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


def _render_answer(q: str, result: Dict[str, Any]) -> str:
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
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
    print("Unified engine talk mode.")
    print(f"Model: {ctx.checkpoint_path}")
    print(f"Features: {ctx.feature_columns} | seq_len={ctx.sequence_length} | shift={ctx.target_shift}")
    print("Type: 'predict csv <path>' or 'predict ticker <SYM>' or 'exit'.")


def _maybe_run_initial_prediction(
    ctx: TalkContext,
    *,
    csv_path: Optional[str],
    ticker: Optional[str],
    period: str,
        df = _load_df_from_csv(csv_path)
        result = _predict_one(ctx, df)
        print(ASSISTANT_PREFIX + _render_answer("predict", result))
        return

    if ticker:
        df = _load_df_from_ticker(ticker, period=period, interval=interval)
        result = _predict_one(ctx, df)
        print(ASSISTANT_PREFIX + _render_answer("predict", result))
        df = _load_df_from_ticker(ticker, period=period, interval=interval)
        result = _predict_one(ctx, df)
        print("Assistant:\n" + _render_answer("predict", result))


def _extract_command_arg(q: str, prefix: str) -> Optional[str]:
    ql = q.lower()
    if not ql.startswith(prefix):
        return None
    return q.split(" ", 2)[2].strip()


def _handle_talk_command(ctx: TalkContext, q: str, *, period: str, interval: str) -> bool:
    ql = q.lower()
    if ql in {"exit", "quit", "q"}:
        print("Bye.")
        return False

    if ql in {"help", "?"}:
        print("Assistant: Try 'predict csv market_data/MSFT_data.csv' or 'predict ticker MSFT'.")
    if csv_arg is not None:
        df = _load_df_from_csv(csv_arg)
        result = _predict_one(ctx, df)
        print(ASSISTANT_PREFIX + _render_answer(q, result))
        return True

    ticker_arg = _extract_command_arg(q, "predict ticker ")
    if ticker_arg is not None:
        df = _load_df_from_ticker(ticker_arg, period=period, interval=interval)
        result = _predict_one(ctx, df)
        print(ASSISTANT_PREFIX + _render_answer(q, result))
        return True
        result = _predict_one(ctx, df)
        print("Assistant:\n" + _render_answer(q, result))
        return True

    # Default: treat as an intent to explain the last prediction needs context.
    print("Assistant: I can run the model if you ask 'predict csv ...' or 'predict ticker ...'.")
    return True


def run_unified_talk(
    config_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    ticker: Optional[str] = None,
    period: str = "5d",
    interval: str = "1h",
) -> None:
    ctx = _load_context(config_path, checkpoint_path=checkpoint_path)

    _print_talk_banner(ctx)
    _maybe_run_initial_prediction(ctx, csv_path=csv_path, ticker=ticker, period=period, interval=interval)

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
            print(f"Assistant: error: {e}")
