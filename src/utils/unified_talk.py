"""Talk to the unified neural engine.

This is a minimal REPL that:
- loads the trained unified checkpoint (model + preprocessing metadata)
- runs predictions on a CSV or yfinance ticker
- answers in a chat-like format using the reasoning engine + head outputs

Supports both PyTorch (.pth) and TensorFlow (.keras) models.
No LLM is required; the "AI" here is the trained neural network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import contextlib
import io

import numpy as np
import pandas as pd
from rich.console import Console
from rich.text import Text
from src.utils.buddy_knowledge import answer_buddy_question
import re
import threading

# PyTorch was removed from this repository. Unified talk supports
# TensorFlow-only checkpoints.
torch = None  # type: ignore

logger = logging.getLogger(__name__)


def safe_torch_load(*args: Any, **kwargs: Any):
    raise ImportError("PyTorch was removed; load a TensorFlow checkpoint instead.")


UnifiedNeuralEngine = None  # type: ignore
from src.utils.reasoning_enhanced import ReasoningEngine  # noqa: E402
from src.utils.utils import load_config  # noqa: E402

# TensorFlow support (lazy import)
from src.models.tensorflow_engine_unified import (  # noqa: E402
    TensorFlowUnifiedEngine,
    is_tensorflow_checkpoint,
    get_default_tf_checkpoint,
)

try:  # Optional dependency (listed in requirements.txt)
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


DEFAULT_ASSISTANT_NAME = "Assistant"

# Constants for execution status display
EXEC_STATUS_LIVE = "🟢 LIVE"
EXEC_STATUS_DRY_RUN = "⏸️ DRY-RUN"


def _assistant_prefix(ctx: "TalkContext") -> str:
    if _planner_chat_enabled():
        return "Buddy Planner 3B: "
    return f"{ctx.assistant_name}: "


def _planner_chat_enabled() -> bool:
    return os.getenv("BUDDY_CHAT_MODE", "").strip().lower() == "planner"


def _remember_chat_turn(ctx: "TalkContext", role: str, content: str) -> None:
    """Keep a short rolling chat history for grounded synthesis."""
    text = (content or "").strip()
    if not text:
        return
    ctx.chat_history.append({"role": role, "content": text})
    if len(ctx.chat_history) > 8:
        ctx.chat_history = ctx.chat_history[-8:]


def _recent_chat_context(ctx: "TalkContext", turns: int = 4) -> str:
    if not ctx.chat_history:
        return "No earlier chat context."
    recent = ctx.chat_history[-turns:]
    lines = [f"{item['role']}: {item['content']}" for item in recent if item.get("content")]
    return "\n".join(lines) if lines else "No earlier chat context."


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
    engine: Union[UnifiedNeuralEngine, TensorFlowUnifiedEngine]  # Supports both PyTorch and TensorFlow
    reasoning: ReasoningEngine
    feature_scaler: Any = None
    target_scaler: Any = None
    active_source: Optional[str] = None
    active_df: Optional[pd.DataFrame] = None
    last_result: Optional[Dict[str, Any]] = None

    # Feature engineering flag (True if model was trained with all features)
    apply_feature_engineering: bool = False

    # Model type flag
    is_tensorflow: bool = False

    # OANDA demo/practice integration (initialized lazily)
    oanda_client: Any = None
    oanda_instrument: str = "USD_JPY"
    oanda_granularity: str = "M5"
    oanda_candles: int = 300
    oanda_price: str = "MBA"
    oanda_execute: bool = True  # Live trading enabled by default

    # Risk management: Stop Loss and Take Profit (in pips)
    stop_loss_pips: float = 20.0  # Default 20 pips stop loss
    take_profit_pips: float = 40.0  # Default 40 pips take profit (2:1 risk/reward)
    use_stop_loss: bool = True
    use_take_profit: bool = True

    # UI/verbosity
    assistant_name: str = DEFAULT_ASSISTANT_NAME
    verbose: bool = False

    # Prevent concurrent model inference (TensorFlow is not reliably thread-safe).
    inference_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # New confidence system components
    confidence_calibrator: Any = None  # ConfidenceCalibrator instance
    position_sizer: Any = None  # DynamicPositionSizer instance
    risk_manager: Any = None  # ConfidenceBasedRiskManager instance
    chat_history: list[dict[str, str]] = field(default_factory=list)
    last_scan_summary: Optional[dict[str, Any]] = None
    planner_runtime: Any = None
    planner_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class TalkPlannedStep:
    """One planned action in a natural-language talk request."""
    action: str
    value: Optional[str] = None


@dataclass
class TalkIntentPlan:
    """A lightweight action plan derived from a natural-language request."""
    steps: list[TalkPlannedStep] = field(default_factory=list)
    answer_after: bool = False


_PLANNER_CONSOLE = Console()


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
    """Get default checkpoint path, preferring TensorFlow over PyTorch.

    Priority:
    1. TensorFlow checkpoint (trained_data/checkpoints/tensorflow/tf_model_best.keras)
    2. PyTorch checkpoint (trained_data/models/unified_market_net.pth)
    """
    # First, check for TensorFlow checkpoint (preferred)
    tf_checkpoint = Path(get_default_tf_checkpoint())
    if tf_checkpoint.exists():
        return str(tf_checkpoint)

    # Fall back to PyTorch checkpoint
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


def _apply_feature_engineering_if_needed(ctx: TalkContext, df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to DataFrame if the model was trained with all features."""
    if not ctx.apply_feature_engineering:
        return df
    from src.data.feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    return fe.create_features(df, include_all=True)


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

    from src.utils.fx_paper import candles_to_ohlcv_df

    try:
        resp = ctx.oanda_client.get_candles(
            instrument,
            granularity=granularity,
            count=int(candles),
            price=price,
        )
    except Exception as e:
        print(_assistant_prefix(ctx) + f"❌ Failed to fetch candles: {e}")
        raise

    df = candles_to_ohlcv_df(resp)
    df = _normalize_ohlcv_columns(df)

    if ctx.apply_feature_engineering:
        print(_assistant_prefix(ctx) + "Applying feature engineering...")

    # Apply feature engineering if model requires it
    df = _apply_feature_engineering_if_needed(ctx, df)

    return df


def _coerce_sequence(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    sequence_length: int,
    feature_scaler: Any,
    use_all_numeric: bool = False,
    expected_features: int = None,
) -> np.ndarray:
    """Extract feature sequence from DataFrame.

    Args:
        df: DataFrame with features
        feature_columns: List of columns to use (ignored if use_all_numeric=True)
        sequence_length: Number of timesteps
        feature_scaler: Optional scaler for normalization
        use_all_numeric: If True, use all numeric columns (for TensorFlow)
        expected_features: If set, select first N numeric features
    """
    if use_all_numeric:
        # Use all numeric columns for TensorFlow models
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # If expected_features is set, limit to that number
        if expected_features and len(numeric_cols) > expected_features:
            numeric_cols = numeric_cols[:expected_features]
        elif expected_features and len(numeric_cols) < expected_features:
            # Pad with zeros if we have fewer features
            pass  # Handle below

        feature_columns = numeric_cols

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    window = df[feature_columns].tail(int(sequence_length)).copy()
    if len(window) < int(sequence_length):
        raise ValueError(f"Need at least {sequence_length} rows; got {len(window)}")

    arr = window.to_numpy(dtype=np.float32, copy=True)

    # Pad or truncate features to match expected_features
    if expected_features and arr.shape[1] != expected_features:
        if arr.shape[1] > expected_features:
            arr = arr[:, :expected_features]
        else:
            # Pad with zeros
            pad_width = expected_features - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

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
    if ctx.engine is None:
        raise RuntimeError(
            "Prediction engine unavailable in this chat session. "
            "Grounded Buddy self-knowledge still works, but predictive commands need a compatible unified checkpoint."
        )

    # TensorFlow models with feature engineering need all numeric features
    if ctx.is_tensorflow and ctx.apply_feature_engineering:
        seq = _coerce_sequence(
            df,
            feature_columns=ctx.feature_columns,
            sequence_length=ctx.sequence_length,
            feature_scaler=ctx.feature_scaler,
            use_all_numeric=True,
            expected_features=128,  # TF model expects 128 features
        )
    else:
        seq = _coerce_sequence(
            df,
            feature_columns=ctx.feature_columns,
            sequence_length=ctx.sequence_length,
            feature_scaler=ctx.feature_scaler,
        )

    # Prepare input - TensorFlow takes numpy, PyTorch takes torch tensor
    if ctx.is_tensorflow:
        x = np.expand_dims(seq, axis=0)  # (1, seq, feat)
    else:
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


def _predict_one_locked(ctx: TalkContext, df: pd.DataFrame) -> Dict[str, Any]:
    """Thread-safe wrapper for _predict_one.

    TensorFlow inference can hang if called concurrently from multiple threads.
    Guard all inference through a single lock shared by the REPL + warmup.
    """
    print(_assistant_prefix(ctx) + "Acquiring inference lock...")
    with ctx.inference_lock:
        print(_assistant_prefix(ctx) + "Lock acquired. Starting prediction...")
        try:
            return _predict_one(ctx, df)
        finally:
            print(_assistant_prefix(ctx) + "Prediction done/failed. Releasing lock...")


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
    confidence = float(max(state_probs))
    state_labels = ["BEARISH", "BULLISH"] if len(state_probs) == 2 else [f"state_{i}" for i in range(len(state_probs))]
    return f"Market state: {state_labels[best]} ({confidence:.1%} confidence)"


def _format_state_answer(state_probs: Any) -> str:
    """Format answer specifically about market state."""
    if state_probs is None or not isinstance(state_probs, list) or len(state_probs) == 0:
        return "Hmm, I don't have state data right now. Try running 'predict' first."
    best = int(np.argmax(np.asarray(state_probs, dtype=float)))
    confidence = float(max(state_probs))
    state_labels = ["BEARISH", "BULLISH"] if len(state_probs) == 2 else [f"state_{i}" for i in range(len(state_probs))]
    label = state_labels[best]
    if confidence > 0.8:
        return f"I'm feeling pretty confident here - looks {label} to me ({confidence:.0%} sure)."
    elif confidence > 0.6:
        return f"Leaning {label} ({confidence:.0%}), but I'm not super certain about it."
    else:
        return f"Honestly? I'm not sure. Slight edge to {label} ({confidence:.0%}), but it's a coin flip."


def _format_trade_note(q: str, trend: Any, risk: Any) -> Optional[str]:
    ql = q.lower()
    if not any(k in ql for k in ("should", "buy", "sell", "trade")):
        return None

    if risk is not None and float(risk) > 0.7:
        return "⚠️ Careful - risk is elevated right now. Maybe size down or wait for a cleaner setup."
    if trend is not None and float(trend) > 0 and (risk is None or float(risk) < 0.6):
        return "📈 Trend looks positive with manageable risk - could favor longs here."
    if trend is not None and float(trend) < 0 and (risk is None or float(risk) < 0.6):
        return "📉 Trend is pointing down - might want to stay cautious or look for shorts."
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


def _risk_emoji(risk: float) -> str:
    """Get emoji for risk level."""
    if risk < 0.35:
        return "🟢"
    if risk < 0.7:
        return "🟡"
    return "🔴"


def _trend_emoji(delta: float) -> str:
    """Get emoji for trend/delta direction."""
    if delta > 0:
        return "📈"
    if delta < 0:
        return "📉"
    return "➡️"


def _confidence_bucket(confidence: float) -> str:
    """Categorize confidence level."""
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def _can_execute_trade(state_probs: Any, risk: Any) -> tuple[bool, str]:
    """Determine if trade should be executed based on confidence and risk.

    Rules:
    - HIGH confidence (>=75%): Always execute
    - MEDIUM confidence (55-75%): Execute only if risk is LOW (<0.35)
    - LOW confidence (<55%): Never execute

    Returns (can_execute, reason)
    """
    if state_probs is None or not isinstance(state_probs, list) or len(state_probs) == 0:
        return False, "no prediction data available"

    confidence = float(max(state_probs))
    conf_bucket = _confidence_bucket(confidence)
    risk_val = float(risk) if risk is not None else 0.5
    risk_level = _risk_bucket(risk_val)

    if conf_bucket == "high":
        return True, f"high confidence ({confidence:.0%})"
    elif conf_bucket == "medium":
        if risk_level == "low":
            return True, f"medium confidence ({confidence:.0%}) with low risk"
        else:
            return False, f"medium confidence ({confidence:.0%}) but risk is {risk_level} - waiting for better setup"
    else:
        return False, f"low confidence ({confidence:.0%}) - not trading"


def _fx_now_override(ctx: Any):
    return getattr(ctx, "fx_now", None) or getattr(ctx, "_fx_now", None)


def _fx_min_confidence(cfg: Dict[str, Any], policy: Any) -> float:
    rm = dict((cfg or {}).get("risk_management") or {})
    mc = rm.get("min_confidence")
    try:
        mc_f = float(mc) if mc is not None else float(getattr(policy.confidence, "low_lt", 0.70))
    except Exception:
        mc_f = float(getattr(policy.confidence, "low_lt", 0.70))
    try:
        low_lt = float(getattr(policy.confidence, "low_lt", mc_f))
    except Exception:
        low_lt = mc_f
    return float(max(mc_f, low_lt))


def _fx_trade_gate(ctx: "TalkContext", *, instrument: str, action: str) -> tuple[bool, str, float]:
    """Tier-1 FX gate used by Buddy trade commands.

    Keeps the trading decision consistent with:
    - `fx_guardrails` (session, entry limits)
    - `fx_paper` costs (spread)
    - `fx_confidence_v1` (confidence mapping)

    Returns: (ok, human_reason, confidence)
    """
    import fx_guardrails as fxg
    from src.utils.fx_paper import atr as fx_atr
    from src.utils.fx_paper import spread_pips_from_df
    from src.utils.reasoning_enhanced import fx_confidence_v1

    cfg = load_config(ctx.config_path)
    policy = fxg.load_fx_policy(cfg)
    state = fxg.load_state(cfg, policy, now=_fx_now_override(ctx))

    ok, why = fxg.can_open_new_trade(policy, state, now=_fx_now_override(ctx))
    if not ok:
        return False, f"Tier-1 blocked: {why}", 0.0

    if ctx.active_df is None:
        return False, "No active candles loaded; run 'use oanda ...' then 'predict' first.", 0.0

    df = ctx.active_df
    spread_pips = spread_pips_from_df(df, instrument)
    if spread_pips is None:
        spread_pips = float(getattr(policy.costs, "spread_fallback_pips", {}).get(instrument, 0.0) or 0.0)

    max_spread = float(getattr(policy.costs, "max_spread_pips", {}).get(instrument, 0.0) or 0.0)
    if max_spread > 0 and float(spread_pips) > max_spread:
        return False, f"Tier-1 blocked: spread too wide ({float(spread_pips):.2f}p > {max_spread:.2f}p)", 0.0

    price = float(df["close"].iloc[-1])
    atr_value = float(fx_atr(df, period=14))

    params = dict((cfg.get("fx", {}) or {}).get("confidence_model") or {})
    msp = max_spread if max_spread > 0 else max(1e-9, float(spread_pips))

    payload = fx_confidence_v1(
        signal=str(action),
        price=float(price),
        atr=float(atr_value),
        spread_pips=float(spread_pips),
        max_spread_pips=float(msp),
        params=params,
    )
    conf = float(payload.get("confidence") or 0.0)
    reasons = ", ".join([str(x) for x in (payload.get("reasons") or [])])

    threshold = _fx_min_confidence(cfg, policy)
    if conf < threshold:
        return False, f"confidence {conf:.2f} < {threshold:.2f} ({reasons})", float(conf)

    band = fxg.confidence_band(policy, conf)
    return True, f"confidence {conf:.2f} band={band} ({reasons})", float(conf)


def _summary_line(pred: Any, last_close: Any) -> Optional[str]:
    if pred is None or last_close is None:
        return None
    try:
        pred_f = float(pred)
        last_f = float(last_close)
        delta = pred_f - last_f
        pct = (delta / last_f) if last_f != 0 else 0.0
        direction = _direction_label(delta)
        emoji = _trend_emoji(delta)
        return f"{emoji} I'm leaning {direction}. Next close estimate: {pred_f:.5f} (currently {last_f:.5f}, that's {pct:+.2%})."
    except Exception:
        return None


def _render_risk_answer(risk: Any) -> str:
    """Render answer for risk-related questions."""
    if risk is None:
        return "I don't have risk data at the moment. Run 'predict' first?"
    r = float(risk)
    bucket = _risk_bucket(r)
    if bucket == "high":
        return f"🔴 Risk is HIGH ({r:.2f}). I'd be careful here - maybe smaller size or just wait."
    if bucket == "medium":
        return f"🟡 Risk is moderate ({r:.2f}). Not terrible, but stay alert."
    return f"🟢 Risk looks low ({r:.2f}). Things are relatively calm right now."


def _render_trend_answer(trend: Any, state_probs: Any) -> str:
    """Render answer for trend-related questions."""
    lines: list[str] = []
    if trend is not None:
        t = float(trend)
        if t > 0.01:
            lines.append(f"📈 Trend is pointing UP ({t:+.4f}).")
        elif t < -0.01:
            lines.append(f"📉 Trend is pointing DOWN ({t:+.4f}).")
        else:
            lines.append(f"➡️ Trend is pretty FLAT right now ({t:+.4f}).")
    if state_probs:
        lines.append(_format_state_answer(state_probs))
    return "\n".join(lines) if lines else "No trend data yet. Try 'predict' first."


def _render_default_answer(result: Dict[str, Any], q: str) -> str:
    """Render default full summary answer."""
    pred = result.get("prediction")
    last_close = result.get("last_close")
    risk = result.get("risk")
    trend = result.get("trend")
    state_probs = result.get("state_probs")

    lines: list[str] = []

    state_line = _format_state_probs_line(state_probs)
    if state_line:
        lines.append(state_line)

    summary = _summary_line(pred, last_close)
    if summary is not None:
        lines.append(summary)
    else:
        lines.extend(_format_prediction_lines(pred, last_close))

    if risk is not None:
        try:
            r = float(risk)
            lines.append(f"Risk looks {_risk_bucket(r)}.")
        except Exception:
            pass

    trade_note = _format_trade_note(q, trend, risk)
    if trade_note:
        lines.append(trade_note)

    if not lines:
        return "I ran the model but didn't get a usable prediction."

    return "\n".join(lines)


def _render_answer(q: str, result: Dict[str, Any]) -> str:
    ql = q.lower()

    # Handle specific questions
    if any(k in ql for k in ("risk", "risky", "dangerous", "safe")):
        return _render_risk_answer(result.get("risk"))

    if any(k in ql for k in ("trend", "direction", "going")):
        return _render_trend_answer(result.get("trend"), result.get("state_probs"))

    if any(k in ql for k in ("state", "bullish", "bearish", "sentiment")):
        return _format_state_answer(result.get("state_probs"))

    return _render_default_answer(result, q)


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


def _compact_float(value: Any, digits: int = 4) -> Optional[str]:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return None


def _prediction_fact_lines(ctx: TalkContext, result: Dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"active_source={ctx.active_source or 'unknown'}")
    lines.append(f"instrument={ctx.oanda_instrument}")
    lines.append(f"granularity={ctx.oanda_granularity}")

    pred = _compact_float(result.get("prediction"), 5)
    last_close = _compact_float(result.get("last_close"), 5)
    if pred is not None:
        lines.append(f"prediction={pred}")
    if last_close is not None:
        lines.append(f"last_close={last_close}")
    if pred is not None and last_close is not None:
        try:
            delta = float(result.get("prediction")) - float(result.get("last_close"))
            lines.append(f"delta={delta:+.5f}")
        except Exception:
            pass

    risk = result.get("risk")
    if risk is not None:
        try:
            risk_f = float(risk)
            lines.append(f"risk={risk_f:.2f} ({_risk_bucket(risk_f)})")
        except Exception:
            pass

    trend = result.get("trend")
    if trend is not None:
        try:
            lines.append(f"trend={float(trend):+.4f}")
        except Exception:
            pass

    state_probs = result.get("state_probs")
    if isinstance(state_probs, list) and state_probs:
        try:
            conf = float(max(state_probs))
            best_idx = int(np.argmax(state_probs))
            lines.append(f"confidence={conf:.0%} ({_confidence_bucket(conf)})")
            lines.append(f"dominant_state_index={best_idx}")
            lines.append(f"state_probs={state_probs}")
        except Exception:
            pass

    can_trade, reason = _can_execute_trade(result.get("state_probs"), result.get("risk"))
    lines.append(f"trade_ready={can_trade}")
    lines.append(f"trade_gate_reason={reason}")
    lines.append(f"execute_mode={'live' if ctx.oanda_execute else 'dry-run'}")
    lines.append(f"stop_loss_pips={ctx.stop_loss_pips if ctx.use_stop_loss else 'off'}")
    lines.append(f"take_profit_pips={ctx.take_profit_pips if ctx.use_take_profit else 'off'}")
    return lines


def _grounded_chat_reply(
    ctx: TalkContext,
    *,
    user_query: str,
    task: str,
    fact_lines: list[str],
    fallback: str,
    max_tokens: int = 320,
) -> str:
    """Synthesize a natural reply from grounded facts, local-first."""
    facts = [line for line in fact_lines if line]
    if not facts:
        return fallback

    try:
        from llm_providers import llm_call, select_buddy_provider_name

        provider = select_buddy_provider_name(None)
        if not provider:
            return fallback

        system_prompt = (
            "You are Buddy, a trading assistant. "
            "Answer ONLY from the provided grounded facts and recent chat context. "
            "Do not invent state, trades, config, or runtime details. "
            "If the facts are insufficient, say exactly what is missing. "
            "Keep the answer concise and natural."
        )
        prompt = (
            f"Task: {task}\n"
            f"User query: {user_query}\n\n"
            f"Recent chat context:\n{_recent_chat_context(ctx)}\n\n"
            f"Grounded facts:\n- " + "\n- ".join(facts) + "\n\n"
            "Write the answer as Buddy speaking directly to the user."
        )
        response = llm_call(
            prompt,
            system_prompt=system_prompt,
            provider=provider,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        if response and response.strip():
            return response.strip()
    except Exception as e:
        logger.debug("Grounded chat synthesis failed: %s", e)

    return fallback


def _render_prediction_chat_answer(ctx: TalkContext, q: str, result: Dict[str, Any]) -> str:
    fallback = _render_answer_verbose(q, result) if ctx.verbose else _render_answer(q, result)
    return _grounded_chat_reply(
        ctx,
        user_query=q,
        task="Answer the user's market question using the latest prediction context.",
        fact_lines=_prediction_fact_lines(ctx, result),
        fallback=fallback,
    )


def _find_pytorch_checkpoint(cfg: Dict[str, Any]) -> Optional[str]:
    """Find a PyTorch checkpoint as fallback."""
    # Check common locations
    paths_to_try = [
        "trained_data/models/unified_market_net.pth",
        cfg.get("paths", {}).get("checkpoint"),
        cfg.get("data", {}).get("checkpoint_path"),
    ]

    for path in paths_to_try:
        if path and Path(path).exists():
            return str(path)

    return None


def _load_tensorflow_model(cfg: Dict[str, Any], ckpt_path: str) -> tuple[TensorFlowUnifiedEngine, dict]:
    """Load TensorFlow model and return engine and metadata."""
    print("🔷 Loading TensorFlow TFT model...")
    engine = TensorFlowUnifiedEngine.from_checkpoint(ckpt_path, device=cfg.get("device"))

    feature_columns = cfg.get("data", {}).get("feature_columns") or ["open", "high", "low", "close", "volume"]
    sequence_length = int(cfg.get("data", {}).get("sequence_length") or cfg.get("sequence_length") or 60)
    target_shift = int(cfg.get("data", {}).get("target_shift") or cfg.get("target_shift") or 1)

    return engine, {
        "feature_columns": feature_columns,
        "sequence_length": sequence_length,
        "target_shift": target_shift,
        "feature_scaler": None,
        "target_scaler": None,
        "apply_feature_engineering": True,
    }


def _load_pytorch_model(cfg: Dict[str, Any], ckpt_path: str) -> tuple[UnifiedNeuralEngine, dict]:
    """Load PyTorch model and return engine and metadata."""
    print("🔶 Loading PyTorch model...")
    engine = UnifiedNeuralEngine.from_checkpoint(ckpt_path, device=cfg.get("device"))
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    data_meta = dict(ckpt.get("data_meta") or {})

    feature_columns = list(data_meta.get("feature_columns") or ["open", "high", "low", "close", "volume"])
    sequence_length = int(data_meta.get("sequence_length") or cfg.get("data", {}).get("sequence_length") or cfg.get("sequence_length") or 60)
    target_shift = int(data_meta.get("target_shift") or cfg.get("data", {}).get("target_shift") or cfg.get("target_shift") or 1)

    base_ohlcv_count = 8
    apply_feature_engineering = len(feature_columns) > base_ohlcv_count

    return engine, {
        "feature_columns": feature_columns,
        "sequence_length": sequence_length,
        "target_shift": target_shift,
        "feature_scaler": data_meta.get("feature_scaler"),
        "target_scaler": data_meta.get("target_scaler"),
        "apply_feature_engineering": apply_feature_engineering,
    }


def _load_context(config_path: str, *, checkpoint_path: Optional[str] = None) -> TalkContext:
    cfg = load_config(config_path)
    ckpt_path = checkpoint_path or _default_checkpoint_path(cfg)
    reasoning = ReasoningEngine(cfg.get("reasoning", {}))

    # Initialize confidence system components
    from confidence_calibration import create_default_calibrator
    from position_sizing import create_default_position_sizer
    from risk_management import create_default_risk_manager

    confidence_calibrator = create_default_calibrator()
    position_sizer = create_default_position_sizer()
    risk_manager = create_default_risk_manager()

    if not Path(ckpt_path).exists():
        logger.info(
            "Unified talk checkpoint not found at %s; starting Buddy chat in grounded knowledge-only mode",
            ckpt_path,
        )
        return TalkContext(
            config_path=config_path,
            checkpoint_path=ckpt_path,
            feature_columns=["open", "high", "low", "close", "volume"],
            sequence_length=int(cfg.get("data", {}).get("sequence_length") or cfg.get("sequence_length") or 60),
            target_shift=int(cfg.get("data", {}).get("target_shift") or cfg.get("target_shift") or 1),
            engine=None,  # type: ignore[arg-type]
            reasoning=reasoning,
            feature_scaler=None,
            target_scaler=None,
            apply_feature_engineering=False,
            is_tensorflow=False,
            confidence_calibrator=confidence_calibrator,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
        )

    is_tf_model = is_tensorflow_checkpoint(ckpt_path)

    if is_tf_model:
        try:
            engine, meta = _load_tensorflow_model(cfg, ckpt_path)
        except Exception as e:
            print(f"⚠ TensorFlow model load failed: {e}")
            pytorch_ckpt = _find_pytorch_checkpoint(cfg)
            if pytorch_ckpt:
                print(f"🔶 Falling back to PyTorch model: {pytorch_ckpt}")
                ckpt_path = pytorch_ckpt
                is_tf_model = False
                engine, meta = _load_pytorch_model(cfg, ckpt_path)
            else:
                raise RuntimeError(
                    "TensorFlow model failed to load and no PyTorch fallback found.\n"
                    "Retrain with: python3 train_visual.py --framework tensorflow --model tft --multi-task --data-dir trained_data/data"
                )
    else:
        engine, meta = _load_pytorch_model(cfg, ckpt_path)

    return TalkContext(
        config_path=config_path,
        checkpoint_path=ckpt_path,
        feature_columns=meta["feature_columns"],
        sequence_length=meta["sequence_length"],
        target_shift=meta["target_shift"],
        engine=engine,
        reasoning=reasoning,
        feature_scaler=meta["feature_scaler"],
        target_scaler=meta["target_scaler"],
        apply_feature_engineering=meta["apply_feature_engineering"],
        is_tensorflow=is_tf_model,
        confidence_calibrator=confidence_calibrator,
        position_sizer=position_sizer,
        risk_manager=risk_manager,
    )


def _print_talk_banner(ctx: TalkContext) -> None:
    import sys
    if ctx.verbose:
        model_type = "Knowledge-Only" if ctx.engine is None else ("TensorFlow TFT" if ctx.is_tensorflow else "PyTorch LSTM")
        print(f"🤖 Buddy - {model_type} Trading Engine")
        print(f"Model: {ctx.checkpoint_path}")
        print(
            f"Features: {len(ctx.feature_columns)} | seq_len={ctx.sequence_length} | shift={ctx.target_shift}"
        )
        print(
            "Commands: 'predict' | 'summary' | 'trade' | 'help' | 'exit'"
        )
        print(
            "Tip: I only execute trades on HIGH confidence, or MEDIUM + LOW risk."
        )
        sys.stdout.flush()
        return

    exec_status = EXEC_STATUS_LIVE if ctx.oanda_execute else EXEC_STATUS_DRY_RUN
    sl_info = f"SL={ctx.stop_loss_pips}p" if ctx.use_stop_loss else "SL=OFF"
    tp_info = f"TP={ctx.take_profit_pips}p" if ctx.use_take_profit else "TP=OFF"
    if _planner_chat_enabled():
        label = Text("Buddy Planner 3B", style="bold color(215)")
        prompt = Text("Prompt: scan-first grounded operator", style="color(117)")
        status = Text()
        status.append("Mode: planner mode", style="color(223)")
        status.append(f" | {sl_info} | {tp_info}. ", style="default")
        status.append("Type 'help' for planner examples.", style="color(246)")
        _PLANNER_CONSOLE.print(label, " ", prompt)
        _PLANNER_CONSOLE.print(Text("Buddy Planner 3B: ", style="bold color(215)"), status)
    else:
        readiness = "knowledge-first mode" if ctx.engine is None else exec_status
        print(_assistant_prefix(ctx) + f"Hey! I'm ready to help. Mode: {readiness} | {sl_info} | {tp_info}. Type 'help' for commands.")
    sys.stdout.flush()


def _run_initial_csv_prediction(ctx: TalkContext, csv_path: str) -> None:
    """Load CSV and run initial prediction."""
    df = _load_df_from_csv(csv_path)
    df = _apply_feature_engineering_if_needed(ctx, df)
    ctx.active_source = f"csv:{csv_path}"
    ctx.active_df = df
    result = _predict_one_locked(ctx, df)
    ctx.last_result = result
    ans = _render_answer_verbose("predict", result) if ctx.verbose else _render_answer("predict", result)
    print(_assistant_prefix(ctx) + ans)


def _run_initial_ticker_prediction(ctx: TalkContext, ticker: str, period: str, interval: str) -> None:
    """Load ticker data and run initial prediction."""
    df = _load_df_from_ticker(ticker, period=period, interval=interval)
    df = _apply_feature_engineering_if_needed(ctx, df)
    ctx.active_source = f"ticker:{ticker}"
    ctx.active_df = df
    result = _predict_one_locked(ctx, df)
    ctx.last_result = result
    ans = _render_answer_verbose("predict", result) if ctx.verbose else _render_answer("predict", result)
    print(_assistant_prefix(ctx) + ans)


def _oanda_warmup_worker(ctx: TalkContext) -> None:
    """Background worker to warm up OANDA connection."""
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
        if ctx.verbose:
            print(_assistant_prefix(ctx) + "OANDA warmup complete. Type 'predict' to run inference.")
    except Exception as e:
        if ctx.verbose:
            print(
                _assistant_prefix(ctx)
                + f"OANDA not ready ({e}). You can still use 'use csv ...' or 'use ticker ...'."
            )


def _start_oanda_warmup(ctx: TalkContext) -> None:
    """Start OANDA warmup in background thread."""
    try:
        t = threading.Thread(target=_oanda_warmup_worker, args=(ctx,), name="buddy-oanda-warmup", daemon=True)
        t.start()
    except Exception as e:
        if ctx.verbose:
            print(_assistant_prefix(ctx) + f"OANDA warmup thread failed ({e}).")


def _maybe_run_initial_prediction(
    ctx: TalkContext,
    *,
    csv_path: Optional[str],
    ticker: Optional[str],
    period: str,
    interval: str,
    oanda: bool,
) -> None:
    """Run initial prediction based on provided data source."""
    if ctx.engine is None:
        return

    if csv_path:
        _run_initial_csv_prediction(ctx, csv_path)
        return

    if ticker:
        _run_initial_ticker_prediction(ctx, ticker, period, interval)
        return

    if oanda:
        _start_oanda_warmup(ctx)


def _extract_command_arg(q: str, prefix: str) -> Optional[str]:
    ql = q.lower()
    if not ql.startswith(prefix):
        return None
    parts = q.split(" ", 2)
    if len(parts) < 3:
        return None
    return parts[2].strip()


_FX_CURRENCY_CODES = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"}


def _is_explicit_talk_command(ql: str) -> bool:
    """Return True for direct command forms that should bypass intent planning."""
    if ql in {"exit", "quit", "q", "help", "?"}:
        return True
    return False


def _extract_instrument_hint(q: str) -> Optional[str]:
    """Extract a likely FX instrument from natural-language text."""
    for match in re.finditer(r"\b([A-Za-z]{3})[\s/_-]?([A-Za-z]{3})\b", q):
        base = match.group(1).upper()
        quote = match.group(2).upper()
        if base in _FX_CURRENCY_CODES and quote in _FX_CURRENCY_CODES:
            return f"{base}_{quote}"
    return None


def _compute_prediction_for_active_source(ctx: TalkContext) -> Dict[str, Any]:
    """Run a fresh prediction for the active source without printing."""
    if ctx.active_df is None:
        raise ValueError("pick a source first: 'use oanda ...' or 'use csv ...' or 'use ticker ...'.")

    if ctx.active_source and ctx.active_source.startswith("oanda:"):
        ctx.active_df = _load_df_from_oanda(
            ctx,
            instrument=ctx.oanda_instrument,
            granularity=ctx.oanda_granularity,
            candles=ctx.oanda_candles,
            price=ctx.oanda_price,
        )

    result = _predict_one_locked(ctx, ctx.active_df)
    ctx.last_result = result
    return result


def _plan_position(q: str, patterns: tuple[str, ...]) -> Optional[int]:
    """Return the earliest position for any regex pattern in text."""
    hits = []
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            hits.append(match.start())
    return min(hits) if hits else None


def _extract_sl_tp_steps(q: str) -> list[tuple[int, TalkPlannedStep]]:
    """Extract natural-language SL/TP changes in source order."""
    steps: list[tuple[int, TalkPlannedStep]] = []

    for match in re.finditer(r"\b(?:set\s+)?(?:stop\s*loss|stoploss|sl)\s+(?:to\s+)?(\d+(?:\.\d+)?)\b", q, flags=re.IGNORECASE):
        steps.append((match.start(), TalkPlannedStep(action="set_sl", value=match.group(1))))
    for match in re.finditer(r"\b(?:set\s+)?(?:take\s*profit|takeprofit|tp)\s+(?:to\s+)?(\d+(?:\.\d+)?)\b", q, flags=re.IGNORECASE):
        steps.append((match.start(), TalkPlannedStep(action="set_tp", value=match.group(1))))

    return steps


def _extract_execute_mode_steps(q: str) -> list[tuple[int, TalkPlannedStep]]:
    steps: list[tuple[int, TalkPlannedStep]] = []
    on_patterns = (
        r"\bexecute\s+on\b",
        r"\bturn\s+(?:execution|execute)\s+on\b",
        r"\benable\s+(?:execution|live trading|live mode)\b",
        r"\bgo\s+live\b",
    )
    off_patterns = (
        r"\bexecute\s+off\b",
        r"\bturn\s+(?:execution|execute)\s+off\b",
        r"\bdisable\s+(?:execution|live trading|live mode)\b",
        r"\bdry[- ]run\s+only\b",
    )

    for pattern in on_patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            steps.append((match.start(), TalkPlannedStep(action="set_execute_mode", value="on")))
            break
    for pattern in off_patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            steps.append((match.start(), TalkPlannedStep(action="set_execute_mode", value="off")))
            break

    return steps


def _plan_talk_request(ctx: TalkContext, q: str, ql: str) -> Optional[TalkIntentPlan]:
    """Map natural-language requests onto existing talk tools."""
    if _is_explicit_talk_command(ql):
        return None

    plan = TalkIntentPlan()
    instrument = _extract_instrument_hint(q)
    instrument_pos = _plan_position(q, (r"\b[A-Za-z]{3}[\s/_-]?[A-Za-z]{3}\b",))
    wants_analysis = any(
        phrase in ql for phrase in (
            "analy",
            "check",
            "look at",
            "what do you think",
            "setup",
            "safe to",
            "should i",
            "how risky",
            "good trade",
        )
    )
    wants_summary = any(
        phrase in ql for phrase in (
            "summary",
            "summarize",
            "overview",
            "what am i looking at",
        )
    )
    wants_execution = any(
        phrase in ql for phrase in (
            "take the trade",
            "place the trade",
            "execute the trade",
            "enter the trade",
            "open the trade",
            "go ahead and trade",
            "put on the trade",
        )
    )

    directional_hint = None
    has_buy = bool(re.search(r"\b(buy|long)\b", ql))
    has_sell = bool(re.search(r"\b(sell|short)\b", ql))
    if has_buy ^ has_sell:
        directional_hint = "buy" if has_buy else "sell"

    ordered_steps: list[tuple[int, TalkPlannedStep]] = []
    if instrument and instrument != ctx.oanda_instrument:
        ordered_steps.append((instrument_pos or 0, TalkPlannedStep(action="use_oanda", value=instrument)))

    ordered_steps.extend(_extract_sl_tp_steps(q))
    ordered_steps.extend(_extract_execute_mode_steps(q))

    summary_pos = _plan_position(q, (r"\bsummar(?:y|ize)\b", r"\boverview\b"))
    if wants_summary:
        ordered_steps.append((summary_pos or len(q), TalkPlannedStep(action="summary")))

    execution_pos = _plan_position(
        q,
        (
            r"take the trade",
            r"place the trade",
            r"execute the trade",
            r"enter the trade",
            r"open the trade",
            r"go ahead and trade",
            r"put on the trade",
        ),
    )
    if wants_execution:
        ordered_steps.append((
            execution_pos or len(q),
            TalkPlannedStep(action="trade_market" if directional_hint else "trade_auto", value=directional_hint),
        ))

    needs_prediction = wants_analysis or wants_execution or wants_summary
    if needs_prediction:
        first_action_pos = min(
            [pos for pos, step in ordered_steps if step.action not in {"use_oanda", "set_sl", "set_tp"}],
            default=len(q),
        )
        ordered_steps.append((max(0, first_action_pos - 1), TalkPlannedStep(action="predict")))

    if wants_analysis and not wants_summary and not wants_execution:
        plan.answer_after = True

    if not ordered_steps and not plan.answer_after:
        return None

    ordered_steps.sort(key=lambda item: item[0])
    seen_predict = False
    for _, step in ordered_steps:
        if step.action == "predict":
            if seen_predict:
                continue
            seen_predict = True
        plan.steps.append(step)

    return plan


def _execute_talk_plan(
    ctx: TalkContext,
    plan: TalkIntentPlan,
    *,
    q: str,
    period: str,
    interval: str,
) -> bool:
    """Execute a planned natural-language request."""
    for step in plan.steps:
        if step.action == "use_oanda" and step.value:
            _handle_use_commands(
                ctx,
                f"use oanda {step.value} {ctx.oanda_granularity} {ctx.oanda_candles}",
                period=period,
                interval=interval,
            )
        elif step.action == "predict":
            _compute_prediction_for_active_source(ctx)
        elif step.action == "summary":
            _handle_summary_command(ctx)
        elif step.action == "set_sl" and step.value:
            _handle_sl_command(ctx, f"sl {step.value}")
        elif step.action == "set_tp" and step.value:
            _handle_tp_command(ctx, f"tp {step.value}")
        elif step.action == "set_execute_mode" and step.value:
            if step.value == "on":
                _handle_basic_commands(ctx, "execute on")
            else:
                _handle_basic_commands(ctx, "execute off")
        elif step.action == "trade_auto":
            _ensure_oanda_client(ctx)
            _handle_trade_auto(ctx, ["trade"])
        elif step.action == "trade_market":
            _ensure_oanda_client(ctx)
            _handle_trade_market(ctx, action=step.value or "buy", parts=[step.value or "buy"])

    if plan.answer_after:
        ans = _render_prediction_chat_answer(ctx, q, ctx.last_result) if ctx.last_result is not None else (
            "I completed the requested steps, but I don't have a fresh prediction result to summarize yet."
        )
        print(_assistant_prefix(ctx) + ans)
        _remember_chat_turn(ctx, "assistant", ans)

    return True


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


def _handle_help_command(ctx: TalkContext) -> None:
    """Handle help command."""
    if _planner_chat_enabled():
        print(
            f"{_assistant_prefix(ctx)}Here's how to talk to me in planner mode:\n\n"
            f"  • 'look for an opportunity' - scan the market for current setups\n"
            f"  • 'scan M5' / 'scan H1 top 3' - run an explicit scan\n"
            f"  • 'ignore session filter' / 'disable session filter for this scan' - bypass session guardrail\n"
            f"  • 'restore session filter' / 'enable session filter' - re-enable session guardrails\n"
            f"  • 'set session window 8 to 21 utc' - temporarily change trading window\n"
            f"  • 'set session window 8-21' (this session) - same as above\n"
            f"  • 'show status' / 'show model status' - inspect runtime and models\n"
            f"  • 'compare EUR/USD and GBP/USD' - compare the last scan context\n"
            f"  • 'journal summary' / 'account status' - grounded account and journal actions\n"
            f"  • 'sl 25' / 'tp 50' / 'execute on' - change runtime settings\n\n"
            f"I default to taking the next grounded action. If you ask for a setup or opportunity, I'll scan rather than bounce you into a menu."
        )
        return

    sl_status = f"{ctx.stop_loss_pips}p" if ctx.use_stop_loss else "OFF"
    tp_status = f"{ctx.take_profit_pips}p" if ctx.use_take_profit else "OFF"
    exec_status = EXEC_STATUS_LIVE if ctx.oanda_execute else EXEC_STATUS_DRY_RUN
    print(
        f"{_assistant_prefix(ctx)}📋 Here's what I can do:\n\n"
        f"📊 Analysis:\n"
        f"  • 'predict' - run fresh prediction\n"
        f"  • 'summary' - overview with trade readiness\n"
        f"  • 'what's the risk?' / 'trend?' / 'bullish or bearish?'\n\n"
        f"📈 Data Sources:\n"
        f"  • 'use oanda <INSTR> <GRAN> <COUNT>' - OANDA data\n"
        f"  • 'use csv <path>' / 'use ticker <SYM>'\n\n"
        f"💰 Trading (confidence-gated):\n"
        f"  • 'trade' - auto trade based on prediction\n"
        f"  • 'buy' / 'sell' - manual direction\n"
        f"  • 'trade close' - close position\n"
        f"  • 'force buy' / 'force sell' - override confidence check\n"
        f"  • 'execute on/off' - toggle live orders\n\n"
        f"🛡️ Risk Management:\n"
        f"  • 'sl <pips>' - set stop loss (e.g., 'sl 25')\n"
        f"  • 'tp <pips>' - set take profit (e.g., 'tp 50')\n"
        f"  • 'sl off' / 'tp off' - disable SL/TP\n"
        f"  • 'risk' - view current risk settings\n"
        f"  Current: SL={sl_status} | TP={tp_status} | {exec_status}\n\n"
        f"🎯 Trade Rules:\n"
        f"  • HIGH confidence (≥75%): ✅ Execute\n"
        f"  • MEDIUM (55-75%) + LOW risk: ✅ Execute\n"
        f"  • Otherwise: ⛔ Won't execute"
    )


def _handle_summary_command(ctx: TalkContext) -> None:
    """Handle summary/status/overview command."""
    if ctx.last_result is None:
        print(_assistant_prefix(ctx) + "I don't have a prediction yet. Run 'predict' and I'll give you the lowdown.")
        return

    result = ctx.last_result
    state_probs = result.get("state_probs")
    risk = result.get("risk")
    trend = result.get("trend")

    lines = [f"📊 Here's what I'm seeing for {ctx.active_source or 'current data'}:"]

    # State with confidence level
    if state_probs:
        best = int(np.argmax(state_probs))
        conf = float(max(state_probs))
        labels = ["🔴 BEARISH", "🟢 BULLISH"] if len(state_probs) == 2 else [f"State {i}" for i in range(len(state_probs))]
        conf_label = _confidence_bucket(conf).upper()
        lines.append(f"  State: {labels[best]} ({conf:.0%} - {conf_label} confidence)")

    # Risk
    if risk is not None:
        r = float(risk)
        emoji = _risk_emoji(r)
        lines.append(f"  Risk: {emoji} {_risk_bucket(r)} ({r:.2f})")

    # Trend
    if trend is not None:
        t = float(trend)
        emoji = _trend_emoji(t)
        lines.append(f"  Trend: {emoji} {t:+.4f}")

    # Trade readiness
    can_trade, reason = _can_execute_trade(state_probs, risk)
    trade_emoji = "✅" if can_trade else "⏸️"
    lines.append(f"  Trade ready: {trade_emoji} {reason}")

    answer = "\n".join(lines)
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)


def _handle_sl_command(ctx: TalkContext, ql: str) -> Optional[bool]:
    """Handle stop loss commands."""
    if not (ql.startswith(("sl ", "stoploss ", "stop loss "))):
        return None

    parts = ql.split()
    if len(parts) < 2:
        return None

    try:
        pips = float(parts[-1])
        ctx.stop_loss_pips = pips
        ctx.use_stop_loss = True
        print(_assistant_prefix(ctx) + f"🛑 Stop Loss set to {pips} pips")
        return True
    except ValueError:
        last_part = parts[-1].lower()
        if last_part in {"off", "false", "disable", "0"}:
            ctx.use_stop_loss = False
            print(_assistant_prefix(ctx) + "🛑 Stop Loss disabled")
            return True
        if last_part in {"on", "true", "enable"}:
            ctx.use_stop_loss = True
            print(_assistant_prefix(ctx) + f"🛑 Stop Loss enabled ({ctx.stop_loss_pips} pips)")
            return True
    return None


def _handle_tp_command(ctx: TalkContext, ql: str) -> Optional[bool]:
    """Handle take profit commands."""
    if not (ql.startswith(("tp ", "takeprofit ", "take profit "))):
        return None

    parts = ql.split()
    if len(parts) < 2:
        return None

    try:
        pips = float(parts[-1])
        ctx.take_profit_pips = pips
        ctx.use_take_profit = True
        print(_assistant_prefix(ctx) + f"🎯 Take Profit set to {pips} pips")
        return True
    except ValueError:
        last_part = parts[-1].lower()
        if last_part in {"off", "false", "disable", "0"}:
            ctx.use_take_profit = False
            print(_assistant_prefix(ctx) + "🎯 Take Profit disabled")
            return True
        if last_part in {"on", "true", "enable"}:
            ctx.use_take_profit = True
            print(_assistant_prefix(ctx) + f"🎯 Take Profit enabled ({ctx.take_profit_pips} pips)")
            return True
    return None


def _handle_risk_settings_command(ctx: TalkContext) -> None:
    """Handle risk settings display command."""
    sl_status = f"ON ({ctx.stop_loss_pips} pips)" if ctx.use_stop_loss else "OFF"
    tp_status = f"ON ({ctx.take_profit_pips} pips)" if ctx.use_take_profit else "OFF"
    exec_status = EXEC_STATUS_LIVE if ctx.oanda_execute else EXEC_STATUS_DRY_RUN
    answer = (
        "📊 Risk Management Settings:\n"
        + f"  Execute Mode: {exec_status}\n"
        + f"  Stop Loss: {sl_status}\n"
        + f"  Take Profit: {tp_status}\n"
        + f"  Risk/Reward Ratio: 1:{ctx.take_profit_pips/ctx.stop_loss_pips:.1f}"
    )
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)


def _handle_basic_commands(ctx: TalkContext, ql: str) -> Optional[bool]:
    if ql in {"exit", "quit", "q"}:
        print("Bye.")
        return False

    if ql in {"help", "?"}:
        _handle_help_command(ctx)
        return True

    if ql in {"summary", "overview"}:
        _handle_summary_command(ctx)
        return True

    if ql in {"execute on", "execute true"}:
        ctx.oanda_execute = True
        print(_assistant_prefix(ctx) + "🟢 Execute mode ON - I'll send real orders to your practice account.")
        return True
    if ql in {"execute off", "execute false"}:
        ctx.oanda_execute = False
        print(_assistant_prefix(ctx) + "⏸️ Execute mode OFF - dry-run only, no orders will be sent.")
        return True

    # Stop Loss and Take Profit commands
    sl_result = _handle_sl_command(ctx, ql)
    if sl_result is not None:
        return sl_result

    tp_result = _handle_tp_command(ctx, ql)
    if tp_result is not None:
        return tp_result

    if ql in {"risk", "risk settings", "sl/tp", "sltp"}:
        _handle_risk_settings_command(ctx)
        return True
    return None


def _looks_like_buddy_self_query(ql: str) -> bool:
    """Detect questions about Buddy's own architecture, models, or memory."""
    phrases = (
        "who are you",
        "what are you",
        "how do you work",
        "how are you wired",
        "what model are you using",
        "what models are you using",
        "what llm",
        "what provider",
        "what backend",
        "how do you remember",
        "what do you know about yourself",
        "what is intelligent mode",
        "how does intelligent mode work",
        "how do you learn",
        "what is in memory",
        "what's in memory",
        "how does buddy work",
        "explain your architecture",
    )
    return any(phrase in ql for phrase in phrases)


def _handle_buddy_self_query(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    """Answer Buddy self-questions from grounded workspace/runtime context."""
    if not _looks_like_buddy_self_query(ql):
        return None

    try:
        answer = answer_buddy_question(
            q,
            config_path=ctx.config_path,
            use_llm=True,
        )
    except Exception as e:
        logger.debug("Buddy self-knowledge lookup failed: %s", e)
        answer = "I couldn't load my grounded self-knowledge cleanly from this workspace."

    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)
    return True


def _looks_like_smalltalk(ql: str) -> bool:
    return ql in {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "buddy",
        "buddy?",
        "are you there",
        "still there",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
    }


def _handle_knowledge_only_chat(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    """Handle casual chat safely when no predictive engine is available."""
    if ctx.engine is not None:
        return None
    if not _looks_like_smalltalk(ql):
        return None

    facts = [
        "prediction_engine_available=False",
        f"active_instrument={ctx.oanda_instrument}",
        f"granularity={ctx.oanda_granularity}",
        f"execute_mode={'live' if ctx.oanda_execute else 'dry-run'}",
        f"stop_loss_pips={ctx.stop_loss_pips if ctx.use_stop_loss else 'off'}",
        f"take_profit_pips={ctx.take_profit_pips if ctx.use_take_profit else 'off'}",
        "available_actions=grounded self-knowledge, status, model status, journal, scan, account, settings",
    ]
    fallback = (
        "I'm here in knowledge-first mode. "
        "I can answer grounded questions about Buddy, show status or decisions, run scans, and manage journal/account tasks. "
        "I just won't fabricate a market prediction without a compatible prediction checkpoint."
    )
    answer = _grounded_chat_reply(
        ctx,
        user_query=q,
        task="Reply naturally to a casual greeting while explaining the current knowledge-first chat mode.",
        fact_lines=facts,
        fallback=fallback,
    )
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)
    return True


def _format_runtime_decision(prefix: str, payload: Any) -> Optional[str]:
    if not isinstance(payload, dict) or not payload:
        return None

    decision_type = payload.get("decision_type") or "unknown"
    instrument = payload.get("instrument") or "unknown"
    direction = payload.get("direction")
    reason = payload.get("reason") or "no stored reason"
    parts = [f"{prefix}: {decision_type}", f"instrument={instrument}"]
    if direction:
        parts.append(f"direction={direction}")
    parts.append(f"reason={reason}")
    return " | ".join(parts)


def _status_fact_lines(ctx: TalkContext) -> list[str]:
    from memory_client import MLEngineMemory
    from llm_providers import select_buddy_provider_name

    cfg = load_config(ctx.config_path)
    memory = MLEngineMemory()
    runtime_state = memory.get_runtime_state("buddy_runtime")
    configured_risk = (cfg.get("fx", {}) or {}).get("risk", {}).get("risk_per_trade_pct")

    lines = [
        f"active_source={ctx.active_source or 'none'}",
        f"active_instrument={ctx.oanda_instrument}",
        f"granularity={ctx.oanda_granularity}",
        f"execute_mode={'live' if ctx.oanda_execute else 'dry-run'}",
        f"stop_loss_pips={ctx.stop_loss_pips if ctx.use_stop_loss else 'off'}",
        f"take_profit_pips={ctx.take_profit_pips if ctx.use_take_profit else 'off'}",
        f"configured_risk_per_trade_pct={configured_risk}",
        f"selected_provider={select_buddy_provider_name(None) or 'none'}",
    ]

    if isinstance(runtime_state, dict):
        trade_decision = _format_runtime_decision("last_trade_execution_decision", runtime_state.get("last_trade_execution_decision"))
        no_trade_decision = _format_runtime_decision("last_no_trade_decision", runtime_state.get("last_no_trade_decision"))
        if trade_decision:
            lines.append(trade_decision)
        if no_trade_decision:
            lines.append(no_trade_decision)

    if ctx.last_result:
        lines.extend(_prediction_fact_lines(ctx, ctx.last_result))

    return lines


def _handle_status_query(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    if not any(
        phrase in ql
        for phrase in (
            "show status",
            "current status",
            "status with",
            "status --decisions",
            "last decisions",
            "last decision",
            "show decisions",
            "show me the decisions",
            "current risk per trade",
            "what provider did you choose",
            "which provider",
            "execute mode",
        )
    ):
        return None

    facts = _status_fact_lines(ctx)
    fallback = "\n".join(
        [
            f"Current status: {ctx.oanda_instrument} {ctx.oanda_granularity} on {ctx.active_source or 'no active source'}.",
            f"Execution is {'LIVE' if ctx.oanda_execute else 'DRY-RUN'} with SL={ctx.stop_loss_pips if ctx.use_stop_loss else 'OFF'} and TP={ctx.take_profit_pips if ctx.use_take_profit else 'OFF'}.",
        ]
        + [line for line in facts if line.startswith("last_")]
    )
    answer = _grounded_chat_reply(
        ctx,
        user_query=q,
        task="Summarize Buddy's current runtime status and persisted decisions.",
        fact_lines=facts,
        fallback=fallback,
    )
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)
    return True


def _journal_fact_lines() -> list[str]:
    from src.utils.trade_journal import TradeJournal

    journal = TradeJournal()
    stats = journal.get_statistics(days=30)
    recent = journal.get_recent_trades(n=3)

    lines = [
        f"journal_total_trades={stats.get('total_trades')}",
        f"journal_open_trades={stats.get('open_trades')}",
        f"journal_closed_trades={stats.get('closed_trades')}",
        f"journal_win_rate={stats.get('win_rate')}",
        f"journal_total_pnl={stats.get('total_pnl')}",
        f"journal_total_pips={stats.get('total_pips')}",
        f"journal_profit_factor={stats.get('profit_factor')}",
    ]
    for idx, trade in enumerate(recent, start=1):
        lines.append(
            "recent_trade_"
            + str(idx)
            + "="
            + f"{trade.instrument} {trade.direction} status={trade.status} pnl={trade.pnl if trade.pnl is not None else 'n/a'} timestamp={trade.timestamp}"
        )
    return lines


def _journal_update_fact_lines() -> list[str]:
    from src.utils.trade_journal import TradeJournal
    from src.utils.oanda_practice import OandaPracticeClient

    journal = TradeJournal()
    client = OandaPracticeClient.from_env()
    sync_results = journal.sync_open_trades(client)
    return [
        f"closed_updated={sync_results.get('closed_updated', 0)}",
        f"open_updated={sync_results.get('open_updated', 0)}",
        f"tracked_live_trades={sync_results.get('tracked_live_trades', 0)}",
        f"untracked_live_trades={sync_results.get('untracked_live_trades', 0)}",
    ]


def _journal_import_fact_lines() -> list[str]:
    from src.utils.trade_journal import TradeJournal
    from src.utils.oanda_practice import OandaPracticeClient

    journal = TradeJournal()
    client = OandaPracticeClient.from_env()
    imported = journal.import_untracked_trades(client)
    return [
        f"imported_trades={imported}",
    ]


def _handle_journal_query(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    if "journal" in ql and any(phrase in ql for phrase in ("import", "bring in", "load untracked")):
        try:
            facts = _journal_import_fact_lines()
        except Exception as e:
            logger.debug("Journal import query failed: %s", e)
            answer = "I couldn't import untracked OANDA trades into the journal right now."
            print(_assistant_prefix(ctx) + answer)
            _remember_chat_turn(ctx, "assistant", answer)
            return True

        imported = next((x.split("=", 1)[1] for x in facts if x.startswith("imported_trades=")), "0")
        fallback = f"Journal import finished. Imported {imported} untracked trades."
        answer = _grounded_chat_reply(
            ctx,
            user_query=q,
            task="Summarize the result of importing untracked OANDA trades into the journal.",
            fact_lines=facts,
            fallback=fallback,
        )
        print(_assistant_prefix(ctx) + answer)
        _remember_chat_turn(ctx, "assistant", answer)
        return True

    if "journal" in ql and any(phrase in ql for phrase in ("update", "sync", "refresh")):
        try:
            facts = _journal_update_fact_lines()
        except Exception as e:
            logger.debug("Journal update query failed: %s", e)
            answer = "I couldn't sync the trade journal with OANDA right now."
            print(_assistant_prefix(ctx) + answer)
            _remember_chat_turn(ctx, "assistant", answer)
            return True

        fallback = (
            f"Journal sync finished. "
            f"Closed updated: {next((x.split('=', 1)[1] for x in facts if x.startswith('closed_updated=')), '0')}, "
            f"open updated: {next((x.split('=', 1)[1] for x in facts if x.startswith('open_updated=')), '0')}."
        )
        answer = _grounded_chat_reply(
            ctx,
            user_query=q,
            task="Summarize the result of syncing the local trade journal with OANDA.",
            fact_lines=facts,
            fallback=fallback,
        )
        print(_assistant_prefix(ctx) + answer)
        _remember_chat_turn(ctx, "assistant", answer)
        return True

    if not any(phrase in ql for phrase in ("journal", "recent trades", "performance", "win rate", "pnl")):
        return None

    try:
        facts = _journal_fact_lines()
    except Exception as e:
        logger.debug("Journal query failed: %s", e)
        print(_assistant_prefix(ctx) + "I couldn't read the trade journal cleanly from this workspace.")
        return True

    fallback = "\n".join(
        [
            f"Journal summary: {next((x.split('=', 1)[1] for x in facts if x.startswith('journal_total_trades=')), 'unknown')} total trades.",
            f"Recent performance facts: {next((x for x in facts if x.startswith('journal_win_rate=')), 'journal_win_rate=unknown')}, {next((x for x in facts if x.startswith('journal_total_pnl=')), 'journal_total_pnl=unknown')}.",
        ]
    )
    answer = _grounded_chat_reply(
        ctx,
        user_query=q,
        task="Summarize recent trade journal performance and recent trades.",
        fact_lines=facts,
        fallback=fallback,
    )
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)
    return True


def _model_status_fact_lines(ctx: TalkContext) -> list[str]:
    model_root = Path("trained_data") / "models"
    pair_rows: list[str] = []
    pair_count = 0
    validated_count = 0

    if model_root.exists():
        for child in sorted(model_root.iterdir()):
            if not child.is_dir():
                continue
            keras_path = child / "transformer_direction.keras"
            meta_path = child / "transformer_direction.meta.pkl"
            if keras_path.exists() and meta_path.exists():
                pair_count += 1
                validated = (child / ".validation_result.json").exists()
                if validated:
                    validated_count += 1
                pair_rows.append(f"pair_model={child.name} validated={validated}")

    legacy_meta = model_root / "modular_ensemble.meta.json"
    facts = [
        f"model_root={model_root}",
        f"pair_model_count={pair_count}",
        f"validated_pair_model_count={validated_count}",
        f"legacy_modular_ensemble_present={legacy_meta.exists()}",
    ]
    facts.extend(pair_rows[:8])
    facts.extend(_status_fact_lines(ctx))
    return facts


def _handle_model_status_query(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    if not any(phrase in ql for phrase in ("model status", "models status", "show models", "show model status", "which models are loaded")):
        return None

    facts = _model_status_fact_lines(ctx)
    fallback_lines = [
        f"Model status: {next((x.split('=', 1)[1] for x in facts if x.startswith('pair_model_count=')), '0')} pair models found.",
        f"Validated pair models: {next((x.split('=', 1)[1] for x in facts if x.startswith('validated_pair_model_count=')), '0')}.",
        f"Legacy modular ensemble present: {next((x.split('=', 1)[1] for x in facts if x.startswith('legacy_modular_ensemble_present=')), 'False')}.",
    ]
    sample_models = [x.split("=", 1)[1] for x in facts if x.startswith("pair_model=")]
    if sample_models:
        fallback_lines.append("Sample loaded models: " + ", ".join(sample_models[:4]))

    answer = _grounded_chat_reply(
        ctx,
        user_query=q,
        task="Summarize Buddy's current model inventory and runtime model status.",
        fact_lines=facts,
        fallback="\n".join(fallback_lines),
    )
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)
    return True


def _handle_account_query(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    if not any(phrase in ql for phrase in ("account", "nav", "balance", "margin available", "margin")):
        return None

    try:
        _ensure_oanda_client(ctx)
        summary = ctx.oanda_client.get_account_summary()
        account = (summary or {}).get("account") or {}
        facts = [
            f"nav={account.get('NAV') or account.get('nav')}",
            f"balance={account.get('balance')}",
            f"margin_available={account.get('marginAvailable')}",
            f"currency={account.get('currency')}",
        ]
        fallback = (
            f"Account status: NAV={account.get('NAV') or account.get('nav')} "
            f"balance={account.get('balance')} marginAvailable={account.get('marginAvailable')}."
        )
        answer = _grounded_chat_reply(
            ctx,
            user_query=q,
            task="Answer the user's account question from the OANDA account summary.",
            fact_lines=facts,
            fallback=fallback,
        )
        print(_assistant_prefix(ctx) + answer)
        _remember_chat_turn(ctx, "assistant", answer)
    except Exception as e:
        logger.debug("Account query failed: %s", e)
        answer = "I couldn't load the account summary right now."
        print(_assistant_prefix(ctx) + answer)
        _remember_chat_turn(ctx, "assistant", answer)
    return True


def _extract_all_instrument_hints(q: str) -> list[str]:
    instruments: list[str] = []
    seen: set[str] = set()
    patterns = (
        r"(?<![A-Za-z])([A-Za-z]{3})/([A-Za-z]{3})(?![A-Za-z])",
        r"(?<![A-Za-z])([A-Za-z]{3})_([A-Za-z]{3})(?![A-Za-z])",
        r"(?<![A-Za-z])([A-Za-z]{3})-([A-Za-z]{3})(?![A-Za-z])",
        r"(?<![A-Za-z])([A-Za-z]{3})\s+([A-Za-z]{3})(?![A-Za-z])",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, q):
            base = match.group(1).upper()
            quote = match.group(2).upper()
            if base in _FX_CURRENCY_CODES and quote in _FX_CURRENCY_CODES:
                instrument = f"{base}_{quote}"
                if instrument not in seen:
                    instruments.append(instrument)
                    seen.add(instrument)
    return instruments


def _summarize_scan_results(raw_results: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    items = raw_results or []
    for item in items:
        result = item[0] if isinstance(item, tuple) else item
        pair = getattr(result, "pair", None) or getattr(result, "instrument", None) or getattr(result, "symbol", None) or "unknown"
        direction = getattr(result, "direction", None) or "NONE"
        gates_passed = bool(getattr(result, "gates_passed", False))
        error = getattr(result, "error", None)
        confidence = getattr(result, "ridge_confidence", None)
        if confidence is None:
            confidence = getattr(result, "confidence", None)
        agent_total = getattr(result, "agent_total", None)
        rows.append(
            {
                "pair": str(pair),
                "direction": str(direction),
                "gates_passed": gates_passed,
                "error": error,
                "confidence": confidence,
                "agent_total": agent_total,
            }
        )

    tradable = [row for row in rows if row["direction"] not in {"NONE", "FLAT"} and row["error"] is None]
    approved = [row for row in tradable if row["gates_passed"]]
    top_pair = tradable[0] if tradable else (rows[0] if rows else None)
    return {
        "count": len(rows),
        "tradable_count": len(tradable),
        "approved_count": len(approved),
        "top_pair": top_pair,
        "rows": rows[:5],
        "all_rows": rows,
    }


def _filter_scan_rows(summary: dict[str, Any], *, approved_only: bool = False, limit: Optional[int] = None) -> list[dict[str, Any]]:
    rows = list(summary.get("all_rows") or [])
    rows = [row for row in rows if row.get("error") is None]
    if approved_only:
        rows = [row for row in rows if row.get("gates_passed")]
    if limit is not None:
        rows = rows[:max(0, limit)]
    return rows


def _extract_scan_pair_reference(q: str) -> list[str]:
    pairs = _extract_all_instrument_hints(q)
    if pairs:
        return pairs

    words = re.findall(r"\b[a-z]{3,}[_/-]?[a-z]{0,3}\b", q.lower())
    known_pairs = [str(row.get("pair")) for row in []]
    del words, known_pairs
    return []


def _find_scan_row(summary: dict[str, Any], pair: str) -> Optional[dict[str, Any]]:
    pair_upper = pair.upper().replace("/", "_")
    for row in summary.get("all_rows") or []:
        if str(row.get("pair", "")).upper() == pair_upper:
            return row
    return None


def _fallback_scan_comparison(summary: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> str:
    lines = [
        f"Comparing {left['pair']} vs {right['pair']}:",
        f"- {left['pair']}: direction={left['direction']}, gates_passed={left['gates_passed']}, confidence={left['confidence']}, agent_total={left['agent_total']}, error={left['error']}",
        f"- {right['pair']}: direction={right['direction']}, gates_passed={right['gates_passed']}, confidence={right['confidence']}, agent_total={right['agent_total']}, error={right['error']}",
    ]

    if left.get("gates_passed") and not right.get("gates_passed"):
        lines.append(f"{left['pair']} is stronger because it passed the trading gates and {right['pair']} did not.")
    elif right.get("gates_passed") and not left.get("gates_passed"):
        lines.append(f"{right['pair']} is stronger because it passed the trading gates and {left['pair']} did not.")
    else:
        left_conf = float(left.get("confidence") or 0.0)
        right_conf = float(right.get("confidence") or 0.0)
        if left_conf > right_conf:
            lines.append(f"{left['pair']} looks stronger on confidence.")
        elif right_conf > left_conf:
            lines.append(f"{right['pair']} looks stronger on confidence.")
        else:
            lines.append("They look similar on the stored scan metrics.")

    return "\n".join(lines)


def _handle_scan_comparison_query(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    summary = ctx.last_scan_summary
    if summary is None:
        return None
    pair_count = len(_extract_all_instrument_hints(q))
    if not (
        any(phrase in ql for phrase in ("why this setup", "why not", "compare", "versus", "vs "))
        or (pair_count >= 2 and "why" in ql and "not" in ql)
    ):
        return None

    pairs = _extract_all_instrument_hints(q)
    if len(pairs) < 2:
        rows = _filter_scan_rows(summary, approved_only=False, limit=2)
        if len(rows) >= 2:
            left, right = rows[0], rows[1]
        else:
            answer = "I need two scan setups to compare, and I don't have enough from the last scan."
            print(_assistant_prefix(ctx) + answer)
            _remember_chat_turn(ctx, "assistant", answer)
            return True
    else:
        left = _find_scan_row(summary, pairs[0])
        right = _find_scan_row(summary, pairs[1])
        if left is None or right is None:
            answer = "I couldn't find both requested pairs in the last scan results."
            print(_assistant_prefix(ctx) + answer)
            _remember_chat_turn(ctx, "assistant", answer)
            return True

    facts = [
        f"left_pair={left['pair']}",
        f"left_direction={left['direction']}",
        f"left_gates_passed={left['gates_passed']}",
        f"left_confidence={left['confidence']}",
        f"left_agent_total={left['agent_total']}",
        f"left_error={left['error']}",
        f"right_pair={right['pair']}",
        f"right_direction={right['direction']}",
        f"right_gates_passed={right['gates_passed']}",
        f"right_confidence={right['confidence']}",
        f"right_agent_total={right['agent_total']}",
        f"right_error={right['error']}",
    ]
    answer = _grounded_chat_reply(
        ctx,
        user_query=q,
        task="Compare two scan setups and explain why one is stronger or why one was not approved.",
        fact_lines=facts,
        fallback=_fallback_scan_comparison(summary, left, right),
    )
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)
    return True


def _render_scan_followup_fallback(summary: dict[str, Any], *, approved_only: bool, limit: Optional[int]) -> str:
    rows = _filter_scan_rows(summary, approved_only=approved_only, limit=limit)
    if not rows:
        return "I don't have any matching scan setups in the last scan context."

    header = "Approved setups from the last scan:" if approved_only else "Top setups from the last scan:"
    lines = [header]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"{idx}. {row['pair']} {row['direction']} | gates_passed={row['gates_passed']} | confidence={row['confidence']} | agent_total={row['agent_total']}"
        )
    return "\n".join(lines)


def _handle_scan_followup_query(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    if ctx.last_scan_summary is None:
        return None
    if "scan" in ql:
        return None
    if not any(phrase in ql for phrase in ("approved setups", "approved only", "top setups", "top 2", "top two", "show approved")):
        return None

    approved_only = any(phrase in ql for phrase in ("approved setups", "approved only", "show approved"))
    limit = None
    if "top two" in ql:
        limit = 2
    else:
        top_match = re.search(r"\btop\s+(\d+)\b", ql)
        if top_match:
            limit = int(top_match.group(1))

    rows = _filter_scan_rows(ctx.last_scan_summary, approved_only=approved_only, limit=limit)
    facts = [
        f"last_scan_result_count={ctx.last_scan_summary.get('count')}",
        f"last_scan_tradable_count={ctx.last_scan_summary.get('tradable_count')}",
        f"last_scan_approved_count={ctx.last_scan_summary.get('approved_count')}",
        f"approved_only={approved_only}",
        f"limit={limit}",
    ]
    for idx, row in enumerate(rows, start=1):
        facts.append(
            f"filtered_scan_row_{idx}={row['pair']} direction={row['direction']} gates_passed={row['gates_passed']} confidence={row['confidence']} agent_total={row['agent_total']}"
        )

    answer = _grounded_chat_reply(
        ctx,
        user_query=q,
        task="Answer a follow-up question about the most recent scan results.",
        fact_lines=facts,
        fallback=_render_scan_followup_fallback(ctx.last_scan_summary, approved_only=approved_only, limit=limit),
    )
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)
    return True


def _handle_scan_query(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    if "scan" not in ql:
        return None

    try:
        from cli.buddy_scanning import buddy_scan
    except Exception as e:
        logger.debug("Buddy scan import failed in chat: %s", e)
        print(_assistant_prefix(ctx) + "I couldn't load the scanner from this workspace.")
        return True

    instruments = _extract_all_instrument_hints(q)
    granularity_match = re.search(r"\b(M1|M5|M15|M30|H1|H4|D|D1)\b", q, flags=re.IGNORECASE)
    granularity = granularity_match.group(1).upper() if granularity_match else ctx.oanda_granularity
    auto_execute = any(phrase in ql for phrase in ("auto-execute", "auto execute", "execute if approved", "execute approved"))
    force = any(phrase in ql for phrase in (" force", "ignore session", "outside hours", "bypass session"))
    diversified = any(phrase in ql for phrase in ("diversified", "avoid correlated", "correlation cluster"))
    top_match = re.search(r"\btop\s+(\d+)\b", ql)
    top_n = int(top_match.group(1)) if top_match else 5
    pairs_arg = ",".join(instruments) if instruments else None

    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            raw_results = buddy_scan(
                config_path=ctx.config_path,
                pairs=pairs_arg,
                granularity=granularity,
                top_n=top_n,
                force=force,
                diversified=diversified,
                auto_execute=auto_execute,
                no_execute=not auto_execute,
                clean_output=True,
            )
    except Exception as e:
        logger.debug("Chat scan failed: %s", e)
        print(_assistant_prefix(ctx) + f"I couldn't complete the scan cleanly: {e}")
        return True

    summary = _summarize_scan_results(raw_results)
    ctx.last_scan_summary = summary
    facts = [
        f"scan_pairs={pairs_arg or 'default majors'}",
        f"scan_granularity={granularity}",
        f"scan_auto_execute={auto_execute}",
        f"scan_force={force}",
        f"scan_diversified={diversified}",
        f"scan_result_count={summary['count']}",
        f"scan_tradable_count={summary['tradable_count']}",
        f"scan_approved_count={summary['approved_count']}",
        f"scan_requested_top_n={top_n}",
    ]
    for idx, row in enumerate(summary["rows"], start=1):
        facts.append(
            f"scan_row_{idx}={row['pair']} direction={row['direction']} gates_passed={row['gates_passed']} confidence={row['confidence']} agent_total={row['agent_total']} error={row['error']}"
        )

    top_pair = summary.get("top_pair")
    fallback_lines = [
        f"Scan finished on {granularity} for {pairs_arg or 'the default major pairs'}.",
        f"I found {summary['tradable_count']} tradable setups out of {summary['count']} scanned results.",
    ]
    if top_pair:
        fallback_lines.append(
            f"Best setup right now looks like {top_pair['pair']} {top_pair['direction']} with gates_passed={top_pair['gates_passed']} and confidence={top_pair['confidence']}."
        )
    if auto_execute:
        fallback_lines.append("Auto-execute was enabled for approved setups.")

    answer = _grounded_chat_reply(
        ctx,
        user_query=q,
        task="Summarize the multi-pair scan and highlight the best setups.",
        fact_lines=facts,
        fallback="\n".join(fallback_lines),
        max_tokens=360,
    )
    print(_assistant_prefix(ctx) + answer)
    _remember_chat_turn(ctx, "assistant", answer)
    return True


def _handle_use_commands(ctx: TalkContext, q: str, *, period: str, interval: str) -> Optional[bool]:
    use_csv_arg = _extract_command_arg(q, "use csv ")
    if use_csv_arg is not None:
        df = _load_df_from_csv(use_csv_arg)
        df = _apply_feature_engineering_if_needed(ctx, df)
        ctx.active_source = f"csv:{use_csv_arg}"
        ctx.active_df = df
        ctx.last_result = None
        print(_assistant_prefix(ctx) + f"loaded {ctx.active_source}.")
        return True

    use_ticker_arg = _extract_command_arg(q, "use ticker ")
    if use_ticker_arg is not None:
        df = _load_df_from_ticker(use_ticker_arg, period=period, interval=interval)
        df = _apply_feature_engineering_if_needed(ctx, df)
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
        df = _apply_feature_engineering_if_needed(ctx, df)
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
        df = _apply_feature_engineering_if_needed(ctx, df)
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
        result = _predict_one_locked(ctx, ctx.active_df)
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
        print(_assistant_prefix(ctx) + f"🔸 Dry-run: would close position for {instrument}. Run 'execute on' to send.")
        return
    ctx.oanda_client.close_position(instrument=instrument)
    print(_assistant_prefix(ctx) + f"✅ Position closed for {instrument}.")


def _parse_trade_units(parts: list[str]) -> int:
    if len(parts) < 3:
        raise ValueError("Usage: trade buy <units> [instrument]")
    try:
        return int(parts[2])
    except Exception:
        raise ValueError("units must be an integer")


def _handle_trade_market(ctx: TalkContext, *, action: str, parts: list[str], skip_confidence_check: bool = False) -> None:
    """Handle direct buy/sell market orders.

    By default, checks confidence before executing:
    - HIGH confidence: Always execute
    - MEDIUM confidence + LOW risk: Execute
    - Otherwise: Refuse to execute

    Set skip_confidence_check=True to bypass (use with caution).
    """
    # Get units - if not provided, use default
    if len(parts) >= 3:
        units = _parse_trade_units(parts)
        instrument = _trade_instrument(parts, default=ctx.oanda_instrument, idx=3)
    else:
        instrument = ctx.oanda_instrument
        units = _default_trade_units(ctx, instrument)

    signed_units = abs(int(units)) * (1 if action == "buy" else -1)

    # Tier-1 gating before executing (unless explicitly skipped)
    if not skip_confidence_check:
        _ensure_last_prediction(ctx)
        can_trade, reason, _ = _fx_trade_gate(ctx, instrument=instrument, action=action)
        if not can_trade:
            print(_assistant_prefix(ctx) + f"⛔ I won't execute this trade - {reason}.")
            return

    if not ctx.oanda_execute:
        print(
            _assistant_prefix(ctx)
            + f"🔸 Dry-run: would place MARKET order {action.upper()} {abs(signed_units)} {instrument}. "
            + "Run 'execute on' to send."
        )
        return

    # Use new confidence-based risk management
    if ctx.risk_manager and ctx.last_result:
        risk_result = ctx.risk_manager.calculate_risk_levels(
            entry_price=ctx.last_result.get("last_close", 0.0),
            calibrated_confidence=None,  # Will use raw confidence
            raw_confidence=float(max(ctx.last_result.get("state_probs", [0.0]))),
            base_stop_loss_pips=ctx.stop_loss_pips if ctx.use_stop_loss else None,
            base_take_profit_pips=ctx.take_profit_pips if ctx.use_take_profit else None,
            instrument=instrument
        )

        if risk_result.is_valid:
            stop_loss_pips = risk_result.stop_loss_pips
            take_profit_pips = risk_result.take_profit_pips
            print(_assistant_prefix(ctx) + f"🎯 Confidence-based risk management: SL={stop_loss_pips:.1f}p, TP={take_profit_pips:.1f}p, R:R={risk_result.risk_reward_ratio:.2f}")
        else:
            stop_loss_pips = ctx.stop_loss_pips if ctx.use_stop_loss else None
            take_profit_pips = ctx.take_profit_pips if ctx.use_take_profit else None
    else:
        stop_loss_pips = ctx.stop_loss_pips if ctx.use_stop_loss else None
        take_profit_pips = ctx.take_profit_pips if ctx.use_take_profit else None

    # Calculate stop loss and take profit prices
    stop_loss_price, take_profit_price = _calculate_sl_tp_prices(ctx, instrument, action, stop_loss_pips, take_profit_pips)

    resp = ctx.oanda_client.create_market_order(
        instrument=instrument,
        units=signed_units,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
    )
    tx = (resp or {}).get("orderFillTransaction") or (resp or {}).get("orderCreateTransaction") or {}

    sl_tp_msg = ""
    if stop_loss_price:
        sl_tp_msg += f" SL={stop_loss_price:.5f}"
    if take_profit_price:
        sl_tp_msg += f" TP={take_profit_price:.5f}"

    print(_assistant_prefix(ctx) + f"✅ Order submitted! {action.upper()} {abs(signed_units)} {instrument}{sl_tp_msg} (id={tx.get('id')}).")


def _get_pip_value(instrument: str) -> float:
    """Get the pip value for an instrument.

    For most forex pairs, 1 pip = 0.0001
    For JPY pairs, 1 pip = 0.01
    """
    instrument_upper = instrument.upper().replace("_", "")
    if "JPY" in instrument_upper:
        return 0.01
    return 0.0001


def _get_current_price(ctx: TalkContext, instrument: str) -> Optional[float]:
    """Get current price from last result or OANDA."""
    if ctx.last_result and ctx.last_result.get("last_close"):
        return float(ctx.last_result["last_close"])

    try:
        _ensure_oanda_client(ctx)
        resp = ctx.oanda_client.get_candles(
            instrument, granularity="S5", count=1, price="M"
        )
        if resp and "candles" in resp and resp["candles"]:
            return float(resp["candles"][-1]["mid"]["c"])
    except Exception:
        pass
    return None


def _calculate_sl_tp_for_action(
    current_price: float,
    action: str,
    sl_distance: float,
    tp_distance: float,
    use_stop_loss: bool,
    use_take_profit: bool,
) -> tuple[Optional[float], Optional[float]]:
    """Calculate SL/TP prices based on action direction."""
    stop_loss_price = None
    take_profit_price = None

    if action.lower() == "buy":
        if use_stop_loss:
            stop_loss_price = current_price - sl_distance
        if use_take_profit:
            take_profit_price = current_price + tp_distance
    else:
        if use_stop_loss:
            stop_loss_price = current_price + sl_distance
        if use_take_profit:
            take_profit_price = current_price - tp_distance

    return stop_loss_price, take_profit_price


def _calculate_sl_tp_prices(
    ctx: TalkContext,
    instrument: str,
    action: str,
    stop_loss_pips: Optional[float] = None,
    take_profit_pips: Optional[float] = None
) -> tuple[Optional[float], Optional[float]]:
    """Calculate stop loss and take profit prices based on current price and pip settings."""
    if not ctx.use_stop_loss and not ctx.use_take_profit:
        return None, None

    # Use provided pips or fall back to context defaults
    sl_pips = stop_loss_pips if stop_loss_pips is not None else ctx.stop_loss_pips
    tp_pips = take_profit_pips if take_profit_pips is not None else ctx.take_profit_pips

    current_price = _get_current_price(ctx, instrument)
    if current_price is None:
        print(_assistant_prefix(ctx) + "⚠️ Could not get current price for SL/TP calculation")
        return None, None

    pip_value = _get_pip_value(instrument)
    sl_distance = sl_pips * pip_value
    tp_distance = tp_pips * pip_value

    return _calculate_sl_tp_for_action(
        current_price, action, sl_distance, tp_distance,
        ctx.use_stop_loss, ctx.use_take_profit
    )


def _get_account_equity(ctx: TalkContext) -> float:
    """Get account equity from OANDA. Returns default if unavailable."""
    try:
        _ensure_oanda_client(ctx)
        summary = ctx.oanda_client.get_account_summary()
        account = summary.get("account", {})
        # Try NAV first (Net Asset Value), then balance
        nav = account.get("NAV") or account.get("nav")
        if nav:
            return float(nav)
        balance = account.get("balance")
        if balance:
            return float(balance)
    except Exception:
        pass
    return 10000.0  # Fallback default


def _calculate_position_size(
    ctx: TalkContext,
    instrument: str,
    risk_pct: float = 0.02,  # 2% risk per trade default
) -> int:
    """Calculate position size based on account equity and risk parameters.

    Formula: units = (equity * risk_pct) / (stop_loss_pips * pip_value)

    For USD/JPY with 20 pip SL and 2% risk on $100k:
    units = (100000 * 0.02) / (20 * 0.01) = 2000 / 0.20 = 10,000 units
    """
    equity = _get_account_equity(ctx)
    risk_amount = equity * risk_pct

    # Get pip value based on instrument
    pip_value = _get_pip_value(instrument)
    stop_pips = ctx.stop_loss_pips if ctx.use_stop_loss else 20.0

    # Risk per pip = pip_value (for JPY pairs, 1 pip = 0.01)
    # For standard lot (100k units), 1 pip = ~$6.45 for USD/JPY
    # For mini lot (10k units), 1 pip = ~$0.645
    # Simplified: pip_value_per_unit ≈ pip_value for most pairs

    # Stop loss in price terms
    stop_distance = stop_pips * pip_value

    if stop_distance <= 0:
        return 10000  # Safe default

    # Position size = risk_amount / stop_distance
    # This gives us the units where losing stop_pips = risk_amount
    units = risk_amount / stop_distance

    # Round to nearest 1000 (standard lot sizing)
    units = int(round(units / 1000) * 1000)

    # Ensure minimum and maximum bounds
    units = max(1000, min(units, 500000))  # Between 1k and 500k units

    return units


def _default_trade_units(ctx: TalkContext = None, instrument: str = None) -> int:
    """Calculate smart position size based on account equity and risk.

    Uses configured risk per trade (default 2%) with the configured stop loss.
    """
    if ctx is None:
        return 10000  # Fallback if no context

    # Try to load risk from config
    risk_pct = 0.02
    try:
        cfg = load_config(ctx.config_path)
        risk_pct = float(cfg.get("fx", {}).get("risk", {}).get("risk_per_trade_pct", 0.02))
    except Exception:
        pass

    return _calculate_position_size(ctx, instrument or ctx.oanda_instrument, risk_pct=risk_pct)


def _try_parse_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except Exception:
        return None


def _ensure_last_prediction(ctx: TalkContext) -> None:
    if ctx.last_result is not None:
        return
    if ctx.active_df is None:
        raise ValueError("No active data source. Run 'use oanda ...' then 'predict' first.")
    if ctx.active_source and ctx.active_source.startswith("oanda:"):
        print(_assistant_prefix(ctx) + "Fetching latest candles...")
        ctx.active_df = _load_df_from_oanda(
            ctx,
            instrument=ctx.oanda_instrument,
            granularity=ctx.oanda_granularity,
            candles=ctx.oanda_candles,
            price=ctx.oanda_price,
        )
        print(_assistant_prefix(ctx) + "Candles fetched. Running inference...")
    else:
        print(_assistant_prefix(ctx) + "Running inference on active data...")

    ctx.last_result = _predict_one_locked(ctx, ctx.active_df)
    print(_assistant_prefix(ctx) + "Inference complete.")


def _parse_trade_auto_units_and_instrument(ctx: TalkContext, parts: list[str]) -> tuple[int, str]:
    # Supported:
    # - trade
    # - trade auto
    # - trade auto <units> [instrument]
    # - trade <units> [instrument]
    instrument = ctx.oanda_instrument
    units: Optional[int] = None

    if len(parts) >= 2 and parts[1].lower().strip() == "auto":
        units = _try_parse_int(parts[2]) if len(parts) >= 3 else None
        if len(parts) >= 4 and parts[3]:
            instrument = parts[3]
    elif len(parts) >= 2:
        units = _try_parse_int(parts[1])
        if units is not None and len(parts) >= 3 and parts[2]:
            instrument = parts[2]

    if units is None:
        units = _default_trade_units(ctx, instrument)
    return int(units), instrument


def _handle_trade_auto(ctx: TalkContext, parts: list[str]) -> None:
    print(_assistant_prefix(ctx) + "Analyzing market data...")
    units, instrument = _parse_trade_auto_units_and_instrument(ctx, parts)

    _ensure_last_prediction(ctx)
    result = ctx.last_result or {}
    pred = result.get("prediction")
    last_close = result.get("last_close")

    if pred is None or last_close is None:
        raise ValueError("Need a valid prediction and last close. Run 'predict' first.")

    pred_f, last_f = float(pred), float(last_close)
    delta = pred_f - last_f

    if delta == 0:
        print(_assistant_prefix(ctx) + "➡️ HOLD - my prediction equals the last close. No edge here.")
        return

    action = "buy" if delta > 0 else "sell"
    signed_units = abs(int(units)) * (1 if action == "buy" else -1)
    can_trade, reason, _ = _fx_trade_gate(ctx, instrument=instrument, action=action)

    if not can_trade:
        emoji = _trend_emoji(delta)
        print(
            _assistant_prefix(ctx)
            + f"{emoji} I want to {action.upper()}, but I'm not confident enough.\n"
            + f"Reason: {reason}\n"
            + "Run 'summary' to see details, or wait for a better setup."
        )
        return

    rationale = f"pred={pred_f:.5f} last={last_f:.5f} Δ={delta:+.5f}"

    if not ctx.oanda_execute:
        print(
            _assistant_prefix(ctx)
            + f"🔸 Dry-run: I'd {action.upper()} {abs(signed_units)} {instrument} ({rationale}).\n"
            + f"Confidence: {reason}\n"
            + "Run 'execute on' to enable live trading."
        )
        return

    _execute_auto_trade(ctx, instrument, action, signed_units, reason)


def _execute_auto_trade(
    ctx: TalkContext,
    instrument: str,
    action: str,
    signed_units: int,
    reason: str,
) -> None:
    """Execute the auto trade order."""
    stop_loss_price, take_profit_price = _calculate_sl_tp_prices(ctx, instrument, action)

    resp = ctx.oanda_client.create_market_order(
        instrument=instrument,
        units=signed_units,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
    )
    tx = (resp or {}).get("orderFillTransaction") or (resp or {}).get("orderCreateTransaction") or {}

    sl_tp_info = ""
    if stop_loss_price:
        sl_tp_info += f"\n   Stop Loss: {stop_loss_price:.5f}"
    if take_profit_price:
        sl_tp_info += f"\n   Take Profit: {take_profit_price:.5f}"

    print(
        _assistant_prefix(ctx)
        + f"✅ Auto-trade executed: {action.upper()} {abs(signed_units)} {instrument}\n"
        + f"   Order ID: {tx.get('id')}{sl_tp_info}\n"
        + f"   Reason: {reason}"
    )


def _handle_trade_commands(ctx: TalkContext, q: str, ql: str) -> Optional[bool]:
    # Handle force buy/sell (bypasses confidence check)
    if ql in {"force buy", "force sell"}:
        _ensure_oanda_client(ctx)
        action = ql.split()[1]  # "buy" or "sell"
        print(_assistant_prefix(ctx) + f"⚠️ Force {action.upper()} - bypassing confidence check...")
        _handle_trade_market(ctx, action=action, parts=[action], skip_confidence_check=True)
        return True

    # Handle direct buy/sell commands (not just "trade buy/sell")
    if ql in {"buy", "sell"}:
        _ensure_oanda_client(ctx)
        _handle_trade_market(ctx, action=ql, parts=[ql])
        return True

    if not ql.startswith("trade"):
        return None

    parts = q.split()

    # Ensure client only when we're actually going to trade.
    _ensure_oanda_client(ctx)

    # Auto trade:
    # - `trade`
    # - `trade auto ...`
    # - `trade <units> [instrument]`
    if len(parts) == 1:
        _handle_trade_auto(ctx, parts)
        return True

    action = parts[1].lower().strip()
    if action in {"buy", "sell", "close"}:
        if action == "close":
            _handle_trade_close(ctx, parts)
            return True
        _handle_trade_market(ctx, action=action, parts=parts)
        return True

    if action == "auto":
        _handle_trade_auto(ctx, parts)
        return True

    # Fallback: interpret `trade <units> [instrument]` as auto-trade.
    _handle_trade_auto(ctx, parts)
    return True


def _planner_runtime_context(ctx: TalkContext) -> dict[str, Any]:
    return {
        "active_instrument": ctx.oanda_instrument,
        "granularity": ctx.oanda_granularity,
        "execute_mode": "live" if ctx.oanda_execute else "dry-run",
        "stop_loss_pips": ctx.stop_loss_pips if ctx.use_stop_loss else "off",
        "take_profit_pips": ctx.take_profit_pips if ctx.use_take_profit else "off",
        "knowledge_only": ctx.engine is None,
        "active_source": ctx.active_source,
        "has_prediction": ctx.last_result is not None,
    }


def _planner_transitions_enabled() -> bool:
    raw = os.getenv("BUDDY_PLANNER_TRANSITIONS", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _planner_tool_status_text(tool_name: str) -> str:
    tool_key = str(tool_name or "").strip().lower()
    labels = {
        "scan_market": "scanning market",
        "run_prediction": "running prediction",
        "compare_scan_results": "comparing scan results",
        "get_status": "checking runtime status",
        "get_account_summary": "checking account summary",
        "get_journal_summary": "checking journal summary",
        "get_model_status": "checking model status",
        "run_trade_command": "preparing trade command",
        "run_runtime_command": "applying runtime settings",
        "override_guardrail": "updating guardrail override",
        "run_oanda_command": "running oanda command",
        "use_market_source": "loading market source",
        "summarize_last_prediction": "summarizing prediction",
        "answer_buddy_self_question": "answering grounded self question",
        "sync_journal": "syncing journal",
        "import_journal_trades": "importing journal trades",
    }
    if tool_key in labels:
        return labels[tool_key]
    return f"running {tool_key.replace('_', ' ')}"


def _planner_progress_callback(ctx: TalkContext):
    if not _planner_transitions_enabled():
        return None

    def callback(event: str, payload: dict[str, Any]) -> None:
        prefix = _assistant_prefix(ctx)
        if event == "thinking_start":
            print(prefix + "thinking...", flush=True)
            return
        if event == "plan_ready":
            step_count = int(payload.get("step_count") or 0)
            if step_count > 0:
                print(prefix + f"plan ready: {step_count} step(s).", flush=True)
            else:
                print(prefix + "plan ready.", flush=True)
            return
        if event == "tool_start":
            tool_name = str(payload.get("tool") or "")
            idx = int(payload.get("index") or 0)
            total = int(payload.get("total") or 0)
            phase = _planner_tool_status_text(tool_name)
            if idx > 0 and total > 0:
                print(prefix + f"{phase} ({idx}/{total})...", flush=True)
            else:
                print(prefix + f"{phase}...", flush=True)
            return
        if event == "tool_done":
            tool_name = str(payload.get("tool") or "").replace("_", " ")
            if bool(payload.get("ok", True)):
                print(prefix + f"{tool_name} complete.", flush=True)
            else:
                print(prefix + f"{tool_name} failed.", flush=True)
            return
        if event == "reasoning_start":
            print(prefix + "reasoning...", flush=True)
            return
        if event == "done":
            print(prefix + "response ready.", flush=True)

    return callback


def _handle_planner_chat(ctx: TalkContext, q: str, *, period: str, interval: str) -> Optional[bool]:
    if not _planner_chat_enabled():
        return None

    try:
        from src.agent.planner_runtime import PlannerRuntime
    except Exception as e:
        logger.debug("Planner runtime unavailable: %s", e)
        return None

    if ctx.planner_runtime is None:
        try:
            ctx.planner_runtime = PlannerRuntime.create_default(ctx.config_path)
        except Exception as e:
            logger.debug("Planner runtime init failed: %s", e)
            return None

    ctx.planner_state.update(_planner_runtime_context(ctx))
    ctx.planner_state["_talk_context"] = ctx
    ctx.planner_state["_period"] = period
    ctx.planner_state["_interval"] = interval
    if ctx.last_scan_summary is not None:
        ctx.planner_state["last_scan_summary"] = ctx.last_scan_summary

    progress_callback = _planner_progress_callback(ctx)
    response = ctx.planner_runtime.handle_request(
        q,
        chat_history=ctx.chat_history,
        runtime_context=_planner_runtime_context(ctx),
        state=ctx.planner_state,
        progress_callback=progress_callback,
    )
    state = (response.metadata or {}).get("state") or {}
    if isinstance(state, dict):
        ctx.planner_state = dict(state)
        if isinstance(state.get("last_scan_summary"), dict):
            ctx.last_scan_summary = state["last_scan_summary"]

    if response.answer:
        print(_assistant_prefix(ctx) + response.answer)
        _remember_chat_turn(ctx, "assistant", response.answer)
        return True
    return None


def _handle_talk_command(ctx: TalkContext, q: str, *, period: str, interval: str) -> bool:
    ql = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", q.lower()).strip()
    _remember_chat_turn(ctx, "user", q)

    if _is_explicit_talk_command(ql):
        basic_out = _handle_basic_commands(ctx, ql)
        if basic_out is not None:
            return basic_out

    if _planner_chat_enabled() and not _is_explicit_talk_command(ql):
        planner_out = _handle_planner_chat(ctx, q, period=period, interval=interval)
        if planner_out is not None:
            return planner_out

    for handler in (
        lambda: _handle_basic_commands(ctx, ql),
        lambda: _handle_buddy_self_query(ctx, q, ql),
        lambda: _handle_knowledge_only_chat(ctx, q, ql),
        lambda: _handle_status_query(ctx, q, ql),
        lambda: _handle_model_status_query(ctx, q, ql),
        lambda: _handle_journal_query(ctx, q, ql),
        lambda: _handle_account_query(ctx, q, ql),
        lambda: _handle_scan_comparison_query(ctx, q, ql),
        lambda: _handle_scan_followup_query(ctx, q, ql),
        lambda: _handle_scan_query(ctx, q, ql),
        lambda: _handle_use_commands(ctx, q, period=period, interval=interval),
        lambda: _handle_predict_commands(ctx, q, ql, period=period, interval=interval),
        lambda: _handle_oanda_info_commands(ctx, q, ql),
        lambda: _handle_trade_commands(ctx, q, ql),
    ):
        out = handler()
        if out is not None:
            return out

    plan = _plan_talk_request(ctx, q, ql)
    if plan is not None:
        return _execute_talk_plan(ctx, plan, q=q, period=period, interval=interval)

    if ctx.engine is not None and ctx.active_df is not None and ctx.last_result is None:
        ctx.last_result = _predict_one_locked(ctx, ctx.active_df)

    if ctx.last_result is None:
        if ctx.engine is None:
            ans = (
                "I'm in knowledge-first mode right now. "
                "I can help with grounded Buddy questions, status, decisions, scans, journal, or account tasks, "
                "but I don't have a compatible prediction checkpoint loaded for free-form market answers."
            )
        else:
            ans = (
                "I don't have a fresh market prediction yet. "
                "Ask me to use OANDA or another source first, or ask a grounded Buddy self-knowledge question."
            )
    else:
        ans = _render_prediction_chat_answer(ctx, q, ctx.last_result)
    print(_assistant_prefix(ctx) + ans)
    _remember_chat_turn(ctx, "assistant", ans)
    return True


def _read_input_with_timeout(prompt: str, timeout: float) -> Optional[str]:
    """Read a single line from stdin with a timeout.

    Uses `signal.setitimer` on Unix-like systems to interrupt `input()`.
    Returns:
      - str: user input (may be empty string)
      - None: timed out
    """

    # If we're not interactive (piped input), don't try to use signals/timeouts.
    # This avoids surprising behavior in non-tty contexts.
    import sys

    if not sys.stdin.isatty():
        return input(prompt)

    try:
        import signal

        class _InputTimeout(Exception):
            pass

        def _alarm_handler(_signum: int, _frame: object) -> None:
            raise _InputTimeout()

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        try:
            return input(prompt)
        except _InputTimeout:
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
    except Exception:
        # Fallback: no timeout support (e.g., signals not available / not main thread).
        return input(prompt)


def _run_repl(ctx: TalkContext, *, period: str, interval: str) -> None:
    INPUT_TIMEOUT = 300.0  # seconds
    idle_prompt_shown = False

    while True:
        try:
            q = _read_input_with_timeout("You: ", timeout=INPUT_TIMEOUT)
            if q is None:
                if not idle_prompt_shown:
                    print(f"\n{_assistant_prefix(ctx)}Still here? Type something or 'exit' to quit.")
                    idle_prompt_shown = True
                continue
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        idle_prompt_shown = False
        if not q:
            continue

        try:
            if not _handle_talk_command(ctx, q, period=period, interval=interval):
                return
        except Exception as e:
            print(_assistant_prefix(ctx) + f"error: {e}")


@dataclass
class OandaSettings:
    """OANDA trading settings."""
    instrument: str = "USD_JPY"
    granularity: str = "M5"
    candles: int = 300
    execute: bool = True


def _apply_oanda_settings(ctx: TalkContext, settings: OandaSettings) -> None:
    """Apply OANDA settings to context."""
    ctx.oanda_instrument = settings.instrument
    ctx.oanda_granularity = settings.granularity
    ctx.oanda_candles = int(settings.candles)
    ctx.oanda_execute = bool(settings.execute)


def run_unified_talk(
    config_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    ticker: Optional[str] = None,
    period: str = "5d",
    interval: str = "1h",
    oanda: bool = False,
    oanda_settings: Optional[OandaSettings] = None,
    all_features: bool = False,
    verbose: bool = False,
    assistant_name: str = DEFAULT_ASSISTANT_NAME,
) -> None:
    """Run the unified talk REPL.

    Args:
        config_path: Path to config file.
        checkpoint_path: Optional model checkpoint path.
        csv_path: Optional CSV data source.
        ticker: Optional ticker symbol.
        period: Ticker data period (default "5d").
        interval: Ticker data interval (default "1h").
        oanda: Enable OANDA data source.
        oanda_settings: OANDA configuration (uses defaults if None).
        all_features: Enable all feature engineering.
        verbose: Enable verbose output.
        assistant_name: Name for the assistant.
    """
    ctx = _load_context(config_path, checkpoint_path=checkpoint_path)

    if all_features:
        ctx.apply_feature_engineering = True

    ctx.verbose = bool(verbose)
    ctx.assistant_name = str(assistant_name or DEFAULT_ASSISTANT_NAME)

    settings = oanda_settings or OandaSettings()
    _apply_oanda_settings(ctx, settings)

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
# — Raynergy-svg —
