#!/usr/bin/env python3
"""CLI command implementations for ML Engine Trading Bot."""
from __future__ import annotations

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
from rich.panel import Panel

from cli.config import BuddyTrainingAdvancedOptions, OandaFetchOptions
from cli.io_utils import (
    console, DEFAULT_CONFIG_PATH, _validate_instrument, _normalize_instrument, BUDDY_META_FILENAME, _meta_path_for_checkpoint,
)
from cli.tf_config import _configure_tf_metal

from src.utils import load_config
from src.core.constants import DIRECTION_DEFAULTS

logger = logging.getLogger(__name__)

# Import from refactored modules
from cli.buddy_helpers import (
    _tier2_apply_calibration, _configure_predict_output, _buddy_live_enabled_from_meta,
    _fx_execution_guard_price_bound, _schedule_auto_close, _build_buddy_model_for_type,
)
from cli.wizard import _buddy_interactive_wizard, launch_buddy_repl_from_wizard


# train_buddy is implemented in cli.training - re-export for backward compatibility
from cli.training import train_buddy  # noqa: F401, E402


# Major FX pairs for "all" mode
_MAJOR_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
]

INFERENCE_PROFILE_BALANCED = "balanced"
INFERENCE_PROFILE_OVERRIDES: dict[str, dict[str, float | int]] = {
    INFERENCE_PROFILE_BALANCED: {},
    # Fewer trades, stricter gate quality.
    "conservative": {
        "min_confidence": 58.0,
        "min_momentum": 0.28,
        "max_drawdown_pct": 0.020,
        "min_atr_pips": 6.0,
        "min_volatility_regime": 2,
        "min_tcn_probability": 0.58,
        "use_rsi_extreme_gate": True,
        "use_trend_contra_gate": True,
    },
    # Higher trade frequency while preserving risk controls.
    "aggressive": {
        "min_confidence": 45.0,
        "min_momentum": 0.12,
        "max_drawdown_pct": 0.035,
        "min_atr_pips": 3.0,
        "min_volatility_regime": 0,
        "min_tcn_probability": 0.51,
        "use_rsi_extreme_gate": False,
        "use_trend_contra_gate": False,
    },
}

INFERENCE_OPPORTUNITY_OVERRIDES: dict[str, Any] = {
    # Signal discovery mode: don't fail-closed on schema/model-contract drift.
    "enforce_model_contracts": False,
    "require_direction_model": False,
    "require_volatility_model": False,
    "require_regime_model": False,
    "require_xgb_model": False,
    "require_rf_model": False,
    "require_ridge_model": False,
    # Scanner parity: contextual overlays should not hard-veto discovery.
    "sentiment_block_enabled": False,
    "enable_meta_labeling": False,
    "overlay_meta_enabled": False,
    "final_score_threshold": 0.45,
}


_POST_TRADE_LLM_SYSTEM = """You are a disciplined FX trade manager.
Give post-execution guidance for an already-open trade.
Keep it practical and concise.
Return JSON only with keys:
quality, summary, if_good, if_bad, take_profit, exit
`quality` must be one of: good, neutral, bad.
Each value should be under 20 words.
"""


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model text."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : idx + 1]
                try:
                    parsed = json.loads(snippet)
                except Exception:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _normalize_llm_news_sentiment(sentiment: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize market-intel sentiment payload for Buddy LLM prompts."""
    if not sentiment:
        return None

    normalized = {
        "sentiment": sentiment.get("sentiment") or sentiment.get("aggregate_label"),
        "score": sentiment.get("score", sentiment.get("aggregate_score", 0.0)),
        "headline_count": sentiment.get("headline_count", sentiment.get("num_headlines", 0)),
    }

    headlines = sentiment.get("headlines")
    if isinstance(headlines, list):
        normalized["headlines"] = headlines

    return normalized


def _build_buddy_price_context(df: Any) -> dict[str, Any]:
    """Build compact market context for intelligent-mode reasoning/validation."""
    if df is None or len(df) == 0:
        return {}

    def _latest(col: str, default: Any = None) -> Any:
        if col not in df.columns:
            return default
        try:
            value = df[col].iloc[-1]
        except Exception:
            return default
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return default
        return float(value) if isinstance(value, (float, int, np.floating, np.integer)) else value

    last_close = _latest("close")
    lookback = min(len(df), 20)
    trend = "unknown"
    if last_close is not None and lookback >= 2 and "close" in df.columns:
        try:
            base_close = float(df["close"].iloc[-lookback])
            trend = "up" if float(last_close) > base_close else "down"
        except Exception:
            trend = "unknown"

    return {
        "last_close": last_close,
        "atr": _latest("atr"),
        "atr_pct": _latest("atr_pct_14"),
        "rsi": _latest("rsi"),
        "adx": _latest("adx"),
        "trend": trend,
    }


def _build_multi_modal_context(intel_data: dict[str, Any] | None) -> Any:
    """Build MultiModalContext from modular inference intelligence metadata."""
    from buddy_intelligent_mode import MultiModalContext

    intel_data = intel_data or {}
    sentiment = intel_data.get("sentiment", {}) if isinstance(intel_data, dict) else {}

    news_summaries: list[str] = []
    headlines = sentiment.get("headlines")
    if isinstance(headlines, list):
        news_summaries.extend(str(item) for item in headlines[:10])

    aggregate_label = sentiment.get("aggregate_label")
    aggregate_score = sentiment.get("aggregate_score")
    if aggregate_label and not news_summaries:
        news_summaries.append(f"Market sentiment is {aggregate_label}")

    upcoming_events: list[str] = []
    next_event = intel_data.get("next_high_impact")
    if isinstance(next_event, dict):
        event_name = next_event.get("name", "High-impact event")
        minutes_until = next_event.get("minutes_until")
        if minutes_until is not None:
            upcoming_events.append(f"{event_name} in {minutes_until} minutes")
        else:
            upcoming_events.append(str(event_name))

    impact_flags: list[str] = []
    if intel_data.get("calendar_high_impact", 0):
        impact_flags.append(f"{intel_data.get('calendar_high_impact', 0)} high-impact events")
    calendar_reason = intel_data.get("calendar_reason")
    if calendar_reason:
        impact_flags.append(str(calendar_reason))

    return MultiModalContext(
        news_summaries=news_summaries,
        sentiment_score=float(aggregate_score or 0.0),
        upcoming_events=upcoming_events,
        event_impact_flags=impact_flags,
    )


def _build_buddy_raw_output(signal: Any, current_price: float | None) -> Any:
    """Convert a modular inference signal into BuddyRawOutput."""
    from buddy_intelligent_mode import BuddyRawOutput

    metadata = signal.metadata or {}
    gate_results = {
        "trade": bool(signal.trade),
        "reason": signal.reason,
        "confidence_gate_passed": bool(signal.confidence_gate_passed),
        "momentum_gate_passed": bool(signal.momentum_gate_passed),
        "risk_gate_passed": bool(signal.risk_gate_passed),
        "regime_gate_passed": bool(signal.regime_gate_passed),
        "meta_gate_passed": bool(signal.meta_gate_passed),
        "gate_observability": metadata.get("gate_observability", {}).get("current", {}),
        "stage_a": metadata.get("stage_a", {}),
        "stage_b": metadata.get("stage_b", {}),
        "thresholds": metadata.get("gate_thresholds", {}),
    }

    return BuddyRawOutput(
        confidence=float(signal.confidence),
        prediction=float(current_price) if current_price is not None else float(signal.tcn_probability),
        probability=float(signal.tcn_probability),
        direction=signal.direction,
        last_price=float(current_price) if current_price is not None else None,
        features={
            "ridge_confidence": float(signal.ridge_confidence),
            "xgb_momentum": float(signal.xgb_momentum),
            "rf_drawdown_pips": float(signal.rf_drawdown_pips),
            "rf_streak_prob": float(signal.rf_streak_prob),
            "meta_confidence": float(signal.meta_confidence),
        },
        gate_results=gate_results,
    )


def _build_trade_journal_metadata(signal: Any) -> dict[str, Any]:
    """Capture entry-time context for later adaptive learning."""
    metadata = dict(getattr(signal, "metadata", {}) or {})
    intel_data = dict(metadata.get("intel_data", {}) or {})
    sentiment = dict(intel_data.get("sentiment", {}) or {})

    return {
        "volatility_regime": getattr(signal, "volatility_regime", None),
        "volatility_regime_name": getattr(signal, "volatility_regime_name", None),
        "volatility_regime_confidence": getattr(signal, "volatility_regime_confidence", None),
        "regime": getattr(signal, "regime", None),
        "regime_confidence": getattr(signal, "regime_confidence", None),
        "calendar_high_impact": intel_data.get("calendar_high_impact"),
        "calendar_reason": intel_data.get("calendar_reason"),
        "next_high_impact": intel_data.get("next_high_impact"),
        "sentiment_label": sentiment.get("aggregate_label") or sentiment.get("sentiment"),
        "sentiment_score": sentiment.get("aggregate_score", sentiment.get("score")),
        "headlines_count": intel_data.get("headlines_count", sentiment.get("num_headlines")),
        "pre_trade_blocked": intel_data.get("pre_trade_blocked"),
        "pre_trade_block_reason": intel_data.get("pre_trade_block_reason"),
        "gate_reason": getattr(signal, "reason", None),
    }


def _update_buddy_runtime_state(**values: Any) -> None:
    """Persist lightweight Buddy runtime state for grounded chat answers."""
    try:
        from memory_client import MLEngineMemory

        memory = MLEngineMemory()
        memory.update_runtime_state("buddy_runtime", values)
    except Exception as e:
        logger.debug("Failed to update Buddy runtime state: %s", e)


def _run_buddy_intelligent_analysis(
    *,
    instrument: str,
    granularity: str,
    signal: Any,
    df: Any,
    ensemble: Any,
    intelligent: bool,
    explain: bool,
    llm_initialized: bool,
) -> dict[str, Any]:
    """Run intelligent-mode validation/reasoning against the live Buddy signal."""
    result: dict[str, Any] = {
        "validation": None,
        "llm_approved": True,
        "reasoning": None,
        "adaptive_policy": None,
    }
    if not llm_initialized or not (intelligent or explain):
        return result

    try:
        from buddy_intelligent_mode import (
            apply_adaptive_trade_policy,
            get_intelligent_mode,
            validate_trade_with_risk_assessment,
        )
    except Exception as e:
        logger.debug("Buddy intelligent analysis unavailable: %s", e)
        return result

    price_context = _build_buddy_price_context(df)
    current_price = price_context.get("last_close")
    buddy_raw = _build_buddy_raw_output(signal, current_price)
    intel_data = signal.metadata.get("intel_data", {}) if signal.metadata else {}
    news_sentiment = _normalize_llm_news_sentiment(intel_data.get("sentiment"))
    adaptive_context = dict(price_context)
    adaptive_context.update(_build_trade_journal_metadata(signal))

    intel_mode = None
    if explain or intelligent:
        try:
            intel_mode = get_intelligent_mode()
        except Exception as e:
            logger.debug("Buddy intelligent mode unavailable: %s", e)

    if explain and intel_mode is not None:
        try:
            multi_modal_context = _build_multi_modal_context(intel_data)
            result["reasoning"] = intel_mode.get_full_intelligent_analysis(
                ticker=instrument,
                timeframe=granularity,
                buddy_raw=buddy_raw,
                context=multi_modal_context,
                price_data=price_context,
            )
        except Exception as e:
            logger.debug("Buddy reasoning generation failed: %s", e)

    if intelligent and signal.trade and signal.direction:
        try:
            validation = validate_trade_with_risk_assessment(
                ticker=instrument,
                direction=signal.direction,
                buddy_raw=buddy_raw,
                price_data=price_context,
                news_sentiment=news_sentiment,
                market_intel=getattr(ensemble, "market_intel", None),
            )
            if intel_mode is not None:
                adaptive_policy = intel_mode.get_adaptation_policy(
                    ticker=instrument,
                    direction=signal.direction,
                    confidence=float(signal.confidence),
                    context=adaptive_context,
                )
                result["adaptive_policy"] = adaptive_policy
                validation = apply_adaptive_trade_policy(validation, adaptive_policy)
            result["validation"] = validation
            result["llm_approved"] = bool(validation.approve)
        except Exception as e:
            logger.debug("Buddy trade validation failed: %s", e)

    return result


def _pair_has_models(pair: str, model_root: Path = Path("trained_data/models")) -> bool:
    """Return True when pair-specific model artifacts exist."""
    normalized_pair = _normalize_instrument(pair)
    pair_dir = model_root / normalized_pair
    if not pair_dir.exists():
        return False
    return any(pair_dir.glob("*.keras")) or any(pair_dir.glob("*.pkl"))


def _resolve_master_pair_for_buddy(
    *,
    target_pair: str,
    explicit_master_pair: str | None,
    interactive: bool,
) -> str:
    """Resolve model source pair for buddy inference."""
    target = _normalize_instrument(target_pair)

    if explicit_master_pair:
        return _normalize_instrument(explicit_master_pair)

    try:
        from src.training.correlation_group_config import get_correlation_group
    except Exception:
        return target

    group = get_correlation_group(target)
    if group is None:
        return target

    recommended = _normalize_instrument(group.master_pair)
    options: list[str] = []
    for pair in [target, recommended, *group.pairs]:
        normalized = _normalize_instrument(pair)
        if normalized not in options:
            options.append(normalized)

    target_has_models = _pair_has_models(target)
    recommended_has_models = _pair_has_models(recommended)
    default_choice = target
    if (not target_has_models) and recommended_has_models:
        default_choice = recommended

    if not interactive or not sys.stdin.isatty() or len(options) <= 1:
        return default_choice

    console.print("")
    console.print("[bold cyan]Master Pair Selection[/bold cyan]")
    console.print(
        f"[dim]Target: {target} | Group: {group.name} | Recommended master: {recommended}[/dim]"
    )
    console.print(
        "[dim]Choose which pair's trained models to use for direction/gate inference.[/dim]"
    )
    for idx, pair in enumerate(options, start=1):
        available = _pair_has_models(pair)
        marker = "✓" if available else "·"
        suffix = " (target)" if pair == target else (" (recommended)" if pair == recommended else "")
        console.print(f"  {idx}. {pair} [{marker} models]{suffix}")

    default_idx = options.index(default_choice) + 1
    while True:
        selection = input(f"Select master pair [1-{len(options)}] (default {default_idx}): ").strip()
        if not selection:
            chosen = options[default_idx - 1]
            break
        if selection.isdigit():
            num = int(selection)
            if 1 <= num <= len(options):
                chosen = options[num - 1]
                break
        console.print("[yellow]Invalid selection. Enter a number from the list.[/yellow]")

    if not _pair_has_models(chosen):
        console.print(
            f"[yellow]⚠ No pair-specific models found for {chosen}; generic/joint fallback may be used.[/yellow]"
        )

    return chosen


def _compute_rl_exit_scenarios(
    *,
    ensemble: Any,
    df: Any,
    direction: str,
    entry_price: float,
    stop_loss_pips: float | None,
    take_profit_pips: float | None,
) -> dict[str, dict[str, Any]]:
    """Run RL exit checks for now/if_good/if_bad scenarios."""
    scenarios: dict[str, dict[str, Any]] = {}
    if ensemble is None or df is None:
        return scenarios

    sl_pips = float(stop_loss_pips) if stop_loss_pips and stop_loss_pips > 0 else 15.0
    tp_pips = float(take_profit_pips) if take_profit_pips and take_profit_pips > 0 else 30.0
    scenario_inputs = {
        "now": (0.0, 1),
        "if_good": (max(tp_pips * 0.65, 10.0), 8),
        "if_bad": (-max(sl_pips * 0.65, 8.0), 8),
    }

    for label, (pnl_pips, bars_held) in scenario_inputs.items():
        try:
            raw = ensemble.check_open_position_exit(
                df=df,
                unrealized_pnl_pips=float(pnl_pips),
                bars_in_trade=int(bars_held),
                direction=direction,
                entry_price=float(entry_price),
                stop_loss_pips=sl_pips,
                take_profit_pips=tp_pips,
            )
            if isinstance(raw, dict):
                scenarios[label] = {
                    "action": str(raw.get("action", "hold")),
                    "confidence": float(raw.get("confidence", 0.0)),
                    "reason": str(raw.get("reason", "")).strip(),
                }
        except Exception as exc:
            scenarios[label] = {
                "action": "hold",
                "confidence": 0.0,
                "reason": f"RL check unavailable: {exc}",
            }

    return scenarios


def _build_post_trade_feedback(
    *,
    instrument: str,
    direction: str,
    entry_price: float,
    fill_price: float,
    stop_loss_price: float | None,
    take_profit_price: float | None,
    stop_loss_pips: float | None,
    take_profit_pips: float | None,
    confidence: float,
    tcn_probability: float,
    llm_enabled: bool,
    rl_scenarios: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Build post-fill trade guidance, using LLM when available."""
    sl_pips = float(stop_loss_pips) if stop_loss_pips and stop_loss_pips > 0 else 15.0
    tp_pips = float(take_profit_pips) if take_profit_pips and take_profit_pips > 0 else 30.0
    rl_scenarios = rl_scenarios or {}

    rl_now = rl_scenarios.get("now", {})
    rl_good = rl_scenarios.get("if_good", {})
    rl_bad = rl_scenarios.get("if_bad", {})
    now_reason = str(rl_now.get("reason", "")).strip() or "Hold and reassess on next candle close."
    good_reason = str(rl_good.get("reason", "")).strip() or "If momentum fades in profit zone, lock gains and reduce risk."
    bad_reason = str(rl_bad.get("reason", "")).strip() or f"Exit quickly if drawdown approaches {-sl_pips:.1f} pips."

    feedback = {
        "quality": "neutral",
        "summary": now_reason,
        "if_good": good_reason,
        "if_bad": bad_reason,
        "take_profit": (
            f"Take profit partials near +{tp_pips * 0.6:.1f} pips, then trail toward +{tp_pips:.1f} pips TP."
        ),
        "exit": f"Hard exit at -{sl_pips:.1f} pips or if trend structure breaks.",
        "source": "rl",
    }

    if not llm_enabled:
        return feedback

    try:
        from buddy_intelligent_mode import llm_call

        prompt = (
            f"Executed trade details:\n"
            f"- Instrument: {instrument}\n"
            f"- Direction: {direction.upper()}\n"
            f"- Entry expected: {entry_price:.5f}\n"
            f"- Fill: {fill_price:.5f}\n"
            f"- Stop Loss: {stop_loss_price if stop_loss_price is not None else 'N/A'}\n"
            f"- Take Profit: {take_profit_price if take_profit_price is not None else 'N/A'}\n"
            f"- Stop Loss pips: {sl_pips:.1f}\n"
            f"- Take Profit pips: {tp_pips:.1f}\n"
            f"- Model confidence: {confidence:.1f}\n"
            f"- TCN probability: {tcn_probability:.2f}\n"
            f"- RL scenarios: {json.dumps(rl_scenarios, ensure_ascii=True)}\n"
            f"- Baseline guidance: {json.dumps(feedback, ensure_ascii=True)}\n\n"
            "Return JSON only with keys: quality, summary, if_good, if_bad, take_profit, exit."
        )

        llm_response = llm_call(
            prompt=prompt,
            system_prompt=_POST_TRADE_LLM_SYSTEM,
            temperature=0.1,
        )
        parsed = _extract_first_json_object(llm_response or "")
        if isinstance(parsed, dict):
            for key in ("quality", "summary", "if_good", "if_bad", "take_profit", "exit"):
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    feedback[key] = val.strip()
            quality = str(feedback.get("quality", "neutral")).lower()
            feedback["quality"] = quality if quality in {"good", "neutral", "bad"} else "neutral"
            feedback["source"] = "llm+rl"
    except Exception:
        pass

    return feedback


def _print_post_trade_feedback(*, instrument: str, feedback: dict[str, str], indent: str = "") -> None:
    """Render post-fill trade management feedback."""
    source = str(feedback.get("source", "rl"))
    quality = str(feedback.get("quality", "neutral")).upper()
    console.print(f"{indent}[cyan]🤖 Post-trade feedback ({source}) for {instrument}[/cyan]")
    console.print(f"{indent}  Quality: {quality}")
    console.print(f"{indent}  Now: {feedback.get('summary', 'N/A')}")
    console.print(f"{indent}  If good: {feedback.get('if_good', 'N/A')}")
    console.print(f"{indent}  If bad: {feedback.get('if_bad', 'N/A')}")
    console.print(f"{indent}  Take profit: {feedback.get('take_profit', 'N/A')}")
    console.print(f"{indent}  Exit: {feedback.get('exit', 'N/A')}")


def _build_inference_config_from_yaml(
    *,
    cfg: dict[str, Any],
    instrument: str,
    config_path: str,
    equity_override: float | None,
    profile: str = INFERENCE_PROFILE_BALANCED,
    opportunity_mode: bool = False,
) -> tuple[Any, bool]:
    """Build InferenceConfig from YAML only (no numeric fallback literals).

    Returns:
        (InferenceConfig instance, has_pair_override)
    """
    from src.core.modular_inference import InferenceConfig

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config loaded from {config_path}: expected mapping at root")

    inf_section = cfg.get("inference")
    if not isinstance(inf_section, dict):
        raise ValueError(
            f"Missing required 'inference' section in {config_path}. "
            "Define thresholds there to avoid implicit hardcoded defaults."
        )

    by_pair = inf_section.get("by_pair", {})
    if by_pair is not None and not isinstance(by_pair, dict):
        raise ValueError(f"Invalid 'inference.by_pair' in {config_path}: expected mapping")

    normalized_instrument = _normalize_instrument(instrument)
    pair_override = {}
    if isinstance(by_pair, dict):
        raw_pair_cfg = by_pair.get(normalized_instrument) or by_pair.get(instrument)
        if raw_pair_cfg is not None:
            if not isinstance(raw_pair_cfg, dict):
                raise ValueError(
                    f"Invalid 'inference.by_pair.{normalized_instrument}' in {config_path}: expected mapping"
                )
            pair_override = raw_pair_cfg

    # Merge: base inference + pair-specific override
    merged_cfg = {k: v for k, v in inf_section.items() if k != "by_pair"}
    merged_cfg.update(pair_override)

    # Pull volatility-gate wiring from config sections when provided.
    if "use_tcn_volatility_filter" in cfg and "use_tcn_volatility_filter" not in merged_cfg:
        merged_cfg["use_tcn_volatility_filter"] = cfg["use_tcn_volatility_filter"]
    vol_cfg = cfg.get("volatility_regime")
    if isinstance(vol_cfg, dict):
        if "min_regime_for_trade" in vol_cfg and "min_volatility_regime" not in merged_cfg:
            merged_cfg["min_volatility_regime"] = vol_cfg["min_regime_for_trade"]
        if "tcn_required" in vol_cfg and "tcn_required" not in merged_cfg:
            merged_cfg["tcn_required"] = vol_cfg["tcn_required"]

    # Enforce that key trade gates are explicitly configured in YAML.
    required_keys = [
        "min_tcn_probability",
        "min_confidence",
        "min_momentum",
        "max_drawdown_pct",
        "max_streak_prob",
        "risk_per_trade_pct",
        "pip_value",
        "min_stop_loss_pips",
        "min_position_lots",
        "position_sizing_mode",
        "rl_min_risk_utilization",
    ]
    missing = [k for k in required_keys if k not in merged_cfg]
    if missing:
        raise ValueError(
            f"Missing required inference keys in {config_path}: {missing}. "
            "Add them under 'inference:' to keep gating fully config-driven."
        )

    # Runtime equity (live/account override) wins when explicitly provided.
    if equity_override is not None:
        merged_cfg["account_equity"] = float(equity_override)

    # Apply CLI inference profile overrides last so they intentionally win.
    resolved_profile = str(profile).strip().lower() if profile is not None else INFERENCE_PROFILE_BALANCED
    if resolved_profile not in INFERENCE_PROFILE_OVERRIDES:
        valid = ", ".join(sorted(INFERENCE_PROFILE_OVERRIDES.keys()))
        raise ValueError(f"Unknown inference profile '{resolved_profile}'. Valid profiles: {valid}")
    merged_cfg.update(INFERENCE_PROFILE_OVERRIDES[resolved_profile])
    logger.info("Inference profile active: %s", resolved_profile)

    if opportunity_mode:
        merged_cfg.update(INFERENCE_OPPORTUNITY_OVERRIDES)
        logger.info("Inference contract mode active: opportunity")

    # Filter unknown keys to keep config resilient across versions.
    valid_fields = set(InferenceConfig.__dataclass_fields__.keys())
    unknown = sorted(k for k in merged_cfg.keys() if k not in valid_fields)
    if unknown:
        logger.warning("Ignoring unknown inference config keys: %s", unknown)

    kwargs = {k: merged_cfg[k] for k in merged_cfg.keys() if k in valid_fields}
    return InferenceConfig(**kwargs), bool(pair_override)


def _buddy_execute_all_pairs(
    config_path: str,
    *,
    granularity: str = "M5",
    candles: int = 500,
    execute: bool = False,
    force_execute: bool = False,
    force_units: int | None = None,
    equity: float | None = None,
    risk_per_trade_pct: float | None = None,
    verbose: bool = False,
    use_rl_sizer: bool = True,
    trades_today: int = 0,
    max_trades_per_day: int = 30,
    llm_initialized: bool = False,
    llm_enhance: bool = True,
    profile: str = INFERENCE_PROFILE_BALANCED,
) -> None:
    """Execute trades on ALL major pairs that pass gates.

    Scans all major pairs, runs inference, and executes trades on any
    that pass all gates. Respects daily trade limits and correlation limits.
    """
    import numpy as np
    from datetime import datetime

    console.print("\n[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  BUDDY ALL-PAIRS EXECUTION MODE[/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[dim]Scanning {len(_MAJOR_PAIRS)} major pairs for trade opportunities...[/dim]\n")
    console.print("[dim]Pre-trade LLM veto: disabled (fast execution mode)[/dim]\n")

    # Suppress verbose logging
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import logging as _logging
    for name in ['modular_trainers', 'modular_inference', 'tensorflow', 'absl']:
        _logging.getLogger(name).setLevel(_logging.ERROR)

    from src.core.modular_inference import ModularEnsembleInference
    from src.utils.fx_paper import candles_to_ohlcv_df
    from src.utils.oanda_practice import OandaPracticeClient
    from src.data.feature_engineering import FeatureEngineering

    cfg = load_config(config_path)
    client = OandaPracticeClient.from_env()
    fe = FeatureEngineering(cfg.get("feature_engineering", {}))

    # Track results
    passed_pairs = []
    failed_pairs = []
    executed_trades = []
    trades_executed_now = 0
    open_position_units: dict[str, int] = {}

    # Preload current net positions for FIFO-safe execution checks.
    try:
        pos_payload = client.get_open_positions()
        for pos in (pos_payload or {}).get("positions", []) or []:
            if not isinstance(pos, dict):
                continue
            inst = str(pos.get("instrument", "")).strip()
            if not inst:
                continue
            long_units = int(float((pos.get("long") or {}).get("units", 0) or 0))
            short_units = int(float((pos.get("short") or {}).get("units", 0) or 0))
            open_position_units[inst] = long_units + short_units
    except Exception:
        open_position_units = {}

    for pair in _MAJOR_PAIRS:
        # Check trade limit
        if trades_today + trades_executed_now >= max_trades_per_day:
            console.print(f"\n[bold yellow]⚠️ Daily trade limit reached ({max_trades_per_day}). Stopping.[/bold yellow]")
            break

        try:
            # Load pair-specific models if available (check both .keras and .pkl)
            pair_model_dir = Path("trained_data/models") / pair
            has_pair_models = pair_model_dir.exists() and (
                any(pair_model_dir.glob("*.keras")) or any(pair_model_dir.glob("*.pkl"))
            )

            enable_llm_integration = llm_enhance and llm_initialized
            inference_config, _ = _build_inference_config_from_yaml(
                cfg=cfg,
                instrument=pair,
                config_path=config_path,
                equity_override=equity,
                profile=profile,
                opportunity_mode=not execute,
            )
            ensemble = ModularEnsembleInference(
                instrument=pair,
                config=inference_config,
                use_rl_sizer=use_rl_sizer,
                enable_llm_integration=enable_llm_integration,
            )
            ensemble.load_models(instrument=pair)

            # Fetch candles
            resp = client.get_candles(pair, granularity=granularity, count=int(candles), price="MBA")
            df = candles_to_ohlcv_df(resp)

            # Feature engineering
            df = fe.create_features(df, include_all=True)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0.0)

            # Run inference
            result = ensemble.predict_verbose(df, equity=equity, instrument=pair)
            signal = result['raw_signal']

            # Check if gates pass
            if signal.trade:
                adjusted_size = signal.size
                passed_pairs.append({
                    'pair': pair,
                    'direction': signal.direction,
                    'size': adjusted_size,
                    'original_size': signal.size,
                    'confidence': signal.confidence,
                    'result': result,
                    'signal': signal,
                    'ensemble': ensemble,
                    'df': df,
                })
                status = f"[green]✓ PASS[/green] → {signal.direction.upper()} {adjusted_size:.1f} lots"
            else:
                failed_pairs.append({
                    'pair': pair,
                    'reason': signal.reason,
                })
                status = f"[red]✗ FAIL[/red] ({signal.reason[:30]}...)" if len(signal.reason) > 30 else f"[red]✗ FAIL[/red] ({signal.reason})"

            # Show compact status
            model_tag = "[green]●[/green]" if has_pair_models else "[yellow]○[/yellow]"
            console.print(f"  {model_tag} {pair}: {status}")

        except Exception as e:
            failed_pairs.append({'pair': pair, 'reason': str(e)[:50]})
            console.print(f"  [red]✗[/red] {pair}: [red]ERROR[/red] ({str(e)[:40]}...)")

    # Summary
    console.print("\n[bold]━━━ SCAN RESULTS ━━━[/bold]")
    console.print(f"  Passed gates: [green]{len(passed_pairs)}[/green] / {len(_MAJOR_PAIRS)}")
    console.print(f"  Failed gates: [red]{len(failed_pairs)}[/red] / {len(_MAJOR_PAIRS)}")

    if not passed_pairs:
        console.print("\n[yellow]No pairs passed all gates. No trades to execute.[/yellow]")
        return

    # Show pairs that passed
    console.print("\n[bold]━━━ TRADEABLE PAIRS ━━━[/bold]")
    for p in passed_pairs:
        dir_color = "green" if p['direction'] == 'long' else "red"
        console.print(f"  [{dir_color}]{p['direction'].upper():5}[/{dir_color}] {p['pair']}: {p['size']:.1f} lots (conf={p['confidence']:.0f}%)")

    # Execute if enabled
    if not execute:
        console.print("\n[yellow]⚠️ DRY RUN - Add --execute to place trades[/yellow]")
        return

    # Check if FX market is likely open (not weekend)
    now_utc = datetime.now(timezone.utc)
    weekday = now_utc.weekday()  # 0=Mon, 5=Sat, 6=Sun
    hour_utc = now_utc.hour

    # FX market closed: Fri 21:00 UTC to Sun 21:00 UTC (approximately)
    market_closed = (weekday == 5) or (weekday == 6 and hour_utc < 21) or (weekday == 4 and hour_utc >= 21)
    if market_closed:
        console.print("\n[yellow]⚠️ WARNING: FX market is likely closed (weekend)[/yellow]")
        console.print(f"[dim]   Current time: {now_utc.strftime('%Y-%m-%d %H:%M UTC')} (day={weekday})[/dim]")
        console.print("[dim]   Orders may be cancelled with MARKET_HALTED[/dim]\n")

    console.print("\n[bold cyan]━━━ EXECUTING TRADES ━━━[/bold cyan]")

    for p in passed_pairs:
        # Re-check trade limit
        if trades_today + trades_executed_now >= max_trades_per_day:
            console.print("[yellow]Trade limit reached. Skipping remaining pairs.[/yellow]")
            break

        try:
            pair = p['pair']
            signal = p['signal']
            result = p['result']
            pair_ensemble = p.get('ensemble')
            pair_df = p.get('df')

            # Get current price for SL/TP
            resp = client.get_candles(pair, granularity="M1", count=1, price="MBA")
            current_price = float(resp['candles'][0]['mid']['c'])

            # Calculate ATR-based SL/TP
            atr = signal.metadata.get('atr', 0.001)
            pip_size = 0.01 if 'JPY' in pair else 0.0001
            price_precision = 3 if 'JPY' in pair else 5

            # Calculate SL/TP with proper precision
            sl_distance = 1.5 * atr
            tp_distance = 3.0 * atr
            ts_distance = max(atr * 0.9, pip_size * 10)  # Min 10 pips trailing stop
            sl_pips = sl_distance / pip_size
            tp_pips = tp_distance / pip_size

            if signal.direction == 'long':
                sl_price = round(current_price - sl_distance, price_precision)
                tp_price = round(current_price + tp_distance, price_precision)
                units = int(abs(signal.size) * 100000)
            else:
                sl_price = round(current_price + sl_distance, price_precision)
                tp_price = round(current_price - tp_distance, price_precision)
                units = -int(abs(signal.size) * 100000)

            # Round trailing stop distance properly
            ts_distance = round(ts_distance, price_precision)

            # Override units if forced
            if force_units:
                units = force_units if signal.direction == 'long' else -force_units

            # FIFO-safe preflight: don't submit an opposite-direction order
            # against an existing net position for this instrument.
            existing_units = int(open_position_units.get(pair, 0))
            if existing_units != 0 and (existing_units * units) < 0:
                existing_side = "LONG" if existing_units > 0 else "SHORT"
                new_side = "LONG" if units > 0 else "SHORT"
                console.print(
                    f"  [yellow]⏭[/yellow] {pair}: Skipped ({new_side}) due to existing "
                    f"{existing_side} position ({abs(existing_units):,} units) [FIFO safeguard]"
                )
                continue

            # Execute
            order_result = client.create_market_order(
                instrument=pair,
                units=units,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
                trailing_stop_distance=ts_distance,
                # Avoid implicit netting behavior that can trigger FIFO safeguards.
                position_fill="OPEN_ONLY",
            )

            # Debug: show full order result if order fails
            trade_id = order_result.get('orderFillTransaction', {}).get('tradeOpened', {}).get('tradeID')
            if trade_id:
                executed_trades.append({'pair': pair, 'direction': signal.direction, 'trade_id': trade_id})
                trades_executed_now += 1
                open_position_units[pair] = int(open_position_units.get(pair, 0)) + int(units)
                console.print(f"  [green]✓[/green] {pair}: {signal.direction.upper()} {abs(units):,} units (Trade #{trade_id})")

                fill_price_str = order_result.get('orderFillTransaction', {}).get('price')
                try:
                    fill_price = float(fill_price_str) if fill_price_str is not None else float(current_price)
                except Exception:
                    fill_price = float(current_price)

                # Log to journal
                try:
                    from src.utils.trade_journal import TradeJournal
                    journal = TradeJournal()
                    journal.log_trade(
                        trade_id=str(trade_id),
                        instrument=pair,
                        direction=signal.direction,
                        units=abs(units),
                        lots=abs(signal.size),
                        expected_price=current_price,
                        fill_price=fill_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        trailing_stop=ts_distance,
                        atr=atr,
                        tcn_probability=signal.tcn_probability,
                        ridge_confidence=signal.ridge_confidence,
                        xgb_momentum=signal.xgb_momentum,
                        rf_drawdown_pips=signal.rf_drawdown_pips,
                        volatility_regime=getattr(signal, "volatility_regime", None),
                        volatility_regime_name=getattr(signal, "volatility_regime_name", None),
                        volatility_regime_confidence=getattr(signal, "volatility_regime_confidence", None),
                        prediction=signal.tcn_probability,
                        confidence=signal.ridge_confidence,
                        metadata=_build_trade_journal_metadata(signal),
                    )
                except Exception:
                    pass

                rl_scenarios = _compute_rl_exit_scenarios(
                    ensemble=pair_ensemble,
                    df=pair_df,
                    direction=signal.direction,
                    entry_price=float(fill_price),
                    stop_loss_pips=float(sl_pips),
                    take_profit_pips=float(tp_pips),
                )
                post_feedback = _build_post_trade_feedback(
                    instrument=pair,
                    direction=signal.direction,
                    entry_price=float(current_price),
                    fill_price=float(fill_price),
                    stop_loss_price=float(sl_price),
                    take_profit_price=float(tp_price),
                    stop_loss_pips=float(sl_pips),
                    take_profit_pips=float(tp_pips),
                    confidence=float(signal.confidence),
                    tcn_probability=float(signal.tcn_probability),
                    llm_enabled=bool(llm_initialized),
                    rl_scenarios=rl_scenarios,
                )
                _print_post_trade_feedback(
                    instrument=pair,
                    feedback=post_feedback,
                    indent="    ",
                )
            else:
                # Check for order cancellation (e.g., market halted on weekends)
                cancel_tx = order_result.get('orderCancelTransaction', {})
                cancel_reason = cancel_tx.get('reason', '')
                reject_reason = order_result.get('orderRejectTransaction', {}).get('rejectReason', '')
                error_msg = order_result.get('errorMessage', '')

                if cancel_reason == 'MARKET_HALTED':
                    console.print(f"  [yellow]⏸[/yellow] {pair}: Market closed (weekend/holiday)")
                elif cancel_reason:
                    console.print(f"  [red]✗[/red] {pair}: Order cancelled - {cancel_reason}")
                elif reject_reason:
                    console.print(f"  [red]✗[/red] {pair}: Rejected - {reject_reason}")
                elif error_msg:
                    console.print(f"  [red]✗[/red] {pair}: Error - {error_msg}")
                else:
                    console.print(f"  [red]✗[/red] {pair}: Order failed")

        except Exception as e:
            console.print(f"  [red]✗[/red] {pair}: Execution error ({str(e)[:40]})")

    # Final summary
    console.print("\n[bold]━━━ EXECUTION SUMMARY ━━━[/bold]")
    console.print(f"  Trades executed: [green]{len(executed_trades)}[/green]")
    console.print(f"  Total trades today: {trades_today + trades_executed_now}/{max_trades_per_day}")

    if executed_trades:
        console.print(f"\n[dim]Trade IDs: {', '.join(t['trade_id'] for t in executed_trades)}[/dim]")


# =============================================================================
# BUDDY: MAIN INFERENCE COMMAND
# =============================================================================


def buddy(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "USD_JPY",
    granularity: str = "M5",
    candles: int = 500,
    execute: bool = False,
    force_execute: bool = False,
    force_units: int | None = None,
    max_hold_minutes: float | None = None,
    equity: float | None = None,
    risk_per_trade_pct: float | None = None,
    all_features: bool = True,
    verbose: bool = False,
    aggressive_scaling: bool = False,
    use_rl_sizer: bool = True,
    intelligent: bool = False,
    explain: bool = False,
    llm_provider: str = "auto",
    llm_enhance: bool = True,
    profile: str = INFERENCE_PROFILE_BALANCED,
    master_pair: str | None = None,
    interactive_master: bool = True,
) -> None:
    """Buddy: run one TF-only inference on fresh OANDA candles; optionally place a trade.

    Args:
        config_path: Path to config YAML file.
        checkpoint_path: Optional path to specific model checkpoint.
        instrument: Currency pair (e.g., "EUR_USD").
        granularity: OANDA timeframe (e.g., "M5", "H1").
        candles: Number of historical candles to fetch.
        execute: If True, place actual trades via OANDA.
        force_execute: Bypass training accuracy gate.
        force_units: Override position size in units.
        max_hold_minutes: Auto-close trade after N minutes (scalping).
        equity: Override account equity for sizing.
        risk_per_trade_pct: Override risk per trade percentage.
        all_features: Use all engineered features.
        verbose: Enable verbose output.
        aggressive_scaling: Enable $100K→$1M aggressive scaling strategy.
        use_rl_sizer: Use RL-based position sizing.
        intelligent: Enable LLM-enhanced intelligent mode.
        explain: Generate detailed trade reasoning.
        llm_provider: LLM provider (ollama|claude|openai|auto).
        llm_enhance: Enable deep LLM integration.
        master_pair: Optional pair to use as model source for inference.
        interactive_master: Prompt for master-pair model source selection.
    """
    _configure_predict_output(verbose)

    # =========================================================================
    # INTELLIGENT MODE INITIALIZATION
    # =========================================================================
    llm_initialized = False
    llm_validation = None
    llm_approved = True

    if intelligent or explain:
        try:
            from llm_providers import initialize_buddy_llm

            provider_pref = None if llm_provider == "auto" else llm_provider
            provider = initialize_buddy_llm(provider=provider_pref)

            if provider.is_available:
                llm_initialized = True
                _update_buddy_runtime_state(
                    selected_llm_provider=provider.name,
                    intelligent_mode_enabled=bool(intelligent or explain),
                )
                console.print(f"[cyan]🧠 Intelligent Mode: {provider.name.upper()}[/cyan]")
            else:
                _update_buddy_runtime_state(
                    selected_llm_provider=provider.name,
                    intelligent_mode_enabled=False,
                )
                console.print("[yellow]⚠ LLM provider not available - intelligent mode disabled[/yellow]")
                console.print("[dim]Install Ollama (brew install ollama && ollama pull llama3.1:7b) or set ANTHROPIC_API_KEY[/dim]")
                intelligent = False
                explain = False
        except ImportError as e:
            console.print(f"[yellow]⚠ Intelligent mode unavailable: {e}[/yellow]")
            intelligent = False
            explain = False

    # Sync journal with OANDA before backfilling online learning
    try:
        from src.utils.trade_journal import TradeJournal
        from src.utils.oanda_practice import OandaPracticeClient
        journal = TradeJournal()
        client = OandaPracticeClient.from_env()
        sync_results = journal.sync_open_trades(client)
        closed_updated = sync_results.get('closed_updated', 0)
        if closed_updated > 0:
            console.print(f"[dim]🔄 Journal synced: {closed_updated} closed trades[/dim]")
    except Exception:
        pass

    # Seed online learning buffer from closed journal trades
    try:
        from src.utils.trade_journal import TradeJournal
        from market_intelligence import MarketIntelligence
        journal = TradeJournal()
        added = journal.backfill_online_learning()

        try:
            intel = MarketIntelligence(enable_online_learning=True)
            buffer_size = len(intel.online_learner.trade_buffer) if intel.online_learner else 0
            if added > 0:
                console.print(f"[dim]🔄 Online Learning: {buffer_size} trades loaded (+{added} new)[/dim]")
            elif buffer_size > 0:
                console.print(f"[dim]🔄 Online Learning: {buffer_size} trades loaded[/dim]")
        except Exception:
            if added > 0:
                console.print(f"[dim]🔄 Online Learning backfilled: {added} trades[/dim]")
    except Exception:
        pass

    # =========================================================================
    # FETCH LIVE ACCOUNT BALANCE FROM OANDA
    # =========================================================================
    live_nav = None
    trades_today = 0
    max_trades_per_day = 30

    # Always fetch live NAV (even if equity was passed as default $10k from argparser)
    _equity_is_default = (equity is not None and abs(equity - 10_000.0) < 1.0)
    if equity is None or _equity_is_default:
        try:
            from src.utils.oanda_practice import OandaPracticeClient

            client = OandaPracticeClient.from_env()
            account_summary = client.get_account_summary()
            account = account_summary.get('account', {})
            live_nav = float(account.get('NAV', 0))

            # Count trades opened today
            today_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            trades_result = client._request(
                'GET',
                f'/accounts/{client._config.account_id}/trades',
                params={'state': 'ALL', 'count': 100}
            )
            for t in trades_result.get('trades', []):
                if t.get('openTime', '').startswith(today_utc):
                    trades_today += 1

            if live_nav > 0:
                equity = live_nav
                # Balance display now handled in account status header section
                # Check if daily trade limit reached
                trades_remaining = max_trades_per_day - trades_today
                if trades_remaining <= 0:
                    console.print(f"[bold red]⛔ DAILY TRADE LIMIT REACHED ({trades_today}/{max_trades_per_day})[/bold red]")
                    console.print("[dim]Come back tomorrow or increase max_trades_per_day[/dim]")
                    return
        except Exception as e:
            if verbose:
                console.print(f"[dim]Could not fetch live balance: {e}[/dim]")

    # Initialize aggressive scaling engine if enabled
    scaling_engine = None
    if aggressive_scaling:
        try:
            from aggressive_scaling import AggressiveScalingEngine
            scaling_engine = AggressiveScalingEngine(
                starting_capital=equity or 100000.0,
                win_rate=0.78,
                avg_win_pips=41.5,
                avg_loss_pips=24.9,
                slippage_at_10_lots=6.0,
            )
            console.print("\n[bold yellow]⚡ AGGRESSIVE SCALING MODE ACTIVE ⚡[/bold yellow]")
            console.print(f"[yellow]  Kelly Fraction: {scaling_engine.kelly_calculator.kelly_fraction:.1%}[/yellow]")
            console.print(f"[yellow]  Current Phase: {scaling_engine.phase}[/yellow]")
            console.print(f"[yellow]  Phase Kelly Multiplier: {scaling_engine._get_kelly_multiplier():.0%}[/yellow]")
        except ImportError:
            console.print("[red]Warning: aggressive_scaling module not found. Using standard sizing.[/red]")
            scaling_engine = None

    t0 = time.perf_counter()
    last = t0

    def _lap(label: str) -> None:
        nonlocal last
        if not verbose:
            return
        now = time.perf_counter()
        console.print(f"[dim]Timing[/dim]: {label} {now - last:.3f}s (total {now - t0:.3f}s)")
        last = now

    cfg = load_config(config_path)
    _lap("load_config")

    # Treat non-explicit execution as signal-discovery mode.
    # Argparser defaults execute=True, so we inspect argv to distinguish
    # an explicit live-trade request (`-x/--execute`) from plain inference.
    argv_flags = set(sys.argv[1:])
    execute_explicit = any(flag in argv_flags for flag in {"-x", "--execute", "--no-execute", "--dry-run"})
    signal_discovery_mode = (not execute) or (not execute_explicit)

    # =========================================================================
    # HANDLE "ALL" PAIRS MODE
    # =========================================================================
    if instrument.upper() == "ALL":
        _buddy_execute_all_pairs(
            config_path=config_path,
            granularity=granularity,
            candles=candles,
            execute=execute,
            force_execute=force_execute,
            force_units=force_units,
            equity=equity,
            risk_per_trade_pct=risk_per_trade_pct,
            verbose=verbose,
            use_rl_sizer=use_rl_sizer,
            trades_today=trades_today,
            max_trades_per_day=max_trades_per_day,
            llm_initialized=llm_initialized,
            llm_enhance=llm_enhance,
            profile=profile,
        )
        return

    # =========================================================================
    # CHECK FOR MODULAR ENSEMBLE FIRST
    # =========================================================================
    # Check pair-specific meta first, then generic fallback
    _norm_inst = _normalize_instrument(instrument)
    _pair_meta = Path("trained_data") / "models" / _norm_inst / "modular_ensemble.meta.json"
    _generic_meta = Path("trained_data") / "models" / "modular_ensemble.meta.json"
    modular_ensemble_meta_path = _pair_meta if _pair_meta.exists() else _generic_meta

    if modular_ensemble_meta_path.exists() and not checkpoint_path:
        # =====================================================================
        # DISPLAY ACCOUNT STATUS HEADER (before banner)
        # =====================================================================
        console.print("")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]               ACCOUNT STATUS[/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")

        # Display live balance and trade count
        if live_nav and live_nav > 0:
            console.print(f"[green]💰 Live Balance: ${live_nav:,.2f}[/green]")
            trades_remaining = max_trades_per_day - trades_today
            if trades_remaining <= 5:
                console.print(f"[yellow]📊 Trades Today: {trades_today}/{max_trades_per_day} (⚠️ {trades_remaining} remaining)[/yellow]")
            else:
                console.print(f"[cyan]📊 Trades Today: {trades_today}/{max_trades_per_day}[/cyan]")
        else:
            console.print("[dim]💰 Balance: Using default equity[/dim]")
            console.print(f"[dim]📊 Trades Today: {trades_today}/{max_trades_per_day}[/dim]")

        # Display RL Position Sizer status
        if use_rl_sizer:
            console.print("[cyan]🤖 RL Position Sizer: ENABLED[/cyan]")
        else:
            console.print("[dim]🤖 RL Position Sizer: DISABLED[/dim]")

        # Display Intelligent Mode status
        enable_llm_integration = llm_enhance and llm_initialized
        if enable_llm_integration:
            console.print("[cyan]🧠 Intelligent Mode: ENABLED[/cyan]")
        else:
            console.print("[dim]🧠 Intelligent Mode: DISABLED[/dim]")

        console.print("")

        # =====================================================================
        # MODULAR ENSEMBLE INFERENCE BANNER
        # =====================================================================
        console.print("[bold magenta]════════════════════════════════════════════════════════════[/bold magenta]")
        console.print("[bold magenta]  MODULAR ENSEMBLE INFERENCE[/bold magenta]")
        console.print("[bold magenta]════════════════════════════════════════════════════════════[/bold magenta]")

        # Check if instrument was in training set
        meta = json.loads(modular_ensemble_meta_path.read_text())
        trained_pairs = meta.get("selected_pairs", [])
        if trained_pairs and instrument not in trained_pairs:
            console.print(f"\n[bold yellow]⚠️  WARNING: {instrument} not in training set[/bold yellow]")
            console.print(f"[dim]Trained: {', '.join(trained_pairs)}[/dim]")
            console.print(f"[dim]Model may perform differently on {instrument}[/dim]\n")

        # Suppress verbose TF/Metal logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        for name in ['modular_trainers', 'modular_inference', 'tensorflow', 'absl']:
            logging.getLogger(name).setLevel(logging.ERROR)

        from src.core.modular_inference import ModularEnsembleInference

        # Normalize instrument
        normalized_instrument = _normalize_instrument(instrument)
        selected_master_pair = _resolve_master_pair_for_buddy(
            target_pair=normalized_instrument,
            explicit_master_pair=master_pair,
            interactive=interactive_master,
        )

        # Check if pair-specific models exist (check both .keras and .pkl)
        has_target_models = _pair_has_models(normalized_instrument)
        has_master_models = _pair_has_models(selected_master_pair)

        # Display model source
        if selected_master_pair == normalized_instrument and has_target_models:
            console.print(f"[green]📊 Loading {normalized_instrument}-specific models[/green]")
        elif selected_master_pair == normalized_instrument and not has_target_models:
            console.print(
                f"[yellow]📊 Loading generic fallback models (no {normalized_instrument}-specific models found)[/yellow]"
            )
        elif has_master_models:
            console.print(
                f"[green]📊 Loading master pair models: {selected_master_pair} (target: {normalized_instrument})[/green]"
            )
        else:
            console.print(
                f"[yellow]📊 Master pair selected: {selected_master_pair}, but no pair-specific artifacts found; "
                "using generic/joint fallback[/yellow]"
            )

        # Load inference configuration from config file (strict, config-driven).
        # If equity is still the CLI default and no live NAV was fetched, preserve YAML account_equity.
        equity_override = None if (_equity_is_default and live_nav is None) else equity
        inference_config, has_pair_override = _build_inference_config_from_yaml(
            cfg=cfg,
            instrument=normalized_instrument,
            config_path=config_path,
            equity_override=equity_override,
            profile=profile,
            opportunity_mode=signal_discovery_mode,
        )
        console.print(f"[dim]⚙️  Loaded inference config from {config_path}[/dim]")
        console.print(f"[dim]📌 Inference profile: {profile}[/dim]")
        if signal_discovery_mode:
            console.print("[dim]🔎 Contract mode: opportunity (soft model contracts)[/dim]")
        console.print(
            f"[dim]🧭 Heuristic vetoes: "
            f"RSI={'on' if inference_config.use_rsi_extreme_gate else 'off'}, "
            f"TrendContra={'on' if inference_config.use_trend_contra_gate else 'off'}[/dim]"
        )
        if has_pair_override:
            console.print(f"[dim]🎯 Applied pair-specific inference overrides for {normalized_instrument}[/dim]")
        console.print(
            f"[dim]💰 Position sizing: equity=${inference_config.account_equity:,.0f}, "
            f"risk={inference_config.risk_per_trade_pct:.0%}[/dim]"
        )

        ensemble = ModularEnsembleInference(
            instrument=selected_master_pair,
            config=inference_config,
            use_rl_sizer=use_rl_sizer,
            enable_llm_integration=enable_llm_integration,
        )
        ensemble.load_models(instrument=selected_master_pair)

        # Show model timestamps to validate correct model loading
        if hasattr(ensemble, "get_model_load_report"):
            report = ensemble.get_model_load_report()
            if report:
                console.print("[dim]Model load order:[/dim]")
                order = [
                    "direction",
                    "volatility",
                    "ensemble",
                    "gates",
                    "rl_sizer",
                    "rl_gates",
                    "rl_exits",
                    "meta",
                    "model",
                ]
                ordered_items = []
                for category in order:
                    ordered_items.extend([i for i in report if i.get("category") == category])

                for item in ordered_items:
                    trained_at = item.get("trained_at", "unknown")
                    age_days = item.get("age_days")
                    age_str = f"{age_days}d ago" if age_days is not None else "unknown"
                    path_source = item.get("path_source", "unknown")
                    ts_source = item.get("timestamp_source", "unknown")
                    label = item.get("label", "model")
                    path = item.get("path", "")

                    metrics = item.get("metrics") or {}
                    acc = metrics.get("val_accuracy") or metrics.get("best_val_accuracy")
                    metric_str = f" | acc={acc:.1%}" if isinstance(acc, (int, float)) else ""

                    console.print(
                        f"[dim]  {label}: {trained_at} ({age_str}, {ts_source}; {path_source}) | {path}{metric_str}[/dim]"
                    )

        # Fetch candles for inference
        from src.utils.fx_paper import candles_to_ohlcv_df
        from src.utils.oanda_practice import OandaPracticeClient

        client = OandaPracticeClient.from_env()
        resp = client.get_candles(instrument, granularity=granularity, count=int(candles), price="MBA")
        df = candles_to_ohlcv_df(resp)
        _lap("oanda_fetch")

        # Feature engineering
        from src.data.feature_engineering import FeatureEngineering
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(df, include_all=True)
        _lap("feature_engineering")

        # Clean up data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0.0)

        # Run modular ensemble inference
        result = ensemble.predict_verbose(
            df,
            equity=equity,
            instrument=instrument,
            model_instrument=selected_master_pair,
        )
        _lap("modular_inference")

        signal = result['raw_signal']
        intelligent_analysis = _run_buddy_intelligent_analysis(
            instrument=instrument,
            granularity=granularity,
            signal=signal,
            df=df,
            ensemble=ensemble,
            intelligent=intelligent,
            explain=explain,
            llm_initialized=llm_initialized,
        )
        llm_validation = intelligent_analysis.get("validation")
        llm_approved = bool(intelligent_analysis.get("llm_approved", True))
        llm_reasoning = intelligent_analysis.get("reasoning")
        adaptive_policy = intelligent_analysis.get("adaptive_policy")

        # =====================================================================
        # DISPLAY GATE CHECKS SECTION
        # =====================================================================
        console.print("")
        console.print("[bold cyan]Gate Checks:[/bold cyan]")
        for check in result['gate_checks']:
            console.print(f"  {check}")

        if llm_validation is not None:
            console.print("")
            status_color = "green" if llm_validation.approve else "yellow"
            status_text = "APPROVED" if llm_validation.approve else "REJECTED"
            review_line = f"  🧠 Intelligent Review: {status_text}"
            if llm_validation.reason:
                review_line += f" - {llm_validation.reason}"
            console.print(f"[{status_color}]{review_line}[/{status_color}]")
            if llm_validation.approve and llm_validation.size_multiplier != 1.0:
                console.print(
                    f"  [dim]LLM size multiplier: {llm_validation.size_multiplier:.2f}x[/dim]"
                )
            if llm_validation.risk_flags:
                console.print(
                    f"  [dim]Risk flags: {', '.join(str(flag) for flag in llm_validation.risk_flags)}[/dim]"
                )
            adaptive_reason = getattr(adaptive_policy, "reason", None)
            adaptive_samples = getattr(adaptive_policy, "sample_size", 0)
            if adaptive_reason and adaptive_samples:
                console.print(
                    f"  [dim]Adaptive memory ({adaptive_samples} trades): {adaptive_reason}[/dim]"
                )

        if explain and llm_reasoning:
            reasoning_text = llm_reasoning.get("reasoning")
            if reasoning_text:
                console.print("")
                console.print(
                    Panel(
                        reasoning_text,
                        title="[bold cyan]Buddy Reasoning[/bold cyan]",
                        border_style="cyan",
                    )
                )

        # =====================================================================
        # DISPLAY FINAL TRADING DECISION
        # =====================================================================
        final_decision = result['decision']
        if signal.trade and not llm_approved:
            llm_reason = llm_validation.reason if llm_validation and llm_validation.reason else "intelligent_review_reject"
            final_decision = f"→ NO TRADE: {llm_reason}"

        execution_reason = (
            "All model gates passed and intelligent review approved"
            if signal.trade and llm_approved and llm_validation is not None
            else "All model gates passed and Buddy generated a trade signal"
        )

        failed_gates: list[str] = []
        if not signal.trade:
            if not signal.confidence_gate_passed:
                failed_gates.append(f"Ridge confidence ({signal.ridge_confidence:.0f}/100)")
            if not signal.momentum_gate_passed:
                failed_gates.append(f"XGBoost momentum ({signal.xgb_momentum:.2f})")
            if not signal.risk_gate_passed:
                rf_dd_pct = signal.rf_drawdown_pips / 10000 if signal.rf_drawdown_pips > 1.0 else signal.rf_drawdown_pips
                failed_gates.append(f"RF risk (dd={rf_dd_pct:.2%}, streak={signal.rf_streak_prob:.2f})")
            if hasattr(signal, 'meta_gate_passed') and not signal.meta_gate_passed:
                failed_gates.append(f"Meta-labeler ({signal.meta_confidence:.2f})")

        if signal.trade and not llm_approved:
            _update_buddy_runtime_state(
                last_no_trade_decision={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "instrument": instrument,
                    "granularity": granularity,
                    "decision_type": "intelligent_reject",
                    "reason": llm_reason,
                    "final_decision": final_decision,
                    "risk_flags": list(getattr(llm_validation, "risk_flags", []) or []),
                    "adaptive_reason": getattr(adaptive_policy, "reason", None),
                }
            )
        elif not signal.trade:
            _update_buddy_runtime_state(
                last_no_trade_decision={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "instrument": instrument,
                    "granularity": granularity,
                    "decision_type": "gate_failure",
                    "reason": signal.reason,
                    "final_decision": final_decision,
                    "failed_gates": failed_gates,
                }
            )
        elif signal.trade:
            _update_buddy_runtime_state(
                last_trade_execution_decision={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "instrument": instrument,
                    "granularity": granularity,
                    "direction": signal.direction,
                    "decision_type": "execute" if execute else "dry_run_signal",
                    "reason": execution_reason,
                    "final_decision": final_decision,
                    "size_lots": float(signal.size),
                    "confidence": float(signal.confidence),
                    "tcn_probability": float(signal.tcn_probability),
                    "intelligent_review_reason": getattr(llm_validation, "reason", None),
                    "adaptive_reason": getattr(adaptive_policy, "reason", None),
                }
            )

        console.print("")
        console.print(f"[bold]{final_decision}[/bold]")
        console.print("")

        # =====================================================================
        # TRADE EXECUTION OR DRY RUN STATUS
        # =====================================================================
        if signal.trade and not llm_approved:
            # Model gates passed but LLM rejected
            console.print("[bold yellow]╔═══════════════════════════════════════════════════════════╗[/bold yellow]")
            console.print("[bold yellow]║              NO TRADE: LLM VALIDATION FAILED              ║[/bold yellow]")
            console.print("[bold yellow]╚═══════════════════════════════════════════════════════════╝[/bold yellow]")
            console.print("[yellow]Model gates passed but LLM identified risk factors[/yellow]")
            console.print("[dim]The intelligent mode override prevented this trade[/dim]")

        elif signal.trade and execute:
            # Trade signal and execution enabled - execute the trade
            from src.utils.oanda_practice import OandaPracticeClient
            trader = OandaPracticeClient.from_env()

            # Apply LLM size adjustment if available
            size_multiplier = 1.0
            if llm_validation and llm_validation.size_multiplier != 1.0:
                size_multiplier = llm_validation.size_multiplier

            units = int(signal.size * 100000 * size_multiplier)
            if signal.direction == 'short':
                units = -units

            # Calculate TP, SL, TS based on ATR
            current_price = float(df['close'].iloc[-1])
            atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else None

            spread = atr * 0.05 if atr else current_price * 0.0001

            if atr and atr > 0:
                is_jpy = instrument.endswith('_JPY')
                pip_mult = 100 if is_jpy else 10000

                sl_distance = atr * 1.5
                sl_pips = sl_distance * pip_mult

                raw_signal = result.get('raw_signal')
                tcn_prob = raw_signal.tcn_probability if raw_signal else 0.5
                if tcn_prob >= 0.65:
                    tp_distance = atr * 3.0
                    console.print(f"[green]✓ High probability ({tcn_prob:.1%}) - TP extended to 3x ATR[/green]")
                else:
                    tp_distance = atr * 2.0
                tp_pips = tp_distance * pip_mult

                ts_distance = atr * 1.0

                if signal.direction == 'long':
                    sl_price = current_price - sl_distance
                    tp_price = current_price + tp_distance
                else:
                    sl_price = current_price + sl_distance
                    tp_price = current_price - tp_distance

                ts_distance * pip_mult

                console.print(f"[bold yellow]Executing: {signal.direction.upper()} {abs(units):,} units @ {current_price:.5f}[/bold yellow]")
                console.print(f"  [cyan]TP:[/cyan] {tp_price:.5f} (+{tp_pips:.1f} pips = {tp_distance/atr:.1f}x ATR)")
                console.print(f"  [red]SL:[/red] {sl_price:.5f} (-{sl_pips:.1f} pips = {sl_distance/atr:.1f}x ATR)")
                console.print(f"  [magenta]TS:[/magenta] {ts_distance:.5f} ({ts_distance*pip_mult:.1f} pips trailing)")
                console.print(f"  [dim]ATR: {atr:.5f} ({atr*pip_mult:.1f} pips) | R:R = 1:{tp_distance/sl_distance:.2f}[/dim]")

                try:
                    order_result = trader.create_market_order(
                        instrument=instrument,
                        units=units,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        trailing_stop_distance=ts_distance,
                    )
                    fill_tx = order_result.get('orderFillTransaction', {})
                    fill_price_str = fill_tx.get('price', 'N/A')
                    trade_id = fill_tx.get('tradeOpened', {}).get('tradeID')
                    if not trade_id:
                        trades_opened = fill_tx.get('tradesOpened', [])
                        if trades_opened:
                            trade_id = trades_opened[0].get('tradeID')
                    if not trade_id:
                        trade_id = getattr(trader, '_last_trade_id', None)
                    if not trade_id:
                        create_tx = order_result.get('orderCreateTransaction', {})
                        trade_id = create_tx.get('id')
                    if not trade_id:
                        trade_id = 'N/A'

                    if fill_price_str != 'N/A':
                        fill_price_actual = float(fill_price_str)
                        slippage = fill_price_actual - current_price
                        slippage_pips = slippage * pip_mult

                        is_favorable = (signal.direction == 'short' and slippage < 0) or (signal.direction == 'long' and slippage > 0)
                        slip_color = 'green' if abs(slippage_pips) < 1.0 else ('yellow' if abs(slippage_pips) < 3.0 else 'red')
                        slip_sign = '+' if slippage_pips > 0 else ''
                        favorable_str = ' (favorable)' if is_favorable else ''

                        console.print(f"[green]✓ Filled @ {fill_price_str} (Trade #{trade_id})[/green]")
                        console.print(f"  [{slip_color}]Slippage: {slip_sign}{slippage_pips:.1f} pips{favorable_str}[/{slip_color}]")
                    else:
                        fill_price_actual = current_price
                        slippage_pips = 0.0
                        console.print(f"[green]✓ Order filled (Trade #{trade_id})[/green]")

                    # Log trade to journal
                    if trade_id and trade_id != 'N/A':
                        try:
                            from src.utils.trade_journal import TradeJournal
                            journal = TradeJournal()
                            journal.log_trade(
                                trade_id=str(trade_id),
                                instrument=instrument,
                                direction=signal.direction,
                                units=abs(units),
                                lots=signal.size,
                                expected_price=current_price,
                                fill_price=fill_price_actual,
                                stop_loss=sl_price,
                                take_profit=tp_price,
                                trailing_stop=ts_distance,
                                atr=atr,
                                tcn_probability=signal.tcn_probability,
                                ridge_confidence=signal.ridge_confidence,
                                xgb_momentum=signal.xgb_momentum,
                                rf_drawdown_pips=signal.rf_drawdown_pips,
                                feature_vector=[
                                    signal.tcn_probability,
                                    signal.ridge_confidence,
                                    signal.xgb_momentum,
                                    signal.rf_drawdown_pips,
                                ],
                                prediction=signal.tcn_probability,
                                confidence=signal.ridge_confidence,
                                volatility_regime=getattr(signal, "volatility_regime", None),
                                volatility_regime_name=getattr(signal, "volatility_regime_name", None),
                                volatility_regime_confidence=getattr(signal, "volatility_regime_confidence", None),
                                metadata=_build_trade_journal_metadata(signal),
                            )
                            console.print("  [green]✓ Trade logged to journal[/green]")
                        except Exception as je:
                            console.print(f"  [yellow]⚠ Journal error: {je}[/yellow]")

                    rl_scenarios = _compute_rl_exit_scenarios(
                        ensemble=ensemble,
                        df=df,
                        direction=signal.direction,
                        entry_price=float(fill_price_actual),
                        stop_loss_pips=float(sl_pips),
                        take_profit_pips=float(tp_pips),
                    )
                    post_feedback = _build_post_trade_feedback(
                        instrument=instrument,
                        direction=signal.direction,
                        entry_price=float(current_price),
                        fill_price=float(fill_price_actual),
                        stop_loss_price=float(sl_price),
                        take_profit_price=float(tp_price),
                        stop_loss_pips=float(sl_pips),
                        take_profit_pips=float(tp_pips),
                        confidence=float(signal.confidence),
                        tcn_probability=float(signal.tcn_probability),
                        llm_enabled=bool(llm_initialized),
                        rl_scenarios=rl_scenarios,
                    )
                    _print_post_trade_feedback(instrument=instrument, feedback=post_feedback)
                except Exception as e:
                    console.print(f"[red]✗ Order failed: {e}[/red]")
            else:
                console.print(f"[bold yellow]Executing: {signal.direction.upper()} {abs(units):,} units[/bold yellow]")
                console.print("[yellow]⚠ ATR not available - placing order without TP/SL[/yellow]")
                try:
                    order_result = trader.create_market_order(instrument=instrument, units=units)
                    fill_tx = order_result.get('orderFillTransaction', {})
                    fill_price_str = fill_tx.get('price')
                    trade_id = fill_tx.get('tradeOpened', {}).get('tradeID')
                    if fill_price_str is not None:
                        console.print(f"[green]✓ Order placed @ {fill_price_str} (Trade #{trade_id or 'N/A'})[/green]")
                    else:
                        console.print(f"[green]✓ Order placed (Trade #{trade_id or 'N/A'})[/green]")

                    try:
                        fill_price_actual = float(fill_price_str) if fill_price_str is not None else float(current_price)
                    except Exception:
                        fill_price_actual = float(current_price)

                    rl_scenarios = _compute_rl_exit_scenarios(
                        ensemble=ensemble,
                        df=df,
                        direction=signal.direction,
                        entry_price=float(fill_price_actual),
                        stop_loss_pips=None,
                        take_profit_pips=None,
                    )
                    post_feedback = _build_post_trade_feedback(
                        instrument=instrument,
                        direction=signal.direction,
                        entry_price=float(current_price),
                        fill_price=float(fill_price_actual),
                        stop_loss_price=None,
                        take_profit_price=None,
                        stop_loss_pips=None,
                        take_profit_pips=None,
                        confidence=float(signal.confidence),
                        tcn_probability=float(signal.tcn_probability),
                        llm_enabled=bool(llm_initialized),
                        rl_scenarios=rl_scenarios,
                    )
                    _print_post_trade_feedback(instrument=instrument, feedback=post_feedback)
                except Exception as e:
                    console.print(f"[red]✗ Order failed: {e}[/red]")

        elif signal.trade and not execute:
            # Gates passed but execution flag not set - DRY RUN
            console.print("[bold cyan]╔═══════════════════════════════════════════════════════════╗[/bold cyan]")
            console.print("[bold cyan]║              DRY RUN MODE (--execute not set)             ║[/bold cyan]")
            console.print("[bold cyan]╚═══════════════════════════════════════════════════════════╝[/bold cyan]")
            console.print("[green]✓ All gates passed - Trade signal generated[/green]")
            console.print(f"  [cyan]Direction:[/cyan] {signal.direction.upper()}")
            dry_run_size = float(signal.size)
            if llm_validation and llm_validation.approve:
                dry_run_size *= float(llm_validation.size_multiplier)
            console.print(
                f"  [cyan]Position Size:[/cyan] {dry_run_size:.2f} lots ({int(dry_run_size * 100000):,} units)"
            )
            console.print("  [dim]To execute this trade, add --execute flag[/dim]")

        else:
            # Gates failed - no trade signal
            console.print("[bold yellow]╔═══════════════════════════════════════════════════════════╗[/bold yellow]")
            console.print("[bold yellow]║              NO TRADE: GATES FAILED                        ║[/bold yellow]")
            console.print("[bold yellow]╚═══════════════════════════════════════════════════════════╝[/bold yellow]")
            console.print(f"[yellow]Reason: {signal.reason}[/yellow]")

            # Show which specific gates failed
            if failed_gates:
                console.print(f"  [dim]Failed gates: {', '.join(failed_gates)}[/dim]")

        # =====================================================================
        # CLOSING SEPARATOR
        # =====================================================================
        console.print("[bold magenta]════════════════════════════════════════════════════════════[/bold magenta]")
        return

    # =========================================================================
    # FALLBACK TO STANDARD MODEL INFERENCE
    # =========================================================================

    # Load model + preprocessing metadata
    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    candidate_meta_path = Path("trained_data") / "models" / "buddy_tf_candidate.meta.json"

    if checkpoint_path:
        ck_meta = _meta_path_for_checkpoint(str(checkpoint_path))
        if ck_meta and ck_meta.exists():
            meta_path = ck_meta

    if candidate_meta_path.exists():
        meta_path = candidate_meta_path
        console.print("[cyan]Using candidate model[/cyan] (buddy_tf_candidate.keras)")
        console.print("[dim]Tip: Promote with: buddy promote-model[/dim]")
    elif meta_path.exists():
        console.print("[green]Model: buddy_tf.keras (production)[/green]")

    if not meta_path.exists():
        if checkpoint_path:
            raise FileNotFoundError(f"Missing Buddy metadata for checkpoint: {checkpoint_path}")
        raise FileNotFoundError("Missing Buddy metadata. Run: python main.py train-buddy")
    meta = json.loads(meta_path.read_text())
    _lap("load_meta")

    all_features = bool(meta.get("all_features", all_features))
    train_smoothing = bool(meta.get("train_smoothing", True))
    median_window = meta.get("median_window", None)
    median_window = int(median_window) if median_window is not None else None
    min_volume = meta.get("min_volume", None)
    min_volume = int(min_volume) if min_volume is not None else None
    spread_filter = bool(meta.get("spread_filter", False))
    spread_pctl = float(meta.get("spread_pctl", 0.99))
    spread_mult = float(meta.get("spread_mult", 3.0))

    model_path = checkpoint_path or meta.get("model_path")
    if not model_path:
        raise ValueError("Missing model path in metadata")
    seq_len_eff = int(meta.get("seq_len") or 64)
    feature_columns: list[str] = list(meta.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("Missing feature_columns in Buddy metadata")

    std = meta.get("standardize") or {}
    mu = np.asarray(std.get("mean") or [], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(std.get("std") or [], dtype=np.float32).reshape(1, -1)
    if mu.size != len(feature_columns) or sigma.size != len(feature_columns):
        raise ValueError("Standardization stats do not match feature_columns")

    pca_info = meta.get("pca")
    pca_components = int(meta.get("pca_components") or 0)
    pca_mean = None
    pca_components_mat = None
    if pca_info and pca_components > 0:
        pca_mean = np.asarray(pca_info.get("mean") or [], dtype=np.float32).reshape(1, -1)
        pca_components_mat = np.asarray(pca_info.get("components") or [], dtype=np.float32)
        if pca_components_mat.ndim != 2:
            raise ValueError("Invalid PCA components in metadata")

    # Trading gate
    threshold = float(meta.get("live_confidence_threshold", 0.75))
    if execute:
        if force_execute:
            console.print(
                "[yellow]FORCE-EXECUTE[/yellow]: bypassing the training accuracy gate; PRACTICE orders may be placed."
            )
        else:
            enabled, _ = _buddy_live_enabled_from_meta()
            if not enabled:
                console.print(
                    "[yellow]Live trading disabled[/yellow]: train Buddy and reach >=62% direction accuracy."
                )
                execute = False
            else:
                console.print(f"[green]Live trading enabled[/green] (confidence threshold {threshold*100:.0f}%).")

    # Fetch candles
    from src.utils.fx_paper import candles_to_ohlcv_df
    from src.utils.oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    resp = client.get_candles(instrument, granularity=granularity, count=int(candles), price="MBA")
    df = candles_to_ohlcv_df(resp)
    _lap("oanda_fetch")

    if all_features and int(candles) < 1000:
        console.print(
            "[yellow]Warning[/yellow]: low candle lookback can make engineered features degenerate. "
            "Consider re-running with `--candles 2000` or `--candles 5000`."
        )

    # Apply data-quality filters
    if min_volume is not None and "volume" in df.columns:
        try:
            df = df.loc[df["volume"].astype(float) >= float(min_volume)].copy()
        except Exception:
            pass

    if spread_filter and ("bid_close" in df.columns and "ask_close" in df.columns):
        try:
            spread = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs()
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
            if not spread.empty:
                pctl_thr = float(np.quantile(spread.to_numpy(), float(spread_pctl)))
                med = float(np.median(spread.to_numpy()))
                thr = max(pctl_thr, float(spread_mult) * med) if med > 0 else pctl_thr
                keep_mask = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs() <= thr
                df = df.loc[keep_mask].copy()
        except Exception:
            pass

    # Feature engineering
    if all_features:
        from src.data.feature_engineering import FeatureEngineering

        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(
            df,
            include_all=True,
            apply_candle_smoothing=bool(train_smoothing),
            median_window=median_window,
        )
    _lap("feature_engineering")

    numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.ffill().bfill().fillna(0.0)

    # Build feature matrix
    available_cols = set(numeric_df.columns)
    expected_cols = list(feature_columns)
    missing_cols = [c for c in expected_cols if c not in available_cols]
    extra_cols = [c for c in numeric_df.columns if c not in set(expected_cols)]
    if missing_cols:
        console.print(f"[yellow]Inference[/yellow]: {len(missing_cols)} missing features will be filled from training means")
    if extra_cols and verbose:
        console.print(f"[dim]Inference[/dim]: dropping {len(extra_cols)} extra numeric features")

    numeric_df = numeric_df.reindex(columns=expected_cols, fill_value=0.0)
    if missing_cols:
        mu_flat = mu.reshape(-1)
        for idx, col in enumerate(expected_cols):
            if col in set(missing_cols):
                numeric_df[col] = float(mu_flat[idx])

    x2d = numeric_df.to_numpy(dtype=np.float32, copy=False)
    if len(x2d) < seq_len_eff:
        raise ValueError(f"Need at least {seq_len_eff} rows after preprocessing; got {len(x2d)}")
    x2d_full = x2d
    x2d = x2d_full[-seq_len_eff:]
    _lap("build_features")

    # Standardize and optional PCA
    x_std = (x2d - mu) / np.where(sigma < 1e-6, 1.0, sigma)
    x_std = np.clip(x_std, -10.0, 10.0).astype(np.float32, copy=False)
    if pca_components_mat is not None and pca_mean is not None:
        x_std = (x_std - pca_mean) @ pca_components_mat.T
    _lap("standardize_pca")

    if verbose:
        try:
            zero_frac = float(np.mean((np.abs(x2d) < 1e-12).astype(np.float32)))
            console.print(
                f"[dim]Inference diag[/dim]: seq_len={seq_len_eff} feat_dim={x2d.shape[1]} zeros={zero_frac*100:.1f}% "
                f"x_std[min/mean/max]=({float(x_std.min()):.3f}/{float(x_std.mean()):.3f}/{float(x_std.max()):.3f})"
            )
        except Exception:
            pass

    x_std.reshape(1, seq_len_eff, -1)

    # Load TensorFlow and model
    from cli.tf_config import _configure_tf_metal
    _configure_tf_metal(verbose=verbose)

    import tensorflow as tf

    from ml_head_engine import MLEngineHead
    from mr_engine import MREngineHead
    from ms_head_engine import MSEngineHead
    from mt_engine import MTEngineHead
    from mx_head_engine import MXEngineHead

    custom_objects = {
        "MLEngineHead": MLEngineHead,
        "MREngineHead": MREngineHead,
        "MSEngineHead": MSEngineHead,
        "MTEngineHead": MTEngineHead,
        "MXEngineHead": MXEngineHead,
    }

    # Load model with fallback strategies
    model = None
    load_errors = []

    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        if verbose:
            console.print("[green]✓ Model loaded (standard)[/green]")
    except Exception as e1:
        load_errors.append(f"Standard: {str(e1)[:100]}")
        if verbose:
            console.print(f"[dim]Strategy 1 failed: {str(e1)[:100]}[/dim]")

        try:
            model = tf.keras.models.load_model(
                model_path, compile=False, custom_objects=custom_objects, safe_mode=False
            )
            if verbose:
                console.print("[green]✓ Model loaded (safe_mode=False)[/green]")
        except Exception as e2:
            load_errors.append(f"SafeMode=False: {str(e2)[:100]}")
            if verbose:
                console.print(f"[dim]Strategy 2 failed: {str(e2)[:100]}[/dim]")

            try:
                model = tf.keras.saving.load_model(
                    model_path, compile=False, custom_objects=custom_objects
                )
                if verbose:
                    console.print("[green]✓ Model loaded (tf.keras.saving)[/green]")
            except Exception as e3:
                load_errors.append(f"tf.keras.saving: {str(e3)[:100]}")
                if verbose:
                    console.print(f"[dim]Strategy 3 failed: {str(e3)[:100]}[/dim]")

                try:
                    feature_dim = len(feature_columns)
                    seq_len_meta = int(meta.get("seq_len", 60))
                    model_type = meta.get("model_type", "tcn")

                    console.print(f"[yellow]Attempting to rebuild {model_type} model from metadata...[/yellow]")

                    model = _build_buddy_model_for_type(
                        model_type=model_type,
                        feature_dim=feature_dim,
                        seq_len=seq_len_meta,
                    )

                    weights_path = str(model_path).replace(".keras", "_weights.h5")
                    if Path(weights_path).exists():
                        model.load_weights(weights_path)
                        console.print("[green]✓ Model rebuilt and weights loaded[/green]")
                    else:
                        import h5py
                        with h5py.File(model_path, 'r') as f:
                            if 'model_weights' in f:
                                model.load_weights(model_path, by_name=True)
                                console.print("[green]✓ Model rebuilt and weights loaded from .keras[/green]")
                            else:
                                raise ValueError("No weights found in model file")
                except Exception as e4:
                    load_errors.append(f"Weights-only: {str(e4)[:100]}")
                    console.print("[red]Failed to load model after 4 attempts[/red]")
                    console.print("[yellow]Load errors:[/yellow]")
                    for i, err in enumerate(load_errors, 1):
                        console.print(f"  {i}. {err}")
                    raise RuntimeError(
                        f"Failed to load model from {model_path}. "
                        "This is a Keras version mismatch."
                    ) from e4

    if model is None:
        raise RuntimeError("Model loading failed - model is None")

    _lap("tf_load_model")

    _inputs_struct = getattr(model, "_inputs_struct", None)
    _expects_features_dict = isinstance(_inputs_struct, dict) and "features" in _inputs_struct

    def _model_call(x_in: np.ndarray):
        return model({"features": x_in}, training=False) if _expects_features_dict else model(x_in, training=False)

    # Ensemble over last N windows
    n_ens = int(meta.get("inference_ensemble_windows", 20) or 20)
    n_ens = max(1, min(n_ens, 100))

    conf_agg = str(meta.get("inference_confidence_agg", "quantile") or "quantile").strip().lower()
    if conf_agg not in {"mean", "max", "quantile"}:
        conf_agg = "quantile"
    conf_q = float(meta.get("inference_confidence_quantile", 0.90) or 0.90)
    conf_q = float(min(1.0, max(0.50, conf_q)))

    def _predict_one(x_one: np.ndarray) -> tuple[float, float]:
        pred_one = _model_call(x_one)
        d = pred_one["direction"]
        c = pred_one["confidence"]
        if hasattr(d, "numpy"):
            d = d.numpy()
        if hasattr(c, "numpy"):
            c = c.numpy()
        return (
            float(np.asarray(d).reshape(-1)[0]),
            float(np.asarray(c).reshape(-1)[0]),
        )

    p_dir_last = None
    p_conf_last = None
    p_dir = None
    p_conf = None
    try:
        x_std_full = (x2d_full - mu) / np.where(sigma < 1e-6, 1.0, sigma)
        x_std_full = np.clip(x_std_full, -10.0, 10.0).astype(np.float32, copy=False)
        if pca_components_mat is not None and pca_mean is not None:
            x_std_full = (x_std_full - pca_mean) @ pca_components_mat.T

        need = seq_len_eff + n_ens - 1
        if n_ens > 1 and len(x_std_full) >= need:
            tail = x_std_full[-need:]
            windows = np.stack([tail[i : i + seq_len_eff] for i in range(n_ens)], axis=0).astype(np.float32)
            pred_batch = _model_call(windows)
            d = pred_batch["direction"]
            c = pred_batch["confidence"]
            if hasattr(d, "numpy"):
                d = d.numpy()
            if hasattr(c, "numpy"):
                c = c.numpy()
            dir_probs = np.asarray(d).reshape(-1)
            conf_probs = np.asarray(c).reshape(-1)
            conf_raw = conf_probs
            conf_probs = np.clip(conf_probs, 0.0, 1.0)
            if verbose:
                try:
                    raw = np.asarray(conf_raw, dtype=np.float64).reshape(-1)
                    console.print(
                        "[dim]Inference diag[/dim]: conf_raw[min/mean/max]="
                        f"({float(raw.min()):.6g}/{float(raw.mean()):.6g}/{float(raw.max()):.6g}) "
                        f"pct<0={float(np.mean(raw < 0.0))*100:.1f}% pct>1={float(np.mean(raw > 1.0))*100:.1f}%"
                    )
                except Exception:
                    pass
            p_dir_last = float(dir_probs[-1])
            p_conf_last = float(conf_probs[-1])
            p_dir = float(np.mean(dir_probs))
            if conf_agg == "mean":
                p_conf = float(np.mean(conf_probs))
            elif conf_agg == "max":
                p_conf = float(np.max(conf_probs))
            else:
                p_conf = float(np.quantile(conf_probs, conf_q))
            if verbose:
                console.print(
                    f"[dim]Inference ensemble[/dim]: windows={n_ens} "
                    f"dir_mean={p_dir:.6g} conf_{conf_agg}={p_conf:.6g} dir_last={p_dir_last:.6g} conf_last={p_conf_last:.6g}"
                )
        else:
            x_one = x_std.reshape(1, seq_len_eff, -1)
            p_dir, p_conf = _predict_one(x_one)
            p_dir_last, p_conf_last = p_dir, p_conf
    except Exception:
        x_one = x_std.reshape(1, seq_len_eff, -1)
        p_dir, p_conf = _predict_one(x_one)
        p_dir_last, p_conf_last = p_dir, p_conf

    if p_dir is None or p_conf is None:
        x_one = x_std.reshape(1, seq_len_eff, -1)
        p_dir, p_conf = _predict_one(x_one)
        p_dir_last, p_conf_last = p_dir, p_conf

    _lap("inference")

    try:
        p_conf = float(np.clip(float(p_conf), 0.0, 1.0))
        p_conf_last = float(np.clip(float(p_conf_last), 0.0, 1.0))
    except Exception:
        pass

    def _fmt_prob(p: float) -> str:
        if p <= 1e-12 or p >= 1.0 - 1e-12 or p < 1e-4 or p > 1.0 - 1e-4:
            return f"{p:.6e}"
        return f"{p:.6f}"

    side = "buy" if p_dir >= 0.5 else "sell"

    # =========================================================================
    # PREDICTION SUMMARY
    # =========================================================================
    dir_strength = abs(p_dir - 0.5) * 200
    signal_emoji = "🟢" if side == "buy" else "🔴"

    console.print("\n" + "=" * 60)
    console.print(f"[bold]{signal_emoji} BUDDY SIGNAL: {side.upper()}[/bold]")
    console.print("=" * 60)

    console.print(f"  Instrument:     {instrument}")
    console.print(f"  Timeframe:      {granularity}")
    console.print(f"  Direction:      {p_dir:.1%} {'(LONG)' if side == 'buy' else '(SHORT)'}")
    console.print(f"  Signal Strength: {dir_strength:.1f}%")
    console.print(f"  Model Conf:     {p_conf:.1%}")

    try:
        if "adx" in df.columns:
            adx_val = float(df["adx"].iloc[-1])
            is_trending = adx_val > 25
            regime = "TRENDING" if is_trending else "RANGING"
            regime_color = "green" if is_trending else "yellow"
            console.print(f"  Market Regime:  [{regime_color}]{regime}[/{regime_color}] (ADX={adx_val:.1f})")
    except Exception:
        pass

    console.print("-" * 60)

    # Tier-2 calibrated win probability
    tier2_cfg = (cfg.get("buddy", {}) or {}).get("tier2", {}) or {}
    tier2_enabled = bool(tier2_cfg.get("enabled", False))
    tier2_p_win = None
    if tier2_enabled:
        try:
            dir_conf = float(p_dir) if side == "buy" else float(1.0 - float(p_dir))
            score = float(max(0.0, min(1.0, dir_conf * float(p_conf))))
            tier2_p_win = _tier2_apply_calibration(meta, score)
        except Exception:
            tier2_p_win = None

    # Tier-1 confidence gate
    tier1_conf = None
    tier1_reasons: list[str] | None = None
    try:
        from reasoning_enhanced import fx_confidence_v1
        from src.utils.fx_paper import atr as atr_from_df
        from src.utils.fx_paper import spread_pips_from_df

        last_px = float(df["close"].iloc[-1])
        atr_val = float(atr_from_df(df, period=14))
        sp_pips = spread_pips_from_df(df, instrument)
        if sp_pips is None:
            sp_pips = (
                cfg.get("fx", {})
                .get("costs", {})
                .get("spread_fallback_pips", {})
                .get(instrument)
            )
        max_spread = (
            cfg.get("fx", {})
            .get("costs", {})
            .get("max_spread_pips", {})
            .get(instrument)
        )

        payload = fx_confidence_v1(
            signal=side,
            price=last_px,
            atr=atr_val,
            spread_pips=float(sp_pips) if sp_pips is not None else float("nan"),
            max_spread_pips=float(max_spread) if max_spread is not None else float("nan"),
            params=cfg.get("fx", {}).get("confidence_model", {}),
        )
        tier1_conf = float(payload.get("confidence", 0.0) or 0.0)
        r = payload.get("reasons")
        if isinstance(r, list):
            tier1_reasons = [str(x) for x in r]
    except Exception:
        tier1_conf = None
        tier1_reasons = None

    gate_use_model = bool(tier2_cfg.get("use_model_confidence_for_gate", False))
    model_threshold_from_tier2 = tier2_cfg.get("model_conf_threshold", None)

    if tier2_p_win is not None and not gate_use_model:
        gate_conf = float(tier2_p_win)
        threshold = float(tier2_cfg.get("p_win_threshold", 0.60))
    else:
        gate_conf = float(p_conf)
        if model_threshold_from_tier2 is not None:
            threshold = float(model_threshold_from_tier2)
        elif tier1_conf is not None:
            threshold = float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))
        else:
            threshold = float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))

    # Display Tier-2 if available
    if tier2_p_win is not None:
        t2_color = "green" if tier2_p_win >= 0.50 else "yellow" if tier2_p_win >= 0.40 else "red"
        console.print(f"  [bold]Win Prob (Tier-2): [{t2_color}]{tier2_p_win:.1%}[/{t2_color}][/bold]")

        stop_loss_pips = float(cfg.get("buddy", {}).get("stop_loss_pips", 12.0))
        take_profit_pips = float(cfg.get("buddy", {}).get("take_profit_pips", 24.0))
        ev = (tier2_p_win * take_profit_pips) - ((1 - tier2_p_win) * stop_loss_pips)
        ev_color = "green" if ev > 0 else "red"
        console.print(f"  Expected Value:  [{ev_color}]{ev:+.1f} pips[/{ev_color}]")
    elif tier1_conf is not None:
        console.print(f"  Tier-1 Conf:     {tier1_conf:.1%}")

    console.print("-" * 60)

    if verbose:
        console.print("[dim]Detailed Diagnostics:[/dim]")
        conf_msg = f"model_conf={_fmt_prob(p_conf)}"
        if tier1_conf is not None:
            conf_msg += f" tier1_conf={_fmt_prob(gate_conf)}"
        if tier2_p_win is not None:
            conf_msg += f" tier2_p_win={_fmt_prob(float(tier2_p_win))}"
        console.print(f"[dim]  Raw values: direction={_fmt_prob(p_dir)} ({side}), {conf_msg}[/dim]")
        if tier1_reasons:
            console.print(f"[dim]  Tier-1 reasons: {', '.join(tier1_reasons)}[/dim]")

    # Confidence gate
    if bool(tier2_cfg.get("remove_confidence_gate", False)):
        if verbose:
            console.print("[yellow]Confidence gate disabled by config (remove_confidence_gate=True)[/yellow]")
    else:
        require_both = bool(tier2_cfg.get("require_both", False))
        if require_both and tier2_p_win is not None:
            model_thr = float(model_threshold_from_tier2) if model_threshold_from_tier2 is not None else float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))
            p_win_thr = float(tier2_cfg.get("p_win_threshold", 0.60))
            p_ok = float(p_conf) >= model_thr
            tier2_ok = float(tier2_p_win) >= p_win_thr
            if not (p_ok and tier2_ok):
                console.print(f"[yellow]No trade[/yellow]: requires model_conf >= {model_thr:.2f} AND tier2_p_win >= {p_win_thr:.2f}")
                return
        else:
            if tier2_p_win is not None and not gate_use_model:
                gate_conf = float(tier2_p_win)
                threshold = float(tier2_cfg.get("p_win_threshold", 0.60))
            else:
                gate_conf = float(p_conf)
                if model_threshold_from_tier2 is not None:
                    threshold = float(model_threshold_from_tier2)
                elif tier1_conf is not None:
                    threshold = float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))
                else:
                    threshold = float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))

            if gate_conf < threshold:
                console.print(f"[yellow]No trade[/yellow]: confidence below {threshold:.2f}")
                return

    if not execute:
        console.print("[cyan]DRY-RUN[/cyan]: would place a market order.")
        return

    # Place a conservative market order with SL/TP
    last_close = float(df["close"].iloc[-1])
    from src.utils.fx_paper import pip_size, position_size_units

    ps = float(pip_size(instrument))
    stop_loss_pips = float(cfg.get("buddy", {}).get("stop_loss_pips", 20.0))
    take_profit_pips = float(cfg.get("buddy", {}).get("take_profit_pips", 40.0))
    stop_distance = stop_loss_pips * ps

    equity_eff = float(
        equity
        if equity is not None
        else cfg.get("buddy", {}).get("equity", cfg.get("fx", {}).get("equity", 10_000.0))
    )

    # Aggressive scaling sizing
    risk_eff = None
    use_smart_execution = False

    if scaling_engine is not None:
        try:
            can_trade, block_reason = scaling_engine.risk_manager.can_trade()
            if not can_trade:
                console.print(f"[red]🛑 CIRCUIT BREAKER: {block_reason}[/red]")
                return

            position_lots = scaling_engine.calculate_position_size(
                equity=equity_eff,
                atr_pips=stop_loss_pips,
            )

            units_from_scaling = int(position_lots * 100000)
            kelly_mult = scaling_engine._get_kelly_multiplier()
            kelly_fraction = scaling_engine.kelly_calculator.kelly_fraction
            slippage_adj = scaling_engine.kelly_calculator.slippage_adjustment
            risk_eff = kelly_fraction * slippage_adj * kelly_mult
            risk_eff = min(risk_eff, 0.03)
            use_smart_execution = position_lots >= 5.0

            if verbose:
                console.print("[yellow]Aggressive Scaling:[/yellow]")
                console.print(f"  Kelly Fraction: {kelly_fraction:.1%}")
                console.print(f"  Slippage Adjustment: {slippage_adj:.1%}")
                console.print(f"  Phase Multiplier: {kelly_mult:.0%}")
                console.print(f"  Effective Risk: {risk_eff:.2%}")
                console.print(f"  Position Size: {position_lots:.2f} lots ({units_from_scaling:,} units)")
                if use_smart_execution:
                    console.print("  [cyan]Smart Execution: Will split large order[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Warning: Aggressive scaling error: {e}. Using standard sizing.[/yellow]")
            risk_eff = None

    # Tier-2 Kelly sizing fallback
    if risk_eff is None and tier2_p_win is not None:
        try:
            stop_loss_pips_eff = float(cfg.get("buddy", {}).get("stop_loss_pips", 20.0))
            take_profit_pips_eff = float(cfg.get("buddy", {}).get("take_profit_pips", 60.0))
            b = float(take_profit_pips_eff / max(stop_loss_pips_eff, 1e-9))
            p = float(max(0.0, min(1.0, float(tier2_p_win))))
            q = 1.0 - p
            kelly = ((b * p) - q) / max(b, 1e-9)
            kelly_fraction = float(tier2_cfg.get("kelly_fraction", 0.5))
            cap = float(tier2_cfg.get("max_risk_per_trade_pct", 0.02))
            risk_eff = float(max(0.0, min(cap, kelly * kelly_fraction)))
        except Exception:
            risk_eff = None

    if risk_eff is None:
        risk_eff = float(
            risk_per_trade_pct
            if risk_per_trade_pct is not None
            else cfg.get("buddy", {}).get(
                "risk_per_trade_pct",
                cfg.get("fx", {}).get("risk", {}).get("risk_per_trade_pct", 0.005),
            )
        )
    units = position_size_units(
        instrument=instrument,
        equity=equity_eff,
        risk_per_trade_pct=risk_eff,
        stop_distance_price=stop_distance,
        price=last_close,
    )
    if side == "sell":
        units = -abs(int(units))
    else:
        units = abs(int(units))

    if force_units is not None:
        try:
            fu = int(force_units)
            units = -abs(int(fu)) if side == "sell" else abs(int(fu))
            if verbose:
                console.print(f"[dim]Force-units override[/dim]: using units={units}")
        except Exception:
            pass

    stop_price = last_close - stop_distance if units > 0 else last_close + stop_distance
    tp_distance = take_profit_pips * ps
    tp_price = last_close + tp_distance if units > 0 else last_close - tp_distance

    try:
        units_final = int(units)
    except Exception:
        units_final = int(float(units))

    if verbose:
        console.print(f"[dim]Order sizing[/dim]: computed_units={units} final_submitted_units={units_final}")

    # Smart order execution for aggressive scaling
    if scaling_engine is not None and use_smart_execution and abs(units_final) >= 500000:
        try:
            from aggressive_scaling import SmartOrderExecutor
            executor = SmartOrderExecutor(atr_pips=stop_loss_pips, spread_pips=1.5)

            split_orders = executor.split_order(
                total_lots=abs(units_final) / 100000,
                current_price=last_close,
                side=side,
            )

            console.print(f"[cyan]Smart Execution: Splitting into {len(split_orders)} orders[/cyan]")

            total_filled = 0
            for i, order in enumerate(split_orders, 1):
                order_units = int(order['size'] * 100000)
                if side == "sell":
                    order_units = -abs(order_units)

                price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=order_units)
                result = client.create_market_order(
                    instrument=instrument,
                    units=order_units,
                    stop_loss_price=float(stop_price),
                    take_profit_price=float(tp_price),
                    price_bound=price_bound,
                    client_tag="buddy_tf_split",
                )
                total_filled += abs(order_units)
                console.print(f"  [green]Order {i}/{len(split_orders)}[/green]: {order_units:,} units - {result}")

                if i < len(split_orders):
                    time.sleep(0.5)

            console.print(f"[green]Total filled[/green]: {total_filled:,} units")

            scaling_engine.record_trade_result(
                win=True,
                pips=0,
                lots=total_filled / 100000,
            )
        except Exception as e:
            console.print(f"[yellow]Smart execution failed: {e}. Using standard order.[/yellow]")
            price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=units_final)
            result = client.create_market_order(
                instrument=instrument,
                units=units_final,
                stop_loss_price=float(stop_price),
                take_profit_price=float(tp_price),
                price_bound=price_bound,
                client_tag="buddy_tf",
            )
            console.print(f"[green]Order submitted[/green]: {result}")
    else:
        # Standard order execution
        price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=units_final)
        result = client.create_market_order(
            instrument=instrument,
            units=units_final,
            stop_loss_price=float(stop_price),
            take_profit_price=float(tp_price),
            price_bound=price_bound,
            client_tag="buddy_tf",
        )
        console.print(f"[green]Order submitted[/green]: {result}")

        if scaling_engine is not None:
            try:
                scaling_engine.record_trade_result(
                    win=True,
                    pips=0,
                    lots=abs(units_final) / 100000,
                )
            except Exception:
                pass

    # Schedule auto-close if requested
    try:
        if max_hold_minutes is not None:
            m = float(max_hold_minutes)
            if m > 0:
                _schedule_auto_close(client, instrument, delay_s=m * 60.0, verbose=bool(verbose))
    except Exception:
        pass


def buddy_loop(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "USD_JPY",
    granularity: str = "M5",
    candles: int = 500,
    execute: bool = False,
    force_execute: bool = False,
    force_units: int | None = None,
    equity: float | None = None,
    risk_per_trade_pct: float | None = None,
    verbose: bool = False,
    max_trades: int = 1,
    **kwargs: Any,
) -> None:
    """Buddy loop: keep TF+model warm; trade once per new candle (practice only)."""
    _configure_predict_output(verbose)

    max_hold_minutes = None
    try:
        m = kwargs.get("max_hold_minutes")
        max_hold_minutes = float(m) if m is not None else None
    except Exception:
        max_hold_minutes = None

    all_features = bool(kwargs.get("all_features", True))

    from collections import deque
    import json
    import math
    import numpy as np
    import pandas as pd  # noqa: F811
    from src.utils.tracing_setup import span, record_exception

    from datetime import datetime, timezone

    def _granularity_seconds(g: str) -> int:
        s = str(g).strip().upper()
        if s.startswith("M") and s[1:].isdigit():
            return int(s[1:]) * 60
        if s.startswith("H") and s[1:].isdigit():
            return int(s[1:]) * 3600
        if s == "D":
            return 86400
        # fallback
        return 300

    def _sleep_until_next_boundary(interval_s: int, *, lag_s: float = 2.0) -> None:
        # Align to the next candle close boundary in UTC (best-effort).
        now = time.time()
        next_boundary = (math.floor(now / interval_s) + 1) * interval_s
        sleep_s = max(0.0, (next_boundary + float(lag_s)) - now)
        if verbose:
            eta = datetime.fromtimestamp(next_boundary, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            console.print(f"[dim]Loop[/dim]: waiting {sleep_s:.1f}s for next {granularity} boundary @ {eta}")
        time.sleep(sleep_s)

    cfg = load_config(config_path)

    equity_eff = float(
        equity
        if equity is not None
        else cfg.get("buddy", {}).get("equity", cfg.get("fx", {}).get("equity", 10_000.0))
    )
    risk_eff = float(
        risk_per_trade_pct
        if risk_per_trade_pct is not None
        else cfg.get("buddy", {}).get(
            "risk_per_trade_pct",
            cfg.get("fx", {}).get("risk", {}).get("risk_per_trade_pct", 0.005),
        )
    )

    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    if checkpoint_path:
        ck_meta = _meta_path_for_checkpoint(str(checkpoint_path))
        if ck_meta and ck_meta.exists():
            meta_path = ck_meta
    if not meta_path.exists():
        raise FileNotFoundError("Missing Buddy metadata. Run: python main.py train-buddy")
    meta = json.loads(meta_path.read_text())

    tier2_cfg = (cfg.get("buddy", {}) or {}).get("tier2", {}) or {}
    tier2_enabled = bool(tier2_cfg.get("enabled", False))

    # Match inference preprocessing to how the model was trained.
    all_features = bool(meta.get("all_features", all_features))
    train_smoothing = bool(meta.get("train_smoothing", True))
    median_window = meta.get("median_window", None)
    median_window = int(median_window) if median_window is not None else None
    min_volume = meta.get("min_volume", None)
    min_volume = int(min_volume) if min_volume is not None else None
    spread_filter = bool(meta.get("spread_filter", False))
    spread_pctl = float(meta.get("spread_pctl", 0.99))
    spread_mult = float(meta.get("spread_mult", 3.0))

    model_path = checkpoint_path or meta.get("model_path")
    if not model_path:
        raise ValueError("Missing model path in metadata")

    seq_len_eff = int(meta.get("seq_len") or 64)
    feature_columns: list[str] = list(meta.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("Missing feature_columns in Buddy metadata")

    std = meta.get("standardize") or {}
    mu = np.asarray(std.get("mean") or [], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(std.get("std") or [], dtype=np.float32).reshape(1, -1)
    if mu.size != len(feature_columns) or sigma.size != len(feature_columns):
        raise ValueError("Standardization stats do not match feature_columns")

    pca_info = meta.get("pca")
    pca_components = int(meta.get("pca_components") or 0)
    pca_mean = None
    pca_components_mat = None
    if pca_info and pca_components > 0:
        pca_mean = np.asarray(pca_info.get("mean") or [], dtype=np.float32).reshape(1, -1)
        pca_components_mat = np.asarray(pca_info.get("components") or [], dtype=np.float32)
        if pca_components_mat.ndim != 2:
            raise ValueError("Invalid PCA components in metadata")

    # Live trading gate.
    threshold = float(meta.get("live_confidence_threshold", 0.75))
    if execute:
        if force_execute:
            console.print(
                "[yellow]FORCE-EXECUTE[/yellow]: bypassing the training accuracy gate; PRACTICE orders may be placed."
            )
        else:
            enabled, _ = _buddy_live_enabled_from_meta()
            if not enabled:
                console.print("[yellow]Live trading disabled[/yellow]: train Buddy and reach >=62% direction accuracy.")
                execute = False
            else:
                console.print(f"[green]Live trading enabled[/green] (confidence threshold {threshold*100:.0f}%).")

    # Load TF/model once (warm process).
    _configure_tf_metal(verbose=verbose)
    import tensorflow as tf

    from ml_head_engine import MLEngineHead
    from mr_engine import MREngineHead
    from ms_head_engine import MSEngineHead
    from mt_engine import MTEngineHead
    from mx_head_engine import MXEngineHead

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "MLEngineHead": MLEngineHead,
            "MREngineHead": MREngineHead,
            "MSEngineHead": MSEngineHead,
            "MTEngineHead": MTEngineHead,
            "MXEngineHead": MXEngineHead,
        },
    )

    from src.utils.fx_paper import candles_to_ohlcv_df
    from src.utils.oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    interval_s = _granularity_seconds(granularity)

    def _to_utc_iso_z(ts: pd.Timestamp) -> str:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat().replace("+00:00", "Z")

    def _normalize_candle_df(df_in: "pd.DataFrame") -> "pd.DataFrame":
        out = df_in
        if out is None or len(out) == 0:
            return pd.DataFrame()
        if "time" not in out.columns:
            return pd.DataFrame()
        out = out.copy()
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
        out = out.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
        return out

    # --- Streaming candle buffer (bounded, no persistence) ---
    candle_buf = deque(maxlen=int(candles))
    last_seen_ts: pd.Timestamp | None = None
    last_seen_from_time: str | None = None

    # Warm start the deque with the last `candles` candles.
    try:
        with span("warm_get_candles", attributes={"instrument": instrument, "granularity": granularity}):
            warm_resp = client.get_candles(instrument, granularity=granularity, count=int(candles), price="MBA")
        warm_df = _normalize_candle_df(candles_to_ohlcv_df(warm_resp))
        if len(warm_df) > 0:
            candle_buf.extend(warm_df.to_dict("records"))
            last_seen_ts = pd.to_datetime(warm_df["time"].iloc[-1], utc=True, errors="coerce")
            if pd.notna(last_seen_ts):
                last_seen_from_time = _to_utc_iso_z(pd.Timestamp(last_seen_ts))
    except Exception:
        try:
            record_exception(Exception("warm_fetch_failed"))
        except Exception:
            pass

    last_candle_id: str | None = None
    primed = False
    trades_done = 0
    # Simple trace counter for debugging / liveness checks.
    trace_iter = 0

    # =========================================================================
    # ONLINE RETRAINER: Connect drift detection to model retraining
    # =========================================================================
    market_intel = None
    try:
        from market_intelligence import MarketIntelligence
        market_intel = MarketIntelligence(
            enable_sentiment=False,
            enable_calendar=False,
            enable_online_learning=True,
            enable_drift_detection=True,
        )
        logger.info("MarketIntelligence initialized for buddy_loop")
    except Exception as _mi_err:
        logger.warning("MarketIntelligence unavailable in buddy_loop: %s", _mi_err)

    try:
        from online_retrainer import OnlineRetrainer, create_retrain_callback

        retrainer = OnlineRetrainer()
        retrain_callback = create_retrain_callback(retrainer=retrainer, market_intel=market_intel)

        # Connect to drift detection
        if hasattr(market_intel, 'drift_manager') and market_intel.drift_manager is not None:
            market_intel.drift_manager.set_retrain_callback(retrain_callback)
            logger.info("Online retrainer connected to drift detection")
        else:
            logger.info("Online retrainer created but no drift_manager available to connect")
    except Exception as _or_err:
        logger.warning("Online retrainer integration skipped: %s", _or_err)

    while True:
        # Quick pre-sleep checks:
        # - If weekend (Sat/Sun) pause briefly.
        # - If last_seen_ts is stale (>5 minutes), assume feed dead: clear buffer,
        #   repull recent candles (200), reset pointers, and skip waiting for boundary.
        skip_sleep = False
        try:
            now_utc = datetime.now(timezone.utc)
            # Weekend check: Saturday=5, Sunday=6
            if now_utc.weekday() >= 5:
                console.print("[dim]Loop[/dim]: Weekend. sleeping 60 seconds.")
                time.sleep(60)
                continue

            if last_seen_ts is not None and pd.notna(last_seen_ts):
                delta_s = (now_utc - pd.to_datetime(last_seen_ts, utc=True)).total_seconds()
                if delta_s > 300:
                    console.print(
                        "[yellow]Loop[/yellow]: last_seen candle >5min behind; assuming feed dead. Repulling recent candles."
                    )
                    # Clear buffer and repull the most recent 200 candles (two * 100)
                    try:
                        candle_buf.clear()
                        with span("repull_recent_candles", attributes={"instrument": instrument, "count": 200}):
                            resp = client.get_candles(instrument, granularity=granularity, count=200, price="MBA")
                        warm_df = _normalize_candle_df(candles_to_ohlcv_df(resp))
                        if len(warm_df) > 0:
                            candle_buf.extend(warm_df.to_dict("records"))
                            last_seen_ts = pd.to_datetime(warm_df["time"].iloc[-1], utc=True, errors="coerce")
                            if pd.notna(last_seen_ts):
                                last_seen_from_time = _to_utc_iso_z(pd.Timestamp(last_seen_ts))
                            # Don't wait for a ghost boundary; allow immediate processing.
                            primed = True
                            last_candle_id = None
                            skip_sleep = True
                        else:
                            console.print("[yellow]Loop[/yellow]: repull returned 0 candles; sleeping 30s")
                            time.sleep(30)
                            continue
                    except Exception as _e:
                        try:
                            record_exception(_e)
                        except Exception:
                            pass
                        console.print("[red]Loop[/red]: repull failed; sleeping 30s")
                        time.sleep(30)
                        continue
        except Exception:
            # Best-effort; don't let the pre-sleep checks crash the loop.
            pass

        if not skip_sleep:
            _sleep_until_next_boundary(interval_s, lag_s=2.0)

        # Heartbeat / trace: increment and occasionally print to help identify where loop hangs.
        try:
            trace_iter += 1
        except Exception:
            trace_iter = 1
        if verbose or (trace_iter % 20 == 0):
            try:
                console.print(
                    f"[dim]Trace[/dim]: still alive — on Kindle number {trace_iter} | last_seen={last_seen_from_time} | buf={len(candle_buf)}"
                )
            except Exception:
                # best-effort trace, never raise
                pass

        # Fetch only new candles since the last seen timestamp (inclusive), then append strictly newer.
        try:
            fetch_count = int(min(max(50, interval_s // 60 * 10), 5000))
        except Exception:
            fetch_count = 200
        try:
            with span("get_incremental_candles", attributes={"instrument": instrument, "count": int(fetch_count)}):
                resp = client.get_candles(
                    instrument,
                    granularity=granularity,
                    count=int(fetch_count),
                    price="MBA",
                    from_time=last_seen_from_time,
                )
        except Exception as _e:
            try:
                record_exception(_e)
            except Exception:
                pass
            resp = None
        inc_df = _normalize_candle_df(candles_to_ohlcv_df(resp))
        if len(inc_df) == 0:
            if verbose:
                console.print("[yellow]Loop[/yellow]: no candles returned")
            continue

        if last_seen_ts is not None and pd.notna(last_seen_ts):
            inc_df = inc_df.loc[inc_df["time"] > pd.Timestamp(last_seen_ts)].copy()

        if len(inc_df) == 0:
            # No new candle since last tick.
            continue

        candle_buf.extend(inc_df.to_dict("records"))
        last_seen_ts = pd.to_datetime(inc_df["time"].iloc[-1], utc=True, errors="coerce")
        if pd.notna(last_seen_ts):
            last_seen_from_time = _to_utc_iso_z(pd.Timestamp(last_seen_ts))

        # Build the working window DataFrame from the deque (bounded memory).
        df = pd.DataFrame(list(candle_buf))
        df = _normalize_candle_df(df)
        if len(df) == 0:
            continue

        # Identify candle by its timestamp if available.
        candle_id = None
        try:
            if "time" in df.columns and len(df) > 0:
                candle_id = str(df["time"].iloc[-1])
        except Exception:
            candle_id = None
        candle_id = candle_id or f"len={len(df)}"

        # Prime on first seen candle to avoid an immediate trade mid-candle.
        if not primed:
            primed = True
            last_candle_id = candle_id
            console.print(f"[dim]Loop[/dim]: primed at candle {candle_id}; waiting for next candle")
            continue

        if candle_id == last_candle_id:
            # Avoid duplicate action if OANDA returns the same last candle.
            continue
        last_candle_id = candle_id

        # Apply training-time filters.
        if min_volume is not None and "volume" in df.columns:
            try:
                df = df.loc[df["volume"].astype(float) >= float(min_volume)].copy()
            except Exception:
                pass

        if spread_filter and ("bid_close" in df.columns and "ask_close" in df.columns):
            try:
                spread = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs()
                spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
                if not spread.empty:
                    pctl_thr = float(np.quantile(spread.to_numpy(), float(spread_pctl)))
                    med = float(np.median(spread.to_numpy()))
                    thr = max(pctl_thr, float(spread_mult) * med) if med > 0 else pctl_thr
                    keep_mask = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs() <= thr
                    df = df.loc[keep_mask].copy()
            except Exception:
                pass

        if all_features:
            from src.data.feature_engineering import FeatureEngineering
            fe = FeatureEngineering(cfg.get("feature_engineering", {}))
            try:
                with span("feature_engineering", attributes={"rows": len(df)}):
                    df = fe.create_features(
                        df,
                        include_all=True,
                        apply_candle_smoothing=bool(train_smoothing),
                        median_window=median_window,
                    )
            except Exception as _e:
                try:
                    record_exception(_e)
                except Exception:
                    pass

        numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.ffill().bfill().fillna(0.0)

        expected_cols = list(feature_columns)
        missing_cols = [c for c in expected_cols if c not in set(numeric_df.columns)]
        numeric_df = numeric_df.reindex(columns=expected_cols, fill_value=0.0)
        if missing_cols:
            mu_flat = mu.reshape(-1)
            for idx, col in enumerate(expected_cols):
                if col in set(missing_cols):
                    numeric_df[col] = float(mu_flat[idx])

        x2d_full = numeric_df.to_numpy(dtype=np.float32, copy=False)
        if len(x2d_full) < seq_len_eff:
            if verbose:
                console.print(f"[yellow]Loop[/yellow]: need {seq_len_eff} rows, got {len(x2d_full)}")
            continue

        x_std_full = (x2d_full - mu) / np.where(sigma < 1e-6, 1.0, sigma)
        x_std_full = np.clip(x_std_full, -10.0, 10.0).astype(np.float32, copy=False)
        if pca_components_mat is not None and pca_mean is not None:
            x_std_full = (x_std_full - pca_mean) @ pca_components_mat.T

        n_ens = int(meta.get("inference_ensemble_windows", 20) or 20)
        n_ens = max(1, min(n_ens, 100))
        need = seq_len_eff + n_ens - 1
        if len(x_std_full) < need:
            n_ens = 1
            need = seq_len_eff

        tail = x_std_full[-need:]
        if n_ens > 1:
            windows = np.stack([tail[i : i + seq_len_eff] for i in range(n_ens)], axis=0).astype(np.float32)
        else:
            windows = tail[-seq_len_eff:].reshape(1, seq_len_eff, -1).astype(np.float32)

        try:
            with span("model_inference", attributes={"n_ens": int(n_ens), "seq_len": int(seq_len_eff)}):
                _inputs_struct = getattr(model, "_inputs_struct", None)
                _expects_features_dict = isinstance(_inputs_struct, dict) and "features" in _inputs_struct
                pred_in = {"features": windows} if _expects_features_dict else windows
                pred = model(pred_in, training=False)
        except Exception as _e:
            try:
                record_exception(_e)
            except Exception:
                pass
            raise
        d = pred["direction"]
        c = pred["confidence"]
        if hasattr(d, "numpy"):
            d = d.numpy()
        if hasattr(c, "numpy"):
            c = c.numpy()
        dir_probs = np.asarray(d).reshape(-1)
        conf_probs = np.asarray(c).reshape(-1)
        conf_probs = np.clip(conf_probs, 0.0, 1.0)
        p_dir = float(np.mean(dir_probs))
        p_conf = float(np.quantile(conf_probs, float(meta.get("inference_confidence_quantile", 0.90) or 0.90)))

        side = "buy" if p_dir >= 0.5 else "sell"
        tier2_p_win = None
        gate_conf = float(p_conf)
        if tier2_enabled:
            try:
                dir_conf = float(p_dir) if side == "buy" else float(1.0 - float(p_dir))
                score = float(max(0.0, min(1.0, dir_conf * float(p_conf))))
                tier2_p_win = _tier2_apply_calibration(meta, score)
                if tier2_p_win is not None:
                    gate_conf = float(tier2_p_win)
            except Exception:
                tier2_p_win = None
        if verbose:
            if tier2_p_win is not None:
                console.print(
                    f"Buddy loop: candle={candle_id} direction={p_dir:.3f} ({side}) model_conf={p_conf:.3f} tier2_p_win={gate_conf:.3f}"
                )
            else:
                console.print(f"Buddy loop: candle={candle_id} direction={p_dir:.3f} ({side}) conf={gate_conf:.3f}")
        else:
            if tier2_p_win is not None:
                console.print(f"Buddy: {side} tier2_p_win={gate_conf:.3f} candle={candle_id}")
            else:
                console.print(f"Buddy: {side} conf={gate_conf:.3f} candle={candle_id}")

        thr = float(tier2_cfg.get("p_win_threshold", 0.60)) if tier2_p_win is not None else float(threshold)
        if gate_conf < thr:
            continue
        if not execute:
            console.print("[cyan]DRY-RUN[/cyan]: would place a market order.")
            continue

        from src.utils.fx_paper import pip_size, position_size_units

        last_close = float(df["close"].iloc[-1])
        ps = float(pip_size(instrument))
        stop_loss_pips = float(cfg.get("buddy", {}).get("stop_loss_pips", 20.0))
        take_profit_pips = float(cfg.get("buddy", {}).get("take_profit_pips", 40.0))
        stop_distance = stop_loss_pips * ps

        risk_eff_trade = float(risk_eff)
        if tier2_p_win is not None:
            try:
                b = float(take_profit_pips / max(stop_loss_pips, 1e-9))
                p = float(max(0.0, min(1.0, float(tier2_p_win))))
                q = 1.0 - p
                kelly = ((b * p) - q) / max(b, 1e-9)
                kelly_fraction = float(tier2_cfg.get("kelly_fraction", 0.5))
                cap = float(tier2_cfg.get("max_risk_per_trade_pct", 0.02))
                risk_eff_trade = float(max(0.0, min(cap, kelly * kelly_fraction)))
            except Exception:
                risk_eff_trade = float(risk_eff)

        units = position_size_units(
            instrument=instrument,
            equity=float(equity_eff),
            risk_per_trade_pct=float(risk_eff_trade),
            stop_distance_price=stop_distance,
            price=last_close,
        )
        units = -abs(int(units)) if side == "sell" else abs(int(units))

        # CLI override: allow user to force exact units (practice / debug)
        if force_units is not None:
            try:
                fu = int(force_units)
                units = -abs(int(fu)) if side == "sell" else abs(int(fu))
                if verbose:
                    console.print(f"[dim]Force-units override[/dim]: using units={units}")
            except Exception:
                pass

        stop_price = last_close - stop_distance if units > 0 else last_close + stop_distance
        tp_distance = take_profit_pips * ps
        tp_price = last_close + tp_distance if units > 0 else last_close - tp_distance

        # Finalize units as integer (preserve sign) and submit exactly that value.
        try:
            units_final = int(units)
        except Exception:
            units_final = int(float(units))

        if verbose:
            console.print(f"[dim]Order sizing[/dim]: computed_units={units} final_submitted_units={units_final}")

        price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=units_final)
        try:
            with span("order_submission", attributes={"instrument": instrument, "units": int(units_final)}):
                result = client.create_market_order(
                    instrument=instrument,
                    units=int(units_final),
                    stop_loss_price=float(stop_price),
                    take_profit_price=float(tp_price),
                    price_bound=price_bound,
                    client_tag="buddy_tf",
                )
        except Exception as _e:
            try:
                record_exception(_e)
            except Exception:
                pass
            raise
        console.print(f"[green]Order submitted[/green]: {result}")
        # Schedule auto-close for scalping if requested via CLI `max_hold_minutes`.
        try:
            if max_hold_minutes is not None:
                m = float(max_hold_minutes)
                if m > 0:
                    _schedule_auto_close(client, instrument, delay_s=m * 60.0, verbose=bool(verbose))
        except Exception:
            pass

        trades_done += 1
        if int(max_trades) > 0 and trades_done >= int(max_trades):
            console.print(f"[dim]Loop[/dim]: reached max_trades={int(max_trades)}; exiting")
            return


def _normalize_command_args(args: Any) -> None:
    if getattr(args, "command", None) == "Buddy":
        console.print("[yellow]Deprecation: use lowercase 'buddy' instead of 'Buddy'[/yellow]")
        args.command = "buddy"


def _maybe_run_buddy_interactive_wizard(*, default_config: str) -> bool:
    if len(sys.argv) == 2 and sys.argv[1] in {"buddy", "Buddy"} and sys.stdin.isatty():
        if sys.argv[1] == "Buddy":
            console.print("[yellow]Deprecation: please use lowercase 'buddy' instead of 'Buddy'[/yellow]")
        try:
            _buddy_interactive_wizard(default_config=default_config)
            return True
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            return True
    return False


def _maybe_launch_buddy_repl(args: Any) -> bool:
    if getattr(args, "command", None) == "buddy" and bool(getattr(args, "repl", False)):
        launch_buddy_repl_from_wizard(
            args.config,
            checkpoint_path=getattr(args, "model_path", None),
            instrument=str(getattr(args, "instrument", "USD_JPY")),
            granularity=str(getattr(args, "granularity", "M5")),
            candles=int(getattr(args, "candles", 300)),
            execute=bool(getattr(args, "execute", True)),
            all_features=bool(getattr(args, "all_features", False)),
            verbose=bool(getattr(args, "verbose", False)),
            assistant_name="Buddy",
        )
        return True
    return False


def _maybe_launch_buddy_chat(args: Any) -> bool:
    if getattr(args, "command", None) != "buddy-chat":
        return False

    explicit_execute = any(token in {"--execute", "-x"} for token in sys.argv[1:])
    execute = explicit_execute and not bool(getattr(args, "dry_run", False))
    launch_buddy_repl_from_wizard(
        args.config,
        checkpoint_path=getattr(args, "model_path", None),
        instrument=str(getattr(args, "instrument", "USD_JPY")),
        granularity=str(getattr(args, "granularity", "M5")),
        candles=int(getattr(args, "candles", 300)),
        execute=execute,
        all_features=bool(getattr(args, "all_features", False)),
        verbose=bool(getattr(args, "verbose", False)),
        assistant_name="Buddy",
    )
    return True


def _compute_force_units(*, force_units_raw: Any, force_margin_raw: Any) -> int | None:
    if force_units_raw is not None:
        return int(force_units_raw)
    if force_margin_raw is not None:
        return int(round(float(force_margin_raw) / 0.05))
    return None


def _dispatch_buddy(args: Any, command_map: dict[str, Any]) -> None:
    should_execute = bool(args.execute) and (not bool(getattr(args, "dry_run", False)))

    # Validate and normalize instrument name
    try:
        instrument_validated = _validate_instrument(str(args.instrument))
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    force_units_eff = _compute_force_units(
        force_units_raw=getattr(args, "force_units", None),
        force_margin_raw=getattr(args, "force_margin", None),
    )
    max_hold_raw = getattr(args, "max_hold_minutes", None)
    max_hold_eff = float(max_hold_raw) if max_hold_raw is not None else None

    # Check for aggressive scaling mode
    aggressive_scaling = bool(getattr(args, "aggressive_scaling", False))

    # Check for RL position sizer
    use_rl_sizer = bool(getattr(args, "use_rl_sizer", False))

    # Intelligent mode flags
    intelligent = bool(getattr(args, "intelligent", False))
    explain = bool(getattr(args, "explain", False))
    llm_provider = str(getattr(args, "llm_provider", "auto"))

    if bool(getattr(args, "loop", False)):
        buddy_loop(
            args.config,
            checkpoint_path=args.model_path,
            instrument=instrument_validated,
            granularity=args.granularity,
            candles=args.candles,
            execute=should_execute,
            force_execute=bool(getattr(args, "force_execute", False)),
            force_units=force_units_eff,
            max_hold_minutes=max_hold_eff,
            equity=float(getattr(args, "equity", 10_000.0)),
            risk_per_trade_pct=float(getattr(args, "risk", 0.005)),
            all_features=getattr(args, "all_features", False),
            verbose=args.verbose,
            max_trades=int(getattr(args, "max_trades", 1)),
            profile=str(getattr(args, "profile", INFERENCE_PROFILE_BALANCED)),
        )
        return

    command_map["buddy"](
        args.config,
        checkpoint_path=args.model_path,
        instrument=instrument_validated,
        granularity=args.granularity,
        candles=args.candles,
        execute=should_execute,
        force_execute=bool(getattr(args, "force_execute", False)),
        force_units=force_units_eff,
        max_hold_minutes=max_hold_eff,
        equity=float(getattr(args, "equity", 10_000.0)),
        risk_per_trade_pct=float(getattr(args, "risk", 0.005)),
        all_features=getattr(args, "all_features", False),
        verbose=args.verbose,
        aggressive_scaling=aggressive_scaling,
        use_rl_sizer=use_rl_sizer,
        intelligent=intelligent,
        explain=explain,
        llm_provider=llm_provider,
        llm_enhance=getattr(args, "llm_enhance", False),
        profile=str(getattr(args, "profile", INFERENCE_PROFILE_BALANCED)),
        master_pair=getattr(args, "master_pair", None),
        interactive_master=bool(getattr(args, "interactive_master", True)),
    )


def _dispatch_train_buddy(args: Any, command_map: dict[str, Any]) -> None:
    # Suppress verbose utils logging during config load for clean output
    import logging as _logging
    _utils_logger = _logging.getLogger('utils')
    _utils_prev = _utils_logger.level
    _utils_logger.setLevel(_logging.WARNING)
    cfg = load_config(args.config)
    _utils_logger.setLevel(_utils_prev)

    buddy_cfg = (cfg.get("buddy") or {}) if isinstance(cfg, dict) else {}
    buddy_train_cfg = (buddy_cfg.get("train_defaults") or buddy_cfg.get("train") or {}) if isinstance(buddy_cfg, dict) else {}
    training_cfg = (cfg.get("training") or {}) if isinstance(cfg, dict) else {}
    model_cfg = (cfg.get("model") or {}) if isinstance(cfg, dict) else {}

    def _cfg_get(name: str, default_val):
        """Get config value from buddy.train_defaults first, then root config."""
        try:
            # First check buddy.train_defaults
            if isinstance(buddy_train_cfg, dict) and name in buddy_train_cfg:
                return buddy_train_cfg[name]
            # Then check root config
            if isinstance(cfg, dict) and name in cfg:
                return cfg[name]
            # Then check training section
            if isinstance(training_cfg, dict) and name in training_cfg:
                return training_cfg[name]
            return default_val
        except Exception:
            return default_val

    def _model_cfg_get(name: str, default_val):
        """Get config value from model section."""
        try:
            return model_cfg.get(name, default_val) if isinstance(model_cfg, dict) else default_val
        except Exception:
            return default_val

    # CLI overrides config; if CLI arg is None, fall back to config then hard default.
    seq_len_eff = int(args.seq_len) if getattr(args, "seq_len", None) is not None else int(_cfg_get("seq_len", 50))
    epochs_eff = int(args.epochs) if getattr(args, "epochs", None) is not None else int(_cfg_get("epochs", 300))
    batch_size_eff = int(args.batch_size) if getattr(args, "batch_size", None) is not None else int(_cfg_get("batch_size", 32))
    lr_eff = float(args.lr) if getattr(args, "lr", None) is not None else float(_cfg_get("lr", 0.001))
    patience_eff = int(args.patience) if getattr(args, "patience", None) is not None else int(_cfg_get("patience", 10))
    es_monitor_eff = str(args.es_monitor) if getattr(args, "es_monitor", None) is not None else str(_cfg_get("es_monitor", "direction"))
    combined_w_dir_eff = float(args.combined_w_dir) if getattr(args, "combined_w_dir", None) is not None else float(_cfg_get("combined_w_dir", 0.7))
    combined_w_conf_eff = float(args.combined_w_conf) if getattr(args, "combined_w_conf", None) is not None else float(_cfg_get("combined_w_conf", 0.3))
    steps_per_execution_eff = int(args.steps_per_execution) if getattr(args, "steps_per_execution", None) is not None else int(_cfg_get("steps_per_execution", 1))
    fit_verbose_eff = int(args.fit_verbose) if getattr(args, "fit_verbose", None) is not None else int(_cfg_get("fit_verbose", 1))
    shared_encoder_eff = (
        bool(args.shared_encoder)
        if getattr(args, "shared_encoder", None) is not None
        else bool(_cfg_get("shared_encoder", False))
    )

    # M1 METAL CRITICAL SETTINGS - now properly read from config
    # For BooleanOptionalAction: None means user didn't pass flag, use config
    _mp_cli = getattr(args, "mixed_precision", None)
    mixed_precision_eff = (
        bool(_mp_cli) if _mp_cli is not None
        else bool(_cfg_get("mixed_precision", False))  # Default False; enable explicitly in config
    )
    jit_compile_eff = (
        True if bool(getattr(args, "jit_compile", False))
        else bool(_cfg_get("jit_compile", False))
    )
    cache_val_eff = (
        True if bool(getattr(args, "cache_val", False))
        else bool(_cfg_get("cache_val", True))  # Default True for M1
    )
    prefetch_eff = (
        getattr(args, "prefetch", None)
        if getattr(args, "prefetch", None) is not None
        else _cfg_get("prefetch", None)
    )
    shuffle_buffer_eff = (
        getattr(args, "shuffle_buffer", None)
        if getattr(args, "shuffle_buffer", None) is not None
        else _cfg_get("shuffle_buffer", None)
    )

    # Model type from CLI or config - TCN is default for M1 Metal performance
    model_type_eff = (
        str(args.model_type) if getattr(args, "model_type", None) is not None
        else str(_model_cfg_get("type", "tcn"))
    )

    tier2_stride_eff = (
        int(getattr(args, "tier2_calibration_stride", 0))
        if getattr(args, "tier2_calibration_stride", None) is not None
        else int(_cfg_get("tier2_calibration_stride", 5))
    )
    tier2_horizon_eff = (
        int(getattr(args, "tier2_horizon_candles", 0))
        if getattr(args, "tier2_horizon_candles", None) is not None
        else int(_cfg_get("tier2_horizon_candles", 0))
    )
    if int(tier2_horizon_eff) <= 0:
        tier2_horizon_eff = None

    # Default to fetching fresh OANDA candles for training unless the user
    # explicitly provides a local CSV.
    oanda_live_eff = bool(getattr(args, "oanda_live", False)) or (getattr(args, "csv", None) is None)

    # Use candles from args (default 5000 for training)
    train_candles = int(args.candles)

    # Check for multi-pair mode - skip single-pair OANDA fetch if enabled
    multi_pair_mode = bool(getattr(args, "multi_pair", False))

    oanda_fetch = None
    if bool(oanda_live_eff) and not multi_pair_mode:
        # Single-pair mode: validate and normalize instrument name
        instrument_raw = str(getattr(args, "instrument", "USD_JPY"))
        try:
            instrument_validated = _validate_instrument(instrument_raw)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise SystemExit(1)

        if instrument_raw.upper().replace("/", "_").replace("-", "_") != instrument_validated:
            console.print(f"[yellow]Normalized instrument: {instrument_raw} -> {instrument_validated}[/yellow]")

        console.print(Panel(
            f"[bold]Fetching Live Market Data[/bold]\n\n"
            f"[dim]Instrument:[/dim] {instrument_validated}\n"
            f"[dim]Granularity:[/dim] {args.granularity}\n"
            f"[dim]Candles:[/dim] {train_candles:,}",
            title="🌐 OANDA API",
            border_style="cyan",
        ))

        oanda_fetch = OandaFetchOptions(
            instrument=instrument_validated,
            granularity=str(args.granularity),
            candles=int(train_candles),
            price=str(getattr(args, "oanda_price", "MBA")),
            save_csv=getattr(args, "oanda_save_csv", None),
        )
    elif multi_pair_mode:
        # Multi-pair mode: create a placeholder oanda_fetch just for granularity/candles info
        # The actual multi-pair fetching happens in _train_buddy_impl
        oanda_fetch = OandaFetchOptions(
            instrument="MULTI_PAIR",  # Placeholder
            granularity=str(args.granularity),
            candles=int(train_candles),
            price=str(getattr(args, "oanda_price", "MBA")),
            save_csv=None,
        )

    init_from = getattr(args, "init_from", None)
    # WARM-START default=True: Always load existing weights for compounding learning
    # BUT only if the checkpoint file actually exists
    if not init_from and bool(getattr(args, "warm_start", True)):
        default_checkpoint = Path("trained_data") / "models" / "buddy_tf.keras"
        if default_checkpoint.exists():
            init_from = str(default_checkpoint)
        else:
            # Checkpoint doesn't exist - skip warm-start silently (first training run)
            pass

    advanced = BuddyTrainingAdvancedOptions(
        init_from=init_from,
        es_monitor=str(es_monitor_eff),
        combined_w_dir=float(combined_w_dir_eff),
        combined_w_conf=float(combined_w_conf_eff),
        train_smoothing=not bool(getattr(args, "no_train_smoothing", None) or _cfg_get("no_train_smoothing", False)),
        top_features=(
            int(getattr(args, "top_features", 0))
            if getattr(args, "top_features", None) is not None
            else (int(_cfg_get("top_features", 0)) if _cfg_get("top_features", None) is not None else None)
        ),
        median_window=(
            int(getattr(args, "median_window", 0))
            if getattr(args, "median_window", None) is not None
            else None
        ),
        vol_norm_window=(
            int(getattr(args, "vol_norm_window", 0))
            if getattr(args, "vol_norm_window", None) is not None
            else None
        ),
        min_volume=(
            int(getattr(args, "min_volume", 0))
            if getattr(args, "min_volume", None) is not None
            else None
        ),
        spread_filter=bool(getattr(args, "spread_filter", False)),
        spread_pctl=float(getattr(args, "spread_pctl", 0.99)),
        spread_mult=float(getattr(args, "spread_mult", 3.0)),
        tier2_calibrate=bool(getattr(args, "tier2_calibrate", True)),
        tier2_horizon_candles=(int(tier2_horizon_eff) if tier2_horizon_eff is not None else None),
        tier2_calibration_stride=int(tier2_stride_eff),
    )

    command_map["train-buddy"](
        args.config,
        None if bool(oanda_live_eff) else args.csv,
        oanda_fetch=oanda_fetch,
        advanced=advanced,
        feature_curriculum=getattr(args, "feature_curriculum", None),
        curriculum_ks=getattr(args, "curriculum_ks", None),
        pca_components=args.pca_components,
        seq_len=int(seq_len_eff),
        epochs=int(epochs_eff),
        batch_size=int(batch_size_eff),
        lr=float(lr_eff),
        patience=int(patience_eff),
        seed=int(args.seed),
        run_tag=args.run_tag,
        all_features=True,
        device=str(getattr(args, "device", "auto")),
        ignore_input_mismatches=bool(getattr(args, "ignore_input_mismatches", False)),
        disable_tier2_on_mismatch=bool(getattr(args, "disable_tier2_on_mismatch", False)),
        mixed_precision=bool(mixed_precision_eff),  # Now reads from config
        steps_per_execution=int(steps_per_execution_eff),
        jit_compile=bool(jit_compile_eff),  # Now reads from config
        shuffle_buffer=shuffle_buffer_eff,  # Now reads from config
        prefetch=prefetch_eff,  # Now reads from config
        cache_val=bool(cache_val_eff),  # Now reads from config
        combined_use_predict=bool(getattr(args, "combined_use_predict", True)),
        shared_encoder=bool(shared_encoder_eff),
        model_type=str(model_type_eff),  # NEW: Pass model type from config
        timing=bool(getattr(args, "timing", False)),
        fit_verbose=int(fit_verbose_eff),
        # Enterprise training options (enabled by default for ensemble)
        enterprise=not bool(getattr(args, "no_enterprise", False)),
        cv_folds=int(getattr(args, "cv_folds", 5)),
        bootstrap=not bool(getattr(args, "no_bootstrap", False)),
        bootstrap_samples=int(getattr(args, "bootstrap_samples", 1000)),
        mlflow_experiment=getattr(args, "mlflow_experiment", None),
        generate_report=not bool(getattr(args, "no_report", False)),
        # Continual learning - EWC enabled by default
        enable_ewc=not bool(getattr(args, "disable_ewc", False)),
        # Multi-pair foundation training
        multi_pair=bool(getattr(args, "multi_pair", False)),
        foundation_pairs=getattr(args, "foundation_pairs", None),
        # RL position sizer training (after ensemble)
        train_rl_sizer=not bool(getattr(args, "skip_rl", False)),
        rl_timesteps=(
            int(getattr(args, "timesteps"))
            if getattr(args, "timesteps", None) is not None
            else None
        ),
        # Weights & Biases experiment tracking
        wandb_enabled=bool(getattr(args, "wandb_enabled", False)),
        wandb_project=str(getattr(args, "wandb_project", "ml_engine_fx")),
        wandb_name=getattr(args, "wandb_name", None),
        wandb_tags=getattr(args, "wandb_tags", None),
        wandb_offline=bool(getattr(args, "wandb_offline", False)),
        wandb_gradients=bool(getattr(args, "wandb_gradients", True)),
        wandb_system=bool(getattr(args, "wandb_system", True)),
        # Correlation-based transfer learning
        correlation_transfer=bool(getattr(args, "correlation_transfer", False)),
        master_pairs=getattr(args, "master_pairs", None),
        target_pairs=getattr(args, "target_pairs", None),
        correlation_threshold=float(getattr(args, "correlation_threshold", 0.70)),
        skip_master_training=bool(getattr(args, "skip_master_training", False)),
        transfer_epochs=int(getattr(args, "transfer_epochs", 50)),
    )


# NOTE: buddy_scan, buddy_predict_78, and _buddy_scan_legacy are imported from cli/buddy_scanning.py
# NOTE: suggest_improvements is imported from cli/analytics_commands.py
# NOTE: retrain_gates is imported from cli/training_ops.py


def _buddy_test_modular_ensemble(
    config_path: str,
    instrument: str,
    granularity: str,
    test_candles: int,
    verbose: bool = False,
) -> None:
    """
    Hindcast test using the modular ensemble (4 independent models with gates).
    Shows BOTH gated results (production) and raw TCN results (evaluation).
    """
    import json
    import numpy as np

    console.print("[bold magenta]Using MODULAR ENSEMBLE (4 models with gates)[/bold magenta]")
    console.print("-" * 60)

    cfg = load_config(config_path)

    # Check if instrument was in training set
    meta_path = Path("trained_data") / "models" / "modular_ensemble.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        trained_pairs = meta.get("selected_pairs", [])
        if trained_pairs and instrument not in trained_pairs:
            console.print(f"\n[bold red]⚠️  WARNING: {instrument} was NOT in the training set![/bold red]")
            console.print(f"[yellow]Trained pairs: {', '.join(trained_pairs)}[/yellow]")
            console.print(f"[yellow]Model may not generalize well to {instrument}.[/yellow]")
            console.print(f"[cyan]For accurate testing, use: buddy test -I {trained_pairs[0]}[/cyan]\n")

    # Fetch candles
    fetch_count = test_candles + 300  # Buffer for feature engineering NaN rows
    console.print(f"[dim]Fetching {fetch_count} candles from OANDA...[/dim]")

    from src.utils.fx_paper import candles_to_ohlcv_df
    from src.utils.oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    resp = client.get_candles(instrument, granularity=granularity, count=fetch_count, price="MBA")
    df_raw = candles_to_ohlcv_df(resp)

    # Feature engineering
    from src.data.feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    df = fe.create_features(df_raw.copy(), include_all=True)

    # Load modular ensemble
    _configure_tf_metal(verbose=verbose)

    from src.core.modular_inference import ModularEnsembleInference

    ensemble = ModularEnsembleInference()
    ensemble.load_models()

    # Run hindcast
    console.print("\n[bold]Testing predictions with modular ensemble...[/bold]\n")

    closes = df["close"].values
    pip_size = 0.01 if "JPY" in instrument else 0.0001

    # Get threshold from MODEL METADATA (not config!) to match how model was trained
    model_meta_path = Path("trained_data") / "models" / "modular_ensemble.meta.json"
    model_meta = json.loads(model_meta_path.read_text()) if model_meta_path.exists() else {}
    model_cfg = model_meta.get("config", {})

    threshold_pct = model_cfg.get("direction_threshold", DIRECTION_DEFAULTS['threshold'])
    lookahead = model_cfg.get("direction_lookahead", DIRECTION_DEFAULTS['lookahead'])

    console.print(f"[dim]Using MODEL training settings: threshold={threshold_pct*100:.2f}%, lookahead={lookahead}[/dim]")

    # Test window - need extra buffer for lookahead
    test_start = len(df) - test_candles - lookahead - 1
    test_end = len(df) - lookahead - 1  # Leave room for lookahead measurement

    # Metrics for GATED trades (production mode)
    gated_correct = 0
    gated_wrong = 0
    gated_wins_pips = []
    gated_losses_pips = []
    trades_taken = 0
    trades_skipped = 0

    # Metrics for RAW TCN direction (evaluation mode - no gates)
    raw_correct = 0
    raw_wrong = 0
    raw_wins_pips = []
    raw_losses_pips = []

    # Metrics for CLEAR signals only (matching training threshold)
    clear_correct = 0
    clear_wrong = 0

    # Track gate failures
    gate_failures = {"confidence": 0, "momentum": 0, "risk": 0, "no_direction": 0}

    # Track actual XGBoost outputs for debugging
    all_momentum_values = []
    all_acceleration_values = []
    all_rf_drawdowns = []
    all_rf_streaks = []

    console.print(f"[dim]Using lookahead={lookahead} bars, threshold={threshold_pct*100:.2f}% to match training config[/dim]\n")

    for i in range(test_start, test_end):
        if i < 60:  # Need at least 60 bars of history
            continue

        # Get features up to this point (no lookahead)
        df_slice = df.iloc[:i+1].copy()

        # Run modular inference
        try:
            signal = ensemble.predict(df_slice)
        except Exception as e:
            if verbose:
                console.print(f"[dim]Prediction failed at {i}: {e}[/dim]")
            continue

        # Track XGBoost outputs for debugging
        all_momentum_values.append(signal.xgb_momentum)
        all_acceleration_values.append(signal.xgb_acceleration)
        all_rf_drawdowns.append(signal.rf_drawdown_pips)
        all_rf_streaks.append(signal.rf_streak_prob)

        # Get timestamp
        try:
            ts = df.index[i]
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)[:16]
        except Exception:
            ts_str = f"candle_{i}"

        # Actual movement over LOOKAHEAD period (must match training!)
        entry_price = closes[i]
        exit_price = closes[i + lookahead]  # lookahead from config
        actual_pips = (exit_price - entry_price) / pip_size
        actual_direction = "↑" if actual_pips > 0 else "↓"

        # Calculate move percentage (for threshold matching)
        move_pct = abs(exit_price - entry_price) / entry_price
        is_clear_move = move_pct >= threshold_pct  # threshold_pct from config

        # RAW TCN evaluation (ignore gates - just evaluate direction prediction)
        if signal.tcn_direction is not None:
            raw_is_correct = (
                (signal.tcn_direction == 1 and actual_pips > 0) or
                (signal.tcn_direction == 0 and actual_pips < 0)
            )
            if raw_is_correct:
                raw_correct += 1
                raw_wins_pips.append(abs(actual_pips))
            else:
                raw_wrong += 1
                raw_losses_pips.append(abs(actual_pips))

            # Also track clear-signal accuracy (matching training labels)
            if is_clear_move:
                if raw_is_correct:
                    clear_correct += 1
                else:
                    clear_wrong += 1

        # GATED production evaluation
        if signal.trade:
            trades_taken += 1
            predicted_side = "BUY" if signal.direction == "long" else "SELL"

            is_correct = (
                (signal.direction == "long" and actual_pips > 0) or
                (signal.direction == "short" and actual_pips < 0)
            )

            if is_correct:
                gated_correct += 1
                gated_wins_pips.append(abs(actual_pips))
                status = "[green]✅[/green]"
            else:
                gated_wrong += 1
                gated_losses_pips.append(abs(actual_pips))
                status = "[red]❌[/red]"

            # Gate status
            gates = []
            gates.append(f"conf={signal.ridge_confidence:.0f}{'✓' if signal.confidence_gate_passed else '✗'}")
            gates.append(f"mom={signal.xgb_momentum:.2f}{'✓' if signal.momentum_gate_passed else '✗'}")
            rf_dd_pct = signal.rf_drawdown_pips / 10000 if signal.rf_drawdown_pips > 1.0 else signal.rf_drawdown_pips
            gates.append(f"dd={rf_dd_pct:.2%}{'✓' if signal.risk_gate_passed else '✗'}")

            console.print(
                f"{ts_str} | {predicted_side:4} [{', '.join(gates)}] | "
                f"Actual: {actual_direction} {actual_pips:+.1f} pips | {status}"
            )
        else:
            trades_skipped += 1
            # Track why gate failed
            if signal.tcn_direction is None:
                gate_failures["no_direction"] += 1
            if not signal.confidence_gate_passed:
                gate_failures["confidence"] += 1
            if not signal.momentum_gate_passed:
                gate_failures["momentum"] += 1
            if not signal.risk_gate_passed:
                gate_failures["risk"] += 1

            if verbose:
                tcn_dir = "LONG" if signal.tcn_direction == 1 else "SHORT" if signal.tcn_direction == 0 else "NONE"
                console.print(
                    f"[dim]{ts_str} | TCN:{tcn_dir} conf={signal.ridge_confidence:.0f} "
                    f"mom={signal.xgb_momentum:.2f} dd={signal.rf_drawdown_pips / 10000 if signal.rf_drawdown_pips > 1.0 else signal.rf_drawdown_pips:.2%} | "
                    f"Actual: {actual_direction} {actual_pips:+.1f} pips | SKIP[/dim]"
                )

    # Calculate metrics
    gated_total = gated_correct + gated_wrong
    gated_accuracy = gated_correct / gated_total * 100 if gated_total > 0 else 0
    gated_avg_win = np.mean(gated_wins_pips) if gated_wins_pips else 0
    gated_avg_loss = np.mean(gated_losses_pips) if gated_losses_pips else 0

    raw_total = raw_correct + raw_wrong
    raw_accuracy = raw_correct / raw_total * 100 if raw_total > 0 else 0
    raw_avg_win = np.mean(raw_wins_pips) if raw_wins_pips else 0
    raw_avg_loss = np.mean(raw_losses_pips) if raw_losses_pips else 0

    # Print results
    console.print("\n" + "=" * 70)
    console.print("[bold]📊 MODULAR ENSEMBLE HINDCAST RESULTS[/bold]")
    console.print("=" * 70)

    # Raw TCN performance (evaluation)
    console.print("\n[bold cyan]TCN Direction (Raw - No Gates):[/bold cyan]")
    console.print(f"  Predictions:        {raw_total}")
    console.print(f"  Correct:            {raw_correct} ({raw_accuracy:.1f}%)")
    console.print(f"  Wrong:              {raw_wrong} ({100-raw_accuracy:.1f}%)")
    console.print(f"  Avg Win:            +{raw_avg_win:.1f} pips")
    console.print(f"  Avg Loss:           -{raw_avg_loss:.1f} pips")
    if raw_total > 0:
        raw_ev = (raw_correct / raw_total) * raw_avg_win - (raw_wrong / raw_total) * raw_avg_loss
        console.print(f"  Expected Value:     {raw_ev:+.2f} pips/trade")

    # Clear signal accuracy (matching training labels - moves >= threshold)
    clear_total = clear_correct + clear_wrong
    if clear_total > 0:
        clear_accuracy = 100 * clear_correct / clear_total
        console.print(f"\n[bold magenta]Clear Signals Only (>= {threshold_pct*100:.1f}% moves, matching training):[/bold magenta]")
        console.print(f"  Clear Signals:      {clear_total} ({100*clear_total/raw_total:.1f}% of all)")
        console.print(f"  Correct:            {clear_correct} ({clear_accuracy:.1f}%)")
        console.print(f"  Wrong:              {clear_wrong} ({100-clear_accuracy:.1f}%)")

    # Gated performance (production)
    console.print("\n[bold yellow]Gated Trades (Production Mode):[/bold yellow]")
    console.print(f"  Trades Taken:       {trades_taken}")
    console.print(f"  Trades Skipped:     {trades_skipped}")
    if gated_total > 0:
        console.print(f"  Correct:            {gated_correct} ({gated_accuracy:.1f}%)")
        console.print(f"  Wrong:              {gated_wrong} ({100-gated_accuracy:.1f}%)")
        console.print(f"  Avg Win:            +{gated_avg_win:.1f} pips")
        console.print(f"  Avg Loss:           -{gated_avg_loss:.1f} pips")
        gated_ev = (gated_correct / gated_total) * gated_avg_win - (gated_wrong / gated_total) * gated_avg_loss
        console.print(f"  Expected Value:     {gated_ev:+.2f} pips/trade")
    else:
        console.print("  [dim]No trades passed gates[/dim]")

    # Gate filter breakdown
    console.print("\n[bold]Gate Filter Breakdown:[/bold]")
    console.print(f"  Momentum gate fail: {gate_failures['momentum']} ({gate_failures['momentum']/max(trades_skipped, 1)*100:.0f}%)")
    console.print(f"  Confidence gate fail: {gate_failures['confidence']} ({gate_failures['confidence']/max(trades_skipped, 1)*100:.0f}%)")
    console.print(f"  Risk gate fail:     {gate_failures['risk']} ({gate_failures['risk']/max(trades_skipped, 1)*100:.0f}%)")
    console.print(f"  No TCN direction:   {gate_failures['no_direction']}")

    # XGBoost momentum statistics (for debugging)
    if all_momentum_values:
        import numpy as np
        mom_arr = np.array(all_momentum_values)
        accel_arr = np.array(all_acceleration_values)
        console.print("\n[bold]XGBoost Momentum Debug:[/bold]")
        console.print(f"  Min:  {mom_arr.min():.4f}")
        console.print(f"  Max:  {mom_arr.max():.4f}")
        console.print(f"  Mean: {mom_arr.mean():.4f}")
        console.print(f"  >= 0.5: {(mom_arr >= 0.5).sum()} / {len(mom_arr)} ({100*(mom_arr >= 0.5).mean():.1f}%)")
        console.print(f"  Acceleration True: {accel_arr.sum()} / {len(accel_arr)} ({100*accel_arr.mean():.1f}%)")

    # RF Risk statistics (for debugging) - convert legacy pips to pct
    if all_rf_drawdowns:
        rf_dd_arr = np.array(all_rf_drawdowns)
        # Auto-detect: if values >1 they're in "pips" (pct*10000), convert to pct
        if rf_dd_arr.mean() > 1.0:
            rf_dd_pct_arr = rf_dd_arr / 10000
        else:
            rf_dd_pct_arr = rf_dd_arr
        rf_streak_arr = np.array(all_rf_streaks)
        console.print("\n[bold]RF Risk Debug:[/bold]")
        console.print(f"  Drawdown - Min: {rf_dd_pct_arr.min():.2%}, Max: {rf_dd_pct_arr.max():.2%}, Mean: {rf_dd_pct_arr.mean():.2%}")
        console.print(f"  Streak Prob - Min: {rf_streak_arr.min():.3f}, Max: {rf_streak_arr.max():.3f}, Mean: {rf_streak_arr.mean():.3f}")
        console.print(f"  Drawdown < 2.5%: {(rf_dd_pct_arr < 0.025).sum()} / {len(rf_dd_pct_arr)} ({100*(rf_dd_pct_arr < 0.025).mean():.1f}%)")
        console.print(f"  Streak prob < 0.3: {(rf_streak_arr < 0.3).sum()} / {len(rf_streak_arr)} ({100*(rf_streak_arr < 0.3).mean():.1f}%)")

    # =========================================================================
    # TRADE FREQUENCY ANALYSIS (Step 6)
    # =========================================================================
    console.print("\n[bold cyan]📊 Trade Frequency Analysis:[/bold cyan]")

    # Calculate trades per day (assuming H1 timeframe: 24 candles/day)
    candles_per_day = 24 if granularity == "H1" else 288 if granularity == "M5" else 96 if granularity == "M15" else 24
    test_days = test_candles / candles_per_day
    trades_per_day = trades_taken / test_days if test_days > 0 else 0

    console.print(f"  Test Period:      {test_days:.1f} days ({test_candles} candles)")
    console.print(f"  Trades Taken:     {trades_taken}")
    console.print(f"  Trades/Day:       {trades_per_day:.2f}")

    # Trade frequency warnings
    if trades_per_day < 1:
        console.print("  [yellow]⚠ LOW FREQUENCY: <1 trade/day. Consider relaxing gates:[/yellow]")
        console.print(f"    - Lower confidence threshold (currently {cfg.get('gates', {}).get('min_confidence', 75)})")
        console.print(f"    - Lower momentum threshold (currently {cfg.get('gates', {}).get('min_momentum', 0.5)})")
    elif trades_per_day < 2:
        console.print("  [yellow]⚠ Below target: <2 trades/day. May need to relax gates.[/yellow]")
    elif trades_per_day > 10:
        console.print(f"  [yellow]⚠ HIGH FREQUENCY: >{10} trades/day. Consider tightening gates.[/yellow]")
    else:
        console.print(f"  [green]✓ Good frequency: {trades_per_day:.1f} trades/day[/green]")

    # Regime breakdown (if regime info available)
    # Note: This requires regime classifier to be active
    console.print("\n[bold]Gate Pass Rates:[/bold]")
    total_candles = test_candles
    conf_pass_rate = (total_candles - gate_failures['confidence']) / total_candles * 100
    mom_pass_rate = (total_candles - gate_failures['momentum']) / total_candles * 100
    risk_pass_rate = (total_candles - gate_failures['risk']) / total_candles * 100
    console.print(f"  Confidence gate: {conf_pass_rate:.1f}% pass rate")
    console.print(f"  Momentum gate:   {mom_pass_rate:.1f}% pass rate")
    console.print(f"  Risk gate:       {risk_pass_rate:.1f}% pass rate")

    console.print("-" * 70)
    if raw_accuracy >= 52:
        console.print(f"[green]✓ TCN direction: {raw_accuracy:.1f}% (above 50% random)[/green]")
    else:
        console.print(f"[yellow]⚠ TCN direction: {raw_accuracy:.1f}% (near random)[/yellow]")

    if trades_taken > 0 and gated_accuracy >= 55:
        console.print(f"[green]✓ Gated trades: {gated_accuracy:.1f}% accuracy with {trades_taken} trades[/green]")
    elif trades_taken == 0:
        console.print("[yellow]⚠ Gates too strict - try training with more data or adjust thresholds[/yellow]")
    console.print("=" * 70)


def buddy_validate(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str = "EUR_USD",
    granularity: str = "H1",
    candles: int = 800,
    lookahead: int = 24,
    all_pairs: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    Professional model validation with bias detection, statistical tests,
    and risk metrics. Use this before live trading to verify model integrity.

    Checks:
    - Direction bias (model stuck predicting one direction)
    - Accuracy vs baselines (random, market bias, buy-and-hold)
    - Statistical significance (binomial test, confidence intervals)
    - Confidence calibration (high confidence = higher accuracy)
    - Time stability (consistent accuracy across periods)
    - Risk metrics (Sharpe, Sortino, max drawdown, profit factor)

    Usage:
        buddy validate --instrument EUR_USD
        buddy validate --all-pairs
        buddy validate --instrument USD_JPY --candles 1000
    """
    from tests.validation.validate_model import validate_pair_model, validate_all_pairs

    # Normalize instrument
    instrument = _normalize_instrument(instrument)

    if all_pairs:
        validate_all_pairs(
            candles=candles,
            granularity=granularity,
            lookahead=lookahead,
        )
    else:
        _validate_instrument(instrument)
        validate_pair_model(
            pair=instrument,
            candles=candles,
            granularity=granularity,
            lookahead=lookahead,
            verbose=verbose,
        )


# Legacy test alias (redirects to validate)
def buddy_test(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str = "EUR_USD",
    granularity: str = "H1",
    test_candles: int = 500,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    [DEPRECATED] Use 'buddy validate' instead.

    This redirects to the professional validation system.
    """
    console.print("[yellow]⚠️ 'buddy test' is deprecated. Using 'buddy validate' instead.[/yellow]")
    console.print("[dim]For professional validation, use: buddy validate --instrument EUR_USD[/dim]\n")

    # Load model metadata - prefer candidate if it exists (newer model)
    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    candidate_meta_path = Path("trained_data") / "models" / "buddy_tf_candidate.meta.json"

    # Prefer candidate model (newer) if it exists
    if candidate_meta_path.exists():
        meta_path = candidate_meta_path
    elif not meta_path.exists():
        console.print("[red]No trained model found. Run: buddy train --oanda-live[/red]")
        return

    meta = json.loads(meta_path.read_text())
    seq_len = int(meta.get("seq_len", 60))
    feature_columns = list(meta.get("feature_columns", []))

    std = meta.get("standardize", {})
    mu = np.asarray(std.get("mean", []), dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(std.get("std", []), dtype=np.float32).reshape(1, -1)

    # Fetch candles: need seq_len + test_candles + buffer for feature engineering NaN rows
    # Feature engineering drops ~200 rows for rolling calculations
    fetch_count = seq_len + test_candles + 300

    console.print(f"[dim]Fetching {fetch_count} candles from OANDA...[/dim]")

    from src.utils.fx_paper import candles_to_ohlcv_df
    from src.utils.oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    resp = client.get_candles(instrument, granularity=granularity, count=fetch_count, price="MBA")
    df_raw = candles_to_ohlcv_df(resp)

    # Feature engineering
    from src.data.feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    df = fe.create_features(df_raw.copy(), include_all=True)

    # Ensure we have required columns
    missing_cols = [c for c in feature_columns if c not in df.columns]
    for c in missing_cols:
        df[c] = 0.0

    # Load model
    _configure_tf_metal(verbose=verbose)
    import tensorflow as tf

    from ml_head_engine import MLEngineHead
    from mr_engine import MREngineHead
    from ms_head_engine import MSEngineHead
    from mt_engine import MTEngineHead
    from mx_head_engine import MXEngineHead

    custom_objects = {
        "MLEngineHead": MLEngineHead,
        "MREngineHead": MREngineHead,
        "MSEngineHead": MSEngineHead,
        "MTEngineHead": MTEngineHead,
        "MXEngineHead": MXEngineHead,
    }

    model_path = meta.get("model_path")
    model = None

    # Try loading with different strategies to handle Keras version mismatches
    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception as e1:
        if "keras.src.models.functional" in str(e1) or "cannot be imported" in str(e1):
            # Keras version mismatch - try with safe_mode=False
            try:
                model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects, safe_mode=False)
            except Exception as e2:
                console.print("[red]Failed to load model (Keras version mismatch)[/red]")
                console.print(f"[dim]Error: {str(e2)[:200]}[/dim]")
                console.print("[yellow]The model was saved with a different Keras version.[/yellow]")
                console.print("[cyan]Solution: Retrain the model with current version:[/cyan]")
                console.print("[cyan]  buddy train --oanda-live --candles 15000 --granularity H1[/cyan]")
                return
        else:
            console.print(f"[red]Failed to load model: {str(e1)[:200]}[/red]")
            console.print("[cyan]Solution: Retrain the model: buddy train --oanda-live[/cyan]")
            return

    if model is None:
        console.print("[red]Could not load model[/red]")
        return

    # Check if model expects dict input
    _inputs_struct = getattr(model, "_inputs_struct", None)
    _expects_features_dict = isinstance(_inputs_struct, dict) and "features" in _inputs_struct

    def _predict(x_in):
        if _expects_features_dict:
            return model({"features": x_in}, training=False)
        return model(x_in, training=False)

    # Run hindcast
    min_confidence = 0.0  # Default confidence threshold
    if min_confidence > 0:
        console.print(f"\n[bold]Testing predictions (min confidence: {min_confidence:.0%})...[/bold]\n")
    else:
        console.print("\n[bold]Testing predictions...[/bold]\n")

    results = []
    x2d = df[feature_columns].values.astype(np.float32)
    closes = df["close"].values

    # Test window: from (seq_len) to (len - 1) so we can see next candle
    test_start = len(df) - test_candles - 1
    test_end = len(df) - 1

    correct = 0
    wrong = 0
    skipped = 0
    wins_pips = []
    losses_pips = []

    for i in range(test_start, test_end):
        # Get features for prediction window
        window_start = i - seq_len
        if window_start < 0:
            continue

        x_window = x2d[window_start:i]
        if len(x_window) != seq_len:
            continue

        # Standardize
        x_std = (x_window - mu) / np.where(sigma < 1e-6, 1.0, sigma)
        x_std = np.clip(x_std, -10.0, 10.0).astype(np.float32)
        x_batch = x_std.reshape(1, seq_len, -1)

        # Predict
        pred = _predict(x_batch)
        p_dir = float(np.asarray(pred["direction"]).reshape(-1)[0])
        p_conf = float(np.asarray(pred["confidence"]).reshape(-1)[0])

        predicted_side = "BUY" if p_dir >= 0.5 else "SELL"

        # Skip if below confidence threshold
        if p_conf < min_confidence:
            skipped += 1
            continue

        # Actual movement
        entry_price = closes[i]
        exit_price = closes[i + 1]
        pip_size = 0.01 if "JPY" in instrument else 0.0001
        actual_pips = (exit_price - entry_price) / pip_size

        actual_direction = "↑" if actual_pips > 0 else "↓"
        is_correct = (predicted_side == "BUY" and actual_pips > 0) or (predicted_side == "SELL" and actual_pips < 0)

        # Record result
        if is_correct:
            correct += 1
            wins_pips.append(abs(actual_pips))
            status = "[green]✅[/green]"
        else:
            wrong += 1
            losses_pips.append(abs(actual_pips))
            status = "[red]❌[/red]"

        # Get timestamp
        try:
            ts = df.index[i]
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)[:16]
        except Exception:
            ts_str = f"candle_{i}"

        # Print result
        if verbose or len(results) < 10 or not is_correct:
            conf_color = "green" if p_conf >= 0.6 else "yellow" if p_conf >= 0.5 else "red"
            console.print(
                f"{ts_str} | Pred: {predicted_side:4} ({p_dir:.1%}) "
                f"[{conf_color}]conf={p_conf:.1%}[/{conf_color}] | "
                f"Actual: {actual_direction} {actual_pips:+.1f} pips | {status}"
            )

        results.append({
            "time": ts_str,
            "predicted": predicted_side,
            "direction_prob": p_dir,
            "confidence": p_conf,
            "actual_pips": actual_pips,
            "correct": is_correct,
        })

    # Summary
    total = correct + wrong
    accuracy = correct / total if total > 0 else 0
    avg_win = np.mean(wins_pips) if wins_pips else 0
    avg_loss = np.mean(losses_pips) if losses_pips else 0

    console.print("\n" + "=" * 60)
    console.print("[bold]📊 HINDCAST RESULTS[/bold]")
    console.print("=" * 60)
    if min_confidence > 0:
        console.print(f"  Confidence Filter:  >={min_confidence:.0%}")
        console.print(f"  Skipped (low conf): {skipped}")
    console.print(f"  Total Predictions:  {total}")
    console.print(f"  Correct:            [green]{correct}[/green] ({accuracy:.1%})")
    console.print(f"  Wrong:              [red]{wrong}[/red] ({1-accuracy:.1%})")
    console.print(f"  Avg Win:            [green]+{avg_win:.1f} pips[/green]")
    console.print(f"  Avg Loss:           [red]-{avg_loss:.1f} pips[/red]")

    # Expected value calculation
    if accuracy > 0 and (avg_win > 0 or avg_loss > 0):
        ev = (accuracy * avg_win) - ((1 - accuracy) * avg_loss)
        ev_color = "green" if ev > 0 else "red"
        console.print(f"  Expected Value:     [{ev_color}]{ev:+.2f} pips/trade[/{ev_color}]")

    # Verdict
    console.print("-" * 60)
    if accuracy >= 0.55:
        console.print("[green]✅ Model shows edge - suitable for live trading[/green]")
    elif accuracy >= 0.50:
        console.print("[yellow]⚠️ Model at breakeven - needs improvement[/yellow]")
    else:
        console.print("[red]❌ Model underperforming - retrain recommended[/red]")
    console.print("=" * 60 + "\n")


# =============================================================================
# TRANSFER LEARNING COMMAND
# =============================================================================


def _scan_trained_models() -> list[dict[str, Any]]:
    """Scan trained_data/models/ for existing trained models.

    Returns:
        List of dicts with pair, model_path, accuracy, trained_at, age_days.
    """
    from pathlib import Path
    import json
    from datetime import datetime, timezone

    models_dir = Path("trained_data/models")
    if not models_dir.exists():
        return []

    models = []

    # Check for pair-specific models
    for pair_dir in models_dir.iterdir():
        if not pair_dir.is_dir():
            continue

        pair_name = pair_dir.name

        # Check for modular ensemble meta
        meta_path = pair_dir / "modular_ensemble.meta.json"
        if not meta_path.exists():
            # Check for buddy meta
            meta_path = pair_dir / "buddy_tf.meta.json"

        if not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text())

            # Get accuracy from meta
            accuracy = None
            if "metrics" in meta:
                metrics = meta["metrics"]
                if isinstance(metrics, dict):
                    accuracy = metrics.get("val_accuracy") or metrics.get("direction_accuracy")
            if accuracy is None:
                accuracy = meta.get("val_accuracy") or meta.get("direction_accuracy")

            # Get trained timestamp
            trained_at = meta.get("trained_at") or meta.get("saved_at")
            age_days = None
            if trained_at:
                try:
                    trained_dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - trained_dt).days
                except Exception:
                    pass

            models.append({
                "pair": pair_name,
                "path": str(pair_dir),
                "accuracy": accuracy,
                "trained_at": trained_at,
                "age_days": age_days,
            })
        except Exception:
            continue

    # Sort by accuracy (highest first), then by age (newest first)
    models.sort(key=lambda x: (-(x.get("accuracy") or 0), x.get("age_days") or 999))

    return models


def _get_correlated_pairs(
    pair: str,
    threshold: float = 0.70,
    all_pairs: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Get pairs correlated with the given pair.

    Args:
        pair: Source pair to find correlations for.
        threshold: Minimum correlation threshold.
        all_pairs: List of all pairs to check (default: major pairs).

    Returns:
        List of (pair, correlation) tuples sorted by correlation.
    """
    if all_pairs is None:
        all_pairs = [
            "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
            "AUD_USD", "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY",
        ]

    # Remove source pair from targets
    target_pairs = [p for p in all_pairs if p != pair]

    try:
        from src.scanner.analysis.correlation import CorrelationAnalyzer
        from src.utils.oanda_practice import OandaPracticeClient
        import pandas as pd

        client = OandaPracticeClient.from_env()

        # Fetch returns for all pairs
        returns_data = {}
        for p in [pair] + target_pairs:
            try:
                resp = client.get_candles(p, granularity="H1", count=100, price="M")
                closes = [float(c["mid"]["c"]) for c in resp.get("candles", [])]
                if len(closes) > 1:
                    returns_data[p] = pd.Series(closes).pct_change().dropna()
            except Exception:
                continue

        if len(returns_data) < 2:
            return []

        # Compute correlations
        analyzer = CorrelationAnalyzer(threshold=threshold)
        results = analyzer.analyze(returns_data)

        if pair not in results:
            return []

        # Extract correlated pairs
        correlated = []
        for target, result in results.items():
            if target == pair:
                continue
            corr = result.correlation_with_leader or 0.0
            if abs(corr) >= threshold:
                correlated.append((target, corr))

        # Sort by absolute correlation (highest first)
        correlated.sort(key=lambda x: abs(x[1]), reverse=True)

        return correlated

    except Exception as e:
        logger.warning(f"Could not compute correlations: {e}")
        return []


def _prompt_for_master_selection(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prompt user to select a master model from available models.

    Args:
        models: List of available models from _scan_trained_models().

    Returns:
        Selected model dict or None if cancelled.
    """
    if not models:
        console.print("[yellow]No trained models found in trained_data/models/[/yellow]")
        console.print("[dim]Train a model first with: buddy train --instrument EUR_USD[/dim]")
        return None

    console.print("\n[bold cyan]Found trained models:[/bold cyan]")
    console.print("-" * 60)

    for i, model in enumerate(models, 1):
        pair = model["pair"]
        accuracy = model.get("accuracy")
        age_days = model.get("age_days")

        acc_str = f"{accuracy:.1%}" if accuracy else "N/A"
        age_str = f"{age_days}d ago" if age_days is not None else "unknown"

        console.print(f"  {i}. {pair:12} | accuracy={acc_str:>6} | trained={age_str}")

    console.print("-" * 60)

    try:
        selection = input("\nSelect master model [1-{}] (or 'q' to quit): ".format(len(models)))
        if selection.lower() == 'q':
            return None

        idx = int(selection) - 1
        if 0 <= idx < len(models):
            return models[idx]
        else:
            console.print("[red]Invalid selection[/red]")
            return None
    except (ValueError, KeyboardInterrupt):
        console.print("\n[yellow]Cancelled[/yellow]")
        return None


def _prompt_for_target_selection(
    correlated: list[tuple[str, float]],
    threshold: float = 0.70,
) -> list[str]:
    """Prompt user to select target pairs from correlated pairs.

    Args:
        correlated: List of (pair, correlation) tuples.
        threshold: Correlation threshold for display.

    Returns:
        List of selected target pairs.
    """
    if not correlated:
        console.print(f"[yellow]No pairs found above correlation threshold {threshold}[/yellow]")
        return []

    console.print(f"\n[bold cyan]Correlated pairs (threshold >= {threshold}):[/bold cyan]")
    console.print("-" * 50)

    for i, (pair, corr) in enumerate(correlated, 1):
        corr_color = "green" if abs(corr) >= 0.8 else "yellow" if abs(corr) >= 0.7 else "dim"
        console.print(f"  {i}. {pair:12} | correlation=[{corr_color}]{corr:.3f}[/{corr_color}]")

    console.print("-" * 50)

    try:
        selection = input(f"\nSelect targets [1-{len(correlated)}, comma-separated] (or 'all'): ")
        selection = selection.strip().lower()

        if selection == 'all':
            return [p for p, _ in correlated]

        indices = [int(s.strip()) - 1 for s in selection.split(",")]
        selected = [correlated[i][0] for i in indices if 0 <= i < len(correlated)]
        return selected

    except (ValueError, KeyboardInterrupt):
        console.print("\n[yellow]Using all correlated pairs[/yellow]")
        return [p for p, _ in correlated]


def handle_transfer_command(args: Any) -> None:
    """Handle the transfer subcommand with progressive disclosure.

    Implements three usage modes:
    1. Interactive Mode (no flags): Scan models, prompt for selection
    2. Auto Mode (--auto): Auto-detect best master and correlated targets
    3. Standard Mode (--from/--to): Explicit master and targets
    4. Expert Mode (--advanced): Full control over transfer parameters

    Args:
        args: Parsed arguments from create_transfer_parser().
    """
    console.print("\n[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  CORRELATION TRANSFER LEARNING[/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Get parameters from args
    from_pair = getattr(args, "from_pair", None)
    to_pairs_str = getattr(args, "to_pairs", None)
    auto_mode = getattr(args, "auto_mode", False)
    fresh = getattr(args, "fresh", False)
    verbose = getattr(args, "verbose", False)

    # Advanced parameters
    threshold = float(getattr(args, "threshold", 0.70))
    epochs = int(getattr(args, "epochs", 50))
    lr_factor = float(getattr(args, "lr_factor", 0.03))
    frozen_layers = int(getattr(args, "frozen_layers", 2))
    ewc_lambda = float(getattr(args, "ewc_lambda", 100.0))
    patience = int(getattr(args, "patience", 15))
    granularity = getattr(args, "granularity", "H1")
    candles = int(getattr(args, "candles", 5000))

    # Legacy flag support
    if from_pair is None:
        from_pair = getattr(args, "master_pairs", None)
        if from_pair:
            from_pair = from_pair.split(",")[0]  # Take first if multiple

    if to_pairs_str is None:
        to_pairs_str = getattr(args, "target_pairs", None)

    # Parse target pairs
    to_pairs = []
    if to_pairs_str:
        to_pairs = [p.strip() for p in to_pairs_str.split(",")]

    # =========================================================================
    # MODE 1: AUTO MODE - Auto-detect everything
    # =========================================================================
    if auto_mode:
        console.print("[cyan]🤖 Auto Mode: Selecting best master and targets...[/cyan]\n")

        # Scan for trained models
        models = _scan_trained_models()

        if not models:
            console.print("[red]No trained models found. Train a model first:[/red]")
            console.print("[dim]  buddy train --instrument EUR_JPY[/dim]")
            return

        # Select best model (highest accuracy)
        best_model = models[0]
        from_pair = best_model["pair"]
        console.print(f"[green]✓ Selected master: {from_pair}[/green]")
        if best_model.get("accuracy"):
            console.print(f"[dim]  Accuracy: {best_model['accuracy']:.1%}[/dim]")

        # Find correlated pairs
        correlated = _get_correlated_pairs(from_pair, threshold=threshold)
        if correlated:
            to_pairs = [p for p, _ in correlated[:5]]  # Top 5 correlated
            console.print(f"[green]✓ Auto-selected targets: {', '.join(to_pairs)}[/green]")
        else:
            console.print("[yellow]⚠ No correlated pairs found above threshold[/yellow]")
            console.print("[dim]  Specify targets manually with --to[/dim]")
            return

    # =========================================================================
    # MODE 2: INTERACTIVE MODE - No flags provided
    # =========================================================================
    elif from_pair is None and not to_pairs:
        console.print("[cyan]📋 Interactive Mode: Select from existing models...[/cyan]\n")

        # Scan for trained models
        models = _scan_trained_models()

        # Prompt for master selection
        selected = _prompt_for_master_selection(models)
        if selected is None:
            return

        from_pair = selected["pair"]
        console.print(f"\n[green]✓ Selected master: {from_pair}[/green]")

        # Find correlated pairs
        correlated = _get_correlated_pairs(from_pair, threshold=threshold)

        if correlated:
            to_pairs = _prompt_for_target_selection(correlated, threshold)
            if not to_pairs:
                console.print("[yellow]No targets selected[/yellow]")
                return
            console.print(f"\n[green]✓ Selected targets: {', '.join(to_pairs)}[/green]")
        else:
            console.print(f"[yellow]No correlated pairs found for {from_pair}[/yellow]")
            console.print("[dim]  Specify targets manually with --to[/dim]")
            return

    # =========================================================================
    # MODE 3: STANDARD/EXPERT MODE - Explicit flags
    # =========================================================================
    elif from_pair:
        console.print(f"[cyan]📊 Standard Mode: Using specified master {from_pair}[/cyan]\n")

        # If no targets specified, find correlated pairs
        if not to_pairs:
            console.print(f"[dim]Finding correlated pairs for {from_pair}...[/dim]")
            correlated = _get_correlated_pairs(from_pair, threshold=threshold)

            if correlated:
                to_pairs = [p for p, _ in correlated]
                console.print(f"[green]✓ Found correlated targets: {', '.join(to_pairs)}[/green]")
            else:
                console.print(f"[red]No correlated pairs found for {from_pair}[/red]")
                console.print("[dim]  Specify targets manually with --to GBP_JPY,AUD_JPY[/dim]")
                return

    else:
        # --to specified without --from
        console.print("[red]Error: --to requires --from to specify the source pair[/red]")
        console.print("[dim]  Example: buddy transfer --from EUR_JPY --to GBP_JPY,AUD_JPY[/dim]")
        return

    # =========================================================================
    # DISPLAY CONFIGURATION SUMMARY
    # =========================================================================
    console.print("\n[bold]Transfer Configuration:[/bold]")
    console.print(f"  Master pair:    {from_pair}")
    console.print(f"  Target pairs:   {', '.join(to_pairs)}")
    console.print(f"  Fresh training: {'Yes' if fresh else 'No (use existing)'}")

    if getattr(args, "advanced", False) or verbose:
        console.print(f"\n[bold]Advanced Settings:[/bold]")
        console.print(f"  Correlation threshold: {threshold}")
        console.print(f"  Transfer epochs:       {epochs}")
        console.print(f"  LR factor:             {lr_factor}")
        console.print(f"  Frozen layers:         {frozen_layers}")
        console.print(f"  EWC lambda:            {ewc_lambda}")
        console.print(f"  Patience:              {patience}")

    # =========================================================================
    # CONFIRM AND EXECUTE
    # =========================================================================
    console.print("")
    try:
        confirm = input("Proceed with transfer? [Y/n]: ").strip().lower()
        if confirm not in ("", "y", "yes"):
            console.print("[yellow]Cancelled[/yellow]")
            return
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        return

    # =========================================================================
    # EXECUTE TRANSFER LEARNING
    # =========================================================================
    console.print("\n[bold cyan]Starting transfer learning...[/bold cyan]\n")

    try:
        from src.training.orchestrators.correlation_transfer import (
            CorrelationTransferOrchestrator,
            CorrelationTransferConfig,
        )
        from src.training.trainers.config import TrainerConfig
        from src.utils.oanda_practice import OandaPracticeClient
        from src.data.feature_engineering import FeatureEngineering
        import pandas as pd

        # Create transfer config with user parameters
        transfer_config = CorrelationTransferConfig(
            correlation_threshold=threshold,
            transfer_epochs=epochs,
            transfer_lr_factor=lr_factor,
            transfer_encoder_layers_to_freeze=frozen_layers,
            ewc_lambda=ewc_lambda,
            transfer_patience=patience,
            use_ewc=True,
            gradual_unfreeze=True,
            unfreeze_after_epochs=10,
        )

        trainer_config = TrainerConfig(
            epochs=epochs,
            patience=patience,
        )

        orchestrator = CorrelationTransferOrchestrator(
            config=transfer_config,
            trainer_config=trainer_config,
        )

        # Fetch data for all pairs
        console.print("[dim]Fetching market data...[/dim]")
        client = OandaPracticeClient.from_env()
        fe = FeatureEngineering()

        dfs = {}
        returns_data = {}

        all_pairs = [from_pair] + to_pairs
        for pair in all_pairs:
            try:
                resp = client.get_candles(pair, granularity=granularity, count=candles, price="MBA")
                candles_data = resp.get("candles", [])

                if not candles_data:
                    console.print(f"[yellow]⚠ No data for {pair}, skipping[/yellow]")
                    continue

                # Build DataFrame
                rows = []
                for c in candles_data:
                    rows.append({
                        "time": c["time"],
                        "open": float(c["mid"]["o"]),
                        "high": float(c["mid"]["h"]),
                        "low": float(c["mid"]["l"]),
                        "close": float(c["mid"]["c"]),
                        "volume": int(c.get("volume", 0)),
                    })

                df = pd.DataFrame(rows)
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time").sort_index()

                # Feature engineering
                df = fe.create_features(df, include_all=True)
                df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

                dfs[pair] = df

                # Returns for correlation
                returns_data[pair] = df["close"].pct_change().dropna()

                console.print(f"[green]✓[/green] {pair}: {len(df)} candles")

            except Exception as e:
                console.print(f"[red]✗[/red] {pair}: {e}")
                continue

        if from_pair not in dfs:
            console.print(f"\n[red]Error: Could not fetch data for master pair {from_pair}[/red]")
            return

        # Run transfer pipeline
        console.print("\n[bold]Running transfer learning pipeline...[/bold]\n")

        results = orchestrator.run_full_pipeline(
            returns_data=returns_data,
            dfs=dfs,
            master_pairs=[from_pair],
            target_pairs=to_pairs,
            skip_master_training=not fresh,
        )

        # Display results
        console.print("\n" + "=" * 60)
        console.print("[bold]TRANSFER LEARNING RESULTS[/bold]")
        console.print("=" * 60)

        transfer_results = results.get("transfer_results", [])
        if transfer_results:
            for result in transfer_results:
                target = result.get("target_pair", "unknown")
                source = result.get("source_pair", "unknown")
                correlation = result.get("correlation", 0)
                accuracy = result.get("post_transfer_accuracy", 0)

                console.print(f"\n  {source} → {target}")
                console.print(f"    Correlation: {correlation:.3f}")
                if accuracy:
                    console.print(f"    Accuracy:    {accuracy:.1%}")
                else:
                    console.print(f"    Accuracy:    N/A")
        else:
            console.print("\n[yellow]No transfer results[/yellow]")

        console.print("\n" + "=" * 60)
        console.print("[green]✓ Transfer learning complete[/green]")
        console.print(f"[dim]Models saved to: trained_data/models/transfer/[/dim]")

    except ImportError as e:
        console.print(f"[red]Error: Required module not available: {e}[/red]")
        console.print("[dim]Ensure all dependencies are installed[/dim]")
        raise
    except Exception as e:
        console.print(f"[red]Transfer learning failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise
