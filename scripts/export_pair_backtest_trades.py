#!/usr/bin/env python3
"""Export rolling per-pair backtest trades for meta-labeler retraining."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scanner.config import ScannerConfig
from src.scanner.engine import Scanner
from src.scanner.results import PairAnalysis

DEFAULT_ACCOUNT_EQUITY = 10_000.0
DEFAULT_HISTORICAL_LABEL_PROFILE = "live"
HISTORICAL_LABEL_PROFILES = {
    "live": {},
    "lenient": {
        "sub_inference_tradeable_only": False,
        "sub_inference_min_confidence": 0.40,
        "sub_inference_vote_threshold": 0.55,
        "agent_promotion_min_confidence": 0.45,
        "weighted_vote_threshold": 0.48,
        "regime_vote_thresholds": {
            "LOW": 0.45,
            "NORMAL": 0.48,
            "HIGH": 0.54,
            "EXTREME": 0.60,
        },
        "max_uncertainty_score": 0.85,
        "max_model_disagreement": 1.0,
        "max_spread_pips": 5.0,
        "min_liquidity_score": 0.15,
    },
}


def _compute_recent_drawdown(df_raw: Any) -> float:
    if "close" not in df_raw.columns or len(df_raw) < 20:
        return 0.02
    close = df_raw["close"].iloc[-100:]
    running_max = close.cummax()
    drawdown = (running_max - close) / running_max.replace(0, math.nan)
    drawdown = drawdown.replace([math.inf, -math.inf], math.nan).fillna(0.0)
    return max(float(drawdown.max()), 0.001)


def _resolve_primary_probability(analysis: PairAnalysis, gate_details: dict[str, Any]) -> tuple[float, str]:
    raw_probability = float(gate_details.get("tcn_probability", analysis.tcn_probability))
    if raw_probability != 0.5:
        return raw_probability, "tcn_probability"

    confidence = min(max(float(analysis.confidence), 0.0), 1.0)
    proxy_probability = min(max(0.5 + (confidence / 2.0), 0.5), 0.999)
    return proxy_probability, "confidence_proxy"


def _apply_historical_label_overrides(
    scan_cfg: ScannerConfig,
    *,
    label_profile: str,
    weighted_vote_threshold: float | None,
    max_uncertainty_score: float | None,
    max_model_disagreement: float | None,
    max_spread_pips: float | None,
    min_liquidity_score: float | None,
) -> None:
    resolved_profile = str(label_profile or DEFAULT_HISTORICAL_LABEL_PROFILE).strip().lower()
    if resolved_profile not in HISTORICAL_LABEL_PROFILES:
        valid = ", ".join(sorted(HISTORICAL_LABEL_PROFILES))
        raise ValueError(f"Unknown historical label profile '{resolved_profile}'. Valid profiles: {valid}")

    profile_overrides = HISTORICAL_LABEL_PROFILES[resolved_profile]
    for key, value in profile_overrides.items():
        setattr(scan_cfg, key, value)

    if weighted_vote_threshold is not None:
        scan_cfg.weighted_vote_threshold = float(weighted_vote_threshold)
    if max_uncertainty_score is not None:
        scan_cfg.max_uncertainty_score = float(max_uncertainty_score)
    if max_model_disagreement is not None:
        scan_cfg.max_model_disagreement = float(max_model_disagreement)
    if max_spread_pips is not None:
        scan_cfg.max_spread_pips = float(max_spread_pips)
    if min_liquidity_score is not None:
        scan_cfg.min_liquidity_score = float(min_liquidity_score)


def _build_pair_analysis(scanner: Scanner, pair: str, df_raw: Any) -> tuple[PairAnalysis, dict[str, Any], int | None]:
    df_feat = scanner._compute_features(df_raw)
    scanner._raw_snapshots[pair] = df_raw.tail(scanner.config.lookback_candles + 50).copy()
    scanner._feature_snapshots[pair] = df_feat.tail(scanner.config.lookback_candles + 50).copy()

    direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, inference_ok, gate_details = (
        scanner._run_inference(df_raw, df_feat, pair)
    )
    gate_details = dict(gate_details or {})

    if direction == "HOLD":
        tech_direction, tech_conf = scanner._infer_from_technicals(df_feat)
        if tech_direction is not None and tech_conf is not None:
            direction = tech_direction
            confidence = max(float(confidence), float(tech_conf) * 0.85)
            gate_details.setdefault("opportunity_bias", "technical_fallback")

    metrics = scanner._calculate_metrics(df_feat, pair)
    pip_value = scanner.config.get_pip_value(pair)
    if "atr" in df_feat.columns:
        atr = float(df_feat["atr"].iloc[-1])
    elif "atr_pct_14" in df_feat.columns and "close" in df_raw.columns:
        atr = float(df_feat["atr_pct_14"].iloc[-1]) * float(df_raw["close"].iloc[-1])
    else:
        atr = 0.0
    atr_pips = atr / pip_value if pip_value > 0 else 0.0

    error_msg = None
    if not inference_ok and direction == "HOLD" and confidence < 0.01:
        error_msg = "No models loaded"
    regime_name = "UNKNOWN"
    if volatility_regime is not None:
        regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
        regime_name = regime_names[volatility_regime] if 0 <= volatility_regime <= 3 else "UNKNOWN"

    momentum_val = float(gate_details.get("momentum", confidence))
    momentum_ok = bool(gate_details.get("momentum_passed", False))
    momentum_accel = bool(gate_details.get("momentum_acceleration", False))
    confidence_ok = bool(gate_details.get("confidence_passed", False))
    confidence_score = float(gate_details.get("confidence_score", ridge_conf))
    risk_ok = bool(gate_details.get("risk_passed", False))
    drawdown_model = gate_details.get("drawdown")
    drawdown_used = float(drawdown_model) if drawdown_model is not None else _compute_recent_drawdown(df_raw)
    vol_gate_ok = bool(gate_details.get("volatility_gate_passed", False))
    final_gates_passed = bool(gates_passed) and error_msg is None

    analysis = PairAnalysis(
        pair=pair,
        direction=direction,
        confidence=float(confidence),
        gates_passed=final_gates_passed,
        tcn_confidence=float(tcn_conf),
        tcn_probability=float(gate_details.get("tcn_probability", 0.5)),
        ridge_confidence=float(ridge_conf),
        momentum=momentum_val,
        xgb_momentum=momentum_val,
        momentum_acceleration=momentum_accel,
        momentum_passed=momentum_ok,
        confidence_score=confidence_score,
        confidence_passed=confidence_ok,
        drawdown=drawdown_used,
        rf_drawdown=drawdown_used,
        risk_passed=risk_ok,
        volatility_gate_passed=vol_gate_ok,
        current_price=float(metrics["current_price"]),
        atr=float(metrics["atr"]),
        atr_pips=float(atr_pips),
        volatility_percentile=float(metrics["volatility_percentile"]),
        trend_strength=float(metrics["trend_strength"]),
        entry_score=float(metrics["entry_score"]),
        sl_pips=float(scanner.config.sl_pips),
        tp_pips=float(scanner.config.tp_pips),
        risk_pct=float(scanner.config.risk_per_trade_pct),
        scan_time=datetime.now(timezone.utc),
        volatility_regime=regime_name,
        master_pair=str(gate_details.get("model_source_pair", pair)).upper().replace("/", "_"),
        error=error_msg,
    )
    analysis = scanner._apply_specialist_agents(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details=gate_details,
    )
    analysis = scanner._run_sub_inference_agents_for_pair(analysis)
    return analysis, gate_details, volatility_regime


def _apply_execution_sizing(
    scanner: Scanner,
    analysis: PairAnalysis,
    volatility_regime: int | None,
    gate_details: dict[str, Any],
) -> PairAnalysis:
    if not scanner._init_executor() or scanner._executor is None:
        return analysis
    if analysis.direction not in {"LONG", "SHORT"}:
        return analysis

    meta_confidence = float(gate_details.get("meta_confidence", 0.0) or 0.0)
    regime_for_sizing = volatility_regime if volatility_regime is not None else 0
    lots, risk_pct, sl_pips, tp_pips, _conf_level, _scale, _regime_name = (
        scanner._executor.calculate_regime_aware_position_size(
            pair=analysis.pair,
            confidence=float(analysis.confidence),
            atr=float(analysis.atr),
            volatility_regime=int(regime_for_sizing),
            meta_confidence=meta_confidence,
            account_equity=float(scanner.config.account_equity or DEFAULT_ACCOUNT_EQUITY),
        )
    )
    analysis.recommended_lots = float(lots)
    analysis.risk_pct = float(risk_pct)
    analysis.sl_pips = float(sl_pips)
    analysis.tp_pips = float(tp_pips)
    analysis.risk_amount = float(scanner.config.account_equity or DEFAULT_ACCOUNT_EQUITY) * float(risk_pct)
    return analysis


def _simulate_trade_path(
    *,
    df_raw: Any,
    start_idx: int,
    direction: str,
    entry_price: float,
    pip_value: float,
    sl_pips: float,
    tp_pips: float,
    max_hold_bars: int,
) -> dict[str, Any]:
    last_idx = min(len(df_raw) - 1, start_idx + max(1, max_hold_bars))
    direction = str(direction).upper()
    if direction == "LONG":
        sl_price = entry_price - (sl_pips * pip_value)
        tp_price = entry_price + (tp_pips * pip_value)
    else:
        sl_price = entry_price + (sl_pips * pip_value)
        tp_price = entry_price - (tp_pips * pip_value)

    exit_idx = last_idx
    exit_price = float(df_raw["close"].iloc[last_idx])
    exit_reason = "time_stop"
    for future_idx in range(start_idx + 1, last_idx + 1):
        row = df_raw.iloc[future_idx]
        high = float(row["high"]) if "high" in df_raw.columns else float(row["close"])
        low = float(row["low"]) if "low" in df_raw.columns else float(row["close"])

        if direction == "LONG":
            hit_sl = low <= sl_price
            hit_tp = high >= tp_price
            if hit_sl and hit_tp:
                exit_idx = future_idx
                exit_price = sl_price
                exit_reason = "both_hit_sl_first"
                break
            if hit_sl:
                exit_idx = future_idx
                exit_price = sl_price
                exit_reason = "stop_loss"
                break
            if hit_tp:
                exit_idx = future_idx
                exit_price = tp_price
                exit_reason = "take_profit"
                break
        else:
            hit_sl = high >= sl_price
            hit_tp = low <= tp_price
            if hit_sl and hit_tp:
                exit_idx = future_idx
                exit_price = sl_price
                exit_reason = "both_hit_sl_first"
                break
            if hit_sl:
                exit_idx = future_idx
                exit_price = sl_price
                exit_reason = "stop_loss"
                break
            if hit_tp:
                exit_idx = future_idx
                exit_price = tp_price
                exit_reason = "take_profit"
                break

    raw_pips = (exit_price - entry_price) / pip_value if pip_value else 0.0
    pnl_pips = raw_pips if direction == "LONG" else -raw_pips
    exit_ts = df_raw.index[exit_idx]
    return {
        "exit_index": int(exit_idx),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "holding_bars": int(exit_idx - start_idx),
        "exit_timestamp": exit_ts.isoformat() if hasattr(exit_ts, "isoformat") else str(exit_ts),
        "pnl_pips": float(pnl_pips),
    }


def export_pair_backtest_trades(
    *,
    pair: str,
    config_path: str,
    granularity: str,
    candle_count: int,
    warmup: int,
    stride: int,
    max_hold_bars: int,
    trade_filter: str,
    historical_label_profile: str,
    historical_weighted_vote_threshold: float | None,
    historical_max_uncertainty_score: float | None,
    historical_max_model_disagreement: float | None,
    historical_max_spread_pips: float | None,
    historical_min_liquidity_score: float | None,
    output_path: Path,
) -> dict[str, Any]:
    pair = pair.upper().replace("/", "_")

    scan_cfg = ScannerConfig.from_cli_args(
        config_path=config_path,
        pairs=[pair],
        granularity=granularity,
        profile="balanced",
        non_interactive=True,
        force=True,
    )
    scan_cfg.account_equity = DEFAULT_ACCOUNT_EQUITY
    scan_cfg.incremental_enabled = False
    _apply_historical_label_overrides(
        scan_cfg,
        label_profile=historical_label_profile,
        weighted_vote_threshold=historical_weighted_vote_threshold,
        max_uncertainty_score=historical_max_uncertainty_score,
        max_model_disagreement=historical_max_model_disagreement,
        max_spread_pips=historical_max_spread_pips,
        min_liquidity_score=historical_min_liquidity_score,
    )
    scanner = Scanner(config=scan_cfg)
    scanner.config.account_equity = DEFAULT_ACCOUNT_EQUITY
    df = scanner._fetch_pair_data(pair, candle_count)
    if df is None:
        raise RuntimeError(f"No data available for {pair}.")
    if len(df) <= warmup + 1:
        raise RuntimeError(f"Not enough candles for {pair}: {len(df)} <= warmup {warmup}.")
    scanner._init_oanda_client()
    scanner._init_gate_evaluator()
    scanner._init_feature_engineer()
    scanner._init_modular_ensemble()
    scanner._init_executor()

    pip_value = scan_cfg.get_pip_value(pair)
    trades: list[dict[str, Any]] = []
    start_idx = max(warmup, 1)
    filtered_out = 0
    approved_count = 0

    for idx in range(start_idx, len(df) - 1, max(1, stride)):
        window = df.iloc[: idx + 1].copy()
        analysis, gate_details, volatility_regime = _build_pair_analysis(scanner, pair, window)
        if analysis.direction not in {"LONG", "SHORT"}:
            continue
        analysis = _apply_execution_sizing(scanner, analysis, volatility_regime, gate_details)

        if trade_filter == "approved" and not analysis.is_tradeable:
            filtered_out += 1
            continue
        if trade_filter == "gated" and not analysis.gates_passed:
            filtered_out += 1
            continue

        entry = float(analysis.current_price)
        ts = window.index[-1]
        timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        primary_probability, probability_source = _resolve_primary_probability(analysis, gate_details)
        path_result = _simulate_trade_path(
            df_raw=df,
            start_idx=idx,
            direction=analysis.direction,
            entry_price=entry,
            pip_value=pip_value,
            sl_pips=float(analysis.sl_pips or scan_cfg.sl_pips),
            tp_pips=float(analysis.tp_pips or scan_cfg.tp_pips),
            max_hold_bars=max_hold_bars,
        )
        pnl_pips = float(path_result["pnl_pips"])
        lots = float(analysis.recommended_lots or 0.01)
        pnl = pnl_pips * lots * 10.0
        approved_count += int(analysis.is_tradeable)

        trades.append(
            {
                "trade_id": f"{pair}_{idx:05d}",
                "instrument": pair,
                "timestamp": timestamp,
                "direction": analysis.direction.lower(),
                "probability": float(primary_probability),
                "tcn_probability": float(primary_probability),
                "probability_source": probability_source,
                "tcn_confidence": float(analysis.tcn_confidence),
                "ridge_confidence": float(analysis.ridge_confidence),
                "xgb_momentum": float(analysis.xgb_momentum),
                "rf_drawdown_pips": float(analysis.rf_drawdown * 10000.0),
                "confidence": float(analysis.confidence),
                "meta_confidence": float(gate_details.get("meta_confidence", 0.0) or 0.0),
                "volatility_regime": analysis.volatility_regime,
                "trade_allowed": bool(analysis.is_tradeable),
                "gates_passed": bool(analysis.gates_passed),
                "agent_promoted": bool(analysis.agent_promoted),
                "reason": ", ".join(analysis.why_trade) if analysis.is_tradeable else ", ".join(analysis.why_no_trade),
                "weighted_vote_score": float(analysis.weighted_vote_score),
                "weighted_vote_threshold": float(analysis.weighted_vote_threshold),
                "uncertainty_score": float(analysis.uncertainty_score),
                "spread_pips": float(analysis.spread_pips),
                "execution_quality_score": float(analysis.execution_quality_score),
                "entry_score": float(analysis.entry_score),
                "trend_strength": float(analysis.trend_strength),
                "recommended_lots": float(analysis.recommended_lots),
                "risk_pct": float(analysis.risk_pct),
                "risk_amount": float(analysis.risk_amount),
                "sl_pips": float(analysis.sl_pips),
                "tp_pips": float(analysis.tp_pips),
                "current_price": float(analysis.current_price),
                "pnl": float(pnl),
                "pnl_pips": pnl_pips,
                "exit_price": float(path_result["exit_price"]),
                "exit_timestamp": path_result["exit_timestamp"],
                "exit_reason": path_result["exit_reason"],
                "holding_bars": int(path_result["holding_bars"]),
            }
        )

    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    payload = {
        "pair": pair,
        "granularity": granularity,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candle_count": int(len(df)),
        "warmup": warmup,
        "stride": stride,
        "max_hold_bars": max_hold_bars,
        "trade_filter": trade_filter,
        "historical_label_profile": historical_label_profile,
        "trade_count": len(trades),
        "approved_trade_count": approved_count,
        "filtered_out_count": filtered_out,
        "win_rate": (wins / len(trades)) if trades else 0.0,
        "trades": trades,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Export rolling per-pair backtest trades.")
    parser.add_argument("--pair", required=True, help="FX pair like AUD_NZD")
    parser.add_argument("--config", default="config/config_improved_H1.yaml", help="Config YAML path")
    parser.add_argument("--granularity", default="H1", help="Timeframe to use")
    parser.add_argument("--candles", type=int, default=500, help="Number of candles to fetch")
    parser.add_argument("--warmup", type=int, default=120, help="Warmup candles before first prediction")
    parser.add_argument("--stride", type=int, default=3, help="Step size between evaluated bars")
    parser.add_argument("--hold-bars", type=int, default=12, help="Maximum bars to hold before time stop")
    parser.add_argument(
        "--trade-filter",
        choices=("approved", "gated", "all"),
        default="gated",
        help="Which setups to export: fully tradeable, gate-passed, or all directional signals",
    )
    parser.add_argument(
        "--historical-label-profile",
        choices=tuple(sorted(HISTORICAL_LABEL_PROFILES)),
        default=DEFAULT_HISTORICAL_LABEL_PROFILE,
        help="Historical-only specialist-agent threshold profile used during export",
    )
    parser.add_argument("--historical-weighted-vote-threshold", type=float, help="Historical-only weighted vote threshold override")
    parser.add_argument("--historical-max-uncertainty-score", type=float, help="Historical-only max uncertainty threshold override")
    parser.add_argument("--historical-max-model-disagreement", type=float, help="Historical-only max disagreement threshold override")
    parser.add_argument("--historical-max-spread-pips", type=float, help="Historical-only spread threshold override")
    parser.add_argument("--historical-min-liquidity-score", type=float, help="Historical-only minimum liquidity override")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in (
        "src.utils.keras_model_loader",
        "src.training.trainers.transformer_trainer",
        "src.training.trainers.hybrid_sft_rl_loss",
        "src.training.trainers.callbacks",
        "src.training.trainers.lightgbm_trainers",
        "src.training.trainers.random_forest_trainer",
        "src.training.trainers.ridge_trainer",
        "src.training.trainers.tcn_volatility_trainer",
        "src.core.modular_inference",
        "src.scanner.engine",
        "src.risk.position_sizing",
        "src.scanner.execution",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    payload = export_pair_backtest_trades(
        pair=args.pair,
        config_path=args.config,
        granularity=args.granularity,
        candle_count=args.candles,
        warmup=args.warmup,
        stride=args.stride,
        max_hold_bars=args.hold_bars,
        trade_filter=args.trade_filter,
        historical_label_profile=args.historical_label_profile,
        historical_weighted_vote_threshold=args.historical_weighted_vote_threshold,
        historical_max_uncertainty_score=args.historical_max_uncertainty_score,
        historical_max_model_disagreement=args.historical_max_model_disagreement,
        historical_max_spread_pips=args.historical_max_spread_pips,
        historical_min_liquidity_score=args.historical_min_liquidity_score,
        output_path=Path(args.output),
    )
    print(json.dumps({
        "pair": payload["pair"],
        "trade_count": payload["trade_count"],
        "approved_trade_count": payload["approved_trade_count"],
        "historical_label_profile": payload["historical_label_profile"],
        "win_rate": payload["win_rate"],
        "output": str(Path(args.output)),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
