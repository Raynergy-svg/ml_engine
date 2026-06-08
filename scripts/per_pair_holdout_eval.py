"""Per-pair M15 holdout evaluator — independent of scheduled_retrain.py.

The script's built-in holdout uses `pairs[:3]` aggregate accuracy which
doesn't tell you per-pair signal. This standalone evaluator:
  - For each pair, fetches the most-recent N M15 candles from OANDA
  - Loads that pair's per-pair model via GateEvaluator with per_pair_routing
  - Runs predictions for last K windows at 24-bar lookahead
  - Computes per-pair accuracy
  - Tabulates pass/fail at the 52% gate

Usage:
    python3 scripts/per_pair_holdout_eval.py
    python3 scripts/per_pair_holdout_eval.py --pairs EUR_USD,GBP_USD
    python3 scripts/per_pair_holdout_eval.py --candles 800 --windows 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("per_pair_eval")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_AUD", "EUR_JPY", "EUR_CHF",
    "GBP_JPY", "GBP_AUD", "GBP_CHF",
    "AUD_JPY", "AUD_NZD",
]


def evaluate_one(evaluator, pair: str, oanda, candles: int, windows: int,
                 lookahead: int, granularity: str = "M15") -> dict:
    """Fetch + predict + compare for one pair. Returns metrics dict."""
    try:
        candles_raw = oanda.get_candles(pair, granularity=granularity, count=candles)
        if isinstance(candles_raw, dict):
            rows = []
            for c in candles_raw.get("candles", []):
                mid = c.get("mid", {})
                rows.append({
                    "time": c.get("time"),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })
            df = pd.DataFrame(rows)
        elif isinstance(candles_raw, pd.DataFrame):
            df = candles_raw
        else:
            return {"pair": pair, "error": f"unexpected candles type {type(candles_raw)}"}

        if len(df) < lookahead + windows:
            return {"pair": pair, "error": f"only {len(df)} candles, need {lookahead + windows}"}
    except Exception as e:
        return {"pair": pair, "error": f"fetch failed: {e}"}

    # Compute features.
    #
    # Training pipeline (scripts/train_single_model_m1.py:813) does
    #     df_feat = apply_features(df) → FeatureEngineering.create_features(df, include_all=True)
    # BEFORE invoking load_direction_data, which produces derived features like
    # macd_hist_momentum, rsi_momentum, obv, macd_x_adx, etc. Without this step
    # the holdout's runtime DataFrame is missing 44/50 of the trained model's
    # expected feature_names, the inference contract gap fires, and the
    # transformer abstains to (None, 0.5) for every window — which historically
    # masqueraded as an "all-SHORT" prediction (see memory 1352).
    from src.core.modular_data_loaders import compute_normalized_features
    from src.data.feature_engineering import FeatureEngineering
    from src.utils import load_config
    _CONFIG_PATH = str(
        Path(__file__).resolve().parent.parent / "config" / "config_m1_optimized.yaml"
    )
    try:
        cfg = load_config(_CONFIG_PATH)
        df_feat = FeatureEngineering(cfg).create_features(df.copy(), include_all=True)
        # Mirror load_direction_data's conditional: compute_normalized_features is
        # idempotent when 'returns_1' is already present.
        feat = df_feat if "returns_1" in df_feat.columns else compute_normalized_features(df_feat)
    except Exception as e:
        return {"pair": pair, "error": f"feature compute failed: {e}"}

    if feat is None or len(feat) < lookahead + windows:
        return {"pair": pair, "error": "insufficient features"}

    # Walk-forward eval at 24-bar lookahead.
    # CRITICAL: pass seq_len=60 rolling-window DataFrames to evaluate_all_gates;
    # passing single-row DataFrames bypasses the transformer (returns None) and
    # falls back to momentum/SHORT — which is what was making every holdout
    # number bogus.
    SEQ_LEN = 60
    correct = 0
    total = 0
    long_correct, long_total, short_correct, short_total = 0, 0, 0, 0
    start = max(SEQ_LEN, len(feat) - lookahead - windows)
    end = len(feat) - lookahead
    for i in range(start, end):
        row = feat.iloc[i - SEQ_LEN + 1:i + 1]  # 60-bar context window
        future_close = df["close"].iloc[i + lookahead]
        current_close = df["close"].iloc[i]
        actual_dir = "LONG" if future_close > current_close else "SHORT"
        try:
            gate_result = evaluator.evaluate_all_gates(row, instrument=pair)
            if not gate_result or not isinstance(gate_result, dict):
                continue
            pred_dir = str(gate_result.get("direction", "")).upper()
            if pred_dir not in ("LONG", "SHORT"):
                continue
            total += 1
            if pred_dir == actual_dir:
                correct += 1
            if actual_dir == "LONG":
                long_total += 1
                if pred_dir == "LONG":
                    long_correct += 1
            else:
                short_total += 1
                if pred_dir == "SHORT":
                    short_correct += 1
        except Exception:
            continue

    if total == 0:
        return {"pair": pair, "error": "no gate predictions returned"}
    return {
        "pair": pair,
        "accuracy": correct / total,
        "long_accuracy": long_correct / max(long_total, 1),
        "short_accuracy": short_correct / max(short_total, 1),
        "n_predictions": total,
        "n_correct": correct,
        "passed_gate": correct / total >= 0.52,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--granularity", default="M15",
                    help="OANDA candle granularity (default M15)")
    ap.add_argument("--candles", type=int, default=800,
                    help="Candles to fetch per pair (default 800)")
    ap.add_argument("--windows", type=int, default=300,
                    help="Test windows per pair (default 300; needs candles >= windows + lookahead)")
    ap.add_argument("--lookahead", type=int, default=24,
                    help="Bars forward for direction label (default 24, matches trainer)")
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "trained_data" / "per_pair_eval.json")
    args = ap.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    log.info("Evaluating %d pairs at %s, lookahead=%d, windows=%d",
             len(pairs), args.granularity, args.lookahead, args.windows)

    from src.scanner.gates import GateEvaluator
    from src.utils.oanda_practice import OandaPracticeClient

    # CRITICAL: use_per_pair_routing=True so the evaluator routes each pair
    # to its own per-pair model (the C1.A scaler-valid ones). Default False
    # falls back to the joint model which is scaler-null and predicts all-SHORT.
    evaluator = GateEvaluator(
        model_dir=str(REPO_ROOT / "trained_data" / "models"),
        use_per_pair_routing=True,
    )
    load_status = evaluator.load_models(require_tcn=False)
    log.info("GateEvaluator loaded: %s", load_status)
    oanda = OandaPracticeClient.from_env()

    results = []
    for pair in pairs:
        log.info("=== %s ===", pair)
        res = evaluate_one(evaluator, pair, oanda, args.candles, args.windows,
                           args.lookahead, args.granularity)
        if "accuracy" in res:
            log.info("  acc=%.1f%% long=%.1f%% short=%.1f%% n=%d  %s",
                     100 * res["accuracy"], 100 * res["long_accuracy"],
                     100 * res["short_accuracy"], res["n_predictions"],
                     "✓" if res["passed_gate"] else "✗")
        else:
            log.warning("  ERROR: %s", res.get("error"))
        results.append(res)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "lookahead": args.lookahead,
        "windows": args.windows,
        "candles": args.candles,
        "gate": 0.52,
        "results": results,
    }
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    passed = [r for r in results if r.get("passed_gate")]
    failed = [r for r in results if not r.get("passed_gate") and "accuracy" in r]
    errors = [r for r in results if "error" in r]
    print()
    print("=" * 60)
    print(f"PER-PAIR M15 EVAL: {len(passed)}/{len(pairs)} passed gate (52%)")
    print("=" * 60)
    print("PASSED:")
    for r in sorted(passed, key=lambda r: -r["accuracy"]):
        print(f"  {r['pair']:10s} {100*r['accuracy']:.1f}% (n={r['n_predictions']})")
    print("FAILED:")
    for r in sorted(failed, key=lambda r: -r["accuracy"]):
        print(f"  {r['pair']:10s} {100*r['accuracy']:.1f}% (n={r['n_predictions']})")
    if errors:
        print("ERRORS:")
        for r in errors:
            print(f"  {r['pair']:10s} {r.get('error')}")
    print(f"Report: {args.report}")
    return 0 if len(passed) >= len(pairs) // 2 else 2


if __name__ == "__main__":
    sys.exit(main())
