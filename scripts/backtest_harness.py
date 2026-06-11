#!/usr/bin/env python
"""Cost-aware backtest harness CLI — the honest-number machine.

Walks cached candles bar-by-bar, asks the real per-pair gate transformer for a
direction on each close (same feature enrichment + contract path as the
holdout), simulates sequential ATR-bracketed trades with spread+slippage, and
emits an equity-curve report + the promotion-style gate verdict.

Usage:
    python scripts/backtest_harness.py --instrument USD_JPY --granularity M15 \
        --candles 6000 [--spread 1.2] [--slippage 0.2] [--out report.json]

Report artifact: trained_data/backtests/{PAIR}_{GRAN}_{UTC}.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backtest_harness")

CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
OUT_DIR = PROJECT_ROOT / "trained_data" / "backtests"

# Window handed to the evaluator per signal call. Must cover the largest
# per-pair seq_len (120) + indicator warm-up margin; gates tails to the
# artifact's own saved seq_len internally.
SIGNAL_WINDOW_ROWS = 150


def main() -> int:
    ap = argparse.ArgumentParser(description="Cost-aware sequential backtest")
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--granularity", default="M15")
    ap.add_argument("--candles", type=int, default=6000,
                    help="bars to backtest (tail of the training cache)")
    ap.add_argument("--cache-candles", type=int, default=65000,
                    help="which cache file to read ({pair}_{gran}_{this}.csv)")
    ap.add_argument("--spread", type=float, default=None,
                    help="round-trip spread pips (default: per-pair table)")
    ap.add_argument("--slippage", type=float, default=0.2)
    ap.add_argument("--sl-mult", type=float, default=1.5)
    ap.add_argument("--tp-mult", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=24)
    ap.add_argument("--out", default=None, help="report JSON path override")
    args = ap.parse_args()

    import pandas as pd
    from src.core.modular_data_loaders import compute_normalized_features
    from src.data.feature_engineering import FeatureEngineering
    from src.scanner.gates import GateEvaluator
    from src.training.backtest_harness import run_backtest
    from src.training.expectancy_gate import pip_size_for, spread_for
    from src.utils import load_config

    pair = args.instrument.upper()
    spread = args.spread if args.spread is not None else spread_for(pair)
    pip_size = pip_size_for(pair)

    cache = CACHE_DIR / f"{pair}_{args.granularity}_{args.cache_candles}.csv"
    if not cache.exists():
        logger.error("No cache at %s — run a training fetch first.", cache)
        return 2
    df = pd.read_csv(cache, parse_dates=["time"]).tail(args.candles).reset_index(drop=True)
    logger.warning("Backtesting %s %s: %d bars from %s -> %s (spread=%.1f slip=%.1f)",
                   pair, args.granularity, len(df), df["time"].iloc[0], df["time"].iloc[-1],
                   spread, args.slippage)

    # TRAIN/SERVE PARITY: enrich once over the full frame (causal rolling
    # features, same as training's apply_features), then hand the evaluator
    # fixed-size tail windows — identical to the holdout's per-window path.
    cfg = load_config(str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml"))
    enriched = FeatureEngineering(cfg).create_features(df.copy(), include_all=True)
    feat = enriched if "returns_1" in enriched.columns else compute_normalized_features(enriched)
    feat = feat.reset_index(drop=True)
    offset = len(df) - len(feat)  # enrichment drops indicator-warmup head rows
    logger.warning("Enriched: %d rows (dropped %d warmup), %d cols",
                   len(feat), offset, feat.shape[1])

    ev = GateEvaluator(model_dir=str(PROJECT_ROOT / "trained_data" / "models"),
                       use_joint_only=False, use_per_pair_routing=True)
    ev.load_models(require_tcn=False)
    sub = ev._get_pair_evaluator(pair)
    if getattr(sub, "_transformer", None) is None:
        logger.error("No transformer loaded for %s — nothing to backtest.", pair)
        return 2

    calls = {"n": 0, "abstain": 0}

    def signal_fn(window_raw: pd.DataFrame):
        """Map the raw-bar window to the enriched frame; abstain near edges."""
        bar = len(window_raw) - 1            # current raw bar index
        fi = bar - offset                    # corresponding enriched row
        if fi + 1 < SIGNAL_WINDOW_ROWS:
            return None
        calls["n"] += 1
        d, _c = sub.evaluate_transformer(
            feat.iloc[fi + 1 - SIGNAL_WINDOW_ROWS: fi + 1], instrument=pair,
        )
        if d is None:
            calls["abstain"] += 1
        return d

    t0 = time.time()
    res = run_backtest(
        df, signal_fn,
        warmup_bars=offset + SIGNAL_WINDOW_ROWS,
        spread_pips=spread, slippage_pips=args.slippage, pip_size=pip_size,
        sl_atr_mult=args.sl_mult, tp_atr_mult=args.tp_mult,
        max_hold_bars=args.max_hold,
    )
    dur = time.time() - t0

    report = {
        "instrument": pair,
        "granularity": args.granularity,
        "bars": len(df),
        "window_start": str(df["time"].iloc[0]),
        "window_end": str(df["time"].iloc[-1]),
        "signal": "per_pair_transformer",
        "signal_calls": calls["n"],
        "signal_abstentions": calls["abstain"],
        "sl_atr_mult": args.sl_mult,
        "tp_atr_mult": args.tp_mult,
        "max_hold_bars": args.max_hold,
        "duration_s": round(dur, 1),
        **res.as_report(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        OUT_DIR / f"{pair}_{args.granularity}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    )
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out_path)

    print(f"REPORT:{json.dumps({k: v for k, v in report.items() if k not in ('equity_curve',)})}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
