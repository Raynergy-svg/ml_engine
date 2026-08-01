"""Path 4: True out-of-sample retrain+holdout for master pairs.

The existing per_pair_holdout_eval fetches the most-recent N candles, but
the trainer ALSO fetched the most-recent 25K candles — so the existing
holdout overlaps with training data. This script fixes that:

  Fetch (25000 + oos_bars) candles, e.g. 25700 H1 bars.
  Train on the OLDER 25000 (true historical, NOT including last `oos_bars`).
  Holdout-test on the LAST `oos_bars` bars (truly post-training).

For 4 master pairs at H1, with oos_bars=700 (~30 days):
  Per pair: ~10 min retrain + ~5 min holdout = ~15 min × 4 = ~1h total.

Usage:
  python scripts/path4_true_oos_holdout.py --pairs EUR_USD,EUR_JPY,USD_JPY,USD_CAD
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("path4_true_oos")

# --- Pre-registered OOS pass criteria (2026-07-31) ------------------------
# Changed from a bare `accuracy >= 0.52` after that gate passed a model which
# scored 0.03pp BELOW its own constant-long control. Directional accuracy is
# only meaningful relative to the majority-class baseline on the same bars.
RAW_MIN_ACCURACY = 0.52          # floor; necessary, nowhere near sufficient
MIN_EDGE_VS_MAJORITY = 0.02      # must beat always-predict-majority by 2pp
MIN_BALANCED_ACCURACY = 0.52     # guards against class-collapse into the drift
# --------------------------------------------------------------------------


def fetch_candles(pair: str, granularity: str, n: int) -> pd.DataFrame:
    from src.utils.oanda_practice import OandaPracticeClient
    from src.training.buddy_training_helpers import _fetch_paginated_candles
    oanda = OandaPracticeClient.from_env()
    rows = _fetch_paginated_candles(oanda, pair, granularity, n)
    if not rows:
        raise RuntimeError(f"OANDA returned no candles for {pair}")
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = df.set_index("time", drop=False)
    return df


def train_master_on_split(
    pair: str,
    df_train: pd.DataFrame,
    save_dir: Path,
    lookahead: int,
    threshold: float,
) -> dict:
    """Retrain a master transformer on df_train. Returns metrics."""
    from src.core.modular_data_loaders import compute_normalized_features, load_direction_data
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    df_feat = compute_normalized_features(df_train)
    log.info("[%s] training features: %d cols on %d rows", pair, len(df_feat.columns), len(df_feat))

    dir_result = load_direction_data(
        df_feat, apply_scaler=False, lookahead=lookahead, threshold=threshold,
    )
    if dir_result is None:
        raise RuntimeError(f"load_direction_data returned None for {pair}")
    log.info("[%s] X_train shape: %s, feature_count: %d", pair,
             dir_result["X_train"].shape, len(dir_result["feature_names"]))

    cfg = TrainerConfig()
    trainer = TransformerDirectionTrainer(cfg)
    metrics = trainer.train(
        dir_result["X_train"], dir_result["y_train"],
        dir_result["X_val"], dir_result["y_val"],
        feature_names=dir_result["feature_names"],
        w_train=dir_result.get("w_train"),
        w_val=dir_result.get("w_val"),
        instrument=pair,
        skip_scaling=False,
        regime_quantiles=dir_result.get("regime_quantiles"),
        regime_atr_col=dir_result.get("regime_atr_col"),
        feature_pipeline_version=dir_result.get("feature_pipeline_version"),
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "transformer_direction.keras"
    trainer.save(str(save_path), instrument=pair)
    log.info("[%s] saved to %s, val_balanced_acc=%.4f", pair, save_path,
             metrics.get("val_balanced_accuracy", 0))
    return metrics


def holdout_oos(
    pair: str,
    df_oos: pd.DataFrame,
    model_dir: Path,
    base_models_dir: Path,
    seq_len: int,
    lookahead: int,
) -> dict:
    """Predict on df_oos rolling windows; compare to actuals."""
    from src.core.modular_data_loaders import compute_normalized_features
    from src.scanner.gates import GateEvaluator

    # Build features with the SAME pipeline the trainer uses. Calling
    # compute_normalized_features alone yields ~29 columns, but a head trained
    # via train_single_model_m1.py expects the FeatureEngineering(include_all)
    # superset — 37 of its 50 contract features (obv, cum_returns,
    # returns_lag_*, di_spread, macd_x_adx, cci_signal, day_cos, ...) are simply
    # absent from the narrow build. The transformer then refuses the contract
    # and returns its (None, 0.5) abstain sentinel on EVERY bar, which this
    # function's `except Exception: continue` rendered as a bare
    # "no predictions returned". Mirrors scripts/per_pair_holdout_eval.py.
    try:
        from src.data.feature_engineering import FeatureEngineering
        from src.utils import load_config
        _cfg_path = REPO_ROOT / "config" / "config_m1_optimized.yaml"
        _df_feat = FeatureEngineering(load_config(str(_cfg_path))).create_features(
            df_oos.copy(), include_all=True
        )
        feat = (
            _df_feat if "returns_1" in _df_feat.columns
            else compute_normalized_features(_df_feat)
        )
    except Exception as exc:
        log.warning("[%s] FeatureEngineering failed (%s); falling back to "
                    "compute_normalized_features — contract gaps likely", pair, exc)
        feat = compute_normalized_features(df_oos)

    if feat is None or len(feat) < seq_len + lookahead:
        return {"pair": pair, "error": f"insufficient OOS rows: {len(feat) if feat is not None else 0}"}

    # Temporarily place the OOS-trained model at base_models_dir/pair/ so per-pair
    # routing finds it. We must NOT clobber the existing model — caller renames.
    # This function assumes model_dir is the active per-pair dir.
    ev = GateEvaluator(model_dir=str(base_models_dir), use_per_pair_routing=True)
    ev.load_models(require_tcn=False)

    # Label alignment. FeatureEngineering may drop warmup rows, so `feat` and
    # `df_oos` are not guaranteed to be positionally aligned; indexing labels
    # off df_oos by the feature-row position would silently mislabel every bar.
    # Prefer feat's own close column; otherwise align on time.
    if "close" in feat.columns:
        closes = feat["close"].to_numpy(dtype=float)
    elif "time" in feat.columns and "time" in df_oos.columns:
        _m = feat[["time"]].merge(df_oos[["time", "close"]], on="time", how="left")
        closes = _m["close"].to_numpy(dtype=float)
    else:
        if len(feat) != len(df_oos):
            return {"pair": pair,
                    "error": f"cannot align labels: len(feat)={len(feat)} "
                             f"!= len(df_oos)={len(df_oos)} and no time/close column"}
        closes = df_oos["close"].to_numpy(dtype=float)

    correct = 0
    total = 0
    long_correct = long_total = short_correct = short_total = 0
    for i in range(seq_len, len(feat) - lookahead):
        row = feat.iloc[i - seq_len + 1 : i + 1]
        future_close = closes[i + lookahead]
        current_close = closes[i]
        if not (future_close == future_close and current_close == current_close):
            continue  # NaN from a failed alignment — skip rather than mislabel
        actual_dir = "LONG" if future_close > current_close else "SHORT"
        try:
            res = ev.evaluate_all_gates(row, instrument=pair)
            if not res or not isinstance(res, dict):
                continue
            pred = str(res.get("direction", "")).upper()
            if pred not in ("LONG", "SHORT"):
                continue
            total += 1
            if pred == actual_dir:
                correct += 1
            if actual_dir == "LONG":
                long_total += 1
                if pred == "LONG":
                    long_correct += 1
            else:
                short_total += 1
                if pred == "SHORT":
                    short_correct += 1
        except Exception:
            continue

    if total == 0:
        return {"pair": pair, "error": "no predictions returned"}

    accuracy = correct / total
    long_acc = long_correct / max(long_total, 1)
    short_acc = short_correct / max(short_total, 1)

    # --- Controls (2026-07-31) -------------------------------------------
    # Raw accuracy alone is not evidence of edge. On the 2026-06-17..07-31
    # USD_JPY M15 window, 57.83% of 24-bar futures closed UP, so a model that
    # always says LONG scores 57.83%. The old gate (accuracy >= 0.52) passed
    # such a model with a "PASS" verdict; the measured model scored 57.80% —
    # 0.03pp WORSE than the trivial control — and still passed.
    #
    # A directional model must beat the majority-class control, and must not
    # be achieving that purely by class-collapsing into the drifting side,
    # which is what balanced accuracy catches.
    majority_accuracy = max(long_total, short_total) / total
    balanced_accuracy = (long_acc + short_acc) / 2.0
    edge_vs_majority = accuracy - majority_accuracy

    passed_gate = (
        accuracy >= RAW_MIN_ACCURACY
        and edge_vs_majority >= MIN_EDGE_VS_MAJORITY
        and balanced_accuracy >= MIN_BALANCED_ACCURACY
    )

    return {
        "pair": pair,
        "accuracy": accuracy,
        "long_accuracy": long_acc,
        "short_accuracy": short_acc,
        "n_predictions": total,
        "n_correct": correct,
        # Controls — always reported so a PASS can be audited.
        "majority_class_accuracy": majority_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "edge_vs_majority": edge_vs_majority,
        "gate_thresholds": {
            "raw_min_accuracy": RAW_MIN_ACCURACY,
            "min_edge_vs_majority": MIN_EDGE_VS_MAJORITY,
            "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
        },
        "passed_gate": passed_gate,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="EUR_USD,EUR_JPY,USD_JPY,USD_CAD")
    ap.add_argument("--granularity", default="H1")
    ap.add_argument("--candles", type=int, default=25000, help="Training candles per pair")
    ap.add_argument("--oos-bars", type=int, default=700, help="OOS test bars (~30 days at H1)")
    ap.add_argument("--lookahead", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument(
        "--save-suffix",
        default="_oos_train",
        help="Suffix for OOS-trained model dirs",
    )
    args = ap.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    base_models = REPO_ROOT / "trained_data" / "models"
    results = []

    for pair in pairs:
        log.info("=" * 60)
        log.info("PATH 4 OOS for %s", pair)
        log.info("=" * 60)

        # 1. Fetch (candles + oos_bars) bars
        total = args.candles + args.oos_bars
        log.info("[%s] fetching %d %s candles (paginated)...", pair, total, args.granularity)
        df_full = fetch_candles(pair, args.granularity, total)
        log.info("[%s] fetched %d candles, span %s → %s", pair, len(df_full),
                 df_full.index[0], df_full.index[-1])

        # 2. Split: oldest `candles` rows → train, newest `oos_bars` → OOS test
        df_train = df_full.iloc[: args.candles].copy()
        df_oos = df_full.iloc[args.candles - args.seq_len :].copy()  # include seq_len of overlap so first OOS prediction has 60-bar context
        log.info("[%s] train rows %d (%s → %s)", pair, len(df_train),
                 df_train.index[0], df_train.index[-1])
        log.info("[%s] OOS test rows %d (%s → %s) — true post-training data", pair, len(df_oos),
                 df_oos.index[0], df_oos.index[-1])

        # 3. Retrain master on train fold
        oos_save_dir = base_models / f"{pair}{args.save_suffix}"
        try:
            metrics = train_master_on_split(pair, df_train, oos_save_dir, args.lookahead, args.threshold)
        except Exception as e:
            log.error("[%s] train failed: %s", pair, e, exc_info=True)
            results.append({"pair": pair, "error": f"train failed: {e}"})
            continue

        # 4. Holdout: temporarily swap pair's prod dir with OOS-trained dir
        prod_dir = base_models / pair
        backup_dir = base_models / f"{pair}_path4_backup"

        # 2026-07-31: train_master_on_split writes ONLY transformer_direction.*
        # into oos_save_dir. The swap below then made that the whole per-pair
        # dir, so the momentum / confidence / risk heads vanished for the
        # duration of the holdout — GateEvaluator logged "momentum=none" and
        # scored a transformer-only stack while appearing to score the full
        # one. Copy the sibling heads across (never the transformer, which
        # must stay the OOS-trained one) so the holdout exercises every gate.
        # Direction-head artifacts must NEVER be copied — the OOS model owns
        # them. transformer_direction.* is obvious, but direction_scaler.pkl and
        # feature_indices.pkl are ALSO written by the transformer trainer (via
        # src/training/feature_alignment.py) and copying the incumbent's copies
        # over silently corrupts the OOS model's inference contract, making it
        # abstain on every bar.
        _DIRECTION_HEAD_FILES = {"direction_scaler.pkl", "feature_indices.pkl"}
        if prod_dir.exists():
            for src in sorted(prod_dir.iterdir()):
                if (
                    src.is_dir()
                    or src.name.startswith("transformer_direction.")
                    or src.name in _DIRECTION_HEAD_FILES
                ):
                    continue
                shutil.copy2(src, oos_save_dir / src.name)
            log.info(
                "[%s] copied %d sibling head/config file(s) into OOS dir so the "
                "full gate stack is exercised",
                pair,
                sum(
                    1 for f in prod_dir.iterdir()
                    if f.is_file() and not f.name.startswith("transformer_direction.")
                ),
            )

        if prod_dir.exists():
            prod_dir.rename(backup_dir)
        oos_save_dir.rename(prod_dir)

        try:
            res = holdout_oos(pair, df_oos, prod_dir, base_models, args.seq_len, args.lookahead)
        except Exception as e:
            log.error("[%s] holdout failed: %s", pair, e, exc_info=True)
            res = {"pair": pair, "error": f"holdout failed: {e}"}
        finally:
            # ALWAYS restore
            prod_dir.rename(oos_save_dir)
            if backup_dir.exists():
                backup_dir.rename(prod_dir)

        # Augment with metrics from training
        res["train_val_acc"] = metrics.get("val_accuracy") if "metrics" in dir() else None
        res["train_val_balanced_acc"] = metrics.get("val_balanced_accuracy")
        res["oos_bars"] = args.oos_bars
        res["train_candles"] = args.candles
        results.append(res)

        if "accuracy" in res:
            log.info(
                "[%s] OOS HOLDOUT: acc=%.2f%% vs majority-class %.2f%% "
                "(edge %+.2fpp) balanced=%.2f%% n=%d  long=%.1f%% short=%.1f%%  %s",
                pair, res["accuracy"] * 100, res["majority_class_accuracy"] * 100,
                res["edge_vs_majority"] * 100, res["balanced_accuracy"] * 100,
                res["n_predictions"], res["long_accuracy"] * 100,
                res["short_accuracy"] * 100,
                "✓" if res["passed_gate"] else "✗",
            )
        else:
            log.warning("[%s] OOS HOLDOUT: %s", pair, res.get("error"))

    # Save report
    report_path = REPO_ROOT / "trained_data" / "h1_path4_true_oos_eval.json"
    payload = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "lookahead": args.lookahead,
        "seq_len": args.seq_len,
        "train_candles": args.candles,
        "oos_bars": args.oos_bars,
        "granularity": args.granularity,
        "gate": {
            "raw_min_accuracy": RAW_MIN_ACCURACY,
            "min_edge_vs_majority": MIN_EDGE_VS_MAJORITY,
            "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
        },
        "results": results,
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    print()
    print("=" * 88)
    # granularity was previously hardcoded "H1" here regardless of the flag —
    # that mislabel made M15 runs read as H1 in the report.
    print(f"PATH 4 (TRUE OOS) — {args.granularity} lookahead={args.lookahead}, "
          f"train={args.candles}, oos={args.oos_bars}")
    print("=" * 88)
    print(f'{"pair":10} {"train_val":>9} {"oos_acc":>9} {"majority":>9} '
          f'{"edge":>8} {"balanced":>9} {"oos_n":>6} {"verdict":>8}')
    print("-" * 88)
    for r in results:
        if "accuracy" in r:
            tv = r.get("train_val_balanced_acc") or 0
            print(f'{r["pair"]:10} {tv*100:>8.1f}% {r["accuracy"]*100:>8.1f}% '
                  f'{r["majority_class_accuracy"]*100:>8.1f}% '
                  f'{r["edge_vs_majority"]*100:>+7.2f}p '
                  f'{r["balanced_accuracy"]*100:>8.1f}% {r["n_predictions"]:>6d} '
                  f'{"✓" if r["passed_gate"] else "✗":>8}')
        else:
            print(f'{r["pair"]:10} ERROR: {r.get("error")}')
    print(f'\nReport: {report_path}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
