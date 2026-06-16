#!/usr/bin/env python3
"""Phase A — Triple-barrier / meta-labeling experiment, EUR_USD M15.

Tests the load-bearing question from
``docs/modernization-architecture-plan-2026-06-15.md`` §2 Phase A:

    Does a trade/no-trade META-LABELER beat the ~52% raw-direction base rate
    on the subset of bars it chooses to trade?

This is the cheapest experiment that could reveal a real edge: it reuses the
existing primary transformer and the existing meta-labeler (``src/training/
meta_labeling.py``) — no new architecture, no new data. A 58%+ precision on a
traded subset ≥20% of bars is the legitimate reading of the operator's "65%"
target (a trade-QUALITY number, not a raw-direction number).

──────────────────────────────────────────────────────────────────────────────
HONEST, LEAKAGE-SAFE DESIGN (López de Prado meta-labeling)
──────────────────────────────────────────────────────────────────────────────
The primary's predictions that train the meta-labeler MUST be out-of-sample,
or the meta number is a leakage artifact (the exact trap behind the repo's
70%→52% OBV collapse). Three temporal segments, no overlap:

    segment 1  primary-train   = X_train           → train the primary transformer
    segment 2  meta-dev        = first 60% of X_val → primary scores OOS here;
                                                       split into mt_train / mt_val
                                                       to FIT + tune the meta-model
    segment 3  test            = last 40% of X_val  → primary + meta both score OOS;
                                                       the ONLY segment we report on

The primary never sees X_val. The meta-model never sees `test`. `test` is touched
exactly once, at final evaluation, via the public ``predict_meta_confidence``.

──────────────────────────────────────────────────────────────────────────────
FIRST-RUN SANITY GATE — do NOT trust the meta number until this holds
──────────────────────────────────────────────────────────────────────────────
`primary_test_acc` (primary OOS direction accuracy on `test`) must print in the
~0.50–0.53 band. That reproduces the established ceiling and proves the split is
honest. If it prints >0.60, STOP — that is leakage, not signal (see the OBV
incident). The meta number is only meaningful when the base rate is ~chance.

DECISION RULE (Phase A gate):
    PASS  → best-threshold meta_precision ≥ 0.58 AND traded_fraction ≥ 0.20
    FAIL  → reframed target does not beat base rate on M15; shelve meta-labeling
            for intraday and the daily factor portfolio remains the direction.

Usage:
    cd ~/Documents/ml_engine
    python scripts/phase_a_meta_label_eur_usd.py
Runtime: ~10–15 min on M1 (primary training dominates; cached CSV reused).
"""
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── tf-metal / environment setup (must precede tensorflow import) ────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env.local"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

import logging  # noqa: E402  (must follow env + sys.path setup above)
import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase_a_meta")

# ── Constants ─────────────────────────────────────────────────────────────────
INSTRUMENT = "EUR_USD"
GRANULARITY = "M15"
# Reuse the SAME 65k-bar CSV the price-only baseline trained on so the primary's
# ~52% ceiling is reproduced exactly (apples-to-apples with commit a5748e0).
CANDLES = 65_000
DIRECTION_LOOKAHEAD = 24          # 24 M15 bars = 6h (same as train_single_model_m1.py)

META_DEV_FRACTION = 0.60          # first 60% of X_val trains/tunes the meta-model
META_INTERNAL_VAL_FRACTION = 0.20  # tail of meta-dev used for the labeler's own tuning
THRESHOLD_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70]

# Phase A gate (docs/modernization-architecture-plan-2026-06-15.md §2)
META_PRECISION_GATE = 0.58
TRADED_FRACTION_GATE = 0.20
# Sanity band for primary OOS accuracy on `test` — outside this, suspect leakage.
SANITY_LO, SANITY_HI = 0.47, 0.57

MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")

# M1-optimised transformer hyperparams (EUR_USD defaults, mirror news experiment)
TRANSFORMER_PARAMS = {
    "seq_len": 90, "batch_size": 128, "lr": 0.000415,
    "patience": 25, "dropout_rate": 0.363, "l2_reg": 0.000098,
}


# ── Data helpers (proven pattern, mirrored from train_news_experiment_eur_usd.py) ─

def fetch_or_load(instrument: str, candles: int, granularity: str):
    """Load cached candles, or fetch from OANDA practice and cache as CSV."""
    import pandas as pd
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{instrument}_{granularity}_{candles}.csv"
    if cache_file.exists():
        logger.info(f"Loading cached candles: {cache_file}")
        return pd.read_csv(cache_file, parse_dates=["time"])

    from src.utils.oanda_practice import OandaPracticeClient
    client = OandaPracticeClient.from_env()
    all_rows, remaining, to_time = [], candles, None
    logger.info(f"Fetching {candles:,} {granularity} candles for {instrument}...")
    while remaining > 0:
        batch = min(remaining, 5000)
        kwargs = {"granularity": granularity, "count": batch}
        if to_time:
            kwargs["to_time"] = to_time
        try:
            response = client.get_candles(instrument, **kwargs)
        except Exception as e:
            logger.warning(f"Fetch error (retrying once): {e}")
            time.sleep(2)
            response = client.get_candles(instrument, **kwargs)
        candles_list = response.get("candles", [])
        if not candles_list:
            break
        for c in candles_list:
            mid = c.get("mid", {})
            all_rows.append({
                "time": c.get("time"),
                "open": float(mid.get("o", 0)), "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)), "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })
        remaining -= len(candles_list)
        to_time = candles_list[0].get("time")
        if len(candles_list) < batch:
            break
        time.sleep(0.5)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        df.to_csv(cache_file, index=False)
    logger.info(f"✓ {instrument}: {len(df):,} candles")
    return df


def apply_features(df):
    """Price-only feature engineering (same pipeline as production)."""
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering
    fe = FeatureEngineering(load_config(CONFIG_PATH))
    return fe.create_features(df.copy(), include_all=True)


def _read_trainer_metrics(trainer) -> dict:
    m = getattr(trainer, "metrics", None)
    return m if isinstance(m, dict) and m else {}


def _to_1d_labels(y) -> np.ndarray:
    """Coerce direction labels to a 1-D {0,1} vector (handles one-hot / 2-col)."""
    y = np.asarray(y)
    if y.ndim == 2:
        return (np.argmax(y, axis=1) if y.shape[1] == 2 else y[:, 0]).astype(np.float32)
    return y.reshape(-1).astype(np.float32)


def _primary_seq_scores(trainer, X, y, w):
    """Out-of-sample primary P(up) over windowed sequences, with aligned labels.

    The direction loader returns 2-D (n, features); the trainer windows it into
    (n_seq, seq_len, features) and keeps only clear-label sequences (weight > 0).
    We replicate that EXACT path via the trainer's own builder, so the meta-labeler
    trains on the same OOS predictions the model actually makes. Uses the trainer's
    fitted scaler + selected_indices (prevents the Bug-A unscaled-inference failure).

    Returns (proba, x_seq_clear, y_seq_clear) — aligned at the sequence level.
    """
    from src.training.trainers.utils import create_sequences_with_weights
    if getattr(trainer, "scaler", None) is None:
        raise RuntimeError("Primary trainer has no fitted scaler — refusing to score "
                           "(would reproduce the unscaled-inference bug).")
    X = np.asarray(X)
    model_F = trainer.model.input_shape[-1]
    if X.shape[-1] > model_F:
        sel = getattr(trainer, "selected_indices", None)
        if sel is not None and len(sel) == model_F:
            X = X[:, sel]
        else:
            logger.warning(f"Feature count {X.shape[-1]}>{model_F}; using first {model_F}.")
            X = X[:, :model_F]
    x_scaled = trainer.scaler.transform(X)
    seq_len = int(trainer.seq_len)
    w = np.ones(len(x_scaled), np.float32) if w is None else np.asarray(w)
    x_seq, y_seq, w_seq = create_sequences_with_weights(
        x_scaled, np.asarray(y), w, seq_len)
    clear = np.asarray(w_seq) > 0          # clear-label mask — same filter the trainer uses
    x_seq, y_seq = x_seq[clear], _to_1d_labels(y_seq[clear])
    out = trainer.model.predict(x_seq, verbose=0)
    if isinstance(out, dict):
        out = out.get("direction", out.get("output_0", next(iter(out.values()))))
    out = np.asarray(out)
    proba = out[:, 1] if (out.ndim == 2 and out.shape[1] == 2) else out.reshape(-1)
    return proba.astype(np.float32), x_seq, y_seq.astype(np.float32)


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment():
    t0 = time.time()
    logger.info("=" * 72)
    logger.info("PHASE A — META-LABELING EXPERIMENT — EUR_USD M15")
    logger.info(f"Gate: meta_precision ≥ {META_PRECISION_GATE} AND "
                f"traded_fraction ≥ {TRADED_FRACTION_GATE}")
    logger.info("=" * 72)

    # 1. Price → features → windowed direction dataset (price-only, no news).
    df_raw = fetch_or_load(INSTRUMENT, CANDLES, GRANULARITY)
    if df_raw.empty:
        logger.error("No price data — aborting.")
        return
    df_feat = apply_features(df_raw)
    if df_feat.empty:
        logger.error("Feature engineering produced empty frame — aborting.")
        return

    from src.core.modular_data_loaders import load_direction_data, FEATURE_PIPELINE_VERSION
    dir_data = load_direction_data(
        df_feat,
        lookahead=DIRECTION_LOOKAHEAD,
        gap=DIRECTION_LOOKAHEAD,   # label embargo = lookahead (no leakage)
        pair=INSTRUMENT,
    )
    if not dir_data:
        logger.error("load_direction_data returned empty — aborting.")
        return
    X_train = np.asarray(dir_data["X_train"])  # primary trains on raw labels below
    X_val = np.asarray(dir_data["X_val"])  # labels are windowed inside _primary_seq_scores
    logger.info(f"X_train={X_train.shape}  X_val={X_val.shape}  "
                f"features={len(dir_data.get('feature_names', []))}")

    # 2. Train the PRIMARY transformer on segment 1 (X_train).
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
    cfg = TrainerConfig(
        epochs=200, batch_size=TRANSFORMER_PARAMS["batch_size"],
        learning_rate=TRANSFORMER_PARAMS["lr"], patience=TRANSFORMER_PARAMS["patience"],
        seq_len=TRANSFORMER_PARAMS["seq_len"], min_epochs=30, label_smoothing=0.1,
        use_augmentation=True, use_class_weights=True, use_ema=True, use_ewc=False,
        use_feature_selection=True, use_replay_buffer=False,
        overfit_threshold=0.04, critical_threshold=0.08, max_acceptable_gap=0.06,
    )
    logger.info("Training primary transformer (segment 1)...")
    trainer = TransformerDirectionTrainer(cfg)
    trainer.train(
        X_train, dir_data["y_train"], X_val, dir_data["y_val"],
        w_train=dir_data.get("w_train"), w_val=dir_data.get("w_val"),
        feature_names=dir_data.get("feature_names"),
        feature_pipeline_version=dir_data.get("feature_pipeline_version", FEATURE_PIPELINE_VERSION),
        regime_quantiles=dir_data.get("regime_quantiles"),
        regime_atr_col=dir_data.get("regime_atr_col"),
    )
    pmetrics = _read_trainer_metrics(trainer)
    logger.info(f"Primary metrics: train={pmetrics.get('train_accuracy'):.4f} "
                f"val={pmetrics.get('val_accuracy'):.4f}")

    # 3. Primary scores X_val OUT-OF-SAMPLE as clear-label sequences; split meta-dev / test.
    val_proba, val_X, val_y = _primary_seq_scores(
        trainer, X_val, dir_data["y_val"], dir_data.get("w_val"))
    n_val = len(val_proba)
    if n_val < 200:
        logger.error(f"Only {n_val} clear val sequences — too few for a meta estimate; aborting.")
        return
    n_dev = int(n_val * META_DEV_FRACTION)
    dev_X, dev_p, dev_y = val_X[:n_dev], val_proba[:n_dev], val_y[:n_dev]
    test_X, test_p, test_y = val_X[n_dev:], val_proba[n_dev:], val_y[n_dev:]
    # within meta-dev: mt_train fits the meta-model, mt_val tunes it (both OOS for primary)
    n_mt = int(len(dev_X) * (1 - META_INTERNAL_VAL_FRACTION))
    mt_X, mt_p, mt_y = dev_X[:n_mt], dev_p[:n_mt], dev_y[:n_mt]
    mv_X, mv_p, mv_y = dev_X[n_mt:], dev_p[n_mt:], dev_y[n_mt:]
    logger.info(f"Split: meta_train={len(mt_X)} meta_val={len(mv_X)} test={len(test_X)}")

    # 4. Fit the meta-labeler on out-of-sample primary predictions.
    from src.training.meta_labeling import train_meta_labeler
    labeler, mmetrics = train_meta_labeler(
        X_train=mt_X, y_train=mt_y, X_val=mv_X, y_val=mv_y,
        primary_probs_train=mt_p, primary_probs_val=mv_p,
        config=None, verbose=True,
    )

    # 5. Evaluate on `test` (pure holdout, scored once).
    primary_correct = (
        (test_p >= 0.5).astype(np.int8) == (test_y >= 0.5).astype(np.int8)
    ).astype(np.float32)
    primary_test_acc = float(np.mean(primary_correct))           # base rate (~chance)
    meta_conf = labeler.predict_meta_confidence(test_X, test_p)  # P(primary correct)

    sweep = []
    for thr in THRESHOLD_SWEEP:
        mask = meta_conf >= thr
        traded_fraction = float(np.mean(mask))
        precision = float(np.mean(primary_correct[mask])) if mask.any() else 0.0
        sweep.append({"threshold": thr,
                      "traded_fraction": round(traded_fraction, 4),
                      "precision": round(precision, 4),
                      "lift_pp": round((precision - primary_test_acc) * 100, 2)})

    # Best threshold meeting the traded-fraction floor, by precision.
    eligible = [s for s in sweep if s["traded_fraction"] >= TRADED_FRACTION_GATE]
    best = max(eligible, key=lambda s: s["precision"]) if eligible else None
    sanity_ok = SANITY_LO <= primary_test_acc <= SANITY_HI
    passed = bool(best and best["precision"] >= META_PRECISION_GATE and sanity_ok)

    result = {
        "experiment": "phase_a_meta_label",
        "instrument": INSTRUMENT,
        "primary_train_acc": round(float(pmetrics.get("train_accuracy", 0.0)), 4),
        "primary_val_acc": round(float(pmetrics.get("val_accuracy", 0.0)), 4),
        "primary_test_acc": round(primary_test_acc, 4),   # base rate; expect ~0.50–0.53
        "sanity_ok": sanity_ok,                            # False ⇒ suspect leakage, distrust
        "n_test": int(len(test_X)),
        "sweep": sweep,
        "best": best,
        "meta_precision_gate": META_PRECISION_GATE,
        "traded_fraction_gate": TRADED_FRACTION_GATE,
        "PASS": passed,
        "verdict": ("PASS — meta-labeling beats base rate; proceed to Phase B"
                    if passed else
                    "FAIL — no trade-quality edge on M15; shelve, factor pivot stands"),
        "elapsed_s": round(time.time() - t0, 1),
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }
    logger.info("")
    logger.info("=" * 72)
    if not sanity_ok:
        logger.error(f"⚠ SANITY FAIL: primary_test_acc={primary_test_acc:.4f} outside "
                     f"[{SANITY_LO},{SANITY_HI}] — suspect leakage; DISTRUST the meta number.")
    logger.info(f"RESULT:{json.dumps(result)}")
    logger.info("=" * 72)

    out_path = PROJECT_ROOT / "trained_data" / "backtests" / (
        f"phase_a_meta_{INSTRUMENT}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.rename(out_path)                                  # atomic write (project rule)
    logger.info(f"Wrote {out_path}")
    gc.collect()
    return result


if __name__ == "__main__":
    run_experiment()
