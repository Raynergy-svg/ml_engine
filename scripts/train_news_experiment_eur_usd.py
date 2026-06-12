#!/usr/bin/env python3
"""
News-fused Transformer experiment — EUR_USD M15.

Goal: measure whether FinBERT + ForexFactory calendar news features push
val_accuracy above the 52% price-only ceiling established 2026-06-10 (dad8624).

Decision rule (from docs/strategy.md):
  SHIP   → val_acc >= 0.55 AND gap < 0.10
  SHELVE → anything less (news doesn't add lift; explore other data sources)

Usage:
    cd ~/Documents/ml_engine
    python scripts/train_news_experiment_eur_usd.py

Runtime: ~15-25 min total (ForexFactory backfill ~4 min, FinBERT embed ~5 min,
training ~10 min on M1 Metal, faster if cache exists from prior run).

Architecture: this script is a thin harness around existing, tested components:
  - ForexFactoryNewsSource (src/data/news/source.py)  — HTML scraper + parquet cache
  - FinBERTEmbedder (src/data/news/embedder.py)       — ProsusAI/finbert, 768-dim
  - load_direction_data with news_* kwargs             — Stage 4-A fusion (modular_data_loaders.py)
  - TransformerDirectionTrainer                        — train/save/metrics (same as M1 script)
  - _quarantine_if_overshipped                         — 10% gap gate (operator directive 2026-05-13)
"""
import gc
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── tf-metal / environment setup (must be first) ─────────────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

env_path = Path(__file__).resolve().parent.parent / ".env.local"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("news_experiment")

# ── Constants ─────────────────────────────────────────────────────────────────
INSTRUMENT = "EUR_USD"
GRANULARITY = "M15"
CANDLES = 65_000             # ~22 months of M15 bars. MUST match the cached
                             # EUR_USD_M15_65000.csv that the price-only baseline
                             # (commit a5748e0, 2026-06-10) trained on — otherwise
                             # the news-vs-price comparison is not apples-to-apples
                             # and fetch_or_load would re-fetch from OANDA.
DIRECTION_LOOKAHEAD = 24      # 24 M15 bars = 6 hours (same as train_single_model_m1.py)
NEWS_PCA_N_COMPONENTS = 32   # PCA compression of 768-dim FinBERT → 32 (per strategy.md design)
NEWS_LOOKBACK_HOURS = 24      # how far back to look for news events per bar
NEWS_BACKFILL_MONTHS = 22    # ForexFactory historical scrape: ~88 weekly pages (~4 min)
# Controlled baseline: the SAME-CSV price-only transformer measured
# train_acc=0.7637 / val_acc=0.5192 / gap=0.2445 (QUARANTINED) on 2026-06-10
# (trained_data/models/EUR_USD/training_status.json, git a5748e0). News must
# beat THIS val_acc, not a rounded doc number.
PRICE_ONLY_BASELINE = 0.5192  # honest controlled val_acc on the identical 65k CSV
PRICE_ONLY_BASELINE_GAP = 0.2445  # baseline overfit gap — news must shrink this < 0.10
SHIP_THRESHOLD_VAL = 0.55    # val_acc needed to ship
HARD_MAX_GAP = 0.10          # 10% gap ship gate (operator directive 2026-05-13)
MAX_GAP = 0.06               # cosmetic PASS/FAIL logging threshold

MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")

# M1-optimised transformer hyperparams (from train_single_model_m1.py DEFAULT_PARAMS EUR_USD)
TRANSFORMER_PARAMS = {
    "seq_len": 90,
    "batch_size": 128,
    "lr": 0.000415,
    "patience": 25,
    "dropout_rate": 0.363,
    "l2_reg": 0.000098,
}


# ── Utility helpers (copied from train_single_model_m1.py) ───────────────────

def _read_trainer_metrics(trainer) -> dict:
    m = getattr(trainer, "metrics", None)
    if isinstance(m, dict) and m:
        return m
    if hasattr(trainer, "get_metrics"):
        try:
            got = trainer.get_metrics() or {}
            return got if isinstance(got, dict) else {}
        except Exception:
            pass
    return {}


def _snapshot_existing_artifacts(model_path) -> "Path | None":
    path = Path(model_path)
    siblings = [p for p in path.parent.glob(f"{path.stem}*") if p.is_file()]
    if not siblings:
        return None
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rollback_dir = path.parent / "_rollback" / f"{path.stem}-{ts}"
    rollback_dir.mkdir(parents=True, exist_ok=True)
    for s in siblings:
        shutil.copy2(s, rollback_dir / s.name)
    return rollback_dir


def _quarantine_if_overshipped(saved_paths, train_acc, val_acc, instrument, model_name, restore_dir=None):
    gap = abs(train_acc - val_acc)
    if gap <= HARD_MAX_GAP:
        return {"quarantined": False, "gap": gap, "quarantine_dir": None}

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    qdir = MODELS_DIR / instrument / "_quarantine" / f"{model_name}-{ts}"
    qdir.mkdir(parents=True, exist_ok=True)
    moved = []
    live_dir = None
    for p in saved_paths:
        path = Path(p)
        live_dir = path.parent
        stem = path.stem
        for sibling in path.parent.glob(f"{stem}*"):
            if sibling.is_file():
                dest = qdir / sibling.name
                sibling.rename(dest)
                moved.append(str(dest))
    restored = 0
    if restore_dir is not None and live_dir is not None:
        rb = Path(restore_dir)
        if rb.is_dir():
            for f in rb.iterdir():
                if f.is_file():
                    shutil.copy2(f, live_dir / f.name)
                    restored += 1
    logger.error(
        f"🚫 SHIP-BLOCKED: {instrument}/{model_name} gap={gap:.4f} > {HARD_MAX_GAP} "
        f"(train={train_acc:.4f}, val={val_acc:.4f}). Quarantined {len(moved)} file(s) → {qdir}; "
        f"restored {restored} prior artifact(s). Operator rule: no models shipped > 10% gap."
    )
    return {"quarantined": True, "gap": gap, "quarantine_dir": str(qdir)}


def gc_cleanup():
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


# ── Stage 1: Fetch / load OANDA candles ─────────────────────────────────────

def fetch_or_load(instrument: str, candles: int, granularity: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{instrument}_{granularity}_{candles}.csv"

    if cache_file.exists():
        logger.info(f"Loading cached candles: {cache_file}")
        import pandas as pd
        df = pd.read_csv(cache_file, parse_dates=["time"])
        return df

    import pandas as pd
    from src.utils.oanda_practice import OandaPracticeClient
    client = OandaPracticeClient.from_env()
    all_rows = []
    remaining = candles
    to_time = None

    logger.info(f"Fetching {candles:,} {granularity} candles for {instrument} from OANDA...")
    while remaining > 0:
        batch = min(remaining, 5000)
        kwargs = {"granularity": granularity, "count": batch}
        if to_time:
            kwargs["to_time"] = to_time
        try:
            response = client.get_candles(instrument, **kwargs)
        except Exception as e:
            logger.warning(f"Fetch error (retrying): {e}")
            time.sleep(2)
            response = client.get_candles(instrument, **kwargs)

        candles_list = response.get("candles", [])
        if not candles_list:
            break
        for c in candles_list:
            mid = c.get("mid", {})
            all_rows.append({
                "time": c.get("time"),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
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

    logger.info(f"✓ {instrument}: {len(df):,} candles fetched")
    return df


def apply_features(df):
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering
    cfg = load_config(CONFIG_PATH)
    fe = FeatureEngineering(cfg)
    return fe.create_features(df.copy(), include_all=True)


# ── Stage 2: Backfill ForexFactory historical news ───────────────────────────

def backfill_news(news_source, instrument: str, months_back: int):
    """Scrape ForexFactory historical calendar and cache to parquet.

    ForexFactory returns one page per week. ~88 pages for 22 months.
    The scraper enforces a 2.5s polite delay between pages, so this takes ~4 min.
    A Parquet cache means subsequent runs skip the scrape entirely.
    """
    since = datetime(2024, 8, 1, tzinfo=timezone.utc)
    until = datetime.now(tz=timezone.utc)
    logger.info(
        f"Backfilling ForexFactory news for {instrument} from {since.date()} to {until.date()} "
        f"(~{months_back} months, ~88 pages, ~4 min first run)"
    )
    try:
        events = news_source.fetch_events_historical(instrument, since=since, until=until)
        logger.info(f"✓ ForexFactory backfill: {len(events)} events cached for {instrument}")
        return events
    except Exception as e:
        logger.error(f"ForexFactory backfill failed: {e}. Continuing without news — experiment will show price-only accuracy.")
        return []


# ── Stage 3: Run the experiment ───────────────────────────────────────────────

def run_experiment():
    t0 = time.time()
    logger.info("=" * 72)
    logger.info("NEWS-FUSED TRANSFORMER EXPERIMENT — EUR_USD M15")
    logger.info(f"Baseline (price-only, dad8624): val_acc ≈ {PRICE_ONLY_BASELINE:.2%}")
    logger.info(f"Ship threshold: val_acc ≥ {SHIP_THRESHOLD_VAL:.2%} AND gap < {HARD_MAX_GAP:.2%}")
    logger.info("=" * 72)

    # 1. Price data
    logger.info("Stage 1/5: Loading price data...")
    df_raw = fetch_or_load(INSTRUMENT, CANDLES, GRANULARITY)
    if df_raw.empty:
        logger.error("No price data — aborting.")
        return

    # 2. Feature engineering (price-only, same pipeline as prod)
    logger.info("Stage 2/5: Applying feature engineering...")
    df_feat = apply_features(df_raw)
    if df_feat.empty:
        logger.error("Feature engineering produced empty dataframe — aborting.")
        return
    logger.info(f"✓ Features: {df_feat.shape[1]} columns, {len(df_feat):,} rows")

    # 3. News source + embedder init
    logger.info("Stage 3/5: Initialising news pipeline (ForexFactory + FinBERT)...")
    from src.data.news.source import ForexFactoryNewsSource
    from src.data.news.embedder import FinBERTEmbedder

    news_cache_dir = str(PROJECT_ROOT / "trained_data" / "news")
    news_source = ForexFactoryNewsSource(cache_dir=news_cache_dir)
    news_embedder = FinBERTEmbedder(device="cpu")  # cpu avoids MPS/TF deadlock
    # Note: FinBERTEmbedder sets torch.set_num_threads(1) internally to prevent
    # TF/PyTorch deadlock when both Metal and PyTorch are loaded in the same process.

    # Backfill ForexFactory historical data (cached parquet; skipped on reruns)
    backfill_news(news_source, INSTRUMENT, NEWS_BACKFILL_MONTHS)

    # 4. load_direction_data WITH news fusion (Stage 4-A)
    logger.info("Stage 4/5: Building training dataset with news fusion (Stage 4-A)...")
    logger.info(f"  news_pca_n_components={NEWS_PCA_N_COMPONENTS}, lookback={NEWS_LOOKBACK_HOURS}h")
    from src.core.modular_data_loaders import load_direction_data, FEATURE_PIPELINE_VERSION

    dir_data = load_direction_data(
        df_feat,
        lookahead=DIRECTION_LOOKAHEAD,
        gap=DIRECTION_LOOKAHEAD,     # label embargo = lookahead (no leakage)
        news_source=news_source,
        news_embedder=news_embedder,
        pair=INSTRUMENT,
        news_lookback_window_hours=NEWS_LOOKBACK_HOURS,
        news_pca_n_components=NEWS_PCA_N_COMPONENTS,
    )
    if not dir_data:
        logger.error("load_direction_data returned empty — aborting.")
        return

    news_pca_obj = dir_data.get("news_pca")
    if news_pca_obj is None:
        logger.warning(
            "Stage 4-A returned news_pca=None — news features were not appended. "
            "Possible causes: no events in the backfill window, align_news_to_bars error. "
            "This run will measure price-only accuracy (not the news experiment)."
        )
    else:
        n_news_features = NEWS_PCA_N_COMPONENTS + len(dir_data.get("news_event_class_count_columns") or [])
        logger.info(f"✓ Stage 4-A: {n_news_features} news features appended to X_train/X_val")

    X_train, y_train = dir_data["X_train"], dir_data["y_train"]
    X_val, y_val = dir_data["X_val"], dir_data["y_val"]
    logger.info(
        f"  X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, "
        f"features: {len(dir_data.get('feature_names', []))}"
    )

    # 5. Train TransformerDirectionTrainer
    logger.info("Stage 5/5: Training transformer (M1-optimised params)...")
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    cfg = TrainerConfig(
        epochs=200,
        batch_size=TRANSFORMER_PARAMS["batch_size"],
        learning_rate=TRANSFORMER_PARAMS["lr"],
        patience=TRANSFORMER_PARAMS["patience"],
        seq_len=TRANSFORMER_PARAMS["seq_len"],
        min_epochs=30,
        label_smoothing=0.1,
        use_augmentation=True,
        use_class_weights=True,
        use_ema=True,
        use_ewc=False,
        use_feature_selection=True,
        use_replay_buffer=False,
        overfit_threshold=0.04,
        critical_threshold=0.08,
        max_acceptable_gap=MAX_GAP,
    )

    save_dir = MODELS_DIR / INSTRUMENT
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "transformer_direction.keras"
    snapshot_dir = _snapshot_existing_artifacts(save_path)

    trainer = TransformerDirectionTrainer(cfg)
    # Forward ALL dir_data inference-contract fields including news_pca.
    # trainer.train() reads them via **kwargs (transformer_trainer.py:2803-2811).
    # trainer.save() embeds them in the .meta.pkl sidecar so gates._load_transformer
    # can reconstruct news features at inference time.
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        w_train=dir_data.get("w_train"),
        w_val=dir_data.get("w_val"),
        feature_names=dir_data.get("feature_names"),
        feature_pipeline_version=dir_data.get("feature_pipeline_version", FEATURE_PIPELINE_VERSION),
        regime_quantiles=dir_data.get("regime_quantiles"),
        regime_atr_col=dir_data.get("regime_atr_col"),
        # Stage 4-A news inference contract (None if news not available)
        news_pca=dir_data.get("news_pca"),
        news_event_class_count_columns=dir_data.get("news_event_class_count_columns"),
        news_lookback_window_hours=dir_data.get("news_lookback_window_hours"),
        news_pca_n_components=dir_data.get("news_pca_n_components"),
    )

    trainer.save(str(save_path))

    # ── Metrics + ship gate ───────────────────────────────────────────────────
    metrics = _read_trainer_metrics(trainer)
    train_acc = float(metrics.get("train_accuracy", 0.0))
    val_acc = float(metrics.get("val_accuracy", 0.0))
    ship = _quarantine_if_overshipped(
        [save_path], train_acc, val_acc, INSTRUMENT, "transformer_direction",
        restore_dir=snapshot_dir,
    )
    gap = ship["gap"]
    quarantined = ship["quarantined"]
    news_lift_pp = round((val_acc - PRICE_ONLY_BASELINE) * 100, 2)

    # ── RESULT line (machine-parseable for downstream automation) ─────────────
    elapsed = round(time.time() - t0, 1)
    result = {
        "instrument": INSTRUMENT,
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "gap": round(gap, 4),
        "price_only_val_acc": PRICE_ONLY_BASELINE,   # controlled same-CSV baseline
        "price_only_gap": PRICE_ONLY_BASELINE_GAP,   # baseline overfit gap (0.2445)
        "news_lift_pp": news_lift_pp,                # pp lift vs controlled baseline
        "gap_delta": round(gap - PRICE_ONLY_BASELINE_GAP, 4),  # <0 = news reduced overfit
        "news_pca_active": news_pca_obj is not None,
        "ship": bool(news_pca_obj is not None and val_acc >= SHIP_THRESHOLD_VAL and not quarantined),
        "quarantined": quarantined,
        "quarantine_dir": ship["quarantine_dir"],
        "elapsed_s": elapsed,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"RESULT:{json.dumps(result)}")
    logger.info("=" * 72)

    # ── Human-readable verdict ────────────────────────────────────────────────
    if not result["news_pca_active"]:
        logger.warning("⚠  News PCA was NOT active — result is price-only, not news experiment.")
    elif quarantined:
        logger.error(
            f"🚫 QUARANTINED: gap={gap:.2%} > 10%. Model NOT shippable. "
            f"News added {news_lift_pp:+.2f}pp vs baseline but overfit."
        )
    elif val_acc >= SHIP_THRESHOLD_VAL and gap < HARD_MAX_GAP:
        logger.info(
            f"✅ SHIP: val_acc={val_acc:.2%} ≥ {SHIP_THRESHOLD_VAL:.2%}, "
            f"gap={gap:.2%} < 10%. News adds {news_lift_pp:+.2f}pp lift. "
            f"Model saved to {save_path}. "
            f"Next: run backtest, then tell David to unhalt the bot."
        )
    else:
        logger.warning(
            f"⚠  SHELVE: val_acc={val_acc:.2%} < {SHIP_THRESHOLD_VAL:.2%} threshold "
            f"(news adds {news_lift_pp:+.2f}pp — not enough). "
            f"News/macro signal does not provide sufficient lift on M15 EUR_USD. "
            f"Consider: longer timeframe (H1/H4), different asset class, "
            f"or alternative data source."
        )

    gc_cleanup()
    logger.info(f"Done in {elapsed:.0f}s")
    return result


if __name__ == "__main__":
    run_experiment()
