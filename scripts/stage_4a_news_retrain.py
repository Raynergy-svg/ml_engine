"""Stage 4-A end-to-end retrain script — single pair news-fusion vs price-only.

Bypasses the joint_trainer / correlation_transfer orchestration to keep the
first Stage 4-A holdout number tight and controllable. Once we have a real
news-vs-no-news lift number, the orchestration plumbing gets a permanent
upgrade.

Usage:
    python scripts/stage_4a_news_retrain.py --pair EUR_USD --candles 25000

Produces:
    trained_data/models/{pair}_news/transformer_direction.{keras,meta.pkl,...}

Holdout follow-up:
    python scripts/per_pair_holdout_eval.py --pairs {pair}_news \\
        --granularity H1 --candles 5000 --windows 1500
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("stage_4a_news_retrain")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="EUR_USD")
    ap.add_argument("--granularity", default="H1")
    ap.add_argument("--candles", type=int, default=25000)
    ap.add_argument("--lookahead", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument(
        "--news-cache",
        default=str(REPO_ROOT / "trained_data" / "news"),
    )
    ap.add_argument(
        "--save-suffix",
        default="_news",
        help="Suffix for the news-fused model dir (e.g. trained_data/models/EUR_USD_news)",
    )
    ap.add_argument("--news-pca-n", type=int, default=64)
    ap.add_argument("--news-lookback-h", type=int, default=24)
    args = ap.parse_args()

    log.info("=" * 60)
    log.info("Stage 4-A retrain: pair=%s gran=%s candles=%d", args.pair, args.granularity, args.candles)
    log.info("News pca_n=%d lookback_h=%d cache=%s", args.news_pca_n, args.news_lookback_h, args.news_cache)
    log.info("=" * 60)

    # 1. Fetch candles from OANDA
    from src.utils.oanda_practice import OandaPracticeClient
    oanda = OandaPracticeClient.from_env()
    log.info("Fetching %d %s candles for %s", args.candles, args.granularity, args.pair)
    candles_raw = oanda.get_candles(args.pair, granularity=args.granularity, count=args.candles)

    import pandas as pd
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
                "volume": float(c.get("volume", 0)),
            })
        df = pd.DataFrame(rows)
    elif isinstance(candles_raw, pd.DataFrame):
        df = candles_raw
    else:
        log.error("Unexpected candles type: %s", type(candles_raw))
        return 1

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time", drop=False)
    log.info("Fetched %d candles, span %s → %s", len(df), df.index[0], df.index[-1])

    # 2. Compute price features
    from src.core.modular_data_loaders import compute_normalized_features, load_direction_data
    df_feat = compute_normalized_features(df)
    log.info("Price features computed: %d cols", len(df_feat.columns))

    # 3. Build news source + embedder
    from src.data.news.source import ForexFactoryNewsSource
    from src.data.news.embedder import FinBERTEmbedder
    news_source = ForexFactoryNewsSource(cache_dir=args.news_cache)
    news_embedder = FinBERTEmbedder(device="cpu")
    log.info("News pipeline initialized (cache=%s)", args.news_cache)

    # 4. load_direction_data with news kwargs
    log.info("Calling load_direction_data with news fusion enabled...")
    dir_result = load_direction_data(
        df_feat,
        apply_scaler=False,
        lookahead=args.lookahead,
        threshold=args.threshold,
        news_source=news_source,
        news_embedder=news_embedder,
        pair=args.pair,
        news_lookback_window_hours=args.news_lookback_h,
        news_pca_n_components=args.news_pca_n,
    )
    if dir_result is None:
        log.error("load_direction_data returned None — aborting")
        return 1
    log.info(
        "Loader done: X_train shape=%s feature_count=%d news_pca_present=%s",
        dir_result["X_train"].shape,
        len(dir_result["feature_names"]),
        dir_result.get("news_pca") is not None,
    )

    if dir_result.get("news_pca") is None:
        log.error("News fusion didn't activate — check news kwargs and parquet cache")
        return 1

    # 5. Train
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    trainer_cfg = TrainerConfig()
    trainer = TransformerDirectionTrainer(trainer_cfg)
    log.info("Starting trainer.train()...")
    metrics = trainer.train(
        dir_result["X_train"], dir_result["y_train"],
        dir_result["X_val"], dir_result["y_val"],
        feature_names=dir_result["feature_names"],
        w_train=dir_result.get("w_train"),
        w_val=dir_result.get("w_val"),
        instrument=args.pair,
        skip_scaling=False,
        regime_quantiles=dir_result.get("regime_quantiles"),
        regime_atr_col=dir_result.get("regime_atr_col"),
        feature_pipeline_version=dir_result.get("feature_pipeline_version"),
        news_pca=dir_result.get("news_pca"),
        news_event_class_count_columns=dir_result.get("news_event_class_count_columns"),
        news_lookback_window_hours=dir_result.get("news_lookback_window_hours"),
        news_pca_n_components=dir_result.get("news_pca_n_components"),
    )
    log.info(
        "Training done: val_acc=%.4f val_balanced_acc=%.4f",
        metrics.get("val_accuracy", 0),
        metrics.get("val_balanced_accuracy", 0),
    )

    # 6. Save
    save_dir = REPO_ROOT / "trained_data" / "models" / f"{args.pair}{args.save_suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "transformer_direction.keras"
    trainer.save(str(save_path), instrument=args.pair)
    log.info("Saved to %s", save_path)

    log.info("=" * 60)
    log.info("Stage 4-A retrain COMPLETE for %s", args.pair)
    log.info("  Save dir: %s", save_dir)
    log.info("  val_balanced_acc: %.4f (compare to price-only baseline)", metrics.get("val_balanced_accuracy", 0))
    log.info("=" * 60)
    log.info("Next: run holdout to compare news vs no-news lift:")
    log.info(
        "  python scripts/per_pair_holdout_eval.py --pairs %s%s --granularity %s --candles 5000 --windows 1500",
        args.pair, args.save_suffix, args.granularity,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
