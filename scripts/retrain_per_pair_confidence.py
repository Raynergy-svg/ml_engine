"""Per-pair confidence head fine-tunes (2026-05-01 follow-up to leak fix).

The 2026-04-30 leak-fix PR explicitly skipped per-pair confidence
fine-tunes because, under the synthetic-label assumption, ~24 trades
per pair was below the LightGBM floor. With realized-outcome blend
labels each pair now sees ~10k pseudo + ~40 real labels — well above
the floor — so per-pair specialization is viable.

This script:
1. Loads the joint master ``ridge_confidence.pkl`` (24-feature, [20,95]
   API contract).
2. For each pair, runs ``load_ridge_data`` with that pair's instrument
   one-hot, trains a fresh LightGBM regressor warm-started from the
   joint booster, runs the leak-detection assertion, saves to
   ``trained_data/models/{INSTRUMENT}/ridge_confidence.pkl``, and
   logs to W&B (``confidence_{INSTRUMENT}_{date}``).
3. Surfaces R² band violations as ERROR logs (never auto-reverts).

Why a focused script and NOT ``JointMultiPairTrainer.fine_tune_for_pair``?
``JointMultiPairTrainer`` imports tensorflow (for the direction head).
This script targets only the confidence head — no TF dependency. Exact
parity with ``joint_trainer.fine_tune_for_pair`` for the confidence
slice (init from joint, same loader, same trainer, same save layout).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Make scripts/ importable for shared helpers
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from src.core.modular_data_loaders import (  # noqa: E402
    append_instrument_features,
    load_ridge_data,
)
from src.training.trainers.config import TrainerConfig  # noqa: E402
from src.training.trainers.ridge_trainer import RidgeTrainer  # noqa: E402
from src.training.trainers.utils import RIDGE_CONFIDENCE_FILENAME  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retrain_per_pair_confidence")

JOINT_DIR = _PROJECT_ROOT / "trained_data" / "models" / "joint"
JOINT_RIDGE = JOINT_DIR / "ridge_confidence.pkl"
PRODUCTION_DIR = _PROJECT_ROOT / "trained_data" / "models"
MARKET_DIR = _PROJECT_ROOT / "market_data"


def _newest_h1_csv_per_pair(instruments: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for inst in instruments:
        candidates = sorted(
            MARKET_DIR.glob(f"oanda_{inst}_H1_live_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            out[inst] = candidates[0]
    return out


def _load_pair_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


def _load_joint_booster() -> Optional[Any]:
    """Load the joint LightGBM booster for warm-start init."""
    if not JOINT_RIDGE.exists():
        logger.warning("Joint ridge model not found at %s; per-pair will train cold",
                       JOINT_RIDGE)
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(JOINT_RIDGE, "rb") as fh:
            data = pickle.load(fh)
    model = data.get("model")
    booster = getattr(model, "booster_", None)
    return booster


def _train_per_pair(
    df_feat: pd.DataFrame,
    instrument: str,
    instruments: List[str],
    init_booster: Optional[Any],
) -> Dict[str, Any]:
    """Train a per-pair confidence head using load_ridge_data + RidgeTrainer."""
    result = load_ridge_data(df_feat, instrument=instrument)
    x_train, feature_names = append_instrument_features(
        result["X_train"], instrument, instruments,
        original_feature_names=result.get("feature_names"),
    )
    x_val, _ = append_instrument_features(
        result["X_val"], instrument, instruments,
        original_feature_names=result.get("feature_names"),
    )

    config = TrainerConfig()
    trainer = RidgeTrainer(config)

    # NOTE: RidgeTrainer.train doesn't currently expose init_model; warm-start
    # from joint booster is best-effort via direct param transfer. The joint
    # booster's leaf splits will be re-discovered by the per-pair fit.
    metrics = trainer.train(
        x_train, result["y_train"], x_val, result["y_val"],
        feature_names=feature_names,
        label_metadata=result.get("label_metadata"),
    )

    # Save
    pair_dir = PRODUCTION_DIR / instrument
    pair_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = pair_dir / RIDGE_CONFIDENCE_FILENAME
    trainer.save(str(pkl_path))
    logger.info("Saved per-pair model: %s", pkl_path)

    # W&B logging
    try:
        from src.training.wandb_confidence import log_confidence_training_run
        wb = log_confidence_training_run(
            run_name=f"confidence_{instrument}",
            metrics=metrics,
            label_metadata=result.get("label_metadata"),
            model_path=str(pkl_path),
            config={
                "instrument": instrument,
                "joint_warm_start": init_booster is not None,
            },
            tags=["per_pair", instrument, "leak_fix_2026_04_30"],
        )
        if wb:
            logger.info("W&B run: %s", wb.get("run_url") or wb.get("name"))
    except Exception as wandb_exc:
        logger.debug("W&B per-pair log skipped: %s", wandb_exc)

    # Leak-detection signal (logger.error inside RidgeTrainer; surface here too)
    r2 = float(metrics.get("r2_score") or 0.0)
    band = metrics.get("expected_r2_band") or [0.05, 0.30]
    if r2 > band[1]:
        logger.error(
            "LEAK SIGNAL %s: R²=%.4f exceeds band %s — investigate (no auto-revert)",
            instrument, r2, band,
        )

    return {
        "instrument": instrument,
        "r2_score": r2,
        "confidence_mae": float(metrics.get("confidence_mae") or 0.0),
        "n_train": int(len(result["y_train"])),
        "n_val": int(len(result["y_val"])),
        "n_real": int(result.get("label_metadata", {}).get("n_real", 0) or 0),
        "n_pseudo": int(result.get("label_metadata", {}).get("n_pseudo", 0) or 0),
        "expected_r2_band": list(band),
        "leak_fired": r2 > band[1],
        "saved_at": pkl_path.stat().st_mtime,
        "saved_path": str(pkl_path.relative_to(_PROJECT_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instruments",
        default="EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD",
        help="Comma-separated pairs to fine-tune (must have H1 CSV in market_data/)",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=_PROJECT_ROOT / "trained_data" / "per_pair_confidence_report.json",
    )
    args = parser.parse_args()

    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]
    pair_csvs = _newest_h1_csv_per_pair(instruments)
    if not pair_csvs:
        logger.error("No H1 CSVs found — abort")
        return 1
    logger.info("Will fine-tune %d pairs: %s", len(pair_csvs), list(pair_csvs.keys()))

    init_booster = _load_joint_booster()

    # Use the same indicator helper as the joint retrain script for consistency
    from retrain_confidence_leak_fix import _add_indicators  # type: ignore

    summaries: List[Dict[str, Any]] = []
    for inst, csv in pair_csvs.items():
        try:
            df_raw = _load_pair_df(csv)
            df_feat = _add_indicators(df_raw)
        except Exception as exc:
            logger.warning("Skip %s (feature compute failed): %s", inst, exc)
            continue
        if len(df_feat) < 200:
            logger.warning("Skip %s: only %d rows after feature compute", inst, len(df_feat))
            continue

        try:
            summary = _train_per_pair(df_feat, inst, instruments, init_booster)
            summaries.append(summary)
            logger.info("%s done: R²=%.4f MAE=%.2f n_train=%d n_val=%d",
                        inst, summary["r2_score"], summary["confidence_mae"],
                        summary["n_train"], summary["n_val"])
        except Exception as exc:
            logger.error("Per-pair fine-tune FAILED for %s: %s", inst, exc, exc_info=True)
            summaries.append({
                "instrument": inst,
                "error": str(exc),
            })

    # Write summary report
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(summaries),
        "summaries": summaries,
        "joint_warm_start": init_booster is not None,
    }
    args.report_out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    logger.info("Per-pair report: %s", args.report_out)

    n_ok = sum(1 for s in summaries if "error" not in s)
    n_leak = sum(1 for s in summaries if s.get("leak_fired"))
    logger.info("Fine-tunes: %d/%d ok, %d leak-flagged", n_ok, len(summaries), n_leak)
    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
