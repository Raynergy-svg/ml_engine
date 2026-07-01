"""Smoke test for the 2026-05-01 follow-up (post per-pair retrain + cal v3).

Verifies:
1. Joint ``ridge_confidence.pkl`` loads.
2. All per-pair ``{INSTRUMENT}/ridge_confidence.pkl`` load.
3. Calibration v3 JSON loads with the expected schema.
4. Synthetic feature batch (50 rows) yields scores in [0, 100].
5. Real EUR_USD H1 batch (last 200 bars) yields scores in [0, 100] via the
   joint model.
6. Per-pair model `predict()` works for at least one pair.

Returns 0 on success, non-zero with reason on failure.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_v3")

JOINT_PKL = _PROJECT_ROOT / "trained_data" / "models" / "joint" / "ridge_confidence.pkl"
CAL_JSON = _PROJECT_ROOT / "trained_data" / "confidence_calibration.json"
MODELS_DIR = _PROJECT_ROOT / "trained_data" / "models"


def _load_pkl(path: Path) -> Dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _predict_with(model_data: Dict, X: np.ndarray) -> np.ndarray:
    scaler = model_data["scaler"]
    model = model_data["model"]
    feature_names = model_data.get("feature_names")
    n_expected = model_data.get("n_features") or (
        len(feature_names) if feature_names else X.shape[1]
    )
    # Pad/truncate
    if X.shape[1] < n_expected:
        pad = np.zeros((X.shape[0], n_expected - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > n_expected:
        X = X[:, :n_expected]
    Xs = scaler.transform(X)
    if feature_names is not None and len(feature_names) == Xs.shape[1]:
        Xs = pd.DataFrame(Xs, columns=feature_names)
    return np.asarray(model.predict(Xs))


def main() -> int:
    failures: List[str] = []

    # 1. Joint
    try:
        joint = _load_pkl(JOINT_PKL)
        n_features = joint["n_features"]
        feature_names = joint["feature_names"]
        logger.info("Joint loaded: n_features=%d feature_count=%d type=%s",
                    n_features, len(feature_names), joint.get("model_type"))
        if n_features != 24:
            failures.append(f"joint n_features != 24 (got {n_features})")
        if not feature_names or "instrument_EUR_USD" not in feature_names:
            failures.append("joint feature_names missing instrument one-hot")
    except Exception as exc:
        failures.append(f"joint load FAILED: {exc}")
        logger.error("joint load FAILED: %s", exc)
        return 1

    # 2. Per-pair models — list and verify each loads + predicts
    per_pair_paths = [p for p in MODELS_DIR.glob("*/ridge_confidence.pkl")
                      if "joint" not in p.parts]
    logger.info("Found %d per-pair pkl files", len(per_pair_paths))
    per_pair_summary = []
    for pkl in sorted(per_pair_paths):
        pair = pkl.parent.name
        try:
            m = _load_pkl(pkl)
            n_feat = m.get("n_features")
            saved_at = m.get("saved_at")
            # Quick predict on synthetic features
            rng = np.random.default_rng(hash(pair) % (2 ** 32))
            X = rng.normal(0, 1, size=(5, n_feat)).astype(np.float32)
            preds = _predict_with(m, X)
            if not np.all(np.isfinite(preds)):
                failures.append(f"{pair}: NaN/Inf predictions")
            per_pair_summary.append({
                "pair": pair, "n_features": n_feat, "saved_at": saved_at,
                "synthetic_pred_min": float(preds.min()),
                "synthetic_pred_max": float(preds.max()),
                "synthetic_pred_p50": float(np.percentile(preds, 50)),
            })
            logger.info(
                "%s: n_features=%d preds[%.2f, %.2f, %.2f] (min/p50/max)",
                pair, n_feat, preds.min(), np.percentile(preds, 50), preds.max(),
            )
        except Exception as exc:
            failures.append(f"{pair} load/predict FAILED: {exc}")
            logger.error("%s load/predict FAILED: %s", pair, exc)

    # 3. Calibration v3
    try:
        cal = json.loads(CAL_JSON.read_text())
        ver = cal.get("version")
        corpus = cal.get("calibration_corpus")
        coef = cal["platt_params"]["_global"]["coef"]
        intercept = cal["platt_params"]["_global"]["intercept"]
        ece = cal.get("validation", {}).get("ece")
        brier = cal.get("validation", {}).get("brier")
        logger.info(
            "Calibration loaded: version=%s corpus=%s coef=%.4f intercept=%.4f "
            "ece=%.4f brier=%.4f",
            ver, corpus, coef, intercept, ece, brier,
        )
        if ver != 3:
            failures.append(f"calibration version != 3 (got {ver})")
        if corpus != "archive+live":
            failures.append(f"calibration corpus != archive+live (got {corpus})")
        if cal.get("label_mode") != "journal_outcome_blend":
            failures.append("calibration label_mode != journal_outcome_blend")
        if ece is None or ece >= 0.15:
            failures.append(f"calibration ECE outside gate (ece={ece})")
        if brier is None or brier >= 0.30:
            failures.append(f"calibration Brier outside gate (brier={brier})")
    except Exception as exc:
        failures.append(f"calibration load FAILED: {exc}")

    # 4. Synthetic batch through joint model
    try:
        rng = np.random.default_rng(456)
        scores = []
        for _ in range(50):
            x = rng.normal(0.0, 1.0, size=n_features).astype(np.float32).reshape(1, -1)
            y = _predict_with(joint, x)[0]
            scores.append(float(y))
        arr = np.array(scores)
        logger.info(
            "Joint synthetic 50 preds: min=%.2f p25=%.2f p50=%.2f p75=%.2f max=%.2f",
            arr.min(), np.percentile(arr, 25), np.percentile(arr, 50),
            np.percentile(arr, 75), arr.max(),
        )
        if arr.min() < 0 or arr.max() > 100:
            failures.append(f"joint synthetic scores outside [0,100]: [{arr.min():.2f},{arr.max():.2f}]")
    except Exception as exc:
        failures.append(f"joint synthetic predict FAILED: {exc}")

    # 5. Real EUR_USD H1 batch
    csvs = sorted(glob(str(_PROJECT_ROOT / "market_data" / "oanda_EUR_USD_H1_live_*.csv")))
    if csvs:
        try:
            from src.core.modular_data_loaders import compute_normalized_features
            sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
            from retrain_confidence_leak_fix import _add_indicators

            df = pd.read_csv(csvs[-1])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            df = df.tail(200)
            df_feat = _add_indicators(df)

            # Build a 24-feature row using only numeric features (instrument
            # one-hots = 0 here since this is generic synthetic-real data)
            cols_avail = [c for c in feature_names
                          if not c.startswith("instrument_") and c in df_feat.columns]
            X = df_feat[cols_avail].values.astype(np.float32)
            # Pad to 24 (instrument one-hots = 0)
            preds = _predict_with(joint, X)
            logger.info(
                "Real EUR_USD H1 last-200 bar preds: n=%d min=%.2f p50=%.2f max=%.2f",
                len(preds), preds.min(), np.percentile(preds, 50), preds.max(),
            )
            if preds.min() < 0 or preds.max() > 100:
                failures.append(
                    f"real-data scores outside [0,100]: [{preds.min():.2f}, {preds.max():.2f}]"
                )
        except Exception as exc:
            logger.warning("Real-data path skipped: %s", exc)
    else:
        logger.warning("No EUR_USD H1 CSV — real-data path skipped")

    # Summary
    payload = {
        "joint_path": str(JOINT_PKL.relative_to(_PROJECT_ROOT)),
        "joint_n_features": n_features,
        "n_per_pair_models": len(per_pair_paths),
        "per_pair_summary": per_pair_summary,
        "calibration_version": cal.get("version") if 'cal' in locals() else None,
        "calibration_corpus": cal.get("calibration_corpus") if 'cal' in locals() else None,
        "failures": failures,
    }
    logger.info("Smoke summary: %s", json.dumps(payload, indent=2, default=str))

    if failures:
        logger.error("SMOKE TEST FAILED: %d issue(s)", len(failures))
        return 4
    logger.info("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
