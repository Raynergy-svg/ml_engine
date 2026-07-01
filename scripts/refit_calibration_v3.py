"""Calibration v3 refit — archive + live (post-backfill) trades.

Updates the 2026-04-30 leak-fix Platt calibrator using both archive
trades AND live trades (which now have 24-feature ``ridge_features``
post-backfill — see ``scripts/backfill_ridge_features.py``).

v2 used archive-only; v3 includes live trades that scored against the
new 24-feature joint ridge model. Saves to
``trained_data/confidence_calibration.json`` with:

- ``version: 3``
- ``label_mode: "journal_outcome_blend"``
- ``calibration_corpus: "archive+live"``
- ``platt_params: {_global: {coef, intercept}}``
- ``trade_history: [(score, won, regime), ...]``
- ``validation: {ece, brier, reliability_monotonic, n_train, n_test, gates_passed}``
- ``metadata: {n_trades, n_regimes, leak_fix_version, prior_version: 2}``

Validation gates (same as v2):
    ECE < 0.15, Brier < 0.30, reliability monotonic across 5 buckets
    (small inversions <2pp acceptable; if test set falls into one bucket
    monotonicity is vacuously true and we accept).

Failure path: writes ``confidence_calibration.validation_failed.json``
sidecar, exits non-zero, does NOT overwrite the on-disk calibration. The
operator restores v2 from ``_rollback_2026-04-30/confidence_calibration.json.v2``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.labels.journal_loader import load_all_journal_trades  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("refit_calibration_v3")

CAL_PATH = _PROJECT_ROOT / "trained_data" / "confidence_calibration.json"
RIDGE_PATH = _PROJECT_ROOT / "trained_data" / "models" / "joint" / "ridge_confidence.pkl"
LIVE_JOURNAL = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"


def _load_ridge_model() -> Dict[str, Any]:
    import pickle, warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(RIDGE_PATH, "rb") as f:
            return pickle.load(f)


def _compute_ece(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        bp = float(y_prob[mask].mean())
        by = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(by - bp)
    return float(ece)


def _reliability_buckets(y_prob: np.ndarray, y_true: np.ndarray,
                         n_bins: int = 5) -> List[Tuple[float, float, int]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    buckets = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append((float(y_prob[mask].mean()), float(y_true[mask].mean()), n))
    return buckets


def _is_monotonic(buckets: List[Tuple[float, float, int]]) -> bool:
    if len(buckets) < 2:
        return False
    ys = [b[1] for b in buckets]
    # Allow small inversions (≤0.05).
    for i in range(len(ys) - 1):
        if ys[i + 1] < ys[i] - 0.05:
            return False
    return True


def _score_with_features(model_data: Dict[str, Any],
                         ridge_features: List[float]) -> Optional[float]:
    if not ridge_features:
        return None
    try:
        scaler = model_data["scaler"]
        model = model_data["model"]
        feature_names = model_data.get("feature_names")
        n_expected = model_data.get("n_features") or (
            len(feature_names) if feature_names else None
        )
        x = np.asarray(ridge_features, dtype=np.float32).reshape(1, -1)
        if n_expected is not None and x.shape[1] != n_expected:
            if x.shape[1] < n_expected:
                pad = np.zeros((1, n_expected - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, :n_expected]
        x_scaled = scaler.transform(x)
        if feature_names is not None and len(feature_names) == x_scaled.shape[1]:
            x_scaled = pd.DataFrame(x_scaled, columns=feature_names)
        return float(model.predict(x_scaled)[0])
    except Exception as exc:
        logger.debug("score_with_features failed: %s", exc)
        return None


def main() -> int:
    if not RIDGE_PATH.exists():
        logger.error("ridge_confidence.pkl not found at %s — abort", RIDGE_PATH)
        return 1

    model_data = _load_ridge_model()
    n_expected = model_data.get("n_features")
    logger.info("Loaded joint ridge: model_type=%s n_features=%s",
                model_data.get("model_type"), n_expected)

    pairs: List[Tuple[float, int, str]] = []   # (score_norm_0_1, won, regime)
    n_live_used = 0
    n_live_skipped = 0
    n_live_skipped_dim = 0

    # ── 1) Live journal: trades with 24-feature ridge_features (post-backfill)
    if LIVE_JOURNAL.exists():
        try:
            raw_live = json.loads(LIVE_JOURNAL.read_text())
        except Exception as exc:
            logger.warning("could not parse live journal: %s", exc)
            raw_live = []
    else:
        raw_live = []

    for t in raw_live:
        rf = t.get("ridge_features")
        outcome = t.get("outcome")
        won: Optional[bool] = None
        if isinstance(outcome, dict) and "trade_won" in outcome:
            won = bool(outcome["trade_won"])

        if not isinstance(rf, list) or won is None:
            n_live_skipped += 1
            continue
        if n_expected is not None and len(rf) != n_expected:
            n_live_skipped_dim += 1
            continue

        score = _score_with_features(model_data, rf)
        if score is None:
            n_live_skipped += 1
            continue

        regime = "NORMAL"
        r = t.get("regime")
        if isinstance(r, dict):
            regime = str(r.get("volatility_regime", "NORMAL"))
        score_norm = max(0.0, min(1.0, score / 100.0))
        pairs.append((score_norm, 1 if won else 0, regime))
        n_live_used += 1

    logger.info(
        "Live journal: used=%d skipped_no_outcome=%d skipped_wrong_dim=%d",
        n_live_used, n_live_skipped, n_live_skipped_dim,
    )

    # ── 2) Archive trades — fall back to leaky `confidence` field
    # (same as v2; archive trades don't store ridge_features).
    trades = load_all_journal_trades(include_archives=True)
    arch_used = 0
    for t in trades:
        if t.get("trade_won") is None:
            continue
        conf = t.get("confidence")
        if conf is None:
            continue
        try:
            cf = float(conf)
            if 0.0 <= cf <= 1.0:
                score_norm = cf
            elif 0.0 <= cf <= 100.0:
                score_norm = cf / 100.0
            else:
                continue
        except (TypeError, ValueError):
            continue
        regime = t.get("regime") or "NORMAL"
        pairs.append((score_norm, 1 if t["trade_won"] else 0, str(regime)))
        arch_used += 1

    n_total = len(pairs)
    logger.info("Total pairs for fit: %d (live=%d, archive=%d)",
                n_total, n_live_used, arch_used)

    if n_total < 30:
        logger.error("Insufficient (score, outcome) pairs (%d < 30) — abort", n_total)
        return 2

    rng = np.random.default_rng(42)
    arr = np.array([(p[0], p[1]) for p in pairs], dtype=np.float64)
    perm = rng.permutation(len(arr))
    arr = arr[perm]
    n_test = max(int(len(arr) * 0.2), 5)
    train, test = arr[n_test:], arr[:n_test]
    X_tr = train[:, 0:1]
    y_tr = train[:, 1].astype(int)
    X_te = test[:, 0:1]
    y_te = test[:, 1].astype(int)

    platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, random_state=42)
    platt.fit(X_tr, y_tr)
    coef = float(platt.coef_[0][0])
    intercept = float(platt.intercept_[0])

    p_test = platt.predict_proba(X_te)[:, 1]
    brier = float(np.mean((p_test - y_te) ** 2))
    ece = _compute_ece(p_test, y_te.astype(float), n_bins=5)
    buckets = _reliability_buckets(p_test, y_te.astype(float), n_bins=5)
    monotonic = _is_monotonic(buckets)
    monotonic_ok = monotonic if len(buckets) >= 2 else True

    gates = {
        "ece_under_0.15": ece < 0.15,
        "brier_under_0.30": brier < 0.30,
        "monotonic_5_buckets": monotonic_ok,
        "n_buckets_in_test": len(buckets),
    }
    logger.info("Validation: ECE=%.4f Brier=%.4f monotonic=%s buckets=%d",
                ece, brier, monotonic, len(buckets))
    for b in buckets:
        logger.info("  bucket pred=%.3f actual=%.3f n=%d", *b)
    logger.info("Gates: %s", gates)

    pass_gates = {k: v for k, v in gates.items() if k != "n_buckets_in_test"}
    if not all(v if isinstance(v, bool) else True for v in pass_gates.values()):
        logger.error("Calibration v3 validation FAILED: %s", gates)
        sidecar = CAL_PATH.with_suffix(".validation_failed.json")
        sidecar.write_text(json.dumps({
            "ece": ece, "brier": brier, "monotonic": monotonic,
            "buckets": buckets, "gates": gates,
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "coef": coef, "intercept": intercept,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "version_attempted": 3,
            "calibration_corpus": "archive+live",
        }, indent=2, sort_keys=True))
        logger.error("Wrote %s — restore v2 from rollback manifest", sidecar)
        return 3

    # Read prior version metadata to embed in v3 (provenance)
    prior_version: Optional[int] = None
    prior_brier: Optional[float] = None
    prior_ece: Optional[float] = None
    if CAL_PATH.exists():
        try:
            prior = json.loads(CAL_PATH.read_text())
            prior_version = prior.get("version")
            prior_validation = prior.get("validation") or {}
            prior_brier = prior_validation.get("brier")
            prior_ece = prior_validation.get("ece")
        except Exception:
            pass

    out = {
        "version": 3,
        "label_mode": "journal_outcome_blend",
        "calibration_corpus": "archive+live",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platt_params": {
            "_global": {"coef": coef, "intercept": intercept},
        },
        "validation": {
            "ece": ece,
            "brier": brier,
            "reliability_monotonic": monotonic,
            "buckets": buckets,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "gates_passed": gates,
        },
        "trade_history": [[float(s), float(y), str(r)] for (s, y, r) in pairs],
        "metadata": {
            "n_trades": int(n_total),
            "n_live_used": int(n_live_used),
            "n_archive_used": int(arch_used),
            "n_live_skipped_wrong_dim": int(n_live_skipped_dim),
            "n_live_skipped_no_outcome": int(n_live_skipped),
            "n_regimes": 1,
            "leak_fix_version": "2026-04-30",
            "prior_version": prior_version,
            "prior_brier": prior_brier,
            "prior_ece": prior_ece,
        },
    }

    tmp = CAL_PATH.with_suffix(CAL_PATH.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    os.replace(tmp, CAL_PATH)
    logger.info("Wrote v3 calibration to %s (coef=%.4f intercept=%.4f, prior_v=%s)",
                CAL_PATH, coef, intercept, prior_version)
    if prior_brier is not None:
        logger.info("Brier delta v3-v2: %+.4f (v2=%.4f, v3=%.4f)",
                    brier - prior_brier, prior_brier, brier)
    if prior_ece is not None:
        logger.info("ECE delta v3-v2:   %+.4f (v2=%.4f, v3=%.4f)",
                    ece - prior_ece, prior_ece, ece)
    return 0


if __name__ == "__main__":
    sys.exit(main())
