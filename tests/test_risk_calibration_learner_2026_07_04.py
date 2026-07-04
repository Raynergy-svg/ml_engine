"""Lightweight online (River-style) risk/calibration learner.

Scope discipline (task requirement): this learner must ONLY ever see
risk/execution/calibration features (regime, rr_ratio, sl/tp pips, mae/mfe,
lane, existing confidence) — never directional/price features (trend,
momentum, price levels). Re-fitting a directional signal from journal
outcomes would overfit noise per the established empirical ceiling
(docs/fx-edge-search-final-verdict-2026-06-18.md: price-only intraday
direction has no edge).

NO MOCKS per CLAUDE.md No-Mock Rule — real dataclasses, no I/O in this file.
"""
from __future__ import annotations

from copy import deepcopy

from src.training.incremental.risk_calibration_learner import (
    ALLOWED_FEATURE_KEYS,
    OnlineLogisticRegression,
    RiskCalibrationLearner,
    brier_score,
    build_features,
)


def test_build_features_only_uses_whitelisted_risk_calibration_keys():
    entry = {
        "confidence": 0.62,
        "regime": {"volatility_regime": "HIGH"},
        "rr_ratio": 1.4,
        "sl_pips": 12.0,
        "tp_pips": 16.8,
        "lane": "scanner",
        "outcome": {"trade_won": True, "mae_pips": 3.0, "mfe_pips": 9.0},
        # A directional field that must NEVER leak into the feature vector.
        "trend_direction_signal": "LONG",
        "macd_histogram": 0.0042,
    }
    feats = build_features(entry)
    assert set(feats) <= ALLOWED_FEATURE_KEYS
    assert "trend_direction_signal" not in feats
    assert "macd_histogram" not in feats
    assert feats["regime_high"] == 1.0
    assert feats["regime_normal"] == 0.0


def test_build_features_handles_string_outcome_shape():
    entry = {
        "confidence": 0.55,
        "regime": {"volatility_regime": "LOW"},
        "outcome": "win",
        "rr_ratio": 1.2,
        "lane": "trend",
    }
    feats = build_features(entry)
    assert feats["regime_low"] == 1.0
    assert feats["lane_trend"] == 1.0
    assert feats["lane_scanner"] == 0.0


def test_online_logistic_regression_learns_a_separable_pattern():
    """Reproduce-then-resolve: an untrained model predicts ~0.5 for
    everything; after learning a clearly separable pattern (high confidence
    -> win, low confidence -> loss) it must discriminate between them."""
    model = OnlineLogisticRegression()
    before_high = model.predict_proba_one({"confidence": 0.9})
    before_low = model.predict_proba_one({"confidence": 0.1})
    assert abs(before_high - before_low) < 1e-9  # untrained: no discrimination

    for _ in range(600):
        model.learn_one({"confidence": 0.9}, 1.0)
        model.learn_one({"confidence": 0.1}, 0.0)

    after_high = model.predict_proba_one({"confidence": 0.9})
    after_low = model.predict_proba_one({"confidence": 0.1})
    assert after_high > 0.7
    assert after_low < 0.3


def test_online_logistic_regression_state_round_trips():
    model = OnlineLogisticRegression()
    for _ in range(20):
        model.learn_one({"confidence": 0.8, "rr_ratio": 1.5}, 1.0)
    state = model.to_state()

    restored = OnlineLogisticRegression.from_state(state)

    x = {"confidence": 0.8, "rr_ratio": 1.5}
    assert restored.predict_proba_one(x) == model.predict_proba_one(x)
    assert restored.n_seen == model.n_seen


def test_size_multiplier_never_exceeds_one_point_zero():
    """Hard risk-decreasing-only invariant: the learner may recommend
    trading SMALLER than the caller's existing size, never larger."""
    learner = RiskCalibrationLearner()
    # Even a model that is wildly overconfident in the outcome's favor
    # (predicts near-certain win) must not push the multiplier above 1.0.
    for _ in range(500):
        learner.model.learn_one({"confidence": 0.5}, 1.0)

    mult = learner.size_multiplier({"confidence": 0.5})
    assert 0.5 <= mult <= 1.0


def test_size_multiplier_is_never_nan_even_with_corrupted_model_state():
    """Regression for a real fragility: min(1.0, nan) == 1.0 in Python (NaN
    comparisons are always False), so a corrupted/NaN model state used to
    silently clamp to full size via argument-order luck rather than an
    explicit finite-check. A future refactor to min(ratio, 1.0) would flip
    that to NaN with no test catching it. The safe value must be asserted
    explicitly, not incidental to clamp ordering."""
    learner = RiskCalibrationLearner()
    learner.model.weights = {"confidence": float("nan")}
    learner.model.bias = float("nan")

    mult = learner.size_multiplier({"confidence": 0.7})

    assert mult == 0.5  # most conservative (de-risk) value, not 1.0
    assert mult == mult  # not NaN (NaN != NaN)


def test_size_multiplier_shrinks_when_calibration_says_overconfident():
    """If realized win-rate at a given confidence level is far below the
    raw confidence, the multiplier must shrink below 1.0 (de-risk)."""
    learner = RiskCalibrationLearner()
    # Confidence says 0.9 (should mean ~90% win rate) but realized outcomes
    # at that confidence level are mostly losses -> overconfident model.
    for _ in range(300):
        learner.model.learn_one({"confidence": 0.9}, 0.0)

    mult = learner.size_multiplier({"confidence": 0.9})
    assert mult < 1.0


def test_brier_score_lower_is_better_and_none_on_empty():
    assert brier_score(OnlineLogisticRegression(), []) is None

    perfect_model = OnlineLogisticRegression()
    for _ in range(300):
        perfect_model.learn_one({"confidence": 0.9}, 1.0)
        perfect_model.learn_one({"confidence": 0.1}, 0.0)

    samples = [({"confidence": 0.9}, 1.0), ({"confidence": 0.1}, 0.0)]
    untrained_score = brier_score(OnlineLogisticRegression(), samples)
    trained_score = brier_score(perfect_model, samples)
    assert trained_score < untrained_score


def test_learner_state_round_trip_preserves_fit(tmp_path):
    learner = RiskCalibrationLearner()
    entries = [
        {"confidence": 0.8, "regime": {"volatility_regime": "NORMAL"}, "outcome": "win", "rr_ratio": 1.5},
        {"confidence": 0.2, "regime": {"volatility_regime": "NORMAL"}, "outcome": "loss", "rr_ratio": 1.2},
    ] * 50
    n = learner.fit_incremental(entries)
    assert n == len(entries)

    state = learner.to_state()
    restored = RiskCalibrationLearner.from_state(state)
    x = build_features(entries[0])
    assert restored.model.predict_proba_one(x) == learner.model.predict_proba_one(x)


def test_fit_incremental_skips_unresolved_entries():
    learner = RiskCalibrationLearner()
    entries = [
        {"confidence": 0.7, "outcome": None},  # still open — must be skipped
        {"confidence": 0.7, "outcome": "win", "regime": {"volatility_regime": "NORMAL"}},
    ]
    n = learner.fit_incremental(entries)
    assert n == 1


def test_deepcopy_of_learner_does_not_share_state():
    """The offline batch job deep-copies the incumbent to build a candidate
    — mutating the candidate must never mutate the incumbent."""
    incumbent = RiskCalibrationLearner()
    incumbent.model.learn_one({"confidence": 0.5}, 1.0)
    incumbent_weights_before = dict(incumbent.model.weights)

    candidate = deepcopy(incumbent)
    for _ in range(50):
        candidate.model.learn_one({"confidence": 0.5}, 0.0)

    assert incumbent.model.weights == incumbent_weights_before
    assert candidate.model.weights != incumbent.model.weights
