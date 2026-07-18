"""Regression tests for the 2026-07-18 pipeline/math audit fixes.

Each test locks in one repaired formula or safety property. No new mocks:
pure-math modules are exercised directly against real arrays/real disk
(tmp_path); the execute_trade halt test reuses the grandfathered helper from
tests/scanner/_breaker_helpers.py (same pattern as the circuit-breaker suite).

Covered fixes:
  1. Sortino downside deviation (src/rl/utils.py)
  2. Sharpe significance per-period units (src/training/enterprise_training.py)
  3. Calibration: no 0.5 floor, is_valid meaningful, honest default calibrator
     (src/risk/confidence_calibration.py)
  4. Purged k-fold: embargo on the trailing TRAIN side (walkforward_validation)
  5. Factor vol-target: NaN cell no longer flattens the book (src/factor/portfolio.py)
  6. carry_signal single-name cross-section -> neutral 0 (src/factor/signals.py)
  7. EWMA correlation: covariance persisted across save/load; legacy files
     cold-start; pair growth preserves history (src/risk/ewma_correlation.py)
  8. Position sizing: quote-currency pip values + <=15x leverage clamp
     (src/risk/position_sizing.py)
  9. Quote->USD risk aggregation factor (src/scanner/execution.py)
 10. execute_trade halt re-check fails CLOSED on corrupt state.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


# ── 1. Sortino ──────────────────────────────────────────────────────────────

def test_sortino_positive_for_profitable_strategy_with_equal_losses():
    """Old bug: np.std of the loser subset zeroed out when losses were equal,
    reporting Sortino = 0 for a strongly profitable strategy."""
    from src.rl.utils import calculate_sortino

    returns = [0.10, 0.10, 0.10, -0.02, -0.02]
    sortino = calculate_sortino(returns, risk_free_rate=0.0, annualization_factor=1.0)
    # Correct: mean=0.052, DD=sqrt((2*0.0004)/5)=0.012649 -> ~4.111
    assert sortino == pytest.approx(0.052 / np.sqrt(0.0008 / 5), rel=1e-6)
    assert sortino > 4.0


def test_sortino_downside_measured_from_target_over_all_periods():
    from src.rl.utils import calculate_sortino

    returns = [0.05, -0.01, 0.02, -0.03]
    got = calculate_sortino(returns, risk_free_rate=0.0, annualization_factor=1.0)
    shortfalls = np.minimum(np.array(returns), 0.0)
    expected = np.mean(returns) / np.sqrt(np.mean(shortfalls**2))
    assert got == pytest.approx(expected, rel=1e-9)


# ── 2. Sharpe significance ──────────────────────────────────────────────────

def test_sharpe_significance_not_inflated_by_annualization():
    """Old bug: annualized Sharpe plugged into Lo's per-period SE formula
    inflated z by ~sqrt(252); a modest strategy reported p ~= 0."""
    from src.training.enterprise_training import StatisticalValidator

    rng = np.random.default_rng(7)
    # Per-period SR ~= 0.1: NOT significant at n=252 (correct z ~= 1.58)
    returns = rng.normal(0.001, 0.01, 252)
    sr_pp = returns.mean() / returns.std(ddof=1)
    res = StatisticalValidator.sharpe_ratio_significance(
        returns, benchmark_sharpe=0.0, periods_per_year=252
    )
    # z must be computed in per-period units (annualization-invariant)
    assert res["z_statistic"] == pytest.approx(
        (sr_pp - 0.0) / np.sqrt((1 + 0.5 * sr_pp**2) / len(returns)), rel=1e-9
    )
    # And the same data must give the same z regardless of periods_per_year
    res_hourly = StatisticalValidator.sharpe_ratio_significance(
        returns, benchmark_sharpe=0.0, periods_per_year=252 * 24
    )
    assert res["z_statistic"] == pytest.approx(res_hourly["z_statistic"], rel=1e-9)


# ── 3. Calibration honesty ──────────────────────────────────────────────────

def test_calibration_low_probability_not_floored_and_validity_meaningful():
    from src.risk.confidence_calibration import (
        CalibrationConfig, ConfidenceCalibrator,
    )

    config = CalibrationConfig(
        method="platt", min_confidence_threshold=0.5, max_confidence_threshold=0.95,
        apply_directional_adjustment=False,
        apply_win_probability_adjustment=False,
        apply_trading_context_adjustment=False,
    )
    cal = ConfidenceCalibrator(config)
    # Real fit: high scores lose, low scores win -> calibrator inverts, so a
    # high raw confidence maps to a LOW probability that must survive un-floored.
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.uniform(0.6, 0.9, 200), rng.uniform(0.1, 0.4, 200)])
    outcomes = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    cal.fit(scores, outcomes)

    res = cal.calibrate_confidence(0.85)
    assert res.calibrated_confidence < 0.5, "low probability must not be floored to 0.5"
    assert res.is_valid is False, "below-threshold probability must be invalid"

    res_good = cal.calibrate_confidence(0.15)
    assert res_good.calibrated_confidence > 0.5
    assert res_good.is_valid is True


def test_default_calibrator_applies_no_multiplicative_shrink():
    from src.risk.confidence_calibration import create_default_calibrator

    cal = create_default_calibrator()
    assert cal.config.apply_directional_adjustment is False
    assert cal.config.apply_win_probability_adjustment is False
    assert cal.config.apply_trading_context_adjustment is False


# ── 4. Purged k-fold embargo side ───────────────────────────────────────────

def test_purged_kfold_embargo_protects_trailing_train_side():
    from src.training.walkforward_validation import purged_kfold_split

    n, splits, purge, embargo = 1000, 5, 10, 5
    for train_idx, test_idx in purged_kfold_split(
        n, n_splits=splits, purge_gap=purge, embargo_gap=embargo
    ):
        test_start, test_end = test_idx.min(), test_idx.max() + 1
        after = train_idx[train_idx >= test_end]
        if len(after):
            # trailing train must sit at least purge+embargo past test_end
            assert after.min() >= test_end + purge + embargo
        before = train_idx[train_idx < test_start]
        if len(before):
            assert before.max() < test_start - purge + 1
        # the test fold itself must be contiguous and un-chopped
        assert len(test_idx) == test_end - test_start


# ── 5. Factor vol-target NaN robustness ─────────────────────────────────────

def test_factor_book_survives_single_nan_cell():
    from src.factor.portfolio import target_weights

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=140)
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.005, (len(dates), 3)), axis=0),
        index=dates, columns=pairs,
    )
    score = pd.DataFrame(1.0, index=dates, columns=pairs)
    # Poison exactly one close mid-sample -> one NaN return cell in the window
    close.iloc[100, 2] = np.nan

    w = target_weights(score, close)
    last = w.iloc[-1]
    assert np.isfinite(last).all()
    # Old bug: the whole row collapsed to zeros for ~VOL_LOOKBACK dates after
    # the gap. The clean columns must retain non-zero exposure.
    assert last.abs().sum() > 0, "one NaN cell must not flatten the entire book"


def test_carry_signal_single_name_cross_section_is_neutral():
    from src.factor.signals import carry_signal

    dates = pd.date_range("2024-01-01", periods=3)
    rd = pd.DataFrame({"EUR_USD": [0.5, 0.5, 0.5], "GBP_USD": [np.nan, 1.0, np.nan]},
                      index=dates)
    sig = carry_signal(rd)
    assert np.isfinite(sig.fillna(0).to_numpy()).all()
    # Dates where only EUR_USD is valid: neutral 0, not inf/NaN
    assert sig.loc[dates[0], "EUR_USD"] == 0.0
    assert sig.loc[dates[2], "EUR_USD"] == 0.0


# ── 7. EWMA correlation persistence ─────────────────────────────────────────

def _feed_correlated(engine, n=60, seed=5):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        common = rng.normal(0, 0.01)
        engine.update({
            "EUR_USD": common + rng.normal(0, 0.003),
            "GBP_USD": common + rng.normal(0, 0.003),
            "USD_JPY": rng.normal(0, 0.01),
        })


def test_ewma_state_roundtrip_preserves_covariance(tmp_path):
    from src.risk.ewma_correlation import EWMACorrelationEngine

    eng = EWMACorrelationEngine()
    _feed_correlated(eng, 60)
    rho_before = eng.get_correlation("EUR_USD", "GBP_USD")
    assert rho_before is not None and 0.3 < rho_before < 1.0

    path = str(tmp_path / "ewma_state.json")
    eng.save_state(path)

    eng2 = EWMACorrelationEngine()
    eng2.load_state(path)
    # One more update must NOT degenerate to ±1 (old bug: zeroed covariance
    # made the first post-load update produce rank-1 ±1 correlations).
    eng2.update({"EUR_USD": 0.001, "GBP_USD": 0.0012, "USD_JPY": -0.0005})
    rho_after = eng2.get_correlation("EUR_USD", "USD_JPY")
    assert rho_after is not None
    assert abs(rho_after) < 0.999, "post-restart correlations must not collapse to ±1"


def test_ewma_legacy_state_without_covariance_cold_starts(tmp_path):
    from src.risk.ewma_correlation import EWMACorrelationEngine

    eng = EWMACorrelationEngine()
    _feed_correlated(eng, 60)
    path = str(tmp_path / "ewma_state.json")
    eng.save_state(path)
    # Simulate a legacy v1.0 file: strip the persisted covariance/means
    data = json.loads(open(path).read())
    data.pop("ewma_covariance", None)
    data.pop("ewma_means", None)
    open(path, "w").write(json.dumps(data))

    eng2 = EWMACorrelationEngine()
    eng2.load_state(path)
    # Without a covariance the warm-up guard must re-engage
    assert eng2.get_state().observation_count == 0


def test_ewma_pair_growth_preserves_existing_history():
    from src.risk.ewma_correlation import EWMACorrelationEngine

    eng = EWMACorrelationEngine()
    _feed_correlated(eng, 60)
    rho_before = eng.get_correlation("EUR_USD", "GBP_USD")
    # New pair arrives mid-stream (old bug: wiped the whole covariance)
    eng.update({"EUR_USD": 0.001, "GBP_USD": 0.0011, "USD_JPY": 0.0, "AUD_USD": 0.002})
    rho_after = eng.get_correlation("EUR_USD", "GBP_USD")
    assert rho_after is not None
    assert rho_after == pytest.approx(rho_before, abs=0.15), \
        "adding a pair must not reset accumulated correlations"


# ── 8. Position sizing currency + leverage ──────────────────────────────────

def test_pip_value_reflects_quote_currency():
    from src.risk.position_sizing import DynamicPositionSizer, PositionSizingConfig

    sizer = DynamicPositionSizer(PositionSizingConfig())
    # GBP-quoted: pip value must be ~0.0001 * GBP->USD (>0.0001), so for the
    # same risk budget the size must be SMALLER than the USD-quoted equivalent.
    v_gbp = sizer._get_stop_loss_value_per_unit(20.0, "EUR_GBP")
    v_usd = sizer._get_stop_loss_value_per_unit(20.0, "EUR_USD")
    v_cad = sizer._get_stop_loss_value_per_unit(20.0, "USD_CAD")
    assert v_gbp > v_usd > v_cad
    # JPY cross uses 0.01 pip * JPY->USD
    v_jpy = sizer._get_stop_loss_value_per_unit(20.0, "EUR_JPY")
    assert v_jpy == pytest.approx(20.0 * 0.01 * (1.0 / 150.0), rel=0.2)


def test_leverage_clamp_caps_units_at_15x_equity():
    from src.risk.position_sizing import DynamicPositionSizer, PositionSizingConfig

    sizer = DynamicPositionSizer(PositionSizingConfig())
    equity = 100_000.0
    capped = sizer._apply_position_constraints(10_000_000, equity, "USD_JPY")
    # USD-base pair: notional(USD) == units, so cap = 15 * equity
    assert capped <= int(equity * 15.0) + 1
    # And well below the old 30x ceiling
    assert capped < int(equity * 30.0)


# ── 9. Quote->USD aggregation factor ────────────────────────────────────────

def test_quote_to_usd_factor():
    from src.scanner.execution import _quote_to_usd_factor

    assert _quote_to_usd_factor("EUR_USD", 1.08) == 1.0
    assert _quote_to_usd_factor("USD_JPY", 150.0) == pytest.approx(1 / 150.0)
    # No price -> static estimate, not raw quote units
    assert _quote_to_usd_factor("USD_JPY", None) == pytest.approx(1 / 150.0, rel=0.2)
    assert _quote_to_usd_factor("EUR_GBP", None) > 1.0


# ── 10. Halt re-check fails closed ──────────────────────────────────────────

def test_execute_trade_blocks_on_corrupt_state_json(tmp_path, monkeypatch):
    from tests.scanner._breaker_helpers import (
        build_minimal_valid_ctx, make_execution_manager,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    # Present-but-corrupt halt file: the old guard silently proceeded
    # (load_state -> default halted=False); the fixed guard must BLOCK.
    (tmp_path / ".claude" / "state.json").write_text("{not valid json")

    mgr = make_execution_manager()
    result = mgr.execute_trade(
        pair="EUR_USD", direction="LONG", confidence=0.6,
        current_price=1.1, atr=0.001,
        analysis_context=build_minimal_valid_ctx(),
    )
    # Active-branch semantics (get_halted_strict, 2026-07-02): ANY unreadable
    # state — including corrupt JSON — is halted. Blocked either way.
    assert result.success is False
    assert "halted" in (result.error or "").lower()


def test_execute_trade_blocks_when_halted_true(tmp_path, monkeypatch):
    from tests.scanner._breaker_helpers import (
        build_minimal_valid_ctx, make_execution_manager,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "state.json").write_text(json.dumps({"halted": True}))

    mgr = make_execution_manager()
    result = mgr.execute_trade(
        pair="EUR_USD", direction="LONG", confidence=0.6,
        current_price=1.1, atr=0.001,
        analysis_context=build_minimal_valid_ctx(),
    )
    assert result.success is False
    assert "halted" in (result.error or "").lower()


# ── 11. Purge/gap must cover the label horizon (operator directive) ─────────

def test_walkforward_gap_raised_to_label_horizon():
    from src.training.walkforward_validation import WalkForwardValidator

    v = WalkForwardValidator(n_splits=3, gap=24, label_horizon=288)
    assert v.gap == 288, "gap must be raised to cover the label horizon"
    v2 = WalkForwardValidator(n_splits=3, gap=300, label_horizon=288)
    assert v2.gap == 300, "an already-sufficient gap must not shrink"


def test_purged_kfold_purge_raised_to_label_horizon():
    from src.training.walkforward_validation import purged_kfold_split

    n, horizon = 4000, 288
    for train_idx, test_idx in purged_kfold_split(
        n, n_splits=4, purge_gap=10, embargo_gap=5, label_horizon=horizon
    ):
        test_start, test_end = test_idx.min(), test_idx.max() + 1
        before = train_idx[train_idx < test_start]
        if len(before):
            assert before.max() < test_start - horizon + 1
        after = train_idx[train_idx >= test_end]
        if len(after):
            assert after.min() >= test_end + horizon + 5


# ── 12. label_horizon hardening (round-trip + mandatory at training boundary) ─

def test_walkforward_config_roundtrips_label_horizon():
    from src.training.walkforward_validation import WalkForwardConfig

    cfg = WalkForwardConfig(label_horizon=288)
    d = cfg.to_dict()
    assert d["label_horizon"] == 288, "to_dict must serialize label_horizon"
    cfg2 = WalkForwardConfig.from_dict(d)
    assert cfg2.label_horizon == 288, "round trip must not drop the leakage bound"


def test_supervised_training_boundary_requires_label_horizon():
    """train_with_walkforward_validation must REFUSE (not warn) when the
    walk-forward config omits label_horizon — an unverifiable leakage bound
    at the supervised boundary means potentially leaked baselines."""
    import pytest as _pytest
    from src.training.buddy_training_helpers import train_with_walkforward_validation

    with _pytest.raises(ValueError, match="label_horizon"):
        train_with_walkforward_validation(
            trainer_class=object, trainer_config=None,
            X_train=np.zeros((100, 4)), y_train=np.zeros(100),
            X_val=np.zeros((20, 4)), y_val=np.zeros(20),
            wf_config={"enabled": True, "n_splits": 2},  # no label_horizon
        )
