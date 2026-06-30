"""Cross-sectional variant bake-off — pure-function tests (offline; running:NO).

Verifies the harness is apples-to-apples (EW arm reproduces
`ship_gate.evaluate_harvester`) and that EACH variant weight construction is
(1) STRICTLY CAUSAL (perturbing a price at t cannot change weights at any index
<= t), (2) long-only, and (3) row-normalized over the active set. Causality is
the #1 lie vector in these backtests.

No mocks: real synthetic panels + real `ship_gate`/`backtest`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equity.hrp import hrp_weights
from src.equity.lowvol import lowvol_weights
from src.equity.quality import quality_tilt_weights
from src.equity.ship_gate import build_equal_weight_panel, evaluate_harvester
from src.equity.universe import UniverseRules, UniverseSnapshot
from src.equity.variant_eval import evaluate_book

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _panels(seed=7, n=700):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n, tz="UTC").normalize()
    membership = pd.DataFrame(
        np.ones((n, len(TICKERS)), dtype=bool), index=dates, columns=TICKERS
    )
    # Give each ticker a distinct vol so low-vol tilt is testable.
    vols = np.array([0.006, 0.009, 0.012, 0.016, 0.022])
    rets = rng.normal(0.0004, 1.0, size=(n, len(TICKERS))) * vols
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=TICKERS
    )
    return membership, prices


def _snap(membership):
    return UniverseSnapshot(
        membership=membership,
        rules=UniverseRules(),
        built_at="2026-06-19T00:00:00Z",
        pipeline_version="2026-06-18-eq1",
        universe_hash="abcd" * 16,
    )


def _assert_long_only_normalized(w, active_mask):
    assert (w.values >= -1e-12).all()  # long-only
    row_sum = w.sum(axis=1)
    # rows with any active member must sum to ~1
    has_active = active_mask.sum(axis=1) > 0
    assert np.allclose(row_sum[has_active].values, 1.0, atol=1e-9)


def test_evaluate_book_ew_reproduces_baseline():
    membership, prices = _panels()
    snap = _snap(membership)
    ew = build_equal_weight_panel(snap, prices)
    got = evaluate_book(snap, prices, ew)
    exp = evaluate_harvester(snap, prices, adv=None)
    assert abs(got["net_sharpe"] - exp.net_sharpe) < 1e-9
    assert abs(got["max_dd"] - exp.max_dd) < 1e-9
    assert got["gate_pass"] == exp.gate_pass


def _causal(weight_fn):
    membership, prices = _panels()
    snap = _snap(membership)
    base = weight_fn(snap, prices)
    t = 300
    p2 = prices.copy()
    p2.iloc[t] = p2.iloc[t] * 1.5  # perturb price at t (still finite/positive)
    perturbed = weight_fn(snap, p2)
    # weights at every index <= t must be byte-identical (depend only on data <= t-1)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], perturbed.iloc[: t + 1])
    # and the perturbation DOES propagate forward (not a frozen panel)
    assert not base.iloc[t + 1:].equals(perturbed.iloc[t + 1:])


def test_lowvol_is_causal():
    _causal(lowvol_weights)


def test_hrp_is_causal():
    _causal(hrp_weights)


def test_lowvol_long_only_normalized_and_tilts_low_vol():
    membership, prices = _panels()
    snap = _snap(membership)
    w = lowvol_weights(snap, prices)
    ew = build_equal_weight_panel(snap, prices)
    _assert_long_only_normalized(w, ew > 0)
    # AAA is the lowest-vol name, EEE the highest -> AAA overweighted vs EEE on average
    tail = w.iloc[400:]
    assert tail["AAA"].mean() > tail["EEE"].mean()


def test_hrp_long_only_normalized():
    membership, prices = _panels()
    snap = _snap(membership)
    w = hrp_weights(snap, prices)
    ew = build_equal_weight_panel(snap, prices)
    _assert_long_only_normalized(w, ew > 0)


def test_quality_mechanism_tilts_to_high_score_long_only():
    membership, prices = _panels()
    snap = _snap(membership)
    ew = build_equal_weight_panel(snap, prices)
    # Synthetic PIT score panel (constant in time): AAA best, EEE worst.
    scores = pd.DataFrame(
        np.tile([5.0, 4.0, 3.0, 2.0, 1.0], (len(ew.index), 1)),
        index=ew.index,
        columns=TICKERS,
    )
    w = quality_tilt_weights(snap, prices, scores)
    _assert_long_only_normalized(w, ew > 0)
    assert w["AAA"].mean() > w["EEE"].mean()  # tilts toward high quality
