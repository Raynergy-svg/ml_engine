"""No-mock regression test for overlay()'s since-inception drawdown floor.

Real pandas/numpy only, no ``unittest.mock`` (project no-mock policy — see
``.claude/rules/improvement.md`` "No-Mock Rule"). This pins the mechanism
identified as the dominant driver behind the 2026-07-04 FINRA short-volume
H2 experiment's misleading "Sharpe 1.231" headline (see
``docs/prereg-finra-short-volume-2026-07-04.md`` and
``trained_data/backtests/finra_short_volume_20260704T155944Z.json``): once a
raw book's SINCE-INCEPTION cumulative drawdown (running peak from bar 0, not
a rolling window) breaches ``dd_hard``, ``overlay()``'s exposure scalar
floors at 0.0 and stays there for as long as equity remains below that
earlier peak — even when every subsequent per-bar return is strongly
positive. 6 of 7.5 years in that experiment contributed exactly zero return
via this floor, not signal decay. This test documents the floor as an
intentional, tested property rather than a surprise to be re-discovered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.equity.backtest import ANN, overlay

N_WARMUP = 5  # flat bars before the crash, so a real pre-crash peak exists
CRASH_IDX = N_WARMUP  # index of the single deep drawdown bar
CRASH_RETURN = -0.40
N_RECOVERY = 25  # bars of sustained positive return after the crash
DD_SOFT = 0.10
DD_HARD = 0.20
TARGET_VOL = 0.10


def _build_since_inception_dd_fixture() -> pd.Series:
    """Flat warm-up -> one deep crash -> sustained positive "recovery".

    The crash alone (-40%) breaches ``dd_hard=0.20`` with room to spare.
    The recovery leg alternates +1.5%/+0.5% (avg +1%/day) for 25 bars --
    long enough to run past the 21-bar rolling-vol window ``overlay()`` also
    uses (so the rolling-vol term genuinely re-arms), but small enough that
    cumulative equity never climbs back above ``peak * (1 - dd_hard)``
    (0.80): the deepest post-crash recovery only reaches ~0.773 of the
    original peak. That gap -- confirmed numerically below, not assumed --
    is what isolates the since-inception DD floor from a rolling-window
    artifact.
    """
    idx = pd.bdate_range(start="2025-01-02", periods=N_WARMUP + 1 + N_RECOVERY)
    recovery = [0.015 if i % 2 == 0 else 0.005 for i in range(N_RECOVERY)]
    rets = [0.0] * N_WARMUP + [CRASH_RETURN] + recovery
    return pd.Series(rets, index=idx)


def test_since_inception_dd_floor_persists_through_positive_recovery():
    """Since-inception DD > dd_hard floors the scalar at 0 for the rest of
    the sample, even though every post-crash return is positive -- because
    equity never reclaims the pre-crash peak (not a rolling-window effect).
    """
    base = _build_since_inception_dd_fixture()
    eq = (1 + base).cumprod()
    peak = eq.cummax()

    # --- Fixture sanity: prove the setup does what the docstring claims ---
    assert (base.iloc[CRASH_IDX + 1:] > 0).all(), "recovery leg must be positive"
    raw_dd_at_crash = 1.0 - eq.iloc[CRASH_IDX] / peak.iloc[CRASH_IDX]
    assert raw_dd_at_crash > DD_HARD, "fixture bug: crash must breach dd_hard"
    # Since-inception (not rolling): running peak never drops, and the whole
    # post-crash path stays under the dd_hard recovery threshold.
    dd_hard_recovery_level = peak.iloc[CRASH_IDX] * (1.0 - DD_HARD)
    assert eq.iloc[CRASH_IDX:].max() < dd_hard_recovery_level, (
        "fixture bug: equity must never climb back above the dd_hard "
        "recovery threshold, or the floor would legitimately lift"
    )

    scalar = overlay(
        base, target_vol=TARGET_VOL, dd_soft=DD_SOFT, dd_hard=DD_HARD, max_lev=1.0
    )

    # Causal (shift(1)): the bar right after the crash is the first whose
    # scalar reflects the breach. It stays 0.0 through the end of the
    # sample -- the entire positive-return "recovery" leg.
    post_crash = scalar.iloc[CRASH_IDX + 1:]
    assert (post_crash == 0.0).all(), (
        "overlay must floor at 0.0 for the entire positive-return recovery "
        "leg -- since-inception DD never drops back below dd_hard"
    )

    # --- Isolate the mechanism: prove it's the DD term, not the 21-bar ---
    # --- rolling-vol term, holding the scalar at zero.                 ---
    # Recompute overlay()'s own vol-target component in isolation. Once the
    # 21-bar rolling window is past its NaN warm-up, this term is genuinely
    # live (non-zero) on the alternating-return recovery leg. If overlay()
    # were still 0.0 in that same window, the zero cannot be coming from
    # the rolling-vol term -- it has to be the since-inception DD term.
    rvol = base.rolling(21).std().mul(np.sqrt(ANN)).shift(1)
    s_vol_only = (TARGET_VOL / rvol).clip(upper=1.0).fillna(0.0)
    live_vol_window = s_vol_only.iloc[-6:]
    assert (live_vol_window > 0.0).all(), (
        "sanity: the rolling-vol term should be live (non-zero, past its "
        "21-bar NaN warm-up) in this window"
    )
    assert (scalar.iloc[-6:] == 0.0).all(), (
        "overlay is 0.0 even where the rolling-vol term alone is non-zero "
        "-- confirms the since-inception DD floor, not a rolling-window "
        "artifact, is what's suppressing exposure"
    )
