"""FP-1 no-mock tests: G10 cross-section breadth.

Crosses (no USD leg) must compute carry differentials with the correct sign and
must not break signal causality / window-invariance over a mixed universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor import CROSSES, ALL_PAIRS, PAIRS
from src.factor.rates import _pair_ccys
from src.factor.signals import carry_signal, tsmom_signal, TSMOM_LOOKBACKS


def test_universe_is_broad_and_crosses_have_no_usd():
    assert len(ALL_PAIRS) >= 12
    assert set(PAIRS).issubset(set(ALL_PAIRS))
    for c in CROSSES:
        base, quote = _pair_ccys(c)
        assert "USD" not in (base, quote), f"{c} is not a cross"


def test_cross_carry_sign_is_base_minus_quote():
    # EUR_JPY long earns r_EUR - r_JPY. With EUR high and JPY low, signal > 0.
    idx = pd.bdate_range("2014-01-01", periods=4, tz="UTC")
    rd = pd.DataFrame(
        {"EUR_JPY": [2.0] * 4, "AUD_NZD": [0.0] * 4, "CHF_JPY": [-2.0] * 4},
        index=idx,
    )
    sig = carry_signal(rd)
    assert sig["EUR_JPY"].iloc[-1] == 1.0     # highest diff -> long
    assert sig["CHF_JPY"].iloc[-1] == -1.0    # lowest diff -> short


def test_tsmom_window_invariance_over_mixed_universe():
    rng = np.random.default_rng(0)
    cols = PAIRS[:3] + CROSSES[:3]
    idx = pd.bdate_range("2014-01-01", periods=400, tz="UTC")
    prices = {c: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 400))),
                           index=idx) for c in cols}
    panel = pd.DataFrame(prices)
    full = tsmom_signal(panel)
    trunc = tsmom_signal(panel.iloc[40:])
    common = full.index.intersection(trunc.index)
    common = common[common > panel.index[max(TSMOM_LOOKBACKS) + 40]]
    assert np.allclose(full.loc[common].to_numpy(),
                       trunc.loc[common].to_numpy(), equal_nan=True)
