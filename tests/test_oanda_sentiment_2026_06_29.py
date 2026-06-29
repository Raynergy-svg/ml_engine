"""OANDA position-book contrarian sentiment — pure-logic tests (no network, no mocks).

Verifies long-fraction extraction, the pre-registered contrarian sign, strict causality
of the signal/return panels (signal[t] vs return t->t+1, last row NaN), and that a
constructed contrarian relationship yields a positive IC + positive portfolio.
"""
from __future__ import annotations

import numpy as np

from src.equity.oanda_sentiment import (
    book_long_fraction,
    build_panels,
    contrarian_portfolio,
    contrarian_signal,
    pooled_ic,
)


def _book(t, price, long_pct):
    """One position book: long_pct% long, rest short (two buckets)."""
    return {"positionBook": {"time": t, "price": f"{price}",
                             "buckets": [{"longCountPercent": long_pct,
                                          "shortCountPercent": 100 - long_pct}]}}


def test_long_fraction_and_contrarian_sign():
    assert abs(book_long_fraction(_book("t", 1.1, 70)) - 0.70) < 1e-9
    assert contrarian_signal(0.70) < 0          # crowded long -> fade (negative)
    assert contrarian_signal(0.30) > 0          # crowded short -> buy (positive)
    assert book_long_fraction({"positionBook": {"buckets": []}}) is None


def test_panels_are_causal_and_last_return_nan():
    # day t price then t+1 price; return[t] must be price[t+1]/price[t]-1
    books = {"EUR_USD": [
        _book("2024-01-01T17:00:00Z", 1.00, 60),
        _book("2024-01-02T17:00:00Z", 1.02, 55),
        _book("2024-01-03T17:00:00Z", 0.99, 50),
    ]}
    sig, ret = build_panels(books)
    assert list(sig.columns) == ["EUR_USD"]
    assert abs(ret["EUR_USD"].iloc[0] - (1.02 / 1.00 - 1)) < 1e-9   # t->t+1
    assert abs(ret["EUR_USD"].iloc[1] - (0.99 / 1.02 - 1)) < 1e-9
    assert np.isnan(ret["EUR_USD"].iloc[-1])                        # no future beyond last
    assert abs(sig["EUR_USD"].iloc[0] - (-(0.60 - 0.5))) < 1e-9     # S = -(NL-0.5)


def test_constructed_contrarian_gives_positive_ic():
    # Build a series where crowded-long (high NL) reliably precedes a price DROP.
    import pandas as pd
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=120)
    price = 1.00
    books_x = []
    for d in dates:
        nl = float(rng.uniform(0.3, 0.7))
        books_x.append(_book(d.isoformat() + "Z", price, int(round(nl * 100))))
        price *= 1.0 - (nl - 0.5) * 0.05   # contrarian truth: crowded long -> next price falls
    sig, ret = build_panels({"X": books_x})
    ic = pooled_ic(sig, ret)
    assert ic["ic"] > 0.3 and ic["t_stat"] > 2     # contrarian signal predicts (by construction)
    net = contrarian_portfolio(sig, ret)
    assert net.sum() > 0                            # fading the crowd makes money here
