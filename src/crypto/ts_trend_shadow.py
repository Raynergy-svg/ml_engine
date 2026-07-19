"""Crypto H5 TIME-SERIES trend SHADOW forward-tracking (research-adjacent, NOT execution).

2026-07-18 readiness report, execution-order step 2: H5 (30-day time-series /
trend momentum) is the crypto strategy with the best IS/OOS consistency of the
campaign (IS +0.91 / OOS +1.13, maxDD 16.2%, beta -0.073 — the ONLY crypto book
clearing 4 of 5 ex-history criteria), while the ACTIVE ``crypto_momentum``
shadow lane tracks H4 cross-sectional momentum — a DIFFERENT and weaker
strategy. Per the report: "Build a separate crypto_ts_trend lane instead of
overwriting H4." This module is that lane. H4's ledger and construction are
untouched; the two accumulate forward evidence independently.

H5 FAILED the significance criterion in its own pre-registration (DSR 0.496
< 0.95; block-bootstrap p 0.031 fails even the lenient Bonferroni), so it is
NOT a verified edge — the wall is structural (crypto effective-N ~ 3.9, ~6.5y
history). This lane exists to let a genuine forward record answer what the
backtest's statistical power cannot. Target per the acceptance rules: >= 52
independent weekly observations and 12 months before a serious read.

HARD LINE — SHADOW ONLY, same contract as ``momentum_shadow``:
  * Zero orders, zero broker/execution imports.
  * The construction is REUSED, not re-derived: every frozen parameter is
    imported from the verifier-confirmed pre-registered harness
    (``scripts/experiment_crypto_round2.py`` — TS_LOOKBACK, REBALANCE_D —
    and ``scripts/experiment_crypto_h2_infra_stress.py`` — vol target /
    window / leverage cap / ANN), never redefined or tuned here.
  * ``tests/test_strategy_lanes_2026_07_18.py::
    test_h5_net_series_matches_frozen_harness`` asserts this module's P&L
    series is bit-for-bit identical to the pre-registered
    ``ts_trend_backtest`` for the same inputs — it can never silently drift
    from the frozen construction.
  * Ledger plumbing (row schema, duplicate-asof guard, forward summary) is
    REUSED from ``momentum_shadow`` (same dataclass, same append/read
    helpers) with this lane's OWN ledger path and OWN frozen manifest.

Data cadence caveat: same free monthly-dump source as H4 — consecutive
cycles on the same calendar day/week are honest no-ops.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_crypto_xs_signals as _h            # noqa: E402  frozen harness (panels, costs)
import experiment_crypto_h2_infra_stress as _h2i     # noqa: E402  frozen overlay constants
import experiment_crypto_round2 as _round2           # noqa: E402  frozen H5 lookback + weekly cadence

from src.crypto.momentum_shadow import (             # noqa: E402  shared panel plumbing (reused, not re-derived)
    MIN_SIGNAL_UNIVERSE,
    ShadowCycleResult,
    _last_populated_date,
)
from src.evidence.forward_ledger import (            # noqa: E402  evidence-safe recording engine
    append_unseen_bars,
    cadence_counts,
    forward_rows,
    read_rows,
)

# --------------------------------------------------------------------------- #
# Frozen construction — imported constants ONLY. Never redefine/tune here.
# Source: docs/experiment-crypto-edge-hunt-round2-2026-06-29.md section 4 (H5).
# --------------------------------------------------------------------------- #
TS_LOOKBACK_D: int = _round2.TS_LOOKBACK                       # 30
REBALANCE_DAYS: int = _round2.REBALANCE_D                       # 7 (weekly)
COST_BPS: float = _h.COST_BPS                                   # 10.0
ANN: float = _h2i.ANN                                            # 365.0
TARGET_ANN_VOL: float = _h2i.TARGET_ANN_VOL                      # 0.10
VOL_WINDOW: int = _h2i.VOL_WINDOW                                 # 30
MAX_LEV: float = _h2i.MAX_LEV                                     # 3.0

SOURCE_DOC = "docs/experiment-crypto-edge-hunt-round2-2026-06-29.md#4-h5"
PRE_REGISTERED_IS_SHARPE = 0.91
PRE_REGISTERED_OOS_SHARPE = 1.13   # context only, NOT a live guarantee
GATE_VERDICT = ("clears_ex_history=FALSE (4 of 5 ex-history criteria clear; "
                "significance FAILS: DSR 0.496, boot p 0.031 — see source_doc)")

LEDGER_PATH_DEFAULT = REPO_ROOT / "trained_data" / "crypto" / "shadow_ts_trend_ledger.jsonl"


def construction_manifest() -> Dict[str, Any]:
    """The frozen H5 construction, exactly as reused — for ledger rows + AXIOM."""
    return {
        "signal": f"ts_trend_sign_{TS_LOOKBACK_D}d",
        "sizing": "inverse_vol_gross_normalized",
        "vol_target_ann": TARGET_ANN_VOL,
        "vol_window_d": VOL_WINDOW,
        "max_leverage": MAX_LEV,
        "rebalance_days": REBALANCE_DAYS,
        "cost_bps": COST_BPS,
        "dollar_neutral": False,   # directional by construction — carries market beta
        "source_doc": SOURCE_DOC,
        "pre_registered_is_sharpe": PRE_REGISTERED_IS_SHARPE,
        "pre_registered_oos_sharpe": PRE_REGISTERED_OOS_SHARPE,
        "gate_verdict": GATE_VERDICT,
    }


# --------------------------------------------------------------------------- #
# Book + P&L — identical math to _round2.ts_trend_backtest(vol_target=True),
# ALSO exposing the levered weight matrix (the harness computes it internally
# as Wlev but discards it). Regression-tested bit-for-bit against the frozen
# harness in tests/test_strategy_lanes_2026_07_18.py.
# --------------------------------------------------------------------------- #
def _compute_book_and_returns(
    close: pd.DataFrame,
    funding_daily: pd.DataFrame,
    eligible: pd.DataFrame,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    px = close[cols]
    fund = funding_daily[cols]
    elig = eligible[cols]
    price_ret = px.pct_change()

    sign = np.sign(px / px.shift(TS_LOOKBACK_D) - 1.0)
    inv_vol = 1.0 / price_ret.rolling(VOL_WINDOW, min_periods=10).std()
    raw = (sign * inv_vol).shift(1)
    raw = raw.where(elig.shift(1).fillna(False))

    dates = list(px.index)
    target_w = pd.Series(0.0, index=cols)
    weights: Dict[Any, pd.Series] = {}
    for i, d in enumerate(dates):
        if i % REBALANCE_DAYS == 0:
            r = raw.loc[d].dropna()
            w = pd.Series(0.0, index=cols)
            if len(r) >= MIN_SIGNAL_UNIVERSE:
                gross = r.abs().sum()
                if gross > 0:
                    w = (r / gross).reindex(cols).fillna(0.0)
            target_w = w
        weights[d] = target_w.copy()
    W = pd.DataFrame(weights).T.reindex(columns=cols).fillna(0.0)

    pr = price_ret.reindex(columns=cols).fillna(0.0)
    fr = fund.reindex(columns=cols).fillna(0.0)
    Wprev = W.shift(1).fillna(0.0)
    gross_ret = (Wprev * pr).sum(axis=1) + (-Wprev * fr).sum(axis=1)

    target_daily = TARGET_ANN_VOL / np.sqrt(ANN)
    realized = gross_ret.rolling(VOL_WINDOW, min_periods=10).std().shift(1)
    lev = (target_daily / realized).clip(upper=MAX_LEV).fillna(0.0)
    Wlev = W.mul(lev, axis=0)
    Wlevprev = Wlev.shift(1).fillna(0.0)
    turnover = (Wlev - Wlevprev).abs().sum(axis=1)
    price_pnl = (Wlevprev * pr).sum(axis=1)
    carry_pnl = (-Wlevprev * fr).sum(axis=1)

    gross = price_pnl + carry_pnl
    cost = turnover * (COST_BPS / 1e4)
    out = pd.DataFrame({
        "price": price_pnl, "carry": carry_pnl, "turnover": turnover,
        "cost": cost, "gross": gross, "net": gross - cost,
    })
    out.index = px.index
    return out, Wlev


def compute_shadow_cycle(refresh_klines: bool = False) -> ShadowCycleResult:
    """Pull the crypto universe (cached unless refresh_klines=True), compute
    the frozen H5 book as of the latest well-populated date, and return
    today's would-be positions + today's realized (mark-to-market) P&L."""
    close, funding_daily, eligible, _btc_ret, cols, _taker = _h.build_panels(False, refresh_klines)
    if close.empty or not cols:
        raise RuntimeError("crypto ts-trend shadow: empty universe/panel — data layer returned no data")
    cutoff = _last_populated_date(close, cols)
    close = close.loc[:cutoff]
    funding_daily = funding_daily.loc[:cutoff]
    eligible = eligible.loc[:cutoff]
    returns, book = _compute_book_and_returns(close, funding_daily, eligible, cols)

    asof = book.index[-1]
    row = book.loc[asof]
    longs = {str(s): float(w) for s, w in row.items() if w > 1e-12}
    shorts = {str(s): float(w) for s, w in row.items() if w < -1e-12}
    ret_row = returns.loc[asof]
    elig_shifted = eligible[cols].shift(1)
    universe_size = int(elig_shifted.loc[asof].sum()) if asof in elig_shifted.index else len(cols)

    return ShadowCycleResult(
        asof_date=str(pd.Timestamp(asof).date()),
        universe_size=universe_size,
        longs=longs,
        shorts=shorts,
        gross_leverage=float(row.abs().sum()),
        today_net_return=float(ret_row["net"]),
        today_price_return=float(ret_row["price"]),
        today_carry_return=float(ret_row["carry"]),
        today_cost=float(ret_row["cost"]),
        today_turnover=float(ret_row["turnover"]),
        construction=construction_manifest(),
    )


def _book_dict(row: "pd.Series") -> Dict[str, Dict[str, float]]:
    return {
        "longs": {str(s): float(w) for s, w in row.items() if w > 1e-12},
        "shorts": {str(s): float(w) for s, w in row.items() if w < -1e-12},
    }


def capture_forward(*, ledger_path: Path = LEDGER_PATH_DEFAULT,
                    cycle_ts_iso: str,
                    refresh_klines: bool = False,
                    panels: Any = None) -> List[Dict[str, Any]]:
    """Evidence-safe forward capture (2026-07-19 review fixes 2+3+5).

    Empty ledger -> activation BASELINE only (book snapshot, no realized
    return — the latest bar's return was already known at activation).
    Otherwise -> appends EVERY bar strictly after the last recorded asof
    (deterministic backfill of the frozen rule; missed scheduler days are
    recovered, never lost). Each forward row carries the period-aligned
    split: ``applied_book`` (Wlev at the PRIOR bar — the book that earned
    ``today_net_return``) vs ``book`` (Wlev at the row's bar — the standing
    next-bar target the hedge registry loads).

    ``panels`` is an injectable (close, funding, eligible, cols) tuple for
    tests / offline runs; production callers leave it None."""
    if panels is None:
        close, funding_daily, eligible, _btc, cols, _tk = _h.build_panels(False, refresh_klines)
    else:
        close, funding_daily, eligible, cols = panels
    if close.empty or not cols:
        raise RuntimeError("crypto ts-trend shadow: empty universe/panel")
    cutoff = _last_populated_date(close, cols)
    close, funding_daily, eligible = close.loc[:cutoff], funding_daily.loc[:cutoff], eligible.loc[:cutoff]
    out, book = _compute_book_and_returns(close, funding_daily, eligible, cols)
    elig_shifted = eligible[cols].shift(1)

    dates = [str(pd.Timestamp(d).date()) for d in book.index]
    pos_by_date = {d: i for i, d in enumerate(dates)}

    def _universe(i: int) -> int:
        idx = book.index[i]
        return int(elig_shifted.loc[idx].sum()) if idx in elig_shifted.index else len(cols)

    def payload_for(d: str) -> Dict[str, Any]:
        i = pos_by_date[d]
        idx = book.index[i]
        ret = out.loc[idx]
        applied = book.iloc[i - 1] if i > 0 else book.iloc[i] * 0.0
        nxt = book.iloc[i]
        return {
            "today_net_return": float(ret["net"]),
            "today_price_return": float(ret["price"]),
            "today_carry_return": float(ret["carry"]),
            "today_cost": float(ret["cost"]),
            "today_turnover": float(ret["turnover"]),
            "applied_book": _book_dict(applied),
            "book": _book_dict(nxt),
            "return_period_start": dates[i - 1] if i > 0 else None,
            "gross_leverage": float(nxt.abs().sum()),
            "universe_size": _universe(i),
            "n_longs": int((nxt > 1e-12).sum()),
            "n_shorts": int((nxt < -1e-12).sum()),
            "construction": construction_manifest(),
        }

    def activation_payload_for(d: str) -> Dict[str, Any]:
        i = pos_by_date[d]
        nxt = book.iloc[i]
        return {
            "book": _book_dict(nxt),
            "gross_leverage": float(nxt.abs().sum()),
            "universe_size": _universe(i),
            "n_longs": int((nxt > 1e-12).sum()),
            "n_shorts": int((nxt < -1e-12).sum()),
            "construction": construction_manifest(),
        }

    return append_unseen_bars(ledger_path, dates=dates, payload_for=payload_for,
                              activation_payload_for=activation_payload_for,
                              cycle_ts_iso=cycle_ts_iso)


def forward_summary(ledger_path: Path = LEDGER_PATH_DEFAULT) -> Dict[str, Any]:
    """Honest forward record: activation excluded; cadence counted in
    weeks/rebalances, never raw daily bars (52 bars != 52 weeks)."""
    rows = read_rows(ledger_path)
    fwd = forward_rows(rows)
    rets = [float(r["today_net_return"]) for r in fwd]
    n = len(rets)
    sharpe = None
    if n >= 2:
        mean = sum(rets) / n
        var = sum((x - mean) ** 2 for x in rets) / (n - 1)
        if var > 0:
            sharpe = (mean / var ** 0.5) * (ANN ** 0.5)
    return {
        "n_cycles": n,
        "first_asof_date": fwd[0]["asof_date"] if fwd else None,
        "last_asof_date": rows[-1]["asof_date"] if rows else None,
        "activated": bool(rows),
        "cumulative_return": float(rows[-1].get("cumulative_shadow_return", 0.0)) if rows else 0.0,
        "forward_sharpe_annualized": sharpe,
        "forward_sharpe_note": ("n<2 forward bars — Sharpe undefined, not reported as zero"
                                if sharpe is None else
                                "annualized from forward bars only (post-activation), NOT the backtest"),
        "cadence": cadence_counts(rows),
        "evidence_target": "52 independent WEEKLY observations (readiness report) — see cadence",
    }


__all__ = [
    "ANN", "COST_BPS", "GATE_VERDICT", "LEDGER_PATH_DEFAULT", "MAX_LEV",
    "PRE_REGISTERED_IS_SHARPE", "PRE_REGISTERED_OOS_SHARPE", "REBALANCE_DAYS",
    "SOURCE_DOC", "TARGET_ANN_VOL", "TS_LOOKBACK_D", "VOL_WINDOW",
    "ShadowCycleResult", "capture_forward", "compute_shadow_cycle",
    "construction_manifest", "forward_summary",
]
