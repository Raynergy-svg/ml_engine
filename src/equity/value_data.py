"""PIT value-panel builder — price-joined value factor (offline; running:NO).

Unlike ``quality_data.build_quality_panel`` (a pure fundamental-ratio panel that
never touches price), the classic VALUE factor requires a PIT fundamentals x PIT
price JOIN: book-to-market and earnings-yield are fundamentals SCALED BY market
cap, and market cap needs same-day price. This module is that new mechanism.

PRE-REGISTERED (docs/experiment-sec-edgar-value-accruals-2026-07-02.md, frozen
before any factor Sharpe number was computed):
  - market_cap(t)      = shares_outstanding_PIT(t) * price(t)
  - book_to_market(t)  = equity_PIT(t) / market_cap(t)
  - earnings_yield(t)  = net_income_PIT(t) / market_cap(t)
  - value_score        = mean available cross-sectional z of
                          {+book_to_market, +earnings_yield}, tilt=1.0

CAUSALITY: ``equity``/``net_income``/``shares_outstanding`` are held at their
real EDGAR ``filed`` availability date (forward-filled, never backfilled — the
same lagging discipline as ``quality_data.build_quality_panel``). ``price(t)``
is same-day (public real-time, no lag needed for a price feed). The join is
causal because the fundamentals leg is ALWAYS the PIT-lagged, filing-available
value at date t; it is never re-fetched from the future. A date/ticker with no
public filing yet (before the earliest 10-K's filed date) is NaN — never
fabricated/zero-filled/backfilled.

SEPARATE module; ``quality_data``/``quality``/baseline (`ship_gate`/`backtest`)
NOT modified. NON-hot-path. OFFLINE evaluation only — nothing here runs in a
process, trades, or unhalts.
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from src.equity.quality_data import record_avail

logger = logging.getLogger(__name__)

# Pre-registered, untuned. Signed so that HIGHER = more attractively valued
# (classic value: cheap relative to book / earnings).
VALUE_COMPONENTS = ("book_to_market", "earnings_yield")
_RAW_FIELDS = ("equity", "net_income", "shares_outstanding")

# BUGFIX 2026-07-02 (independent Model QA verification): an UNBOUNDED forward-fill
# let a stale shares_outstanding observation (e.g. Berkshire Hathaway's last dei
# EntityCommonStockSharesOutstanding filing was 2011-02-28) ride forward for 15
# years against a live, still-updating equity/net_income series, producing
# book_to_market values ~1500x too large for BRK-B and 11 other large-cap names
# (CMCSA, CME, F, HSY, IBKR, MA, REGN, SPG, TAP, UPS, V). A raw fundamental is
# knowable from its filed date, but silently carrying it forward for years after
# a peer field (equity/net_income) has moved on is the same "fabricate beyond
# what's knowable" failure mode the "never zero-fill missing features silently"
# gate targets — just for time-since-last-update instead of missingness.
# Cap every raw field's forward-fill at this many days past its own last real
# observation; beyond that, NaN (never stale-carried indefinitely).
_MAX_STALENESS_DAYS = 730


def _raw_fundamentals_frame(records: Sequence[Mapping]) -> pd.DataFrame:
    """Per-ticker frame indexed by PIT availability date, one column per raw field.

    Uses the real EDGAR ``filed`` date (never a lag proxy — these are true-PIT
    annual records from ``edgar_fundamentals.quality_records_from_facts``).
    """
    rows = []
    for r in records:
        rp = r.get("report_period")
        if not rp:
            continue
        try:
            avail = record_avail(r)
        except (ValueError, TypeError, KeyError):
            continue
        row = {"_avail": avail}
        for col in _RAW_FIELDS:
            v = r.get(col)
            row[col] = float(v) if isinstance(v, (int, float)) else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["_avail", *_RAW_FIELDS]).set_index("_avail")
    df = pd.DataFrame(rows).sort_values("_avail")
    df = df.drop_duplicates(subset="_avail", keep="last").set_index("_avail")
    return df


def _ffill_with_staleness_cap(
    s: pd.Series, idx: pd.DatetimeIndex, *, max_age_days: int = _MAX_STALENESS_DAYS,
) -> pd.Series:
    """Forward-fill ``s`` (dated observations) onto ``idx``, but NaN out any date
    more than ``max_age_days`` past the observation being carried forward.

    Prevents an issuer that stopped filing a concept (e.g. Berkshire's dei
    shares-outstanding tag going dormant in 2011 while other concepts kept
    updating) from silently propagating a years-stale value indefinitely.
    """
    filled = s.reindex(idx, method="ffill")
    last_obs_date = s.index.to_series().reindex(idx, method="ffill")
    age_days = (idx.to_series().values - last_obs_date.values) / np.timedelta64(1, "D")
    stale = age_days > float(max_age_days)
    return filled.where(~stale)


def _pit_raw_panel(
    dump: Mapping[str, Sequence[Mapping]],
    price_index: pd.DatetimeIndex,
    members: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Forward-filled PIT panels for each raw field, aligned to the DAILY price grid.

    Each field's forward-fill is capped at ``_MAX_STALENESS_DAYS`` past its own
    last real observation (see ``_ffill_with_staleness_cap``) — a value can only
    be carried forward for so long before it's no longer a defensible PIT proxy.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(price_index, utc=True)).normalize()
    members = list(members)
    panels: Dict[str, pd.DataFrame] = {
        col: pd.DataFrame(np.nan, index=idx, columns=members) for col in _RAW_FIELDS
    }
    for tkr in members:
        recs = dump.get(tkr) or []
        rf = _raw_fundamentals_frame(recs)
        if rf.empty:
            continue
        for col in _RAW_FIELDS:
            s = rf[col].dropna() if col in rf.columns else pd.Series(dtype=float)
            if s.empty:
                continue
            panels[col][tkr] = _ffill_with_staleness_cap(s, idx)
    return panels


def build_value_panel(
    dump: Mapping[str, Sequence[Mapping]],
    prices: pd.DataFrame,
    membership_index: pd.DatetimeIndex,
    members: Sequence[str],
    *,
    components: Sequence[str] = VALUE_COMPONENTS,
    clip_z: float = None,
) -> pd.DataFrame:
    """Build a date x ticker PIT value-SCORE panel, causal fundamentals x price join.

    ``prices`` must be a date x ticker close-price DataFrame covering (at least)
    ``membership_index``. Market cap = PIT shares_outstanding * same-day price;
    book-to-market = PIT equity / market_cap; earnings_yield = PIT net_income /
    market_cap. Each component is cross-sectionally z-scored per date, then
    averaged over the AVAILABLE components (mirrors
    ``quality_data.build_quality_panel``'s stage-2 combine). ``components``
    overrides the pre-registered set (robustness sensitivities, e.g.
    earnings-yield-only). ``clip_z`` winsorizes each component's z before
    combining. A date/ticker with no computable component is NaN.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(membership_index, utc=True)).normalize()
    members = list(members)
    px = prices.reindex(index=idx, columns=members)

    raw = _pit_raw_panel(dump, idx, members)
    shares = raw["shares_outstanding"]
    equity = raw["equity"]
    net_income = raw["net_income"]

    market_cap = shares * px
    market_cap = market_cap.where(market_cap > 0.0)  # non-positive cap is not usable

    comp_panels: Dict[str, pd.DataFrame] = {}
    comp_panels["book_to_market"] = equity / market_cap
    comp_panels["earnings_yield"] = net_income / market_cap

    zsum = pd.DataFrame(0.0, index=idx, columns=members)
    zcnt = pd.DataFrame(0.0, index=idx, columns=members)
    for col in components:
        p = comp_panels[col]
        mu = p.mean(axis=1)
        sd = p.std(axis=1).replace(0.0, np.nan)
        z = p.sub(mu, axis=0).div(sd, axis=0)  # signed +1: HIGHER = cheaper = better
        if clip_z is not None:
            z = z.clip(-float(clip_z), float(clip_z))
        present = z.notna()
        zsum = zsum.add(z.where(present, 0.0), fill_value=0.0)
        zcnt = zcnt.add(present.astype(float), fill_value=0.0)
    score = zsum.div(zcnt.replace(0.0, np.nan))  # mean of available z-scores; NaN if none
    return score


def panel_coverage(panel: pd.DataFrame, membership: pd.DataFrame) -> float:
    """Fraction of ACTIVE (member, date) cells that have a non-NaN value score.

    Duplicated (not imported) from ``quality_data.panel_coverage`` — kept local
    so this module has no runtime dependency beyond ``quality_data.record_avail``.
    """
    active = membership.reindex(index=panel.index, columns=panel.columns).fillna(False).astype(bool)
    total = int(active.values.sum())
    if total == 0:
        return 0.0
    have = int((panel.notna() & active).values.sum())
    return have / total


def validate_value_panel_pit(
    panel: pd.DataFrame,
    dump: Mapping[str, Sequence[Mapping]],
) -> None:
    """Independently re-derive the no-lookahead guarantee for the VALUE panel.

    For every ticker, the FIRST date at which the panel has a non-NaN score must
    be on/after that ticker's earliest fundamentals public-availability date
    (the earliest ``filed`` date among records carrying ANY of equity /
    net_income / shares_outstanding). This mirrors
    ``quality_data.validate_panel_pit`` but is independently re-implemented
    (not imported) so a bug in one validator cannot silently mask a bug in the
    other. It specifically catches: (a) a lookahead bug in the price join
    (e.g. indexing market_cap by ``report_period`` instead of ``filed``), and
    (b) a lookahead bug from the dei shares-outstanding filed-date join itself.
    """
    for tkr in panel.columns:
        col = panel[tkr].dropna()
        if col.empty:
            continue
        recs = dump.get(tkr) or []
        avails = []
        for r in recs:
            rp = r.get("report_period")
            if rp and any(isinstance(r.get(c), (int, float)) for c in _RAW_FIELDS):
                try:
                    avails.append(record_avail(r))
                except (ValueError, TypeError, KeyError):
                    continue
        if not avails:
            raise ValueError(f"PIT violation: {tkr} has value scores but no source filing in dump")
        earliest_known = min(avails)
        first_scored = col.index.min()
        if first_scored < earliest_known:
            raise ValueError(
                f"PIT violation: {tkr} value-scored at {first_scored.date()} but earliest "
                f"public filing is {earliest_known.date()} (lookahead of "
                f"{(earliest_known - first_scored).days}d)"
            )
