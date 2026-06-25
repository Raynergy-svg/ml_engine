"""PIT quality-panel builder — turns dumped fundamentals into a lookahead-proof
quality score panel for the quality sleeve (offline; running:NO).

This is the RECEIVING END for the operator-authorized PIT fundamental-data
purchase (2026-06-25). It fabricates NO data: it consumes a dump of per-ticker
quarterly fundamental records (the financial-datasets / Sharadar schema — each
record carries a fiscal ``report_period`` and the quality fields) and produces a
date×ticker quality-SCORE panel aligned to the universe membership grid, suitable
for ``quality.quality_tilt_weights``.

POINT-IN-TIME DISCIPLINE (the #1 lie vector, L-018): a quarterly filing for the
period ENDING ``report_period`` is not public until it is FILED ~40-60 days later.
Using ``report_period`` as the as-of date injects future knowledge (announcement
lag) and, with later vendor restatements, lookahead. So every record is lagged to
a conservative public-availability date ``report_period + lag_days`` (default 90d,
which safely clears even late large-cap filers) and forward-filled from there. A
value can therefore only influence a weight STRICTLY AFTER it was knowable.
``validate_panel_pit`` re-derives this independently and refuses on any violation.

PRE-REGISTERED (anti-p-hacking, fixed BEFORE any quality number exists):
  - score Q = mean of the AVAILABLE cross-sectional z-scores of
    {gross_margin, net_margin, -debt_to_equity}  (profitability minus leverage;
    canonical Novy-Marx gross-profitability + a leverage penalty), tilt=1.0.
  - availability lag = 90 calendar days; quarterly cadence; forward-fill to grid.
  - single comparison: quality-tilt book vs baseline EW, full + OOS, SAME ship
    gate, plus the disjoint-5y sub-period stability bar the other sleeves use.

SEPARATE module; baseline (`ship_gate`/`backtest`) NOT modified. NON-hot-path.
OFFLINE evaluation only — nothing here runs in a process, trades, or unhalts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pre-registered, untuned. Components are signed so that HIGHER = better quality.
QUALITY_COMPONENTS = (
    ("gross_margin", +1.0),   # Novy-Marx gross profitability
    ("net_margin", +1.0),     # bottom-line profitability
    ("debt_to_equity", -1.0),  # leverage penalty (lower is higher quality)
)
DEFAULT_LAG_DAYS = 90          # conservative public-availability lag (no lookahead)
DEFAULT_MIN_COVERAGE = 0.60    # below this, the panel is too sparse to trust -> caller refuses


def load_metrics_dump(path: Path) -> Dict[str, List[dict]]:
    """Load a {ticker: [record, ...]} fundamentals dump written from the data vendor.

    Graceful: a missing/corrupt file yields an empty mapping (caller treats an
    empty panel as NOT_EVALUATED — never a fabricated score). Each record is a
    raw vendor dict with at least ``report_period`` and some QUALITY_COMPONENTS.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("quality dump unreadable at %s (%s) -> empty panel", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("quality dump at %s is not a {ticker: [...]} object -> empty", path)
        return {}
    out: Dict[str, List[dict]] = {}
    for ticker, recs in raw.items():
        if isinstance(recs, list):
            out[str(ticker)] = [r for r in recs if isinstance(r, dict)]
    return out


def availability_date(report_period: str, *, lag_days: int = DEFAULT_LAG_DAYS) -> pd.Timestamp:
    """Conservative PIT date a filing became public: fiscal period end + lag_days."""
    return (pd.Timestamp(report_period, tz="UTC").normalize()
            + pd.Timedelta(days=int(lag_days)))


def _raw_quality_frame(records: Sequence[Mapping]) -> pd.DataFrame:
    """Per-ticker frame indexed by availability_date with one column per component
    plus the source report_period (kept for independent PIT re-derivation)."""
    rows = []
    for r in records:
        rp = r.get("report_period")
        if not rp:
            continue
        try:
            avail = availability_date(rp)
        except (ValueError, TypeError):
            continue
        row = {"report_period": pd.Timestamp(rp, tz="UTC").normalize(), "_avail": avail}
        for col, _sign in QUALITY_COMPONENTS:
            v = r.get(col)
            row[col] = float(v) if isinstance(v, (int, float)) else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["report_period", "_avail", *[c for c, _ in QUALITY_COMPONENTS]])
    df = pd.DataFrame(rows).sort_values("_avail")
    # If two filings share an availability date, the later-reported one wins.
    df = df.drop_duplicates(subset="_avail", keep="last").set_index("_avail")
    return df


def build_quality_panel(
    dump: Mapping[str, Sequence[Mapping]],
    membership_index: pd.DatetimeIndex,
    members: Sequence[str],
    *,
    lag_days: int = DEFAULT_LAG_DAYS,
) -> pd.DataFrame:
    """Build a date×ticker PIT quality-SCORE panel aligned to the membership grid.

    For each ticker: lag every record to ``report_period + lag_days``, forward-fill
    the raw components onto the grid (a fundamental persists until the next filing
    is public), z-score each component cross-sectionally per date, sign-combine the
    AVAILABLE components into the composite Q, and return that single score panel.
    A date/ticker with no public filing yet is NaN (never zero-filled).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(membership_index, utc=True)).normalize()
    members = list(members)
    # Stage 1: forward-filled raw component panels, strictly causal (lagged source).
    comp_panels: Dict[str, pd.DataFrame] = {
        col: pd.DataFrame(np.nan, index=idx, columns=members) for col, _ in QUALITY_COMPONENTS
    }
    for tkr in members:
        recs = dump.get(tkr) or []
        rf = _raw_quality_frame(recs)
        if rf.empty:
            continue
        for col, _sign in QUALITY_COMPONENTS:
            s = rf[col].dropna()
            if s.empty:
                continue
            # reindex onto the grid by last-known-public value (ffill from _avail)
            comp_panels[col][tkr] = s.reindex(idx, method="ffill")
    # Stage 2: cross-sectional z per date, sign-combined into the composite score.
    zsum = pd.DataFrame(0.0, index=idx, columns=members)
    zcnt = pd.DataFrame(0.0, index=idx, columns=members)
    for col, sign in QUALITY_COMPONENTS:
        p = comp_panels[col]
        mu = p.mean(axis=1)
        sd = p.std(axis=1).replace(0.0, np.nan)
        z = p.sub(mu, axis=0).div(sd, axis=0) * float(sign)
        present = z.notna()
        zsum = zsum.add(z.where(present, 0.0), fill_value=0.0)
        zcnt = zcnt.add(present.astype(float), fill_value=0.0)
    score = zsum.div(zcnt.replace(0.0, np.nan))   # mean of available z-scores; NaN if none
    return score


def panel_coverage(panel: pd.DataFrame, membership: pd.DataFrame) -> float:
    """Fraction of ACTIVE (member, date) cells that have a non-NaN quality score."""
    active = membership.reindex(index=panel.index, columns=panel.columns).fillna(False).astype(bool)
    total = int(active.values.sum())
    if total == 0:
        return 0.0
    have = int((panel.notna() & active).values.sum())
    return have / total


def validate_panel_pit(
    panel: pd.DataFrame,
    dump: Mapping[str, Sequence[Mapping]],
    *,
    lag_days: int = DEFAULT_LAG_DAYS,
) -> None:
    """Independently re-derive the no-lookahead guarantee; raise on any violation.

    For every ticker, the FIRST date at which the panel has a non-NaN score must be
    on/after that ticker's earliest public-availability date. This catches the
    classic bug of indexing fundamentals by ``report_period`` (period end) instead
    of the filing-availability date — which would surface a score before it could
    have been known.
    """
    for tkr in panel.columns:
        col = panel[tkr].dropna()
        if col.empty:
            continue
        recs = dump.get(tkr) or []
        avails = []
        for r in recs:
            rp = r.get("report_period")
            if rp and any(isinstance(r.get(c), (int, float)) for c, _ in QUALITY_COMPONENTS):
                avails.append(availability_date(rp, lag_days=lag_days))
        if not avails:
            raise ValueError(f"PIT violation: {tkr} has scores but no source filing in dump")
        earliest_known = min(avails)
        first_scored = col.index.min()
        if first_scored < earliest_known:
            raise ValueError(
                f"PIT violation: {tkr} scored at {first_scored.date()} but earliest "
                f"public filing is {earliest_known.date()} (lookahead of "
                f"{(earliest_known - first_scored).days}d)"
            )


def run_quality_bakeoff(
    *,
    dump_path: Path = Path("market_data/equity/quality_metrics.json"),
    snapshot_path: Path = Path("market_data/equity/universe_snapshot_pit.json"),
    out_path: Path = Path("trained_data/backtests/variant_bakeoff.json"),
    lag_days: int = DEFAULT_LAG_DAYS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    oos_fraction: float = 0.34,
    generated_at: str = "",
) -> Dict[str, object]:
    """OFFLINE runbook entry that fires the moment a real PIT dump is present.

    Loads the dump -> builds the PIT panel -> INDEPENDENTLY validates no-lookahead
    -> enforces a coverage floor (below it, refuse with NOT_EVALUATED rather than
    score a thin/biased panel, per L-018) -> runs the existing 4-arm bake-off with
    the real quality panel. running:NO — backtest only. Returns the bake-off payload
    on success, or a {status: NOT_EVALUATED, reason} dict if the data isn't usable.
    """
    from src.equity.universe import UniverseSnapshot
    from src.equity.variant_eval import run_bakeoff

    dump = load_metrics_dump(Path(dump_path))
    if not dump:
        return {"status": "NOT_EVALUATED", "reason": f"no quality dump at {dump_path} "
                "(operator-authorized PIT data not yet sourced); quality stays unevaluated."}
    snap = UniverseSnapshot.load(Path(snapshot_path))
    members = list(snap.membership.columns)
    panel = build_quality_panel(dump, snap.membership.index, members, lag_days=lag_days)
    validate_panel_pit(panel, dump, lag_days=lag_days)  # raises on any lookahead
    cov = panel_coverage(panel, snap.membership.astype(bool))
    if cov < float(min_coverage):
        return {"status": "NOT_EVALUATED", "reason": f"PIT quality coverage {cov:.2%} "
                f"< floor {min_coverage:.0%} — too sparse to trust; widen the purchase "
                "or raise lag tolerance before evaluating.", "coverage": round(cov, 4)}
    logger.info("quality bake-off: PIT panel validated, coverage=%.2f%%", cov * 100)
    return run_bakeoff(
        snapshot_path=Path(snapshot_path),
        out_path=Path(out_path),
        oos_fraction=oos_fraction,
        quality_panel=panel,
        generated_at=generated_at,
    )
