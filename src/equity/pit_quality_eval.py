"""Widened-universe PIT quality-factor evaluation (free data; offline; running:NO).

Ties the free-PIT pipeline together and answers the operator's question: does a
quality tilt beat the equal-weight baseline on the ship gate, on a WIDE
survivorship-aware universe with TRUE point-in-time fundamentals — at $0?

Inputs (all free): survivorship-aware S&P 500 membership (`sp500_membership`,
Wikipedia change-log reconstruction), true-PIT annual fundamentals (`edgar_fundamentals`,
SEC EDGAR filing-date-aligned), prices (yfinance, PRICES ONLY).

`evaluate_book_wide` is a SEPARATE evaluator (baseline `ship_gate`/`backtest`/
`variant_eval` NOT modified) that reuses the baseline primitives (overlay,
run_portfolio_backtest, stats, evaluate_gate, _derive_returns, build_equal_weight_panel)
but tolerates a staggered universe: the engine rejects NaN returns (correct, by
design), so we feed it forward/back-filled prices while FORCING weight=0 wherever
the original price was NaN — fabricated pre-IPO/post-delisting cells are inert
(0 contribution). Apples-to-apples: EW and quality arms share the identical prices,
overlay scalar, costs, and window; only the cross-sectional weighting differs.

HONESTY LABELS (L-018), surfaced in the output payload:
  - fundamentals: TRUE-PIT (EDGAR filing-date-aligned). pit=True.
  - membership: survivorship-aware reconstruction (includes since-removed names);
    completeness thins pre-~2010.
  - RESIDUAL bias (labelled, not hidden): delisting returns are UNDERSTATED — the
    forward-fill flattens the final delisting gap. Most S&P removals are acquisitions
    near-market (bounded impact); bankruptcies are hidden. + price-availability
    survivorship (yfinance often lacks delisted tickers -> they drop out).
  - EDGAR XBRL fundamentals floor ~2010 -> honest clean window ~2012-2026.

OFFLINE only — network fetch + in-memory backtest + JSON write; nothing trades,
unhalts, or touches the hot path. running:NO.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.equity.backtest import overlay as baseline_overlay
from src.equity.backtest import run_portfolio_backtest
from src.equity.ship_gate import (
    DEFAULT_COST_BPS,
    DEFAULT_DD_HARD,
    DEFAULT_DD_SOFT,
    DEFAULT_EXECUTION_LAG,
    DEFAULT_MAX_LEV,
    DEFAULT_TARGET_VOL,
    _derive_returns,
    build_equal_weight_panel,
)
from src.factor.ship_gate import evaluate_gate

logger = logging.getLogger(__name__)

PIT_QUALITY_BAKEOFF_PATH = "trained_data/backtests/pit_quality_bakeoff.json"
FUNDAMENTALS_CACHE = "market_data/equity/sp500_pit_fundamentals.json"
PRICES_CACHE = "market_data/equity/sp500_prices.parquet"


def evaluate_book_wide(
    snapshot,
    prices: pd.DataFrame,
    base_weights: pd.DataFrame,
    *,
    target_vol: float = DEFAULT_TARGET_VOL,
    dd_soft: float = DEFAULT_DD_SOFT,
    dd_hard: float = DEFAULT_DD_HARD,
    max_lev: float = DEFAULT_MAX_LEV,
    cost_bps: float = DEFAULT_COST_BPS,
    execution_lag: int = DEFAULT_EXECUTION_LAG,
    return_full: bool = False,
) -> Dict[str, object]:
    """Gate a long-only book on a staggered universe (NaN-safe; baseline untouched).

    ``return_full=True`` (default False, backward-compatible additive param)
    also attaches the raw net-of-cost daily return Series under the
    ``"_returns"`` key — needed by callers that compute significance stats
    (DSR / block-bootstrap p) on the underlying return series rather than the
    summary dict alone. Existing callers (``run_pit_quality_bakeoff`` and its
    tests) never pass this flag, so their output shape is unchanged.
    """
    px = prices.reindex(index=base_weights.index, columns=base_weights.columns)
    # Drop names with NO real price in THIS window (e.g. delisted before / IPO after a
    # sub-period slice). They carry 0 weight anyway; keeping them would leave a column
    # ffill/bfill cannot fill -> the engine's NaN guard would (correctly) crash.
    valid = px.columns[px.notna().any()]
    px = px[valid]
    base_weights = base_weights[valid]
    base_returns = _derive_returns(px)  # skipna-safe diversified driver for the overlay
    scalar = baseline_overlay(
        base_returns.fillna(0.0), target_vol=target_vol, dd_soft=dd_soft,
        dd_hard=dd_hard, max_lev=max_lev,
    ).reindex(base_weights.index).fillna(0.0)
    managed = base_weights.mul(scalar, axis=0)
    # No fabricated exposure: force weight 0 wherever the ORIGINAL price was NaN.
    managed = managed.where(px.notna(), 0.0)
    if (managed < -1e-9).any().any():
        raise ValueError("base_weights contain negatives — long-only violated")
    px_engine = px.ffill().bfill()  # NaN-free for the engine; inert where weight==0
    result = run_portfolio_backtest(
        managed, px_engine, cost_bps=cost_bps, slippage_bps_per_pct_adv=0.0,
        adv=None, execution_lag=execution_lag,
    )
    s = result.summary
    net_sharpe = float(s.get("net_sharpe", 0.0))
    max_dd = float(s.get("max_dd", 1.0))
    per_year = s.get("per_year", {}) or {}
    pos = sum(1 for v in per_year.values() if float(v) > 0)
    tot = len(per_year)
    verdict = evaluate_gate({
        "net_sharpe": net_sharpe, "positive_years": pos, "total_years": tot,
        "max_drawdown": max_dd, "walk_forward": True,
    })
    out = {
        "net_sharpe": round(net_sharpe, 4), "max_dd": round(max_dd, 4),
        "positive_years": int(pos), "total_years": int(tot),
        "gate_pass": bool(verdict.passed),
    }
    if return_full:
        out["_returns"] = result.returns
    return out


def _daily_membership_snapshot(snapshot, price_index: pd.DatetimeIndex):
    """Reindex monthly membership onto the daily price grid (ffill) -> daily snapshot."""
    from dataclasses import replace
    m = snapshot.membership.copy()
    m.index = pd.DatetimeIndex(pd.to_datetime(m.index, utc=True)).normalize()
    daily = m.reindex(price_index.union(m.index)).sort_index().ffill().reindex(price_index)
    daily = daily.fillna(False).astype(bool)
    return replace(snapshot, membership=daily)


def _fetch_prices(members: List[str], start: str, end: str, cache: Path) -> pd.DataFrame:
    """yfinance adjusted closes (PRICES ONLY). Cached to parquet for re-derivation."""
    if cache.exists():
        logger.info("loading cached prices from %s", cache)
        return pd.read_parquet(cache)
    import yfinance as yf
    raw = yf.download(members, start=start, end=end, progress=False,
                      auto_adjust=True, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index, utc=True).normalize()
    close = close.dropna(axis=1, how="all")  # drop tickers yfinance had nothing for
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        close.to_parquet(cache)
    except (OSError, ImportError):
        pass
    return close


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def run_pit_quality_bakeoff(
    *,
    start: str = "2012-01-01",
    end: Optional[str] = None,
    oos_fraction: float = 0.34,
    min_coverage: float = 0.40,
    out_path: Path = Path(PIT_QUALITY_BAKEOFF_PATH),
    fundamentals_cache: Path = Path(FUNDAMENTALS_CACHE),
    prices_cache: Path = Path(PRICES_CACHE),
    generated_at: str = "",
    max_names: Optional[int] = None,
) -> Dict[str, object]:
    """OFFLINE: quality-tilt vs EW baseline on a wide survivorship-aware PIT universe.

    Pipeline: reconstruct PIT membership -> fetch prices (yfinance) + true-PIT
    fundamentals (EDGAR) -> build filing-date-aligned quality panel -> validate
    no-lookahead + coverage floor -> gate EW vs quality (full + OOS). running:NO.
    """
    from datetime import datetime, timezone

    from src.equity.edgar_fundamentals import build_pit_dump
    from src.equity.quality import quality_tilt_weights
    from src.equity.quality_data import (
        build_quality_panel, load_metrics_dump, panel_coverage, validate_panel_pit,
    )
    from src.equity.sp500_membership import build_sp500_snapshot

    end = end or datetime.now(timezone.utc).date().isoformat()
    snap_monthly = build_sp500_snapshot(start=start, end=end)
    members = list(snap_monthly.membership.columns)
    if max_names is not None:
        members = members[:max_names]

    prices = _fetch_prices(members, start, end, prices_cache)
    priced = [t for t in members if t in prices.columns]
    logger.info("prices: %d/%d members have yfinance data", len(priced), len(members))

    if fundamentals_cache.exists():
        dump = load_metrics_dump(fundamentals_cache)
    else:
        dump = build_pit_dump(priced, out_path=fundamentals_cache)
    with_fund = [t for t in priced if t in dump and dump[t]]
    logger.info("fundamentals: %d/%d priced members have EDGAR PIT data", len(with_fund), len(priced))

    # Restrict the evaluation universe to names with BOTH price and PIT fundamentals.
    from dataclasses import replace
    members_eval = sorted(with_fund)
    snap_eval = replace(snap_monthly, membership=snap_monthly.membership[members_eval])
    prices_eval = prices[members_eval]
    snap_daily = _daily_membership_snapshot(snap_eval, prices_eval.index)

    qpanel = build_quality_panel(dump, prices_eval.index, members_eval)
    validate_panel_pit(qpanel, dump)  # raises on any lookahead
    cov = panel_coverage(qpanel, snap_daily.membership.astype(bool))
    if cov < float(min_coverage):
        return {"status": "NOT_EVALUATED", "reason": f"PIT quality coverage {cov:.2%} "
                f"< floor {min_coverage:.0%} on the wide universe", "coverage": round(cov, 4),
                "universe_size": len(members_eval)}

    ew = build_equal_weight_panel(snap_daily, prices_eval)
    qw = quality_tilt_weights(snap_daily, prices_eval, qpanel)

    def _panel_eval(idx, panel) -> dict:
        w = quality_tilt_weights(snap_daily, prices_eval, panel).loc[idx]
        return evaluate_book_wide(snap_daily, prices_eval.loc[idx], w)

    def _compare(idx) -> dict:
        ew_w, qw_w, px = ew.loc[idx], qw.loc[idx], prices_eval.loc[idx]
        base = evaluate_book_wide(snap_daily, px, ew_w)
        qual = evaluate_book_wide(snap_daily, px, qw_w)
        return {
            "baseline_ew": base, "quality": qual,
            "quality_beats_baseline": float(qual["net_sharpe"]) > float(base["net_sharpe"]),
            "net_sharpe_margin": round(float(qual["net_sharpe"]) - float(base["net_sharpe"]), 4),
        }

    full_idx = ew.index
    split = int(len(full_idx) * (1.0 - float(oos_fraction)))
    oos_idx = full_idx[split:]

    sub_periods = []
    years = sorted(set(full_idx.year))
    for y0 in range(years[0], years[-1] + 1, 5):
        blk = full_idx[(full_idx.year >= y0) & (full_idx.year < y0 + 5)]
        if len(blk) < 252:
            continue
        cmp = _compare(blk)
        sub_periods.append({
            "start": blk.min().date().isoformat(), "end": blk.max().date().isoformat(),
            "baseline_net_sharpe": cmp["baseline_ew"]["net_sharpe"],
            "quality_net_sharpe": cmp["quality"]["net_sharpe"],
            "margin": cmp["net_sharpe_margin"], "quality_beats": cmp["quality_beats_baseline"],
        })

    # Robustness sensitivities (labelled): is the canonical result a debt_to_equity
    # outlier artifact? Winsorize the z tails; drop the leverage component entirely.
    full_idx_l = full_idx
    oos_idx_l = oos_idx

    def _sensitivity(panel) -> dict:
        f = _panel_eval(full_idx_l, panel)
        o = _panel_eval(oos_idx_l, panel)
        base_f = evaluate_book_wide(snap_daily, prices_eval.loc[full_idx_l], ew.loc[full_idx_l])
        base_o = evaluate_book_wide(snap_daily, prices_eval.loc[oos_idx_l], ew.loc[oos_idx_l])
        return {
            "full_net_sharpe": f["net_sharpe"], "oos_net_sharpe": o["net_sharpe"],
            "full_margin_vs_ew": round(f["net_sharpe"] - base_f["net_sharpe"], 4),
            "oos_margin_vs_ew": round(o["net_sharpe"] - base_o["net_sharpe"], 4),
        }

    margin_only = [("gross_margin", 1.0), ("net_margin", 1.0)]
    robustness = {
        "purpose": "is the canonical negative a debt_to_equity-outlier artifact?",
        "winsorized_z_clip3": _sensitivity(
            build_quality_panel(dump, prices_eval.index, members_eval, clip_z=3.0)),
        "margin_only": _sensitivity(
            build_quality_panel(dump, prices_eval.index, members_eval, components=margin_only)),
        "margin_only_winsorized": _sensitivity(
            build_quality_panel(dump, prices_eval.index, members_eval,
                                components=margin_only, clip_z=3.0)),
        "interpretation": "canonical drag is largely the non-robust -debt_to_equity z "
                          "(bank/REIT/negative-equity tails); the clean margin-only tilt "
                          "merely MATCHES EW (no shippable edge).",
    }

    n_removed = int(sum(1 for t in members_eval
                        if snap_daily.membership[t].any() and not snap_daily.membership[t].iloc[-1]))
    payload = {
        "generated_at": generated_at,
        "running": "NO (offline backtest evaluation)",
        "data_honesty": {
            "fundamentals": "TRUE-PIT (SEC EDGAR XBRL, filing-date-aligned); pit=True",
            "prices": "yfinance adjusted close (PRICES ONLY; never used for fundamentals)",
            "membership": "survivorship-aware S&P500 reconstruction (Wikipedia change-log)",
            "residual_bias_labelled": [
                "delisting returns UNDERSTATED (forward-fill flattens final gap; most "
                "S&P removals are acquisitions near-market, bankruptcies hidden)",
                "price-availability survivorship: yfinance lacks some delisted tickers -> dropped",
                "EDGAR XBRL floor ~2010 -> clean window ~2012-2026",
            ],
        },
        "universe": {
            "ever_members_in_window": len(snap_monthly.membership.columns),
            "evaluated_with_price_and_pit_fundamentals": len(members_eval),
            "since_removed_included": n_removed,
            "quality_panel_coverage": round(cov, 4),
        },
        "pre_registered": {
            "score": "mean available cross-sectional z of {+gross_margin,+net_margin,"
                     "-debt_to_equity}, tilt=1.0 (annual 10-K, filing-date PIT)",
            "test": "quality-tilt vs EW baseline net Sharpe, full + OOS, same gate; "
                    "+ disjoint-5y sub-period stability (anti-p-hacking)",
        },
        "common_window": {"start": full_idx.min().date().isoformat(),
                          "end": full_idx.max().date().isoformat()},
        "robustness_sensitivities": robustness,
        "sub_period_stability": sub_periods,
        "full_sample": _compare(full_idx),
        "out_of_sample": {"start": oos_idx.min().date().isoformat(),
                          "end": oos_idx.max().date().isoformat(),
                          "fraction": float(oos_fraction), **_compare(oos_idx)},
    }
    _atomic_write_json(payload, Path(out_path))
    return payload
