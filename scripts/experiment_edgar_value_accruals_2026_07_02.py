#!/usr/bin/env python3
"""VALUE + ACCRUALS factor bake-off on true-PIT SEC EDGAR fundamentals (offline).

Pre-registered: docs/experiment-sec-edgar-value-accruals-2026-07-02.md (frozen
BEFORE any Sharpe/DSR number is computed; no post-hoc window/universe/lag
changes after seeing results). This is trial 2 (value) and trial 3 (accruals)
of the "SEC-EDGAR-true-PIT equity fundamental factor" family — trial 1 was the
already-run, already-negative quality composite
(trained_data/backtests/pit_quality_bakeoff.json, DO NOT re-run or overwrite).

Pipeline (mirrors ``pit_quality_eval.run_pit_quality_bakeoff`` structure):
  1. Build the EXTENDED EDGAR fundamentals dump (adds CashFlowFromOps,
     TotalAssets, SharesOutstanding, accruals, equity, net_income raw exports)
     to a NEW cache path -- never touches the original quality-only dump.
  2. Reuse the cached yfinance prices parquet (no re-fetch).
  3. Build the survivorship-aware S&P 500 universe (2012-01-01 -> present,
     same window as the quality bake-off).
  4. Build EW baseline (recomputed fresh for this run's coverage), value-tilt
     weights (``value_data.build_value_panel`` + ``quality.quality_tilt_weights``
     reused generically), accruals-tilt weights (``quality_data.build_quality_panel``
     with ``components=[("accruals", -1.0)]``, reused UNMODIFIED).
  5. Gate EW/value/accruals full-sample + OOS (trailing 34%) via
     ``pit_quality_eval.evaluate_book_wide`` (extended with ``return_full=True``
     to get the raw net-return series for significance testing).
  6. DSR + block-bootstrap p per factor's OOS net-return series, N_TRIALS=3
     (Bonferroni alpha = 0.05/3 ~= 0.0167), reusing (copied verbatim, with
     attribution) the ``psr``/``deflated_sr``/``block_bootstrap_p`` helpers from
     ``scripts/experiment_crypto_xs_signals.py`` -- not reimplemented.
  7. One labelled robustness sensitivity per factor (value: earnings-yield-only;
     accruals: winsorized z-clip=3), reported regardless of sign.
  8. Atomic write to trained_data/backtests/edgar_value_accruals_bakeoff.json.

OFFLINE only -- network fetch (EDGAR + cached yfinance) + in-memory backtest +
JSON write. Nothing here trades, unhalts, or touches src/scanner/execution.py,
src/brokers/, or any order-submission path. running:NO.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.equity.edgar_fundamentals import build_pit_dump  # noqa: E402
from src.equity.pit_quality_eval import (  # noqa: E402
    _daily_membership_snapshot,
    _fetch_prices,
    evaluate_book_wide,
)
from src.equity.quality import quality_tilt_weights  # noqa: E402
from src.equity.quality_data import (  # noqa: E402
    build_quality_panel,
    load_metrics_dump,
    validate_panel_pit,
)
from src.equity.quality_data import panel_coverage as quality_panel_coverage  # noqa: E402
from src.equity.sp500_membership import build_sp500_snapshot  # noqa: E402
from src.equity.ship_gate import build_equal_weight_panel  # noqa: E402
from src.equity.value_data import (  # noqa: E402
    build_value_panel,
    panel_coverage as value_panel_coverage,
    validate_value_panel_pit,
)

logger = logging.getLogger(__name__)

OUT_PATH = Path("trained_data/backtests/edgar_value_accruals_bakeoff.json")
EXT_FUNDAMENTALS_CACHE = Path("market_data/equity/sp500_pit_fundamentals_ext.json")
PRICES_CACHE = Path("market_data/equity/sp500_prices.parquet")
N_TRIALS = 3          # family count: quality (trial 1, already run) + value (2) + accruals (3)
BONFERRONI_ALPHA = 0.05 / N_TRIALS
EULER = 0.5772156649
ANN = 252
# value/accruals need more inputs than quality -> lower floor is honest, not tuned;
# reported either way, never silently raised to pass.
MIN_COVERAGE = 0.20

ACCRUALS_COMPONENTS = [("accruals", -1.0)]   # low accruals -> higher predicted return (Sloan 1996)


# --------------------------------------------------------------------------- #
# Significance-testing helpers -- COPIED VERBATIM (with attribution) from
# scripts/experiment_crypto_xs_signals.py per the pre-registration's explicit
# reuse instruction. Not reimplemented; do not fork the math independently.
# --------------------------------------------------------------------------- #
def psr(x, bench=0.0):
    x = x.dropna()
    t = len(x)
    if t < 30 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    sk = float(stats.skew(x))
    ku = float(stats.kurtosis(x, fisher=False))
    denom = np.sqrt((1 - sk * sr + (ku - 1) / 4 * sr ** 2) / (t - 1))
    return float(stats.norm.cdf((sr - bench) / denom))


def deflated_sr(x, n_trials=N_TRIALS):
    x = x.dropna()
    t = len(x)
    if t < 30:
        return float("nan"), float("nan")
    e_max = ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
             + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr0 = e_max / np.sqrt(t - 1)
    return float(psr(x, sr0)), float(sr0 * np.sqrt(ANN))


def block_bootstrap_p(x, n=5000, block=5, seed=12345):
    x = x.dropna().to_numpy()
    t = len(x)
    if t < 30:
        return float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(t / block))
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, t, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % t
        means[i] = x[idx[:t]].mean()
    return float((means <= 0).mean())


# --------------------------------------------------------------------------- #
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


def _significance(returns: pd.Series) -> Dict[str, object]:
    dsr, sr0_ann = deflated_sr(returns, n_trials=N_TRIALS)
    p = block_bootstrap_p(returns)
    dsr_val = float(dsr) if dsr == dsr else None   # NaN -> None (JSON-safe)
    significant = bool(dsr_val is not None and dsr_val >= 0.95 and p == p and p < BONFERRONI_ALPHA)
    return {
        "n_trials": N_TRIALS,
        "bonferroni_alpha": round(BONFERRONI_ALPHA, 5),
        "deflated_sharpe_ratio": round(dsr_val, 4) if dsr_val is not None else None,
        "benchmark_sr0_annualized": round(float(sr0_ann), 4) if sr0_ann == sr0_ann else None,
        "block_bootstrap_p": round(float(p), 5) if p == p else None,
        "significant_at_bar": significant,
        "bar": "DSR >= 0.95 AND bootstrap_p < bonferroni_alpha",
    }


def run_edgar_value_accruals_bakeoff(
    *,
    start: str = "2012-01-01",
    end: Optional[str] = None,
    oos_fraction: float = 0.34,
    min_coverage: float = MIN_COVERAGE,
    out_path: Path = OUT_PATH,
    fundamentals_cache: Path = EXT_FUNDAMENTALS_CACHE,
    prices_cache: Path = PRICES_CACHE,
    generated_at: str = "",
    max_names: Optional[int] = None,
) -> Dict[str, object]:
    """OFFLINE: value-tilt + accruals-tilt vs EW baseline on the wide PIT universe."""
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

    members_eval = sorted(with_fund)
    snap_eval = replace(snap_monthly, membership=snap_monthly.membership[members_eval])
    prices_eval = prices[members_eval]
    snap_daily = _daily_membership_snapshot(snap_eval, prices_eval.index)

    # --- Panels ---
    vpanel = build_value_panel(dump, prices_eval, prices_eval.index, members_eval)
    validate_value_panel_pit(vpanel, dump)  # raises on any lookahead
    v_cov = value_panel_coverage(vpanel, snap_daily.membership.astype(bool))

    apanel = build_quality_panel(dump, prices_eval.index, members_eval, components=ACCRUALS_COMPONENTS)
    validate_panel_pit(apanel, dump, components=ACCRUALS_COMPONENTS)  # raises on any lookahead
    a_cov = quality_panel_coverage(apanel, snap_daily.membership.astype(bool))

    coverage_gate: Dict[str, object] = {}
    if v_cov < float(min_coverage):
        coverage_gate["value"] = {
            "status": "NOT_EVALUATED",
            "reason": f"value panel coverage {v_cov:.2%} < floor {min_coverage:.0%}",
            "coverage": round(v_cov, 4),
        }
    if a_cov < float(min_coverage):
        coverage_gate["accruals"] = {
            "status": "NOT_EVALUATED",
            "reason": f"accruals panel coverage {a_cov:.2%} < floor {min_coverage:.0%}",
            "coverage": round(a_cov, 4),
        }
    if coverage_gate:
        return {
            "status": "PARTIAL_NOT_EVALUATED" if len(coverage_gate) < 2 else "NOT_EVALUATED",
            "coverage_gate": coverage_gate,
            "universe_size": len(members_eval),
        }

    ew = build_equal_weight_panel(snap_daily, prices_eval)
    vw = quality_tilt_weights(snap_daily, prices_eval, vpanel)
    aw = quality_tilt_weights(snap_daily, prices_eval, apanel)

    def _eval(idx, weights) -> dict:
        res = evaluate_book_wide(snap_daily, prices_eval.loc[idx], weights.loc[idx], return_full=True)
        returns = res.pop("_returns")
        return res, returns

    def _compare(idx) -> Dict[str, object]:
        base, base_ret = _eval(idx, ew)
        val, val_ret = _eval(idx, vw)
        acc, acc_ret = _eval(idx, aw)
        return {
            "baseline_ew": base,
            "value": val,
            "accruals": acc,
            "value_beats_baseline": float(val["net_sharpe"]) > float(base["net_sharpe"]),
            "accruals_beats_baseline": float(acc["net_sharpe"]) > float(base["net_sharpe"]),
            "value_net_sharpe_margin": round(float(val["net_sharpe"]) - float(base["net_sharpe"]), 4),
            "accruals_net_sharpe_margin": round(float(acc["net_sharpe"]) - float(base["net_sharpe"]), 4),
            "_returns": {"baseline_ew": base_ret, "value": val_ret, "accruals": acc_ret},
        }

    full_idx = ew.index
    split = int(len(full_idx) * (1.0 - float(oos_fraction)))
    oos_idx = full_idx[split:]

    full_cmp = _compare(full_idx)
    oos_cmp = _compare(oos_idx)

    value_sig = _significance(oos_cmp["_returns"]["value"])
    accruals_sig = _significance(oos_cmp["_returns"]["accruals"])

    def _strip(cmp: dict) -> dict:
        d = dict(cmp)
        d.pop("_returns", None)
        return d

    # --- Robustness sensitivities (one per factor, reported regardless of sign) ---
    ey_only_panel = build_value_panel(dump, prices_eval, prices_eval.index, members_eval,
                                      components=("earnings_yield",))
    accr_wz3_panel = build_quality_panel(dump, prices_eval.index, members_eval,
                                         components=ACCRUALS_COMPONENTS, clip_z=3.0)

    def _sensitivity_book(panel, idx) -> dict:
        w = quality_tilt_weights(snap_daily, prices_eval, panel).loc[idx]
        base = evaluate_book_wide(snap_daily, prices_eval.loc[idx], ew.loc[idx])
        arm = evaluate_book_wide(snap_daily, prices_eval.loc[idx], w)
        return {"net_sharpe": arm["net_sharpe"], "margin_vs_ew": round(arm["net_sharpe"] - base["net_sharpe"], 4)}

    robustness = {
        "value_earnings_yield_only": {
            "purpose": "is book-to-market dragged by bank/REIT-style outliers, mirroring the "
                       "quality factor's debt_to_equity artifact?",
            "full_sample": _sensitivity_book(ey_only_panel, full_idx),
            "out_of_sample": _sensitivity_book(ey_only_panel, oos_idx),
        },
        "accruals_winsorized_z_clip3": {
            "purpose": "is the accruals result driven by extreme working-capital-shock outliers?",
            "full_sample": _sensitivity_book(accr_wz3_panel, full_idx),
            "out_of_sample": _sensitivity_book(accr_wz3_panel, oos_idx),
        },
    }

    n_removed = int(sum(1 for t in members_eval
                        if snap_daily.membership[t].any() and not snap_daily.membership[t].iloc[-1]))
    payload = {
        "generated_at": generated_at,
        "running": "NO (offline backtest evaluation)",
        "data_honesty": {
            "fundamentals": "TRUE-PIT (SEC EDGAR XBRL, filing-date-aligned); pit=True. Extended "
                            "dump adds CashFlowFromOps/Assets/SharesOutstanding (dei cover-page) "
                            "-> accruals ratio + raw equity/net_income/shares_outstanding exports.",
            "prices": "yfinance adjusted close (PRICES ONLY; reused cached parquet, not re-fetched)",
            "membership": "survivorship-aware S&P500 reconstruction (Wikipedia change-log)",
            "shares_outstanding_dei_finding": (
                "dei:EntityCommonStockSharesOutstanding is a 10-K cover-page fact whose 'end' "
                "date is NOT the fiscal year-end (verified empirically against live EDGAR for "
                "AAPL + MSFT, 2026-07-02) -- e.g. AAPL FY2009: Assets.end=2009-09-26 but "
                "dei.end=2009-10-16 (same accession, filed=2009-10-27). Fixed by keying "
                "shares-outstanding on its OWN filed date "
                "(src.equity.edgar_fundamentals._shares_outstanding_by_filing) and matching to "
                "each period's own filed date via the co-filed us-gaap concepts, never by 'end'."
            ),
            "residual_bias_labelled": [
                "delisting returns UNDERSTATED (forward-fill flattens final gap)",
                "price-availability survivorship: yfinance lacks some delisted tickers -> dropped",
                "EDGAR XBRL floor ~2010 -> clean window ~2012-2026",
                "value/accruals need MORE simultaneous inputs than quality (3-4 vs 2-3 concepts) "
                "-> lower realized coverage than the quality bake-off, reported honestly below",
                "FUNDAMENTALS-SIDE survivorship gap (independent verifier finding, 2026-07-02): "
                "load_cik_map() sources SEC's CURRENT-tickers-only registry "
                "(company_tickers.json) -- a delisted/acquired/renamed constituent (confirmed: "
                "TWX) loses CIK resolvability entirely and silently drops out of the "
                "fundamentals-covered universe even though it remains correctly present in the "
                "price panel. since_removed_included below counts index-membership removals, "
                "NOT delistings covered by fundamentals -- treat as a joint-panel survivorship "
                "gap in addition to the price-side one above, biasing toward SEC-current names.",
                "STALENESS GUARD (independent verifier finding, 2026-07-02, fixed same day): "
                "shares_outstanding forward-fill is now capped at "
                "value_data._MAX_STALENESS_DAYS=730 days past its own last real filing -- an "
                "earlier unbounded-ffill version let >=12 large-cap names (incl. Berkshire "
                "Hathaway, dei tag dormant since 2011) carry a years-stale share count against "
                "a live price, inflating book_to_market by ~1500x for those names. This run "
                "reflects the fix; NaN replaces the stale value beyond the cap rather than "
                "fabricating a market cap.",
            ],
        },
        "universe": {
            "ever_members_in_window": len(snap_monthly.membership.columns),
            "evaluated_with_price_and_pit_fundamentals": len(members_eval),
            "since_removed_included": n_removed,
            "value_panel_coverage": round(v_cov, 4),
            "accruals_panel_coverage": round(a_cov, 4),
        },
        "pre_registered": {
            "value": "value_score = mean available cross-sectional z of {+book_to_market, "
                     "+earnings_yield}; book_to_market=equity_PIT/market_cap, "
                     "earnings_yield=net_income_PIT/market_cap, market_cap=shares_PIT*price(t), "
                     "tilt=1.0",
            "accruals": (
                "accruals_score = z of {-accruals}; accruals=(NetIncomeLoss_PIT-"
                "CashFlowFromOps_PIT)/TotalAssets_PIT, all annual filing-date PIT, tilt=1.0"
            ),
            "test": "value-tilt / accruals-tilt vs EW baseline net Sharpe, full + OOS, same "
                    "canonical ship gate as the quality bake-off",
            "multiple_testing": (
                "N_TRIALS=3 (quality=trial1 already run+negative, value=trial2, accruals=trial3); "
                "Bonferroni alpha=0.05/3~=0.0167; DSR>=0.95 AND bootstrap_p<alpha required for "
                "'statistically significant'"
            ),
        },
        "common_window": {
            "start": full_idx.min().date().isoformat(),
            "end": full_idx.max().date().isoformat(),
        },
        "full_sample": _strip(full_cmp),
        "out_of_sample": {
            "start": oos_idx.min().date().isoformat(),
            "end": oos_idx.max().date().isoformat(),
            "fraction": float(oos_fraction), **_strip(oos_cmp),
        },
        "significance": {
            "value_oos": value_sig,
            "accruals_oos": accruals_sig,
        },
        "robustness_sensitivities": robustness,
    }
    _atomic_write_json(payload, Path(out_path))
    return payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_edgar_value_accruals_bakeoff(
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    keys = ("status", "full_sample", "out_of_sample", "significance", "universe", "coverage_gate")
    summary = {k: v for k, v in result.items() if k in keys}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
