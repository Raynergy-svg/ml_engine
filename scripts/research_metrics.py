#!/usr/bin/env python
"""Consolidated research metrics — ONE schema for EVERY strategy (readiness step 5).

Emits the same metric schema per registered strategy — returns, turnover,
costs, Sharpe, drawdown, DSR, bootstrap p, beta, residual fraction, benchmark
comparison — computed from each strategy's OWN committed evidence (forward
ledgers + hedge twin-lane rows). Every metric that cannot be honestly
computed is null WITH a reason; nothing is imputed, and forward-ledger
metrics are never blended with backtest history.

Sources per strategy:
  * crypto_momentum / crypto_ts_trend / multi_asset_trend / track_b — their
    strategy-owned shadow ledgers (``today_net_return`` per forward cycle).
  * equity_harvester / fx_trend / oanda_fx — the hedge twin-lane ledger's
    resolved raw marks (their own ledgers carry no return series).
  * residual_fraction (phi) — trained_data/hedge/residual_attribution_report.json.
  * beta / benchmark — NOT IMPLEMENTED yet: no pipeline computes them and no
    benchmark series is designated per lane, so both are null with an
    accurate reason (2026-07-19 audit: the earlier wording overclaimed).

Mathematical safety rules (2026-07-19 audit):
  * DSR only against a REGISTERED campaign trial count (n_trials=1
    degenerates the benchmark to -inf — never computed).
  * No annualization below MIN_OBS_FOR_ANNUALIZATION observations, and the
    factor is DERIVED from the observed cadence, never hardcoded.
  * Drawdown measures from initial capital (peak starts at 1.0).
  * One asof_date = one observation (deduped on both ledger kinds).

Usage:
    python scripts/research_metrics.py                # print + persist report
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_crypto_xs_signals as _sig  # noqa: E402  frozen DSR/bootstrap fns

from src.hedge.hedged_shadow_lane import (  # noqa: E402
    CRYPTO_MOMENTUM_LEDGER_PATH,
    CRYPTO_TS_TREND_LEDGER_PATH,
    MULTI_ASSET_TREND_LEDGER_PATH,
    RAW_VS_HEDGED_LEDGER_PATH,
    STRATEGIES,
    TRACK_B_LEDGER_PATH,
    _read_jsonl_rows,
)

RESIDUAL_REPORT_PATH = REPO_ROOT / "trained_data" / "hedge" / "residual_attribution_report.json"
OUT_PATH = REPO_ROOT / "trained_data" / "research" / "strategy_metrics_report.json"

# Registered multiple-testing trial counts (2026-07-19 audit fix): DSR is
# ONLY computed against a pre-registered campaign trial count. n_trials=1
# degenerates the benchmark to norm.ppf(0) = -inf, making any finite result
# look maximally significant — DSR is null WITH a reason for any strategy
# without a registered count, never silently computed.
REGISTERED_TRIAL_COUNTS = {
    "crypto_momentum": 15,     # Round-2 frozen N_TRIALS (crypto campaign)
    "crypto_ts_trend": 15,     # Round-2 frozen N_TRIALS (H5 pre-registered under it)
    "multi_asset_trend": 20,   # Round-3/4 frozen N_TRIALS (edge-hunt campaign)
}

# 2026-07-19 audit fix: never annualize a handful of observations (one
# +1.024% bar is NOT a 373% annual return). Below this floor, ann_return /
# ann_vol / Sharpe stay null with a reason; cumulative + drawdown (which
# need no extrapolation) are still reported.
MIN_OBS_FOR_ANNUALIZATION = 8

_OWN_LEDGERS = {
    "crypto_momentum": CRYPTO_MOMENTUM_LEDGER_PATH,
    "crypto_ts_trend": CRYPTO_TS_TREND_LEDGER_PATH,
    "multi_asset_trend": MULTI_ASSET_TREND_LEDGER_PATH,
    "track_b": TRACK_B_LEDGER_PATH,
}


def _own_ledger_series(path: Path) -> Dict[str, Any]:
    from src.evidence.forward_ledger import cadence_counts, forward_rows, read_rows
    all_rows = read_rows(path)
    rows = forward_rows(all_rows)   # activation baselines are NOT observations
    rets, turns, costs, dates = [], [], [], []
    seen_asof = set()
    for r in rows:
        # defensive dedup (audit fix): one asof_date = one observation, even
        # if a duplicate ever reached a lane ledger.
        d = str(r.get("asof_date"))
        if d in seen_asof:
            continue
        seen_asof.add(d)
        rets.append(float(r["today_net_return"]))
        dates.append(d)
        t = r.get("today_turnover")
        c = r.get("today_cost")
        turns.append(float(t) if isinstance(t, (int, float)) else None)
        costs.append(float(c) if isinstance(c, (int, float)) else None)
    try:
        source = str(path.relative_to(REPO_ROOT))
    except ValueError:
        source = str(path)   # e.g. tmp_path ledgers in tests
    return {"returns": rets, "dates": dates, "turnover": turns, "costs": costs,
            "source": source, "cadence": cadence_counts(all_rows)}


def _hedge_lane_series(strategy: str, ledger_path: Path) -> Dict[str, Any]:
    from src.hedge.hedged_shadow_lane import dedupe_ledger_rows
    rows = [r for r in dedupe_ledger_rows(_read_jsonl_rows(ledger_path))
            if r.get("strategy") == strategy]
    rets, dates = [], []
    for r in sorted(rows, key=lambda x: x.get("asof_date", "")):
        v = (r.get("raw") or {}).get("net_return")
        if isinstance(v, (int, float)):
            rets.append(float(v))
            dates.append(str(r.get("asof_date")))
    weeks = {d[:4] + "-" + d[5:7] + "w" for d in dates}  # coarse month bucket fallback
    return {"returns": rets, "dates": dates, "turnover": [], "costs": [],
            "source": f"{ledger_path.name}:strategy={strategy} (resolved raw marks, deduped)",
            "cadence": {"n_return_bars": len(rets), "n_calendar_weeks": None,
                        "n_calendar_months": len(weeks),
                        "n_completed_rebalances": None,
                        "n_independent_holding_periods": None,
                        "note": "hedge-lane marks; rebalance accounting lives in the strategy's own ledger"}}


def _derive_ann_factor(dates: List[str]) -> Optional[float]:
    """Annualization factor DERIVED from the observed cadence (2026-07-19
    audit fix: never hardcoded — hedge rows are one-bar daily marks even for
    weekly-rebalanced strategies). Median calendar-day gap between
    observations -> periods/year. None below 2 dated observations."""
    ds = []
    for d in dates:
        try:
            ds.append(pd.Timestamp(d))
        except (ValueError, TypeError):
            continue
    if len(ds) < 2:
        return None
    gaps = sorted((b - a).days for a, b in zip(ds, ds[1:]) if (b - a).days > 0)
    if not gaps:
        return None
    median_gap = gaps[len(gaps) // 2]
    return 365.25 / float(median_gap)


def _metrics(returns: List[float], dates: List[str],
             n_trials: Optional[int]) -> Dict[str, Any]:
    n = len(returns)
    out: Dict[str, Any] = {"n_observations": n}
    if n == 0:
        out.update({"cumulative_return": 0.0, "ann_return": None, "ann_vol": None,
                    "sharpe": None, "max_drawdown": None, "ann_factor_derived": None,
                    "unavailable_reason": "no forward observations recorded yet"})
    else:
        s = pd.Series(returns, dtype=float)
        out["cumulative_return"] = round(float((1.0 + s).prod() - 1.0), 6)
        # drawdown from INITIAL CAPITAL: the peak starts at 1.0, so a first
        # observation of -10% is a 10% drawdown, never zero (audit fix).
        eq = pd.concat([pd.Series([1.0]), (1.0 + s).cumprod()], ignore_index=True)
        out["max_drawdown"] = round(float(abs((eq / eq.cummax() - 1.0).min())), 5)
        ann = _derive_ann_factor(dates)
        out["ann_factor_derived"] = None if ann is None else round(ann, 2)
        if n >= MIN_OBS_FOR_ANNUALIZATION and ann is not None and s.std(ddof=1) > 0:
            out["ann_return"] = round(float(s.mean() * ann), 5)
            out["ann_vol"] = round(float(s.std(ddof=1) * np.sqrt(ann)), 5)
            out["sharpe"] = round(float(np.sqrt(ann) * s.mean() / s.std(ddof=1)), 4)
        else:
            out["ann_return"] = None
            out["ann_vol"] = None
            out["sharpe"] = None
            out["unavailable_reason"] = (
                f"n={n} < {MIN_OBS_FOR_ANNUALIZATION} observations (or cadence "
                "underivable) — annualizing would be mechanical extrapolation, "
                "not a forward estimate")
    # significance: frozen harness fns, ONLY against a registered trial count
    if n >= 30 and n_trials is not None and n_trials > 1:
        s = pd.Series(returns, dtype=float)
        dsr, _ = _sig.deflated_sr(s, n_trials=int(n_trials))
        out["deflated_sr_prob"] = None if np.isnan(dsr) else round(float(dsr), 4)
        out["dsr_n_trials"] = int(n_trials)
        out["block_bootstrap_p"] = round(float(_sig.block_bootstrap_p(s)), 5)
    else:
        out["deflated_sr_prob"] = None
        out["dsr_n_trials"] = n_trials
        out["block_bootstrap_p"] = (
            round(float(_sig.block_bootstrap_p(pd.Series(returns, dtype=float))), 5)
            if n >= 30 else None)
        out["significance_reason"] = (
            f"n={n} < 30 — significance undefined (warm-up, not proof)" if n < 30 else
            "no registered multiple-testing trial count — DSR undefined "
            "(n_trials=1 degenerates the benchmark to -inf; never computed)")
    return out


def _residual_fraction(strategy: str) -> Dict[str, Any]:
    try:
        report = json.loads(RESIDUAL_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"phi": None, "reason": "residual attribution report missing/unreadable"}
    entry = (report.get("strategies") or {}).get(strategy)
    if not entry:
        return {"phi": None, "reason": "strategy not in residual attribution report"}
    att = entry.get("attribution") or entry
    phi = att.get("residual_fraction")
    return {"phi": phi,
            "n_aligned_cycles": att.get("n_aligned_cycles"),
            "reason": att.get("residual_fraction_reason")}


def build_report() -> Dict[str, Any]:
    strategies: Dict[str, Any] = {}
    for s in STRATEGIES:
        if s in _OWN_LEDGERS:
            series = _own_ledger_series(_OWN_LEDGERS[s])
        else:
            series = _hedge_lane_series(s, RAW_VS_HEDGED_LEDGER_PATH)
        m = _metrics(series["returns"], series["dates"],
                     REGISTERED_TRIAL_COUNTS.get(s))
        turns = [t for t in series["turnover"] if t is not None]
        costs = [c for c in series["costs"] if c is not None]
        strategies[s] = {
            "source": series["source"],
            "first_asof": series["dates"][0] if series["dates"] else None,
            "last_asof": series["dates"][-1] if series["dates"] else None,
            # 2026-07-19 review: evidence requirements count weeks/rebalances,
            # never raw daily bars — 52 daily bars != 52 weekly observations.
            "cadence": series.get("cadence"),
            **m,
            "avg_turnover_per_cycle": round(float(np.mean(turns)), 5) if turns else None,
            "avg_cost_per_cycle": round(float(np.mean(costs)), 6) if costs else None,
            "residual_fraction": _residual_fraction(s),
            # No benchmark series is invented: beta/benchmark stay null until a
            # hedged lane provides them (scorecard) or an operator names one.
            "beta": None,
            "beta_reason": ("NOT IMPLEMENTED by any current pipeline — no benchmark "
                            "series is designated for this lane and none is imputed; "
                            "field populates only when a pipeline actually computes it"),
            "benchmark_comparison": None,
            "benchmark_reason": "no operator-designated benchmark series for this lane yet",
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": ["n_observations", "cadence", "cumulative_return", "ann_return",
                   "ann_vol", "ann_factor_derived", "sharpe", "max_drawdown",
                   "deflated_sr_prob", "dsr_n_trials", "block_bootstrap_p",
                   "avg_turnover_per_cycle", "avg_cost_per_cycle",
                   "residual_fraction", "beta", "benchmark_comparison"],
        "note": ("forward-ledger evidence ONLY — never blended with backtest history; "
                 "nulls carry reasons; significance below 30 obs is undefined by design"),
        "strategies": strategies,
        "runtime_allowed": False, "paper_only": True, "human_review_required": True,
    }


def _atomic_write(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".metrics_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, str(path))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main() -> int:
    report = build_report()
    _atomic_write(report, OUT_PATH)
    for s, m in report["strategies"].items():
        print(f"{s:>20}: n={m['n_observations']:>3}  cum={m['cumulative_return']!s:>10}  "
              f"sharpe={m['sharpe']!s:>8}  dd={m['max_drawdown']!s:>8}  "
              f"phi={m['residual_fraction']['phi']!s}")
    print(f"\nwritten: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
