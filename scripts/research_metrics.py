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

Mathematical safety rules (2026-07-19 audit + re-review):
  * ONE canonical evidence view: active_observations only — legacy
    pre-engine rows EXCLUDED (reported, never counted); same-period
    conflicts fail closed; returns AND cadence derive from the same view.
  * Annualization + significance are gated by the FROZEN acceptance target
    (explicit holding periods / completed rebalances + elapsed span) with
    structured reason codes — raw daily bars never unlock them.
  * Annualization factor = observations per ELAPSED year over validated
    strictly-increasing dates; geometric and arithmetic annualized returns
    are reported under EXPLICIT names.
  * DSR only against a REGISTERED campaign trial count (n_trials=1
    degenerates the benchmark to -inf — never computed).
  * Drawdown measures from initial capital (peak starts at 1.0).

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

# FROZEN acceptance targets (readiness report) gate annualization AND
# significance (2026-07-19 re-review item 3): raw daily bars never satisfy
# weekly/monthly evidence requirements. Until a strategy's explicit
# holding-period/rebalance counts and elapsed span meet its target, ann/
# Sharpe/DSR/bootstrap stay null with a STRUCTURED reason code.
ACCEPTANCE_REQUIREMENTS = {
    "crypto_momentum":   {"min_independent_periods": 52, "min_elapsed_days": 365},
    "crypto_ts_trend":   {"min_independent_periods": 52, "min_elapsed_days": 365},
    "multi_asset_trend": {"min_completed_rebalances": 24, "min_elapsed_days": 365},
    "track_b":           {"min_independent_periods": 24, "min_elapsed_days": 365},
}

_OWN_LEDGERS = {
    "crypto_momentum": CRYPTO_MOMENTUM_LEDGER_PATH,
    "crypto_ts_trend": CRYPTO_TS_TREND_LEDGER_PATH,
    "multi_asset_trend": MULTI_ASSET_TREND_LEDGER_PATH,
    "track_b": TRACK_B_LEDGER_PATH,
}


def _own_ledger_series(path: Path) -> Dict[str, Any]:
    """ONE canonical evidence view (re-review item 2): rows are
    canonicalized ONCE (active_observations — engine forward rows only, one
    active revision per period, legacy pre-engine rows EXCLUDED, conflicts
    fail closed) and returns/dates/turnover/costs AND cadence all derive
    from that same view. Legacy exclusions are reported, never counted."""
    from src.evidence.forward_ledger import (
        active_observations, cadence_counts, legacy_rows, read_rows)
    all_rows = read_rows(path)
    canon = active_observations(all_rows)
    rets, turns, costs, dates = [], [], [], []
    for r in canon:
        rets.append(float(r["today_net_return"]))
        dates.append(str(r.get("asof_date")))
        t = r.get("today_turnover")
        c = r.get("today_cost")
        turns.append(float(t) if isinstance(t, (int, float)) else None)
        costs.append(float(c) if isinstance(c, (int, float)) else None)
    try:
        source = str(path.relative_to(REPO_ROOT))
    except ValueError:
        source = str(path)   # e.g. tmp_path ledgers in tests
    return {"returns": rets, "dates": dates, "turnover": turns, "costs": costs,
            "source": source, "cadence": cadence_counts(all_rows),
            "legacy_rows_excluded": len(legacy_rows(all_rows)),
            "legacy_exclusion_reason": ("pre-engine rows use latest-bar semantics "
                                        "(return already realized when recorded) — "
                                        "NOT forward evidence; excluded from every count")}


def _hedge_lane_series(strategy: str, ledger_path: Path) -> Dict[str, Any]:
    # STRICT + canonical active set (2026-07-19 audit): corruption fails
    # closed; one active observation per evidence period.
    from src.evidence.forward_ledger import read_rows as _strict_read
    from src.hedge.hedged_shadow_lane import active_ledger_rows
    all_rows = active_ledger_rows(_strict_read(ledger_path)) if ledger_path.exists() else []
    rows = [r for r in all_rows if r.get("strategy") == strategy]
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


def _validated_span(dates: List[str]):
    """(elapsed_years, reason) — dates must be parseable and STRICTLY
    increasing (re-review item 4); invalid/non-monotonic -> (None, reason)."""
    ds = []
    for d in dates:
        try:
            ds.append(pd.Timestamp(d))
        except (ValueError, TypeError):
            return None, "invalid_dates"
    for a, b in zip(ds, ds[1:]):
        if b <= a:
            return None, "invalid_dates"
    if len(ds) < 2:
        return None, "insufficient_observations"
    elapsed_years = (ds[-1] - ds[0]).days / 365.25
    if elapsed_years <= 0:
        return None, "invalid_dates"
    return elapsed_years, None


def _derive_ann_factor(dates: List[str]) -> Optional[float]:
    """Observations-per-elapsed-year (re-review item 4 — the prior
    median-gap version mishandled even counts and mapped business-day data
    to ~365/yr): (n-1) / elapsed_years over VALIDATED strictly-increasing
    dates. None with the span's reason otherwise."""
    elapsed_years, _reason = _validated_span(dates)
    if elapsed_years is None:
        return None
    return (len(dates) - 1) / elapsed_years


def evidence_eligibility(strategy: str, cadence: Optional[Dict[str, Any]],
                         dates: List[str]) -> Dict[str, Any]:
    """Per-strategy gate for annualization + significance (re-review item 3):
    explicit holding-period/rebalance counts and elapsed span must meet the
    FROZEN acceptance target. Structured reason codes:
    insufficient_observations / insufficient_elapsed_span /
    insufficient_independent_periods / invalid_dates / cadence_unavailable /
    no_registered_acceptance_target."""
    if not dates:
        return {"eligible": False, "reason_code": "insufficient_observations",
                "detail": "no forward observations"}
    elapsed_years, span_reason = _validated_span(dates)
    if span_reason == "invalid_dates":
        return {"eligible": False, "reason_code": "invalid_dates",
                "detail": "dates unparseable or not strictly increasing"}
    req = ACCEPTANCE_REQUIREMENTS.get(strategy)
    if req is None:
        return {"eligible": False, "reason_code": "no_registered_acceptance_target",
                "detail": f"{strategy} has no frozen acceptance target registered"}
    if (not cadence or cadence.get("n_completed_rebalances") is None
            or cadence.get("n_independent_holding_periods") is None):
        return {"eligible": False, "reason_code": "cadence_unavailable",
                "detail": "no explicit rebalance/holding-period fields — raw bars "
                          "never satisfy weekly/monthly evidence requirements"}
    elapsed_days = (elapsed_years or 0) * 365.25
    if elapsed_days < req["min_elapsed_days"]:
        return {"eligible": False, "reason_code": "insufficient_elapsed_span",
                "detail": f"elapsed {elapsed_days:.0f}d < required "
                          f"{req['min_elapsed_days']}d"}
    if ("min_completed_rebalances" in req
            and cadence["n_completed_rebalances"] < req["min_completed_rebalances"]):
        return {"eligible": False, "reason_code": "insufficient_independent_periods",
                "detail": f"{cadence['n_completed_rebalances']} completed rebalances "
                          f"< required {req['min_completed_rebalances']}"}
    if ("min_independent_periods" in req
            and cadence["n_independent_holding_periods"] < req["min_independent_periods"]):
        return {"eligible": False, "reason_code": "insufficient_independent_periods",
                "detail": f"{cadence['n_independent_holding_periods']} independent "
                          f"holding periods < required {req['min_independent_periods']}"}
    return {"eligible": True, "reason_code": None, "detail": "acceptance target met"}


def _metrics(returns: List[float], dates: List[str],
             n_trials: Optional[int], gate: Dict[str, Any]) -> Dict[str, Any]:
    n = len(returns)
    out: Dict[str, Any] = {"n_observations": n, "metrics_gate": dict(gate)}
    if n == 0:
        out.update({"cumulative_return": 0.0, "ann_return_geometric": None,
                    "ann_return_arithmetic": None, "ann_vol": None,
                    "sharpe": None, "max_drawdown": None, "ann_factor_derived": None})
    else:
        s = pd.Series(returns, dtype=float)
        out["cumulative_return"] = round(float((1.0 + s).prod() - 1.0), 6)
        # drawdown from INITIAL CAPITAL: peak starts at 1.0.
        eq = pd.concat([pd.Series([1.0]), (1.0 + s).cumprod()], ignore_index=True)
        out["max_drawdown"] = round(float(abs((eq / eq.cummax() - 1.0).min())), 5)
        ann = _derive_ann_factor(dates)
        elapsed_years, _span_reason = _validated_span(dates)
        out["ann_factor_derived"] = None if ann is None else round(ann, 2)
        eligible = bool(gate.get("eligible"))
        if eligible and ann is not None and elapsed_years and s.std(ddof=1) > 0:
            cum = float((1.0 + s).prod())
            out["ann_return_geometric"] = round(cum ** (1.0 / elapsed_years) - 1.0, 5)
            out["ann_return_arithmetic"] = round(float(s.mean() * ann), 5)
            out["ann_vol"] = round(float(s.std(ddof=1) * np.sqrt(ann)), 5)
            out["sharpe"] = round(float(np.sqrt(ann) * s.mean() / s.std(ddof=1)), 4)
        else:
            out["ann_return_geometric"] = None
            out["ann_return_arithmetic"] = None
            out["ann_vol"] = None
            out["sharpe"] = None
            if eligible and n >= 2 and s.std(ddof=1) == 0:
                out["metrics_gate"] = {**gate, "eligible": False,
                                       "reason_code": "zero_variance",
                                       "detail": "return variance is exactly zero"}
    # significance: registered trial count AND acceptance eligibility
    eligible = bool(out["metrics_gate"].get("eligible"))
    if eligible and n >= 30 and n_trials is not None and n_trials > 1:
        s = pd.Series(returns, dtype=float)
        dsr, _ = _sig.deflated_sr(s, n_trials=int(n_trials))
        out["deflated_sr_prob"] = None if np.isnan(dsr) else round(float(dsr), 4)
        out["dsr_n_trials"] = int(n_trials)
        out["block_bootstrap_p"] = round(float(_sig.block_bootstrap_p(s)), 5)
    else:
        out["deflated_sr_prob"] = None
        out["dsr_n_trials"] = n_trials
        out["block_bootstrap_p"] = None
        if eligible and n_trials is None:
            out["significance_reason"] = ("no registered multiple-testing trial "
                                          "count — DSR undefined (never computed)")
        elif not eligible:
            out["significance_reason"] = ("gated by acceptance target: "
                                          + str(out["metrics_gate"].get("reason_code")))
        else:
            out["significance_reason"] = f"n={n} < 30"
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
        gate = evidence_eligibility(s, series.get("cadence"), series["dates"])
        m = _metrics(series["returns"], series["dates"],
                     REGISTERED_TRIAL_COUNTS.get(s), gate)
        turns = [t for t in series["turnover"] if t is not None]
        costs = [c for c in series["costs"] if c is not None]
        strategies[s] = {
            "source": series["source"],
            "first_asof": series["dates"][0] if series["dates"] else None,
            "last_asof": series["dates"][-1] if series["dates"] else None,
            # 2026-07-19 review: evidence requirements count weeks/rebalances,
            # never raw daily bars — 52 daily bars != 52 weekly observations.
            "cadence": series.get("cadence"),
            "legacy_rows_excluded": series.get("legacy_rows_excluded", 0),
            "legacy_exclusion_reason": series.get("legacy_exclusion_reason"),
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
        "schema": ["n_observations", "cadence", "legacy_rows_excluded",
                   "metrics_gate", "cumulative_return", "ann_return_geometric",
                   "ann_return_arithmetic", "ann_vol", "ann_factor_derived",
                   "sharpe", "max_drawdown", "deflated_sr_prob", "dsr_n_trials",
                   "block_bootstrap_p", "avg_turnover_per_cycle",
                   "avg_cost_per_cycle", "residual_fraction", "beta",
                   "benchmark_comparison"],
        "note": ("forward ENGINE evidence only — legacy pre-engine rows are "
                 "excluded and reported, never counted; never blended with "
                 "backtest history; nulls carry reasons; annualization and "
                 "significance unlock ONLY at each strategy's frozen "
                 "acceptance target"),
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
