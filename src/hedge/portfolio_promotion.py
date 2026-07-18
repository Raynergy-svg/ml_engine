"""Universal portfolio-level promotion gate (2026-07-18, rev 2).

Standalone validation asks: does the signal work on its own data? This gate
asks the portfolio question: does the strategy still improve the COMBINED
book once exposure overlap, hedge cost, duplication and correlation are
priced in? Every covered strategy gets a verdict against the rest.

rev 2 (operator review of 84d4348) — correctness fixes:
  #1 exposure values in the ledger are DOLLAR risk (risk_home = |w| x
     notional); they are normalized by each row's own notional before any
     cap comparison, and combined with explicit per-strategy allocations.
  #2 scorecard verdicts are imported from hedge_scorecard — no duplicated
     string constants (the previous copy misspelled weak_or_dead and
     referenced a verdict the scorecard could not emit).
  #3 a missing scorecard verdict is UNKNOWN, never a silent pass.
  #4 duplication is truly fail-closed: any corr >= threshold -> FAIL, else
     ANY unresolved incumbent pair (insufficient overlap OR undefined
     correlation) -> UNKNOWN, else PASS.
  #5 bucket crowding uses ONE synchronized evaluation date — the latest
     asof_date at which EVERY covered strategy has a non-fail-closed
     exposure row. A missing incumbent exposure or no common date is
     UNKNOWN, never a partial "portfolio".
  #6 marginal contribution is allocation-weighted with per-date
     renormalization over the union grid (capital is deployed across the
     strategies LIVE that date — renormalized weights, not zero-fill), so
     mixed cadences (FX daily / equity weekly / Track B ~21d) can still
     accrue evidence instead of demanding an exact all-lane intersection.
  #7 the report iterates the FULL covered-strategy set (STRATEGIES from
     hedged_shadow_lane, union ledger keys) — a lane with no rows receives
     an explicit CONTINUE_SHADOW verdict, never silence.

Decision (fail-closed, in order): any FAIL -> REJECT; any UNKNOWN ->
CONTINUE_SHADOW; all PASS -> PROMOTE_TO_OPERATOR_REVIEW. The strongest
verdict this gate emits is a recommendation for operator review — it
promotes nothing itself. Same authority contract as the rest of src/hedge:
zero orders, no gate/halt/leverage mutation, structural isolation.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.hedge.hedge_scorecard import (
    VERDICT_BETA_NOT_ALPHA,
    VERDICT_GENUINE,
    VERDICT_SIGNAL_REAL_BUT_NOISY,
    VERDICT_WEAK_OR_DEAD,
)
from src.hedge.residual_attribution import residual_series, strategy_attribution

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HEDGE_LEDGER_DIR = REPO_ROOT / "trained_data" / "hedge"
RAW_VS_HEDGED_LEDGER_PATH = HEDGE_LEDGER_DIR / "raw_vs_hedged_ledger.jsonl"
SCORECARD_REPORT_PATH = HEDGE_LEDGER_DIR / "hedge_scorecard_report.json"
PROMOTION_REPORT_PATH = HEDGE_LEDGER_DIR / "portfolio_promotion_report.json"
# The operator's capital plan: {"strategy_name": relative_weight, ...}.
# rev 2.1 (operator review of 735d897): the automatic report path previously
# defaulted to EQUAL allocations — allocation-aware code operating under a
# planning assumption that is not grounded in the actual book. Combined-book
# checks now require a real allocation plan (this file, or the ``allocations``
# argument); without one they are UNKNOWN -> CONTINUE_SHADOW, fail-closed.
PORTFOLIO_ALLOCATIONS_PATH = HEDGE_LEDGER_DIR / "portfolio_allocations.json"

RUNTIME_ALLOWED = False
PAPER_ONLY = True
HUMAN_REVIEW_REQUIRED = True

VERDICT_PROMOTE = "PROMOTE_TO_OPERATOR_REVIEW"
VERDICT_SHADOW = "CONTINUE_SHADOW"
VERDICT_REJECT = "REJECT"

# Scorecard verdict partition — imported constants, single source of truth.
# Everything not in either set (inconclusive, insufficient_history,
# no_hedge_available, cost_unmodeled:*-prefixed, None/missing) is UNKNOWN.
_ALPHA_PASS_VERDICTS = {VERDICT_GENUINE, VERDICT_SIGNAL_REAL_BUT_NOISY}
_ALPHA_FAIL_VERDICTS = {VERDICT_BETA_NOT_ALPHA, VERDICT_WEAK_OR_DEAD}


@dataclass(frozen=True)
class GateConfig:
    """Operator-tunable thresholds. Planning constants, not fitted values —
    serialized into every report so a threshold change is auditable."""
    min_aligned_cycles: int = 8          # standalone evidence floor
    min_phi: float = 0.25                # residual fraction below this = beta-dominated
    dup_corr_threshold: float = 0.85     # residual correlation at/above = duplication
    min_overlap_cycles: int = 6          # shared dates before correlation/marginal count
    max_abs_beta: float = 0.50           # combined |net beta| cap, fraction of allocated notional
    max_abs_bucket: float = 0.75         # combined |net| cap per bucket, fraction of allocated notional
    dd_tolerance: float = 1.10           # combined max-DD may grow at most 10% with candidate
    bucket_fields: tuple = (
        "net_currency_exposure", "net_sector_exposure",
        "net_correlation_bucket_exposure",
    )


# --------------------------------------------------------------------- #
# Small pure helpers                                                    #
# --------------------------------------------------------------------- #
def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None  # degenerate series: correlation undefined, never assumed
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = peak = max_dd = 0.0
    for r in returns:
        equity += r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _residual_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """asof_date -> residual return (aligned cycles only)."""
    return {p["asof_date"]: p["residual"] for p in residual_series(rows)["points"]}


def _exposure_by_date(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """asof_date -> {exposure (non-fail-closed), notional} for one strategy.

    Exposure and notional come from the SAME row — the fix for the dollar-
    vs-fraction unit mismatch (#1): every exposure figure is normalized by
    the notional it was computed against, never against a cap directly.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in sorted(rows, key=lambda x: x.get("asof_date", "")):
        exp = r.get("exposure")
        notional = r.get("notional")
        if (isinstance(exp, dict) and not exp.get("fail_closed", False)
                and isinstance(notional, (int, float)) and notional > 0):
            out[r["asof_date"]] = {"exposure": exp, "notional": float(notional)}
    return out


def _normalized_buckets(exp: Dict[str, Any], notional: float,
                        fields: Sequence[str]) -> Dict[str, float]:
    """bucket key -> signed exposure as a FRACTION of the strategy's notional."""
    out: Dict[str, float] = {}
    for fld in fields:
        for bucket, val in (exp.get(fld) or {}).items():
            out[f"{fld}:{bucket}"] = float(val or 0.0) / notional
    beta = exp.get("net_beta_exposure")
    if beta is not None:
        out["beta"] = float(beta) / notional
    return out


def _check(name: str, status: str, detail: str, **extra: Any) -> Dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **extra}


def _resolve_allocations(strategies: Sequence[str],
                         allocations: Optional[Dict[str, float]]
                         ) -> Optional[Dict[str, float]]:
    """Per-strategy portfolio allocation a_s, normalized to sum 1 over the
    given strategies. ``None`` when no usable plan was supplied (rev 2.1:
    equal weight is a planning ASSUMPTION, not evidence about the real book —
    combined-book checks must go UNKNOWN, never silently assume). The single-
    strategy case needs no plan: its allocation is exactly 1 by identity.

    rev 2.2 (review of 722ffd4): COMPLETE coverage is required — every listed
    strategy (candidate AND every incumbent) must carry a positive finite
    weight. The previous version zero-filled omissions, which silently
    removed an omitted incumbent from the combined exposure, gave a
    zero-allocation "without-candidate" portfolio (division by zero in
    marginal contribution), and let an omitted candidate look harmless
    because with == without. An incomplete plan is UNKNOWN, never repaired."""
    if len(strategies) == 1:
        return {strategies[0]: 1.0}
    if not allocations:
        return None
    vals: Dict[str, float] = {}
    for s in strategies:
        v = allocations.get(s)
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(float(v)) or float(v) <= 0):
            return None  # missing / zero / negative / non-numeric: no plan
        vals[s] = float(v)
    total = sum(vals.values())
    return {s: v / total for s, v in vals.items()}


def load_operator_allocations(
        path: Path = PORTFOLIO_ALLOCATIONS_PATH) -> Optional[Dict[str, float]]:
    """The operator's capital plan from disk. ``None`` (fail-closed, logged)
    when the file is missing, unparseable, empty, or contains ANY invalid
    entry — rev 2.2: a malformed plan is rejected WHOLE, never reduced to its
    positive subset (a partial plan silently drops strategies from the
    combined book, which is exactly the failure _resolve_allocations guards
    against). The report path then leaves combined-book checks UNKNOWN rather
    than inventing a book."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError:
        return None
    except ValueError:
        logger.warning("portfolio_promotion: allocations file unparseable at %s", path)
        return None
    if isinstance(data, dict) and isinstance(data.get("allocations"), dict):
        data = data["allocations"]
    if not isinstance(data, dict) or not data:
        logger.warning("portfolio_promotion: allocations file has no mapping at %s", path)
        return None
    out: Dict[str, float] = {}
    for k, v in data.items():
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(float(v)) or float(v) <= 0):
            logger.warning(
                "portfolio_promotion: invalid allocation %r=%r in %s — rejecting the "
                "WHOLE plan (fail-closed; fix the file, no partial plan is usable)",
                k, v, path)
            return None
        out[str(k)] = float(v)
    return out


# --------------------------------------------------------------------- #
# The gate                                                              #
# --------------------------------------------------------------------- #
def evaluate_strategy(candidate: str,
                      rows_by_strategy: Dict[str, List[Dict[str, Any]]],
                      scorecards: Optional[Dict[str, Any]] = None,
                      config: GateConfig = GateConfig(),
                      allocations: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Portfolio-level verdict for ``candidate`` against every other covered
    strategy. Pure computation over ledger rows + scorecards + allocations."""
    checks: List[Dict[str, Any]] = []
    cand_rows = rows_by_strategy.get(candidate, [])
    incumbents = {s: r for s, r in rows_by_strategy.items() if s != candidate and r}
    alloc = _resolve_allocations([candidate, *incumbents.keys()], allocations)

    # 1. standalone evidence -------------------------------------------------
    att = strategy_attribution(cand_rows)
    n = att["n_aligned_cycles"]
    if n < config.min_aligned_cycles:
        checks.append(_check(
            "standalone_evidence", "unknown",
            f"{n} aligned after-cost cycles < {config.min_aligned_cycles} required",
            n_aligned_cycles=n))
    else:
        checks.append(_check("standalone_evidence", "pass",
                             f"{n} aligned after-cost cycles", n_aligned_cycles=n))

    # 2. residual alpha ------------------------------------------------------
    phi = att.get("residual_fraction")
    verdict = None
    if scorecards and candidate in scorecards:
        verdict = ((scorecards[candidate].get("decision") or {}).get("verdict"))
    if phi is None:
        checks.append(_check("residual_alpha", "unknown",
                             att.get("residual_fraction_reason") or "phi undefined",
                             phi=None, scorecard_verdict=verdict))
    elif verdict in _ALPHA_FAIL_VERDICTS or phi < config.min_phi:
        checks.append(_check(
            "residual_alpha", "fail",
            f"phi={phi:.3f} (min {config.min_phi}) / scorecard verdict={verdict} — "
            "the return does not survive systematic hedging",
            phi=phi, scorecard_verdict=verdict))
    elif verdict not in _ALPHA_PASS_VERDICTS:
        # Covers verdict None (missing scorecard — #3), inconclusive,
        # insufficient_history, no_hedge_available and cost_unmodeled:*.
        checks.append(_check(
            "residual_alpha", "unknown",
            f"scorecard verdict {verdict!r} is not affirmative evidence "
            f"(phi={phi:.3f} alone is not the full contract)",
            phi=phi, scorecard_verdict=verdict))
    else:
        checks.append(_check("residual_alpha", "pass",
                             f"phi={phi:.3f}, scorecard verdict={verdict}",
                             phi=phi, scorecard_verdict=verdict))

    # 3. hedge cost ----------------------------------------------------------
    if att["n_aligned_cycles"] == 0:
        checks.append(_check("hedge_cost", "unknown", "no aligned cycles"))
    elif not att.get("all_net_basis", False):
        checks.append(_check("hedge_cost", "unknown",
                             "gross-basis cycles present — hedge cost unmodeled for venue"))
    elif att["mean_residual"] is not None and att["mean_residual"] <= 0:
        checks.append(_check(
            "hedge_cost", "fail",
            f"hedged (residual) after-cost expectancy {att['mean_residual']:.6f} <= 0 — "
            "hedge costs / systematic removal consume the edge",
            mean_residual=att["mean_residual"]))
    else:
        checks.append(_check("hedge_cost", "pass",
                             f"hedged after-cost expectancy {att['mean_residual']:.6f} > 0",
                             mean_residual=att["mean_residual"]))

    # 4. duplication (fail > unknown > pass; every incumbent must resolve) ---
    cand_res = _residual_map(cand_rows)
    dup_details = []
    any_fail = False
    any_unresolved = False
    for name, rows in incumbents.items():
        inc_res = _residual_map(rows)
        shared = sorted(set(cand_res) & set(inc_res))
        if len(shared) < config.min_overlap_cycles:
            dup_details.append({"incumbent": name, "overlap": len(shared),
                                "corr": None, "resolved": False,
                                "why": "insufficient_overlap"})
            any_unresolved = True
            continue
        corr = _pearson([cand_res[d] for d in shared], [inc_res[d] for d in shared])
        if corr is None:
            # Enough overlap but correlation undefined (degenerate series):
            # unresolved — undefined is never assumed safe (#4).
            dup_details.append({"incumbent": name, "overlap": len(shared),
                                "corr": None, "resolved": False,
                                "why": "correlation_undefined_degenerate_series"})
            any_unresolved = True
            continue
        dup_details.append({"incumbent": name, "overlap": len(shared),
                            "corr": corr, "resolved": True, "why": None})
        if corr >= config.dup_corr_threshold:
            any_fail = True
    if any_fail:
        dup_status, dup_detail = "fail", (
            f"residual correlation >= {config.dup_corr_threshold} with an incumbent — "
            "duplicates an existing strategy")
    elif any_unresolved:
        dup_status, dup_detail = "unknown", (
            "one or more incumbent pairs unresolved (insufficient overlap or "
            "undefined correlation) — duplication cannot be ruled out")
    elif not incumbents:
        dup_status, dup_detail = "pass", "no incumbent to duplicate"
    else:
        dup_status, dup_detail = "pass", (
            f"every incumbent pair resolved below {config.dup_corr_threshold}")
    checks.append(_check("duplication", dup_status, dup_detail, pairs=dup_details))

    # 5. bucket crowding — ONE synchronized snapshot date (#1, #5) -----------
    # rev 2.1: combining strategies requires the operator's REAL capital plan;
    # equal weighting is an assumption, not evidence about the actual book.
    exp_by_strat = {candidate: _exposure_by_date(cand_rows)}
    for name, rows in incumbents.items():
        exp_by_strat[name] = _exposure_by_date(rows)
    missing_exp = sorted(s for s, m in exp_by_strat.items() if not m)
    if alloc is None:
        checks.append(_check(
            "bucket_crowding", "unknown",
            "no operator allocation plan — a combined book cannot be formed "
            "from assumed weights (supply allocations or "
            "trained_data/hedge/portfolio_allocations.json)"))
    elif missing_exp:
        checks.append(_check(
            "bucket_crowding", "unknown",
            f"no non-fail-closed exposure rows for: {missing_exp} — a partial "
            "portfolio cannot prove the combined book is inside its caps",
            missing=missing_exp))
    else:
        common_dates = set.intersection(*(set(m) for m in exp_by_strat.values()))
        if not common_dates:
            checks.append(_check(
                "bucket_crowding", "unknown",
                "no common exposure date across all covered strategies — "
                "time-misaligned snapshots do not form one portfolio"))
        else:
            eval_date = max(common_dates)
            combined: Dict[str, float] = {}
            cand_norm: Dict[str, float] = {}
            for name, m in exp_by_strat.items():
                rec = m[eval_date]
                norm = _normalized_buckets(rec["exposure"], rec["notional"],
                                           config.bucket_fields)
                if name == candidate:
                    cand_norm = norm
                a = alloc.get(name, 0.0)
                for key, frac in norm.items():
                    combined[key] = combined.get(key, 0.0) + a * frac
            breaches = []
            for key, total in combined.items():
                cap = config.max_abs_beta if key == "beta" else config.max_abs_bucket
                cand_leg = alloc.get(candidate, 0.0) * cand_norm.get(key, 0.0)
                # Breach counts against the candidate only in the breaching
                # direction — offsetting a crowded bucket is diversification.
                if abs(total) > cap and cand_leg * total > 0:
                    breaches.append({"bucket": key, "combined": total, "cap": cap,
                                     "candidate_leg": cand_leg})
            if breaches:
                checks.append(_check(
                    "bucket_crowding", "fail",
                    f"candidate pushes an already-crowded bucket past its cap "
                    f"(allocation-weighted, synchronized at {eval_date})",
                    breaches=breaches, eval_date=eval_date))
            else:
                checks.append(_check(
                    "bucket_crowding", "pass",
                    f"combined allocation-weighted book inside all caps at {eval_date} "
                    "(or candidate offsets the crowded side)",
                    n_buckets=len(combined), eval_date=eval_date))

    # 6. marginal contribution — allocation-weighted, union grid (#6) --------
    if not incumbents:
        checks.append(_check("marginal_contribution", "pass",
                             "first covered strategy — no incumbent portfolio to harm"))
    elif alloc is None:
        checks.append(_check(
            "marginal_contribution", "unknown",
            "no operator allocation plan — the combined residual portfolio "
            "cannot be formed from assumed weights"))
    else:
        inc_maps = {s: _residual_map(r) for s, r in incumbents.items()}
        inc_maps = {s: m for s, m in inc_maps.items() if m}
        # Comparison dates: candidate resolved AND at least one incumbent
        # resolved. Per date, capital is renormalized across the strategies
        # LIVE that date (allocation-weighted available-case portfolio — the
        # honest treatment of mixed cadences; NOT zero-fill: an absent
        # strategy's weight is redistributed, its return never invented).
        dates = sorted(d for d in cand_res
                       if any(d in m for m in inc_maps.values()))
        if not inc_maps or len(dates) < config.min_overlap_cycles:
            checks.append(_check(
                "marginal_contribution", "unknown",
                f"{len(dates)} dates with candidate + >=1 incumbent resolved "
                f"< {config.min_overlap_cycles} required"))
        else:
            def _portfolio(date: str, include_candidate: bool) -> Optional[float]:
                legs = []
                for s, m in inc_maps.items():
                    if date in m:
                        legs.append((alloc[s], m[date]))
                if include_candidate and date in cand_res:
                    legs.append((alloc[candidate], cand_res[date]))
                wsum = sum(a for a, _ in legs)
                if wsum <= 0:
                    return None
                return sum(a * r for a, r in legs) / wsum

            without = [v for v in (_portfolio(d, False) for d in dates) if v is not None]
            with_c = [v for v in (_portfolio(d, True) for d in dates) if v is not None]
            e_without = sum(without) / len(without)
            e_with = sum(with_c) / len(with_c)
            dd_without = _max_drawdown(without)
            dd_with = _max_drawdown(with_c)
            dd_ok = dd_with <= dd_without * config.dd_tolerance or dd_with <= 0
            # Risk-adjusted criterion: reward/vol must not degrade (raw
            # expectancy alone penalizes dilution even when an uncorrelated
            # candidate improves return-per-unit-risk).
            n_d = len(dates)
            std_without = math.sqrt(sum((x - e_without) ** 2 for x in without) / n_d)
            std_with = math.sqrt(sum((x - e_with) ** 2 for x in with_c) / n_d)
            ratio_without = e_without / std_without if std_without > 0 else None
            ratio_with = e_with / std_with if std_with > 0 else None
            risk_adj_ok = (ratio_without is not None and ratio_with is not None
                           and ratio_with >= ratio_without)
            e_ok = risk_adj_ok or e_with >= e_without
            detail = (f"allocation-weighted over {n_d} union-grid dates; "
                      f"expectancy {e_without:.6f} -> {e_with:.6f}; "
                      f"reward/vol {None if ratio_without is None else round(ratio_without, 4)}"
                      f" -> {None if ratio_with is None else round(ratio_with, 4)}; "
                      f"maxDD {dd_without:.6f} -> {dd_with:.6f} (tol x{config.dd_tolerance})")
            if e_ok and dd_ok:
                checks.append(_check("marginal_contribution", "pass", detail,
                                     expectancy_without=e_without, expectancy_with=e_with,
                                     max_dd_without=dd_without, max_dd_with=dd_with,
                                     n_dates=n_d))
            else:
                checks.append(_check(
                    "marginal_contribution", "fail",
                    "candidate worsens the combined residual portfolio: " + detail,
                    expectancy_without=e_without, expectancy_with=e_with,
                    max_dd_without=dd_without, max_dd_with=dd_with, n_dates=n_d))

    # Decision (fail-closed ordering) ---------------------------------------
    statuses = [c["status"] for c in checks]
    if "fail" in statuses:
        verdict_out = VERDICT_REJECT
    elif "unknown" in statuses:
        verdict_out = VERDICT_SHADOW
    else:
        verdict_out = VERDICT_PROMOTE
    return {
        "strategy": candidate,
        "verdict": verdict_out,
        "checks": checks,
        "incumbents": sorted(incumbents.keys()),
        "allocations": alloc,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
        "note": ("The strongest verdict this gate emits is a recommendation for "
                 "operator review — it promotes nothing itself."),
    }


# --------------------------------------------------------------------- #
# Report build                                                          #
# --------------------------------------------------------------------- #
def build_portfolio_promotion_report(
    ledger_path: Path = RAW_VS_HEDGED_LEDGER_PATH,
    scorecard_path: Path = SCORECARD_REPORT_PATH,
    out_path: Path = PROMOTION_REPORT_PATH,
    config: GateConfig = GateConfig(),
    allocations: Optional[Dict[str, float]] = None,
    allocations_path: Path = PORTFOLIO_ALLOCATIONS_PATH,
) -> Dict[str, Any]:
    """Evaluate the FULL covered-strategy set (#7): the lane registry's
    STRATEGIES union whatever the ledger contains. A covered lane with no
    rows receives an explicit CONTINUE_SHADOW verdict, never silence.

    rev 2.1: when no ``allocations`` argument is given, the operator's capital
    plan is read from ``allocations_path``; if neither exists, combined-book
    checks are UNKNOWN and no strategy can leave CONTINUE_SHADOW on the gate's
    say-so — the report never invents an equal-weight book."""
    from datetime import datetime, timezone

    allocations_source = "call_argument"
    if allocations is None:
        allocations = load_operator_allocations(allocations_path)
        allocations_source = (
            f"operator_file:{allocations_path}" if allocations is not None
            else "missing_or_invalid_fail_closed")

    rows_by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with open(ledger_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                strat = row.get("strategy")
                if strat:
                    rows_by_strategy.setdefault(strat, []).append(row)
    except OSError:
        pass

    try:
        from src.hedge.hedged_shadow_lane import STRATEGIES as _COVERED
    except Exception:  # noqa: BLE001 — registry unavailable: ledger keys only
        _COVERED = ()
    covered = sorted(set(_COVERED) | set(rows_by_strategy))
    for s in covered:
        rows_by_strategy.setdefault(s, [])

    scorecards: Dict[str, Any] = {}
    try:
        with open(scorecard_path, encoding="utf-8") as fh:
            scorecards = (json.load(fh) or {}).get("scorecards", {}) or {}
    except (OSError, ValueError):
        scorecards = {}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ledger": str(ledger_path),
        "covered_strategies": covered,
        "config": {
            "min_aligned_cycles": config.min_aligned_cycles,
            "min_phi": config.min_phi,
            "dup_corr_threshold": config.dup_corr_threshold,
            "min_overlap_cycles": config.min_overlap_cycles,
            "max_abs_beta": config.max_abs_beta,
            "max_abs_bucket": config.max_abs_bucket,
            "dd_tolerance": config.dd_tolerance,
        },
        "allocations_provided": bool(allocations),
        "allocations_source": allocations_source,
        "verdicts": {
            s: evaluate_strategy(s, rows_by_strategy, scorecards, config, allocations)
            for s in covered
        },
        "runtime_allowed": RUNTIME_ALLOWED,
        "paper_only": PAPER_ONLY,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
    }
    _atomic_write_json(report, out_path)
    return report


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".portfolio_gate_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, str(path))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
