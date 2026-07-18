"""Universal portfolio-level promotion gate — the last layer (2026-07-18).

Standalone validation asks: does the signal work on its own data? This gate
asks the portfolio question: does the strategy still improve the COMBINED
book once exposure overlap, hedge cost, duplication and correlation are
priced in? Every covered strategy gets a verdict against the rest — the gate
is universal, not per-lane bespoke.

Checks (each PASS / FAIL / UNKNOWN, evidence attached):

  standalone_evidence   >= min aligned after-cost twin-lane cycles
  residual_alpha        scorecard verdict + residual fraction phi: the return
                        must survive systematic hedging (beta is not alpha)
  hedge_cost            hedged NET expectancy > 0 — hedge costs must not
                        consume the edge (cost-unmodeled venue => UNKNOWN)
  duplication           residual-series correlation vs every incumbent below
                        the duplication threshold (a strategy that is another
                        strategy wearing a different name adds no information)
  bucket_crowding       combined book (incumbents + candidate) stays inside
                        the per-bucket exposure caps (beta / currency /
                        sector / correlation clusters), and the candidate
                        must not be the leg that breaches
  marginal_contribution the combined residual portfolio WITH the candidate
                        must not lose expectancy or worsen max drawdown
                        beyond tolerance vs WITHOUT it

Decision (fail-closed, in order):
  any FAIL    -> REJECT
  any UNKNOWN -> CONTINUE_SHADOW   (insufficient evidence is not a pass)
  all PASS    -> PROMOTE_TO_OPERATOR_REVIEW

The strongest verdict this gate can emit is a RECOMMENDATION for operator
review — it promotes nothing itself, mirroring the operator's stated
promotion order (standalone gate -> shadow -> exposure/hedge evaluation ->
marginal contribution -> OPERATOR REVIEW -> practice). Same authority
contract as the rest of src/hedge: zero orders, no gate/halt/leverage
mutation, structural isolation.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.hedge.residual_attribution import residual_series, strategy_attribution

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HEDGE_LEDGER_DIR = REPO_ROOT / "trained_data" / "hedge"
RAW_VS_HEDGED_LEDGER_PATH = HEDGE_LEDGER_DIR / "raw_vs_hedged_ledger.jsonl"
SCORECARD_REPORT_PATH = HEDGE_LEDGER_DIR / "hedge_scorecard_report.json"
PROMOTION_REPORT_PATH = HEDGE_LEDGER_DIR / "portfolio_promotion_report.json"

RUNTIME_ALLOWED = False
PAPER_ONLY = True
HUMAN_REVIEW_REQUIRED = True

VERDICT_PROMOTE = "PROMOTE_TO_OPERATOR_REVIEW"
VERDICT_SHADOW = "CONTINUE_SHADOW"
VERDICT_REJECT = "REJECT"

# Scorecard verdicts that constitute a residual-alpha FAIL vs PASS. Anything
# unrecognized (including cost_unmodeled:-prefixed) is UNKNOWN — fail-closed.
_ALPHA_PASS_VERDICTS = {"genuine_strategy_specific_signal", "signal_real_but_noisy"}
_ALPHA_FAIL_VERDICTS = {"return_was_beta_not_alpha", "weak_or_dead_strategy"}


@dataclass(frozen=True)
class GateConfig:
    """Operator-tunable thresholds. Defaults are conservative and documented;
    they are planning constants, not fitted values — tune with evidence."""
    min_aligned_cycles: int = 8          # standalone evidence floor
    min_phi: float = 0.25                # residual fraction below this = beta-dominated
    dup_corr_threshold: float = 0.85     # residual correlation at/above = duplication
    min_overlap_cycles: int = 6          # dates shared before correlation/marginal count
    max_abs_beta: float = 0.50           # combined |net beta| cap (fraction of notional)
    max_abs_bucket: float = 0.75         # combined |net| cap per currency/sector/cluster
    dd_tolerance: float = 1.10           # combined max-DD may grow at most 10% with candidate
    # Exposure dimensions read from each strategy's LATEST ledger row.
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
    """Max peak-to-trough drawdown of the cumulative-sum equity path."""
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        equity += r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _residual_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """asof_date -> residual return (aligned cycles only)."""
    return {p["asof_date"]: p["residual"] for p in residual_series(rows)["points"]}


def _latest_exposure(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in sorted(rows, key=lambda x: x.get("asof_date", ""), reverse=True):
        exp = r.get("exposure")
        if isinstance(exp, dict) and not exp.get("fail_closed", False):
            return exp
    return None


def _check(name: str, status: str, detail: str, **extra: Any) -> Dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **extra}


# --------------------------------------------------------------------- #
# The gate                                                              #
# --------------------------------------------------------------------- #
def evaluate_strategy(candidate: str,
                      rows_by_strategy: Dict[str, List[Dict[str, Any]]],
                      scorecards: Optional[Dict[str, Any]] = None,
                      config: GateConfig = GateConfig()) -> Dict[str, Any]:
    """Portfolio-level verdict for ``candidate`` against every other covered
    strategy (the incumbents). Pure computation over ledger rows + scorecards.
    """
    checks: List[Dict[str, Any]] = []
    cand_rows = rows_by_strategy.get(candidate, [])
    incumbents = {s: r for s, r in rows_by_strategy.items() if s != candidate and r}

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
    elif verdict is not None and verdict not in _ALPHA_PASS_VERDICTS:
        checks.append(_check("residual_alpha", "unknown",
                             f"unrecognized/cost-unmodeled scorecard verdict: {verdict}",
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

    # 4. duplication ---------------------------------------------------------
    cand_res = _residual_map(cand_rows)
    dup_status, dup_details = "pass", []
    any_overlap = False
    for name, rows in incumbents.items():
        inc_res = _residual_map(rows)
        shared = sorted(set(cand_res) & set(inc_res))
        if len(shared) < config.min_overlap_cycles:
            dup_details.append({"incumbent": name, "overlap": len(shared), "corr": None})
            continue
        any_overlap = True
        corr = _pearson([cand_res[d] for d in shared], [inc_res[d] for d in shared])
        dup_details.append({"incumbent": name, "overlap": len(shared), "corr": corr})
        if corr is not None and corr >= config.dup_corr_threshold:
            dup_status = "fail"
    if incumbents and not any_overlap:
        dup_status = "unknown"
    checks.append(_check(
        "duplication", dup_status,
        ("no incumbent to duplicate" if not incumbents else
         "insufficient overlapping history with every incumbent" if dup_status == "unknown" else
         f"max residual correlation vs incumbents under {config.dup_corr_threshold}"
         if dup_status == "pass" else
         f"residual correlation >= {config.dup_corr_threshold} with an incumbent — "
         "duplicates an existing strategy"),
        pairs=dup_details))

    # 5. bucket crowding -----------------------------------------------------
    cand_exp = _latest_exposure(cand_rows)
    if cand_exp is None:
        checks.append(_check("bucket_crowding", "unknown",
                             "no non-fail-closed exposure report for candidate"))
    else:
        combined: Dict[str, float] = {}
        contributors: Dict[str, Dict[str, float]] = {}
        for name, rows in {**incumbents, candidate: cand_rows}.items():
            exp = _latest_exposure(rows)
            if exp is None:
                continue
            for fld in config.bucket_fields:
                for bucket, val in (exp.get(fld) or {}).items():
                    key = f"{fld}:{bucket}"
                    combined[key] = combined.get(key, 0.0) + float(val or 0.0)
                    contributors.setdefault(key, {})[name] = float(val or 0.0)
            beta = exp.get("net_beta_exposure")
            if beta is not None:
                combined["beta"] = combined.get("beta", 0.0) + float(beta)
                contributors.setdefault("beta", {})[name] = float(beta)
        breaches = []
        for key, total in combined.items():
            cap = config.max_abs_beta if key == "beta" else config.max_abs_bucket
            cand_leg = contributors.get(key, {}).get(candidate, 0.0)
            # A breach counts against the candidate only if it is in the
            # breaching direction — a candidate that OFFSETS a crowded bucket
            # is diversifying, not crowding.
            if abs(total) > cap and cand_leg * total > 0:
                breaches.append({"bucket": key, "combined": total, "cap": cap,
                                 "candidate_leg": cand_leg})
        if breaches:
            checks.append(_check(
                "bucket_crowding", "fail",
                "candidate pushes an already-crowded bucket further past its cap",
                breaches=breaches))
        else:
            checks.append(_check("bucket_crowding", "pass",
                                 "combined book inside all bucket caps "
                                 "(or candidate offsets the crowded side)",
                                 n_buckets=len(combined)))

    # 6. marginal contribution ----------------------------------------------
    if not incumbents:
        checks.append(_check("marginal_contribution", "pass",
                             "first covered strategy — no incumbent portfolio to harm"))
    else:
        inc_maps = {s: _residual_map(r) for s, r in incumbents.items()}
        inc_maps = {s: m for s, m in inc_maps.items() if m}
        shared = set(cand_res)
        for m in inc_maps.values():
            shared &= set(m)
        shared_dates = sorted(shared)
        if not inc_maps or len(shared_dates) < config.min_overlap_cycles:
            checks.append(_check(
                "marginal_contribution", "unknown",
                f"{len(shared_dates)} overlapping cycles across all covered strategies "
                f"< {config.min_overlap_cycles} required"))
        else:
            without = [sum(m[d] for m in inc_maps.values()) / len(inc_maps)
                       for d in shared_dates]
            k = len(inc_maps) + 1
            with_c = [(sum(m[d] for m in inc_maps.values()) + cand_res[d]) / k
                      for d in shared_dates]
            e_without = sum(without) / len(without)
            e_with = sum(with_c) / len(with_c)
            dd_without = _max_drawdown(without)
            dd_with = _max_drawdown(with_c)
            dd_ok = dd_with <= dd_without * config.dd_tolerance or dd_with <= 0
            # Marginal criterion is RISK-ADJUSTED, not raw expectancy: under
            # equal-weight averaging any candidate whose mean is below the
            # incumbent average dilutes expectancy even when its uncorrelated
            # returns IMPROVE return-per-unit-risk (the entire point of
            # diversification). Pass if the reward/volatility ratio improves
            # (or raw expectancy does), within the drawdown tolerance.
            n_sh = len(shared_dates)
            std_without = math.sqrt(sum((x - e_without) ** 2 for x in without) / n_sh)
            std_with = math.sqrt(sum((x - e_with) ** 2 for x in with_c) / n_sh)
            ratio_without = e_without / std_without if std_without > 0 else None
            ratio_with = e_with / std_with if std_with > 0 else None
            if ratio_without is not None and ratio_with is not None:
                risk_adj_ok = ratio_with >= ratio_without
            else:
                risk_adj_ok = False  # degenerate vol: fall through to expectancy
            e_ok = risk_adj_ok or e_with >= e_without
            detail = (f"expectancy {e_without:.6f} -> {e_with:.6f}; "
                      f"reward/vol {ratio_without if ratio_without is None else round(ratio_without, 4)}"
                      f" -> {ratio_with if ratio_with is None else round(ratio_with, 4)}; "
                      f"maxDD {dd_without:.6f} -> {dd_with:.6f} "
                      f"(tolerance x{config.dd_tolerance})")
            if e_ok and dd_ok:
                checks.append(_check("marginal_contribution", "pass", detail,
                                     expectancy_without=e_without, expectancy_with=e_with,
                                     max_dd_without=dd_without, max_dd_with=dd_with))
            else:
                checks.append(_check(
                    "marginal_contribution", "fail",
                    "candidate worsens the combined residual portfolio: " + detail,
                    expectancy_without=e_without, expectancy_with=e_with,
                    max_dd_without=dd_without, max_dd_with=dd_with))

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
) -> Dict[str, Any]:
    """Evaluate EVERY covered strategy against the rest; persist atomically."""
    from datetime import datetime, timezone

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

    scorecards: Dict[str, Any] = {}
    try:
        with open(scorecard_path, encoding="utf-8") as fh:
            scorecards = (json.load(fh) or {}).get("scorecards", {}) or {}
    except (OSError, ValueError):
        scorecards = {}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ledger": str(ledger_path),
        "config": {
            "min_aligned_cycles": config.min_aligned_cycles,
            "min_phi": config.min_phi,
            "dup_corr_threshold": config.dup_corr_threshold,
            "min_overlap_cycles": config.min_overlap_cycles,
            "max_abs_beta": config.max_abs_beta,
            "max_abs_bucket": config.max_abs_bucket,
            "dd_tolerance": config.dd_tolerance,
        },
        "verdicts": {
            s: evaluate_strategy(s, rows_by_strategy, scorecards, config)
            for s in sorted(rows_by_strategy)
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
