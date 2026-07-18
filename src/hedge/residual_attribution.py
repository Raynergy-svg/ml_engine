"""Residual-alpha attribution — Layer 8 of the hedge plane (2026-07-18).

Separates each covered strategy's return into the part the systematic hedge
explains and the part it does not, then exposes that residual as the credit
signal the learning layer SHOULD reward instead of raw P&L.

Identity (per aligned twin-lane cycle, F1-fixed schema):

    R_hedged,t   = R_raw,t + R_overlay,t            (hedged_shadow_lane)
    R_residual,t = R_hedged,t                       (market-neutralized book)
    R_system,t   = R_raw,t - R_residual,t           (= -R_overlay,t)

so ``residual + systematic == raw`` exactly, cycle by cycle. The residual is
only defined on cycles where a hedge was actually APPLIED and both lanes
resolved — everything else fails closed (excluded, reasons recorded), never
imputed. Strategies without lane coverage (today: the 15-agent scanner's own
book) get an explicit ``no_lane_coverage`` pass-through — raw credit remains
the honest signal until the exposure engine covers that book; we never borrow
another strategy's residual fraction.

Authority contract (same as the rest of src/hedge): ZERO orders, no gate,
halt or leverage mutation. This module computes and reports. The reward hook
is consumed by the RL sync ONLY in shadow (journal annotation) unless the
operator flips ``enable_residual_alpha_rewards`` after reviewing shadow
evidence — promotion requires evidence, per the repo's governance doctrine.

The exponential credit update requested by the operator,

    w_{j,t+1} = w_{j,t} * exp(eta * contribution_j * R_residual,t)

is provided as a pure primitive (`exponential_credit_update`) for the future
strategy-allocation layer; the existing per-trade agent updater instead
consumes `residual_reward` (a bounded multiplier on its shaped reward), which
composes with its EMA damping and [0.1, 2.0] bounds without replacing them.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HEDGE_LEDGER_DIR = REPO_ROOT / "trained_data" / "hedge"
RAW_VS_HEDGED_LEDGER_PATH = HEDGE_LEDGER_DIR / "raw_vs_hedged_ledger.jsonl"
RESIDUAL_REPORT_PATH = HEDGE_LEDGER_DIR / "residual_attribution_report.json"

# Same advisory triplet as the rest of the package — descriptive metadata;
# isolation is structural (no order path exists in src/hedge).
RUNTIME_ALLOWED = False
PAPER_ONLY = True
HUMAN_REVIEW_REQUIRED = True

# Minimum aligned residual cycles before a residual fraction may modulate any
# reward (mirrors hedge_scorecard.MIN_HISTORY_FOR_VERDICT's spirit; slightly
# stricter because this number would touch learning, not just reporting).
MIN_HISTORY_FOR_REWARD = 5

# Bounds for the reward multiplier. Floor 0.0: an agent never gets NEGATIVE
# credit flips from this layer (the existing updater owns win/loss semantics);
# cap 2.0 mirrors the RL sync's reward-ratio clamp so no single mechanism can
# dominate the weight update.
MULTIPLIER_FLOOR = 0.0
MULTIPLIER_CAP = 2.0

# |mean raw| below this is treated as "no measurable raw expectancy" — a
# residual FRACTION of ~0/0 is meaningless and fails closed to None.
RAW_EPSILON = 1e-9


# --------------------------------------------------------------------- #
# Per-cycle residual series                                             #
# --------------------------------------------------------------------- #
def residual_series(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aligned per-cycle residual decomposition for ONE strategy's rows.

    Returns {"points": [...], "excluded": [{asof, reason}, ...]}. A point is
    emitted only when the cycle's hedge was APPLIED and both lanes resolved;
    the basis records whether the residual is after-cost ("net") or gross
    ("gross_cost_unknown" — equity ETF hedges without a cost model). Gross
    and net points are kept in ONE series with the basis tagged per point;
    consumers that need strict after-cost residuals filter on basis.
    """
    points: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: x.get("asof_date", "")):
        asof = r.get("asof_date")
        status = (r.get("hedge") or {}).get("status")
        raw = (r.get("raw") or {}).get("net_return")
        if status != "applied":
            excluded.append({"asof_date": asof, "reason": f"hedge_not_applied:{status}"})
            continue
        if raw is None:
            excluded.append({"asof_date": asof, "reason": "raw_unresolved"})
            continue
        hedged = r.get("hedged") or {}
        if hedged.get("net_return") is not None:
            residual, basis = float(hedged["net_return"]), "net"
        elif hedged.get("gross_return") is not None:
            residual, basis = float(hedged["gross_return"]), "gross_cost_unknown"
        else:
            excluded.append({"asof_date": asof, "reason": "hedged_unresolved"})
            continue
        raw_f = float(raw)
        points.append({
            "asof_date": asof,
            "raw": raw_f,
            "residual": residual,
            "systematic": raw_f - residual,
            "basis": basis,
        })
    return {"points": points, "excluded": excluded}


# --------------------------------------------------------------------- #
# Per-strategy attribution summary                                      #
# --------------------------------------------------------------------- #
def strategy_attribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize one strategy's residual decomposition.

    ``residual_fraction`` (phi) = mean(residual) / mean(raw) over aligned
    cycles — the share of the strategy's expectancy that survives systematic
    hedging. phi ~= 1: return is residual (candidate alpha). phi ~= 0: return
    was the systematic exposure (beta, not alpha). phi > 1: the systematic
    component was HURTING the strategy. phi < 0: residual expectancy has the
    opposite sign of raw. None (with reason) when |mean raw| < RAW_EPSILON or
    history is empty — never a fabricated fraction.
    """
    series = residual_series(rows)
    pts = series["points"]
    n = len(pts)
    out: Dict[str, Any] = {
        "n_aligned_cycles": n,
        "n_excluded": len(series["excluded"]),
        "excluded_reasons": sorted({e["reason"] for e in series["excluded"]}),
        "asof_range": [pts[0]["asof_date"], pts[-1]["asof_date"]] if pts else [None, None],
        "all_net_basis": bool(pts) and all(p["basis"] == "net" for p in pts),
        "mean_raw": None,
        "mean_residual": None,
        "mean_systematic": None,
        "residual_fraction": None,
        "residual_fraction_reason": None,
    }
    if n == 0:
        out["residual_fraction_reason"] = "no_aligned_residual_cycles"
        return out
    mean_raw = sum(p["raw"] for p in pts) / n
    mean_res = sum(p["residual"] for p in pts) / n
    out["mean_raw"] = mean_raw
    out["mean_residual"] = mean_res
    out["mean_systematic"] = mean_raw - mean_res
    if abs(mean_raw) < RAW_EPSILON:
        out["residual_fraction_reason"] = "raw_expectancy_indistinguishable_from_zero"
        return out
    out["residual_fraction"] = mean_res / mean_raw
    return out


# --------------------------------------------------------------------- #
# Reward hook (consumed by the RL sync — shadow-first)                  #
# --------------------------------------------------------------------- #
def residual_reward(raw_reward: float, attribution: Optional[Dict[str, Any]],
                    min_history: int = MIN_HISTORY_FOR_REWARD) -> Dict[str, Any]:
    """Map a raw trade reward to its residual-scaled value.

    Fail-closed pass-through (multiplier 1.0, applied=False, reason recorded)
    when: no attribution exists for the strategy (no lane coverage), history
    is below ``min_history``, any aligned cycle is gross-basis only, or the
    residual fraction is undefined. Otherwise the multiplier is
    clamp(phi, MULTIPLIER_FLOOR, MULTIPLIER_CAP) — a pure-beta strategy's
    agents earn ~0 credit, a pure-alpha strategy's earn full credit, and the
    unclamped phi is reported alongside for the shadow evidence trail.
    """
    def _pass(reason: str) -> Dict[str, Any]:
        return {"reward": float(raw_reward), "multiplier": 1.0, "applied": False,
                "reason": reason, "phi": None}

    if attribution is None:
        return _pass("no_lane_coverage")
    phi = attribution.get("residual_fraction")
    if phi is None:
        return _pass(attribution.get("residual_fraction_reason") or "residual_fraction_undefined")
    n = int(attribution.get("n_aligned_cycles") or 0)
    if n < min_history:
        return _pass(f"insufficient_history:{n}<{min_history}")
    if not attribution.get("all_net_basis", False):
        return _pass("gross_basis_cycles_present_cost_unmodeled")
    mult = max(MULTIPLIER_FLOOR, min(MULTIPLIER_CAP, float(phi)))
    return {"reward": float(raw_reward) * mult, "multiplier": mult, "applied": True,
            "reason": "residual_fraction_applied", "phi": float(phi)}


# --------------------------------------------------------------------- #
# Exponential credit update (operator's Layer-8 formula, pure)          #
# --------------------------------------------------------------------- #
def exponential_credit_update(weight: float, signed_contribution: float,
                              residual_return: float, eta: float = 0.5,
                              min_w: float = 0.1, max_w: float = 2.0) -> float:
    """w' = clamp(w * exp(eta * contribution * R_residual), [min_w, max_w]).

    The multiplicative-weights update the operator specified for residual-
    based credit. ``signed_contribution`` in [-1, 1] (FOR=+, AGAINST=-,
    scaled by conviction); ``residual_return`` is the period's residual.
    Non-finite inputs fail closed to the unchanged weight — a corrupt
    residual must never move a weight.
    """
    if not (math.isfinite(weight) and math.isfinite(signed_contribution)
            and math.isfinite(residual_return) and math.isfinite(eta)):
        return weight
    return max(min_w, min(max_w, weight * math.exp(eta * signed_contribution * residual_return)))


# --------------------------------------------------------------------- #
# Report build + fail-soft consumer-side loader                         #
# --------------------------------------------------------------------- #
def build_residual_attribution_report(
    ledger_path: Path = RAW_VS_HEDGED_LEDGER_PATH,
    out_path: Path = RESIDUAL_REPORT_PATH,
) -> Dict[str, Any]:
    """Read the twin-lane ledger, attribute per strategy, persist atomically."""
    from datetime import datetime, timezone

    rows: List[Dict[str, Any]] = []
    try:
        with open(ledger_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # corrupt line: skipped, surfaced via counts below
    except OSError:
        rows = []

    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        strat = r.get("strategy")
        if strat:
            by_strategy.setdefault(strat, []).append(r)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(ledger_path),
        "n_ledger_rows": len(rows),
        "strategies": {s: strategy_attribution(sr) for s, sr in sorted(by_strategy.items())},
        "min_history_for_reward": MIN_HISTORY_FOR_REWARD,
        "multiplier_bounds": [MULTIPLIER_FLOOR, MULTIPLIER_CAP],
        "runtime_allowed": RUNTIME_ALLOWED,
        "paper_only": PAPER_ONLY,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
        "note": ("Advisory shadow evidence. Reward modulation only activates via "
                 "ScannerConfig.enable_residual_alpha_rewards after operator review."),
    }
    _atomic_write_json(report, out_path)
    return report


def load_attribution_for(strategy: str,
                         report_path: Path = RESIDUAL_REPORT_PATH) -> Optional[Dict[str, Any]]:
    """Fail-soft consumer-side lookup: None (=> pass-through) on any problem."""
    try:
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
        att = (report.get("strategies") or {}).get(strategy)
        return att if isinstance(att, dict) else None
    except (OSError, ValueError):
        return None


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".residual_attr_", suffix=".tmp")
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
