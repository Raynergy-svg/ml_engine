"""AXIOM Hedge Layer — Phase 4: raw-vs-hedged scorecard (SHADOW / ANALYSIS-ONLY).

Reads ``trained_data/hedge/raw_vs_hedged_ledger.jsonl`` (written by
``hedged_shadow_lane.py``) and turns the accumulated Lane-A/Lane-B history
into the decisive readout: **does hedging improve after-cost expectancy for
this strategy, or was its return just market beta?**

Nothing here touches execution, halt state, or a gate. Pure read + compute
over an already-logged, already-analysis-only ledger.

Decision readout (the roadmap's three-way test, evaluated per strategy):

* hedge improves after-cost expectancy AND cuts max drawdown
  -> ``"signal_real_but_noisy"`` — the return survives losing its market
  beta; what's left is real, just noisy.
* hedge cuts drawdown but after-cost expectancy is still <= 0
  -> ``"weak_or_dead"`` — vol goes down, but there was no positive
  expectancy to protect in the first place.
* raw wins (raw expectancy > 0) but hedged fails (hedged expectancy <= 0)
  -> ``"return_was_beta_not_alpha"`` — the raw book's return doesn't survive
  losing its market/sector exposure; it WAS the exposure.
* anything else (mixed signal, or fewer than ``MIN_HISTORY_FOR_VERDICT``
  cycles logged) -> ``"insufficient_history"`` / ``"inconclusive"`` — never
  forces a verdict onto a thin sample. Matches the same "never fabricates a
  Sharpe on n<2" discipline ``momentum_shadow.forward_oos_summary`` and
  ``track_b_shadow.forward_oos_summary`` already use.

Honesty on cost (the equity/SPY case this module exists to get right): when
a strategy's applied hedge has ``cost_known=False`` (no OANDA fills exist
for SPY/sector-ETF instruments — see ``hedged_shadow_lane`` docstring), the
NET after-cost expectancy for Lane B is reported as ``None``, never
defaulted to the gross figure. The scorecard instead surfaces
``hedged.gross_after_cost_expectancy`` alongside an explicit
``cost_unmodeled_for_venue: true`` flag and reruns the SAME three-way
decision logic on the GROSS number, prefixing the verdict with
``"cost_unmodeled:"`` so it can never be read as a genuine after-cost
conclusion.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_VS_HEDGED_LEDGER_PATH = REPO_ROOT / "trained_data" / "hedge" / "raw_vs_hedged_ledger.jsonl"
SCORECARD_REPORT_PATH = REPO_ROOT / "trained_data" / "hedge" / "hedge_scorecard_report.json"

RUNTIME_ALLOWED = False
PAPER_ONLY = True
HUMAN_REVIEW_REQUIRED = True

# Below this many logged cycles, no verdict is forced — a Sharpe/drawdown/
# expectancy read on 1-2 points is noise dressed up as a conclusion.
MIN_HISTORY_FOR_VERDICT = 3
# Calendar days per year, used to annualize an irregular-cadence cycle
# series via the median day-gap between logged cycles (not an assumed
# trading-day count, since strategies here rebalance on very different
# cadences: crypto momentum ~4h/1d, equity harvester ~weekly, Track B ~21d).
CALENDAR_DAYS_PER_YEAR = 365.25

VERDICT_GENUINE = "genuine_strategy_specific_signal"
VERDICT_SIGNAL_REAL_BUT_NOISY = "signal_real_but_noisy"
VERDICT_WEAK_OR_DEAD = "weak_or_dead"
VERDICT_BETA_NOT_ALPHA = "return_was_beta_not_alpha"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_INSUFFICIENT_HISTORY = "insufficient_history"


def read_ledger_rows(ledger_path: Path = RAW_VS_HEDGED_LEDGER_PATH) -> List[Dict[str, Any]]:
    """Tolerant JSONL reader — corrupt lines are skipped and logged, never
    crash the scorecard. Missing file -> [] (fail soft)."""
    if not ledger_path.exists():
        logger.warning("hedge_scorecard: ledger missing at %s", ledger_path)
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(ledger_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("hedge_scorecard: skipping corrupt ledger line")
    except OSError as exc:
        logger.warning("hedge_scorecard: ledger unreadable at %s (%s)", ledger_path, exc)
        return []
    return rows


def group_by_strategy(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        strategy = row.get("strategy")
        if not strategy:
            continue
        grouped.setdefault(strategy, []).append(row)
    for strategy_rows in grouped.values():
        strategy_rows.sort(key=lambda r: r.get("asof_date", ""))
    return grouped


# --------------------------------------------------------------------- #
# Per-lane statistics                                                   #
# --------------------------------------------------------------------- #
def _median_gap_days(asof_dates: Sequence[str]) -> Optional[float]:
    """Median calendar-day gap between consecutive ``asof_date`` strings.
    ``None`` if fewer than 2 distinct/parseable dates (annualization would
    be meaningless)."""
    from datetime import date

    parsed = []
    for d in asof_dates:
        try:
            parsed.append(date.fromisoformat(str(d)[:10]))
        except ValueError:
            continue
    parsed = sorted(set(parsed))
    if len(parsed) < 2:
        return None
    gaps = [(b - a).days for a, b in zip(parsed[:-1], parsed[1:]) if (b - a).days > 0]
    if not gaps:
        return None
    gaps.sort()
    mid = len(gaps) // 2
    return float(gaps[mid]) if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2.0


def _mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _sample_std(values: Sequence[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _max_drawdown(returns: Sequence[float]) -> Optional[float]:
    """Max peak-to-trough drawdown (positive magnitude, e.g. 0.12 = 12%) of
    the compounded equity curve implied by ``returns``. ``None`` on an
    empty series."""
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def _series_stats(returns: Sequence[float], asof_dates: Sequence[str]) -> Dict[str, Any]:
    """Core stats for one return series: expectancy, hit rate, drawdown,
    Sharpe/Sortino (annualized via the series' own median cadence).
    Sharpe/Sortino are ``None`` below ``MIN_HISTORY_FOR_VERDICT`` points or
    when std is degenerate (0 / undefined) — never a fabricated ratio."""
    n = len(returns)
    expectancy = _mean(returns)
    hit_rate = (sum(1 for r in returns if r > 0) / n) if n else None
    max_dd = _max_drawdown(returns)

    sharpe = None
    sortino = None
    if n >= MIN_HISTORY_FOR_VERDICT:
        std = _sample_std(returns)
        gap_days = _median_gap_days(asof_dates)
        if std and std > 0 and gap_days and gap_days > 0:
            periods_per_year = CALENDAR_DAYS_PER_YEAR / gap_days
            sharpe = (expectancy / std) * math.sqrt(periods_per_year)
        downside = [r for r in returns if r < 0]
        downside_std = _sample_std(downside) if len(downside) >= 2 else None
        if downside_std and downside_std > 0 and gap_days and gap_days > 0:
            periods_per_year = CALENDAR_DAYS_PER_YEAR / gap_days
            sortino = (expectancy / downside_std) * math.sqrt(periods_per_year)

    return {
        "n": n,
        "after_cost_expectancy": expectancy,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "sharpe_annualized": sharpe,
        "sortino_annualized": sortino,
    }


def _exposure_drift(exposure_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How much the book's net-beta / concentration picture moved cycle to
    cycle — a large drift means the "book" a Sharpe was computed over wasn't
    stable, which matters for how much to trust the ratio."""
    betas = [e.get("net_beta_exposure") for e in exposure_reports if e.get("net_beta_exposure") is not None]
    warning_counts = [len(e.get("concentration_warnings") or []) for e in exposure_reports]
    beta_std = _sample_std(betas) if len(betas) >= 2 else None
    return {
        "net_beta_mean": _mean(betas),
        "net_beta_std": beta_std,
        "avg_concentration_warnings": _mean(warning_counts),
        "any_fail_closed": any(e.get("fail_closed") for e in exposure_reports),
    }


# --------------------------------------------------------------------- #
# Decision readout                                                      #
# --------------------------------------------------------------------- #
def _three_way_decision(raw_expectancy: Optional[float], raw_max_dd: Optional[float],
                        hedged_expectancy: Optional[float], hedged_max_dd: Optional[float]) -> str:
    if raw_expectancy is None or hedged_expectancy is None or raw_max_dd is None or hedged_max_dd is None:
        return VERDICT_INCONCLUSIVE
    expectancy_improved = hedged_expectancy > raw_expectancy
    dd_cut = hedged_max_dd < raw_max_dd
    if expectancy_improved and dd_cut:
        return VERDICT_SIGNAL_REAL_BUT_NOISY
    if dd_cut and hedged_expectancy <= 0:
        return VERDICT_WEAK_OR_DEAD
    if raw_expectancy > 0 and hedged_expectancy <= 0:
        return VERDICT_BETA_NOT_ALPHA
    # 2026-07-18 (gate-contract fix): the "raw profitable AND hedged
    # profitable" quadrant — the return SURVIVES removal of the systematic
    # exposure — is the documented "likely genuine strategy-specific signal"
    # verdict from the interpretation table. It previously fell through to
    # INCONCLUSIVE, so the promotion gate's pass-set referenced a verdict the
    # scorecard could never emit.
    if raw_expectancy > 0 and hedged_expectancy > 0:
        return VERDICT_GENUINE
    return VERDICT_INCONCLUSIVE


def decision_readout(raw_stats: Dict[str, Any], hedged_stats: Dict[str, Any],
                     hedged_gross_stats: Optional[Dict[str, Any]],
                     cost_unmodeled: bool) -> Dict[str, Any]:
    n = raw_stats["n"]
    if n < MIN_HISTORY_FOR_VERDICT:
        return {"verdict": VERDICT_INSUFFICIENT_HISTORY,
                "basis": "net",
                "reason": f"only {n} logged cycle(s) (< {MIN_HISTORY_FOR_VERDICT}) — "
                          "no verdict forced on this thin a sample."}

    if not cost_unmodeled:
        verdict = _three_way_decision(
            raw_stats["after_cost_expectancy"], raw_stats["max_drawdown"],
            hedged_stats["after_cost_expectancy"], hedged_stats["max_drawdown"])
        return {"verdict": verdict, "basis": "net", "reason": None}

    # Cost-unmodeled venue (e.g. equity SPY/sector-ETF hedge, no OANDA fill
    # history): report the SAME 3-way logic on GROSS numbers, but the
    # verdict is unmistakably prefixed so it is never read as a genuine
    # after-cost conclusion. See module docstring "Honesty on cost".
    gross_verdict = _three_way_decision(
        raw_stats["after_cost_expectancy"], raw_stats["max_drawdown"],
        hedged_gross_stats["after_cost_expectancy"] if hedged_gross_stats else None,
        hedged_gross_stats["max_drawdown"] if hedged_gross_stats else None)
    return {
        "verdict": f"cost_unmodeled:{gross_verdict}",
        "basis": "gross",
        "reason": "hedge cost unmodeled for this venue (no OANDA fill history for the hedge "
                  "instrument) — this verdict is computed on GROSS hedged performance, NOT "
                  "after-cost. Treat as advisory, not a genuine after-cost conclusion.",
    }


# --------------------------------------------------------------------- #
# Top-level scorecard                                                   #
# --------------------------------------------------------------------- #
def compute_strategy_scorecard(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = sorted(rows, key=lambda r: r.get("asof_date", ""))
    asof_dates = [r.get("asof_date", "") for r in rows]  # full logged range, for asof_range only

    # Each lane's Sharpe/Sortino must annualize using the SAME dates its own
    # return series was drawn from -- pairing an all-rows date list with a
    # returns list already filtered down to the resolved subset silently
    # miscomputes the cadence (e.g. a stale-price cycle interleaved among
    # dense ones would understate the median gap and inflate the ratio).
    raw_rows = [r for r in rows if r["raw"]["net_return"] is not None]
    raw_returns = [r["raw"]["net_return"] for r in raw_rows]
    raw_dates = [r.get("asof_date", "") for r in raw_rows]
    raw_stats = _series_stats(raw_returns, raw_dates)

    # Only cycles where a hedge was actually APPLIED belong in the hedged
    # series -- a cycle with no valid hedge sets hedged == raw by
    # construction (hedged_shadow_lane.run_cycle_for_strategy), and mixing
    # those trivial rows in would silently dilute the alpha-vs-beta read
    # toward "hedged tracks raw" for reasons that have nothing to do with
    # hedge economics.
    applied_rows = [r for r in rows if r["hedge"]["status"] == "applied"]

    hedged_net_rows = [r for r in applied_rows if r["hedged"]["net_return"] is not None]
    hedged_net_returns = [r["hedged"]["net_return"] for r in hedged_net_rows]
    hedged_net_dates = [r.get("asof_date", "") for r in hedged_net_rows]
    hedged_stats = _series_stats(hedged_net_returns, hedged_net_dates)

    hedged_gross_rows = [r for r in applied_rows if r["hedged"]["gross_return"] is not None]
    hedged_gross_returns = [r["hedged"]["gross_return"] for r in hedged_gross_rows]
    hedged_gross_dates = [r.get("asof_date", "") for r in hedged_gross_rows]
    hedged_gross_stats = _series_stats(hedged_gross_returns, hedged_gross_dates) if hedged_gross_rows else None

    # 2026-07-18 audit fix (F2): the alpha-vs-beta decision must compare raw
    # and hedged over the SAME cycles. raw_stats above spans ALL resolved
    # cycles (reported for reference), but if the hedge only applied on a
    # subset, comparing full-history raw expectancy to subset hedged
    # expectancy attributes sample-composition differences to the hedge.
    # The PAIRED baseline is the raw leg of exactly the rows that entered the
    # hedged series (hedged net_return non-null implies raw non-null by
    # construction in run_cycle_for_strategy).
    _paired_src = [
        r for r in (hedged_net_rows if hedged_net_rows else hedged_gross_rows)
        if r["raw"]["net_return"] is not None
    ]
    raw_paired_returns = [r["raw"]["net_return"] for r in _paired_src]
    raw_paired_dates = [r.get("asof_date", "") for r in _paired_src]
    raw_paired_stats = _series_stats(raw_paired_returns, raw_paired_dates)

    # "Cost unmodeled" is TRUE for this strategy's hedge iff every applied
    # cycle came back cost_known=False (equity SPY/sector-ETF case) — never
    # inferred from a single cycle.
    cost_known_flags = [r["hedge"]["overlay_cost_known"] for r in applied_rows]
    cost_unmodeled = bool(applied_rows) and not any(cost_known_flags)
    hedge_unavailable = not applied_rows

    exposure_reports = [r.get("exposure", {}) for r in rows]
    drift = _exposure_drift(exposure_reports)

    decision = decision_readout(raw_paired_stats, hedged_stats, hedged_gross_stats, cost_unmodeled)
    if hedge_unavailable:
        statuses = sorted({r["hedge"]["status"] for r in rows})
        # Same n (resolved-raw-return count) as decision_readout's own gate
        # -- otherwise a strategy with plenty of logged rows but few
        # resolved raw returns could get "no_hedge_available" instead of
        # the correct "insufficient_history".
        if raw_stats["n"] < MIN_HISTORY_FOR_VERDICT:
            no_hedge_verdict = VERDICT_INSUFFICIENT_HISTORY
        else:
            no_hedge_verdict = "no_hedge_available"
        decision = {
            "verdict": no_hedge_verdict,
            "basis": "n/a",
            "reason": f"no cycle in this history had an applied hedge (status(es): {statuses}) — "
                      "raw and hedged lanes are identical by construction; no alpha-vs-beta "
                      "read is possible from this history.",
        }

    return {
        "n_cycles": len(rows),
        "asof_range": [asof_dates[0], asof_dates[-1]] if asof_dates else [None, None],
        "raw": raw_stats,
        # F2 (2026-07-18): the baseline actually used for the verdict —
        # raw returns of exactly the cycles in the hedged series.
        "raw_paired": raw_paired_stats,
        "hedged_net": hedged_stats,
        "hedged_gross": hedged_gross_stats,
        "cost_unmodeled_for_venue": cost_unmodeled,
        "hedge_unavailable": hedge_unavailable,
        "exposure_drift": drift,
        "decision": decision,
        "runtime_allowed": RUNTIME_ALLOWED,
        "paper_only": PAPER_ONLY,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
    }


def build_full_scorecard(ledger_path: Path = RAW_VS_HEDGED_LEDGER_PATH) -> Dict[str, Any]:
    rows = read_ledger_rows(ledger_path)
    grouped = group_by_strategy(rows)
    scorecards = {strategy: compute_strategy_scorecard(strategy_rows)
                  for strategy, strategy_rows in grouped.items()}
    return {
        "scorecards": scorecards,
        "strategies_with_history": sorted(scorecards.keys()),
        "runtime_allowed": RUNTIME_ALLOWED,
        "paper_only": PAPER_ONLY,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
    }


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".hedge_scorecard_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main() -> Dict[str, Any]:
    report = build_full_scorecard()
    _atomic_write_json(report, SCORECARD_REPORT_PATH)
    # Layer 8 (2026-07-18): the residual-attribution report is produced in the
    # same pass so the learning layer's shadow evidence stays as fresh as the
    # scorecard it derives from. Fail-soft: attribution problems never block
    # the scorecard itself.
    try:
        from src.hedge.residual_attribution import build_residual_attribution_report
        build_residual_attribution_report()
    except Exception as _ra_err:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "residual attribution report not produced: %s", _ra_err)
    # Universal portfolio promotion gate (2026-07-18): verdict per covered
    # strategy against the rest, from the same ledger + the scorecard just
    # written. Fail-soft: gate problems never block the scorecard.
    try:
        from src.hedge.portfolio_promotion import build_portfolio_promotion_report
        build_portfolio_promotion_report()
    except Exception as _pg_err:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "portfolio promotion report not produced: %s", _pg_err)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _report = main()
    for _strategy, _card in _report["scorecards"].items():
        print(f"=== {_strategy} (n={_card['n_cycles']}) ===")
        print(f"  raw expectancy={_card['raw']['after_cost_expectancy']} "
              f"max_dd={_card['raw']['max_drawdown']}")
        print(f"  hedged(net) expectancy={_card['hedged_net']['after_cost_expectancy']} "
              f"max_dd={_card['hedged_net']['max_drawdown']}")
        print(f"  cost_unmodeled_for_venue={_card['cost_unmodeled_for_venue']}")
        print(f"  decision={_card['decision']['verdict']}")
