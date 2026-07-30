"""Deterministic forward-return scorecard for the crypto XS-momentum lane.

This is the crypto-momentum analogue of
:mod:`src.equity.research.rank_ic_scorecard`: a pure, offline, RNG-free,
network-free scorer that turns an *already-realized* per-period net-return series
into an honest verdict on whether the frozen cross-sectional momentum
construction actually earns a statistically distinguishable, cost-and-funding-
aware, drawdown-bounded, stress-surviving edge FORWARD — or whether it stays
where the pre-registrations left it: shadow-only, because significance never
cleared (the frozen construction FAILED DSR at N=15 and N=3; see
``src.crypto.momentum_shadow`` and
``docs/experiment-crypto-edge-hunt-round2-2026-06-29.md``).

The Phase-H parallel unit is ``construction × fold × stress``:

1. :func:`compute_fold_scorecard` scores ONE cell — one contiguous forward
   window ("fold") of one construction under one stress scenario. Its headline is
   the annualized net Sharpe of the fold's realized net-return series, plus
   drawdown, turnover and funding contribution. A cell is scored fully
   independently of every other cell.

2. :func:`aggregate_construction_scorecard` is the DETERMINISTIC AGGREGATOR: it
   folds many per-cell scorecards for ONE construction into a construction
   verdict, refusing an incomplete or duplicated cell set. The admissibility
   verdict requires ALL of the roadmap J3 promotion conditions at once — adequate
   out-of-sample folds, a pooled OOS Sharpe that clears the bar with an
   across-fold t-statistic distinguishable from zero, a maximum drawdown within
   limits, robustness to dropping the single best fold (no dependence on one
   market episode), AND survival of every adverse stress scenario. A construction
   whose OOS Sharpe is positive but not significant is
   ``shadow_insufficient_significance`` — an honest null preserved as a
   first-class result, never rescued by an attractive in-sample or single-episode
   number.

This module reads nothing from disk on its own and imports nothing that can reach
a broker, the trade path, or the control state. Determinism is structural: pure
float arithmetic over the parsed rows, so two runs on identical partition bytes
produce identical metrics and identical verdicts — the property the local
metric-replay verification depends on.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from src.research.drawdown_convention import drawdown_fraction, drawdown_limit

# A fold needs at least this many realized periods to estimate a Sharpe; below it
# the fold reports ``sharpe=None`` (undefined, never a fabricated number).
MIN_PERIODS_FOR_SHARPE = 8

# A construction needs at least this many out-of-sample (forward) folds to
# conclude anything; below it the verdict is ``insufficient_folds`` (never forced).
MIN_OOS_FOLDS_FOR_VERDICT = 3

# Annualization factor for a daily net-return series (crypto trades 365 d/yr,
# matching the frozen construction's ``ANN`` in src.crypto.momentum_shadow).
ANN = 365.0

# Pre-registered admissibility bars for a construction's OUT-OF-SAMPLE edge.
# ALL must clear. The Sharpe/t-stat bars are deliberately stringent: the frozen
# construction's own pre-registration failed its significance criterion, so an
# honest forward scorecard must not admit it either.
SHARPE_BAR = 0.5          # pooled annualized OOS net Sharpe floor
SHARPE_TSTAT_BAR = 2.0    # across-fold t-statistic of per-fold Sharpes floor
MAX_DRAWDOWN_LIMIT = 0.25  # pooled OOS peak-to-trough drawdown ceiling (fraction)
STRESS_SHARPE_FLOOR = 0.0  # every adverse stress scenario must stay >= this OOS
DROP_ONE_SHARPE_FLOOR = 0.25  # pooled OOS Sharpe with the single best fold removed

# The baseline (un-stressed) scenario tag. Every construction must carry it; the
# significance verdict is computed on the baseline OOS folds, and the other
# scenarios are adverse stresses the construction must additionally survive.
BASELINE_STRESS = "baseline"

# Construction verdicts.
VERDICT_SIGNIFICANCE_CLEARS = "significance_clears"                    # admissible for quarantine
VERDICT_SHADOW_INSUFFICIENT_SIGNIFICANCE = "shadow_insufficient_significance"  # honest null — stays shadow
VERDICT_NO_EDGE = "no_edge"                                           # pooled OOS Sharpe <= 0
VERDICT_FAILS_STRESS = "fails_stress"                                 # significant but breaks under stress
VERDICT_INSUFFICIENT_FOLDS = "insufficient_folds"                     # too few OOS folds to conclude

# Per-period row keys the fold series carries.
_RETURN_KEY = "net_return"


def _finite_float(value: Any) -> float:
    """Coerce to a finite float or raise — a non-finite datum is never scored.

    ``json.loads`` accepts ``NaN``/``Infinity`` by default; a non-finite value
    would poison the Sharpe/drawdown math and later blow up Canonical-JSON
    encoding, so reject it at the source (fail-closed parse).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {type(value).__name__}: {value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"non-finite value {value!r}")
    return f


def _annualized_sharpe(returns: Sequence[float]) -> float | None:
    """Annualized Sharpe of a per-period net-return series, or ``None``.

    ``None`` (undefined, never 0.0) when there are too few periods or the series
    has zero variance — a degenerate series carries no measurable risk-adjusted
    edge and must not be reported as a real null.
    """
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0.0:
        return None
    return (mean / math.sqrt(var)) * math.sqrt(ANN)


def _max_drawdown(returns: Sequence[float]) -> float:
    """Peak-to-trough drawdown of the compounded equity curve (>= 0.0).

    Compounds the per-period net returns into an equity curve and returns the
    largest fractional decline from a running peak. A monotonically rising curve
    returns 0.0. This is the canonical positive-fraction convention -- see
    :mod:`src.research.drawdown_convention` -- validated at this boundary.
    """
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        if peak > 0.0:
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
    return drawdown_fraction(max_dd, source="crypto.momentum_scorecard._max_drawdown")


def compute_fold_scorecard(fold_id: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score ONE cell (one forward window of one construction under one stress).

    Every row must belong to this ``fold_id`` and carry the same
    ``construction``, ``stress``, ``fold_index`` and ``in_sample`` flag — a
    foreign or inconsistent row is a hard failure (fail-closed), because the
    cell's bytes were hash-verified against the signed dataset manifest before we
    got here. Rows are consumed IN THE ORDER SUPPLIED (the realized time order);
    a ``period_index`` field, when present, must be strictly increasing so a
    shuffled series cannot masquerade as an ordered one.

    Returns a deterministic dict. ``sharpe`` is ``None`` (absent signal, not a
    fabricated 0.0) when the cell has fewer than :data:`MIN_PERIODS_FOR_SHARPE`
    periods or a degenerate (zero-variance) return series.
    """
    if not rows:
        raise ValueError(f"fold {fold_id!r} has no rows")

    first = rows[0]
    construction = first.get("construction")
    stress = first.get("stress")
    fold_index = first.get("fold_index")
    in_sample = first.get("in_sample")
    if not isinstance(construction, str) or not construction:
        raise ValueError(f"fold {fold_id!r} row missing string 'construction'")
    if not isinstance(stress, str) or not stress:
        raise ValueError(f"fold {fold_id!r} row missing string 'stress'")
    if not isinstance(in_sample, bool):
        raise ValueError(f"fold {fold_id!r} row missing boolean 'in_sample'")

    returns: List[float] = []
    turnovers: List[float] = []
    funding: List[float] = []
    last_period: float | None = None
    for lineno, row in enumerate(rows, start=1):
        if row.get("fold_id") != fold_id:
            raise ValueError(f"fold {fold_id!r} row {lineno} carries fold_id {row.get('fold_id')!r}")
        if (row.get("construction") != construction or row.get("stress") != stress
                or row.get("fold_index") != fold_index):
            raise ValueError(f"fold {fold_id!r} row {lineno} has inconsistent construction/stress/fold_index")
        if row.get("in_sample") != in_sample:
            raise ValueError(f"fold {fold_id!r} row {lineno} has inconsistent in_sample")
        if _RETURN_KEY not in row:
            raise ValueError(f"fold {fold_id!r} row {lineno} missing {_RETURN_KEY!r}")
        if "period_index" in row:
            period = _finite_float(row["period_index"])
            if last_period is not None and period <= last_period:
                raise ValueError(f"fold {fold_id!r} row {lineno} period_index not strictly increasing")
            last_period = period
        returns.append(_finite_float(row[_RETURN_KEY]))
        turnovers.append(_finite_float(row.get("turnover", 0.0)))
        funding.append(_finite_float(row.get("funding_return", 0.0)))

    n = len(returns)
    sharpe = _annualized_sharpe(returns) if n >= MIN_PERIODS_FOR_SHARPE else None
    return {
        "fold_id": fold_id,
        "construction": construction,
        "stress": stress,
        "fold_index": fold_index,
        "in_sample": in_sample,
        "n_periods": n,
        "sharpe": sharpe,
        "mean_net_return": sum(returns) / n,
        "cumulative_return": math.prod(1.0 + r for r in returns) - 1.0,
        "max_drawdown": _max_drawdown(returns),
        "mean_turnover": sum(turnovers) / n,
        "funding_contribution": sum(funding),
    }


def _tstat(values: Sequence[float]) -> float | None:
    """One-sample t-statistic of ``values`` against 0, or ``None`` if undefined.

    Zero across-fold variance means the standard error is not estimable, so the
    t-statistic is reported as ``None`` (undefined) rather than ``±inf`` — a
    degenerate zero-variance sample cannot be *claimed* significant.
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    if var <= 0.0:
        return None
    return mean / math.sqrt(var / n)


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def aggregate_construction_scorecard(
    construction: str,
    fold_cards: Sequence[Mapping[str, Any]],
    expected_fold_ids: Sequence[str],
    *,
    min_oos_folds: int = MIN_OOS_FOLDS_FOR_VERDICT,
    sharpe_bar: float = SHARPE_BAR,
    sharpe_tstat_bar: float = SHARPE_TSTAT_BAR,
    max_drawdown_limit: float = MAX_DRAWDOWN_LIMIT,
    stress_sharpe_floor: float = STRESS_SHARPE_FLOOR,
    drop_one_sharpe_floor: float = DROP_ONE_SHARPE_FLOOR,
) -> Dict[str, Any]:
    """Deterministically fold per-cell scorecards into ONE construction verdict.

    This is the aggregator the roadmap requires to *refuse missing or duplicate
    folds*: the produced cell set must equal the declared ``expected_fold_ids``
    exactly. A missing cell or a cell submitted twice raises ``ValueError`` — an
    incomplete aggregation can never be silently scored as complete.

    The verdict encodes the full roadmap J3 promotion gate at once. The baseline
    (un-stressed) out-of-sample folds drive the significance test; the other
    scenarios are adverse stresses (funding/basis/liquidation) the construction
    must additionally survive. ``significance_clears`` (admissible) requires ALL
    of: enough baseline OOS folds; pooled OOS Sharpe >= ``sharpe_bar`` with an
    across-fold t-statistic >= ``sharpe_tstat_bar``; pooled OOS drawdown within
    ``max_drawdown_limit``; robustness to dropping the single best fold
    (>= ``drop_one_sharpe_floor``, so the edge does not rest on one market
    episode); and every adverse stress scenario's mean OOS Sharpe >=
    ``stress_sharpe_floor``. Anything short of significance is
    ``shadow_insufficient_significance`` — the honest shadow-only null.

    All bars default to the module constants but are accepted as arguments so the
    caller's SIGNED, digest-bound bars are the ones actually enforced.
    """
    expected = list(expected_fold_ids)
    if len(set(expected)) != len(expected):
        raise ValueError(f"construction {construction!r} declared duplicate expected fold ids")

    produced = [card["fold_id"] for card in fold_cards]
    if len(set(produced)) != len(produced):
        raise ValueError(f"construction {construction!r} received duplicate fold scorecards: {sorted(produced)}")
    if set(produced) != set(expected):
        missing = sorted(set(expected) - set(produced))
        extra = sorted(set(produced) - set(expected))
        raise ValueError(
            f"construction {construction!r} fold set mismatch: missing={missing}, unexpected={extra}"
        )
    for card in fold_cards:
        if card.get("construction") != construction:
            raise ValueError(
                f"construction {construction!r} received a fold from construction {card.get('construction')!r}"
            )

    baseline_oos = [
        c for c in fold_cards
        if c["stress"] == BASELINE_STRESS and c["in_sample"] is False and c.get("sharpe") is not None
    ]
    baseline_is = [
        c for c in fold_cards
        if c["stress"] == BASELINE_STRESS and c["in_sample"] is True and c.get("sharpe") is not None
    ]
    baseline_oos_sharpes = [float(c["sharpe"]) for c in baseline_oos]

    pooled_oos_sharpe = _mean(baseline_oos_sharpes)
    pooled_is_sharpe = _mean([float(c["sharpe"]) for c in baseline_is])
    sharpe_tstat = _tstat(baseline_oos_sharpes)
    # Drawdown is a tail-risk observation, not a Sharpe-derived statistic. A
    # short catastrophic fold can have undefined Sharpe and must still count.
    baseline_oos_all = [
        c
        for c in fold_cards
        if c["stress"] == BASELINE_STRESS and c["in_sample"] is False
    ]
    pooled_oos_drawdown = max(
        (float(c["max_drawdown"]) for c in baseline_oos_all), default=None
    )

    # Drop-one robustness: pooled OOS Sharpe with the single best baseline fold
    # removed (no dependence on one market episode). ``None`` when < 2 folds.
    if len(baseline_oos_sharpes) >= 2:
        worst_kept = sorted(baseline_oos_sharpes)[:-1]
        drop_one_sharpe = sum(worst_kept) / len(worst_kept)
    else:
        drop_one_sharpe = None

    # Per-stress survival: mean OOS Sharpe of each adverse scenario.
    stress_scenarios = sorted({c["stress"] for c in fold_cards if c["stress"] != BASELINE_STRESS})
    stress_oos_sharpe: Dict[str, float | None] = {}
    for scenario in stress_scenarios:
        all_cells = [
            c for c in fold_cards
            if c["stress"] == scenario and c["in_sample"] is False
        ]
        # A declared OOS stress cell with undefined Sharpe is missing usable
        # stress evidence, not a cell the aggregator may silently discard.
        if not all_cells or any(c.get("sharpe") is None for c in all_cells):
            stress_oos_sharpe[scenario] = None
        else:
            stress_oos_sharpe[scenario] = _mean(
                [float(c["sharpe"]) for c in all_cells]
            )

    decision = _construction_decision(
        n_oos=len(baseline_oos),
        pooled_oos_sharpe=pooled_oos_sharpe,
        sharpe_tstat=sharpe_tstat,
        pooled_oos_drawdown=pooled_oos_drawdown,
        drop_one_sharpe=drop_one_sharpe,
        stress_oos_sharpe=stress_oos_sharpe,
        min_oos_folds=min_oos_folds,
        sharpe_bar=sharpe_bar,
        sharpe_tstat_bar=sharpe_tstat_bar,
        max_drawdown_limit=max_drawdown_limit,
        stress_sharpe_floor=stress_sharpe_floor,
        drop_one_sharpe_floor=drop_one_sharpe_floor,
    )

    return {
        "construction": construction,
        "n_folds": len(fold_cards),
        "n_baseline_oos_folds": len(baseline_oos),
        "n_baseline_is_folds": len(baseline_is),
        "pooled_oos_sharpe": pooled_oos_sharpe,
        "pooled_is_sharpe": pooled_is_sharpe,
        "oos_sharpe_tstat": sharpe_tstat,
        "pooled_oos_max_drawdown": pooled_oos_drawdown,
        "drop_one_oos_sharpe": drop_one_sharpe,
        "stress_scenarios": stress_scenarios,
        "stress_oos_sharpe": stress_oos_sharpe,
        "sharpe_bar": sharpe_bar,
        "sharpe_tstat_bar": sharpe_tstat_bar,
        "max_drawdown_limit": max_drawdown_limit,
        "stress_sharpe_floor": stress_sharpe_floor,
        "drop_one_sharpe_floor": drop_one_sharpe_floor,
        "min_oos_folds": min_oos_folds,
        "decision": decision,
    }


def _construction_decision(
    *,
    n_oos: int,
    pooled_oos_sharpe: float | None,
    sharpe_tstat: float | None,
    pooled_oos_drawdown: float | None,
    drop_one_sharpe: float | None,
    stress_oos_sharpe: Mapping[str, float | None],
    min_oos_folds: int,
    sharpe_bar: float,
    sharpe_tstat_bar: float,
    max_drawdown_limit: float,
    stress_sharpe_floor: float,
    drop_one_sharpe_floor: float,
) -> Dict[str, Any]:
    """The frozen construction verdict from the pooled baseline OOS statistics.

    Fail-closed at every undefined statistic: an undefined t-statistic (zero
    across-fold variance) or an undefined drop-one/stress reading cannot be
    *asserted* to clear its bar, so it falls through to the honest-null verdict
    rather than being admitted. The gate never rescues a construction on an
    in-sample or single-episode number.
    """
    if n_oos < min_oos_folds or pooled_oos_sharpe is None:
        return {
            "verdict": VERDICT_INSUFFICIENT_FOLDS,
            "reason": (
                f"only {n_oos} baseline out-of-sample fold(s) with a defined Sharpe "
                f"(< {min_oos_folds}) — refusing to force a verdict"
            ),
        }
    if pooled_oos_sharpe <= 0.0:
        return {
            "verdict": VERDICT_NO_EDGE,
            "reason": f"pooled OOS Sharpe {pooled_oos_sharpe:.4f} <= 0 — no forward edge",
        }

    significant = (
        sharpe_tstat is not None
        and pooled_oos_sharpe >= sharpe_bar
        and sharpe_tstat >= sharpe_tstat_bar
        and drop_one_sharpe is not None
        and drop_one_sharpe >= drop_one_sharpe_floor
        and pooled_oos_drawdown is not None
        and drawdown_fraction(
            pooled_oos_drawdown, source="crypto.momentum_scorecard.verdict"
        )
        <= drawdown_limit(
            max_drawdown_limit, source="crypto.momentum_scorecard.verdict"
        )
    )
    if not significant:
        tstat_str = "undefined" if sharpe_tstat is None else f"{sharpe_tstat:.2f}"
        return {
            "verdict": VERDICT_SHADOW_INSUFFICIENT_SIGNIFICANCE,
            "reason": (
                f"pooled OOS Sharpe {pooled_oos_sharpe:.4f} (t={tstat_str}) is positive but does not "
                f"clear Sharpe>={sharpe_bar} AND t>={sharpe_tstat_bar} AND drop-one>={drop_one_sharpe_floor} "
                f"AND drawdown<={max_drawdown_limit} — stays shadow (an honest null, not an edge)"
            ),
        }

    # Significant on baseline — but roadmap J3 forbids promotion on a headline
    # that breaks under stress. Every declared adverse scenario must survive.
    failed_stress = [
        scenario for scenario, sharpe in stress_oos_sharpe.items()
        if sharpe is None or sharpe < stress_sharpe_floor
    ]
    if failed_stress:
        return {
            "verdict": VERDICT_FAILS_STRESS,
            "reason": (
                f"baseline significance clears but adverse stress scenario(s) "
                f"{sorted(failed_stress)} fall below Sharpe {stress_sharpe_floor} — "
                f"depends on stress-free conditions"
            ),
        }
    return {"verdict": VERDICT_SIGNIFICANCE_CLEARS, "reason": None}


__all__ = [
    "MIN_PERIODS_FOR_SHARPE",
    "MIN_OOS_FOLDS_FOR_VERDICT",
    "ANN",
    "SHARPE_BAR",
    "SHARPE_TSTAT_BAR",
    "MAX_DRAWDOWN_LIMIT",
    "STRESS_SHARPE_FLOOR",
    "DROP_ONE_SHARPE_FLOOR",
    "BASELINE_STRESS",
    "VERDICT_SIGNIFICANCE_CLEARS",
    "VERDICT_SHADOW_INSUFFICIENT_SIGNIFICANCE",
    "VERDICT_NO_EDGE",
    "VERDICT_FAILS_STRESS",
    "VERDICT_INSUFFICIENT_FOLDS",
    "compute_fold_scorecard",
    "aggregate_construction_scorecard",
]
