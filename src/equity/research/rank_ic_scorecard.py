"""Deterministic cross-sectional rank-IC scorecard for the equity-research lane.

This is the equity-research analogue of :mod:`src.hedge.hedge_scorecard`: a pure,
offline, RNG-free, network-free scorer that turns *already-realized* per-name
research rows into an honest verdict on whether a research hypothesis actually
predicts forward cross-sectional returns — and, crucially, whether that edge
*survives* as the universe widens (the Track B finding: a curated top-N rank-IC
of +0.16 fades to a non-significant +0.09 once the survivorship-corrected wide
universe is scored).

Two levels, both deterministic pure-Python float arithmetic:

1. :func:`compute_fold_scorecard` scores ONE fold — a single evaluation window
   over one universe snapshot. Its headline is the *rank information coefficient*
   (Spearman correlation of the pre-registered composite score against the
   realized forward return across the fold's names). A fold is the Phase-H
   parallel unit; each is scored fully independently of every other fold.

2. :func:`aggregate_tier_scorecard` is the DETERMINISTIC AGGREGATOR: it folds
   many per-fold scorecards for one universe tier into a tier verdict, pooling
   the in-sample and out-of-sample rank-IC, computing the across-fold IC
   t-statistic, and refusing an incomplete or duplicated fold set. A tier whose
   out-of-sample rank-IC fades to statistical insignificance is
   ``signal_fades_with_breadth`` — an honest null preserved as a first-class
   result, never rescued by a strong in-sample or curated number.

The composite is the FROZEN :data:`~src.equity.research.contracts.PRIMARY_WEIGHTS`
(reused, never redefined here) so the research book cannot silently diverge from
the pre-registration. This module reads nothing from disk on its own and imports
nothing that can reach a broker, the trade path, or the control state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from src.equity.research.contracts import PRIMARY_WEIGHTS

# A fold needs at least this many priced, scored names to form an honest
# cross-section; below it a rank-IC is noise, so the fold reports ``rank_ic=None``
# rather than a fabricated correlation.
MIN_NAMES_FOR_IC = 5

# A tier needs at least this many out-of-sample folds to conclude the edge
# persists; below it the tier verdict is ``insufficient_folds`` (never forced).
MIN_OOS_FOLDS_FOR_VERDICT = 3

# The pre-registered admissibility bars for a tier's OUT-OF-SAMPLE edge. Both
# must clear: a positive mean OOS rank-IC of at least ``IC_BAR`` AND an
# across-fold t-statistic of at least ``IC_TSTAT_BAR`` (i.e. the edge is not only
# positive on average but statistically distinguishable from zero across folds).
IC_BAR = 0.03
IC_TSTAT_BAR = 2.0

# Quintile fraction for the long/short spread diagnostic (top/bottom 20%).
QUINTILE = 5

# Tier verdicts.
VERDICT_SIGNAL_PERSISTS = "signal_persists_oos"          # admissible for quarantine
VERDICT_FADES_WITH_BREADTH = "signal_fades_with_breadth"  # honest null — positive but not significant OOS
VERDICT_NO_SIGNAL = "no_signal"                          # rank-IC ~ 0 everywhere
VERDICT_INSUFFICIENT_FOLDS = "insufficient_folds"         # too few OOS folds to conclude

# Per-name row keys that carry the composite inputs. Reusing the composite
# formula from the frozen contracts guarantees the research book cannot diverge.
_COMPOSITE_INPUT_KEYS = ("fundamental_quality", "accounting_red_flags", "forward_outlook")


def _finite_float(value: Any) -> float:
    """Coerce to a finite float or raise — a non-finite datum is never scored.

    ``json.loads`` accepts ``NaN``/``Infinity`` by default; a non-finite value
    would poison the correlation math and later blow up Canonical-JSON encoding,
    so reject it at the source (mirrors the hedge scorecard's fail-closed parse).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {type(value).__name__}: {value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"non-finite value {value!r}")
    return f


def _composite(row: Mapping[str, Any]) -> float:
    """The frozen pre-registered composite of one row's score fields.

    Uses :data:`PRIMARY_WEIGHTS` verbatim — the same formula
    :meth:`ResearchScore.composite` applies — so a fold scored here is scored
    exactly as the deployable research book would score it.
    """
    total = 0.0
    for key in _COMPOSITE_INPUT_KEYS:
        if key not in row:
            raise ValueError(f"row missing composite input {key!r}")
        total += PRIMARY_WEIGHTS.get(key, 0.0) * _finite_float(row[key])
    return total


def _average_ranks(values: Sequence[float]) -> List[float]:
    """Fractional (tie-averaged) ranks of ``values``, deterministic.

    Ties receive the average of the ranks they span, so the rank vector is a
    pure function of the multiset of values and their order — no RNG, no
    arbitrary tie-break. This is the standard Spearman rank transform.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0  # 0-based average position of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` when a series has zero variance.

    Zero variance (e.g. every composite identical) means the correlation is
    undefined — reported as ``None`` (no signal to measure), never as 0.0 which
    would read as a real, measured null.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation (Pearson on tie-averaged ranks)."""
    if len(xs) != len(ys):
        raise ValueError("spearman inputs must be equal length")
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _quintile_spread(composites: Sequence[float], forwards: Sequence[float]) -> float | None:
    """Top-quintile minus bottom-quintile mean forward return (Q5 - Q1).

    A tradeable-book proxy: rank names by composite, take the top and bottom
    20%, and difference their mean realized forward returns. ``None`` when the
    cross-section is too thin to cut a quintile.
    """
    n = len(composites)
    q = n // QUINTILE
    if q < 1:
        return None
    order = sorted(range(n), key=lambda i: (composites[i], i))
    bottom = order[:q]
    top = order[-q:]
    top_mean = sum(forwards[i] for i in top) / q
    bottom_mean = sum(forwards[i] for i in bottom) / q
    return top_mean - bottom_mean


def compute_fold_scorecard(fold_id: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score ONE fold (one window over one universe snapshot) into a rank-IC card.

    Every row must belong to this ``fold_id`` and carry the same ``tier``,
    ``fold_index``, ``in_sample`` flag and window bounds — a foreign or
    inconsistent row is a hard failure (fail-closed), because the fold's bytes
    were hash-verified against the signed dataset manifest before we got here.

    Returns a deterministic dict. ``rank_ic`` is ``None`` (absent signal, not a
    fabricated 0.0) when the fold has fewer than :data:`MIN_NAMES_FOR_IC` names
    or a degenerate (zero-variance) composite/return cross-section.
    """
    if not rows:
        raise ValueError(f"fold {fold_id!r} has no rows")

    first = rows[0]
    tier = first.get("tier")
    fold_index = first.get("fold_index")
    in_sample = first.get("in_sample")
    window = (first.get("window_start"), first.get("window_end"))
    if not isinstance(in_sample, bool):
        raise ValueError(f"fold {fold_id!r} row missing boolean 'in_sample'")

    composites: List[float] = []
    forwards: List[float] = []
    tickers: List[str] = []
    for lineno, row in enumerate(rows, start=1):
        if row.get("fold_id") != fold_id:
            raise ValueError(f"fold {fold_id!r} row {lineno} carries fold_id {row.get('fold_id')!r}")
        if row.get("tier") != tier or row.get("fold_index") != fold_index:
            raise ValueError(f"fold {fold_id!r} row {lineno} has inconsistent tier/fold_index")
        if row.get("in_sample") != in_sample or (row.get("window_start"), row.get("window_end")) != window:
            raise ValueError(f"fold {fold_id!r} row {lineno} has inconsistent window/in_sample")
        ticker = row.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError(f"fold {fold_id!r} row {lineno} missing 'ticker'")
        if "forward_return" not in row:
            raise ValueError(f"fold {fold_id!r} row {lineno} missing 'forward_return'")
        composites.append(_composite(row))
        forwards.append(_finite_float(row["forward_return"]))
        tickers.append(ticker)

    if len(set(tickers)) != len(tickers):
        raise ValueError(f"fold {fold_id!r} has duplicate tickers in its cross-section")

    n_names = len(tickers)
    rank_ic = _spearman(composites, forwards) if n_names >= MIN_NAMES_FOR_IC else None
    q5_q1 = _quintile_spread(composites, forwards) if n_names >= MIN_NAMES_FOR_IC else None
    max_weight = 1.0 / n_names if n_names else None  # equal-weight book concentration proxy

    return {
        "fold_id": fold_id,
        "tier": tier,
        "fold_index": fold_index,
        "in_sample": in_sample,
        "window": list(window),
        "n_names": n_names,
        "rank_ic": rank_ic,
        "q5_q1_spread": q5_q1,
        "mean_forward_return": sum(forwards) / n_names,
        "concentration_max_weight": max_weight,
    }


def _tstat(values: Sequence[float]) -> float | None:
    """One-sample t-statistic of ``values`` against 0, or ``None`` if undefined.

    Zero across-fold variance means the standard error is not estimable, so the
    t-statistic is reported as ``None`` (undefined) rather than ``±inf`` — a
    degenerate zero-variance sample cannot be *claimed* significant, and a
    non-finite value would in any case be rejected by Canonical-JSON encoding.
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    if var <= 0.0:
        return None
    return mean / math.sqrt(var / n)


def _weighted_mean(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Breadth-weighted mean of (value, weight) pairs, or ``None`` if no weight."""
    total_w = sum(w for _, w in pairs)
    if total_w <= 0.0:
        return None
    return sum(v * w for v, w in pairs) / total_w


def aggregate_tier_scorecard(
    tier: str,
    fold_cards: Sequence[Mapping[str, Any]],
    expected_fold_ids: Sequence[str],
    *,
    min_oos_folds: int = MIN_OOS_FOLDS_FOR_VERDICT,
    ic_bar: float = IC_BAR,
    ic_tstat_bar: float = IC_TSTAT_BAR,
) -> Dict[str, Any]:
    """Deterministically fold per-fold scorecards into ONE universe-tier verdict.

    This is the aggregator the roadmap requires to *refuse missing or duplicate
    folds*: the produced fold set must equal the declared ``expected_fold_ids``
    exactly. A missing fold or a fold submitted twice raises ``ValueError`` — an
    incomplete aggregation can never be silently scored as complete.

    Out-of-sample folds (``in_sample=False``, i.e. after the research model's
    training cutoff) drive the admissibility verdict; in-sample folds are pooled
    for display only (they are the *expected-contaminated* arm). The tier is
    ``signal_persists_oos`` (admissible) only when its breadth-weighted mean OOS
    rank-IC clears ``ic_bar`` AND the across-fold t-statistic clears
    ``ic_tstat_bar``; a positive-but-insignificant OOS edge is
    ``signal_fades_with_breadth`` — the honest Track B null.

    ``min_oos_folds`` / ``ic_bar`` / ``ic_tstat_bar`` default to the module
    constants but are accepted as arguments so the caller's SIGNED, digest-bound
    bars are the ones actually enforced — a value that is signed into the job
    manifest but not enforced here would be a dishonest attestation.
    """
    expected = list(expected_fold_ids)
    if len(set(expected)) != len(expected):
        raise ValueError(f"tier {tier!r} declared duplicate expected fold ids")

    produced = [card["fold_id"] for card in fold_cards]
    if len(set(produced)) != len(produced):
        raise ValueError(f"tier {tier!r} received duplicate fold scorecards: {sorted(produced)}")
    if set(produced) != set(expected):
        missing = sorted(set(expected) - set(produced))
        extra = sorted(set(produced) - set(expected))
        raise ValueError(
            f"tier {tier!r} fold set mismatch: missing={missing}, unexpected={extra}"
        )
    for card in fold_cards:
        if card.get("tier") != tier:
            raise ValueError(f"tier {tier!r} received a fold from tier {card.get('tier')!r}")

    oos = [c for c in fold_cards if c["in_sample"] is False and c.get("rank_ic") is not None]
    ins = [c for c in fold_cards if c["in_sample"] is True and c.get("rank_ic") is not None]

    oos_ic_values = [float(c["rank_ic"]) for c in oos]
    pooled_oos_ic = _weighted_mean([(float(c["rank_ic"]), float(c["n_names"])) for c in oos])
    pooled_is_ic = _weighted_mean([(float(c["rank_ic"]), float(c["n_names"])) for c in ins])
    ic_tstat = _tstat(oos_ic_values)
    breadth = (sum(float(c["n_names"]) for c in oos) / len(oos)) if oos else None
    q5_q1_values = [float(c["q5_q1_spread"]) for c in oos if c.get("q5_q1_spread") is not None]
    mean_oos_q5_q1 = (sum(q5_q1_values) / len(q5_q1_values)) if q5_q1_values else None

    decision = _tier_decision(
        len(oos), pooled_oos_ic, ic_tstat,
        min_oos_folds=min_oos_folds, ic_bar=ic_bar, ic_tstat_bar=ic_tstat_bar,
    )

    return {
        "tier": tier,
        "n_folds": len(fold_cards),
        "n_oos_folds": len(oos),
        "n_is_folds": len(ins),
        "breadth_mean_names": breadth,
        "pooled_oos_rank_ic": pooled_oos_ic,
        "pooled_is_rank_ic": pooled_is_ic,
        "oos_ic_tstat": ic_tstat,
        "mean_oos_q5_q1_spread": mean_oos_q5_q1,
        "ic_bar": ic_bar,
        "ic_tstat_bar": ic_tstat_bar,
        "min_oos_folds": min_oos_folds,
        "decision": decision,
    }


def _tier_decision(
    n_oos: int,
    pooled_oos_ic: float | None,
    ic_tstat: float | None,
    *,
    min_oos_folds: int = MIN_OOS_FOLDS_FOR_VERDICT,
    ic_bar: float = IC_BAR,
    ic_tstat_bar: float = IC_TSTAT_BAR,
) -> Dict[str, Any]:
    """The frozen three-way tier verdict from the pooled OOS statistics.

    An undefined across-fold t-statistic (zero variance) is treated as *not
    significant* (fail-closed): a signal we cannot estimate a standard error for
    cannot be asserted to persist, so it falls through to ``no_signal`` /
    ``signal_fades_with_breadth`` rather than being admitted.
    """
    if n_oos < min_oos_folds or pooled_oos_ic is None:
        return {
            "verdict": VERDICT_INSUFFICIENT_FOLDS,
            "basis": "oos",
            "reason": (
                f"only {n_oos} out-of-sample fold(s) with a defined rank-IC "
                f"(< {min_oos_folds}) — refusing to force a verdict"
            ),
        }
    persists = ic_tstat is not None and pooled_oos_ic >= ic_bar and ic_tstat >= ic_tstat_bar
    if persists:
        return {"verdict": VERDICT_SIGNAL_PERSISTS, "basis": "oos", "reason": None}
    if pooled_oos_ic <= 0.0:
        return {
            "verdict": VERDICT_NO_SIGNAL,
            "basis": "oos",
            "reason": f"pooled OOS rank-IC {pooled_oos_ic:.4f} <= 0 — no cross-sectional edge",
        }
    tstat_str = "undefined" if ic_tstat is None else f"{ic_tstat:.2f}"
    return {
        "verdict": VERDICT_FADES_WITH_BREADTH,
        "basis": "oos",
        "reason": (
            f"pooled OOS rank-IC {pooled_oos_ic:.4f} (t={tstat_str}) is positive but "
            f"does not clear IC>={ic_bar} AND t>={ic_tstat_bar} — an honest null, not an edge"
        ),
    }


__all__ = [
    "MIN_NAMES_FOR_IC",
    "MIN_OOS_FOLDS_FOR_VERDICT",
    "IC_BAR",
    "IC_TSTAT_BAR",
    "VERDICT_SIGNAL_PERSISTS",
    "VERDICT_FADES_WITH_BREADTH",
    "VERDICT_NO_SIGNAL",
    "VERDICT_INSUFFICIENT_FOLDS",
    "compute_fold_scorecard",
    "aggregate_tier_scorecard",
]
