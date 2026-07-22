"""Deterministic, authority-free portfolio-book allocation and combined-book gate.

Given a set of already-verified per-sleeve daily net-return partitions (one
partition per contributing lane/strategy sleeve — e.g. risk_target, hedge_eval,
equity_research, crypto_momentum, crypto_carry — each already independently
QUARANTINED by its own evidence slice), this module:

1. parses each sleeve partition and requires an identical trading calendar
   across sleeves (refuses misaligned or short calendars, same "refuse rather
   than silently patch" discipline as the equity-research fold aggregator),
2. splits the calendar at a frozen chronological cutoff into an ESTIMATION
   window (used only to derive allocation weights) and an OOS window (used
   only to score the resulting book) — weights are NEVER fit on the window
   they are scored on,
3. computes four allocation methods on the estimation window as named in
   roadmap §14 — inverse-volatility, equal-risk-contribution, correlation-
   penalized, and hierarchical risk parity (HRP) — and PRE-REGISTERS HRP as
   the one policy that is actually applied (mirrors the existing equity-only
   combiner's pre-registered choice in ``src.equity.sleeve_combiner``); the
   other three are retained as evidence-package diagnostics, not silently
   dropped,
4. resolves lane capacity / min-max allocation / cash reserve via iterative
   waterfilling, then applies tail-correlation damping and a drawdown-budget
   scale-down (never redistributed onto other sleeves — freed weight reverts
   to cash, the conservative choice),
5. applies the FINAL constrained weights STATICALLY over the OOS window (no
   look-ahead, no in-sample rebalancing tuned to the scored window) and scores
   the resulting book against six gates.

This module imports only the evidence contracts and the standard library —
never the broker path, the trade execution/gate path, or the state engine. It
is handed its data as bytes and has no repository-root or control-file
parameter, so it can neither read ``.claude/state.json`` nor reach a broker
credential. It also NEVER re-derives capital-activation authority: a passing
book gate here means the combined book is eligible for local QUARANTINE, the
same and only terminal state every other AXIOM evidence slice reaches through
this authority-free path. Promotion to CHAMPION requires the separate,
human-operated promotion service (Phase L) — no lane, and no book, becomes
capital-active from a standalone gate.

Determinism is structural: every computation below is closed-form or a
fixed-iteration-count deterministic algorithm over parsed floats (no RNG, no
external solver, no environment-dependent library) — two runs on identical
partition bytes and identical params produce byte-identical artifacts, the
property the local metric-replay verification depends on. Correlation-based
clustering (HRP) is reimplemented here in pure Python rather than reusing
``src.equity.hrp`` (which depends on numpy/pandas/scipy) — sleeve counts are
small (single digits), the reimplementation is a direct O(n^3) transliteration
of the same López de Prado (2016) single-linkage / quasi-diagonalization /
recursive-bisection algorithm, and it keeps this replay path free of any
third-party numerical library whose version could drift between the producer
and the local importer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import GateResult, GateStatus

from .models import BookResult

METRICS = (
    "oos_sharpe",
    "oos_max_drawdown",
    "oos_total_return",
    "oos_ann_vol",
    "n_sleeves",
    "cash_reserve_frac",
    "max_observed_tail_correlation",
    "n_drawdown_budget_violations",
)
REQUIRED_ROW_FIELDS = ("date", "net_return", "gross_exposure", "turnover")
TRADING_DAYS_PER_YEAR = 252.0
_EPS = 1e-12


@dataclass(frozen=True)
class EvaluationParams:
    oos_start: str = "2026-01-01"
    min_est_days: int = 60
    min_oos_days: int = 20
    cash_reserve_frac: float = 0.10
    max_allocation: float = 0.40
    min_allocation: float = 0.0
    min_sleeve_vol: float = 1e-6
    lane_capacity: Mapping[str, float] = field(default_factory=dict)
    tail_corr_threshold: float = 0.75
    tail_quantile: float = 0.10
    drawdown_budget: float = 0.15
    max_book_drawdown: float = 0.25
    min_book_sharpe: float = 0.30
    erc_max_iterations: int = 500
    erc_tolerance: float = 1e-10
    capacity_max_rounds: int = 50
    replay_tolerance: float = 1e-9


def _parse_date(value: object, field_name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid ISO-8601 date: {value!r}") from exc


def _rows(partitions: Mapping[str, bytes]) -> dict[str, list[dict]]:
    if len(partitions) < 2:
        raise ValueError("a portfolio book requires at least 2 sleeve partitions")
    import json

    sleeves: dict[str, list[dict]] = {}
    for sleeve_id, data in sorted(partitions.items()):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"sleeve {sleeve_id!r} partition is not UTF-8") from exc
        rows: list[dict] = []
        seen_dates: set[date] = set()
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"sleeve {sleeve_id!r} line {line_number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"sleeve {sleeve_id!r} line {line_number}: each row must be a JSON object")
            missing = sorted(set(REQUIRED_ROW_FIELDS) - set(row))
            if missing:
                raise ValueError(f"sleeve {sleeve_id!r} row missing fields {missing}")
            row_date = _parse_date(row["date"], f"sleeve {sleeve_id!r} date")
            if row_date in seen_dates:
                raise ValueError(f"sleeve {sleeve_id!r} has a duplicate date {row_date.isoformat()}")
            seen_dates.add(row_date)
            numeric: dict[str, float] = {}
            for name in ("net_return", "gross_exposure", "turnover"):
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"sleeve {sleeve_id!r} has non-numeric {name}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"sleeve {sleeve_id!r} has non-finite {name}")
                numeric[name] = value
            if numeric["gross_exposure"] < 0 or numeric["turnover"] < 0:
                raise ValueError(f"sleeve {sleeve_id!r} has a negative exposure or turnover")
            rows.append({"_date": row_date, **numeric})
        if not rows:
            raise ValueError(f"sleeve {sleeve_id!r} partition contains no records")
        rows.sort(key=lambda row: row["_date"])
        sleeves[sleeve_id] = rows
    return sleeves


def _aligned_calendar(sleeves: Mapping[str, list[dict]]) -> list[date]:
    calendars = {sleeve_id: tuple(row["_date"] for row in rows) for sleeve_id, rows in sleeves.items()}
    reference_id, reference = next(iter(sorted(calendars.items())))
    for sleeve_id, calendar in sorted(calendars.items()):
        if calendar != reference:
            raise ValueError(
                f"sleeve {sleeve_id!r} calendar does not match sleeve {reference_id!r}; "
                "the portfolio book requires an identical aligned trading calendar across sleeves"
            )
    return list(reference)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_stdev(values: list[float]) -> float:
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _covariance_matrix(returns: list[list[float]]) -> list[list[float]]:
    n = len(returns)
    means = [_mean(series) for series in returns]
    t = len(returns[0])
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cov[i][j] = sum((returns[i][k] - means[i]) * (returns[j][k] - means[j]) for k in range(t)) / t
    return cov


def _correlation_matrix(cov: list[list[float]]) -> list[list[float]]:
    n = len(cov)
    vols = [math.sqrt(max(cov[i][i], 0.0)) for i in range(n)]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            denom = max(vols[i] * vols[j], _EPS)
            corr[i][j] = 1.0 if i == j else max(-1.0, min(1.0, cov[i][j] / denom))
    return corr


def _inverse_vol_weights(vols: list[float]) -> list[float]:
    inv = [1.0 / max(v, _EPS) for v in vols]
    s = sum(inv)
    return [x / s for x in inv]


def _erc_weights(cov: list[list[float]], params: EvaluationParams) -> list[float]:
    n = len(cov)
    w = [1.0 / n] * n
    for _ in range(params.erc_max_iterations):
        sw = [sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
        rc = [w[i] * sw[i] for i in range(n)]
        total_var = sum(rc)
        if total_var <= _EPS:
            return [1.0 / n] * n
        target = total_var / n
        new_w = [
            w[i] * math.sqrt(target / rc[i]) if rc[i] > _EPS else w[i]
            for i in range(n)
        ]
        s = sum(new_w)
        new_w = [x / s for x in new_w] if s > _EPS else [1.0 / n] * n
        diff = max(abs(new_w[i] - w[i]) for i in range(n))
        w = new_w
        if diff < params.erc_tolerance:
            break
    return w


def _corr_penalized_weights(vols: list[float], corr: list[list[float]]) -> list[float]:
    n = len(vols)
    raw = []
    for i in range(n):
        others = [corr[i][j] for j in range(n) if j != i]
        avg_corr = _mean(others) if others else 0.0
        raw.append((1.0 / max(vols[i], _EPS)) * max(0.05, 1.0 - avg_corr))
    s = sum(raw)
    return [x / s for x in raw] if s > _EPS else [1.0 / n] * n


def _single_linkage(dist: list[list[float]], n: int) -> list[tuple[int, int, float]]:
    """Deterministic single-linkage agglomerative clustering. Returns n-1 merges
    as (id_a, id_b, distance); merged-cluster ids are offset by n, matching the
    scipy ``linkage`` convention consumed by ``_quasi_diag``."""
    clusters: dict[int, set[int]] = {i: {i} for i in range(n)}
    active = list(range(n))
    merges: list[tuple[int, int, float]] = []
    next_id = n

    def cluster_distance(a: set[int], b: set[int]) -> float:
        return min(dist[i][j] for i in a for j in b)

    while len(active) > 1:
        best: tuple[float, int, int] | None = None
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a, b = active[ii], active[jj]
                d = cluster_distance(clusters[a], clusters[b])
                lo, hi = (a, b) if a < b else (b, a)
                if best is None or d < best[0] - 1e-15 or (abs(d - best[0]) <= 1e-15 and (lo, hi) < (best[1], best[2])):
                    best = (d, lo, hi)
        d, a, b = best
        merges.append((a, b, d))
        clusters[next_id] = clusters[a] | clusters[b]
        del clusters[a]
        del clusters[b]
        active.remove(a)
        active.remove(b)
        active.append(next_id)
        next_id += 1
    return merges


def _quasi_diag(merges: list[tuple[int, int, float]], n: int) -> list[int]:
    if n == 1:
        return [0]
    order = [merges[-1][0], merges[-1][1]]
    while any(x >= n for x in order):
        expanded: list[int] = []
        for x in order:
            if x >= n:
                a, b, _ = merges[x - n]
                expanded.extend([a, b])
            else:
                expanded.append(x)
        order = expanded
    return order


def _cluster_var(cov: list[list[float]], items: list[int]) -> float:
    diag = [cov[i][i] for i in items]
    ivp = [1.0 / d if d > _EPS else 0.0 for d in diag]
    s = sum(ivp)
    ivp = [x / s for x in ivp] if s > _EPS else [1.0 / len(items)] * len(items)
    total = 0.0
    for a_idx, a in enumerate(items):
        row_sum = sum(cov[a][b] * ivp[b_idx] for b_idx, b in enumerate(items))
        total += ivp[a_idx] * row_sum
    return total


def _recursive_bisection(cov: list[list[float]], order: list[int]) -> dict[int, float]:
    w = {i: 1.0 for i in order}
    clusters = [list(order)]
    while clusters:
        new_clusters: list[list[int]] = []
        for c in clusters:
            if len(c) > 1:
                mid = len(c) // 2
                new_clusters.append(c[:mid])
                new_clusters.append(c[mid:])
        clusters = new_clusters
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1) if (v0 + v1) > _EPS else 0.5
            for k in c0:
                w[k] *= alpha
            for k in c1:
                w[k] *= (1.0 - alpha)
    return w


def _hrp_weights(cov: list[list[float]], corr: list[list[float]]) -> list[float]:
    n = len(cov)
    if n <= 2:
        return _inverse_vol_weights([math.sqrt(max(cov[i][i], 0.0)) for i in range(n)])
    dist = [[math.sqrt(max((1.0 - corr[i][j]) / 2.0, 0.0)) for j in range(n)] for i in range(n)]
    merges = _single_linkage(dist, n)
    order = _quasi_diag(merges, n)
    w_by_index = _recursive_bisection(cov, order)
    total = sum(w_by_index.values())
    if total <= _EPS:
        return [1.0 / n] * n
    return [w_by_index[i] / total for i in range(n)]


def _max_drawdown(returns: list[float]) -> float:
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cumulative *= (1.0 + r)
        peak = max(peak, cumulative)
        max_dd = max(max_dd, (peak - cumulative) / peak if peak > 0 else 0.0)
    return max_dd


def _resolve_capacity(
    raw: dict[str, float], caps: dict[str, float], floor: float, target_sum: float, max_rounds: int,
) -> dict[str, float]:
    total = sum(raw.values())
    w = {k: (v / total * target_sum if total > _EPS else target_sum / len(raw)) for k, v in raw.items()}
    for _ in range(max_rounds):
        clipped = {k: max(floor, min(caps[k], v)) for k, v in w.items()}
        clipped_sum = sum(clipped.values())
        if abs(clipped_sum - target_sum) <= 1e-9:
            w = clipped
            break
        unclipped = [k for k in w if floor + 1e-12 < clipped[k] < caps[k] - 1e-12]
        if not unclipped:
            w = clipped
            break
        deficit = target_sum - clipped_sum
        total_unclipped = sum(clipped[k] for k in unclipped)
        w = dict(clipped)
        for k in unclipped:
            w[k] = clipped[k] + deficit * (clipped[k] / total_unclipped)
    else:
        w = {k: max(floor, min(caps[k], v)) for k, v in w.items()}
    return w


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = _mean(xs), _mean(ys)
    cov = sum((xs[k] - mx) * (ys[k] - my) for k in range(n)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    denom = max(sx * sy, _EPS)
    return max(-1.0, min(1.0, cov / denom))


def _tail_correlation_matrix(
    sleeve_ids: list[str], est_returns: Mapping[str, list[float]], params: EvaluationParams,
) -> tuple[dict[tuple[str, str], float], float]:
    """Tail-day correlation between every sleeve pair, using a FIXED reference
    tail (the equal-weight book's worst-quantile estimation days). The reference
    is independent of any allocator weight so it cannot be gamed or drift as
    weights are resolved — it is a property of the sleeves themselves."""
    n_days = len(next(iter(est_returns.values())))
    tail_n = max(1, int(math.ceil(params.tail_quantile * n_days)))
    equal_w = 1.0 / len(sleeve_ids)
    combined_ref = [sum(equal_w * est_returns[s][t] for s in sleeve_ids) for t in range(n_days)]
    tail_days = sorted(range(n_days), key=lambda t: combined_ref[t])[:tail_n]
    corr: dict[tuple[str, str], float] = {}
    for i, a in enumerate(sleeve_ids):
        for b in sleeve_ids[i + 1:]:
            xs = [est_returns[a][t] for t in tail_days]
            ys = [est_returns[b][t] for t in tail_days]
            corr[(a, b)] = _pearson(xs, ys) if tail_n > 1 else 1.0
    max_observed = max(corr.values()) if corr else 0.0
    return corr, max_observed


def _resolve_tail_correlation(
    weights: dict[str, float],
    sleeve_ids: list[str],
    tail_corr: Mapping[tuple[str, str], float],
    sharpe_by_sleeve: Mapping[str, float],
    caps: dict[str, float],
    params: EvaluationParams,
    target_sum: float,
) -> tuple[dict[str, float], float, list[str]]:
    """Greedily exclude the weaker (lower estimation-Sharpe) sleeve of the
    first remaining tail-correlation violation, one exclusion per pass, then
    redistribute its freed capacity among the surviving sleeves. Each pass
    strictly shrinks the active set, so this always terminates within
    ``len(sleeve_ids) - 1`` passes — unlike scaling weights toward zero, which
    never changes the underlying correlation and can loop without resolving."""
    excluded: list[str] = []
    w = dict(weights)
    for _ in range(len(sleeve_ids)):
        active = [s for s in sleeve_ids if s not in excluded]
        violating = sorted(
            pair for pair in tail_corr
            if pair[0] in active and pair[1] in active and tail_corr[pair] > params.tail_corr_threshold
        )
        if not violating:
            break
        a, b = violating[0]
        drop = a if sharpe_by_sleeve[a] < sharpe_by_sleeve[b] else b
        excluded.append(drop)
        w[drop] = 0.0
        remaining = [s for s in sleeve_ids if s not in excluded]
        remaining_caps = {s: caps[s] for s in remaining}
        remaining_raw = {s: w[s] if w[s] > 0 else 1e-9 for s in remaining}
        resolved = _resolve_capacity(remaining_raw, remaining_caps, 0.0, target_sum, params.capacity_max_rounds)
        w = {s: (resolved[s] if s in resolved else 0.0) for s in sleeve_ids}
    active = [s for s in sleeve_ids if s not in excluded]
    remaining_pairs = [tail_corr[p] for p in tail_corr if p[0] in active and p[1] in active]
    max_remaining = max(remaining_pairs) if remaining_pairs else 0.0
    return w, max_remaining, excluded


def _apply_drawdown_budget(
    weights: dict[str, float], est_maxdd: dict[str, float], budget: float,
) -> tuple[dict[str, float], int]:
    w = dict(weights)
    violations = 0
    for sleeve_id, maxdd in est_maxdd.items():
        used = w[sleeve_id] * maxdd
        if used > budget + 1e-9:
            violations += 1
            if maxdd > _EPS:
                w[sleeve_id] = budget / maxdd
    return w, violations


def _sharpe(returns: list[float]) -> float:
    vol = _population_stdev(returns)
    if vol <= _EPS:
        return 0.0
    return (_mean(returns) / vol) * math.sqrt(TRADING_DAYS_PER_YEAR)


def evaluate_partitions(partitions: Mapping[str, bytes], params: EvaluationParams | None = None) -> BookResult:
    params = params or EvaluationParams()
    if not (0.0 <= params.cash_reserve_frac < 1.0):
        raise ValueError("cash_reserve_frac must be in [0, 1)")
    sleeves = _rows(partitions)
    calendar = _aligned_calendar(sleeves)
    sleeve_ids = sorted(sleeves)
    n = len(sleeve_ids)
    cutoff = _parse_date(params.oos_start, "oos_start")
    est_idx = [i for i, d in enumerate(calendar) if d < cutoff]
    oos_idx = [i for i, d in enumerate(calendar) if d >= cutoff]
    if len(est_idx) < params.min_est_days or len(oos_idx) < params.min_oos_days:
        raise ValueError(
            f"insufficient chronological rows: estimation={len(est_idx)}, oos={len(oos_idx)}; "
            f"required={params.min_est_days}/{params.min_oos_days}"
        )
    all_returns = {s: [sleeves[s][i]["net_return"] for i in range(len(calendar))] for s in sleeve_ids}
    est_returns = {s: [all_returns[s][i] for i in est_idx] for s in sleeve_ids}
    all_turnover = {s: [sleeves[s][i]["turnover"] for i in range(len(calendar))] for s in sleeve_ids}
    oos_turnover = {s: [all_turnover[s][i] for i in oos_idx] for s in sleeve_ids}

    est_matrix = [est_returns[s] for s in sleeve_ids]
    cov = _covariance_matrix(est_matrix)
    corr = _correlation_matrix(cov)
    vols = [math.sqrt(max(cov[i][i], 0.0)) for i in range(n)]
    degenerate = [sleeve_ids[i] for i in range(n) if vols[i] < params.min_sleeve_vol]
    if degenerate:
        raise ValueError(
            f"sleeve(s) {degenerate} have near-zero estimation-window volatility (< {params.min_sleeve_vol}); "
            "a flat return stream is almost always missing or placeholder data, not a genuine risk-free "
            "sleeve, and inverse-vol/HRP would otherwise reward it with a dominant weight — refusing to allocate"
        )

    inv_vol = _inverse_vol_weights(vols)
    erc = _erc_weights(cov, params)
    corr_pen = _corr_penalized_weights(vols, corr)
    hrp = _hrp_weights(cov, corr)

    hrp_by_sleeve = {sleeve_ids[i]: hrp[i] for i in range(n)}
    caps = {s: float(params.lane_capacity.get(s, params.max_allocation)) for s in sleeve_ids}
    target_sum = 1.0 - params.cash_reserve_frac
    if sum(caps.values()) < target_sum - 1e-9:
        raise ValueError("sum of lane caps cannot satisfy the target allocation net of the cash reserve")

    constrained = _resolve_capacity(hrp_by_sleeve, caps, params.min_allocation, target_sum, params.capacity_max_rounds)
    tail_corr_matrix, tail_corr_pre_resolution_max = _tail_correlation_matrix(sleeve_ids, est_returns, params)
    sharpe_by_sleeve = {s: _sharpe(est_returns[s]) for s in sleeve_ids}
    constrained, max_tail_corr, tail_excluded_sleeves = _resolve_tail_correlation(
        constrained, sleeve_ids, tail_corr_matrix, sharpe_by_sleeve, caps, params, target_sum,
    )
    est_maxdd = {s: _max_drawdown(est_returns[s]) for s in sleeve_ids}
    constrained, n_drawdown_violations = _apply_drawdown_budget(constrained, est_maxdd, params.drawdown_budget)

    def _book_returns(weights: Mapping[str, float], idx: list[int]) -> list[float]:
        return [sum(weights.get(s, 0.0) * all_returns[s][i] for s in sleeve_ids) for i in idx]

    oos_book = _book_returns(constrained, oos_idx)
    oos_sharpe = _sharpe(oos_book)
    oos_maxdd = _max_drawdown(oos_book)
    oos_total_return = math.prod(1.0 + r for r in oos_book) - 1.0
    oos_ann_vol = _population_stdev(oos_book) * math.sqrt(TRADING_DAYS_PER_YEAR)
    turnover_lookthrough = _mean([_mean(oos_turnover[s]) * constrained.get(s, 0.0) for s in sleeve_ids])

    incremental_contribution: dict[str, float] = {}
    for excluded in sleeve_ids:
        remaining = {s: w for s, w in constrained.items() if s != excluded}
        remaining_total = sum(remaining.values())
        if remaining_total > _EPS:
            rescaled = {s: w / remaining_total * sum(constrained.values()) for s, w in remaining.items()}
        else:
            rescaled = remaining
        loo_book = _book_returns(rescaled, oos_idx)
        incremental_contribution[excluded] = round(oos_sharpe - _sharpe(loo_book), 6)

    worst_day = {s: min(est_returns[s]) for s in sleeve_ids}
    stress_worst_day = sum(constrained.get(s, 0.0) * worst_day[s] for s in sleeve_ids)
    stress_corr_breakdown = sum(-3.0 * vols[i] * constrained.get(sleeve_ids[i], 0.0) for i in range(n))

    lane_capacity_ok = all(constrained[s] <= caps[s] + 1e-9 for s in sleeve_ids)
    cash_reserve_ok = sum(constrained.values()) <= target_sum + 1e-9
    tail_corr_ok = max_tail_corr <= params.tail_corr_threshold
    drawdown_budget_ok = all(constrained[s] * est_maxdd[s] <= params.drawdown_budget + 1e-9 for s in sleeve_ids)

    gates = (
        GateResult(gate_id="oos_sharpe_floor", status=GateStatus.PASS if oos_sharpe >= params.min_book_sharpe else GateStatus.FAIL,
                   observed=oos_sharpe, threshold=params.min_book_sharpe, reason="combined-book OOS Sharpe must clear the frozen floor"),
        GateResult(gate_id="oos_drawdown_ceiling", status=GateStatus.PASS if oos_maxdd <= params.max_book_drawdown else GateStatus.FAIL,
                   observed=oos_maxdd, threshold=params.max_book_drawdown, reason="combined-book OOS drawdown must not exceed the frozen ceiling"),
        GateResult(gate_id="lane_capacity_respected", status=GateStatus.PASS if lane_capacity_ok else GateStatus.FAIL,
                   observed=max(constrained.values()), threshold=max(caps.values()), reason="no sleeve may exceed its declared lane capacity"),
        GateResult(gate_id="cash_reserve_maintained", status=GateStatus.PASS if cash_reserve_ok else GateStatus.FAIL,
                   observed=sum(constrained.values()), threshold=target_sum, reason="allocated weight must not exceed 1 minus the cash reserve"),
        GateResult(gate_id="tail_correlation_within_limit", status=GateStatus.PASS if tail_corr_ok else GateStatus.FAIL,
                   observed=max_tail_corr, threshold=params.tail_corr_threshold, reason="tail-day correlation between active sleeves must clear after damping"),
        GateResult(gate_id="drawdown_budget_respected", status=GateStatus.PASS if drawdown_budget_ok else GateStatus.FAIL,
                   observed=n_drawdown_violations, threshold=0, reason="each sleeve's weight times its estimation-window max drawdown must stay within its risk budget"),
    )
    passed = all(g.status == GateStatus.PASS for g in gates)

    artifact = {
        "schema_version": 1,
        "sleeves": sleeve_ids,
        "oos_start": cutoff.isoformat(),
        "estimation_window": {"start": calendar[est_idx[0]].isoformat(), "end": calendar[est_idx[-1]].isoformat(), "n_days": len(est_idx)},
        "oos_window": {"start": calendar[oos_idx[0]].isoformat(), "end": calendar[oos_idx[-1]].isoformat(), "n_days": len(oos_idx)},
        "per_sleeve_estimation_stats": {
            sleeve_ids[i]: {"mean_return": _mean(est_returns[sleeve_ids[i]]), "vol": vols[i], "max_drawdown": est_maxdd[sleeve_ids[i]]}
            for i in range(n)
        },
        "correlation_matrix": {sleeve_ids[i]: {sleeve_ids[j]: corr[i][j] for j in range(n)} for i in range(n)},
        "diagnostic_weights": {
            "inverse_volatility": dict(zip(sleeve_ids, inv_vol)),
            "equal_risk_contribution": dict(zip(sleeve_ids, erc)),
            "correlation_penalized": dict(zip(sleeve_ids, corr_pen)),
            "hrp_raw": dict(zip(sleeve_ids, hrp)),
        },
        "final_policy": "hrp_across_sleeves",
        "constrained_weights": constrained,
        "cash_reserve_frac": params.cash_reserve_frac,
        "resolved_caps": caps,
        "tail_correlation": {
            "threshold": params.tail_corr_threshold, "quantile": params.tail_quantile,
            "pre_resolution_max": tail_corr_pre_resolution_max,
            "max_observed": max_tail_corr, "excluded_sleeves": tail_excluded_sleeves,
        },
        "drawdown_budget": {"budget": params.drawdown_budget, "violations_resolved": n_drawdown_violations},
        "book_oos": {
            "total_return": oos_total_return, "ann_vol": oos_ann_vol, "sharpe": oos_sharpe,
            "max_drawdown": oos_maxdd, "turnover_lookthrough": turnover_lookthrough,
        },
        "incremental_contribution": incremental_contribution,
        "stress_scenarios": {
            "historical_worst_day_combined": stress_worst_day,
            "correlation_breakdown_3sigma": stress_corr_breakdown,
        },
    }
    metrics = {
        "oos_sharpe": oos_sharpe, "oos_max_drawdown": oos_maxdd, "oos_total_return": oos_total_return,
        "oos_ann_vol": oos_ann_vol, "n_sleeves": float(n), "cash_reserve_frac": params.cash_reserve_frac,
        "max_observed_tail_correlation": max_tail_corr, "n_drawdown_budget_violations": float(n_drawdown_violations),
    }
    return BookResult(
        metrics=metrics,
        gates=gates,
        passed=passed,
        artifact_bytes=canonical_bytes(artifact),
        effective_sample_size=float(len(oos_idx)),
        temporal_holdout=f"dates at or after {cutoff.isoformat()} (strict chronological holdout; weights fit only on prior dates)",
        incumbent_comparison={"baseline": "equal-weight-across-sleeves", "policy": "hrp_across_sleeves"},
    )


__all__ = ["METRICS", "REQUIRED_ROW_FIELDS", "EvaluationParams", "evaluate_partitions"]
