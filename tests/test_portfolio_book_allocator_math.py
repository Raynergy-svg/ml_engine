"""Unit tests for the pure-Python portfolio-book allocator math
(``src.evidence.portfolio_book.evaluation``) — no evidence store, no signing,
just the deterministic allocation/gate arithmetic against hand-constructed
synthetic sleeve return series with known properties."""

from __future__ import annotations

import datetime as dt
import json
import random
from datetime import date

import pytest

from src.evidence.portfolio_book.evaluation import EvaluationParams, evaluate_partitions


def _partition(returns: list[float], *, start: str = "2025-01-01") -> bytes:
    day = date.fromisoformat(start)
    rows = []
    for r in returns:
        while day.weekday() >= 5:
            day += dt.timedelta(days=1)
        rows.append({"date": day.isoformat(), "net_return": r, "gross_exposure": 1.0, "turnover": 0.05})
        day += dt.timedelta(days=1)
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8")


def _gauss_partition(mu: float, sigma: float, n: int, *, seed: int) -> tuple[bytes, list[float]]:
    rng = random.Random(seed)
    values = [rng.gauss(mu, sigma) for _ in range(n)]
    return _partition(values), values


DEFAULT = EvaluationParams(oos_start="2025-01-01", min_est_days=1, min_oos_days=1)


def _params(**overrides) -> EvaluationParams:
    base = dict(oos_start="2025-11-01", min_est_days=150, min_oos_days=20)
    base.update(overrides)
    return EvaluationParams(**base)


def test_requires_at_least_two_sleeves():
    part, _ = _gauss_partition(0.0005, 0.006, 300, seed=1)
    with pytest.raises(ValueError, match="at least 2 sleeve"):
        evaluate_partitions({"a": part}, _params())


def test_misaligned_calendar_is_refused():
    a, _ = _gauss_partition(0.0005, 0.006, 300, seed=1)
    b, _ = _gauss_partition(0.0004, 0.005, 299, seed=2)  # one fewer trading day
    with pytest.raises(ValueError, match="calendar"):
        evaluate_partitions({"a": a, "b": b}, _params())


def test_flat_sleeve_is_refused_rather_than_rewarded():
    # A sleeve with near-zero estimation-window volatility almost always means
    # missing or placeholder data, not a genuine risk-free asset — inverse-vol
    # and HRP would otherwise hand it a dominant weight. Must refuse, not silently
    # allocate the book to what looks like a "free lunch".
    flat = _partition([0.0001] * 300)
    normal, _ = _gauss_partition(0.0005, 0.006, 300, seed=1)
    with pytest.raises(ValueError, match="near-zero estimation-window volatility"):
        evaluate_partitions({"flat": flat, "normal": normal}, _params(max_allocation=0.9))


def test_duplicate_date_within_a_sleeve_is_refused():
    part, values = _gauss_partition(0.0005, 0.006, 300, seed=1)
    rows = part.decode().splitlines()
    corrupted = (rows[0] + "\n" + "\n".join(rows) + "\n").encode()  # duplicate first row's date
    b, _ = _gauss_partition(0.0004, 0.005, 300, seed=2)
    with pytest.raises(ValueError, match="duplicate date"):
        evaluate_partitions({"a": corrupted, "b": b}, _params())


def test_deterministic_replay_is_byte_identical():
    a, _ = _gauss_partition(0.0006, 0.006, 300, seed=1)
    b, _ = _gauss_partition(0.0004, 0.004, 300, seed=2)
    c, _ = _gauss_partition(0.0005, 0.008, 300, seed=3)
    partitions = {"a": a, "b": b, "c": c}
    r1 = evaluate_partitions(partitions, _params())
    r2 = evaluate_partitions(partitions, _params())
    assert r1.artifact_bytes == r2.artifact_bytes
    assert r1.metrics == r2.metrics


def test_inverse_vol_favors_the_lower_volatility_sleeve():
    a, _ = _gauss_partition(0.0005, 0.002, 300, seed=1)  # low vol
    b, _ = _gauss_partition(0.0005, 0.02, 300, seed=2)   # high vol
    result = evaluate_partitions({"a": a, "b": b}, _params(max_allocation=0.9))
    diag = json.loads(result.artifact_bytes)["diagnostic_weights"]["inverse_volatility"]
    assert diag["a"] > diag["b"]


def test_correlation_penalized_favors_less_correlated_sleeve():
    # Three equal-volatility sleeves so inverse-vol alone cannot differentiate
    # them: 'a' and 'b' share a common factor (highly correlated with each
    # other); 'c' is independent of both. Correlation-penalization must reward
    # 'c' — the diversifier — over 'a' and 'b', which a naive inverse-vol split
    # (or an n=2 book, where the penalty term is identical for both sides)
    # would miss entirely.
    rng = random.Random(11)
    n = 300
    sigma = 0.008
    base = [rng.gauss(0.0005, sigma) for _ in range(n)]
    a = _partition([x + rng.gauss(0, 0.0005) for x in base])
    b = _partition([x + rng.gauss(0, 0.0005) for x in base])
    c = _partition([rng.gauss(0.0005, sigma) for _ in range(n)])
    result = evaluate_partitions({"a": a, "b": b, "c": c}, _params(max_allocation=0.9))
    diag = json.loads(result.artifact_bytes)["diagnostic_weights"]["correlation_penalized"]
    assert diag["c"] > diag["a"]
    assert diag["c"] > diag["b"]
    assert abs(sum(diag.values()) - 1.0) < 1e-9


def test_hrp_and_erc_diagnostic_weights_sum_to_one():
    a, _ = _gauss_partition(0.0006, 0.006, 300, seed=1)
    b, _ = _gauss_partition(0.0004, 0.004, 300, seed=2)
    c, _ = _gauss_partition(0.0005, 0.008, 300, seed=3)
    d, _ = _gauss_partition(0.0003, 0.003, 300, seed=4)
    result = evaluate_partitions({"a": a, "b": b, "c": c, "d": d}, _params(max_allocation=0.9))
    diag = json.loads(result.artifact_bytes)["diagnostic_weights"]
    for method in ("inverse_volatility", "equal_risk_contribution", "correlation_penalized", "hrp_raw"):
        total = sum(diag[method].values())
        assert abs(total - 1.0) < 1e-6, f"{method} weights do not sum to 1: {total}"
        assert all(w >= -1e-9 for w in diag[method].values()), f"{method} produced a negative weight"


def test_erc_roughly_equalizes_risk_contribution():
    a, _ = _gauss_partition(0.0005, 0.004, 400, seed=1)
    b, _ = _gauss_partition(0.0005, 0.012, 400, seed=2)  # 3x the vol of 'a'
    result = evaluate_partitions(
        {"a": a, "b": b},
        _params(oos_start="2026-01-28", max_allocation=0.9, min_est_days=280, min_oos_days=20),
    )
    diag = json.loads(result.artifact_bytes)["diagnostic_weights"]["equal_risk_contribution"]
    # ERC on two assets reduces to inverse-vol weighting (equal risk contribution
    # with zero correlation implies w_i ~ 1/sigma_i), so the low-vol sleeve gets
    # the larger allocation, unlike a naive equal-weight split.
    assert diag["a"] > diag["b"]
    assert abs(diag["a"] + diag["b"] - 1.0) < 1e-6


def test_lane_capacity_and_cash_reserve_are_respected():
    a, _ = _gauss_partition(0.001, 0.003, 300, seed=1)  # will dominate HRP raw weight
    b, _ = _gauss_partition(0.0002, 0.015, 300, seed=2)
    c, _ = _gauss_partition(0.0002, 0.015, 300, seed=3)
    params = _params(max_allocation=0.35, cash_reserve_frac=0.15)
    result = evaluate_partitions({"a": a, "b": b, "c": c}, params)
    artifact = json.loads(result.artifact_bytes)
    weights = artifact["constrained_weights"]
    assert all(w <= 0.35 + 1e-9 for w in weights.values())
    assert sum(weights.values()) <= 0.85 + 1e-9


def test_per_sleeve_lane_capacity_override_is_respected():
    a, _ = _gauss_partition(0.001, 0.003, 300, seed=1)
    b, _ = _gauss_partition(0.0002, 0.015, 300, seed=2)
    params = _params(max_allocation=0.9, lane_capacity={"a": 0.20})
    result = evaluate_partitions({"a": a, "b": b}, params)
    weights = json.loads(result.artifact_bytes)["constrained_weights"]
    assert weights["a"] <= 0.20 + 1e-9


def test_infeasible_capacity_raises_before_any_allocation():
    a, _ = _gauss_partition(0.0005, 0.006, 300, seed=1)
    b, _ = _gauss_partition(0.0004, 0.005, 300, seed=2)
    params = _params(max_allocation=0.30, cash_reserve_frac=0.10)  # 2*0.30=0.60 < 0.90 target
    with pytest.raises(ValueError, match="lane caps cannot satisfy"):
        evaluate_partitions({"a": a, "b": b}, params)


def test_tail_correlation_excludes_the_redundant_sleeve():
    rng = random.Random(7)
    n = 250
    base = [rng.gauss(0.0005, 0.01) for _ in range(n)]
    x = _partition([v + rng.gauss(0, 0.0005) for v in base])
    y = _partition([v + rng.gauss(0, 0.0005) for v in base])  # near-duplicate of x
    z, _ = _gauss_partition(0.0004, 0.006, n, seed=99)
    params = _params(oos_start="2025-10-01", min_est_days=100, min_oos_days=15, tail_corr_threshold=0.5, max_allocation=0.9)
    result = evaluate_partitions({"x": x, "y": y, "z": z}, params)
    artifact = json.loads(result.artifact_bytes)
    excluded = artifact["tail_correlation"]["excluded_sleeves"]
    assert len(excluded) == 1
    assert excluded[0] in ("x", "y")
    assert artifact["constrained_weights"][excluded[0]] == 0.0
    tail_gate = next(g for g in result.gates if g.gate_id == "tail_correlation_within_limit")
    assert tail_gate.status.value == "PASS"


def test_drawdown_budget_scales_down_a_high_drawdown_sleeve():
    n = 300
    # 'crash' sleeve: mostly flat, with one severe drawdown day INSIDE the
    # estimation window (index 200 of 250 estimation days) — the drawdown
    # budget is deliberately causal (estimation-window only), so the shock
    # must fall before the oos_start cutoff to be visible to the constraint.
    crash_returns = [0.0005] * 200 + [-0.30] + [0.0005] * (n - 201)
    crash = _partition(crash_returns)
    calm, _ = _gauss_partition(0.0004, 0.004, n, seed=4)
    params = _params(oos_start="2025-12-17", min_est_days=250, min_oos_days=20, max_allocation=0.9, drawdown_budget=0.02)
    result = evaluate_partitions({"crash": crash, "calm": calm}, params)
    artifact = json.loads(result.artifact_bytes)
    used = artifact["constrained_weights"]["crash"] * artifact["per_sleeve_estimation_stats"]["crash"]["max_drawdown"]
    assert used <= 0.02 + 1e-6
    assert artifact["drawdown_budget"]["violations_resolved"] >= 1


def test_incremental_contribution_reports_one_entry_per_sleeve():
    a, _ = _gauss_partition(0.0006, 0.006, 300, seed=1)
    b, _ = _gauss_partition(0.0004, 0.004, 300, seed=2)
    c, _ = _gauss_partition(0.0005, 0.008, 300, seed=3)
    result = evaluate_partitions({"a": a, "b": b, "c": c}, _params(max_allocation=0.9))
    artifact = json.loads(result.artifact_bytes)
    assert set(artifact["incremental_contribution"]) == {"a", "b", "c"}


def test_stress_scenarios_are_present_and_non_positive():
    a, _ = _gauss_partition(0.0006, 0.006, 300, seed=1)
    b, _ = _gauss_partition(0.0004, 0.004, 300, seed=2)
    result = evaluate_partitions({"a": a, "b": b}, _params(max_allocation=0.9))
    stress = json.loads(result.artifact_bytes)["stress_scenarios"]
    assert stress["historical_worst_day_combined"] <= 0.0
    assert stress["correlation_breakdown_3sigma"] <= 0.0


def test_gate_fails_when_oos_sharpe_below_floor():
    rng = random.Random(5)
    n = 300
    noisy = [rng.gauss(0.0, 0.02) for _ in range(n)]  # ~zero-mean, will not clear a positive Sharpe floor
    a = _partition(noisy)
    b, _ = _gauss_partition(0.0, 0.02, n, seed=6)
    params = _params(max_allocation=0.9, min_book_sharpe=5.0)  # deliberately unreachable floor
    result = evaluate_partitions({"a": a, "b": b}, params)
    assert result.passed is False
    sharpe_gate = next(g for g in result.gates if g.gate_id == "oos_sharpe_floor")
    assert sharpe_gate.status.value == "FAIL"


def test_gate_passed_matches_all_gates_pass_invariant():
    a, _ = _gauss_partition(0.0006, 0.006, 300, seed=1)
    b, _ = _gauss_partition(0.0004, 0.004, 300, seed=2)
    result = evaluate_partitions({"a": a, "b": b}, _params(max_allocation=0.9))
    any_failed = any(g.status.value == "FAIL" for g in result.gates)
    assert result.passed == (not any_failed)
