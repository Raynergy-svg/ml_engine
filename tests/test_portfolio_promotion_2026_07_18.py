"""Universal portfolio-level promotion gate (src/hedge/portfolio_promotion.py).

No mocks: synthetic twin-lane ledger rows (the real F1-aligned schema), real
disk via tmp_path. Locks in every check and the fail-closed decision order:
any FAIL -> REJECT; any UNKNOWN -> CONTINUE_SHADOW; all PASS -> the gate's
strongest verdict, PROMOTE_TO_OPERATOR_REVIEW (never an actual promotion).
"""
from __future__ import annotations

import json
import math
import random

import pytest

from src.hedge.portfolio_promotion import (
    GateConfig,
    VERDICT_PROMOTE,
    VERDICT_REJECT,
    VERDICT_SHADOW,
    build_portfolio_promotion_report,
    evaluate_strategy,
    _max_drawdown,
    _pearson,
)


def _row(strategy, asof, raw, hedged_net, beta=0.1, currency=None):
    return {
        "strategy": strategy,
        "asof_date": asof,
        "raw": {"net_return": raw},
        "hedge": {"status": "applied"},
        "hedged": {"net_return": hedged_net, "gross_return": None},
        "exposure": {
            "net_currency_exposure": currency or {},
            "net_sector_exposure": {},
            "net_correlation_bucket_exposure": {},
            "net_beta_exposure": beta,
            "fail_closed": False,
        },
    }


def _dates(n):
    return [f"2026-06-{d:02d}" for d in range(1, n + 1)]


def _series(strategy, returns, residual_of=None, beta=0.1, currency=None):
    """residual_of: callable raw -> residual (defaults to 80% of raw)."""
    f = residual_of or (lambda r: r * 0.8)
    return [_row(strategy, d, r, f(r), beta=beta, currency=currency)
            for d, r in zip(_dates(len(returns)), returns)]


def _alpha_returns(seed, n=10, mu=0.004, sigma=0.002):
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


# ── helpers ─────────────────────────────────────────────────────────────────

def test_pearson_and_drawdown_helpers():
    assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert _pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert _pearson([1, 1, 1], [1, 2, 3]) is None  # degenerate: undefined, not 0
    assert _max_drawdown([0.05, -0.03, -0.04, 0.02]) == pytest.approx(0.07)
    assert _max_drawdown([0.01, 0.01]) == 0.0


# ── verdicts ────────────────────────────────────────────────────────────────

def test_good_uncorrelated_candidate_promotes_to_operator_review():
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1)),
        "candidate":   _series("candidate", _alpha_returns(99)),
    }
    scorecards = {"candidate": {"decision": {"verdict": "genuine_strategy_specific_signal"}}}
    out = evaluate_strategy("candidate", rows, scorecards)
    assert out["verdict"] == VERDICT_PROMOTE, [c for c in out["checks"] if c["status"] != "pass"]
    assert out["human_review_required"] is True
    assert {c["name"] for c in out["checks"]} == {
        "standalone_evidence", "residual_alpha", "hedge_cost",
        "duplication", "bucket_crowding", "marginal_contribution",
    }


def test_pure_beta_candidate_rejected():
    # Residual ~ 0 while raw is strongly positive: phi ~ 0 -> residual_alpha FAIL
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1)),
        "candidate":   _series("candidate", _alpha_returns(7),
                               residual_of=lambda r: r * 0.02),
    }
    out = evaluate_strategy("candidate", rows,
                            {"candidate": {"decision": {"verdict": "return_was_beta_not_alpha"}}})
    assert out["verdict"] == VERDICT_REJECT
    alpha = next(c for c in out["checks"] if c["name"] == "residual_alpha")
    assert alpha["status"] == "fail"


def test_duplicate_candidate_rejected():
    base = _alpha_returns(3)
    rows = {
        "incumbent_a": _series("incumbent_a", base),
        # near-copy of incumbent_a's residual path -> correlation ~ 1
        "candidate":   _series("candidate", [r * 1.01 for r in base]),
    }
    out = evaluate_strategy("candidate", rows,
                            {"candidate": {"decision": {"verdict": "genuine_strategy_specific_signal"}}})
    assert out["verdict"] == VERDICT_REJECT
    dup = next(c for c in out["checks"] if c["name"] == "duplication")
    assert dup["status"] == "fail"
    assert any(p["corr"] is not None and p["corr"] >= 0.85 for p in dup["pairs"])


def test_bucket_crowding_rejects_same_direction_but_not_offsetting():
    cfg = GateConfig()
    # Incumbent already long 0.6 USD; candidate adds +0.4 -> combined 1.0 > cap 0.75
    rows_crowding = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1),
                               currency={"USD": 0.6}),
        "candidate":   _series("candidate", _alpha_returns(99),
                               currency={"USD": 0.4}),
    }
    sc = {"candidate": {"decision": {"verdict": "genuine_strategy_specific_signal"}}}
    out = evaluate_strategy("candidate", rows_crowding, sc, cfg)
    assert out["verdict"] == VERDICT_REJECT
    crowd = next(c for c in out["checks"] if c["name"] == "bucket_crowding")
    assert crowd["status"] == "fail"
    assert crowd["breaches"][0]["bucket"] == "net_currency_exposure:USD"

    # Same magnitudes, but the candidate OFFSETS the crowded bucket -> diversifying
    rows_offset = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1),
                               currency={"USD": 0.9}),
        "candidate":   _series("candidate", _alpha_returns(99),
                               currency={"USD": -0.4}),
    }
    out2 = evaluate_strategy("candidate", rows_offset, sc, cfg)
    crowd2 = next(c for c in out2["checks"] if c["name"] == "bucket_crowding")
    assert crowd2["status"] == "pass"


def test_harmful_marginal_contribution_rejected():
    # Candidate residual is the NEGATIVE of a strong incumbent on every date:
    # combined expectancy collapses -> marginal_contribution FAIL
    base = [0.004] * 10
    rows = {
        "incumbent_a": _series("incumbent_a", base),
        "candidate":   _series("candidate", [-r * 3 for r in base]),
    }
    out = evaluate_strategy("candidate", rows,
                            {"candidate": {"decision": {"verdict": "genuine_strategy_specific_signal"}}})
    marg = next(c for c in out["checks"] if c["name"] == "marginal_contribution")
    # (phi is still 0.8 here so residual_alpha passes; the marginal check must
    # be what catches the harm)
    assert marg["status"] == "fail"
    assert out["verdict"] == VERDICT_REJECT


def test_insufficient_history_stays_in_shadow_never_promotes():
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1)),
        "candidate":   _series("candidate", _alpha_returns(99, n=3)),  # < 8 cycles
    }
    out = evaluate_strategy("candidate", rows,
                            {"candidate": {"decision": {"verdict": "genuine_strategy_specific_signal"}}})
    assert out["verdict"] == VERDICT_SHADOW
    ev = next(c for c in out["checks"] if c["name"] == "standalone_evidence")
    assert ev["status"] == "unknown"


def test_first_covered_strategy_can_promote_without_incumbents():
    rows = {"candidate": _series("candidate", _alpha_returns(99))}
    out = evaluate_strategy("candidate", rows,
                            {"candidate": {"decision": {"verdict": "genuine_strategy_specific_signal"}}})
    assert out["verdict"] == VERDICT_PROMOTE
    marg = next(c for c in out["checks"] if c["name"] == "marginal_contribution")
    assert "no incumbent" in marg["detail"]


# ── report build (real disk) ────────────────────────────────────────────────

def test_report_build_roundtrip(tmp_path):
    ledger = tmp_path / "raw_vs_hedged_ledger.jsonl"
    with open(ledger, "w", encoding="utf-8") as fh:
        for row in (_series("incumbent_a", _alpha_returns(1))
                    + _series("candidate", _alpha_returns(99))):
            fh.write(json.dumps(row) + "\n")
    scorecard = tmp_path / "hedge_scorecard_report.json"
    scorecard.write_text(json.dumps({"scorecards": {
        "incumbent_a": {"decision": {"verdict": "genuine_strategy_specific_signal"}},
        "candidate": {"decision": {"verdict": "genuine_strategy_specific_signal"}},
    }}))
    out = tmp_path / "portfolio_promotion_report.json"

    report = build_portfolio_promotion_report(
        ledger_path=ledger, scorecard_path=scorecard, out_path=out)
    assert out.exists()
    assert set(report["verdicts"]) == {"incumbent_a", "candidate"}  # universal
    assert report["verdicts"]["candidate"]["verdict"] == VERDICT_PROMOTE
    assert report["human_review_required"] is True and report["runtime_allowed"] is False
    persisted = json.loads(out.read_text())
    assert persisted["verdicts"]["candidate"]["verdict"] == VERDICT_PROMOTE
    assert list(tmp_path.glob(".portfolio_gate_*")) == []
    # every recorded number is finite (no NaN smuggled into the report)
    def _walk(o):
        if isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
        elif isinstance(o, float):
            assert math.isfinite(o)
    _walk(persisted)
