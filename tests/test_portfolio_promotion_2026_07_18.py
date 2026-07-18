"""Universal portfolio-level promotion gate (src/hedge/portfolio_promotion.py, rev 2).

No mocks: synthetic twin-lane ledger rows in the REAL F1-aligned schema —
including the rev-2 unit contract the first version of this suite hid: the
lane persists DOLLAR exposures (risk_home = |w| x notional) plus the row's
``notional``; the gate must normalize before any cap comparison (#1).
Scorecard verdicts are the imported hedge_scorecard constants (#2). Locks in
every check, every UNKNOWN path, and the fail-closed decision order:
any FAIL -> REJECT; any UNKNOWN -> CONTINUE_SHADOW; all PASS -> the gate's
strongest verdict, PROMOTE_TO_OPERATOR_REVIEW (never an actual promotion).
"""
from __future__ import annotations

import json
import math
import random

import pytest

from src.hedge.hedge_scorecard import (
    VERDICT_BETA_NOT_ALPHA,
    VERDICT_GENUINE,
)
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

NOTIONAL = 100_000.0


def _row(strategy, asof, raw, hedged_net, beta=0.1, currency=None,
         notional=NOTIONAL, exposure_ok=True):
    """One F1-schema ledger row. ``beta``/``currency`` are given as fractions
    of notional for test readability but persisted as DOLLARS — exactly what
    the real lane writes (risk_home = |w| x notional)."""
    return {
        "strategy": strategy,
        "asof_date": asof,
        "notional": notional,
        "raw": {"net_return": raw},
        "hedge": {"status": "applied"},
        "hedged": {"net_return": hedged_net, "gross_return": None},
        "exposure": {
            "net_currency_exposure": {k: v * notional
                                      for k, v in (currency or {}).items()},
            "net_sector_exposure": {},
            "net_correlation_bucket_exposure": {},
            "net_beta_exposure": beta * notional,
            "fail_closed": not exposure_ok,
        },
    }


def _dates(n, start=1):
    return [f"2026-06-{d:02d}" for d in range(start, start + n)]


def _series(strategy, returns, residual_of=None, beta=0.1, currency=None,
            notional=NOTIONAL, exposure_ok=True, start=1):
    """residual_of: callable raw -> residual (defaults to 80% of raw)."""
    f = residual_of or (lambda r: r * 0.8)
    return [_row(strategy, d, r, f(r), beta=beta, currency=currency,
                 notional=notional, exposure_ok=exposure_ok)
            for d, r in zip(_dates(len(returns), start=start), returns)]


def _alpha_returns(seed, n=10, mu=0.004, sigma=0.002):
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _sc(verdict):
    return {"candidate": {"decision": {"verdict": verdict}}}


def _status(out, name):
    return next(c for c in out["checks"] if c["name"] == name)


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
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    assert out["verdict"] == VERDICT_PROMOTE, [c for c in out["checks"] if c["status"] != "pass"]
    assert out["human_review_required"] is True
    assert {c["name"] for c in out["checks"]} == {
        "standalone_evidence", "residual_alpha", "hedge_cost",
        "duplication", "bucket_crowding", "marginal_contribution",
    }
    assert sum(out["allocations"].values()) == pytest.approx(1.0)


def test_pure_beta_candidate_rejected():
    # Residual ~ 0 while raw is strongly positive: phi ~ 0 -> residual_alpha FAIL
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1)),
        "candidate":   _series("candidate", _alpha_returns(7),
                               residual_of=lambda r: r * 0.02),
    }
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_BETA_NOT_ALPHA))
    assert out["verdict"] == VERDICT_REJECT
    assert _status(out, "residual_alpha")["status"] == "fail"


def test_missing_scorecard_verdict_is_unknown_never_pass():
    # #3: phi alone (0.8 here) is NOT the full contract — an absent scorecard
    # verdict must land UNKNOWN -> CONTINUE_SHADOW, never a silent pass.
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1)),
        "candidate":   _series("candidate", _alpha_returns(99)),
    }
    out = evaluate_strategy("candidate", rows, scorecards={})
    alpha = _status(out, "residual_alpha")
    assert alpha["status"] == "unknown" and alpha["scorecard_verdict"] is None
    assert out["verdict"] == VERDICT_SHADOW


def test_duplicate_candidate_rejected():
    base = _alpha_returns(3)
    rows = {
        "incumbent_a": _series("incumbent_a", base),
        # near-copy of incumbent_a's residual path -> correlation ~ 1
        "candidate":   _series("candidate", [r * 1.01 for r in base]),
    }
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    assert out["verdict"] == VERDICT_REJECT
    dup = _status(out, "duplication")
    assert dup["status"] == "fail"
    assert any(p["corr"] is not None and p["corr"] >= 0.85 for p in dup["pairs"])


def test_degenerate_incumbent_correlation_is_unknown_not_pass():
    # #4: enough overlap but the incumbent residual series is CONSTANT (0.0 is
    # exactly representable, so the variance is exactly zero) -> correlation
    # undefined -> the pair is unresolved -> UNKNOWN, never "no duplication
    # detected".
    # (candidate returns kept strictly positive so the zero-baseline incumbent
    # portfolio's multiplicative DD tolerance is not the thing under test)
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1),
                               residual_of=lambda r: 0.0),
        "candidate":   _series("candidate", [0.004] * 10),
    }
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    dup = _status(out, "duplication")
    assert dup["status"] == "unknown"
    assert any(p["why"] == "correlation_undefined_degenerate_series"
               for p in dup["pairs"])
    assert out["verdict"] == VERDICT_SHADOW


def test_bucket_crowding_rejects_same_direction_but_not_offsetting():
    cfg = GateConfig()
    sc = _sc(VERDICT_GENUINE)
    # Equal allocations (0.5 each): 0.5*1.2 + 0.5*0.5 = 0.85 > 0.75 cap, and
    # the candidate's leg points the SAME way as the breach -> fail.
    rows_crowding = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1),
                               currency={"USD": 1.2}),
        "candidate":   _series("candidate", _alpha_returns(99),
                               currency={"USD": 0.5}),
    }
    out = evaluate_strategy("candidate", rows_crowding, sc, cfg)
    assert out["verdict"] == VERDICT_REJECT
    crowd = _status(out, "bucket_crowding")
    assert crowd["status"] == "fail"
    assert crowd["breaches"][0]["bucket"] == "net_currency_exposure:USD"

    # Candidate OFFSETS the crowded bucket -> combined 0.5*1.8 - 0.5*0.4 = 0.7
    # inside the cap: diversification, not crowding.
    rows_offset = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1),
                               currency={"USD": 1.8}),
        "candidate":   _series("candidate", _alpha_returns(99),
                               currency={"USD": -0.4}),
    }
    out2 = evaluate_strategy("candidate", rows_offset, sc, cfg)
    assert _status(out2, "bucket_crowding")["status"] == "pass"


def test_crowding_normalizes_dollar_exposures_by_each_rows_own_notional():
    # #1: the ledger stores DOLLARS. candidate runs at 2x the incumbent's
    # notional; $80k on $200k is 0.4, so combined 0.5*0.6 + 0.5*0.4 = 0.5 is
    # inside the 0.75 cap. Comparing raw dollars against the cap (the rev-1
    # bug) would have called this a massive breach.
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1),
                               currency={"USD": 0.6}, notional=100_000.0),
        "candidate":   _series("candidate", _alpha_returns(99),
                               currency={"USD": 0.4}, notional=200_000.0),
    }
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    assert _status(out, "bucket_crowding")["status"] == "pass"
    assert out["verdict"] == VERDICT_PROMOTE


def test_crowding_honors_operator_allocations():
    # Same books, different capital split: equal weight is fine
    # (0.5*1.0 + 0.5*0.2 = 0.6), but 90/10 concentrates the incumbent's
    # long USD past the cap (0.9*1.0 + 0.1*0.2 = 0.92) with the candidate
    # on the breaching side -> the allocations parameter must matter.
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1),
                               currency={"USD": 1.0}),
        "candidate":   _series("candidate", _alpha_returns(99),
                               currency={"USD": 0.2}),
    }
    sc = _sc(VERDICT_GENUINE)
    out_equal = evaluate_strategy("candidate", rows, sc)
    assert _status(out_equal, "bucket_crowding")["status"] == "pass"
    out_conc = evaluate_strategy(
        "candidate", rows, sc,
        allocations={"incumbent_a": 0.9, "candidate": 0.1})
    assert _status(out_conc, "bucket_crowding")["status"] == "fail"
    assert out_conc["allocations"]["incumbent_a"] == pytest.approx(0.9)


def test_missing_incumbent_exposure_is_unknown_not_partial_portfolio():
    # #5: an incumbent whose exposure rows are all fail-closed cannot be
    # dropped from the "portfolio" — the combined book is unknowable.
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1),
                               exposure_ok=False),
        "candidate":   _series("candidate", _alpha_returns(99)),
    }
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    crowd = _status(out, "bucket_crowding")
    assert crowd["status"] == "unknown"
    assert "incumbent_a" in crowd["missing"]
    assert out["verdict"] == VERDICT_SHADOW


def test_time_misaligned_snapshots_are_unknown_not_a_portfolio():
    # #5: no single date at which every covered strategy has an exposure row
    # -> the snapshots do not form ONE portfolio -> UNKNOWN.
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1), start=11),
        "candidate":   _series("candidate", _alpha_returns(99), start=1),
    }
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    crowd = _status(out, "bucket_crowding")
    assert crowd["status"] == "unknown"
    assert "no common exposure date" in crowd["detail"]
    assert out["verdict"] == VERDICT_SHADOW


def test_harmful_marginal_contribution_rejected():
    # Candidate residual is the NEGATIVE of a strong incumbent on every date:
    # combined expectancy and reward/vol collapse -> marginal_contribution FAIL
    base = [0.004] * 10
    rows = {
        "incumbent_a": _series("incumbent_a", base),
        "candidate":   _series("candidate", [-r * 3 for r in base]),
    }
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    assert _status(out, "marginal_contribution")["status"] == "fail"
    assert out["verdict"] == VERDICT_REJECT


def test_insufficient_history_stays_in_shadow_never_promotes():
    rows = {
        "incumbent_a": _series("incumbent_a", _alpha_returns(1)),
        "candidate":   _series("candidate", _alpha_returns(99, n=3)),  # < 8 cycles
    }
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    assert out["verdict"] == VERDICT_SHADOW
    assert _status(out, "standalone_evidence")["status"] == "unknown"


def test_first_covered_strategy_can_promote_without_incumbents():
    rows = {"candidate": _series("candidate", _alpha_returns(99))}
    out = evaluate_strategy("candidate", rows, _sc(VERDICT_GENUINE))
    assert out["verdict"] == VERDICT_PROMOTE
    marg = _status(out, "marginal_contribution")
    assert "no incumbent" in marg["detail"]


# ── report build (real disk) ────────────────────────────────────────────────

def test_report_build_is_universal_and_roundtrips(tmp_path):
    ledger = tmp_path / "raw_vs_hedged_ledger.jsonl"
    with open(ledger, "w", encoding="utf-8") as fh:
        for row in (_series("incumbent_a", _alpha_returns(1))
                    + _series("candidate", _alpha_returns(99))):
            fh.write(json.dumps(row) + "\n")
    scorecard = tmp_path / "hedge_scorecard_report.json"
    scorecard.write_text(json.dumps({"scorecards": {
        "incumbent_a": {"decision": {"verdict": VERDICT_GENUINE}},
        "candidate": {"decision": {"verdict": VERDICT_GENUINE}},
    }}))
    out = tmp_path / "portfolio_promotion_report.json"

    report = build_portfolio_promotion_report(
        ledger_path=ledger, scorecard_path=scorecard, out_path=out)
    assert out.exists()

    # #7 universal: the FULL covered-strategy registry is evaluated, union the
    # ledger's own keys — a lane with no rows gets an explicit verdict.
    from src.hedge.hedged_shadow_lane import STRATEGIES
    assert set(report["verdicts"]) == set(STRATEGIES) | {"incumbent_a", "candidate"}
    for lane in STRATEGIES:
        assert report["verdicts"][lane]["verdict"] == VERDICT_SHADOW, (
            f"{lane} has no ledger rows — must stay in shadow, never silence")

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
