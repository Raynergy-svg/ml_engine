"""Layer 8 — residual-alpha attribution (src/hedge/residual_attribution.py).

No mocks: real lane-row dicts (the F1-fixed twin-lane schema), real disk via
tmp_path, real ScannerConfig. Locks in:
  1. the residual identity residual + systematic == raw, cycle by cycle;
  2. fail-closed exclusion of non-applied / unresolved cycles;
  3. phi semantics (pure beta -> ~0 credit, pure alpha -> ~1, bounds);
  4. reward-hook pass-through reasons (no coverage, short history, gross basis);
  5. the operator's exponential credit update (math, bounds, non-finite guard);
  6. report build + consumer-side loader round trip on real files;
  7. the ScannerConfig flag exists and defaults OFF (shadow-first).
"""
from __future__ import annotations

import json
import math

import pytest

from src.hedge.residual_attribution import (
    MULTIPLIER_CAP,
    build_residual_attribution_report,
    exponential_credit_update,
    load_attribution_for,
    residual_reward,
    residual_series,
    strategy_attribution,
)


def _row(asof: str, raw, hedged_net, status: str = "applied", hedged_gross=None):
    return {
        "strategy": "test_strat",
        "asof_date": asof,
        "raw": {"net_return": raw},
        "hedge": {"status": status},
        "hedged": {"net_return": hedged_net, "gross_return": hedged_gross},
    }


# ── 1-2. Identity + fail-closed exclusion ───────────────────────────────────

def test_residual_identity_and_exclusions():
    rows = [
        _row("2026-07-01", 0.010, 0.004),                        # aligned
        _row("2026-07-02", 0.006, None, hedged_gross=0.002),     # gross basis
        _row("2026-07-03", 0.005, None),                         # hedged unresolved
        _row("2026-07-04", None, 0.001),                         # raw unresolved
        _row("2026-07-05", 0.007, 0.003, status="no_valid_hedge"),
    ]
    series = residual_series(rows)
    assert len(series["points"]) == 2
    for p in series["points"]:
        assert p["residual"] + p["systematic"] == pytest.approx(p["raw"], abs=1e-12)
    p0 = series["points"][0]
    assert p0["residual"] == pytest.approx(0.004) and p0["systematic"] == pytest.approx(0.006)
    assert p0["basis"] == "net"
    assert series["points"][1]["basis"] == "gross_cost_unknown"
    reasons = {e["reason"] for e in series["excluded"]}
    assert {"hedged_unresolved", "raw_unresolved", "hedge_not_applied:no_valid_hedge"} <= reasons


# ── 3. phi semantics ────────────────────────────────────────────────────────

def test_pure_beta_strategy_gets_near_zero_credit():
    # Raw profitable, hedged ~= 0 -> return was the systematic exposure.
    rows = [_row(f"2026-07-{d:02d}", 0.010, 0.0005) for d in range(1, 9)]
    att = strategy_attribution(rows)
    assert att["residual_fraction"] == pytest.approx(0.05)
    res = residual_reward(1.0, att)
    assert res["applied"] is True
    assert res["reward"] == pytest.approx(0.05)


def test_pure_alpha_strategy_keeps_full_credit():
    rows = [_row(f"2026-07-{d:02d}", 0.010, 0.0098) for d in range(1, 9)]
    res = residual_reward(2.0, strategy_attribution(rows))
    assert res["applied"] is True
    assert res["multiplier"] == pytest.approx(0.98)
    assert res["reward"] == pytest.approx(1.96)


def test_multiplier_bounds():
    # Systematic exposure was HURTING the strategy: phi > 1, capped at 2.0.
    rows = [_row(f"2026-07-{d:02d}", 0.002, 0.008) for d in range(1, 9)]
    res = residual_reward(1.0, strategy_attribution(rows))
    assert res["phi"] == pytest.approx(4.0)
    assert res["multiplier"] == MULTIPLIER_CAP
    # Residual sign opposite to raw: floored at 0, never a sign flip.
    rows_neg = [_row(f"2026-07-{d:02d}", 0.004, -0.002) for d in range(1, 9)]
    res_neg = residual_reward(1.0, strategy_attribution(rows_neg))
    assert res_neg["multiplier"] == 0.0 and res_neg["reward"] == 0.0


def test_zero_raw_expectancy_fails_closed():
    rows = [_row("2026-07-01", 1e-12, 0.001)] * 6
    att = strategy_attribution(rows)
    assert att["residual_fraction"] is None
    res = residual_reward(1.0, att)
    assert res["applied"] is False and res["reward"] == 1.0


# ── 4. Pass-through reasons ─────────────────────────────────────────────────

def test_reward_pass_through_reasons():
    assert residual_reward(1.5, None) == {
        "reward": 1.5, "multiplier": 1.0, "applied": False,
        "reason": "no_lane_coverage", "phi": None,
    }
    short = strategy_attribution([_row("2026-07-01", 0.01, 0.005)] * 3)
    r = residual_reward(1.0, short)
    assert r["applied"] is False and r["reason"].startswith("insufficient_history")
    gross = strategy_attribution(
        [_row(f"2026-07-{d:02d}", 0.01, None, hedged_gross=0.005) for d in range(1, 9)])
    rg = residual_reward(1.0, gross)
    assert rg["applied"] is False
    assert rg["reason"] == "gross_basis_cycles_present_cost_unmodeled"


# ── 5. Exponential credit update ────────────────────────────────────────────

def test_exponential_credit_update_math_and_bounds():
    w = exponential_credit_update(1.0, signed_contribution=1.0, residual_return=0.02, eta=0.5)
    assert w == pytest.approx(math.exp(0.01))
    # AGAINST a positive residual -> weight shrinks symmetrically
    w2 = exponential_credit_update(1.0, -1.0, 0.02, eta=0.5)
    assert w2 == pytest.approx(math.exp(-0.01))
    # eta=0 is a no-op; bounds clamp; non-finite fails closed to unchanged
    assert exponential_credit_update(1.3, 1.0, 0.5, eta=0.0) == 1.3
    assert exponential_credit_update(1.9, 1.0, 10.0, eta=1.0) == 2.0
    assert exponential_credit_update(0.11, -1.0, 10.0, eta=1.0) == 0.1
    assert exponential_credit_update(1.2, 1.0, float("nan"), eta=0.5) == 1.2


# ── 6. Report build + loader round trip (real disk) ─────────────────────────

def test_report_build_and_loader_roundtrip(tmp_path):
    ledger = tmp_path / "raw_vs_hedged_ledger.jsonl"
    # 2026-07-19 scheduler-safety audit (finding 2): corruption FAILS CLOSED —
    # the report build refuses an unparseable ledger, never analyzes around it.
    with open(ledger, "w", encoding="utf-8") as fh:
        for d in range(1, 9):
            fh.write(json.dumps(_row(f"2026-07-{d:02d}", 0.010, 0.004)) + "\n")
        fh.write("{corrupt line\n")
    out = tmp_path / "residual_attribution_report.json"
    from src.evidence.forward_ledger import LedgerCorruptionError
    with pytest.raises(LedgerCorruptionError):
        build_residual_attribution_report(ledger_path=ledger, out_path=out)
    assert not out.exists(), "no report from a corrupt ledger"

    # clean ledger: full roundtrip
    with open(ledger, "w", encoding="utf-8") as fh:
        for d in range(1, 9):
            fh.write(json.dumps(_row(f"2026-07-{d:02d}", 0.010, 0.004)) + "\n")
    import pathlib
    pathlib.Path(str(ledger) + ".corrupt.json").unlink()
    report = build_residual_attribution_report(ledger_path=ledger, out_path=out)
    assert out.exists()
    assert report["n_ledger_rows"] == 8
    att = report["strategies"]["test_strat"]
    assert att["n_aligned_cycles"] == 8
    assert att["residual_fraction"] == pytest.approx(0.4)
    assert report["paper_only"] is True and report["runtime_allowed"] is False

    loaded = load_attribution_for("test_strat", report_path=out)
    assert loaded is not None and loaded["residual_fraction"] == pytest.approx(0.4)
    assert load_attribution_for("unknown_strategy", report_path=out) is None
    assert load_attribution_for("test_strat", report_path=tmp_path / "missing.json") is None
    # no temp files left behind by the atomic writer
    assert list(tmp_path.glob(".residual_attr_*")) == []


# ── 7. Shadow-first flag ────────────────────────────────────────────────────

def test_scanner_config_flag_exists_and_defaults_off():
    from src.scanner.config import ScannerConfig

    cfg = ScannerConfig()
    assert hasattr(cfg, "enable_residual_alpha_rewards")
    assert cfg.enable_residual_alpha_rewards is False
