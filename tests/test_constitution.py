"""Tests for the meta-pipeline Constitution invariant engine.

Each clause in the seed constitution is verified with at least one
known-violating ChangePackage and at least one known-passing case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.constitution import Constitution
from src.scanner.automation.meta_types import (
    ChangePackage,
    ChangeKind,
    DeployStage,
    Proposal,
    Scorecard,
    SubScore,
)


SEED_CONSTITUTION_PATH = Path(__file__).resolve().parent.parent / ".claude" / "rules" / "constitution.json"


@pytest.fixture
def seeded_constitution(tmp_path):
    """Use the real seed constitution shipped in .claude/rules/."""
    return Constitution(path=SEED_CONSTITUTION_PATH)


def _pkg(
    *,
    config_delta=None,
    changed_files=None,
    deploy_target=DeployStage.SHADOW,
    rollback_plan="rollback by reverting via ConfigAdjuster.revert_by_id",
    scorecard=None,
    kind=ChangeKind.CONFIG,
) -> ChangePackage:
    p = ChangePackage(kind=kind, deploy_target=deploy_target, rollback_plan=rollback_plan)
    p.proposal = Proposal(
        diff="",
        changed_files=list(changed_files or []),
        config_delta=dict(config_delta or {}),
    )
    p.scorecard = scorecard
    return p


def test_constitution_loads_seed(seeded_constitution):
    assert len(seeded_constitution.clauses) >= 7
    ids = {c["id"] for c in seeded_constitution.clauses}
    assert {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}.issubset(ids)


def test_c1_widening_max_risk_fails(seeded_constitution):
    pkg = _pkg(config_delta={"max_portfolio_risk": {"old": 0.15, "new": 0.20}})
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C1" in failing
    assert not att.passed


def test_c1_tightening_max_risk_passes(seeded_constitution):
    pkg = _pkg(config_delta={"max_portfolio_risk": {"old": 0.15, "new": 0.10}})
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C1" not in failing


def test_c2_disable_supervision_fails(seeded_constitution):
    # 2026-04-27 audit: C2 was rebound from `enable_supervision_mode` (orphan)
    # to `enable_meta_manager` (real). The cybernetic loop must not be able
    # to disable itself via a meta-pipeline change.
    pkg = _pkg(config_delta={"enable_meta_manager": {"old": True, "new": False}})
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C2" in failing


def test_c2_enable_supervision_passes(seeded_constitution):
    pkg = _pkg(config_delta={"enable_meta_manager": {"old": False, "new": True}})
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C2" not in failing


def test_c3_freshness_floor_blow_through_fails(seeded_constitution):
    # 2026-04-27 audit: C3 was rebound from `max_component_age_days` (orphan)
    # to `staleness_age_threshold_days` (real). Constant raised 7 → 14 to
    # match the operating threshold cited in real incidents.
    pkg = _pkg(config_delta={"staleness_age_threshold_days": {"old": 5, "new": 20}})
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C3" in failing


def test_c3_freshness_within_bound_passes(seeded_constitution):
    pkg = _pkg(config_delta={"staleness_age_threshold_days": {"old": 5, "new": 14}})
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C3" not in failing


def test_c4_low_eval_confidence_fails(seeded_constitution):
    score = Scorecard(
        trading_quality=SubScore(
            name="trading_quality",
            passed=False,
            score=0.4,
            details={"win_rate": 0.42},
        )
    )
    pkg = _pkg(scorecard=score)
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C4" in failing


def test_c4_high_eval_confidence_passes(seeded_constitution):
    score = Scorecard(
        trading_quality=SubScore(
            name="trading_quality",
            passed=True,
            score=0.62,
            details={"win_rate": 0.62},
        )
    )
    pkg = _pkg(scorecard=score)
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C4" not in failing


def test_c5_broker_change_without_test_fails(seeded_constitution):
    pkg = _pkg(changed_files=["src/scanner/execution.py"])
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C5" in failing


def test_c5_broker_change_with_test_passes(seeded_constitution):
    pkg = _pkg(
        changed_files=[
            "src/scanner/execution.py",
            "tests/scanner/test_execution_smoke.py",
        ]
    )
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C5" not in failing


def test_c6_live_deploy_without_rollback_fails(seeded_constitution):
    pkg = _pkg(deploy_target=DeployStage.LIVE, rollback_plan="")
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C6" in failing


def test_c6_shadow_deploy_without_rollback_passes(seeded_constitution):
    pkg = _pkg(deploy_target=DeployStage.SHADOW, rollback_plan="")
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C6" not in failing


def test_c7_rr_floor_violation_fails(seeded_constitution):
    # 2026-04-27 audit: C7 was rebound from `min_rr_ratio` (orphan) to
    # `min_risk_reward_ratio` (real ScannerConfig field). Enforces the
    # CLAUDE.md hard-gate floor of 1.2:1 R:R.
    pkg = _pkg(config_delta={"min_risk_reward_ratio": {"old": 1.2, "new": 1.0}})
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C7" in failing


def test_c7_rr_floor_held_passes(seeded_constitution):
    pkg = _pkg(config_delta={"min_rr_ratio": {"old": 1.2, "new": 1.5}})
    att = seeded_constitution.check(pkg)
    failing = {c.clause_id for c in att.failed_clauses()}
    assert "C7" not in failing


def test_unrelated_keys_dont_trip_pattern_clauses(seeded_constitution):
    pkg = _pkg(config_delta={"min_confidence": {"old": 0.5, "new": 0.55}})
    att = seeded_constitution.check(pkg)
    assert att.passed, [c for c in att.failed_clauses()]


def test_invariant_evaluation_doesnt_crash_on_partial_package(seeded_constitution):
    pkg = ChangePackage()
    att = seeded_constitution.check(pkg)
    assert att.passed


def test_historical_loss_rate_clause_vetoes_above_threshold(tmp_path):
    """G7: incident.historical_outcomes with loss_rate > threshold and
    matched_episodes >= n_min triggers a veto."""
    rules_path = tmp_path / "constitution.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "clauses": [{
            "id": "hist_loss_rate_test",
            "kind": "historical_loss_rate",
            "name": "Historical loss rate veto",
            "rule": "block proposals matching past loss patterns",
            "threshold": 0.7,
            "n_min": 5,
        }],
    }))
    c = Constitution(path=rules_path)
    pkg = ChangePackage(
        incident={
            "kind": "any",
            "historical_outcomes": {
                "matched_episodes": 8,
                "loss_rate": 0.75,
            },
        }
    )
    attestation = c.check(pkg)
    failed_ids = [r.clause_id for r in attestation.clauses if not r.passed]
    assert "hist_loss_rate_test" in failed_ids
    assert not attestation.passed


def test_historical_loss_rate_clause_abstains_on_insufficient_history(tmp_path):
    """G7: when matched_episodes < n_min, the clause abstains
    (passes=True with detail mentioning insufficient_history)."""
    rules_path = tmp_path / "constitution.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "clauses": [{
            "id": "hist_loss_rate_test",
            "kind": "historical_loss_rate",
            "name": "Historical loss rate veto",
            "rule": "block proposals matching past loss patterns",
            "threshold": 0.7,
            "n_min": 5,
        }],
    }))
    c = Constitution(path=rules_path)
    pkg = ChangePackage(
        incident={
            "kind": "any",
            "historical_outcomes": {
                "matched_episodes": 3,  # < n_min (5)
                "loss_rate": 0.95,
            },
        }
    )
    attestation = c.check(pkg)
    matching = [r for r in attestation.clauses if r.clause_id == "hist_loss_rate_test"]
    assert matching and matching[0].passed is True
    assert "insufficient_history" in (matching[0].detail or "")


def test_historical_loss_rate_clause_handles_malformed_history(tmp_path):
    """Hardening: when historical_outcomes is the wrong type (list) or has
    non-numeric values, the clause abstains with a 'malformed_history' detail
    instead of raising. Defensive against intake-side schema corruption."""
    rules_path = tmp_path / "constitution.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "clauses": [{
            "id": "hist_loss_rate_test",
            "kind": "historical_loss_rate",
            "name": "Historical loss rate veto",
            "rule": "block proposals matching past loss patterns",
            "threshold": 0.7,
            "n_min": 5,
        }],
    }))
    c = Constitution(path=rules_path)

    # Case 1: historical_outcomes is a list, not a dict.
    pkg_list = ChangePackage(incident={"kind": "any", "historical_outcomes": ["broken"]})
    att1 = c.check(pkg_list)
    matching = [r for r in att1.clauses if r.clause_id == "hist_loss_rate_test"]
    assert matching and matching[0].passed is True
    assert "malformed_history" in (matching[0].detail or "")

    # Case 2: numeric coercion fails on non-numeric strings.
    pkg_str = ChangePackage(incident={
        "kind": "any",
        "historical_outcomes": {"matched_episodes": "not_a_number", "loss_rate": 0.9},
    })
    att2 = c.check(pkg_str)
    matching = [r for r in att2.clauses if r.clause_id == "hist_loss_rate_test"]
    assert matching and matching[0].passed is True
    assert "malformed_history" in (matching[0].detail or "")
