"""Tests for the meta-pipeline plumbing: ApprovalQueue, StagedDeployer,
PostDeployCritic, and the ConfigAdjuster.revert_by_id extension.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.scanner.automation.approval_queue import ApprovalQueue
from src.scanner.automation.config_adjuster import ConfigAdjuster
from src.scanner.automation.meta_types import (
    ChangeKind,
    ChangePackage,
    ChangeStage,
    DeployStage,
    Proposal,
    Scorecard,
    SubScore,
)
from src.scanner.automation.post_deploy_critic import PostDeployCritic
from src.scanner.automation.staged_deployer import StagedDeployer


def _pkg(**kw) -> ChangePackage:
    pkg = ChangePackage(kind=ChangeKind.CONFIG, **kw)
    pkg.proposal = Proposal(config_delta={"min_confidence": {"old": 0.5, "new": 0.55}})
    return pkg


# ---- ApprovalQueue ----

def test_enqueue_writes_to_pending(tmp_path):
    q = ApprovalQueue(root=tmp_path)
    pkg = _pkg()
    path = q.enqueue(pkg)
    assert path.exists()
    assert path.parent.name == "pending"
    assert pkg.stage == ChangeStage.AWAITING_APPROVAL


def test_approve_moves_to_approved_dir(tmp_path):
    q = ApprovalQueue(root=tmp_path)
    pkg = _pkg()
    q.enqueue(pkg)
    out = q.approve(pkg.change_id, deploy_target=DeployStage.SHADOW, reason="smoke")
    assert out is not None
    assert out.stage == ChangeStage.APPROVED
    assert out.deploy_target == DeployStage.SHADOW
    assert not (tmp_path / "pending" / f"{pkg.change_id}.json").exists()
    assert (tmp_path / "approved" / f"{pkg.change_id}.json").exists()


def test_reject_moves_to_rejected_dir(tmp_path):
    q = ApprovalQueue(root=tmp_path)
    pkg = _pkg()
    q.enqueue(pkg)
    out = q.reject(pkg.change_id, reason="not now")
    assert out is not None
    assert out.stage == ChangeStage.REJECTED
    assert out.rejection_reason == "not now"
    assert (tmp_path / "rejected" / f"{pkg.change_id}.json").exists()


def test_drain_approved_clears_dir(tmp_path):
    q = ApprovalQueue(root=tmp_path)
    pkg = _pkg()
    q.enqueue(pkg)
    q.approve(pkg.change_id, deploy_target=DeployStage.SHADOW)
    drained = q.drain_approved()
    assert len(drained) == 1
    assert drained[0].change_id == pkg.change_id
    assert q.list_approved() == []


def test_get_finds_in_any_directory(tmp_path):
    q = ApprovalQueue(root=tmp_path)
    pkg = _pkg()
    q.enqueue(pkg)
    found = q.get(pkg.change_id)
    assert found is not None and found.change_id == pkg.change_id


# ---- StagedDeployer ----

def test_advance_shadow_writes_namespaced_attrs():
    cfg = SimpleNamespace()
    deployer = StagedDeployer(config=cfg, config_adjuster=None)
    pkg = _pkg(deploy_target=DeployStage.SHADOW)
    deployer.advance(pkg)
    assert hasattr(cfg, "shadow_min_confidence")
    assert getattr(cfg, "shadow_min_confidence") == 0.55
    assert pkg.stage == ChangeStage.DEPLOYED_SHADOW
    assert len(pkg.deployments) == 1


def test_advance_idempotent_per_stage():
    cfg = SimpleNamespace()
    deployer = StagedDeployer(config=cfg, config_adjuster=None)
    pkg = _pkg(deploy_target=DeployStage.SHADOW)
    deployer.advance(pkg)
    deployer.advance(pkg)
    assert len(pkg.deployments) == 1


def test_promote_next_walks_chain():
    cfg = SimpleNamespace()
    adjuster = ConfigAdjuster(persistence_path=Path("/tmp/_meta_test_adjuster.json"))
    adjuster._history.clear()
    adjuster._pending.clear()
    deployer = StagedDeployer(config=cfg, config_adjuster=adjuster, shadow_cycles=1, canary_trades=1)
    pkg = _pkg(deploy_target=DeployStage.SHADOW)
    deployer.advance(pkg)
    deployer.promote_next(pkg, current_cycle=20)
    assert pkg.deploy_target == DeployStage.CANARY
    assert pkg.stage == ChangeStage.DEPLOYED_CANARY
    deployer.promote_next(pkg, current_cycle=40)
    assert pkg.deploy_target == DeployStage.LIVE
    assert pkg.stage == ChangeStage.DEPLOYED_LIVE


def test_rollback_marks_record_and_undoes_shadow():
    cfg = SimpleNamespace()
    deployer = StagedDeployer(config=cfg, config_adjuster=None)
    pkg = _pkg(deploy_target=DeployStage.SHADOW)
    deployer.advance(pkg)
    assert hasattr(cfg, "shadow_min_confidence")
    deployer.rollback(pkg, reason="shadow_miss")
    assert pkg.stage == ChangeStage.REJECTED
    assert "shadow_miss" in (pkg.rejection_reason or "")
    assert pkg.deployments[-1].rolled_back_at is not None
    assert not hasattr(cfg, "shadow_min_confidence")


# ---- ConfigAdjuster.revert_by_id ----

def test_canary_routes_through_approver(tmp_path, caplog):
    """Full-chain integration: _apply_canary must move proposals from
    pending → approved-history via AdjustmentApprover, so apply_adjustments
    can read them without raising BypassAttempt (US-508 write-guard).
    """
    from src.scanner.automation.adjustment_approver import AdjustmentApprover

    pending = tmp_path / "pending.json"
    approved = tmp_path / "approved.json"
    cfg = SimpleNamespace(min_confidence=50.0)
    adjuster = ConfigAdjuster(
        persistence_path=approved,
        pending_path=pending,
    )
    approver = AdjustmentApprover(pending_path=pending, approved_path=approved)
    deployer = StagedDeployer(
        config=cfg,
        config_adjuster=adjuster,
        adjustment_approver=approver,
    )
    pkg = ChangePackage(kind=ChangeKind.CONFIG, deploy_target=DeployStage.CANARY)
    # Use in-bounds value (validator: min_confidence ∈ [10.0, 100.0])
    pkg.proposal = Proposal(config_delta={"min_confidence": {"old": 50.0, "new": 60.0}})

    deployer.advance(pkg, current_cycle=100)

    assert pkg.stage == ChangeStage.DEPLOYED_CANARY
    assert cfg.min_confidence == 60.0
    # And revert_by_id should pick up the now-recorded history entry
    reverted = adjuster.revert_by_id(cfg, source_substring=pkg.change_id)
    assert len(reverted) == 1
    assert cfg.min_confidence == 50.0


def test_canary_redeploy_with_same_change_id_actually_mutates(tmp_path):
    """Regression: ConfigAdjuster._write_pending_proposal short-circuits on a
    duplicate (source, key, value) signature. Before the fix, collect_adjustment
    still returned a fresh proposal_id whose row had never been written, causing
    AdjustmentApprover.approve() to fail to find it and the canary to silently
    no-op. Re-deploys (rollback → retry, or any idempotent re-advance) must
    still mutate config correctly.
    """
    from src.scanner.automation.adjustment_approver import AdjustmentApprover

    pending = tmp_path / "pending.json"
    approved = tmp_path / "approved.json"
    cfg = SimpleNamespace(min_confidence=50.0)
    adjuster = ConfigAdjuster(persistence_path=approved, pending_path=pending)
    approver = AdjustmentApprover(pending_path=pending, approved_path=approved)
    deployer = StagedDeployer(
        config=cfg,
        config_adjuster=adjuster,
        adjustment_approver=approver,
    )
    pkg = ChangePackage(kind=ChangeKind.CONFIG, deploy_target=DeployStage.CANARY)
    pkg.proposal = Proposal(config_delta={"min_confidence": {"old": 50.0, "new": 60.0}})

    # First deploy lands cleanly.
    deployer.advance(pkg, current_cycle=100)
    assert cfg.min_confidence == 60.0

    # Simulate a rollback — revert through the meta path, then redeploy the
    # SAME package. The (source, key, value) signature collides with the row
    # already in pending_adjustments.json. Pre-fix: collect_adjustment returns
    # a phantom id, approve() doesn't find it, apply_adjustments is a no-op.
    deployer.rollback(pkg, reason="forced_for_redeploy_test")
    assert cfg.min_confidence == 50.0

    # Reset stage so advance() will re-apply rather than be idempotent-skipped.
    pkg.deploy_target = DeployStage.CANARY
    pkg.stage = ChangeStage.APPROVED
    deployer.advance(pkg, current_cycle=200)
    assert cfg.min_confidence == 60.0, (
        "Re-deploy of same change_id failed to mutate config — "
        "duplicate-suppression in _write_pending_proposal returned a phantom id"
    )


def test_canary_skips_when_approver_missing(tmp_path, caplog):
    """Without an approver, the meta pipeline cannot move its own proposals
    past the human-approval gate. _apply_canary must log loudly and bail
    rather than letting apply_adjustments raise BypassAttempt.
    """
    import logging as _logging
    cfg = SimpleNamespace(min_confidence=50.0)
    adjuster = ConfigAdjuster(persistence_path=tmp_path / "approved.json")
    deployer = StagedDeployer(
        config=cfg,
        config_adjuster=adjuster,
        adjustment_approver=None,
    )
    pkg = ChangePackage(kind=ChangeKind.CONFIG, deploy_target=DeployStage.CANARY)
    pkg.proposal = Proposal(config_delta={"min_confidence": {"old": 50.0, "new": 60.0}})

    with caplog.at_level(_logging.ERROR, logger="src.scanner.automation.staged_deployer"):
        deployer.advance(pkg, current_cycle=100)

    assert any("canary_no_approver" in rec.message for rec in caplog.records)
    assert cfg.min_confidence == 50.0  # unchanged


def test_revert_by_id_restores_old_values(tmp_path):
    # revert_by_id walks self._history (in-memory list of applied adjustments)
    # in reverse and restores each entry's old_value. Seed history directly so
    # the test exercises the revert contract in isolation from the approver
    # pipeline (collect → AdjustmentApprover.approve → apply_adjustments).
    cfg = SimpleNamespace(min_confidence=60.0)  # current state after a hypothetical apply
    adjuster = ConfigAdjuster(persistence_path=tmp_path / "adj.json")
    adjuster._history.append({
        "key": "min_confidence",
        "old_value": 50.0,
        "new_value": 60.0,
        "source": "meta_manager_canary:abc123",
        "reason": "canary deploy",
        "cycle": 100,
    })
    reverted = adjuster.revert_by_id(cfg, source_substring="abc123")
    assert len(reverted) == 1
    assert cfg.min_confidence == 50.0


# ---- PostDeployCritic ----

def _scored_pkg(predicted_win_rate=0.6) -> ChangePackage:
    pkg = _pkg(deploy_target=DeployStage.SHADOW)
    pkg.scorecard = Scorecard(
        trading_quality=SubScore(
            name="trading_quality",
            passed=True,
            score=0.5,
            details={"win_rate": predicted_win_rate, "pnl_delta_r": 0.5, "sharpe": 0.4},
        )
    )
    return pkg


def test_critic_passes_when_realized_close(tmp_path):
    critic = PostDeployCritic(
        metrics_slicer=lambda p, s: {"sample_size": 12, "win_rate": 0.58, "pnl_delta_r": 0.3},
        ledger_path=tmp_path / "changes.jsonl",
    )
    pkg = _scored_pkg(predicted_win_rate=0.6)
    review = critic.review(pkg, DeployStage.SHADOW)
    assert review.passed is True


def test_critic_fails_when_realized_drifts(tmp_path):
    critic = PostDeployCritic(
        metrics_slicer=lambda p, s: {"sample_size": 10, "win_rate": 0.30, "pnl_delta_r": -3.0},
        ledger_path=tmp_path / "changes.jsonl",
    )
    pkg = _scored_pkg(predicted_win_rate=0.6)
    review = critic.review(pkg, DeployStage.SHADOW)
    assert review.passed is False
    assert "outside" in review.notes


def test_critic_treats_no_sample_as_pass(tmp_path):
    critic = PostDeployCritic(
        metrics_slicer=lambda p, s: {"sample_size": 0},
        ledger_path=tmp_path / "changes.jsonl",
    )
    review = critic.review(_scored_pkg(), DeployStage.SHADOW)
    assert review.passed is True


def test_critic_appends_ledger_jsonl(tmp_path):
    ledger = tmp_path / "changes.jsonl"
    critic = PostDeployCritic(
        metrics_slicer=lambda p, s: {"sample_size": 5, "win_rate": 0.6, "pnl_delta_r": 0.4},
        ledger_path=ledger,
    )
    pkg = _scored_pkg()
    critic.review(pkg, DeployStage.SHADOW)
    assert ledger.exists()
    record = json.loads(ledger.read_text().strip().splitlines()[-1])
    assert record["change_id"] == pkg.change_id
    assert record["stage"] == "shadow"


import hashlib
from src.scanner.automation.meta_types import (
    ChangePackage, DeploymentRecord, DeployStage, Proposal,
)


def test_deployment_record_has_regime_and_trade_count_fields():
    rec = DeploymentRecord(stage=DeployStage.SHADOW)
    assert rec.regime_at_deploy is None
    assert rec.closed_trade_count_at_deploy == 0
    rec.regime_at_deploy = "LOW"
    rec.closed_trade_count_at_deploy = 1234
    from dataclasses import asdict
    blob = asdict(rec)
    assert blob["regime_at_deploy"] == "LOW"
    assert blob["closed_trade_count_at_deploy"] == 1234


def test_change_package_dedup_hash_stable_and_collision_correct():
    delta = {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}}
    pkg_a = ChangePackage(
        incident={"kind": "tp_too_fast", "summary": "..."},
        proposal=Proposal(config_delta=delta),
    )
    pkg_b = ChangePackage(
        incident={"kind": "tp_too_fast", "summary": "different summary, same delta"},
        proposal=Proposal(config_delta=delta),
    )
    pkg_c = ChangePackage(
        incident={"kind": "sl_too_wide", "summary": "..."},
        proposal=Proposal(config_delta=delta),
    )
    assert pkg_a.dedup_hash() == pkg_b.dedup_hash()
    assert pkg_a.dedup_hash() != pkg_c.dedup_hash()
    assert pkg_a.dedup_hash() == pkg_a.dedup_hash()
    assert len(pkg_a.dedup_hash()) >= 16


def test_change_package_dedup_hash_handles_missing_proposal():
    pkg = ChangePackage(incident={"kind": "any"}, proposal=None)
    h = pkg.dedup_hash()
    assert isinstance(h, str) and len(h) >= 16
