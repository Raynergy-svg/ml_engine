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

def test_revert_by_id_restores_old_values(tmp_path):
    cfg = SimpleNamespace(min_confidence=0.5)
    adjuster = ConfigAdjuster(persistence_path=tmp_path / "adj.json")
    adjuster.collect_adjustment(
        source="meta_manager_canary:abc123",
        key="min_confidence",
        value=0.6,
        reason="canary deploy",
        cycle=0,
    )
    adjuster.apply_adjustments(cfg, current_cycle=100)
    assert cfg.min_confidence == 0.6
    reverted = adjuster.revert_by_id(cfg, source_substring="abc123")
    assert len(reverted) == 1
    assert cfg.min_confidence == 0.5


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
