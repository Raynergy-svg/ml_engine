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
    DeploymentRecord,
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


def test_critic_defers_no_sample(tmp_path):
    critic = PostDeployCritic(
        metrics_slicer=lambda p, s: {"sample_size": 0},
        ledger_path=tmp_path / "changes.jsonl",
    )
    review = critic.review(_scored_pkg(), DeployStage.SHADOW)
    assert review.passed is False
    assert review.notes == "insufficient_sample_deferred"


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


def test_change_package_round_trips_deployment_regime_and_trade_count():
    """Deployment metadata must survive disk persistence.

    The post-deploy critic and trade-count promotion gates depend on these
    fields after `ApprovalQueue` and `MetaManager` reload packages from JSON.
    """
    pkg = ChangePackage()
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.SHADOW,
        deployed_at_cycle=42,
        closed_trade_count_at_deploy=1234,
        regime_at_deploy="LOW",
    ))

    restored = ChangePackage.from_dict(pkg.to_dict())
    rec = restored.deployments[0]

    assert rec.deployed_at_cycle == 42
    assert rec.closed_trade_count_at_deploy == 1234
    assert rec.regime_at_deploy == "LOW"


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
    assert len(pkg_a.dedup_hash()) == 16


def test_change_package_dedup_hash_handles_missing_proposal():
    pkg = ChangePackage(incident={"kind": "any"}, proposal=None)
    h = pkg.dedup_hash()
    assert isinstance(h, str) and len(h) == 16


def test_change_package_dedup_hash_canonicalizes_nested_dict_order():
    """Two deltas with the same nested dict but different key insertion
    order must produce the SAME hash (G8 anti-dup intent)."""
    pkg_a = ChangePackage(
        incident={"kind": "tp_too_fast"},
        proposal=Proposal(config_delta={
            "atr_tp_multiplier": {"new": 1.6, "old": 1.5},
        }),
    )
    pkg_b = ChangePackage(
        incident={"kind": "tp_too_fast"},
        proposal=Proposal(config_delta={
            "atr_tp_multiplier": {"old": 1.5, "new": 1.6},  # reversed key order
        }),
    )
    assert pkg_a.dedup_hash() == pkg_b.dedup_hash()


def test_staged_deployer_shadow_promote_requires_real_trades(monkeypatch):
    """Shadow → canary requires >= SHADOW_MIN_TRADES (15) closed trades
    since deploy AND R_mean >= R_baseline - R_FLOOR_DELTA (0.5)."""
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import (
        ChangePackage, DeploymentRecord, DeployStage, Proposal,
    )
    sd = StagedDeployer(config=None)
    pkg = ChangePackage(proposal=Proposal(config_delta={"atr_tp_multiplier": {"new": 1.6, "old": 1.5}}))
    rec = DeploymentRecord(
        stage=DeployStage.SHADOW,
        deployed_at_cycle=0,
        closed_trade_count_at_deploy=0,
    )
    pkg.deployments.append(rec)
    # 0 trades → not enough.
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=0) is False
    # 14 trades (< 15) → not enough.
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=14) is False
    # 15+ trades AND R_mean >= R_baseline - 0.5 → promote.
    monkeypatch.setattr(
        StagedDeployer, "_compute_window_r_stats",
        lambda self, rec, n: {"R_mean": 0.6, "R_baseline": 0.0},
    )
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=15) is True
    # 15+ trades but R below floor → no.
    monkeypatch.setattr(
        StagedDeployer, "_compute_window_r_stats",
        lambda self, rec, n: {"R_mean": -0.6, "R_baseline": 0.0},
    )
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=15) is False


def test_staged_deployer_canary_to_live_requires_30_trades(monkeypatch):
    """Canary → live requires >= CANARY_MIN_TRADES (30) closed trades
    since canary deploy AND R_mean >= R_baseline - R_FLOOR_DELTA."""
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import (
        ChangePackage, DeploymentRecord, DeployStage, Proposal,
    )
    sd = StagedDeployer(config=None)
    pkg = ChangePackage(proposal=Proposal(config_delta={}))
    rec = DeploymentRecord(
        stage=DeployStage.CANARY,
        deployed_at_cycle=0,
        closed_trade_count_at_deploy=100,
    )
    pkg.deployments.append(rec)
    monkeypatch.setattr(
        StagedDeployer, "_compute_window_r_stats",
        lambda self, rec, n: {"R_mean": 0.5, "R_baseline": 0.0},
    )
    # 100 + 29 = 129 — 1 short of 30.
    assert sd.should_promote_canary_to_live(pkg, current_trade_count=129) is False
    # 100 + 30 = 130 — exactly enough.
    assert sd.should_promote_canary_to_live(pkg, current_trade_count=130) is True


def test_staged_deployer_should_promote_returns_false_without_active_record(monkeypatch):
    """If there's no active record at the relevant stage (e.g., already
    rolled back), promotion is False — guards against operating on a
    non-existent deploy record."""
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import ChangePackage, Proposal
    sd = StagedDeployer(config=None)
    pkg = ChangePackage(proposal=Proposal(config_delta={}))
    # No deployments at all.
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=999) is False
    assert sd.should_promote_canary_to_live(pkg, current_trade_count=999) is False


def test_apply_shadow_captures_regime(monkeypatch):
    """At promotion-to-shadow, DeploymentRecord.regime_at_deploy is
    snapshotted from regime_detector.current() and
    closed_trade_count_at_deploy from the trade journal."""
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import (
        ChangePackage, DeployStage, Proposal,
    )

    class FakeConfig:
        pass

    class FakeDetector:
        def current(self):
            return "LOW"

    sd = StagedDeployer(config=FakeConfig(), regime_detector=FakeDetector())
    # Stub the trade-count read so the captured count is deterministic.
    monkeypatch.setattr(StagedDeployer, "_read_trade_journal_count", lambda self: 1234)
    pkg = ChangePackage(proposal=Proposal(config_delta={"atr_tp_multiplier": {"new": 1.6, "old": 1.5}}))
    pkg.deploy_target = DeployStage.SHADOW
    sd.advance(pkg, current_cycle=42)
    rec = pkg.deployments[-1]
    assert rec.regime_at_deploy == "LOW"
    assert rec.closed_trade_count_at_deploy == 1234
    assert rec.deployed_at_cycle == 42  # existing field still populated


def test_apply_shadow_handles_none_regime_detector(monkeypatch):
    """Without an injected regime_detector, regime_at_deploy stays None
    (no crash). closed_trade_count_at_deploy is still captured."""
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import (
        ChangePackage, DeployStage, Proposal,
    )

    sd = StagedDeployer(config=None)  # no regime_detector
    monkeypatch.setattr(StagedDeployer, "_read_trade_journal_count", lambda self: 0)
    pkg = ChangePackage(proposal=Proposal(config_delta={}))
    pkg.deploy_target = DeployStage.SHADOW
    sd.advance(pkg, current_cycle=0)
    rec = pkg.deployments[-1]
    assert rec.regime_at_deploy is None
    assert rec.closed_trade_count_at_deploy == 0


def test_apply_shadow_handles_regime_detector_exception(monkeypatch, caplog):
    """If regime_detector.current() raises, capture falls back to None and
    a warning is logged. Promotion proceeds (regime tagging is observability,
    not gating)."""
    import logging
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import (
        ChangePackage, DeployStage, Proposal,
    )

    class CrashyDetector:
        def current(self):
            raise RuntimeError("regime detector down")

    sd = StagedDeployer(config=None, regime_detector=CrashyDetector())
    monkeypatch.setattr(StagedDeployer, "_read_trade_journal_count", lambda self: 50)
    pkg = ChangePackage(proposal=Proposal(config_delta={}))
    pkg.deploy_target = DeployStage.SHADOW
    with caplog.at_level(logging.WARNING):
        sd.advance(pkg, current_cycle=10)
    rec = pkg.deployments[-1]
    assert rec.regime_at_deploy is None
    assert rec.closed_trade_count_at_deploy == 50
    assert any("regime_capture_failed" in r.message for r in caplog.records)


def test_compute_window_r_stats_returns_zeros_on_stale_deploy_index(monkeypatch, tmp_path):
    """Hardening: if the trade journal was trimmed/rotated since deploy,
    deploy_idx may exceed the journal length. The helper must return
    zeros (and log a warning) rather than producing empty/garbage windows."""
    import json
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import DeploymentRecord, DeployStage
    # Synthesize a tiny journal at the canonical path.
    journal_path = tmp_path / "trained_data" / "trade_journal_rl.json"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(json.dumps([
        {"outcome": {"r_multiple": 0.5}},
        {"outcome": {"r_multiple": -0.3}},
    ]))
    # Re-point the helper's path resolution.
    from src.scanner.automation import staged_deployer as sd_mod
    monkeypatch.setattr(
        sd_mod, "Path", lambda *a, **k: type("FakePath", (), {
            "resolve": lambda self: type("Resolved", (), {
                "parents": [None, None, None, tmp_path],
            })(),
        })(),
        raising=False,
    )
    sd = StagedDeployer(config=None)
    # Record claims deploy_idx=100, but journal only has 2 entries.
    rec = DeploymentRecord(
        stage=DeployStage.SHADOW,
        closed_trade_count_at_deploy=100,
    )
    stats = sd._compute_window_r_stats(rec, 15)
    # Must not produce garbage; must return zeros on the stale-index path.
    assert stats == {"R_mean": 0.0, "R_baseline": 0.0}


def test_metrics_slicer_includes_r_mean(tmp_path, monkeypatch):
    """Slicer adds r_mean alongside win_rate and pnl_total_pips. Closes G9
    (post-deploy critic now reports R-multiple — the metric Phase 1 actually
    cares about for SL/TP-modifying deltas)."""
    import json
    from src.scanner.automation import post_deploy_critic as pdc_mod
    from src.scanner.automation.post_deploy_critic import _default_metrics_slicer
    from src.scanner.automation.meta_types import (
        ChangePackage, DeploymentRecord, DeployStage,
    )
    target = tmp_path / "trained_data" / "trade_journal_rl.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-26T10:00:00", "outcome": {"trade_won": True, "pnl_pips": 10, "r_multiple": 1.0}},
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-27T10:00:00", "outcome": {"trade_won": True, "pnl_pips": 15, "r_multiple": 1.5}},
        {"pair": "EUR_USD", "direction": "short", "timestamp": "2026-04-28T10:00:00", "outcome": {"trade_won": False, "pnl_pips": -5, "r_multiple": -0.5}},
    ]))
    monkeypatch.setattr(pdc_mod, "_PROJECT_ROOT", tmp_path)

    pkg = ChangePackage()
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.SHADOW,
        deployed_at="2026-04-26T00:00:00",
    ))
    out = _default_metrics_slicer(pkg, DeployStage.SHADOW)
    assert out["sample_size"] == 3
    assert out["win_rate"] == pytest.approx(2/3)
    # NEW: r_mean must be present and correctly averaged.
    assert out["r_mean"] == pytest.approx((1.0 + 1.5 + -0.5) / 3)


def test_metrics_slicer_slices_by_regime_when_present(tmp_path, monkeypatch):
    """When DeploymentRecord.regime_at_deploy is set, slicer adds a
    'regime_slices' breakdown keyed by the regime found in each trade's
    outcome dict. The breakdown enables the 14-day deferred-G2 review
    (per-regime distribution comparison)."""
    import json
    from src.scanner.automation import post_deploy_critic as pdc_mod
    from src.scanner.automation.post_deploy_critic import _default_metrics_slicer
    from src.scanner.automation.meta_types import (
        ChangePackage, DeploymentRecord, DeployStage,
    )
    target = tmp_path / "trained_data" / "trade_journal_rl.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-26T10:00:00", "outcome": {"trade_won": True, "pnl_pips": 10, "r_multiple": 1.0, "regime": "LOW"}},
        {"pair": "EUR_USD", "direction": "short", "timestamp": "2026-04-27T10:00:00", "outcome": {"trade_won": False, "pnl_pips": -5, "r_multiple": -0.5, "regime": "HIGH"}},
    ]))
    monkeypatch.setattr(pdc_mod, "_PROJECT_ROOT", tmp_path)

    pkg = ChangePackage()
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.SHADOW,
        deployed_at="2026-04-26T00:00:00",
        regime_at_deploy="LOW",
    ))
    out = _default_metrics_slicer(pkg, DeployStage.SHADOW)
    assert "regime_slices" in out
    assert "LOW" in out["regime_slices"]
    assert out["regime_slices"]["LOW"]["sample_size"] == 1
    assert out["regime_slices"]["LOW"]["win_rate"] == 1.0
    assert out["regime_slices"]["LOW"]["r_mean"] == pytest.approx(1.0)
    assert "HIGH" in out["regime_slices"]
    assert out["regime_slices"]["HIGH"]["sample_size"] == 1
    assert out["regime_slices"]["HIGH"]["r_mean"] == pytest.approx(-0.5)


def test_metrics_slicer_no_regime_slices_when_record_lacks_regime(tmp_path, monkeypatch):
    """When regime_at_deploy is None on the DeploymentRecord, no
    regime_slices key is added. Backward-compat for pre-Task-7 records."""
    import json
    from src.scanner.automation import post_deploy_critic as pdc_mod
    from src.scanner.automation.post_deploy_critic import _default_metrics_slicer
    from src.scanner.automation.meta_types import (
        ChangePackage, DeploymentRecord, DeployStage,
    )
    target = tmp_path / "trained_data" / "trade_journal_rl.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-27T10:00:00", "outcome": {"trade_won": True, "pnl_pips": 10, "r_multiple": 1.0, "regime": "LOW"}},
    ]))
    monkeypatch.setattr(pdc_mod, "_PROJECT_ROOT", tmp_path)

    pkg = ChangePackage()
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.SHADOW,
        deployed_at="2026-04-26T00:00:00",
        regime_at_deploy=None,  # explicit
    ))
    out = _default_metrics_slicer(pkg, DeployStage.SHADOW)
    assert "regime_slices" not in out
    # But r_mean should still be present (that part is unconditional).
    assert "r_mean" in out


def test_critic_writes_calibration_feedback_on_overshoot(tmp_path, monkeypatch):
    """G4: when actual win_rate or r_mean diverges from scorecard.predicted
    beyond WIN_RATE_TOLERANCE / PNL_DELTA_TOLERANCE_R, an append-only
    calibration entry is written to model_bandit.json keyed by change_id."""
    from src.scanner.automation import post_deploy_critic as pdc_mod

    bandit_path = tmp_path / "trained_data" / "model_bandit.json"
    bandit_path.parent.mkdir(parents=True, exist_ok=True)
    bandit_path.write_text("{}")
    monkeypatch.setattr(pdc_mod, "_PROJECT_ROOT", tmp_path)

    # Inject a synthetic slicer so we don't depend on the trade journal.
    def slicer(pkg, stage):
        return {
            "sample_size": 30,
            "win_rate": 0.4,         # actual far below predicted 0.6
            "pnl_total_pips": -50.0,
            "r_mean": -0.5,           # actual far below predicted 0.5
        }

    critic = PostDeployCritic(
        metrics_slicer=slicer,
        ledger_path=tmp_path / "changes.jsonl",
    )
    pkg = ChangePackage(change_id="testchange01")
    pkg.scorecard = Scorecard(
        trading_quality=SubScore(
            name="trading_quality",
            passed=True,
            score=0.5,
            details={"win_rate": 0.6, "pnl_delta_r": 0.5, "sharpe": 0.4, "r_mean": 0.5},
        )
    )
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.LIVE,
        deployed_at="2026-04-26T00:00:00",
        regime_at_deploy="LOW",
    ))

    critic.review(pkg, DeployStage.LIVE)

    # Assert calibration_feedback was written.
    blob = json.loads(bandit_path.read_text())
    assert "calibration_feedback" in blob
    entry = blob["calibration_feedback"]["testchange01"]
    assert entry["predicted_win_rate"] == 0.6
    assert entry["actual_win_rate"] == 0.4
    assert entry["win_rate_overshoot"] == pytest.approx(-0.2, abs=1e-9)
    assert entry["predicted_r_mean"] == 0.5
    assert entry["actual_r_mean"] == -0.5
    assert entry["r_overshoot"] == pytest.approx(-1.0, abs=1e-9)
    assert entry["regime_at_deploy"] == "LOW"
    assert entry["n_trades"] == 30
    assert "ts" in entry  # ISO timestamp present


def test_critic_does_not_write_calibration_when_within_tolerance(tmp_path, monkeypatch):
    """When actuals are within WIN_RATE_TOLERANCE and PNL_DELTA_TOLERANCE_R
    of predicted, no calibration entry is written. Avoids noise in
    model_bandit.json."""
    from src.scanner.automation import post_deploy_critic as pdc_mod

    bandit_path = tmp_path / "trained_data" / "model_bandit.json"
    bandit_path.parent.mkdir(parents=True, exist_ok=True)
    bandit_path.write_text("{}")
    monkeypatch.setattr(pdc_mod, "_PROJECT_ROOT", tmp_path)

    def slicer(pkg, stage):
        return {
            "sample_size": 30,
            "win_rate": 0.55,  # 0.05 below predicted 0.6 — within 0.10 tolerance
            "pnl_total_pips": 5.0,
            "r_mean": 0.6,      # 0.1 above predicted 0.5 — within 1.5 R tolerance
        }

    critic = PostDeployCritic(
        metrics_slicer=slicer,
        ledger_path=tmp_path / "changes.jsonl",
    )
    pkg = ChangePackage(change_id="testchange02")
    pkg.scorecard = Scorecard(
        trading_quality=SubScore(
            name="trading_quality",
            passed=True,
            score=0.5,
            details={"win_rate": 0.6, "pnl_delta_r": 0.5, "sharpe": 0.4, "r_mean": 0.5},
        )
    )
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.LIVE,
        deployed_at="2026-04-26T00:00:00",
    ))

    critic.review(pkg, DeployStage.LIVE)

    blob = json.loads(bandit_path.read_text())
    # Either no calibration_feedback key, or the testchange02 entry is absent.
    assert "testchange02" not in blob.get("calibration_feedback", {})
