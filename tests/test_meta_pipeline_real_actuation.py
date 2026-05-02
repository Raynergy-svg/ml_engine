"""Real-disk end-to-end test for the autonomous loop final yard.

Mythos audit 2026-05-01 — proves that a meta-pipeline ChangePackage
promoted to LIVE actually mutates a live ScannerConfig instance, both
in-memory (setattr on the same instance the StagedDeployer holds) and
on-disk (config_adjustments.json["history"] gets the new entry that
EmbeddedScanner's own ConfigAdjuster reads on the next cycle).

NO MOCKS. Real ScannerConfig, real ConfigAdjuster, real
AdjustmentApprover, real StagedDeployer, real disk via tmp_path.

This test would have caught the 2026-04-30 bug where StagedDeployer
was constructed in MetaManager without `config_adjuster=ConfigAdjuster()`
— 11 packages walked through shadow → canary → live → closed cleanly
with `live_no_adjuster` warnings and zero actual config mutation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.adjustment_approver import AdjustmentApprover
from src.scanner.automation.config_adjuster import ConfigAdjuster
from src.scanner.automation.meta_types import (
    ChangePackage,
    ChangeStage,
    DeployStage,
    Proposal,
)
from src.scanner.automation.staged_deployer import StagedDeployer
from src.scanner.config import ScannerConfig


@pytest.fixture
def tmp_disk(tmp_path: Path) -> dict:
    """Real disk paths shared by ConfigAdjuster + AdjustmentApprover.

    Constructor args (no monkeypatch needed). The two collaborators
    MUST use the same pending + approved paths or
    AdjustmentApprover.approve(pid) won't find the proposal that
    ConfigAdjuster.collect_adjustment wrote.
    """
    return {
        "approved": tmp_path / "config_adjustments.json",
        "pending": tmp_path / "pending_adjustments.json",
        "root": tmp_path,
    }


def _make_promoted_package(config_delta: dict) -> ChangePackage:
    """Build a real ChangePackage staged at APPROVED with config_delta.

    Produced by upstream pipeline stages (intake → diagnosis → proposal →
    eval → policy → human approval) — this test exercises the LAST yard:
    StagedDeployer.advance shadow → canary → live → live_apply.
    """
    pkg = ChangePackage()
    pkg.stage = ChangeStage.APPROVED
    pkg.deploy_target = DeployStage.LIVE
    pkg.proposal = Proposal(
        diff="",
        changed_files=[],
        config_delta=config_delta,
        branch=None,
        worktree_path=None,
    )
    return pkg


def test_promoted_package_actually_mutates_live_config(tmp_disk: dict) -> None:
    """Real disk + real classes: promoted ChangePackage mutates the live ScannerConfig.

    REGRESSION GUARD: pre-fix MetaManager.build constructed StagedDeployer
    without config_adjuster=ConfigAdjuster(). Result: _apply_canary +
    _apply_live early-returned at `if not self._adjuster:` and the
    config_delta never reached the running ScannerConfig. With the fix
    in meta_manager.py:814 (config_adjuster=ConfigAdjuster() now wired),
    this test asserts the actual setattr happens.
    """
    # Real running config (mirrors what Scanner.__init__ holds).
    cfg = ScannerConfig()
    baseline_min_conf = cfg.min_confidence
    baseline_wvs = cfg.weighted_vote_threshold

    # Real ConfigAdjuster + AdjustmentApprover sharing the SAME pending +
    # approved paths. Production wiring: EmbeddedScanner's own
    # ConfigAdjuster reads the same on-disk file the meta-pipeline's
    # collaborators write to.
    adjuster = ConfigAdjuster(
        persistence_path=tmp_disk["approved"],
        pending_path=tmp_disk["pending"],
    )
    approver = AdjustmentApprover(
        pending_path=tmp_disk["pending"],
        approved_path=tmp_disk["approved"],
    )

    # Real StagedDeployer wired exactly like MetaManager.build does it
    # post-fix. shadow_cycles=0 + canary_trades=0 so the soak gates
    # don't block the test on real cycle/trade counts.
    deployer = StagedDeployer(
        config=cfg,
        config_adjuster=adjuster,
        adjustment_approver=approver,
        shadow_cycles=0,
        canary_trades=0,
    )

    # Promoted ChangePackage with a real config_delta the pipeline
    # could plausibly produce (auto-halt loss-streak proposed these
    # exact two keys on 2026-04-30).
    pkg = _make_promoted_package({
        "min_confidence": baseline_min_conf - 5.0,
        "weighted_vote_threshold": baseline_wvs - 0.05,
    })

    # Drive the deployer through stages. With cycles=trades=0 the gates
    # pass immediately so a single advance() can leap shadow→live.
    pkg.stage = ChangeStage.DEPLOYED_SHADOW
    for _ in range(4):
        deployer.advance(pkg, current_cycle=1)
        if pkg.stage in (ChangeStage.DEPLOYED_LIVE, ChangeStage.CLOSED):
            break

    assert pkg.stage in (ChangeStage.DEPLOYED_LIVE, ChangeStage.CLOSED), (
        f"deployer never reached live; pkg.stage={pkg.stage}"
    )

    # ASSERTION 1 (the regression guard): live ScannerConfig instance
    # was actually mutated by setattr — pre-fix this would have been
    # untouched because _apply_canary early-returned on _adjuster=None.
    assert cfg.min_confidence == pytest.approx(baseline_min_conf - 5.0), (
        f"live config NOT mutated — min_confidence still {cfg.min_confidence} "
        f"(expected {baseline_min_conf - 5.0}). The autonomous loop's last "
        f"yard is open-circuit. Check StagedDeployer._adjuster wiring in "
        f"MetaManager.build (meta_manager.py:814)."
    )
    assert cfg.weighted_vote_threshold == pytest.approx(baseline_wvs - 0.05), (
        f"live config NOT mutated — weighted_vote_threshold still "
        f"{cfg.weighted_vote_threshold}"
    )

    # ASSERTION 2: on-disk config_adjustments.json["history"] has the
    # new entries — this is the bridge EmbeddedScanner's own ConfigAdjuster
    # reads on each cycle to mutate the live Scanner config in the
    # production runtime.
    assert tmp_disk["approved"].exists(), "config_adjustments.json was not written"
    on_disk = json.loads(tmp_disk["approved"].read_text())
    history = on_disk.get("history", [])
    history_keys = {entry.get("key") for entry in history}
    assert "min_confidence" in history_keys, (
        f"min_confidence not in on-disk history; bridge to EmbeddedScanner "
        f"is broken. on-disk history keys: {history_keys}"
    )
    assert "weighted_vote_threshold" in history_keys, (
        f"weighted_vote_threshold not in on-disk history; on-disk history "
        f"keys: {history_keys}"
    )


def test_apply_live_without_adjuster_does_not_mutate(tmp_disk: dict) -> None:
    """Negative control: prove _adjuster=None silently no-ops.

    This is what was happening in production for the 11 packages
    promoted on 2026-04-30. Asserts the bug shape so the regression
    guard above has a known-bad counter-example.
    """
    cfg = ScannerConfig()
    baseline = cfg.min_confidence

    approver = AdjustmentApprover(
        pending_path=tmp_disk["pending"],
        approved_path=tmp_disk["approved"],
    )

    # Construct StagedDeployer the OLD broken way: config_adjuster
    # missing (defaults to None).
    deployer = StagedDeployer(
        config=cfg,
        adjustment_approver=approver,
        shadow_cycles=0,
        canary_trades=0,
    )

    pkg = _make_promoted_package({"min_confidence": baseline - 5.0})
    pkg.stage = ChangeStage.DEPLOYED_SHADOW
    for _ in range(4):
        deployer.advance(pkg, current_cycle=1)
        if pkg.stage in (ChangeStage.DEPLOYED_LIVE, ChangeStage.CLOSED):
            break

    # Even though stage advanced, the live config is UNCHANGED —
    # this is the bug shape.
    assert pkg.stage in (ChangeStage.DEPLOYED_LIVE, ChangeStage.CLOSED)
    assert cfg.min_confidence == baseline, (
        "negative control failed: live config WAS mutated even with "
        "_adjuster=None — the early-return guard in _apply_canary may "
        "have been removed; revisit assumptions."
    )
