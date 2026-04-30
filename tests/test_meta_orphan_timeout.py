"""Mythos audit 2026-04-29 — orphan-package timeout on the
meta-pipeline concurrency throttle.

Production case discovered live: package be14953921fb got stuck at
stage=policy_check on 2026-04-28T14:43:52Z (scorecard=None,
constitution=None — producer crashed mid-pipeline). For 22 hours every
new incident the orchestrator tried to route hit the throttle because
_concurrent_count saw a non-terminal package. Result: meta-pipeline
silently locked despite nothing actually being in flight.

The fix: packages whose updated_at is older than _STALE_PACKAGE_AGE_S
(default 2 hours) are not counted toward the concurrency throttle.
The stale package itself stays on disk — operators can still inspect
it via the F2 inbox or revert_by_id — but it stops blocking new
work.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.scanner.automation.meta_manager import MetaManager
from src.scanner.automation.meta_types import (
    ChangeKind,
    ChangePackage,
    ChangeStage,
)


def _make_pkg(stage: ChangeStage, age_hours: float, change_id: str) -> ChangePackage:
    """Create a package with a forced updated_at age."""
    pkg = ChangePackage(
        incident={"kind": "self_heal"},
        kind=ChangeKind.CONFIG,
    )
    pkg.change_id = change_id
    pkg.stage = stage
    pkg.updated_at = (
        datetime.now(timezone.utc) - timedelta(hours=age_hours)
    ).isoformat()
    return pkg


def _mgr(tmp_path) -> MetaManager:
    """Build a bare MetaManager with the changes_dir pointed at a clean tmp dir."""
    return MetaManager(
        specialist_invoker=lambda mode, prompt: "",
        changes_dir=tmp_path / "changes",
        ledger_path=tmp_path / "ledger.jsonl",
    )


def test_terminal_stages_not_counted(tmp_path):
    """Pre-existing behavior: closed/rejected/aborted don't count."""
    mgr = _mgr(tmp_path)
    pkgs = [
        _make_pkg(ChangeStage.CLOSED, 0.1, "term1"),
        _make_pkg(ChangeStage.REJECTED, 0.1, "term2"),
        _make_pkg(ChangeStage.ABORTED, 0.1, "term3"),
    ]
    with patch.object(mgr, "list_packages", return_value=pkgs):
        assert mgr._concurrent_count() == 0


def test_awaiting_approval_does_not_block_throttle(tmp_path):
    """Mythos audit 2026-04-30 — awaiting_approval is HUMAN-WAITING,
    not pipeline-in-flight. Must not consume the max_concurrent slot.
    Production case: 4 awaiting_approval packages stacked up over 13h
    with max_concurrent=1, blocking every new orchestrator routing →
    5 throttled-aborts."""
    mgr = _mgr(tmp_path)
    pkgs = [
        _make_pkg(ChangeStage.AWAITING_APPROVAL, 0.5, "queued1"),
        _make_pkg(ChangeStage.AWAITING_APPROVAL, 1.0, "queued2"),
        _make_pkg(ChangeStage.AWAITING_APPROVAL, 0.1, "queued3"),
    ]
    with patch.object(mgr, "list_packages", return_value=pkgs):
        assert mgr._concurrent_count() == 0


def test_deployed_stages_do_not_block_throttle(tmp_path):
    """Deployed packages are SOAKING (shadow/canary observation
    period), not actively running pipeline work. New incidents must
    not throttle on them."""
    mgr = _mgr(tmp_path)
    pkgs = [
        _make_pkg(ChangeStage.DEPLOYED_SHADOW, 0.5, "shadow1"),
        _make_pkg(ChangeStage.DEPLOYED_CANARY, 0.5, "canary1"),
        _make_pkg(ChangeStage.DEPLOYED_LIVE, 0.5, "live1"),
        _make_pkg(ChangeStage.POST_DEPLOY_REVIEW, 0.5, "review1"),
    ]
    with patch.object(mgr, "list_packages", return_value=pkgs):
        assert mgr._concurrent_count() == 0


def test_active_pipeline_stages_count(tmp_path):
    """Stages where the pipeline is actively executing CPU/IO work
    DO count — that's the legitimate purpose of the throttle."""
    mgr = _mgr(tmp_path)
    for stage in (
        ChangeStage.INTAKE,
        ChangeStage.DIAGNOSING,
        ChangeStage.PROPOSING,
        ChangeStage.EVALUATING,
        ChangeStage.POLICY_CHECK,
        ChangeStage.APPROVED,
    ):
        pkgs = [_make_pkg(stage, 0.01, f"active_{stage.value}")]
        with patch.object(mgr, "list_packages", return_value=pkgs):
            assert mgr._concurrent_count() == 1, (
                f"stage={stage.value} should count toward throttle"
            )


def test_recent_active_packages_counted(tmp_path):
    """Recent non-terminal packages still count — concurrency
    throttle continues to function for legitimate in-flight work."""
    mgr = _mgr(tmp_path)
    pkgs = [
        _make_pkg(ChangeStage.INTAKE, 0.01, "fresh1"),
        _make_pkg(ChangeStage.PROPOSING, 0.05, "fresh2"),
        _make_pkg(ChangeStage.POLICY_CHECK, 0.1, "fresh3"),
    ]
    with patch.object(mgr, "list_packages", return_value=pkgs):
        assert mgr._concurrent_count() == 3


def test_stale_package_excluded_from_throttle(tmp_path):
    """The bug fix: a 22h-old policy_check package must NOT count
    toward concurrency. Otherwise it blocks the pipeline forever."""
    mgr = _mgr(tmp_path)
    pkgs = [
        _make_pkg(ChangeStage.POLICY_CHECK, age_hours=22.0, change_id="stale"),
    ]
    with patch.object(mgr, "list_packages", return_value=pkgs):
        assert mgr._concurrent_count() == 0


def test_mixed_fresh_and_stale_only_counts_fresh(tmp_path):
    mgr = _mgr(tmp_path)
    pkgs = [
        _make_pkg(ChangeStage.INTAKE, age_hours=0.01, change_id="fresh"),
        _make_pkg(ChangeStage.POLICY_CHECK, age_hours=22.0, change_id="stale"),
        _make_pkg(ChangeStage.EVALUATING, age_hours=5.0, change_id="also_stale"),
    ]
    with patch.object(mgr, "list_packages", return_value=pkgs):
        assert mgr._concurrent_count() == 1


def test_threshold_exactly_two_hours_treated_as_active(tmp_path):
    """Boundary: a 1.99h package must still count (active), 2.01h must not.
    Defends against off-by-one in the comparison operator."""
    mgr = _mgr(tmp_path)
    pkgs = [_make_pkg(ChangeStage.INTAKE, age_hours=1.99, change_id="just_under")]
    with patch.object(mgr, "list_packages", return_value=pkgs):
        assert mgr._concurrent_count() == 1
    pkgs = [_make_pkg(ChangeStage.INTAKE, age_hours=2.01, change_id="just_over")]
    with patch.object(mgr, "list_packages", return_value=pkgs):
        assert mgr._concurrent_count() == 0


def test_missing_updated_at_counted_active(tmp_path):
    """Defensive: a package with no updated_at value (corrupt/partial
    write) should be conservatively counted — better to throttle a
    new incident than to silently let a real in-flight package race
    against new work."""
    mgr = _mgr(tmp_path)
    pkg = _make_pkg(ChangeStage.INTAKE, age_hours=0.01, change_id="no_ts")
    pkg.updated_at = ""  # simulate missing timestamp
    with patch.object(mgr, "list_packages", return_value=[pkg]):
        assert mgr._concurrent_count() == 1


def test_orphan_timeout_does_not_modify_stale_package(tmp_path):
    """The stale package must remain on disk — operators rely on it
    being reviewable via the F2 inbox / revert_by_id. We're skipping
    its throttle contribution, NOT auto-aborting it."""
    mgr = _mgr(tmp_path)
    stale = _make_pkg(ChangeStage.POLICY_CHECK, 22.0, "stale_keepme")
    original_stage = stale.stage
    original_updated = stale.updated_at
    with patch.object(mgr, "list_packages", return_value=[stale]):
        mgr._concurrent_count()
    assert stale.stage == original_stage
    assert stale.updated_at == original_updated
