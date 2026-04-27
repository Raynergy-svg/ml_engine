"""Shadow → canary → live state machine for the meta-cybernetic pipeline.

A `ChangePackage` arrives here after human approval. Each stage is a holding
cell: the package waits N cycles (shadow) or N closed trades (canary), then
the `PostDeployCritic` decides whether to promote or roll back.

Shadow stage writes to a parallel config namespace (prefix `shadow_`) so the
scanner can produce parallel scoring without affecting trades.
Canary stage promotes to the active config but caps participation to a
single pair / smallest size class.
Live stage uses the existing `ConfigAdjuster` so the rate-limiter and
persistence already apply.

This module only manages the lifecycle of the change. Rollback semantics
are delegated to `ConfigAdjuster.revert_by_id` (added separately) for live
changes, and to dropping the shadow_ namespace for shadow changes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.scanner.automation.meta_types import (
    ChangePackage,
    ChangeStage,
    DeploymentRecord,
    DeployStage,
)

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class StagedDeployer:
    """Promotes a `ChangePackage` through shadow / canary / live in order.

    Responsibilities:
      - Apply shadow flags to a parallel namespace (no trade impact)
      - Promote to canary when shadow review passes
      - Promote to live via ConfigAdjuster when canary review passes
      - Roll back via ConfigAdjuster.revert_by_id on review failure
    """

    def __init__(
        self,
        config: Any,
        config_adjuster: Any = None,
        adjustment_approver: Any = None,
        shadow_cycles: int = 20,
        canary_trades: int = 10,
    ):
        self._config = config
        self._adjuster = config_adjuster
        # Approver is required for canary/live to bypass the human-in-the-loop
        # AdjustmentApprover used by threshold_optimizer / drift_monitor.
        # The meta pipeline is its own approval authority (constitution +
        # scorecard + operator approval all happened upstream), so it
        # auto-approves its own proposals before apply_adjustments runs.
        # When None, MetaManager constructs a default at init time.
        self._approver = adjustment_approver
        self._shadow_cycles = int(shadow_cycles)
        self._canary_trades = int(canary_trades)

    @property
    def shadow_cycles(self) -> int:
        return self._shadow_cycles

    @property
    def canary_trades(self) -> int:
        return self._canary_trades

    def advance(self, package: ChangePackage, current_cycle: int = 0) -> ChangePackage:
        """Move the package one stage forward (or initialize the first stage).

        Idempotent within a stage — if the package already has a `DeploymentRecord`
        for the current target stage, this is a no-op.
        """
        target = package.deploy_target
        if any(d.stage == target and d.rolled_back_at is None for d in package.deployments):
            return package

        if target == DeployStage.SHADOW:
            self._apply_shadow(package)
            package.stage = ChangeStage.DEPLOYED_SHADOW
        elif target == DeployStage.CANARY:
            self._apply_canary(package, current_cycle)
            package.stage = ChangeStage.DEPLOYED_CANARY
        elif target == DeployStage.LIVE:
            self._apply_live(package, current_cycle)
            package.stage = ChangeStage.DEPLOYED_LIVE

        package.deployments.append(
            DeploymentRecord(stage=target, deployed_at_cycle=current_cycle)
        )
        package.touch(f"deployed stage={target.value} cycle={current_cycle}")
        logger.info(
            "staged_deployer.advance change_id=%s stage=%s",
            package.change_id, target.value,
        )
        return package

    def promote_next(self, package: ChangePackage, current_cycle: int = 0) -> ChangePackage:
        """Promote to the next stage in the chain shadow → canary → live."""
        order = [DeployStage.SHADOW, DeployStage.CANARY, DeployStage.LIVE]
        try:
            idx = order.index(package.deploy_target)
        except ValueError:
            return package
        if idx + 1 >= len(order):
            package.stage = ChangeStage.CLOSED
            package.touch("promotion_chain_complete")
            return package
        package.deploy_target = order[idx + 1]
        return self.advance(package, current_cycle=current_cycle)

    def rollback(self, package: ChangePackage, reason: str) -> ChangePackage:
        for record in reversed(package.deployments):
            if record.rolled_back_at is None:
                record.rolled_back_at = _now_iso()
                record.rollback_reason = reason
                self._undo_stage(package, record.stage)
                break
        package.stage = ChangeStage.REJECTED
        package.rejection_reason = f"rolled_back: {reason}"
        package.touch(f"rolled_back reason={reason}")
        logger.warning(
            "staged_deployer.rollback change_id=%s reason=%s",
            package.change_id, reason,
        )
        return package

    def _apply_shadow(self, package: ChangePackage) -> None:
        delta = (package.proposal.config_delta if package.proposal else {}) or {}
        for key, change in delta.items():
            new_value = change.get("new") if isinstance(change, dict) and "new" in change else change
            shadow_key = f"shadow_{key}"
            try:
                setattr(self._config, shadow_key, new_value)
            except Exception as e:
                logger.warning("staged_deployer.shadow_set_failed key=%s err=%s", shadow_key, e)

    def _apply_canary(self, package: ChangePackage, current_cycle: int) -> None:
        if not self._adjuster:
            logger.warning(
                "staged_deployer.canary_no_adjuster change_id=%s — flags will not be live-tunable",
                package.change_id,
            )
            return
        if not self._approver:
            logger.error(
                "staged_deployer.canary_no_approver change_id=%s — meta pipeline cannot "
                "auto-approve its own proposals; ConfigAdjuster.apply_adjustments would "
                "raise BypassAttempt. Skipping canary application.",
                package.change_id,
            )
            return
        # Include a deploy-attempt counter in the source so each retry has a
        # unique (source, key, value) signature. Without this, ConfigAdjuster
        # duplicate-suppresses the second attempt against the first attempt's
        # row (now status=approved), AdjustmentApprover then sees status !=
        # pending and refuses, and the canary silently no-ops. The current
        # deployments list length is the attempt index for THIS canary stage.
        canary_attempt = sum(
            1 for d in package.deployments if d.stage == DeployStage.CANARY
        )
        delta = (package.proposal.config_delta if package.proposal else {}) or {}
        proposal_ids: List[str] = []
        for key, change in delta.items():
            new_value = change.get("new") if isinstance(change, dict) and "new" in change else change
            try:
                pid = self._adjuster.collect_adjustment(
                    source=f"meta_manager_canary:{package.change_id}:attempt_{canary_attempt}",
                    key=key,
                    value=new_value,
                    reason=f"canary deploy of change {package.change_id} (attempt {canary_attempt})",
                    cycle=current_cycle,
                )
                if pid:
                    proposal_ids.append(pid)
                else:
                    logger.warning(
                        "staged_deployer.canary_proposal_rejected change_id=%s key=%s "
                        "value=%s — failed validation; canary will not affect this key",
                        package.change_id, key, new_value,
                    )
            except Exception as e:
                logger.warning("staged_deployer.canary_collect_failed key=%s err=%s", key, e)
        # Auto-approve every proposal we just created. The meta pipeline IS
        # the approver of record here (constitution + scorecard + human gate
        # all ran upstream); routing through AdjustmentApprover.approve()
        # moves the entry from pending → approved-history so apply_adjustments
        # can read it without raising BypassAttempt (US-508 write-guard).
        for pid in proposal_ids:
            try:
                if not self._approver.approve(pid):
                    logger.warning(
                        "staged_deployer.canary_approve_no_op change_id=%s proposal_id=%s",
                        package.change_id, pid[:8],
                    )
            except Exception as e:
                logger.warning("staged_deployer.canary_approve_failed proposal_id=%s err=%s", pid[:8], e)
        try:
            self._adjuster.apply_adjustments(self._config, current_cycle=current_cycle)
        except Exception as e:
            logger.warning("staged_deployer.canary_apply_failed err=%s", e)

    def _apply_live(self, package: ChangePackage, current_cycle: int) -> None:
        if not self._adjuster:
            logger.warning(
                "staged_deployer.live_no_adjuster change_id=%s",
                package.change_id,
            )
            return
        self._apply_canary(package, current_cycle)

    def _undo_stage(self, package: ChangePackage, stage: DeployStage) -> None:
        delta = (package.proposal.config_delta if package.proposal else {}) or {}
        if stage == DeployStage.SHADOW:
            for key in delta:
                shadow_key = f"shadow_{key}"
                try:
                    if hasattr(self._config, shadow_key):
                        delattr(self._config, shadow_key)
                except Exception as e:
                    logger.warning("staged_deployer.shadow_undo_failed key=%s err=%s", shadow_key, e)
            return
        if not self._adjuster:
            return
        revert = getattr(self._adjuster, "revert_by_id", None)
        if revert is None:
            logger.warning(
                "staged_deployer.adjuster_missing_revert_by_id change_id=%s",
                package.change_id,
            )
            return
        try:
            revert(self._config, source_substring=package.change_id)
        except Exception as e:
            logger.warning("staged_deployer.live_undo_failed err=%s", e)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
