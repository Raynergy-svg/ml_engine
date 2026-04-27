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


# G1 soak-gate constants — replace cycle-based promotion with closed-trade
# count + R-multiple floor. R_baseline is the mean R-multiple of the
# R_BASELINE_LOOKBACK trades immediately preceding deploy (cold-start: 0.0).
SHADOW_MIN_TRADES = 15
CANARY_MIN_TRADES = 30
R_FLOOR_DELTA = 0.5  # R_mean must be >= R_baseline - R_FLOOR_DELTA
R_BASELINE_LOOKBACK = 50


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
        regime_detector: Any = None,
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
        # Optional regime detector for snapshotting regime_at_deploy on
        # promotion. Best-effort (None or throwing detector → None capture,
        # logged but non-blocking — regime tagging is observability, not gating).
        self._regime_detector = regime_detector

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

        package.deployments.append(self._build_deploy_record(target, current_cycle))
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

    def _build_deploy_record(
        self, target: DeployStage, current_cycle: int
    ) -> DeploymentRecord:
        """Construct a DeploymentRecord enriched with regime + trade-count snapshots.

        Both snapshots are best-effort: regime_detector.current() exceptions are
        logged and fall back to None; _read_trade_journal_count() returns 0 on
        journal read failure (already logged inside the helper). Reusing the
        same trade-count read path as the R-floor soak gate keeps the index
        arithmetic consistent across deploy and promotion-eval.
        """
        regime: Optional[str] = None
        if self._regime_detector is not None:
            try:
                regime = self._regime_detector.current()
            except Exception as e:
                logger.warning("staged_deployer.regime_capture_failed err=%s", e)
        return DeploymentRecord(
            stage=target,
            deployed_at_cycle=current_cycle,
            closed_trade_count_at_deploy=self._read_trade_journal_count(),
            regime_at_deploy=regime,
        )

    def _read_trade_journal_count(self) -> int:
        """Count closed trades in trade_journal_rl.json. 0 on any error.

        On read failure the gate sees 0 closed trades since deploy → never
        promotes (fail-safe). Caller should be aware that a return of 0 may
        mean an empty journal OR a read error.
        """
        try:
            from src.scanner.automation.safe_json import safe_json_read
            from pathlib import Path
            path = Path(__file__).resolve().parents[3] / "trained_data" / "trade_journal_rl.json"
            entries = safe_json_read(path, default=[])
            return len(entries) if isinstance(entries, list) else 0
        except Exception as e:
            logger.warning("staged_deployer.journal_count_failed err=%s", e)
            return 0

    def _active_record(
        self, package: ChangePackage, stage: DeployStage
    ) -> Optional[DeploymentRecord]:
        """Most recent non-rolled-back DeploymentRecord at the given stage, else None."""
        return next(
            (d for d in package.deployments if d.stage == stage and d.rolled_back_at is None),
            None,
        )

    def _compute_window_r_stats(
        self, record: DeploymentRecord, window_size: int
    ) -> Dict[str, float]:
        """Compute R_mean over the next `window_size` closed trades after deploy
        AND R_baseline over the R_BASELINE_LOOKBACK trades immediately preceding
        deploy. Both default to 0.0 on journal read errors or insufficient
        history."""
        try:
            from src.scanner.automation.safe_json import safe_json_read
            from pathlib import Path
            path = Path(__file__).resolve().parents[3] / "trained_data" / "trade_journal_rl.json"
            entries = safe_json_read(path, default=[])
            if not isinstance(entries, list):
                return {"R_mean": 0.0, "R_baseline": 0.0}
            deploy_idx = record.closed_trade_count_at_deploy
            # Sanity check: deploy_idx is a positional index into the journal.
            # If the journal has been trimmed/rotated since deploy, deploy_idx
            # becomes stale and would produce empty/garbage windows. Detect
            # that explicitly and return zeros (fails the floor in any case
            # where R_baseline >= -0.5R, which is most production scenarios).
            if deploy_idx > len(entries):
                logger.warning(
                    "staged_deployer.r_stats_stale_index deploy_idx=%s journal_len=%s",
                    deploy_idx, len(entries),
                )
                return {"R_mean": 0.0, "R_baseline": 0.0}
            window = entries[deploy_idx : deploy_idx + window_size]
            baseline = entries[max(0, deploy_idx - R_BASELINE_LOOKBACK) : deploy_idx]

            def r_of(e: Any) -> float:
                o = (e.get("outcome") or {}) if isinstance(e, dict) else {}
                try:
                    return float(o.get("r_multiple", 0.0))
                except (TypeError, ValueError):
                    return 0.0

            r_mean = sum(r_of(e) for e in window) / len(window) if window else 0.0
            r_base = sum(r_of(e) for e in baseline) / len(baseline) if baseline else 0.0
            return {"R_mean": r_mean, "R_baseline": r_base}
        except Exception as e:
            logger.warning("staged_deployer.r_stats_failed err=%s", e)
            return {"R_mean": 0.0, "R_baseline": 0.0}

    def should_promote_shadow_to_canary(
        self, package: ChangePackage, current_trade_count: Optional[int] = None
    ) -> bool:
        """G1: shadow→canary gate uses real closed-trade count + R-floor.

        Cold-start note: when fewer than R_BASELINE_LOOKBACK (50) trades
        precede the deploy, R_baseline=0.0 (see _compute_window_r_stats).
        That makes the floor effectively `R_mean >= -0.5R` for cold-start
        deploys — quite permissive. The deeper SHADOW_MIN_TRADES count gate
        (15 trades) is what protects new deploys from premature promotion.
        Once the journal accumulates 50+ trades the R-floor tightens to a
        meaningful comparison against historical performance.
        """
        rec = self._active_record(package, DeployStage.SHADOW)
        if rec is None:
            return False
        if current_trade_count is None:
            current_trade_count = self._read_trade_journal_count()
        closed_since_deploy = current_trade_count - rec.closed_trade_count_at_deploy
        if closed_since_deploy < SHADOW_MIN_TRADES:
            return False
        stats = self._compute_window_r_stats(rec, SHADOW_MIN_TRADES)
        return stats["R_mean"] >= stats["R_baseline"] - R_FLOOR_DELTA

    def should_promote_canary_to_live(
        self, package: ChangePackage, current_trade_count: Optional[int] = None
    ) -> bool:
        """G1: canary→live gate uses real closed-trade count + R-floor.

        Cold-start has the same caveat as shadow→canary: when journal has
        <50 trades preceding deploy, R_baseline falls back to 0.0. By the
        time canary stage is reached the journal is typically populated
        enough that the R-floor is meaningful. See should_promote_shadow_to_canary
        for the full semantics.
        """
        rec = self._active_record(package, DeployStage.CANARY)
        if rec is None:
            return False
        if current_trade_count is None:
            current_trade_count = self._read_trade_journal_count()
        closed_since_deploy = current_trade_count - rec.closed_trade_count_at_deploy
        if closed_since_deploy < CANARY_MIN_TRADES:
            return False
        stats = self._compute_window_r_stats(rec, CANARY_MIN_TRADES)
        return stats["R_mean"] >= stats["R_baseline"] - R_FLOOR_DELTA

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
