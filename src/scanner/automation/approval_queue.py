"""File-backed approval queue for the meta-cybernetic change pipeline.

Pending packages live under `.claude/approval_queue/pending/<change_id>.json`.
A human (or a hook) moves the file to `approved/` or `rejected/` to advance
the pipeline. `MetaManager.drain()` polls these directories every cycle.

Atomic moves use `os.replace` so a half-written approval is never readable.
Reads use `safe_json_read` (fcntl-locked) — the same primitive used by
config_adjuster, episodic_memory, and the trade journal.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.scanner.automation.meta_types import ChangePackage, ChangeStage, DeployStage
from src.scanner.automation.safe_json import safe_json_read, safe_json_write

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_ROOT = _PROJECT_ROOT / ".claude" / "approval_queue"


@dataclass
class ApprovalDecision:
    """Result of a human's approve/reject action."""

    change_id: str
    approved: bool
    deploy_target: Optional[DeployStage] = None
    reason: str = ""


class ApprovalQueue:
    """File-backed pending/approved/rejected directories with atomic moves."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else _DEFAULT_ROOT
        self.pending_dir = self.root / "pending"
        self.approved_dir = self.root / "approved"
        self.rejected_dir = self.root / "rejected"
        for d in (self.pending_dir, self.approved_dir, self.rejected_dir):
            d.mkdir(parents=True, exist_ok=True)

    def enqueue(self, package: ChangePackage) -> Path:
        path = self.pending_dir / f"{package.change_id}.json"
        package.stage = ChangeStage.AWAITING_APPROVAL
        package.touch("enqueued_for_approval")
        safe_json_write(path, package.to_dict())
        logger.info(
            "approval_queue.enqueued change_id=%s deploy_target=%s",
            package.change_id, package.deploy_target.value,
        )
        return path

    def list_pending(self) -> List[ChangePackage]:
        return self._scan(self.pending_dir)

    def list_approved(self) -> List[ChangePackage]:
        return self._scan(self.approved_dir)

    def list_rejected(self) -> List[ChangePackage]:
        return self._scan(self.rejected_dir)

    def get(self, change_id: str) -> Optional[ChangePackage]:
        for d in (self.pending_dir, self.approved_dir, self.rejected_dir):
            path = d / f"{change_id}.json"
            if path.exists():
                data = safe_json_read(path, default={})
                if data:
                    return ChangePackage.from_dict(data)
        return None

    def approve(
        self,
        change_id: str,
        deploy_target: DeployStage = DeployStage.SHADOW,
        reason: str = "",
    ) -> Optional[ChangePackage]:
        pending = self.pending_dir / f"{change_id}.json"
        if not pending.exists():
            logger.warning("approval_queue.approve_missing change_id=%s", change_id)
            return None
        data = safe_json_read(pending, default={})
        if not data:
            logger.warning("approval_queue.approve_unreadable change_id=%s", change_id)
            return None
        pkg = ChangePackage.from_dict(data)
        pkg.stage = ChangeStage.APPROVED
        pkg.deploy_target = deploy_target
        pkg.touch(f"approved target={deploy_target.value} reason={reason}")
        target_path = self.approved_dir / f"{change_id}.json"
        safe_json_write(target_path, pkg.to_dict())
        try:
            os.remove(pending)
        except OSError as e:
            logger.warning("approval_queue.approve_cleanup_failed change_id=%s err=%s", change_id, e)
        logger.info(
            "approval_queue.approved change_id=%s target=%s",
            change_id, deploy_target.value,
        )
        return pkg

    def reject(self, change_id: str, reason: str) -> Optional[ChangePackage]:
        pending = self.pending_dir / f"{change_id}.json"
        if not pending.exists():
            logger.warning("approval_queue.reject_missing change_id=%s", change_id)
            return None
        data = safe_json_read(pending, default={})
        if not data:
            return None
        pkg = ChangePackage.from_dict(data)
        pkg.stage = ChangeStage.REJECTED
        pkg.rejection_reason = reason
        pkg.touch(f"rejected reason={reason}")
        target = self.rejected_dir / f"{change_id}.json"
        safe_json_write(target, pkg.to_dict())
        try:
            os.remove(pending)
        except OSError as e:
            logger.warning("approval_queue.reject_cleanup_failed change_id=%s err=%s", change_id, e)
        logger.info("approval_queue.rejected change_id=%s reason=%s", change_id, reason)
        return pkg

    def drain_approved(self) -> List[ChangePackage]:
        """Read approved packages and clear them from disk so they aren't re-acted on.

        The caller is responsible for advancing the package after this returns;
        the package is then re-persisted to `.claude/meta/changes/<id>.json` by
        the MetaManager.
        """
        pkgs = self._scan(self.approved_dir)
        for pkg in pkgs:
            path = self.approved_dir / f"{pkg.change_id}.json"
            try:
                os.remove(path)
            except OSError as e:
                logger.warning("approval_queue.drain_failed change_id=%s err=%s", pkg.change_id, e)
        return pkgs

    def drain_rejected(self) -> List[ChangePackage]:
        pkgs = self._scan(self.rejected_dir)
        for pkg in pkgs:
            path = self.rejected_dir / f"{pkg.change_id}.json"
            try:
                os.remove(path)
            except OSError as e:
                logger.warning("approval_queue.drain_rejected_failed change_id=%s err=%s", pkg.change_id, e)
        return pkgs

    def _scan(self, directory: Path) -> List[ChangePackage]:
        out: List[ChangePackage] = []
        for path in sorted(directory.glob("*.json")):
            data = safe_json_read(path, default={})
            if data:
                try:
                    out.append(ChangePackage.from_dict(data))
                except Exception as e:
                    logger.warning("approval_queue.bad_package path=%s err=%s", path, e)
        return out
