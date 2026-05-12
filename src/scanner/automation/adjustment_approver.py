"""Approval handler for pending config adjustments.

US-508: The ONLY legitimate path from pending_adjustments.json to
config_adjustments.json. Operators approve/reject/snooze proposals here.
No proposal reaches the running ScannerConfig without passing through this handler.

Design invariant: ConfigAdjuster.apply_adjustments() reads exclusively from
config_adjustments.json. This module is the sole writer of approved entries.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PENDING_PATH = _PROJECT_ROOT / ".claude" / "pending_adjustments.json"
_APPROVED_PATH = _PROJECT_ROOT / ".claude" / "config_adjustments.json"
_STATE_PATH = _PROJECT_ROOT / ".claude" / "state.json"


class AdjustmentApprover:
    """Moves proposals from pending_adjustments.json to config_adjustments.json.

    Usage:
        approver = AdjustmentApprover()
        approver.approve("proposal-uuid")       # → lands in config_adjustments.json
        approver.reject("proposal-uuid", "bad key")
        approver.snooze("proposal-uuid", hours=24)

    Tier 2 T8: every successful `_save_approved` write flips
    `state.json:config_dirty=true`. The TUI scanner consumes the flag at
    the top of each cycle and reloads the ConfigAdjuster cache so newly
    approved adjustments take effect within ~1 cycle (vs the previous
    "wait for the next ConfigAdjuster apply tick" delay).
    """

    def __init__(
        self,
        pending_path: Optional[Path] = None,
        approved_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
    ):
        self.pending_path = pending_path or _PENDING_PATH
        self.approved_path = approved_path or _APPROVED_PATH
        # Tier 2 T8: dirty-flag sink. Tests pass tmp_path/"state.json"; the
        # default points at the real .claude/state.json the TUI reads.
        self.state_path = state_path or _STATE_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def approve(self, proposal_id: str) -> bool:
        """Approve a pending proposal — moves it to config_adjustments.json.

        This is the only path by which a proposal becomes an approved adjustment.
        Returns True on success, False if not found or already processed.
        """
        pending_data = self._load_pending()
        proposal = self._find_proposal(pending_data.get("proposals", []), proposal_id)

        if proposal is None:
            logger.warning("AdjustmentApprover.approve: proposal %s not found", proposal_id)
            return False

        if proposal.get("status") not in ("pending", "snoozed"):
            logger.warning(
                "AdjustmentApprover.approve: proposal %s not actionable (status=%s)",
                proposal_id, proposal.get("status"),
            )
            return False

        # Validation guard — re-validate before writing to approved
        from src.scanner.automation.adjustment_validator import validate_adjustment, ValidationResult
        key = proposal.get("key", "")
        value = proposal.get("proposed_value")
        validation = validate_adjustment(key, value, proposal.get("reason", ""))
        if not validation.valid:
            logger.error(
                "AdjustmentApprover.approve: proposal %s failed re-validation: %s",
                proposal_id, validation.error_message,
            )
            proposal["status"] = "invalid"
            self._save_pending(pending_data)
            return False

        now = datetime.now(timezone.utc).isoformat()

        # Mark approved in pending file
        proposal["status"] = "approved"
        proposal["approved_at"] = now
        self._save_pending(pending_data)

        # Write to config_adjustments.json (the approved store)
        approved_data = self._load_approved()
        history: list[dict[str, Any]] = approved_data.get("history", [])
        approved_entry = {
            "key": key,
            "old_value": proposal.get("current_value"),
            "new_value": value,
            "source": proposal.get("source", "unknown"),
            "reason": proposal.get("reason", ""),
            "proposal_id": proposal_id,
            "timestamp": now,
            "approved_at": now,
        }
        if not self._history_has_duplicate(history, approved_entry):
            history.append(approved_entry)
        else:
            logger.info(
                "AdjustmentApprover: approved %s without duplicating history for %s = %s",
                proposal_id[:8], key, value,
            )
        approved_data["history"] = history[-200:]
        approved_data["last_updated"] = now
        approved_data["version"] = approved_data.get("version", 1)
        approved_data["total_adjustments"] = len(approved_data["history"])
        self._save_approved(approved_data)

        self._emit_event("adjustment.applied", {
            "proposal_id": proposal_id,
            "key": key,
            "value": value,
            "source": proposal.get("source", "unknown"),
        })

        logger.info(
            "AdjustmentApprover: approved %s — %s = %s (source=%s)",
            proposal_id[:8], key, value, proposal.get("source", "unknown"),
        )
        return True

    def reject(self, proposal_id: str, reason: str = "") -> bool:
        """Reject a pending proposal. The proposal stays in pending with status='rejected'."""
        pending_data = self._load_pending()
        proposal = self._find_proposal(pending_data.get("proposals", []), proposal_id)

        if proposal is None:
            logger.warning("AdjustmentApprover.reject: proposal %s not found", proposal_id)
            return False

        proposal["status"] = "rejected"
        proposal["rejected_at"] = datetime.now(timezone.utc).isoformat()
        proposal["rejection_reason"] = reason
        self._save_pending(pending_data)

        self._emit_event("adjustment.rejected", {
            "proposal_id": proposal_id,
            "key": proposal.get("key"),
            "reason": reason,
        })

        logger.info("AdjustmentApprover: rejected %s — %s", proposal_id[:8], reason)
        return True

    def approve_all(self) -> dict[str, Any]:
        """Approve every valid actionable proposal.

        Invalid proposals are left in the inbox so an operator can reject or
        inspect them explicitly.
        """
        pending_data = self._load_pending()
        proposals = pending_data.get("proposals", [])
        normalized_invalid = False
        for proposal in proposals:
            if (
                proposal.get("status") in ("pending", "snoozed")
                and not (proposal.get("validation") or {}).get("valid")
            ):
                proposal["status"] = "invalid"
                normalized_invalid = True
        if normalized_invalid:
            self._save_pending(pending_data)

        actionable_ids = [
            p.get("id")
            for p in proposals
            if p.get("id")
            and p.get("status") in ("pending", "snoozed")
            and (p.get("validation") or {}).get("valid")
        ]

        approved = 0
        failed: list[str] = []
        for proposal_id in actionable_ids:
            if self.approve(str(proposal_id)):
                approved += 1
            else:
                failed.append(str(proposal_id))

        skipped = len([
            p for p in proposals
            if p.get("status") in ("pending", "snoozed", "invalid")
            and not (p.get("validation") or {}).get("valid")
        ])
        return {"approved": approved, "failed": failed, "skipped": skipped}

    def reject_all(self, reason: str = "bulk reject") -> dict[str, Any]:
        """Reject every actionable proposal in the inbox."""
        pending_data = self._load_pending()
        proposals = pending_data.get("proposals", [])
        actionable_ids = [
            p.get("id")
            for p in proposals
            if p.get("id") and p.get("status") in ("pending", "snoozed", "invalid")
        ]

        rejected = 0
        failed: list[str] = []
        for proposal_id in actionable_ids:
            if self.reject(str(proposal_id), reason):
                rejected += 1
            else:
                failed.append(str(proposal_id))

        return {"rejected": rejected, "failed": failed}

    def snooze(self, proposal_id: str, hours: float) -> bool:
        """Snooze a proposal — it stays pending until snooze_until expires."""
        pending_data = self._load_pending()
        proposal = self._find_proposal(pending_data.get("proposals", []), proposal_id)

        if proposal is None:
            logger.warning("AdjustmentApprover.snooze: proposal %s not found", proposal_id)
            return False

        snooze_until = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
        proposal["status"] = "snoozed"
        proposal["snooze_until"] = snooze_until
        self._save_pending(pending_data)

        logger.info(
            "AdjustmentApprover: snoozed %s until %s",
            proposal_id[:8], snooze_until,
        )
        return True

    def list_pending(self) -> list[dict[str, Any]]:
        """Return all proposals with status='pending'."""
        data = self._load_pending()
        return [p for p in data.get("proposals", []) if p.get("status") == "pending"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_proposal(self, proposals: list[dict], proposal_id: str) -> Optional[dict]:
        for p in proposals:
            if p.get("id") == proposal_id:
                return p
        return None

    def _load_pending(self) -> dict[str, Any]:
        if not self.pending_path.exists():
            return {"proposals": []}
        try:
            return json.loads(self.pending_path.read_text())
        except Exception:
            return {"proposals": []}

    def _save_pending(self, data: dict[str, Any]) -> None:
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pending_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.rename(self.pending_path)

    def _load_approved(self) -> dict[str, Any]:
        if not self.approved_path.exists():
            return {"version": 1, "history": [], "pending": {}, "last_applied": {}}
        try:
            raw = json.loads(self.approved_path.read_text())
            if isinstance(raw, list):
                # Legacy config_tuner format: bare list of adjustments
                return {
                    "version": 1,
                    "history": raw,
                    "pending": {},
                    "last_applied": {},
                }
            return raw
        except (OSError, json.JSONDecodeError) as e:
            # Mythos audit 2026-04-30 — replaced bare `except: return empty`.
            # The silent-empty fallback was the path that wiped 200 entries
            # → 5 on 2026-04-29: a transient parse failure → empty load →
            # next save truncates against a fresh empty list. Now we log
            # loudly; the _save_approved write-side tripwire is the actual
            # safety net that refuses to clobber a longer on-disk history.
            logger.warning(
                "AdjustmentApprover._load_approved: corrupt/unreadable "
                "approved_path=%s err=%r — returning empty (write-side "
                "tripwire will refuse to shrink on-disk history)",
                self.approved_path, e,
            )
            return {"version": 1, "history": [], "pending": {}, "last_applied": {}}

    def _save_approved(self, data: dict[str, Any]) -> None:
        """Persist approved-state with a history-shrinkage tripwire.

        Pre-fix the writer was append-only-with-truncation: read existing,
        append new entry, truncate to last 200, write back. If the read
        ever returned `{history: []}` (from _load_approved's silent
        bare-except fallback) we'd overwrite hundreds of legitimate
        entries with a single-entry blob.

        Defense-in-depth: re-read disk RIGHT BEFORE the rename and
        compare the proposed history length against on-disk. If the new
        write would shrink history, abort and dump the proposed payload
        to a `.recovered` shadow file so the operator can reconcile
        manually rather than silently losing data.
        """
        self.approved_path.parent.mkdir(parents=True, exist_ok=True)

        proposed_history = data.get("history") or []
        proposed_n = len(proposed_history) if isinstance(proposed_history, list) else 0

        # Re-read disk just before write — closes the load→save race
        # window. If anything was appended in between (another writer,
        # a manual edit), we'll see it and refuse the shrinking write.
        disk_n = 0
        if self.approved_path.exists():
            try:
                disk_blob = json.loads(self.approved_path.read_text())
                if isinstance(disk_blob, dict):
                    disk_history = disk_blob.get("history") or []
                    if isinstance(disk_history, list):
                        disk_n = len(disk_history)
                elif isinstance(disk_blob, list):
                    disk_n = len(disk_blob)
            except (OSError, json.JSONDecodeError):
                disk_n = 0

        if proposed_n < disk_n:
            recovered = self.approved_path.with_suffix(".json.recovered")
            try:
                recovered.write_text(json.dumps(data, indent=2, sort_keys=True))
            except OSError as rec_err:
                logger.error(
                    "AdjustmentApprover._save_approved: shadow-write also "
                    "failed path=%s err=%r", recovered, rec_err,
                )
            logger.critical(
                "AdjustmentApprover._save_approved: REFUSED to shrink "
                "on-disk history disk=%d → proposed=%d. Operator: "
                "the proposed payload was written to %s for inspection. "
                "Disk file LEFT UNTOUCHED.",
                disk_n, proposed_n, recovered,
            )
            return

        tmp = self.approved_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.rename(self.approved_path)

        # Tier 2 T8: signal that the scanner should reload the
        # ConfigAdjuster cache on its next cycle. Best-effort —
        # a dirty-flag write failure should not crash an otherwise
        # successful approval.
        self._mark_config_dirty()

    def _mark_config_dirty(self) -> None:
        """Set `state.json:config_dirty=true` atomically.

        Read existing state (graceful fallback to empty dict on
        corrupt/missing), merge `config_dirty=true`, write via
        tmp+rename. Specific exceptions only — never bare-except.

        Failure here is non-fatal: the next cycle will just miss the
        immediate-reload optimization and the regular ConfigAdjuster
        apply tick will pick up the new adjustments later.
        """
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                if not isinstance(data, dict):
                    data = {}
            else:
                data = {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "AdjustmentApprover._mark_config_dirty: state read failed "
                "path=%s err=%r — starting from empty state for the dirty flag",
                self.state_path, e,
            )
            data = {}

        data["config_dirty"] = True

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
            tmp.replace(self.state_path)
        except OSError as e:
            logger.warning(
                "AdjustmentApprover._mark_config_dirty: state write failed "
                "path=%s err=%r — scanner will pick up the change on the "
                "next ConfigAdjuster apply tick instead of immediately",
                self.state_path, e,
            )

    def _history_has_duplicate(
        self,
        history: list[dict[str, Any]],
        entry: dict[str, Any],
    ) -> bool:
        signature = _approval_signature(entry)
        return any(_approval_signature(existing) == signature for existing in history)

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            from src.scanner.automation.event_bus import get_event_bus
            get_event_bus().publish(event_type, payload)
        except Exception:
            pass  # Event emission is best-effort; never block approval


def _approval_signature(entry: dict[str, Any]) -> tuple[str, str, str]:
    try:
        value = json.dumps(entry.get("new_value"), sort_keys=True)
    except TypeError:
        value = str(entry.get("new_value"))
    return (
        str(entry.get("source", "unknown")),
        str(entry.get("key", "")),
        value,
    )
