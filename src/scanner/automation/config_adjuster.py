"""Central config adjustment manager with persistence.

US-131: Collects threshold recommendations from ThresholdOptimizer,
ObservationConsumer, and DriftMonitor, persists them, and applies
them to the running ScannerConfig each cycle.

US-508: Approval gate added. Flow:
1. Modules call collect_adjustment(source, key, value, reason)
   → validated by AdjustmentValidator → written to pending_adjustments.json
2. Operator approves via AdjustmentApprover.approve(proposal_id)
   → moves proposal to config_adjustments.json (only path)
3. apply_adjustments() reads from config_adjustments.json
   → raises BypassAttempt if self._pending is non-empty (bypass detected)
4. Rate-limiter prevents more than 1 change per key per 10 cycles
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ADJUSTMENTS_PATH = _PROJECT_ROOT / ".claude" / "config_adjustments.json"
_PENDING_PATH = _PROJECT_ROOT / ".claude" / "pending_adjustments.json"

# Rate limit: max 1 change per key per N cycles
RATE_LIMIT_CYCLES = 10


class BypassAttempt(Exception):
    """Raised when apply_adjustments() detects proposals that bypassed the approval flow.

    Root cause of 2026-04-16 $3,527 loss: orphan keys were silently applied
    via setattr() without validation. This exception makes bypass visible.
    """


class ConfigAdjuster:
    """Collects, rate-limits, persists, and applies config adjustments.

    Usage:
        adjuster = ConfigAdjuster()
        adjuster.collect_adjustment("threshold_optimizer", "min_confidence", 0.48, "win_rate below target in HIGH regime")
        adjuster.apply_adjustments(scanner_config, current_cycle=42)
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        pending_path: Optional[Path] = None,
        ttl_seconds: float = 5.0,
    ):
        self.persistence_path = persistence_path or _ADJUSTMENTS_PATH
        self._pending_path = pending_path or _PENDING_PATH

        # Bypass-detection guard: should remain empty in the approval flow.
        # If non-empty when apply_adjustments() is called → BypassAttempt.
        self._pending: Dict[str, Dict[str, Any]] = {}

        # Applied history: [{key, old_value, new_value, source, reason, cycle, timestamp}]
        self._history: List[Dict[str, Any]] = []

        # Rate limiter: {key: last_applied_cycle}
        self._last_applied: Dict[str, int] = {}

        # Applied proposal IDs — prevents double-application across restarts
        self._applied_ids: set[str] = set()

        # Tier 1 T5: TTL cache around _load_state — skips redundant disk reads
        # within a short window. Invalidated on every persistent write so the
        # cache stays consistent with disk. See _load_state / _invalidate_cache.
        self._ttl_seconds: float = float(ttl_seconds)
        self._last_load_ts: float = 0.0
        self._load_count: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_adjustment(
        self,
        source: str,
        key: str,
        value: Any,
        reason: str,
        cycle: int = 0,
    ) -> Optional[str]:
        """Propose an adjustment from any module.

        US-508: validates key against ScannerConfig schema and writes to
        pending_adjustments.json. Does NOT populate self._pending.

        Args:
            source: Module name (e.g., "threshold_optimizer", "drift_monitor").
            key: Config key to adjust — must be a ScannerConfig field name.
            value: New value to set.
            reason: Human-readable reason for the change.
            cycle: Current scan cycle number.

        Returns:
            The proposal_id if validation passed (proposal queued), else None.
            The meta pipeline's StagedDeployer needs the id to auto-approve
            its own canary proposals via AdjustmentApprover.
        """
        from src.scanner.automation.adjustment_validator import validate_adjustment

        validation = validate_adjustment(key, value, reason)

        proposal_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        proposal = {
            "id": proposal_id,
            "timestamp": now,
            "key": key,
            "current_value": None,  # unknown at proposal time; filled at apply
            "proposed_value": _safe_serialize(value),
            "reason": reason,
            "source": source,
            "validation": validation.to_dict(),
            "status": "pending" if validation.valid else "invalid",
            "snooze_until": None,
        }

        # _write_pending_proposal returns the id actually present in the
        # pending file — either the new one or, if a duplicate (source, key,
        # value) was already pending, the existing one. The caller needs
        # whichever id resolves so AdjustmentApprover.approve() can find it.
        effective_id = self._write_pending_proposal(proposal) or proposal_id

        self._emit_event("adjustment.proposed", {
            "proposal_id": effective_id,
            "key": key,
            "value": value,
            "source": source,
            "valid": validation.valid,
        })

        if not validation.valid:
            logger.warning(
                "ConfigAdjuster: rejected proposal from %s: %s = %s — %s",
                source, key, value, validation.error_message,
            )
            return None

        logger.info(
            "ConfigAdjuster: proposal %s queued from %s: %s = %s",
            effective_id[:8], source, key, value,
        )
        return effective_id

    def apply_adjustments(self, config: Any, current_cycle: int = 0) -> List[Dict[str, Any]]:
        """Apply approved adjustments from config_adjustments.json to config.

        US-508: Raises BypassAttempt if self._pending is non-empty — that means
        someone populated it directly (old path) instead of routing through
        collect_adjustment() → approve() → this method.

        Args:
            config: ScannerConfig instance to modify.
            current_cycle: Current scan cycle for rate limiting.

        Returns:
            List of adjustments that were actually applied.

        Raises:
            BypassAttempt: If self._pending contains proposals that bypassed approval.
        """
        # Write-guard: self._pending must be empty in the approval flow
        if self._pending:
            keys = list(self._pending.keys())
            raise BypassAttempt(
                f"apply_adjustments() called with {len(keys)} unapproved proposals "
                f"in self._pending: {keys}. "
                "Route via collect_adjustment() → AdjustmentApprover.approve() → apply_adjustments()."
            )

        approved_history = self._load_approved_history()
        applied = []

        for entry in approved_history:
            entry_id = (
                entry.get("proposal_id")
                or f"{entry.get('key')}@{entry.get('timestamp', '')}"
            )

            if entry_id in self._applied_ids:
                continue

            key = entry.get("key")
            new_value = entry.get("new_value")

            if key is None or new_value is None:
                self._applied_ids.add(entry_id)
                continue

            # Hard NO (L-003, 2026-07-03): protected fields never setattr, even
            # from a hand-edited history file — last-line defense mirroring
            # adjustment_validator.PROTECTED_FIELDS (validated at proposal AND
            # approval; this guards the direct-file-tamper path).
            from src.scanner.automation.adjustment_validator import PROTECTED_FIELDS
            if key in PROTECTED_FIELDS:
                logger.error(
                    "ConfigAdjuster: PROTECTED key %r REFUSED at apply (Hard NO, "
                    "L-003) — history entry bypassed validation; investigate. "
                    "source=%s", key, entry.get("source", "unknown"))
                self._applied_ids.add(entry_id)
                continue

            # Rate limit
            last_cycle = self._last_applied.get(key, -RATE_LIMIT_CYCLES - 1)
            if current_cycle - last_cycle < RATE_LIMIT_CYCLES:
                continue

            old_value = getattr(config, key, None)
            if old_value == new_value:
                self._applied_ids.add(entry_id)
                continue

            try:
                setattr(config, key, new_value)
                record = {
                    "key": key,
                    "old_value": _safe_serialize(old_value),
                    "new_value": _safe_serialize(new_value),
                    "source": entry.get("source", "unknown"),
                    "reason": entry.get("reason", ""),
                    "cycle": current_cycle,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                applied.append(record)
                self._history.append(record)
                self._last_applied[key] = current_cycle
                self._applied_ids.add(entry_id)

                logger.info(
                    "ConfigAdjuster: applied approved %s = %s → %s (source=%s)",
                    key, old_value, new_value, entry.get("source", "unknown"),
                )
            except Exception as e:
                logger.debug("ConfigAdjuster: Failed to apply %s: %s", key, e)
                self._applied_ids.add(entry_id)

        if len(self._history) > 200:
            self._history = self._history[-100:]

        return applied

    def get_status(self) -> Dict[str, Any]:
        """Get adjuster status for observability."""
        return {
            "pending_count": len(self._pending),
            "total_applied": len(self._history),
            "recent": self._history[-5:],
            "pending": dict(self._pending),
        }

    def revert_by_id(self, config: Any, source_substring: str) -> List[Dict[str, Any]]:
        """Revert every history entry whose `source` contains `source_substring`.

        Used by the meta pipeline's StagedDeployer when a canary/live deploy
        fails its post-deploy review. Walks the history in reverse and resets
        each affected key to its prior `old_value`.
        """
        reverted: List[Dict[str, Any]] = []
        for record in reversed(list(self._history)):
            source = record.get("source", "")
            if source_substring not in source:
                continue
            key = record.get("key")
            if not key:
                continue
            try:
                setattr(config, key, record.get("old_value"))
                reverted.append({**record, "reverted_at": datetime.now(timezone.utc).isoformat()})
                logger.info(
                    "ConfigAdjuster.revert_by_id: %s = %s (source=%s)",
                    key, record.get("old_value"), source,
                )
            except Exception as e:
                logger.warning("ConfigAdjuster.revert_by_id failed key=%s err=%s", key, e)
        return reverted

    def save_state(self) -> None:
        """Persist applied_ids and rate-limit state to config_adjustments.json.

        Reads the existing file first and preserves the 'history' array written
        by AdjustmentApprover. Only updates meta fields so _applied_ids are
        correctly restored on restart without losing approved entries.
        """
        try:
            # Preserve existing approved history — don't clobber AdjustmentApprover writes
            existing: Dict[str, Any] = {}
            if self.persistence_path.exists():
                try:
                    raw = self.persistence_path.read_text()
                    existing = json.loads(raw)
                except Exception:
                    existing = {}

            if isinstance(existing, list):
                existing = {"version": 1, "history": existing, "pending": {}, "last_applied": {}}

            existing["version"] = 2
            existing["pending"] = {}
            existing["last_applied"] = dict(self._last_applied)
            existing["applied_ids"] = list(self._applied_ids)
            existing["last_updated"] = datetime.now(timezone.utc).isoformat()
            existing["total_adjustments"] = len(existing.get("history", []))

            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, existing)
            except ImportError:
                self.persistence_path.write_text(json.dumps(existing, indent=2))
            # Tier 1 T5: invalidate _load_state cache so the next read sees the
            # bytes we just wrote (not a stale copy from within the TTL window).
            self._invalidate_cache()
        except Exception as e:
            logger.debug("ConfigAdjuster: Failed to save: %s", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load previous state from disk.

        Tier 1 T5: cached for `self._ttl_seconds` to skip redundant disk reads
        on hot paths (e.g. multiple status queries per cycle). The cache is
        invalidated by `_invalidate_cache()`, which fires after every write to
        `self.persistence_path` so the in-memory view stays consistent with disk.
        """
        now = time.monotonic()
        if self._last_load_ts > 0.0 and (now - self._last_load_ts) < self._ttl_seconds:
            return
        if not self.persistence_path.exists():
            # Still bump the cache stamp so we don't re-stat a missing file
            # on every call within the TTL window.
            self._last_load_ts = now
            self._load_count += 1
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text())

            if isinstance(data, dict):
                # US-508: do not restore _pending — it must always start empty
                self._history = data.get("history", [])
                self._last_applied = {
                    k: int(v) for k, v in data.get("last_applied", {}).items()
                    if isinstance(v, (int, float))
                }
                self._applied_ids = set(data.get("applied_ids", []))
        except Exception as e:
            logger.debug("ConfigAdjuster: Failed to load: %s", e)
        finally:
            self._last_load_ts = now
            self._load_count += 1

    def _invalidate_cache(self) -> None:
        """Force the next `_load_state()` to re-read from disk.

        Called after every persistent write to `self.persistence_path` so the
        in-memory cache stays consistent with on-disk state. External writers
        (e.g. AdjustmentApprover) that modify the same file can also call this
        if they hold a reference to the adjuster instance.
        """
        self._last_load_ts = 0.0

    def _load_approved_history(self) -> List[Dict[str, Any]]:
        """Read approved adjustments from config_adjustments.json."""
        if not self.persistence_path.exists():
            return []
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text())

            if isinstance(data, list):
                return data  # legacy config_tuner format
            if isinstance(data, dict):
                return data.get("history", [])
        except Exception as e:
            logger.debug("ConfigAdjuster: Failed to load approved history: %s", e)
        return []

    def _write_pending_proposal(self, proposal: Dict[str, Any]) -> Optional[str]:
        """Atomically append one proposal to pending_adjustments.json.

        Identical proposals are idempotent: once a source has suggested the same
        key/value, keep the existing row instead of creating another operator
        approval item every scan cycle.

        Returns the proposal_id of the row now in the pending file — either
        the freshly-written one or the existing duplicate. The meta pipeline's
        StagedDeployer needs this id to call AdjustmentApprover.approve(); if
        we returned None on duplicate, the auto-approval would fail-silently
        and the canary would appear to deploy without actually mutating config.
        """
        data: Dict[str, Any] = {"proposals": []}
        if self._pending_path.exists():
            try:
                data = json.loads(self._pending_path.read_text())
            except Exception:
                data = {"proposals": []}

        proposals = data.setdefault("proposals", [])
        signature = _proposal_signature(proposal)
        for existing in proposals:
            if _proposal_signature(existing) == signature:
                existing_id = existing.get("id")
                logger.debug(
                    "ConfigAdjuster: duplicate proposal suppressed from %s: %s = %s "
                    "(reusing existing id=%s)",
                    proposal.get("source", "unknown"),
                    proposal.get("key"),
                    proposal.get("proposed_value"),
                    (existing_id or "")[:8],
                )
                return existing_id

        proposals.append(proposal)
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._pending_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.rename(self._pending_path)
        return proposal.get("id")

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            from src.scanner.automation.event_bus import get_event_bus
            get_event_bus().publish(event_type, payload)
        except Exception:
            pass  # best-effort; never block adjustment flow


def _safe_serialize(value: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return str(value)


def _proposal_signature(proposal: Dict[str, Any]) -> tuple[str, str, str]:
    """Stable identity for semantically identical adjustment proposals."""
    try:
        value = json.dumps(proposal.get("proposed_value"), sort_keys=True)
    except TypeError:
        value = str(proposal.get("proposed_value"))
    return (
        str(proposal.get("source", "unknown")),
        str(proposal.get("key", "")),
        value,
    )
