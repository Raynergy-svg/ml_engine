"""HomeworkReviewer — operator A/R/E/S transitions + training signal emit.

State transitions:
  pending → approved   (signal emitted with Buddy's proposed deltas)
  pending → rejected   (note required; signal carries operator's correction;
                        heuristic_fired logged for future tuning)
  pending → edited     (operator's edits replace Buddy's proposed deltas)
  pending → snoozed    (entry stays in pending with status=snoozed_until_X)

See spec §4.5 (Training Signal payload).
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.automation.homework.store import (
    DEFAULT_PENDING_PATH,
    HomeworkStore,
)
from src.scanner.automation.homework.types import HomeworkEntry, TrainingSignal

logger = logging.getLogger(__name__)

DEFAULT_AGENT_WEIGHT_DELTA = 0.02

DEFAULT_REJECTED_LOG_PATH = (
    DEFAULT_PENDING_PATH.parent / "rejected_heuristics_log.jsonl"
)


class HomeworkReviewer:
    """Process operator decisions on homework entries; emit training signals.

    Args:
        store: HomeworkStore for pending → history transitions.
        rejected_log_path: where to log heuristics that operator rejected
            (for future tuning). Default: .claude/rejected_heuristics_log.jsonl
    """

    def __init__(
        self,
        store: HomeworkStore,
        rejected_log_path: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.rejected_log_path = rejected_log_path or DEFAULT_REJECTED_LOG_PATH

    # ---------------- public API ----------------

    def approve(self, homework_id: str) -> Optional[TrainingSignal]:
        """Apply Buddy's proposed deltas as-is."""
        entry = self._find(homework_id)
        if entry is None:
            return None
        signal = self._build_signal(entry, action="approved", note=None)
        moved = self.store.move_to_history(homework_id, grade="approved", note=None, edits=None)
        if not moved:
            return None
        return signal

    def reject(self, homework_id: str, note: str) -> Optional[TrainingSignal]:
        """Discard Buddy's deltas; record note + log heuristic for future tuning."""
        if not note or not note.strip():
            raise ValueError("reject() requires a non-empty note explaining the rejection")
        entry = self._find(homework_id)
        if entry is None:
            return None

        # Log the heuristic that fired so we can tune the catalog later.
        self._log_rejected_heuristic(entry, note)

        signal = self._build_signal(entry, action="rejected", note=note.strip())
        # Rejected signals carry empty deltas — operator's note is the future training data
        signal = dataclasses.replace(
            signal,
            agent_weight_deltas={},
            regime_prior_deltas={},
        )
        moved = self.store.move_to_history(
            homework_id, grade="rejected", note=note.strip(), edits=None
        )
        if not moved:
            return None
        return signal

    def edit(
        self, homework_id: str, edits: Dict[str, Any]
    ) -> Optional[TrainingSignal]:
        """Apply operator's edited deltas instead of Buddy's proposed deltas."""
        entry = self._find(homework_id)
        if entry is None:
            return None
        signal = self._build_signal(
            entry,
            action="edited",
            note=edits.get("note"),
        )
        # Override deltas if edits supply them
        if "agent_weight_deltas" in edits:
            signal = dataclasses.replace(
                signal, agent_weight_deltas=dict(edits["agent_weight_deltas"])
            )
        if "regime_prior_deltas" in edits:
            signal = dataclasses.replace(
                signal, regime_prior_deltas=dict(edits["regime_prior_deltas"])
            )
        moved = self.store.move_to_history(
            homework_id, grade="edited", note=edits.get("note"), edits=edits
        )
        if not moved:
            return None
        return signal

    def snooze(self, homework_id: str, hours: float = 24.0) -> bool:
        """Hide entry until ts; remains in pending file with status=snoozed_until_X."""
        pending = self.store.list_pending()
        target = next((e for e in pending if e.homework_id == homework_id), None)
        if target is None:
            return False
        until = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
        new_pending: List[HomeworkEntry] = []
        for e in pending:
            if e.homework_id == homework_id:
                new_pending.append(
                    dataclasses.replace(e, status=f"snoozed_until_{until}")
                )
            else:
                new_pending.append(e)
        self.store.rewrite_pending(new_pending)
        return True

    # ---------------- internals ----------------

    def _find(self, homework_id: str) -> Optional[HomeworkEntry]:
        return next(
            (e for e in self.store.list_pending() if e.homework_id == homework_id),
            None,
        )

    def _build_signal(
        self,
        entry: HomeworkEntry,
        action: str,
        note: Optional[str],
    ) -> TrainingSignal:
        """Convert HomeworkEntry into the proposed TrainingSignal payload."""
        deltas: Dict[str, float] = {}
        for name in entry.agents_to_reinforce:
            deltas[name] = deltas.get(name, 0.0) + DEFAULT_AGENT_WEIGHT_DELTA
        for name in entry.agents_to_penalize:
            deltas[name] = deltas.get(name, 0.0) - DEFAULT_AGENT_WEIGHT_DELTA

        return TrainingSignal(
            homework_id=entry.homework_id,
            trade_id=entry.trade_id,
            outcome=entry.close_reason,
            agent_weight_deltas=deltas,
            regime_prior_deltas={},  # populated when proposed_lesson includes a regime adjustment
            heuristic_fired=None,  # the generator doesn't currently surface this; future work
            operator_action=action,
            operator_note=note,
        )

    def _log_rejected_heuristic(self, entry: HomeworkEntry, note: str) -> None:
        """Append rejected-heuristic event to log for future catalog tuning."""
        try:
            self.rejected_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.rejected_log_path, "a", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(
                    json.dumps({
                        "homework_id": entry.homework_id,
                        "trade_id": entry.trade_id,
                        "proposed_lesson": entry.proposed_lesson,
                        "operator_note": note,
                        "rejected_at": datetime.now(timezone.utc).isoformat(),
                    })
                    + "\n"
                )
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            logger.exception("HomeworkReviewer._log_rejected_heuristic failed: %s", e)
