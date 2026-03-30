"""Gate rejection tracking — per-gate, per-pair rejection diagnostics.

Phase 56 US-346: Track which gate blocks each pair on each scan.
Surfaces the bottleneck that prevents trade execution so calibration
can be targeted rather than guessed.

Usage:
    tracker = GateRejectionTracker()
    tracker.record_rejection(pair="EUR_USD", gate_name="confidence", reason="0.21 < 0.60", confidence=0.21)
    stats = tracker.get_stats()
    top = tracker.get_top_blocking_gate()
    tracker.save_state()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PERSIST_PATH = _PROJECT_ROOT / "trained_data" / "gate_rejection_stats.json"

# Version for forward compatibility
_STATE_VERSION = 1


@dataclass
class RejectionRecord:
    """Single gate rejection event."""

    pair: str
    gate_name: str
    reason: str
    confidence: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "gate_name": self.gate_name,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp,
        }


class GateRejectionTracker:
    """Track per-gate rejection counts and surface the execution bottleneck.

    Records every gate rejection with pair, gate name, reason, and
    confidence at time of rejection. Provides stats so the system can
    identify which gate is responsible for blocking trade generation.

    Usage:
        tracker = GateRejectionTracker()
        tracker.record_rejection("EUR_USD", "confidence", "0.21 < 0.60", 0.21)
        tracker.record_rejection("GBP_USD", "agent", "agent_passed=False", 0.55)
        stats = tracker.get_stats()
        top = tracker.get_top_blocking_gate()
        tracker.save_state()
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        max_recent_records: int = 500,
    ):
        self.persistence_path = Path(persistence_path or _DEFAULT_PERSIST_PATH)
        self.max_recent_records = max_recent_records

        # Total rejection counts per gate name
        self._gate_counts: Dict[str, int] = defaultdict(int)

        # Per-pair rejection counts per gate: {pair: {gate: count}}
        self._pair_gate_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Recent rejection records (capped)
        self._recent_records: List[RejectionRecord] = []

        # Lifetime totals
        self._total_rejections: int = 0
        self._scans_observed: int = 0

        self._load_state()

    # ── Public API ──────────────────────────────────────────────────────────

    def record_rejection(
        self,
        pair: str,
        gate_name: str,
        reason: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Record a gate rejection for a pair.

        Args:
            pair: Currency pair (e.g. "EUR_USD")
            gate_name: Name of the gate that blocked (e.g. "confidence", "agent", "risk")
            reason: Human-readable reason string (e.g. "0.21 < 0.60")
            confidence: Confidence score at time of rejection
        """
        gate_name = gate_name.strip().lower() or "unknown"
        record = RejectionRecord(
            pair=pair,
            gate_name=gate_name,
            reason=reason,
            confidence=float(confidence),
        )

        self._gate_counts[gate_name] += 1
        self._pair_gate_counts[pair][gate_name] += 1
        self._total_rejections += 1

        self._recent_records.append(record)
        # Cap in-memory list
        if len(self._recent_records) > self.max_recent_records:
            self._recent_records = self._recent_records[-self.max_recent_records :]

    def mark_scan_complete(self) -> None:
        """Call once per scan cycle to increment scan counter."""
        self._scans_observed += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return rejection statistics across all gates and pairs.

        Returns:
            Dict with:
                total_rejections: int
                scans_observed: int
                gate_counts: {gate_name: count}
                pair_counts: {pair: {gate: count}}
                top_blocking_gate: str
                rejections_per_scan: float
        """
        total_gates = dict(self._gate_counts)
        top_gate = self.get_top_blocking_gate()
        rps = (
            round(self._total_rejections / self._scans_observed, 2)
            if self._scans_observed > 0
            else 0.0
        )

        pair_counts: Dict[str, Dict[str, int]] = {}
        for pair, gate_map in self._pair_gate_counts.items():
            pair_counts[pair] = dict(gate_map)

        return {
            "total_rejections": self._total_rejections,
            "scans_observed": self._scans_observed,
            "gate_counts": total_gates,
            "pair_counts": pair_counts,
            "top_blocking_gate": top_gate,
            "rejections_per_scan": rps,
        }

    def get_top_blocking_gate(self) -> str:
        """Return the gate name responsible for the most rejections.

        Returns:
            Gate name string, or "none" if no rejections recorded.
        """
        if not self._gate_counts:
            return "none"
        return max(self._gate_counts, key=lambda g: self._gate_counts[g])

    def get_gate_rejection_rate(self, gate_name: str) -> float:
        """Return the fraction of all rejections attributable to this gate.

        Returns:
            Float in [0, 1]. 0.0 if no rejections recorded.
        """
        if self._total_rejections == 0:
            return 0.0
        return self._gate_counts.get(gate_name.lower(), 0) / self._total_rejections

    def get_recent_records(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the last n rejection records as dicts."""
        return [r.to_dict() for r in self._recent_records[-n:]]

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Atomically persist state to disk."""
        state = {
            "version": _STATE_VERSION,
            "total_rejections": self._total_rejections,
            "scans_observed": self._scans_observed,
            "gate_counts": dict(self._gate_counts),
            "pair_gate_counts": {
                pair: dict(gate_map)
                for pair, gate_map in self._pair_gate_counts.items()
            },
            "recent_records": [
                r.to_dict()
                for r in self._recent_records[-self.max_recent_records :]
            ],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.persistence_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(state, f, indent=2, sort_keys=True)
            os.rename(tmp_path, self.persistence_path)
            logger.debug(
                "GateRejectionTracker: Saved state (%d rejections, %d scans)",
                self._total_rejections,
                self._scans_observed,
            )
        except Exception as e:
            logger.warning("GateRejectionTracker: Save failed: %s", e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load_state(self) -> None:
        """Load persisted state from disk. Graceful fallback on any error."""
        if not self.persistence_path.exists():
            return
        try:
            raw = json.loads(self.persistence_path.read_text())
            if not isinstance(raw, dict):
                return

            self._total_rejections = int(raw.get("total_rejections", 0))
            self._scans_observed = int(raw.get("scans_observed", 0))

            for gate, count in raw.get("gate_counts", {}).items():
                self._gate_counts[gate] = int(count)

            for pair, gate_map in raw.get("pair_gate_counts", {}).items():
                for gate, count in gate_map.items():
                    self._pair_gate_counts[pair][gate] = int(count)

            for rec_dict in raw.get("recent_records", []):
                try:
                    rec = RejectionRecord(
                        pair=str(rec_dict.get("pair", "")),
                        gate_name=str(rec_dict.get("gate_name", "unknown")),
                        reason=str(rec_dict.get("reason", "")),
                        confidence=float(rec_dict.get("confidence", 0.0)),
                        timestamp=str(rec_dict.get("timestamp", "")),
                    )
                    self._recent_records.append(rec)
                except Exception:
                    continue

            logger.debug(
                "GateRejectionTracker: Loaded state (%d rejections, %d scans)",
                self._total_rejections,
                self._scans_observed,
            )
        except Exception as e:
            logger.warning(
                "GateRejectionTracker: Failed to load state, starting fresh: %s", e
            )
            self._gate_counts = defaultdict(int)
            self._pair_gate_counts = defaultdict(lambda: defaultdict(int))
            self._recent_records = []
            self._total_rejections = 0
            self._scans_observed = 0
