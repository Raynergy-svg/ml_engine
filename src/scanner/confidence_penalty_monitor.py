"""Confidence penalty stack detector — alert when cumulative penalties exceed threshold.

Phase 56 US-347: Detect when multiple active penalty systems (overconfidence,
concept drift, explainability, disagreement) stack to push a pair's confidence
below the execution floor. Logs a floor_breach alert with full penalty breakdown.

Usage:
    monitor = ConfidencePenaltyMonitor()
    monitor.record_penalties("EUR_USD", {"overconfidence": 0.08, "drift": 0.06, "disagreement": 0.04})
    total = monitor.get_total_penalty("EUR_USD")
    breach = monitor.is_floor_breach("EUR_USD")
    report = monitor.generate_report()
    monitor.save_state()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PERSIST_PATH = _PROJECT_ROOT / "trained_data" / "confidence_floor_breaches.json"

# Default threshold above which cumulative penalties are flagged
DEFAULT_BREACH_THRESHOLD = 0.20

_STATE_VERSION = 1


@dataclass
class PenaltySnapshot:
    """Cumulative penalty snapshot for a pair in a single scan."""

    pair: str
    penalties: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_penalty(self) -> float:
        return sum(self.penalties.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
            "total_penalty": round(self.total_penalty, 4),
            "timestamp": self.timestamp,
        }


class ConfidencePenaltyMonitor:
    """Detect when stacked confidence penalties push pairs below the execution floor.

    Records per-source penalty contributions for each pair on each scan.
    When the cumulative penalty exceeds the breach threshold, logs a WARNING
    and appends to the breach history file.

    This reveals whether the system is self-suppressing (confidence penalties
    collectively eliminating all trade opportunities) so calibration can be
    targeted rather than guessed.

    Usage:
        monitor = ConfidencePenaltyMonitor()
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.08, "drift": 0.06})
        if monitor.is_floor_breach("EUR_USD"):
            print("Confidence floor breached!")
        report = monitor.generate_report()
        monitor.save_state()
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        breach_threshold: float = DEFAULT_BREACH_THRESHOLD,
        max_history: int = 500,
    ):
        self.persistence_path = Path(persistence_path or _DEFAULT_PERSIST_PATH)
        self.breach_threshold = float(breach_threshold)
        self.max_history = max_history

        # Current scan's snapshots: {pair: PenaltySnapshot}
        self._current_scan: Dict[str, PenaltySnapshot] = {}

        # Historical breach records
        self._breach_history: List[Dict[str, Any]] = []

        # Counters
        self._total_scans: int = 0
        self._total_breaches: int = 0

        self._load_state()

    # ── Public API ──────────────────────────────────────────────────────────

    def record_penalties(
        self,
        pair: str,
        penalties: Dict[str, float],
    ) -> None:
        """Record penalty contributions for a pair in the current scan.

        Args:
            pair: Currency pair (e.g. "EUR_USD")
            penalties: Dict mapping source name to penalty value.
                       E.g. {"overconfidence": 0.08, "concept_drift": 0.06}
                       Penalty values should be positive floats (they are deducted
                       from confidence).
        """
        # Sanitize — ensure all values are positive floats
        clean_penalties = {
            str(src): max(0.0, float(v))
            for src, v in penalties.items()
        }

        if pair in self._current_scan:
            # Merge with existing snapshot (accumulate sources)
            existing = self._current_scan[pair]
            existing.penalties.update(clean_penalties)
        else:
            self._current_scan[pair] = PenaltySnapshot(
                pair=pair,
                penalties=clean_penalties,
            )

        # Check for floor breach and log immediately
        snapshot = self._current_scan[pair]
        if snapshot.total_penalty >= self.breach_threshold:
            logger.warning(
                "Phase 56 (US-347): Confidence floor breach on %s — "
                "total_penalty=%.3f (threshold=%.3f) sources=%s",
                pair,
                snapshot.total_penalty,
                self.breach_threshold,
                {s: f"{v:.3f}" for s, v in snapshot.penalties.items()},
            )

    def get_total_penalty(self, pair: str) -> float:
        """Return total cumulative penalty for a pair in the current scan."""
        if pair not in self._current_scan:
            return 0.0
        return self._current_scan[pair].total_penalty

    def is_floor_breach(
        self,
        pair: str,
        threshold: Optional[float] = None,
    ) -> bool:
        """Return True when cumulative penalty for pair exceeds threshold.

        Args:
            pair: Currency pair
            threshold: Override threshold (default: self.breach_threshold)
        """
        limit = threshold if threshold is not None else self.breach_threshold
        return self.get_total_penalty(pair) >= limit

    def generate_report(self) -> Dict[str, Any]:
        """Generate a report of the current scan's penalty landscape.

        Returns:
            Dict keyed by pair with:
                total_penalty: float
                sources: {source: penalty}
                floor_breach: bool
        """
        report: Dict[str, Any] = {}
        for pair, snapshot in self._current_scan.items():
            report[pair] = {
                "total_penalty": round(snapshot.total_penalty, 4),
                "sources": {s: round(v, 4) for s, v in snapshot.penalties.items()},
                "floor_breach": snapshot.total_penalty >= self.breach_threshold,
            }
        return report

    def get_breach_rate(self) -> float:
        """Return fraction of scans where at least one pair had a floor breach.

        Returns:
            Float in [0, 1]. 0.0 if no scans recorded.
        """
        if self._total_scans == 0:
            return 0.0
        return self._total_breaches / self._total_scans

    def flush_scan(self) -> None:
        """Call at the end of each scan to persist breach history and reset."""
        breaches = [
            snap.to_dict()
            for snap in self._current_scan.values()
            if snap.total_penalty >= self.breach_threshold
        ]

        self._total_scans += 1
        if breaches:
            self._total_breaches += 1
            self._breach_history.extend(breaches)
            # Cap history
            if len(self._breach_history) > self.max_history:
                self._breach_history = self._breach_history[-self.max_history:]

        # Reset for next scan
        self._current_scan = {}

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Atomically persist breach history to disk."""
        state = {
            "version": _STATE_VERSION,
            "total_scans": self._total_scans,
            "total_breaches": self._total_breaches,
            "breach_threshold": self.breach_threshold,
            "breach_history": self._breach_history[-self.max_history:],
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
                "ConfidencePenaltyMonitor: Saved state (%d scans, %d breaches)",
                self._total_scans,
                self._total_breaches,
            )
        except Exception as e:
            logger.warning("ConfidencePenaltyMonitor: Save failed: %s", e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load_state(self) -> None:
        """Load persisted breach history. Graceful fallback on any error."""
        if not self.persistence_path.exists():
            return
        try:
            raw = json.loads(self.persistence_path.read_text())
            if not isinstance(raw, dict):
                return
            self._total_scans = int(raw.get("total_scans", 0))
            self._total_breaches = int(raw.get("total_breaches", 0))
            self._breach_history = list(raw.get("breach_history", []))
            logger.debug(
                "ConfidencePenaltyMonitor: Loaded state (%d scans, %d breaches)",
                self._total_scans,
                self._total_breaches,
            )
        except Exception as e:
            logger.warning(
                "ConfidencePenaltyMonitor: Failed to load state, starting fresh: %s", e
            )
            self._total_scans = 0
            self._total_breaches = 0
            self._breach_history = []
