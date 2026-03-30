"""Phase 58 US-358: RawConfidenceRecorder.

Records pre-penalty confidence scores into a rolling deque so the system
can observe the natural ensemble output distribution independently of any
penalty mechanism. This is the observability foundation for Phase 58's
threshold gap analysis and adaptive confidence floor.

Design:
- Insert BEFORE any Phase 57 penalty block in engine.py
- Max 500 entries (rolling; evicts oldest on overflow)
- Persists as JSONL for high-frequency appends
- Thread-safe via threading.Lock
"""

from __future__ import annotations

import json
import logging
import tempfile
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("trained_data/raw_confidence_state.jsonl")
DEFAULT_MAX_SIZE = 500


@dataclass
class ConfidenceSnapshot:
    """Single pre-penalty confidence observation."""
    pair: str
    raw_confidence: float
    scan_id: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "raw_confidence": self.raw_confidence,
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfidenceSnapshot":
        return cls(
            pair=str(d.get("pair", "")),
            raw_confidence=float(d.get("raw_confidence", 0.0)),
            scan_id=str(d.get("scan_id", "")),
            timestamp=str(d.get("timestamp", "")),
        )


class RawConfidenceRecorder:
    """Rolling buffer of pre-penalty confidence scores.

    Records raw confidence BEFORE any penalty modification to expose
    the natural signal distribution of the ensemble models.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        self._max_size = max_size
        self._state_path = Path(state_path)
        self._lock = threading.Lock()
        self._records: deque[ConfidenceSnapshot] = deque(maxlen=max_size)
        self._load_state()

    # ------------------------------------------------------------------ #
    # Write side                                                           #
    # ------------------------------------------------------------------ #

    def record(
        self,
        pair: str,
        raw_confidence: float,
        scan_id: str = "",
    ) -> None:
        """Store a pre-penalty confidence observation."""
        snap = ConfidenceSnapshot(
            pair=pair,
            raw_confidence=float(raw_confidence),
            scan_id=scan_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._records.append(snap)
        logger.debug(
            "RawConfidenceRecorder.record: pair=%s raw_conf=%.2f scan=%s",
            pair, raw_confidence, scan_id,
        )

    # ------------------------------------------------------------------ #
    # Read side                                                            #
    # ------------------------------------------------------------------ #

    def get_distribution(self) -> Dict[str, Any]:
        """Return statistical distribution of all recorded raw confidences."""
        with self._lock:
            values = [r.raw_confidence for r in self._records]

        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def percentile(p: float) -> float:
            idx = (p / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

        return {
            "count": n,
            "mean": round(sum(sorted_vals) / n, 4),
            "median": round(percentile(50), 4),
            "p25": round(percentile(25), 4),
            "p75": round(percentile(75), 4),
            "p90": round(percentile(90), 4),
            "min": round(sorted_vals[0], 4),
            "max": round(sorted_vals[-1], 4),
        }

    def get_distribution_by_pair(self, pair: str) -> Dict[str, Any]:
        """Return distribution stats for a specific currency pair."""
        with self._lock:
            values = [
                r.raw_confidence for r in self._records if r.pair == pair
            ]

        if not values:
            return {"count": 0, "pair": pair, "mean": 0.0, "median": 0.0,
                    "p75": 0.0, "min": 0.0, "max": 0.0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        median = (
            sorted_vals[mid] if n % 2 == 1
            else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        )
        p75_idx = min(int(0.75 * (n - 1)), n - 1)

        return {
            "count": n,
            "pair": pair,
            "mean": round(sum(sorted_vals) / n, 4),
            "median": round(median, 4),
            "p75": round(sorted_vals[p75_idx], 4),
            "min": round(sorted_vals[0], 4),
            "max": round(sorted_vals[-1], 4),
        }

    def get_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the last n records (most recent last)."""
        with self._lock:
            records = list(self._records)
        return [r.to_dict() for r in records[-n:]]

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_state(self, path: Optional[Path] = None) -> None:
        """Atomically persist the rolling deque to JSONL."""
        target = Path(path) if path else self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            records = list(self._records)

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target.parent), suffix=".tmp"
            )
            with os.fdopen(fd, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            os.rename(tmp_path, str(target))
            logger.debug(
                "RawConfidenceRecorder: saved %d records to %s",
                len(records), target,
            )
        except Exception as exc:
            logger.error("RawConfidenceRecorder.save_state failed: %s", exc)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _load_state(self) -> None:
        """Load persisted JSONL records into the deque."""
        if not self._state_path.exists():
            return
        try:
            loaded: List[ConfidenceSnapshot] = []
            with open(self._state_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        loaded.append(ConfidenceSnapshot.from_dict(d))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
            # Respect max_size: take the most recent entries
            start = max(0, len(loaded) - self._max_size)
            with self._lock:
                self._records = deque(loaded[start:], maxlen=self._max_size)
            logger.debug(
                "RawConfidenceRecorder: loaded %d records from %s",
                len(self._records), self._state_path,
            )
        except Exception as exc:
            logger.error("RawConfidenceRecorder._load_state failed: %s", exc)
            with self._lock:
                self._records = deque(maxlen=self._max_size)
