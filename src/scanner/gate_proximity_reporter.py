"""Phase 59 US-364: GateProximityReporter.

For pairs rejected at the confidence gate, records how close each was
to crossing the threshold. Classifies each rejection as:
  NEAR_MISS    — gap <= 3.0 pts  (actionable: 1-3pt relaxation would unblock)
  MODERATE_MISS — 3.0 < gap <= 8.0 pts
  FAR_MISS     — gap > 8.0 pts  (structural; threshold too high or signal too weak)

When near_miss_rate > 50%, combined with NEAR_THRESHOLD from ThresholdGapAnalyzer,
this is a strong two-signal confirmation for AdaptiveConfidenceFloor activation.

Design: wired into engine.py where per-pair results and config.min_confidence
are both available after score computation.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("trained_data/gate_proximity_state.jsonl")
DEFAULT_MAX_SIZE = 500

# Gap classification thresholds (points on 0-100 scale)
NEAR_MISS_THRESHOLD = 3.0
MODERATE_MISS_THRESHOLD = 8.0

MISS_CLASS_NEAR = "NEAR_MISS"
MISS_CLASS_MODERATE = "MODERATE_MISS"
MISS_CLASS_FAR = "FAR_MISS"
NEAR_MISS_ALERT_RATE = 50.0   # Alert when near_miss_rate > 50%


def _classify_gap(gap: float) -> str:
    if gap <= NEAR_MISS_THRESHOLD:
        return MISS_CLASS_NEAR
    elif gap <= MODERATE_MISS_THRESHOLD:
        return MISS_CLASS_MODERATE
    return MISS_CLASS_FAR


class GateProximityReporter:
    """Record confidence-gate near-misses to identify actionable threshold relaxation candidates."""

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        self._max_size = max_size
        self._state_path = Path(state_path)
        self._lock = threading.Lock()
        self._records: deque[Dict[str, Any]] = deque(maxlen=max_size)
        self._load_state()

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record_scan(self, analyses: List[Any], min_confidence: float) -> None:
        """Record proximity data for pairs that failed the confidence gate.

        Args:
            analyses: List of PairAnalysis objects or dicts.
            min_confidence: Current engine threshold (0-100 scale).
        """
        def get_float(a: Any, *names: str) -> Optional[float]:
            for name in names:
                v = getattr(a, name, None)
                if v is None:
                    v = a.get(name) if isinstance(a, dict) else None
                if v is not None:
                    return float(v)
            return None

        def get_flag(a: Any, *names: str) -> bool:
            for name in names:
                v = getattr(a, name, None)
                if v is None:
                    v = a.get(name) if isinstance(a, dict) else None
                if v is not None:
                    return bool(v)
            return False

        def get_str(a: Any, *names: str) -> str:
            for name in names:
                v = getattr(a, name, None)
                if v is None:
                    v = a.get(name) if isinstance(a, dict) else None
                if v is not None:
                    return str(v)
            return "UNKNOWN"

        new_records = []
        for a in analyses:
            conf_passed = get_flag(a, "confidence_gate_passed", "confidence_passed")
            if conf_passed:
                continue  # Only record confidence rejections
            confidence = get_float(a, "confidence")
            if confidence is None:
                continue
            gap = min_confidence - confidence
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pair": get_str(a, "pair"),
                "confidence": round(confidence, 4),
                "min_confidence": round(min_confidence, 4),
                "gap": round(gap, 4),
                "miss_class": _classify_gap(gap),
            }
            new_records.append(record)

        if not new_records:
            return

        with self._lock:
            for rec in new_records:
                self._records.append(rec)

        near_miss_count = sum(1 for r in new_records if r["miss_class"] == MISS_CLASS_NEAR)
        near_miss_rate = (near_miss_count / len(new_records)) * 100 if new_records else 0.0
        if near_miss_rate > NEAR_MISS_ALERT_RATE:
            logger.info(
                "GateProximityReporter: %.0f%% of confidence rejects are near-misses "
                "(gap<=%.0fpt) — threshold relaxation may unblock them",
                near_miss_rate, NEAR_MISS_THRESHOLD,
            )

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def get_near_miss_rate(self) -> float:
        """Return fraction of all records classified as NEAR_MISS."""
        with self._lock:
            records = list(self._records)
        if not records:
            return 0.0
        near = sum(1 for r in records if r.get("miss_class") == MISS_CLASS_NEAR)
        return near / len(records)

    def get_proximity_summary(self) -> Dict[str, Any]:
        """Return aggregate proximity stats."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "total_conf_rejects": 0,
                "near_miss_count": 0,
                "near_miss_rate_pct": 0.0,
                "avg_gap": 0.0,
                "closest_gap": 0.0,
                "closest_pair": "",
            }

        near = [r for r in records if r.get("miss_class") == MISS_CLASS_NEAR]
        gaps = [r.get("gap", 0.0) for r in records]
        min_gap_rec = min(records, key=lambda r: r.get("gap", float("inf")))

        return {
            "total_conf_rejects": len(records),
            "near_miss_count": len(near),
            "near_miss_rate_pct": round((len(near) / len(records)) * 100, 2),
            "avg_gap": round(sum(gaps) / len(gaps), 4),
            "closest_gap": round(min_gap_rec.get("gap", 0.0), 4),
            "closest_pair": min_gap_rec.get("pair", ""),
        }

    def get_actionable_threshold(
        self,
        target_near_miss_unblock: float = 0.10,
    ) -> float:
        """Return min_confidence that would unblock at least target_near_miss_unblock of records.

        Scans all records and finds the threshold at which at least
        target_near_miss_unblock * total_records would be unblocked.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return 0.0

        # Get current threshold from most recent record
        current_threshold = max(r.get("min_confidence", 58.0) for r in records)
        target_unblock_count = max(1, int(target_near_miss_unblock * len(records)))

        # Sort by confidence descending — try from highest to lowest
        confs = sorted([r.get("confidence", 0.0) for r in records], reverse=True)

        if len(confs) < target_unblock_count:
            return round(confs[-1], 1) if confs else current_threshold

        # The threshold that unblocks exactly target_unblock_count pairs
        # = confidence of the (target_unblock_count)th highest rejected pair
        unblock_at = confs[target_unblock_count - 1]
        # Return slightly below that confidence to ensure the pair passes
        return round(max(unblock_at - 0.1, 0.0), 1)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_state(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path else self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._records)
        try:
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            os.rename(tmp, str(target))
        except Exception as exc:
            logger.error("GateProximityReporter.save_state failed: %s", exc)
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            loaded = []
            with open(self._state_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            loaded.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            start = max(0, len(loaded) - self._max_size)
            with self._lock:
                self._records = deque(loaded[start:], maxlen=self._max_size)
        except Exception as exc:
            logger.error("GateProximityReporter._load_state failed: %s", exc)
