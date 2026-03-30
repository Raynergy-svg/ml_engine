"""Phase 75: DisagreementRecorder.

Rolling buffer of model_disagreement values from the uncertainty agent.
Records the heuristic disagreement score (0.0-1.0) per pair per scan.

model_disagreement is computed from 3 heuristics in _team.py:
  - price vs SMA_20
  - RSI position (>50 or <50)
  - 5-bar returns direction
oppose/total gives the ratio. With 3 heuristics: 0.0, 0.33, 0.67, 1.0.

DISAGREEMENT_HARD_FLOOR (default 0.30) hard-blocks any trade where
model_disagreement > floor. With 3 heuristics, 1/3 = 0.33 > 0.30 blocks.

This recorder enables gap analysis for adaptive threshold tuning.
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

DEFAULT_STATE_PATH = Path("trained_data/disagreement_state.jsonl")
DEFAULT_MAX_SIZE = 500


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


class DisagreementRecorder:
    """Record model_disagreement per pair per scan."""

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

    def record(
        self,
        pair: str,
        disagreement: float,
        hard_blocked: bool,
        scan_id: str = "",
    ) -> None:
        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "disagreement": round(float(disagreement), 6),
            "hard_blocked": bool(hard_blocked),
            "scan_id": scan_id,
        }
        with self._lock:
            self._records.append(snap)

    def get_distribution(self) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records)
        return self._compute_dist(records)

    def get_fail_distribution(self) -> Dict[str, Any]:
        with self._lock:
            records = [r for r in self._records if r.get("hard_blocked", False)]
        return self._compute_dist(records)

    def get_pass_rate(self) -> float:
        with self._lock:
            records = list(self._records)
        if not records:
            return 0.0
        return sum(1 for r in records if not r.get("hard_blocked", True)) / len(records)

    def _compute_dist(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {"count": 0, "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
        scores = sorted(float(r["disagreement"]) for r in records)
        n = len(scores)
        return {
            "count": n,
            "mean": round(sum(scores) / n, 6),
            "p25": round(_percentile(scores, 25), 6),
            "p50": round(_percentile(scores, 50), 6),
            "p75": round(_percentile(scores, 75), 6),
            "p90": round(_percentile(scores, 90), 6),
            "min": round(scores[0], 6),
            "max": round(scores[-1], 6),
        }

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
            logger.error("DisagreementRecorder.save_state failed: %s", exc)
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            loaded: List[Dict[str, Any]] = []
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
            logger.error("DisagreementRecorder._load_state failed: %s", exc)
