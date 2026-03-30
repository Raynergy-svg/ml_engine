"""Phase 61 US-371: MomentumScoreRecorder.

Rolling buffer of raw XGBoost momentum scores, recorded per pair per scan
BEFORE gate decision is used. Mirrors RawConfidenceRecorder (Phase 58 US-358)
but for the momentum gate (signal.xgb_momentum vs config.min_momentum).

Momentum scores are in 0.0-1.0 range (XGBoost output probability).
min_momentum threshold is typically 0.40-0.60.

Design: wired into engine.py where signal.xgb_momentum is available.
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

DEFAULT_STATE_PATH = Path("trained_data/momentum_score_state.jsonl")
DEFAULT_MAX_SIZE = 500


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Compute percentile from a pre-sorted list (no scipy needed)."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


class MomentumScoreRecorder:
    """Record raw XGBoost momentum scores per pair per scan.

    Usage:
        recorder.record(pair, momentum_score=float(signal.xgb_momentum),
                        momentum_passed=bool(signal.momentum_gate_passed))
        dist = recorder.get_distribution()
        fail_dist = recorder.get_fail_distribution()
    """

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

    def record(
        self,
        pair: str,
        momentum_score: float,
        momentum_passed: bool,
        scan_id: str = "",
    ) -> None:
        """Record one momentum snapshot."""
        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "momentum_score": round(float(momentum_score), 6),
            "momentum_passed": bool(momentum_passed),
            "scan_id": scan_id,
        }
        with self._lock:
            self._records.append(snap)
        logger.debug(
            "MomentumScoreRecorder: pair=%s score=%.4f passed=%s",
            pair, momentum_score, momentum_passed,
        )

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def get_distribution(self) -> Dict[str, Any]:
        """Return distribution stats across ALL recorded scores."""
        with self._lock:
            records = list(self._records)
        return self._compute_dist(records)

    def get_fail_distribution(self) -> Dict[str, Any]:
        """Return distribution stats for records where momentum_passed=False."""
        with self._lock:
            records = [r for r in self._records if not r.get("momentum_passed", True)]
        return self._compute_dist(records)

    def get_pass_rate(self) -> float:
        """Return fraction of records where momentum_passed=True."""
        with self._lock:
            records = list(self._records)
        if not records:
            return 0.0
        return sum(1 for r in records if r.get("momentum_passed", False)) / len(records)

    def _compute_dist(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {
                "count": 0,
                "mean": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        scores = sorted(float(r["momentum_score"]) for r in records)
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
            logger.error("MomentumScoreRecorder.save_state failed: %s", exc)
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
            logger.error("MomentumScoreRecorder._load_state failed: %s", exc)
