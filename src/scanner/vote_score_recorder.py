"""Phase 66 US-388: VoteScoreRecorder.

Rolling buffer of weighted_vote_score values, recorded per pair per scan
AFTER agent team evaluation. Mirrors MomentumScoreRecorder (Phase 61 US-371)
but for the agent consensus gate (weighted_vote_score vs weighted_vote_threshold).

Weighted vote scores are in 0.0-1.0 range (agent team weighted average).
weighted_vote_threshold is typically 0.50-0.60.

Design: wired into engine.py after _agent_team.evaluate() returns,
inside _run_sub_inference_agents_for_pair().
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

DEFAULT_STATE_PATH = Path("trained_data/vote_score_state.jsonl")
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


class VoteScoreRecorder:
    """Record weighted_vote_score per pair per scan.

    Usage:
        recorder.record(pair, vote_score=float(analysis.weighted_vote_score),
                        agent_passed=bool(analysis.agent_passed))
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
        vote_score: float,
        agent_passed: bool,
        scan_id: str = "",
    ) -> None:
        """Record one vote score snapshot."""
        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "vote_score": round(float(vote_score), 6),
            "agent_passed": bool(agent_passed),
            "scan_id": scan_id,
        }
        with self._lock:
            self._records.append(snap)
        logger.debug(
            "VoteScoreRecorder: pair=%s score=%.4f passed=%s",
            pair, vote_score, agent_passed,
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
        """Return distribution stats for records where agent_passed=False."""
        with self._lock:
            records = [r for r in self._records if not r.get("agent_passed", True)]
        return self._compute_dist(records)

    def get_pass_rate(self) -> float:
        """Return fraction of records where agent_passed=True."""
        with self._lock:
            records = list(self._records)
        if not records:
            return 0.0
        return sum(1 for r in records if r.get("agent_passed", False)) / len(records)

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
        scores = sorted(float(r["vote_score"]) for r in records)
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
            logger.error("VoteScoreRecorder.save_state failed: %s", exc)
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
            logger.error("VoteScoreRecorder._load_state failed: %s", exc)
