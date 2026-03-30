"""Phase 67 US-391: SubInferenceRecorder.

Rolling buffer of sub-inference consensus ratios (votes/total), recorded per
pair per scan AFTER the sub-inference loop completes. Captures the actual data
behind `agent_passed` — the universal blocker identified in virtual_trades.

The sub-inference gate computes:
  vote_required = ceil(total * sub_inference_vote_threshold)
  agent_passed = votes >= vote_required

This recorder captures: votes, total, consensus_ratio (votes/total),
and agent_passed. Enables gap analysis for adaptive threshold tuning.

Design: wired into engine.py at the agent_passed assignment site (~line 2365).
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

DEFAULT_STATE_PATH = Path("trained_data/sub_inference_state.jsonl")
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


class SubInferenceRecorder:
    """Record sub-inference consensus ratios per pair per scan.

    Usage:
        recorder.record(pair, votes=1, total=3, agent_passed=False)
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
        votes: int,
        total: int,
        agent_passed: bool,
        scan_id: str = "",
    ) -> None:
        """Record one sub-inference consensus snapshot."""
        consensus_ratio = round(votes / total, 6) if total > 0 else 0.0
        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "votes": int(votes),
            "total": int(total),
            "consensus_ratio": consensus_ratio,
            "agent_passed": bool(agent_passed),
            "scan_id": scan_id,
        }
        with self._lock:
            self._records.append(snap)
        logger.debug(
            "SubInferenceRecorder: pair=%s votes=%d/%d ratio=%.4f passed=%s",
            pair, votes, total, consensus_ratio, agent_passed,
        )

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def get_distribution(self) -> Dict[str, Any]:
        """Return distribution stats across ALL recorded consensus ratios."""
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
        ratios = sorted(float(r["consensus_ratio"]) for r in records)
        n = len(ratios)
        return {
            "count": n,
            "mean": round(sum(ratios) / n, 6),
            "p25": round(_percentile(ratios, 25), 6),
            "p50": round(_percentile(ratios, 50), 6),
            "p75": round(_percentile(ratios, 75), 6),
            "p90": round(_percentile(ratios, 90), 6),
            "min": round(ratios[0], 6),
            "max": round(ratios[-1], 6),
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
            logger.error("SubInferenceRecorder.save_state failed: %s", exc)
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
            logger.error("SubInferenceRecorder._load_state failed: %s", exc)
