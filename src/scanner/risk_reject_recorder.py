"""Phase 63 US-378: RiskRejectRecorder.

Rolling buffer of raw rf_drawdown_pct and rf_streak_prob per pair per scan,
recorded at the risk gate decision site. Mirrors MomentumScoreRecorder (Phase 61)
but for the two-sub-gate risk gate.

Risk gate: rf_drawdown_pct <= max_drawdown_pct AND rf_streak_prob <= max_streak_prob
  - max_drawdown_pct default: 0.025 (2.5%)
  - max_streak_prob default: 0.95 (very permissive — rarely triggered)

Gap direction is INVERTED vs confidence/momentum:
  gap = rf_drawdown_pct - max_drawdown_pct (positive = over threshold = fail)

Design: wired into engine.py where signal.risk_gate_passed is available.
State: trained_data/risk_reject_state.jsonl
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

DEFAULT_STATE_PATH = Path("trained_data/risk_reject_state.jsonl")
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


class RiskRejectRecorder:
    """Record raw rf_drawdown_pct and rf_streak_prob per pair per scan.

    Tracks which sub-gate caused each risk gate failure so operators can
    distinguish drawdown kills (primary) from streak kills (rare, max_streak=0.95).

    Usage:
        recorder.record(
            pair="EUR_USD",
            rf_drawdown_pct=float(signal.rf_drawdown_pct),
            rf_streak_prob=float(signal.rf_streak_prob),
            risk_passed=bool(signal.risk_gate_passed),
            max_drawdown_pct=float(self.config.max_drawdown_pct),
        )
        dist = recorder.get_distribution()
        breakdown = recorder.get_kill_breakdown()
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
        rf_drawdown_pct: float,
        rf_streak_prob: float,
        risk_passed: bool,
        max_drawdown_pct: float,
        max_streak_prob: float = 0.95,
        scan_id: str = "",
    ) -> None:
        """Record one risk gate snapshot.

        drawdown_kill: True when rf_drawdown_pct > max_drawdown_pct
        streak_kill:   True when rf_streak_prob > max_streak_prob (and not drawdown_kill
                       alone — inferred from risk_passed=False AND not drawdown_kill)
        """
        drawdown_kill = bool(float(rf_drawdown_pct) > float(max_drawdown_pct))
        # streak_kill: over the streak threshold
        streak_kill = bool(float(rf_streak_prob) > float(max_streak_prob))

        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "rf_drawdown_pct": round(float(rf_drawdown_pct), 8),
            "rf_streak_prob": round(float(rf_streak_prob), 8),
            "risk_passed": bool(risk_passed),
            "drawdown_kill": drawdown_kill,
            "streak_kill": streak_kill,
            "max_drawdown_pct": round(float(max_drawdown_pct), 8),
            "scan_id": scan_id,
        }
        with self._lock:
            self._records.append(snap)
        logger.debug(
            "RiskRejectRecorder: pair=%s drawdown=%.5f streak=%.3f passed=%s "
            "drawdown_kill=%s streak_kill=%s",
            pair, rf_drawdown_pct, rf_streak_prob, risk_passed,
            drawdown_kill, streak_kill,
        )

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def get_distribution(self) -> Dict[str, Any]:
        """Return distribution stats for ALL rf_drawdown_pct scores."""
        with self._lock:
            records = list(self._records)
        return self._compute_dist(records)

    def get_fail_distribution(self) -> Dict[str, Any]:
        """Return distribution stats for rf_drawdown_pct where risk_passed=False."""
        with self._lock:
            records = [r for r in self._records if not r.get("risk_passed", True)]
        return self._compute_dist(records)

    def get_pass_rate(self) -> float:
        """Return fraction of records where risk_passed=True. 0.0 if empty."""
        with self._lock:
            records = list(self._records)
        if not records:
            return 0.0
        return sum(1 for r in records if r.get("risk_passed", False)) / len(records)

    def get_kill_breakdown(self) -> Dict[str, Any]:
        """Return kill source breakdown for failed records.

        Returns:
            drawdown_kills: count where drawdown_kill=True
            streak_kills:   count where streak_kill=True (may overlap)
            both_kills:     count where both drawdown_kill AND streak_kill
            total_fails:    count where risk_passed=False
            drawdown_kill_pct: drawdown_kills / total_fails * 100 (0 if no fails)
            streak_kill_pct:   streak_kills / total_fails * 100 (0 if no fails)
        """
        with self._lock:
            fails = [r for r in self._records if not r.get("risk_passed", True)]

        total_fails = len(fails)
        drawdown_kills = sum(1 for r in fails if r.get("drawdown_kill", False))
        streak_kills = sum(1 for r in fails if r.get("streak_kill", False))
        both_kills = sum(
            1 for r in fails
            if r.get("drawdown_kill", False) and r.get("streak_kill", False)
        )

        drawdown_kill_pct = (drawdown_kills / total_fails * 100.0) if total_fails > 0 else 0.0
        streak_kill_pct = (streak_kills / total_fails * 100.0) if total_fails > 0 else 0.0

        return {
            "drawdown_kills": drawdown_kills,
            "streak_kills": streak_kills,
            "both_kills": both_kills,
            "total_fails": total_fails,
            "drawdown_kill_pct": round(drawdown_kill_pct, 2),
            "streak_kill_pct": round(streak_kill_pct, 2),
        }

    def _compute_dist(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute distribution stats over rf_drawdown_pct values."""
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
        scores = sorted(float(r["rf_drawdown_pct"]) for r in records)
        n = len(scores)
        return {
            "count": n,
            "mean": round(sum(scores) / n, 8),
            "p25": round(_percentile(scores, 25), 8),
            "p50": round(_percentile(scores, 50), 8),
            "p75": round(_percentile(scores, 75), 8),
            "p90": round(_percentile(scores, 90), 8),
            "min": round(scores[0], 8),
            "max": round(scores[-1], 8),
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_state(self, path: Optional[Path] = None) -> None:
        """Atomically persist rolling buffer to JSONL file."""
        target = Path(path) if path else self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._records)
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            os.rename(tmp_path, str(target))
            tmp_path = None
        except Exception as exc:
            logger.error("RiskRejectRecorder.save_state failed: %s", exc)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _load_state(self) -> None:
        """Load rolling buffer from JSONL file on init."""
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
            logger.error("RiskRejectRecorder._load_state failed: %s", exc)
