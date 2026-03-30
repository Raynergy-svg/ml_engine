"""Phase 59 US-363: SignalFunnelTracker.

Tracks per-scan pipeline attrition from total evaluated pairs through each
gate stage to final is_tradeable verdict.

Pipeline stages:
  total_evaluated → passed_confidence → passed_momentum → passed_risk → is_tradeable

This answers: "signals cleared confidence — where exactly do they die?"
If dominant_kill_stage = 'momentum', Phase 57/58 fixes are working but
the momentum gate needs attention. If 'confidence', the threshold is still
too high despite Phase 57/58.

Design: wired into continuous.py where the full ScanResult is available.
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

DEFAULT_STATE_PATH = Path("trained_data/signal_funnel_state.jsonl")
DEFAULT_MAX_SCANS = 200

# Stage drop thresholds
KILL_STAGE_NONE = "none"
KILL_STAGE_CONFIDENCE = "confidence"
KILL_STAGE_MOMENTUM = "momentum"
KILL_STAGE_RISK = "risk"
KILL_STAGE_VOTE = "vote"


class SignalFunnelTracker:
    """Track per-scan gate attrition to identify the dominant kill stage."""

    def __init__(
        self,
        max_scans: int = DEFAULT_MAX_SCANS,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        self._max_scans = max_scans
        self._state_path = Path(state_path)
        self._lock = threading.Lock()
        self._history: deque[Dict[str, Any]] = deque(maxlen=max_scans)
        self._load_state()

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record_scan(self, analyses: List[Any]) -> Dict[str, Any]:
        """Record one scan's gate funnel.

        Args:
            analyses: List of PairAnalysis objects or dicts with gate boolean fields:
                - confidence_gate_passed (bool)
                - momentum_gate_passed (bool)
                - risk_gate_passed (bool)
                - is_tradeable (bool)

        Returns:
            Funnel dict for this scan.
        """
        total = len(analyses)
        if total == 0:
            funnel = self._empty_funnel()
            with self._lock:
                self._history.append(funnel)
            return funnel

        def get_flag(a: Any, *names: str) -> bool:
            for name in names:
                v = getattr(a, name, None)
                if v is None:
                    v = a.get(name) if isinstance(a, dict) else None
                if v is not None:
                    return bool(v)
            return False

        passed_conf = sum(1 for a in analyses if get_flag(a, "confidence_gate_passed", "confidence_passed"))
        passed_mom = sum(1 for a in analyses if get_flag(a, "momentum_gate_passed", "momentum_passed"))
        passed_risk = sum(1 for a in analyses if get_flag(a, "risk_gate_passed", "risk_passed"))
        tradeable = sum(1 for a in analyses if get_flag(a, "is_tradeable"))

        funnel = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_evaluated": total,
            "passed_confidence": passed_conf,
            "passed_momentum": passed_mom,
            "passed_risk": passed_risk,
            "is_tradeable": tradeable,
            "passed_confidence_pct": round((passed_conf / total) * 100, 2),
            "passed_momentum_pct": round((passed_mom / max(passed_conf, 1)) * 100, 2),
            "passed_risk_pct": round((passed_risk / max(passed_mom, 1)) * 100, 2),
            "tradeable_pct": round((tradeable / total) * 100, 2),
            "dominant_kill_stage": self._compute_kill_stage(
                total, passed_conf, passed_mom, passed_risk, tradeable
            ),
        }

        with self._lock:
            self._history.append(funnel)

        # Alert when non-confidence gate is the dominant killer with 0 tradeables
        if (
            funnel["dominant_kill_stage"] not in (KILL_STAGE_CONFIDENCE, KILL_STAGE_NONE)
            and tradeable == 0
            and passed_conf > 0
        ):
            logger.warning(
                "SignalFunnelTracker: %d pairs cleared confidence but 0 tradeable — "
                "dominant kill stage: %s",
                passed_conf, funnel["dominant_kill_stage"],
            )

        return funnel

    def _compute_kill_stage(
        self,
        total: int,
        passed_conf: int,
        passed_mom: int,
        passed_risk: int,
        tradeable: int,
    ) -> str:
        """Identify the stage where the most pairs are lost."""
        if total == 0:
            return KILL_STAGE_NONE

        drops = {
            KILL_STAGE_CONFIDENCE: total - passed_conf,
            KILL_STAGE_MOMENTUM: passed_conf - passed_mom,
            KILL_STAGE_RISK: passed_mom - passed_risk,
            KILL_STAGE_VOTE: passed_risk - tradeable,
        }

        max_drop = max(drops.values())
        if max_drop == 0:
            return KILL_STAGE_NONE

        # Pick the earliest stage with the max drop
        for stage in (KILL_STAGE_CONFIDENCE, KILL_STAGE_MOMENTUM, KILL_STAGE_RISK, KILL_STAGE_VOTE):
            if drops[stage] == max_drop:
                return stage

        return KILL_STAGE_NONE

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def get_funnel_summary(self) -> Dict[str, Any]:
        """Aggregate funnel stats over all recorded scans."""
        with self._lock:
            history = list(self._history)

        if not history:
            return {
                "scan_count": 0,
                "total_evaluated": 0,
                "passed_confidence_pct": 0.0,
                "passed_momentum_pct": 0.0,
                "passed_risk_pct": 0.0,
                "tradeable_pct": 0.0,
                "dominant_kill_stage": KILL_STAGE_NONE,
            }

        total_evals = sum(h["total_evaluated"] for h in history)
        total_conf = sum(h["passed_confidence"] for h in history)
        total_mom = sum(h["passed_momentum"] for h in history)
        total_risk = sum(h["passed_risk"] for h in history)
        total_trade = sum(h["is_tradeable"] for h in history)

        return {
            "scan_count": len(history),
            "total_evaluated": total_evals,
            "passed_confidence_pct": round((total_conf / max(total_evals, 1)) * 100, 2),
            "passed_momentum_pct": round((total_mom / max(total_conf, 1)) * 100, 2),
            "passed_risk_pct": round((total_risk / max(total_mom, 1)) * 100, 2),
            "tradeable_pct": round((total_trade / max(total_evals, 1)) * 100, 2),
            "dominant_kill_stage": self._compute_kill_stage(
                total_evals, total_conf, total_mom, total_risk, total_trade
            ),
        }

    def get_dominant_kill_stage(self) -> str:
        return self.get_funnel_summary()["dominant_kill_stage"]

    def get_recent_funnels(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            history = list(self._history)
        return history[-n:]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _empty_funnel(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_evaluated": 0,
            "passed_confidence": 0,
            "passed_momentum": 0,
            "passed_risk": 0,
            "is_tradeable": 0,
            "passed_confidence_pct": 0.0,
            "passed_momentum_pct": 0.0,
            "passed_risk_pct": 0.0,
            "tradeable_pct": 0.0,
            "dominant_kill_stage": KILL_STAGE_NONE,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_state(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path else self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._history)
        try:
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            os.rename(tmp, str(target))
        except Exception as exc:
            logger.error("SignalFunnelTracker.save_state failed: %s", exc)
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
            start = max(0, len(loaded) - self._max_scans)
            with self._lock:
                self._history = deque(loaded[start:], maxlen=self._max_scans)
        except Exception as exc:
            logger.error("SignalFunnelTracker._load_state failed: %s", exc)
