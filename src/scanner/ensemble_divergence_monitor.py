"""Phase 59 US-365: EnsembleDivergenceMonitor.

Tracks disagreement between TCN and Ridge confidence predictions.
When divergence is high (> 20pp), the ensemble_disagreement multiplier
(Phase 56 root cause) suppresses confidence even before Phase 57 fixes kick in.

This module answers: "Is the ensemble structurally disagreeing, or is this
a transient disagreement that resolved after Phase 57's penalty fixes?"

High divergence rate (> 30% of snapshots) → models are structurally divergent
→ ensemble needs retraining / feature drift investigation.

Design: wired into engine.py where both TCN and Ridge confidences are computed.
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

DEFAULT_STATE_PATH = Path("trained_data/ensemble_divergence_state.jsonl")
DEFAULT_MAX_SIZE = 500
DEFAULT_HIGH_DIVERGENCE_THRESHOLD = 20.0   # pp on 0-100 scale
DEFAULT_HIGH_RATE_THRESHOLD = 0.30          # 30% of snapshots

REC_HIGH_DISAGREEMENT = "HIGH_ENSEMBLE_DISAGREEMENT"
REC_OK = "OK"


class EnsembleDivergenceMonitor:
    """Monitor TCN vs Ridge confidence divergence.

    Usage:
        monitor.record(pair, tcn_confidence=72.0, ridge_confidence=55.0, scan_id="s1")
        if monitor.is_high_divergence():
            logger.warning("Ensemble divergence elevated")
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        high_divergence_threshold: float = DEFAULT_HIGH_DIVERGENCE_THRESHOLD,
        high_rate_threshold: float = DEFAULT_HIGH_RATE_THRESHOLD,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        self._max_size = max_size
        self._high_div_threshold = high_divergence_threshold
        self._high_rate_threshold = high_rate_threshold
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
        tcn_confidence: float,
        ridge_confidence: float,
        scan_id: str = "",
    ) -> float:
        """Record a TCN vs Ridge divergence snapshot.

        Returns:
            Divergence value (abs(tcn - ridge) on 0-100 scale).
        """
        divergence = abs(float(tcn_confidence) - float(ridge_confidence))
        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "tcn_confidence": round(float(tcn_confidence), 4),
            "ridge_confidence": round(float(ridge_confidence), 4),
            "divergence": round(divergence, 4),
            "scan_id": scan_id,
            "high": divergence > self._high_div_threshold,
        }
        with self._lock:
            self._records.append(snap)
        logger.debug(
            "EnsembleDivergenceMonitor: pair=%s tcn=%.2f ridge=%.2f div=%.2f",
            pair, tcn_confidence, ridge_confidence, divergence,
        )
        return divergence

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def get_mean_divergence(self) -> float:
        """Return mean divergence across all recorded snapshots."""
        with self._lock:
            records = list(self._records)
        if not records:
            return 0.0
        return sum(r["divergence"] for r in records) / len(records)

    def is_high_divergence(self, threshold: Optional[float] = None) -> bool:
        """Return True when mean divergence exceeds threshold."""
        thresh = threshold if threshold is not None else self._high_div_threshold
        return self.get_mean_divergence() > thresh

    def get_divergence_summary(self) -> Dict[str, Any]:
        """Return aggregate divergence stats with recommendation."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "count": 0,
                "mean_divergence": 0.0,
                "max_divergence": 0.0,
                "high_divergence_rate_pct": 0.0,
                "recommendation": REC_OK,
            }

        divergences = [r["divergence"] for r in records]
        high_count = sum(1 for r in records if r.get("high", False))
        high_rate = high_count / len(records)

        recommendation = (
            REC_HIGH_DISAGREEMENT if high_rate > self._high_rate_threshold else REC_OK
        )

        return {
            "count": len(records),
            "mean_divergence": round(sum(divergences) / len(divergences), 4),
            "max_divergence": round(max(divergences), 4),
            "high_divergence_rate_pct": round(high_rate * 100, 2),
            "recommendation": recommendation,
        }

    def get_per_pair_summary(self, pair: str) -> Dict[str, Any]:
        """Return divergence stats for a specific pair."""
        with self._lock:
            records = [r for r in self._records if r.get("pair") == pair]

        if not records:
            return {"count": 0, "pair": pair, "mean_divergence": 0.0, "max_divergence": 0.0}

        divergences = [r["divergence"] for r in records]
        return {
            "count": len(records),
            "pair": pair,
            "mean_divergence": round(sum(divergences) / len(divergences), 4),
            "max_divergence": round(max(divergences), 4),
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
            logger.error("EnsembleDivergenceMonitor.save_state failed: %s", exc)
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
            logger.error("EnsembleDivergenceMonitor._load_state failed: %s", exc)
