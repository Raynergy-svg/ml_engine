"""Signal timing and scan frequency optimizer.

US-114: Track when signals are generated vs when they would have been
optimal, and adjust scan frequency accordingly. Detects whether signals
arrive too early or too late, and recommends scan frequency adjustments
per pair/session.

Approach:
1. Record signal timing: signal_time, entry_time, peak_time
2. Compute timing gaps: signal_to_entry, signal_to_peak
3. Score timing quality: closer to optimal = higher score
4. Recommend scan intervals per pair/session based on patterns
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "signal_timing_log.json"

# Timing parameters
ROLLING_WINDOW = 50
MIN_SAMPLES = 10

# Scan interval bounds (seconds)
DEFAULT_SCAN_INTERVAL = 300  # 5 minutes
MIN_SCAN_INTERVAL = 60       # 1 minute
MAX_SCAN_INTERVAL = 900      # 15 minutes

# Timing quality scoring
OPTIMAL_SIGNAL_TO_ENTRY_SEC = 120   # 2 minutes = ideal lead time
OPTIMAL_SIGNAL_TO_PEAK_SEC = 600    # 10 minutes = ideal signal-to-peak
MAX_ACCEPTABLE_LAG_SEC = 1800       # 30 minutes = too late

# Session definitions (UTC hours).
# Sessions overlap intentionally (markets trade concurrently).
# Resolution: first match wins in insertion order. During overlaps:
#   TOKYO wins hours 0-6 (over SYDNEY), LONDON wins hour 7 (over TOKYO),
#   LONDON wins hours 13-15 (over NEW_YORK), NEW_YORK wins hours 16-21.
SESSIONS = {
    "TOKYO": (0, 8),
    "LONDON": (7, 16),
    "NEW_YORK": (13, 22),
    "SYDNEY": (21, 6),  # Wraps midnight
}


class SignalTimingOptimizer:
    """Tracks signal timing and optimizes scan frequency.

    Bridges signal generation with trade execution timing. Reveals
    whether the scanner fires too early (signal decays before entry)
    or too late (move already happened), and adjusts scan intervals.

    Usage:
        optimizer = SignalTimingOptimizer()
        optimizer.record_signal("EUR_USD", signal_time=1000, entry_time=1120, peak_time=1600)
        report = optimizer.get_timing_report("EUR_USD")
        interval = optimizer.recommend_scan_interval("EUR_USD", "LONDON")
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        # Timing data: {pair: [records]}
        self._signals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Session-specific timing: {pair_session: [records]}
        self._session_signals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Recommendation log
        self._recommendations: List[Dict[str, Any]] = []
        self._total_signals: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_signal(
        self,
        pair: str,
        signal_time: float,
        entry_time: float,
        peak_time: float,
    ) -> None:
        """Record signal timing for a trade.

        Args:
            pair: Currency pair.
            signal_time: Unix timestamp when signal was generated.
            entry_time: Unix timestamp when trade was entered.
            peak_time: Unix timestamp when price hit the peak (max favorable).
        """
        signal_to_entry = max(0.0, entry_time - signal_time)
        signal_to_peak = max(0.0, peak_time - signal_time)
        entry_to_peak = max(0.0, peak_time - entry_time)

        record = {
            "signal_to_entry": signal_to_entry,
            "signal_to_peak": signal_to_peak,
            "entry_to_peak": entry_to_peak,
            "timing_score": self._compute_timing_score(
                signal_to_entry, signal_to_peak
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._signals[pair].append(record)
        if len(self._signals[pair]) > ROLLING_WINDOW * 2:
            self._signals[pair] = self._signals[pair][-ROLLING_WINDOW:]

        # Also track by session
        session = self._detect_session(signal_time)
        session_key = f"{pair}_{session}"
        self._session_signals[session_key].append(record)
        if len(self._session_signals[session_key]) > ROLLING_WINDOW * 2:
            self._session_signals[session_key] = (
                self._session_signals[session_key][-ROLLING_WINDOW:]
            )

        self._total_signals += 1

    def get_timing_report(self, pair: str) -> Dict[str, Any]:
        """Get timing report for a pair.

        Returns:
            Dict with avg timing gaps, timing score, and diagnosis.
        """
        signals = self._signals.get(pair, [])
        if len(signals) < MIN_SAMPLES:
            return {
                "pair": pair,
                "n_signals": len(signals),
                "sufficient_data": False,
            }

        recent = signals[-ROLLING_WINDOW:]

        avg_sig_to_entry = float(np.mean([r["signal_to_entry"] for r in recent]))
        avg_sig_to_peak = float(np.mean([r["signal_to_peak"] for r in recent]))
        avg_entry_to_peak = float(np.mean([r["entry_to_peak"] for r in recent]))
        avg_score = float(np.mean([r["timing_score"] for r in recent]))

        # Diagnose timing
        if avg_sig_to_entry > MAX_ACCEPTABLE_LAG_SEC:
            diagnosis = "too_slow"
        elif avg_sig_to_entry < 30:
            diagnosis = "too_early"
        elif avg_entry_to_peak < 60:
            diagnosis = "late_entry"
        else:
            diagnosis = "good"

        return {
            "pair": pair,
            "avg_signal_to_entry": round(avg_sig_to_entry, 1),
            "avg_signal_to_peak": round(avg_sig_to_peak, 1),
            "avg_entry_to_peak": round(avg_entry_to_peak, 1),
            "timing_score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "n_signals": len(recent),
            "sufficient_data": True,
        }

    def recommend_scan_interval(
        self, pair: str, session: str = "LONDON"
    ) -> int:
        """Recommend optimal scan interval for a pair/session.

        Args:
            pair: Currency pair.
            session: Trading session (TOKYO/LONDON/NEW_YORK/SYDNEY).

        Returns:
            Recommended scan interval in seconds.
        """
        session_key = f"{pair}_{session}"
        signals = self._session_signals.get(session_key, [])

        if len(signals) < MIN_SAMPLES:
            # Fall back to pair-level data
            signals = self._signals.get(pair, [])
            if len(signals) < MIN_SAMPLES:
                return DEFAULT_SCAN_INTERVAL

        recent = signals[-ROLLING_WINDOW:]

        avg_sig_to_entry = float(np.mean([r["signal_to_entry"] for r in recent]))
        avg_score = float(np.mean([r["timing_score"] for r in recent]))

        # If signals are too early → scan less frequently
        # If signals are too late → scan more frequently
        if avg_sig_to_entry > OPTIMAL_SIGNAL_TO_ENTRY_SEC * 3:
            # Signals way too early — slow down scanning
            interval = int(DEFAULT_SCAN_INTERVAL * 1.5)
        elif avg_sig_to_entry > OPTIMAL_SIGNAL_TO_ENTRY_SEC * 1.5:
            # Slightly early
            interval = int(DEFAULT_SCAN_INTERVAL * 1.2)
        elif avg_score < 0.30:
            # Poor timing quality — scan faster to catch better entries
            interval = int(DEFAULT_SCAN_INTERVAL * 0.6)
        elif avg_score < 0.50:
            # Mediocre timing — scan slightly faster
            interval = int(DEFAULT_SCAN_INTERVAL * 0.8)
        else:
            interval = DEFAULT_SCAN_INTERVAL

        interval = max(MIN_SCAN_INTERVAL, min(interval, MAX_SCAN_INTERVAL))

        # Log recommendation
        self._recommendations.append({
            "pair": pair,
            "session": session,
            "interval": interval,
            "avg_sig_to_entry": round(avg_sig_to_entry, 1),
            "timing_score": round(avg_score, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._recommendations) > 200:
            self._recommendations = self._recommendations[-100:]

        return interval

    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status for observability."""
        return {
            "total_signals": self._total_signals,
            "pairs_tracked": list(self._signals.keys()),
            "pair_counts": {
                pair: len(signals) for pair, signals in self._signals.items()
            },
            "recent_recommendations": self._recommendations[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_timing_score(
        self, signal_to_entry: float, signal_to_peak: float
    ) -> float:
        """Compute timing quality score [0.0, 1.0].

        Perfect timing: signal arrives OPTIMAL_SIGNAL_TO_ENTRY_SEC before entry,
        and entry arrives well before peak.
        """
        # Score for signal-to-entry gap
        entry_diff = abs(signal_to_entry - OPTIMAL_SIGNAL_TO_ENTRY_SEC)
        entry_score = max(0.0, 1.0 - entry_diff / OPTIMAL_SIGNAL_TO_ENTRY_SEC)

        # Score for signal-to-peak gap
        peak_diff = abs(signal_to_peak - OPTIMAL_SIGNAL_TO_PEAK_SEC)
        peak_score = max(0.0, 1.0 - peak_diff / OPTIMAL_SIGNAL_TO_PEAK_SEC)

        # Penalty for very late signals
        if signal_to_entry > MAX_ACCEPTABLE_LAG_SEC:
            late_penalty = 0.5
        else:
            late_penalty = 0.0

        score = 0.6 * entry_score + 0.4 * peak_score - late_penalty
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def _detect_session(self, timestamp: float) -> str:
        """Detect trading session from Unix timestamp."""
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            hour = dt.hour

            for session, (start, end) in SESSIONS.items():
                if start <= end:
                    if start <= hour < end:
                        return session
                else:  # Wraps midnight (e.g., SYDNEY 21-6)
                    if hour >= start or hour < end:
                        return session

            return "OFF_HOURS"
        except Exception:
            return "UNKNOWN"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist timing data."""
        try:
            signals_plain = {
                pair: records[-ROLLING_WINDOW:]
                for pair, records in self._signals.items()
            }
            session_plain = {
                key: records[-ROLLING_WINDOW:]
                for key, records in self._session_signals.items()
            }

            data = {
                "version": 1,
                "total_signals": self._total_signals,
                "signals": signals_plain,
                "session_signals": session_plain,
                "recommendations": self._recommendations[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"SignalTimingOptimizer: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                self._total_signals = data.get("total_signals", 0)
                self._recommendations = data.get("recommendations", [])
                _required_keys = ("signal_to_entry", "signal_to_peak", "timing_score")
                raw_signals = data.get("signals", {})
                for pair, records in raw_signals.items():
                    if not isinstance(records, list):
                        continue
                    valid = [
                        r for r in records
                        if isinstance(r, dict)
                        and all(k in r for k in _required_keys)
                    ]
                    self._signals[pair] = valid
                raw_session = data.get("session_signals", {})
                for key, records in raw_session.items():
                    if not isinstance(records, list):
                        continue
                    valid = [
                        r for r in records
                        if isinstance(r, dict)
                        and all(k in r for k in _required_keys)
                    ]
                    self._session_signals[key] = valid
        except Exception as e:
            logger.warning(f"SignalTimingOptimizer: Failed to load: {e}")
