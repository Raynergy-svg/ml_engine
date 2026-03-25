"""Derived macro stress indicator for position sizing.

US-066: Computes a macro stress score from FX-derived signals without
external APIs. Uses USD pair volatility, regime persistence, and
spread widening trends to detect market stress periods.

Usage:
    detector = MacroStressDetector()
    detector.update(avg_usd_volatility=0.012, regime="HIGH", avg_spread_ratio=1.8)
    modifier = detector.get_stress_modifier()  # 0.70 during stress
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STRESS_PATH = _PROJECT_ROOT / "trained_data" / "macro_stress.json"

# Stress thresholds
VOLATILITY_STRESS_THRESHOLD = 0.015   # Daily USD vol > 1.5% = stressed
SPREAD_STRESS_THRESHOLD = 2.0         # Spread > 2x normal = stressed
REGIME_STRESS_PERSISTENCE = 3         # 3+ consecutive HIGH/EXTREME scans = stressed

# Modifiers
MILD_STRESS_MODIFIER = 0.85     # Mild stress: reduce 15%
SEVERE_STRESS_MODIFIER = 0.60   # Severe stress: reduce 40%
EXTREME_STRESS_MODIFIER = 0.50  # Extreme stress: reduce 50%


class MacroStressDetector:
    """Detects macro stress from FX-derived signals.

    Computes a composite stress score from three signals:
    1. USD pair average volatility (proxy for DXY movement)
    2. Regime persistence (how long in HIGH/EXTREME)
    3. Spread widening trend (liquidity drying up)

    No external API required — derived from OANDA data already available.
    """

    def __init__(
        self,
        window_size: int = 20,
        persistence_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.persistence_path = persistence_path or STRESS_PATH

        # Rolling windows for each signal
        self._volatility_window: deque = deque(maxlen=window_size)
        self._spread_window: deque = deque(maxlen=window_size)
        self._regime_history: deque = deque(maxlen=window_size)
        self._high_regime_streak: int = 0
        self._stress_score: float = 0.0
        self._update_count: int = 0

        self._load()

    def _load(self) -> None:
        """Load persisted stress state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            for v in data.get("volatility_window", []):
                self._volatility_window.append(float(v))
            for s in data.get("spread_window", []):
                self._spread_window.append(float(s))
            for r in data.get("regime_history", []):
                self._regime_history.append(str(r))
            self._high_regime_streak = int(data.get("high_regime_streak", 0))
            self._stress_score = float(data.get("stress_score", 0.0))
            self._update_count = int(data.get("update_count", 0))
        except Exception as e:
            logger.debug(f"Failed to load macro stress state: {e}")

    def _save(self) -> None:
        """Persist stress state."""
        data = {
            "volatility_window": list(self._volatility_window),
            "spread_window": list(self._spread_window),
            "regime_history": list(self._regime_history),
            "high_regime_streak": self._high_regime_streak,
            "stress_score": round(self._stress_score, 4),
            "update_count": self._update_count,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                self.persistence_path.write_text(
                    json.dumps(data, indent=2), encoding="utf-8"
                )
        except Exception as e:
            logger.debug(f"Failed to save macro stress state: {e}")

    def update(
        self,
        avg_usd_volatility: float = 0.0,
        regime: str = "NORMAL",
        avg_spread_ratio: float = 1.0,
    ) -> None:
        """Update stress signals with latest scan data.

        Args:
            avg_usd_volatility: Average volatility of USD pairs (fraction, e.g., 0.012 = 1.2%)
            regime: Current volatility regime (LOW/NORMAL/HIGH/EXTREME)
            avg_spread_ratio: Average spread as ratio of normal (e.g., 1.5 = 50% wider)
        """
        self._volatility_window.append(avg_usd_volatility)
        self._spread_window.append(avg_spread_ratio)
        self._regime_history.append(regime.upper())

        # Track HIGH/EXTREME streak
        if regime.upper() in ("HIGH", "EXTREME"):
            self._high_regime_streak += 1
        else:
            self._high_regime_streak = 0

        # Compute composite stress score
        self._stress_score = self._compute_stress_score()
        self._update_count += 1

        # Persist every 5 updates
        if self._update_count % 5 == 0:
            self._save()

        # Log state transitions
        if self._stress_score > 0.7:
            logger.warning(
                f"Macro stress SEVERE: score={self._stress_score:.2f} "
                f"(vol={avg_usd_volatility:.4f}, regime={regime}, spread_ratio={avg_spread_ratio:.1f})"
            )
            # Log to observation log
            try:
                from src.scanner.automation.observation_log import ObservationLog
                ObservationLog().log_observation(
                    pair="ALL",
                    category="macro_stress",
                    description=f"Macro stress score={self._stress_score:.2f}",
                    metadata={
                        "stress_score": round(self._stress_score, 4),
                        "avg_usd_volatility": round(avg_usd_volatility, 6),
                        "regime": regime,
                        "avg_spread_ratio": round(avg_spread_ratio, 2),
                        "high_regime_streak": self._high_regime_streak,
                    },
                )
            except Exception:
                pass

    def _compute_stress_score(self) -> float:
        """Compute composite stress score from 0.0 (calm) to 1.0 (severe).

        Three signals, equally weighted:
        1. Volatility signal: avg USD vol relative to threshold
        2. Regime persistence: consecutive HIGH/EXTREME scans
        3. Spread signal: avg spread ratio relative to threshold
        """
        # Signal 1: Volatility
        vol_signal = 0.0
        if self._volatility_window:
            avg_vol = sum(self._volatility_window) / len(self._volatility_window)
            vol_signal = min(1.0, avg_vol / VOLATILITY_STRESS_THRESHOLD)

        # Signal 2: Regime persistence
        regime_signal = min(1.0, self._high_regime_streak / REGIME_STRESS_PERSISTENCE)

        # Signal 3: Spread widening
        spread_signal = 0.0
        if self._spread_window:
            avg_spread = sum(self._spread_window) / len(self._spread_window)
            spread_signal = min(1.0, max(0.0, (avg_spread - 1.0) / (SPREAD_STRESS_THRESHOLD - 1.0)))

        # Composite: equal weight
        return round((vol_signal + regime_signal + spread_signal) / 3.0, 4)

    def get_stress_modifier(self) -> float:
        """Get position sizing modifier based on stress level.

        Returns:
            Float 0.5-1.0. Lower = more stress = reduce position size.
        """
        if self._stress_score > 0.7:
            return EXTREME_STRESS_MODIFIER
        if self._stress_score > 0.5:
            return SEVERE_STRESS_MODIFIER
        if self._stress_score > 0.3:
            return MILD_STRESS_MODIFIER
        return 1.0

    def get_status(self) -> Dict[str, Any]:
        """Get detector status for observability."""
        return {
            "stress_score": round(self._stress_score, 4),
            "stress_modifier": self.get_stress_modifier(),
            "high_regime_streak": self._high_regime_streak,
            "update_count": self._update_count,
            "volatility_avg": (
                round(sum(self._volatility_window) / len(self._volatility_window), 6)
                if self._volatility_window else 0.0
            ),
            "spread_avg": (
                round(sum(self._spread_window) / len(self._spread_window), 2)
                if self._spread_window else 1.0
            ),
        }
