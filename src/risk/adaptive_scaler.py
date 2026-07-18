"""Adaptive Risk Scaler — Scales position sizing based on recent trading performance.

After consecutive losses, reduces position size to protect capital (drawdown defense).
After proven winning streaks, cautiously increases size back toward baseline.

Scale factor decays toward 1.0 over time when no trades are made, preventing
stale scale factors from persisting indefinitely.

Integration: Multiply base risk percentage by get_scale_factor() before position sizing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.scanner.automation.safe_json import safe_json_write

logger = logging.getLogger(__name__)


class AdaptiveRiskScaler:
    """Scales position size based on recent win/loss streaks.

    Rules:
      - 3 consecutive losses: scale down by 30% (factor = 0.70)
      - 5 consecutive losses: scale down by 50% (factor = 0.50, floor)
      - 5 consecutive wins (with accuracy > 60%): scale up by 15% (factor = 1.15, capped at 1.0)
      - No trades for 48h: decay toward 1.0 (baseline)
      - Scale factor bounded: [0.50, 1.15]

    Usage:
        scaler = AdaptiveRiskScaler()
        scaler.record_outcome(won=False)
        factor = scaler.get_scale_factor()
        position_risk = base_risk * factor
    """

    MIN_SCALE = 0.50
    MAX_SCALE = 1.15
    LOSS_SCALE_3 = 0.70  # After 3 consecutive losses
    LOSS_SCALE_5 = 0.50  # After 5 consecutive losses
    WIN_BOOST = 1.15     # Ceiling reachable via win streaks (== MAX_SCALE)
    WIN_STEP = 1.05      # Per-win multiplicative step once streak >= 5
    DECAY_HOURS = 48     # Hours to decay fully back to 1.0

    def __init__(
        self,
        state_path: str = "trained_data/risk_scaler_state.json",
    ):
        self.state_path = Path(state_path)
        self._scale_factor: float = 1.0
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0
        self._last_trade_time: Optional[str] = None
        self._total_trades: int = 0
        self._load_state()

    def record_outcome(self, won: bool) -> float:
        """Record a trade outcome and update the scale factor.

        Args:
            won: True if the trade was profitable, False otherwise

        Returns:
            Updated scale factor
        """
        self._total_trades += 1
        self._last_trade_time = datetime.now(timezone.utc).isoformat()

        if won:
            self._consecutive_wins += 1
            self._consecutive_losses = 0

            # Scale up after 5 consecutive wins. 2026-07-18 audit cleanup: the
            # per-win step is an explicit constant now (WIN_STEP); WIN_BOOST=1.15
            # was advertised but never referenced — the effective ceiling is
            # MAX_SCALE either way.
            if self._consecutive_wins >= 5:
                self._scale_factor = min(self.MAX_SCALE, self._scale_factor * self.WIN_STEP)
                logger.info(
                    f"Win streak ({self._consecutive_wins}): "
                    f"scale factor → {self._scale_factor:.2f}"
                )
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

            if self._consecutive_losses >= 5:
                self._scale_factor = self.LOSS_SCALE_5
            elif self._consecutive_losses >= 3:
                self._scale_factor = min(self._scale_factor, self.LOSS_SCALE_3)

            logger.info(
                f"Loss streak ({self._consecutive_losses}): "
                f"scale factor → {self._scale_factor:.2f}"
            )

        # Enforce bounds
        self._scale_factor = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale_factor))

        self._save_state()
        self._log_change()

        return self._scale_factor

    def get_scale_factor(self) -> float:
        """Get current scale factor, applying time decay if needed.

        Returns:
            Scale factor between MIN_SCALE and MAX_SCALE
        """
        self._apply_time_decay()
        return self._scale_factor

    def _apply_time_decay(self) -> None:
        """Decay scale factor toward 1.0 based on time since last trade."""
        if self._last_trade_time is None:
            return

        try:
            last = datetime.fromisoformat(self._last_trade_time)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            hours_since = (now - last).total_seconds() / 3600

            if hours_since <= 0:
                return

            # Linear decay toward 1.0 over DECAY_HOURS
            decay_fraction = min(1.0, hours_since / self.DECAY_HOURS)
            self._scale_factor = self._scale_factor + (1.0 - self._scale_factor) * decay_fraction

            # Clamp
            self._scale_factor = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale_factor))

        except Exception as e:
            logger.debug(f"Time decay error: {e}")

    def get_stats(self) -> dict:
        """Get current scaler statistics."""
        return {
            "scale_factor": round(self._scale_factor, 3),
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "total_trades": self._total_trades,
            "last_trade_time": self._last_trade_time,
        }

    def _load_state(self) -> None:
        """Load scaler state from disk."""
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                self._scale_factor = float(data.get("scale_factor", 1.0))
                self._consecutive_wins = int(data.get("consecutive_wins", 0))
                self._consecutive_losses = int(data.get("consecutive_losses", 0))
                self._last_trade_time = data.get("last_trade_time")
                self._total_trades = int(data.get("total_trades", 0))
        except Exception as e:
            logger.debug(f"Scaler state load failed: {e}")

    def _save_state(self) -> None:
        """Persist scaler state to disk."""
        data = {
            "scale_factor": round(self._scale_factor, 4),
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "last_trade_time": self._last_trade_time,
            "total_trades": self._total_trades,
        }
        # H-2: atomic write — prevents loss of drawdown state on crash
        if not safe_json_write(self.state_path, data):
            logger.error("Scaler state save failed")

    def _log_change(self) -> None:
        """Log scale factor change to observations."""
        try:
            from src.scanner.automation.observation_log import ObservationLog

            obs = ObservationLog()
            obs.log_observation(
                pair="_RISK",
                category="risk_scaling",
                description=(
                    f"Scale factor: {self._scale_factor:.2f} "
                    f"(W{self._consecutive_wins}/L{self._consecutive_losses})"
                ),
                metadata=self.get_stats(),
            )
        except Exception:
            pass  # Non-critical
