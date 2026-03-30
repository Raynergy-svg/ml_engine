"""Dynamic drawdown threshold manager.

Adjusts trailing stop lock/breakeven percentages based on current drawdown
relative to maximum allowed. High drawdown = lock profits earlier (protective).
Low drawdown = allow more room to run (aggressive).

Usage:
    dd_mgr = DynamicDrawdownManager()
    thresholds = dd_mgr.get_dynamic_thresholds(current_dd=0.08, max_dd=0.15)
    # thresholds = {"breakeven_pct": 0.40, "lock_pct": 0.62}
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class DrawdownThresholds:
    """Dynamic trailing stop thresholds."""

    breakeven_pct: float  # Move SL to breakeven when this % of TP is reached
    lock_pct: float  # Lock profits when this % of TP is reached
    severity: float  # 0.0 = no drawdown, 1.0 = at max drawdown
    mode: str  # "normal", "protective", "aggressive"


class DynamicDrawdownManager:
    """Computes dynamic trailing stop thresholds from equity curve health.

    Drawdown severity = current_drawdown / max_drawdown_pct:
    - severity < 0.3: "aggressive" mode (wider trailing stops)
    - severity 0.3-0.6: "normal" mode (default thresholds)
    - severity > 0.6: "protective" mode (tighter trailing stops)
    """

    def __init__(
        self,
        # Normal defaults (match execution.py defaults)
        normal_breakeven_pct: float = 0.50,
        normal_lock_pct: float = 0.75,
        # Protective (high drawdown): tighten thresholds
        protective_breakeven_pct: float = 0.30,
        protective_lock_pct: float = 0.50,
        # Aggressive (low drawdown): widen thresholds
        aggressive_breakeven_pct: float = 0.60,
        aggressive_lock_pct: float = 0.85,
        # Severity boundaries
        protective_threshold: float = 0.6,
        aggressive_threshold: float = 0.3,
    ):
        self.normal_breakeven = normal_breakeven_pct
        self.normal_lock = normal_lock_pct
        self.protective_breakeven = protective_breakeven_pct
        self.protective_lock = protective_lock_pct
        self.aggressive_breakeven = aggressive_breakeven_pct
        self.aggressive_lock = aggressive_lock_pct
        self.protective_threshold = protective_threshold
        self.aggressive_threshold = aggressive_threshold

    def get_dynamic_thresholds(
        self, current_dd: float, max_dd: float
    ) -> DrawdownThresholds:
        """Compute dynamic thresholds based on drawdown severity.

        Args:
            current_dd: Current drawdown as fraction (e.g., 0.08 = 8%)
            max_dd: Maximum allowed drawdown as fraction (e.g., 0.15 = 15%)

        Returns:
            DrawdownThresholds with adjusted breakeven and lock percentages
        """
        if max_dd <= 0:
            return DrawdownThresholds(
                breakeven_pct=self.normal_breakeven,
                lock_pct=self.normal_lock,
                severity=0.0,
                mode="normal",
            )

        severity = min(1.0, max(0.0, abs(current_dd) / max(max_dd, 1e-8)))

        if severity > self.protective_threshold:
            # High drawdown — protect profits aggressively
            # Interpolate between normal and protective based on how far past threshold
            t = min(1.0, (severity - self.protective_threshold) / (1.0 - self.protective_threshold))
            breakeven = self.normal_breakeven + t * (self.protective_breakeven - self.normal_breakeven)
            lock = self.normal_lock + t * (self.protective_lock - self.normal_lock)
            mode = "protective"

            logger.info(
                f"Dynamic drawdown PROTECTIVE: severity={severity:.2f}, "
                f"breakeven={breakeven:.2f}, lock={lock:.2f} "
                f"(dd={current_dd:.1%}/{max_dd:.1%})"
            )

        elif severity < self.aggressive_threshold:
            # Low drawdown — let trades run
            t = 1.0 - (severity / self.aggressive_threshold) if self.aggressive_threshold > 0 else 1.0
            breakeven = self.normal_breakeven + t * (self.aggressive_breakeven - self.normal_breakeven)
            lock = self.normal_lock + t * (self.aggressive_lock - self.normal_lock)
            mode = "aggressive"

            logger.debug(
                f"Dynamic drawdown AGGRESSIVE: severity={severity:.2f}, "
                f"breakeven={breakeven:.2f}, lock={lock:.2f}"
            )

        else:
            # Normal range
            breakeven = self.normal_breakeven
            lock = self.normal_lock
            mode = "normal"

        return DrawdownThresholds(
            breakeven_pct=round(breakeven, 4),
            lock_pct=round(lock, 4),
            severity=round(severity, 4),
            mode=mode,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return manager configuration."""
        return {
            "normal": {"breakeven": self.normal_breakeven, "lock": self.normal_lock},
            "protective": {"breakeven": self.protective_breakeven, "lock": self.protective_lock},
            "aggressive": {"breakeven": self.aggressive_breakeven, "lock": self.aggressive_lock},
            "thresholds": {
                "protective": self.protective_threshold,
                "aggressive": self.aggressive_threshold,
            },
        }
