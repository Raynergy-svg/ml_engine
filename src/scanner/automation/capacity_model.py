"""Capacity model and adaptive rearm timer for the supervision-first runtime.

CapacityModel determines WHEN to scan and WHAT pairs to include.
RearmTimer replaces the fixed 5-minute interval with adaptive timing.
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CapacityModel:
    """Determines scan eligibility based on portfolio state."""

    target_open: int = 3
    max_open: int = 5
    scan_trigger_threshold: int = 2  # Scan when open < this

    def should_scan(
        self,
        open_count: int,
        has_cooldown_active: bool = False,
        session_active: bool = True,
    ) -> bool:
        """Return True if the bot should trigger a scan."""
        if not session_active:
            return False
        if open_count >= self.max_open:
            return False
        if open_count >= self.scan_trigger_threshold and has_cooldown_active:
            return False  # Near threshold + cooldowns active = wait
        return open_count < self.scan_trigger_threshold

    def get_scannable_pairs(
        self,
        all_pairs: List[str],
        open_pairs: Set[str],
        cooldown_pairs: Set[str],
        blocked_pairs: Set[str],
        correlated_pairs: Set[str],
    ) -> List[str]:
        """Return only pairs that can actually be traded right now."""
        excluded = open_pairs | cooldown_pairs | blocked_pairs | correlated_pairs
        scannable = [p for p in all_pairs if p not in excluded]
        logger.debug(
            "capacity_model.scannable_pairs",
            total=len(all_pairs),
            excluded=len(excluded),
            scannable=len(scannable),
        )
        return scannable

    def available_slots(self, open_count: int) -> int:
        """How many more trades can be opened."""
        return max(0, self.max_open - open_count)


class RearmTimer:
    """Adaptive scan interval — replaces fixed 5-minute sleep.

    Good results → scan more often. Bad results → back off.
    3+ losses/hour → sit out at max interval.
    """

    def __init__(
        self,
        base_interval: float = 300.0,
        min_interval: float = 120.0,
        max_interval: float = 1800.0,
    ):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.current_interval = base_interval
        self._last_scan_time: float = 0.0

    def should_rearm(self) -> bool:
        """True if enough time has passed since last scan."""
        if self._last_scan_time == 0.0:
            return True  # Never scanned
        elapsed = time.time() - self._last_scan_time
        return elapsed >= self.current_interval

    def record_scan(self) -> None:
        """Mark that a scan just happened."""
        self._last_scan_time = time.time()

    def record_outcome(self, was_profitable: bool) -> None:
        """Adjust interval based on trade outcome."""
        if was_profitable:
            self.current_interval = max(
                self.min_interval,
                self.current_interval * 0.85,
            )
        else:
            self.current_interval = min(
                self.max_interval,
                self.current_interval * 1.3,
            )
        logger.debug(
            "rearm_timer.adjusted",
            interval=round(self.current_interval, 1),
            profitable=was_profitable,
        )

    def record_consecutive_losses(self, count: int) -> None:
        """3+ losses → sit out at max interval."""
        if count >= 3:
            self.current_interval = self.max_interval
            logger.warning(
                "rearm_timer.sitout",
                consecutive_losses=count,
                interval=self.max_interval,
            )

    def reset(self) -> None:
        """Reset to base interval."""
        self.current_interval = self.base_interval
        self._last_scan_time = 0.0
