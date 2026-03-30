"""Futures margin validation for risk management.

This module implements margin checking for futures contracts to ensure
that position sizing respects account margin constraints.

Key features:
- Track total margin used across open positions
- Check if new positions exceed margin limits
- Support configurable max margin percentage
- Per-instrument margin requirements (stored in Instrument.margin_requirement)
- Release margin when positions close
- Get margin utilization metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.brokers.instrument import Instrument

logger = logging.getLogger(__name__)


@dataclass
class MarginChecker:
    """Tracks and validates futures margin across open positions.

    Attributes:
        max_margin_pct: Maximum margin utilization allowed (default 0.80 = 80%).
        positions_margin: Dict mapping position_id -> margin_used for tracking.
    """

    max_margin_pct: float = 0.80
    positions_margin: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 < self.max_margin_pct <= 1.0:
            raise ValueError(
                f"max_margin_pct must be between 0 and 1, got {self.max_margin_pct}"
            )

    def track_margin(
        self, position_id: str, instrument: Instrument, contracts: int
    ) -> None:
        """Track margin for a new or updated position.

        Args:
            position_id: Unique identifier for the position.
            instrument: Trading instrument with margin_requirement.
            contracts: Number of contracts in the position.

        Raises:
            ValueError: If position_id is empty or contracts is negative.
        """
        if not position_id:
            raise ValueError("position_id cannot be empty")
        if contracts < 0:
            raise ValueError(f"contracts cannot be negative, got {contracts}")

        margin_used = instrument.margin_requirement * contracts
        self.positions_margin[position_id] = margin_used
        logger.debug(
            f"Tracked margin for {position_id}: {margin_used:.2f} "
            f"({contracts} contracts @ {instrument.margin_requirement:.2f} per contract)"
        )

    def release_margin(self, position_id: str) -> float:
        """Release margin for a closed position.

        Args:
            position_id: Unique identifier for the position to release.

        Returns:
            Margin amount that was released (0.0 if position not found).
        """
        margin_released = self.positions_margin.pop(position_id, 0.0)
        if margin_released > 0.0:
            logger.debug(f"Released margin for {position_id}: {margin_released:.2f}")
        return margin_released

    def check_margin(
        self,
        instrument: Instrument,
        contracts: int,
        account_nav: float,
    ) -> bool:
        """Check if a new position would exceed margin limits.

        Validates that:
            total_margin_used + (contracts * margin_per_contract) < account_nav * max_margin_pct

        Args:
            instrument: Trading instrument with margin_requirement.
            contracts: Number of contracts to check.
            account_nav: Current account NAV (net asset value).

        Returns:
            True if margin check passes, False otherwise.

        Raises:
            ValueError: If account_nav is non-positive.
        """
        if account_nav <= 0.0:
            raise ValueError(f"account_nav must be positive, got {account_nav}")

        # Calculate new margin requirement
        new_margin = instrument.margin_requirement * contracts

        # Get total margin already in use
        total_margin_used = sum(self.positions_margin.values())

        # Calculate projected margin
        projected_margin = total_margin_used + new_margin

        # Calculate allowed margin
        allowed_margin = account_nav * self.max_margin_pct

        # Log detailed check result
        passes = projected_margin < allowed_margin
        logger.debug(
            f"Margin check: projected={projected_margin:.2f}, "
            f"allowed={allowed_margin:.2f}, passes={passes}"
        )

        return passes

    def get_margin_utilization(self) -> Dict[str, float]:
        """Get margin utilization metrics.

        Returns:
            Dict with keys:
                - total_margin_used: Sum of all position margins
                - num_positions: Number of tracked positions
                - avg_position_margin: Average margin per position (0 if no positions)
        """
        total_margin = sum(self.positions_margin.values())
        num_positions = len(self.positions_margin)
        avg_margin = total_margin / num_positions if num_positions > 0 else 0.0

        return {
            "total_margin_used": total_margin,
            "num_positions": num_positions,
            "avg_position_margin": avg_margin,
        }
