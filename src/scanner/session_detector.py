"""
Session-Aware Trading Module.

Detects current FX trading session and provides session-specific confidence
modifiers and position sizing multipliers.

FX Sessions (UTC):
- Tokyo: 23:00-08:00 (wraps midnight)
- London: 08:00-17:00 (peak liquidity)
- New York: 13:00-22:00 (high volatility)
- London-NY Overlap: 13:00-17:00 (peak volatility)
- Off Hours: 22:00-23:00 (minimal liquidity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about current FX trading session."""

    name: str  # "TOKYO", "LONDON", "NEW_YORK", "OFF_HOURS"
    is_overlap: bool  # True during London-NY overlap
    overlap_name: Optional[str]  # "LONDON_NY" if overlapping
    confidence_modifiers: Dict[str, float]  # per-strategy confidence adjustments
    position_size_multiplier: float  # sizing adjustment for session
    description: str
    start_hour_utc: int
    end_hour_utc: int


@dataclass
class SessionConfig:
    """Configuration for session detection and adjustments."""

    version: str = "1.0"
    enabled: bool = True

    # Session hour boundaries (UTC, 24-hour format)
    tokyo_start: int = 23
    tokyo_end: int = 8  # wraps midnight
    london_start: int = 8
    london_end: int = 17
    new_york_start: int = 13
    new_york_end: int = 22
    london_ny_overlap_start: int = 13
    london_ny_overlap_end: int = 17

    # Confidence modifiers per session per strategy type
    # Tokyo: mean_reversion +0.10, trend -0.15 (low vol, range-bound)
    tokyo_modifiers: Dict[str, float] = field(
        default_factory=lambda: {
            "mean_reversion": 0.10,
            "trend": -0.15,
            "volatility": -0.10,
            "momentum": -0.05,
            "breakout": -0.20,
        }
    )

    # London: trend +0.10, breakout +0.05 (high vol, directional)
    london_modifiers: Dict[str, float] = field(
        default_factory=lambda: {
            "mean_reversion": -0.05,
            "trend": 0.10,
            "volatility": 0.10,
            "momentum": 0.05,
            "breakout": 0.05,
        }
    )

    # New York: data_driven +0.05 (news/data releases)
    new_york_modifiers: Dict[str, float] = field(
        default_factory=lambda: {
            "mean_reversion": 0.05,
            "trend": 0.05,
            "volatility": 0.05,
            "momentum": 0.05,
            "data_driven": 0.05,
        }
    )

    # London-NY overlap: momentum +0.10 (peak vol, strong moves)
    london_ny_overlap_modifiers: Dict[str, float] = field(
        default_factory=lambda: {
            "mean_reversion": -0.10,
            "trend": 0.15,
            "volatility": 0.15,
            "momentum": 0.10,
            "breakout": 0.10,
        }
    )

    # Off hours: all strategies -0.20 (low liquidity, unreliable)
    off_hours_modifiers: Dict[str, float] = field(
        default_factory=lambda: {
            "mean_reversion": -0.20,
            "trend": -0.20,
            "volatility": -0.20,
            "momentum": -0.20,
            "breakout": -0.20,
        }
    )

    # Position sizing multipliers per session
    # Tokyo: 0.70 (lower liquidity)
    tokyo_position_multiplier: float = 0.70

    # London: 1.00 (peak liquidity)
    london_position_multiplier: float = 1.00

    # New York: 0.85 (good liquidity but afternoon thins)
    new_york_position_multiplier: float = 0.85

    # London-NY overlap: 0.75 (high vol = reduce size for protection)
    london_ny_overlap_position_multiplier: float = 0.75

    # Off hours: 0.50 (minimal liquidity)
    off_hours_position_multiplier: float = 0.50

    def validate(self) -> None:
        """Validate config values at load time."""
        # Hour validation (0-23)
        for attr in [
            "tokyo_start",
            "tokyo_end",
            "london_start",
            "london_end",
            "new_york_start",
            "new_york_end",
            "london_ny_overlap_start",
            "london_ny_overlap_end",
        ]:
            value = getattr(self, attr)
            if not isinstance(value, int) or value < 0 or value > 23:
                raise ValueError(
                    f"{attr} must be an integer between 0 and 23, got {value}"
                )

        # Multiplier validation (> 0)
        for attr in [
            "tokyo_position_multiplier",
            "london_position_multiplier",
            "new_york_position_multiplier",
            "london_ny_overlap_position_multiplier",
            "off_hours_position_multiplier",
        ]:
            value = getattr(self, attr)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{attr} must be > 0, got {value}")

        logger.info(f"SessionConfig v{self.version} validated successfully")


class SessionDetector:
    """Detects current FX trading session and provides session-specific adjustments."""

    SESSIONS = {
        "TOKYO": (23, 8),      # 23:00-08:00 UTC (wraps midnight)
        "LONDON": (8, 17),     # 08:00-17:00 UTC
        "NEW_YORK": (13, 22),  # 13:00-22:00 UTC
    }

    OVERLAP = {
        "LONDON_NY": (13, 17),  # 13:00-17:00 UTC — peak volatility
    }

    def __init__(self, config: Optional[SessionConfig] = None):
        """Initialize session detector with optional config."""
        self.config = config or SessionConfig()
        self.config.validate()

    def get_current_session(self, utc_time: Optional[datetime] = None) -> SessionInfo:
        """Get current session with all modifiers.

        Args:
            utc_time: UTC datetime. If None, uses current UTC time.

        Returns:
            SessionInfo with name, modifiers, and position size multiplier.
        """
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)

        if not self.config.enabled:
            return self._get_neutral_session(utc_time)

        hour = utc_time.hour

        # Check for London-NY overlap first (13:00-17:00 UTC)
        if self._is_in_session(
            hour,
            self.config.london_ny_overlap_start,
            self.config.london_ny_overlap_end,
            wrap=False,
        ):
            return SessionInfo(
                name="LONDON_NY_OVERLAP",
                is_overlap=True,
                overlap_name="LONDON_NY",
                confidence_modifiers=self.config.london_ny_overlap_modifiers,
                position_size_multiplier=self.config.london_ny_overlap_position_multiplier,
                description="London-New York Overlap (Peak Volatility)",
                start_hour_utc=self.config.london_ny_overlap_start,
                end_hour_utc=self.config.london_ny_overlap_end,
            )

        # Check Tokyo (23:00-08:00, wraps midnight)
        if self._is_in_session(
            hour,
            self.config.tokyo_start,
            self.config.tokyo_end,
            wrap=True,
        ):
            return SessionInfo(
                name="TOKYO",
                is_overlap=False,
                overlap_name=None,
                confidence_modifiers=self.config.tokyo_modifiers,
                position_size_multiplier=self.config.tokyo_position_multiplier,
                description="Tokyo Session",
                start_hour_utc=self.config.tokyo_start,
                end_hour_utc=self.config.tokyo_end,
            )

        # Check London (08:00-17:00)
        if self._is_in_session(
            hour,
            self.config.london_start,
            self.config.london_end,
            wrap=False,
        ):
            return SessionInfo(
                name="LONDON",
                is_overlap=False,
                overlap_name=None,
                confidence_modifiers=self.config.london_modifiers,
                position_size_multiplier=self.config.london_position_multiplier,
                description="London Session",
                start_hour_utc=self.config.london_start,
                end_hour_utc=self.config.london_end,
            )

        # Check New York (13:00-22:00)
        if self._is_in_session(
            hour,
            self.config.new_york_start,
            self.config.new_york_end,
            wrap=False,
        ):
            return SessionInfo(
                name="NEW_YORK",
                is_overlap=False,
                overlap_name=None,
                confidence_modifiers=self.config.new_york_modifiers,
                position_size_multiplier=self.config.new_york_position_multiplier,
                description="New York Session",
                start_hour_utc=self.config.new_york_start,
                end_hour_utc=self.config.new_york_end,
            )

        # Off hours (22:00-23:00 UTC gap)
        return SessionInfo(
            name="OFF_HOURS",
            is_overlap=False,
            overlap_name=None,
            confidence_modifiers=self.config.off_hours_modifiers,
            position_size_multiplier=self.config.off_hours_position_multiplier,
            description="Off Hours (Low Liquidity)",
            start_hour_utc=22,
            end_hour_utc=23,
        )

    def get_confidence_modifier(
        self, strategy_type: str, session: Optional[SessionInfo] = None
    ) -> float:
        """Get confidence modifier for a strategy type in the given session.

        Args:
            strategy_type: Name of strategy (e.g., "trend", "mean_reversion")
            session: SessionInfo. If None, uses current session.

        Returns:
            Confidence modifier (can be negative). Returns 0.0 for unknown strategies.
        """
        if session is None:
            session = self.get_current_session()

        return session.confidence_modifiers.get(strategy_type, 0.0)

    def get_position_size_multiplier(
        self, session: Optional[SessionInfo] = None
    ) -> float:
        """Get position sizing multiplier for the given session.

        Args:
            session: SessionInfo. If None, uses current session.

        Returns:
            Position sizing multiplier (> 0).
        """
        if session is None:
            session = self.get_current_session()

        return session.position_size_multiplier

    def get_session_summary(self, utc_time: Optional[datetime] = None) -> Dict:
        """Get summary of all active sessions at given time.

        Args:
            utc_time: UTC datetime. If None, uses current UTC time.

        Returns:
            Dict with all session statuses and current active session.
        """
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)

        if not self.config.enabled:
            return {
                "enabled": False,
                "current_session": None,
                "message": "Session detection disabled in config",
            }

        hour = utc_time.hour
        current = self.get_current_session(utc_time)

        active_sessions = []
        tokyo_active = self._is_in_session(
            hour, self.config.tokyo_start, self.config.tokyo_end, wrap=True
        )
        london_active = self._is_in_session(
            hour, self.config.london_start, self.config.london_end, wrap=False
        )
        ny_active = self._is_in_session(
            hour, self.config.new_york_start, self.config.new_york_end, wrap=False
        )

        if tokyo_active:
            active_sessions.append("TOKYO")
        if london_active:
            active_sessions.append("LONDON")
        if ny_active:
            active_sessions.append("NEW_YORK")
        if not active_sessions:
            active_sessions.append("OFF_HOURS")

        return {
            "enabled": True,
            "utc_time": utc_time.isoformat(),
            "utc_hour": hour,
            "current_session": current.name,
            "current_description": current.description,
            "active_sessions": active_sessions,
            "position_size_multiplier": current.position_size_multiplier,
            "is_overlap": current.is_overlap,
        }

    @staticmethod
    def _is_in_session(hour: int, start: int, end: int, wrap: bool = False) -> bool:
        """Check if hour is within session range.

        Args:
            hour: Current hour (0-23)
            start: Session start hour
            end: Session end hour
            wrap: If True, session wraps midnight (e.g., 23-8)

        Returns:
            True if hour is within session.
        """
        if wrap:
            # Session wraps midnight (e.g., 23:00-08:00)
            return hour >= start or hour < end
        else:
            # Session does not wrap (e.g., 08:00-17:00)
            return start <= hour < end

    @staticmethod
    def _get_neutral_session(utc_time: datetime) -> SessionInfo:
        """Return neutral session when detection is disabled."""
        return SessionInfo(
            name="DISABLED",
            is_overlap=False,
            overlap_name=None,
            confidence_modifiers={},
            position_size_multiplier=1.0,
            description="Session detection disabled",
            start_hour_utc=0,
            end_hour_utc=24,
        )
