"""
Regime-Conditional Gate Profiles and Logic.

This module provides adaptive gate thresholds based on market volatility regime
detected by the BOCPD (Bayesian Online Changepoint Detection) regime detector.

Instead of using static thresholds for all market conditions, regime_gates allows
confidence, momentum, and risk thresholds to adapt to LOW/NORMAL/HIGH/EXTREME
volatility regimes, improving entry quality across market cycles.

Regime Profiles:
- LOW: Low volatility, raise bar to filter noise, boost position size for quality setups
- NORMAL: Standard balanced thresholds (baseline)
- HIGH: High volatility, require momentum alignment, reduce position size
- EXTREME: Extreme volatility, very selective, small defensive positions

Integration with gates.py:
- evaluate_confidence() uses regime_specific threshold
- evaluate_momentum() checks momentum_alignment requirement in HIGH/EXTREME
- evaluate_all_gates() applies position_size_multiplier for regime-aware sizing
- Graceful fallback to static thresholds if regime is None or unknown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Regime constants (match regime_detector.py)
REGIME_LOW = "LOW"
REGIME_NORMAL = "NORMAL"
REGIME_HIGH = "HIGH"
REGIME_EXTREME = "EXTREME"

VALID_REGIMES = {REGIME_LOW, REGIME_NORMAL, REGIME_HIGH, REGIME_EXTREME}


@dataclass
class RegimeGateProfile:
    """Per-regime confidence/momentum/risk thresholds and sizing.

    Attributes:
        regime_name: One of LOW, NORMAL, HIGH, EXTREME
        confidence_threshold: Minimum confidence (0-100) to pass gate
        momentum_threshold: Minimum momentum (0-1) to pass gate
        risk_threshold: Maximum drawdown/risk (0-1) allowed
        min_rr_ratio: Minimum reward:risk ratio for trade (e.g. 1.2 means 1.2:1)
        position_size_multiplier: Scaling factor for position sizing (1.0 = baseline)
        require_momentum_alignment: If True, momentum must align with direction in HIGH/EXTREME
        description: Human-readable description of this regime's trading approach
    """

    regime_name: str
    confidence_threshold: float
    momentum_threshold: float
    risk_threshold: float
    min_rr_ratio: float
    position_size_multiplier: float
    require_momentum_alignment: bool
    description: str

    def validate(self) -> None:
        """Validate profile values at load time per improvement rules.

        Raises:
            ValueError: If any threshold is out of acceptable range.
        """
        if self.regime_name not in VALID_REGIMES:
            raise ValueError(
                f"regime_name '{self.regime_name}' not in {VALID_REGIMES}"
            )

        if not 0.0 <= self.confidence_threshold <= 100.0:
            raise ValueError(
                f"confidence_threshold {self.confidence_threshold} not in [0, 100]"
            )

        if not 0.0 <= self.momentum_threshold <= 1.0:
            raise ValueError(
                f"momentum_threshold {self.momentum_threshold} not in [0, 1]"
            )

        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError(
                f"risk_threshold {self.risk_threshold} not in [0, 1]"
            )

        if self.min_rr_ratio < 0.5:
            raise ValueError(
                f"min_rr_ratio {self.min_rr_ratio} should be >= 0.5"
            )

        if self.position_size_multiplier <= 0.0:
            raise ValueError(
                f"position_size_multiplier {self.position_size_multiplier} must be > 0"
            )


@dataclass
class RegimeGateConfig:
    """Configuration for regime-conditional gates.

    Attributes:
        enabled: If True, use regime-conditional thresholds; else use static
        profiles: Dict mapping regime_name -> RegimeGateProfile
        version: Config version for forward compatibility per improvement rules
    """

    enabled: bool = True
    profiles: Dict[str, RegimeGateProfile] = field(default_factory=dict)
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Initialize default profiles if not provided."""
        if not self.profiles:
            self.profiles = _get_default_profiles()
        else:
            # Validate all provided profiles
            for profile in self.profiles.values():
                try:
                    profile.validate()
                except ValueError as e:
                    logger.warning(f"Invalid regime profile: {e}")
                    raise


def _get_default_profiles() -> Dict[str, RegimeGateProfile]:
    """Create default regime profiles per specification.

    Returns:
        Dict mapping regime_name -> RegimeGateProfile
    """
    return {
        REGIME_LOW: RegimeGateProfile(
            regime_name=REGIME_LOW,
            confidence_threshold=0.60,
            momentum_threshold=0.40,
            risk_threshold=0.70,
            min_rr_ratio=1.5,
            position_size_multiplier=1.3,
            require_momentum_alignment=False,
            description="Low vol, raise bar to filter noise, boost size for quality setups",
        ),
        REGIME_NORMAL: RegimeGateProfile(
            regime_name=REGIME_NORMAL,
            confidence_threshold=0.50,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Standard balanced thresholds",
        ),
        REGIME_HIGH: RegimeGateProfile(
            regime_name=REGIME_HIGH,
            confidence_threshold=0.55,
            momentum_threshold=0.45,
            risk_threshold=0.65,
            min_rr_ratio=1.3,
            position_size_multiplier=0.65,
            require_momentum_alignment=True,
            description="High vol, require momentum alignment, reduce size",
        ),
        REGIME_EXTREME: RegimeGateProfile(
            regime_name=REGIME_EXTREME,
            confidence_threshold=0.65,
            momentum_threshold=0.50,
            risk_threshold=0.50,
            min_rr_ratio=1.5,
            position_size_multiplier=0.40,
            require_momentum_alignment=True,
            description="Extreme vol, very selective, small defensive positions",
        ),
    }


def get_regime_profile(regime_name: Optional[str]) -> Optional[RegimeGateProfile]:
    """Get regime profile by name, defaults to NORMAL if unknown.

    Args:
        regime_name: One of LOW, NORMAL, HIGH, EXTREME, or None

    Returns:
        RegimeGateProfile for the regime, or None if regime_name is None/empty.
        Unknown regimes default to NORMAL profile.
    """
    if not regime_name:
        return None

    profiles = _get_default_profiles()

    if regime_name not in profiles:
        logger.warning(
            f"Unknown regime '{regime_name}', defaulting to NORMAL profile"
        )
        return profiles[REGIME_NORMAL]

    return profiles[regime_name]


def apply_regime_gates(
    regime_name: Optional[str],
    base_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Apply regime-conditional thresholds to base thresholds.

    If regime_name is None/empty or regime gates are disabled, returns base_thresholds
    with defaults applied. Otherwise, returns adjusted thresholds from the regime profile.

    Args:
        regime_name: Current market regime (LOW, NORMAL, HIGH, EXTREME) or None
        base_thresholds: Dict with optional keys:
            - confidence_threshold: baseline confidence (default 50.0)
            - momentum_threshold: baseline momentum (default 0.35)
            - risk_threshold: baseline risk (default 0.80)
            - min_rr_ratio: baseline R:R (default 1.2)
            - position_size_multiplier: baseline size mult (default 1.0)
            - require_momentum_alignment: baseline alignment req (default False)

    Returns:
        Dict with adjusted thresholds:
        - confidence_threshold
        - momentum_threshold
        - risk_threshold
        - min_rr_ratio
        - position_size_multiplier
        - require_momentum_alignment
    """
    if base_thresholds is None:
        base_thresholds = {}

    # Default static thresholds
    defaults = {
        "confidence_threshold": 50.0,
        "momentum_threshold": 0.35,
        "risk_threshold": 0.80,
        "min_rr_ratio": 1.2,
        "position_size_multiplier": 1.0,
        "require_momentum_alignment": False,
    }

    # Merge base thresholds with defaults
    result = {**defaults, **base_thresholds}

    # If no regime or regime gates disabled, return static thresholds
    if not regime_name:
        logger.debug("No regime specified, using static thresholds")
        return result

    # Get regime profile
    profile = get_regime_profile(regime_name)
    if profile is None:
        logger.debug(f"Regime '{regime_name}' not available, using static thresholds")
        return result

    # Apply regime-specific thresholds
    regime_adjusted = {
        "confidence_threshold": profile.confidence_threshold * 100.0,  # Scale to 0-100
        "momentum_threshold": profile.momentum_threshold,
        "risk_threshold": profile.risk_threshold,
        "min_rr_ratio": profile.min_rr_ratio,
        "position_size_multiplier": profile.position_size_multiplier,
        "require_momentum_alignment": profile.require_momentum_alignment,
    }

    logger.debug(
        f"Applied regime '{regime_name}' gates: "
        f"conf={regime_adjusted['confidence_threshold']:.1f}, "
        f"mom={regime_adjusted['momentum_threshold']:.2f}, "
        f"risk={regime_adjusted['risk_threshold']:.2f}, "
        f"mult={regime_adjusted['position_size_multiplier']:.2f}"
    )

    return regime_adjusted
