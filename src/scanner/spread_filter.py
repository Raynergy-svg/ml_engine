"""
Spread-Aware Entry Gate for Buddy FX Trading Bot.

Normalizes current spread as a percentage of ATR to filter out
high-spread conditions that erode edge. Research shows spread > 20%
of ATR significantly degrades trade expectancy.

US-300 (Phase 48): Trade Intelligence & Adaptive Filtering.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class SpreadCheckResult:
    """Result of a spread filter evaluation."""
    passed: bool
    spread_pips: float
    atr_pips: float
    normalized_spread: float  # spread_pips / atr_pips (0.0 - 1.0+)
    threshold: float
    reason: str

    @property
    def size_multiplier(self) -> float:
        """Position size multiplier based on spread conditions.

        Even when passed, wider spreads should reduce size slightly.
        Returns 1.0 for tight spreads, scales down linearly.
        """
        if not self.passed:
            return 0.0
        if self.normalized_spread <= 0.05:
            return 1.0
        # Linear scale: 5% spread → 1.0, threshold → 0.7
        ratio = (self.normalized_spread - 0.05) / max(self.threshold - 0.05, 0.01)
        return max(0.7, 1.0 - 0.3 * ratio)


@dataclass
class SpreadFilterConfig:
    """Configuration for spread-aware entry filtering."""

    # Maximum normalized spread (spread/ATR) to allow entry
    max_normalized_spread: float = 0.20  # 20% of ATR

    # Per-pair overrides (some pairs naturally have wider spreads)
    pair_overrides: Optional[Dict[str, float]] = None

    # Whether to enable the filter (allows runtime toggle)
    enabled: bool = True

    # Version field for forward compatibility
    version: int = 1

    def __post_init__(self):
        if self.pair_overrides is None:
            self.pair_overrides = {
                # Exotic/cross pairs with naturally wider spreads
                "GBP_NZD": 0.25,
                "EUR_NZD": 0.25,
                "GBP_AUD": 0.25,
                "GBP_CAD": 0.25,
                "NZD_JPY": 0.25,
                "CAD_JPY": 0.25,
                "CHF_JPY": 0.25,
                # Major pairs can be tighter
                "EUR_USD": 0.15,
                "GBP_USD": 0.15,
                "USD_JPY": 0.15,
            }

    def validate(self) -> None:
        """Validate configuration values at load time."""
        if not 0.01 <= self.max_normalized_spread <= 1.0:
            raise ValueError(
                f"max_normalized_spread must be 0.01-1.0, got {self.max_normalized_spread}"
            )
        if self.pair_overrides:
            for pair, threshold in self.pair_overrides.items():
                if not 0.01 <= threshold <= 1.0:
                    raise ValueError(
                        f"pair override for {pair} must be 0.01-1.0, got {threshold}"
                    )

    def get_threshold(self, pair: str) -> float:
        """Get the spread threshold for a specific pair."""
        if self.pair_overrides and pair in self.pair_overrides:
            return self.pair_overrides[pair]
        return self.max_normalized_spread


# Pip value lookup for spread calculation
_PIP_VALUES = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "USD_JPY": 0.01,
    "USD_CHF": 0.0001, "AUD_USD": 0.0001, "USD_CAD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001, "EUR_JPY": 0.01,
    "GBP_JPY": 0.01, "AUD_JPY": 0.01, "EUR_AUD": 0.0001,
    "GBP_AUD": 0.0001, "EUR_CHF": 0.0001, "GBP_CHF": 0.0001,
    "EUR_NZD": 0.0001, "GBP_NZD": 0.0001, "AUD_NZD": 0.0001,
    "NZD_JPY": 0.01, "CAD_JPY": 0.01, "CHF_JPY": 0.01,
    "USD_SGD": 0.0001, "EUR_CAD": 0.0001, "GBP_CAD": 0.0001,
}


class SpreadFilter:
    """Filters trade entries based on current spread relative to ATR.

    Normalizes spread as a percentage of ATR to provide a universal
    threshold that works across all pairs regardless of pip scale.

    Usage:
        filter = SpreadFilter()
        result = filter.check_spread(pair="EUR_USD", bid=1.0850, ask=1.0852, atr_pips=30.0)
        if not result.passed:
            logger.info(f"Skipping {pair}: spread too wide ({result.reason})")
    """

    def __init__(self, config: Optional[SpreadFilterConfig] = None):
        self._config = config or SpreadFilterConfig()
        try:
            self._config.validate()
        except ValueError as e:
            logger.warning(f"SpreadFilter config validation failed: {e}, using defaults")
            self._config = SpreadFilterConfig()

    @property
    def config(self) -> SpreadFilterConfig:
        return self._config

    def check_spread(
        self,
        pair: str,
        bid: float,
        ask: float,
        atr_pips: float,
    ) -> SpreadCheckResult:
        """Check if spread is acceptable for entry.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            bid: Current bid price
            ask: Current ask price
            atr_pips: Current ATR in pips

        Returns:
            SpreadCheckResult with pass/fail and details
        """
        if not self._config.enabled:
            return SpreadCheckResult(
                passed=True,
                spread_pips=0.0,
                atr_pips=atr_pips,
                normalized_spread=0.0,
                threshold=self._config.get_threshold(pair),
                reason="spread_filter_disabled",
            )

        # Calculate spread in pips
        pip_value = _PIP_VALUES.get(pair, 0.0001)
        spread_raw = abs(ask - bid)
        spread_pips = spread_raw / pip_value

        # Validate ATR
        if atr_pips <= 0:
            logger.warning(
                f"SpreadFilter: ATR is {atr_pips} for {pair}, "
                "allowing trade with warning (cannot normalize spread)"
            )
            return SpreadCheckResult(
                passed=True,
                spread_pips=spread_pips,
                atr_pips=atr_pips,
                normalized_spread=0.0,
                threshold=self._config.get_threshold(pair),
                reason="atr_zero_or_negative_fallback",
            )

        # Normalize spread
        normalized_spread = spread_pips / atr_pips
        threshold = self._config.get_threshold(pair)

        passed = normalized_spread <= threshold
        reason = "spread_acceptable" if passed else (
            f"spread_too_wide: {normalized_spread:.3f} > {threshold:.3f} "
            f"({spread_pips:.1f} pips / {atr_pips:.1f} ATR)"
        )

        if not passed:
            logger.info(
                f"SpreadFilter BLOCKED {pair}: normalized_spread={normalized_spread:.3f} "
                f"(threshold={threshold:.3f}, spread={spread_pips:.1f}pips, ATR={atr_pips:.1f}pips)"
            )

        return SpreadCheckResult(
            passed=passed,
            spread_pips=spread_pips,
            atr_pips=atr_pips,
            normalized_spread=normalized_spread,
            threshold=threshold,
            reason=reason,
        )

    def check_spread_from_price(
        self,
        pair: str,
        price: float,
        spread_pips: float,
        atr_pips: float,
    ) -> SpreadCheckResult:
        """Check spread when bid/ask aren't available but spread in pips is known.

        Args:
            pair: Currency pair
            price: Current price (mid)
            spread_pips: Known spread in pips
            atr_pips: Current ATR in pips

        Returns:
            SpreadCheckResult
        """
        if not self._config.enabled:
            return SpreadCheckResult(
                passed=True,
                spread_pips=spread_pips,
                atr_pips=atr_pips,
                normalized_spread=0.0,
                threshold=self._config.get_threshold(pair),
                reason="spread_filter_disabled",
            )

        if atr_pips <= 0:
            logger.warning(
                f"SpreadFilter: ATR is {atr_pips} for {pair}, "
                "allowing trade with warning"
            )
            return SpreadCheckResult(
                passed=True,
                spread_pips=spread_pips,
                atr_pips=atr_pips,
                normalized_spread=0.0,
                threshold=self._config.get_threshold(pair),
                reason="atr_zero_or_negative_fallback",
            )

        normalized_spread = spread_pips / atr_pips
        threshold = self._config.get_threshold(pair)
        passed = normalized_spread <= threshold

        reason = "spread_acceptable" if passed else (
            f"spread_too_wide: {normalized_spread:.3f} > {threshold:.3f}"
        )

        if not passed:
            logger.info(
                f"SpreadFilter BLOCKED {pair}: {spread_pips:.1f}pips / "
                f"{atr_pips:.1f}ATR = {normalized_spread:.3f} > {threshold:.3f}"
            )

        return SpreadCheckResult(
            passed=passed,
            spread_pips=spread_pips,
            atr_pips=atr_pips,
            normalized_spread=normalized_spread,
            threshold=threshold,
            reason=reason,
        )


def create_default_spread_filter(
    config: Optional[SpreadFilterConfig] = None,
) -> SpreadFilter:
    """Factory function for SpreadFilter with optional config override."""
    return SpreadFilter(config=config)
