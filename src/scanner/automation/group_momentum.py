"""Cross-pair signal aggregation with group momentum scoring.

Groups currency pairs by their component currencies (EUR, USD, JPY, etc.)
and computes directional momentum per group from scan results. Enables:
- Trade filtering when individual signal conflicts with group direction
- Position boost when individual signal aligns with strong group momentum

Usage:
    aggregator = GroupMomentumAggregator()
    group_scores = aggregator.compute_group_momentum(scan_results)
    action = aggregator.evaluate_trade("EUR_USD", "LONG", 0.65, group_scores)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Currency groups: map each tradeable pair to its component currencies
PAIR_CURRENCIES: Dict[str, Tuple[str, str]] = {
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "USD_JPY": ("USD", "JPY"),
    "USD_CHF": ("USD", "CHF"),
    "AUD_USD": ("AUD", "USD"),
    "NZD_USD": ("NZD", "USD"),
    "USD_CAD": ("USD", "CAD"),
    "EUR_GBP": ("EUR", "GBP"),
    "EUR_JPY": ("EUR", "JPY"),
    "GBP_JPY": ("GBP", "JPY"),
    "EUR_CHF": ("EUR", "CHF"),
    "AUD_JPY": ("AUD", "JPY"),
    "EUR_AUD": ("EUR", "AUD"),
    "GBP_AUD": ("GBP", "AUD"),
    "NZD_JPY": ("NZD", "JPY"),
    "CAD_JPY": ("CAD", "JPY"),
    "EUR_CAD": ("EUR", "CAD"),
    "GBP_CHF": ("GBP", "CHF"),
    "AUD_CAD": ("AUD", "CAD"),
    "EUR_NZD": ("EUR", "NZD"),
    "GBP_NZD": ("GBP", "NZD"),
    "AUD_CHF": ("AUD", "CHF"),
    "AUD_NZD": ("AUD", "NZD"),
    "NZD_CAD": ("NZD", "CAD"),
    "NZD_CHF": ("NZD", "CHF"),
    "CHF_JPY": ("CHF", "JPY"),
    "CAD_CHF": ("CAD", "CHF"),
}

# Direction multipliers: for base/quote currency contribution
# LONG EUR_USD → EUR bullish (+1), USD bearish (-1)
# SHORT EUR_USD → EUR bearish (-1), USD bullish (+1)
DIRECTION_SIGN = {"LONG": 1.0, "SHORT": -1.0}


@dataclass
class GroupMomentumResult:
    """Result of group momentum evaluation for a trade."""

    action: str  # "allow", "block", "boost"
    reason: str
    group_scores: Dict[str, float] = field(default_factory=dict)
    base_score: float = 0.0
    quote_score: float = 0.0
    alignment: float = 0.0  # -1 to +1 alignment with trade direction
    position_modifier: float = 1.0  # Multiplier for position size


class GroupMomentumAggregator:
    """Aggregates directional signals across currency groups.

    Computes per-currency momentum from scan results, then evaluates
    whether an individual trade aligns with or conflicts against its
    currency group's overall direction.
    """

    def __init__(
        self,
        conflict_threshold: float = -0.2,
        boost_threshold: float = 0.5,
        boost_modifier: float = 1.05,
        min_pairs_for_group: int = 2,
    ):
        """Initialize aggregator.

        Args:
            conflict_threshold: Block trade if alignment below this (default -0.2)
            boost_threshold: Boost position if group momentum above this (default 0.5)
            boost_modifier: Position size multiplier when boosted (default 1.05 = +5%)
            min_pairs_for_group: Minimum pairs contributing to a group for it to be valid
        """
        self.conflict_threshold = conflict_threshold
        self.boost_threshold = boost_threshold
        self.boost_modifier = boost_modifier
        self.min_pairs_for_group = min_pairs_for_group

    def compute_group_momentum(
        self, scan_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute directional momentum per currency group.

        Each scan result contributes a directional signal weighted by confidence.
        A LONG signal on EUR_USD contributes +confidence to EUR, -confidence to USD.

        Args:
            scan_results: List of scan analysis dicts with keys:
                - pair: str (e.g., "EUR_USD")
                - direction: str ("LONG" or "SHORT")
                - confidence: float (0-1)

        Returns:
            Dict mapping currency code to momentum score (-1.0 to +1.0)
        """
        # Accumulate weighted directional votes per currency
        currency_votes: Dict[str, List[float]] = {}

        for result in scan_results:
            pair = result.get("pair", "")
            direction = result.get("direction", "").upper()
            confidence = float(result.get("confidence", 0.0))

            if not pair or direction not in DIRECTION_SIGN or confidence <= 0:
                continue

            currencies = PAIR_CURRENCIES.get(pair)
            if not currencies:
                # Try to parse unknown pair format
                parts = pair.split("_")
                if len(parts) == 2:
                    currencies = (parts[0], parts[1])
                else:
                    continue

            base_ccy, quote_ccy = currencies
            sign = DIRECTION_SIGN[direction]

            # Base currency gets the direction, quote gets the inverse
            currency_votes.setdefault(base_ccy, []).append(sign * confidence)
            currency_votes.setdefault(quote_ccy, []).append(-sign * confidence)

        # Compute normalized momentum per group
        group_momentum: Dict[str, float] = {}
        for ccy, votes in currency_votes.items():
            if len(votes) < self.min_pairs_for_group:
                # Not enough data — assign neutral
                group_momentum[ccy] = 0.0
            else:
                # Average votes, clamped to [-1, 1]
                avg = sum(votes) / len(votes)
                group_momentum[ccy] = max(-1.0, min(1.0, round(avg, 4)))

        if group_momentum:
            nonzero = {k: v for k, v in group_momentum.items() if abs(v) > 0.01}
            if nonzero:
                logger.debug(
                    f"Group momentum: {nonzero} "
                    f"(from {len(scan_results)} scan results)"
                )

        return group_momentum

    def evaluate_trade(
        self,
        pair: str,
        direction: str,
        confidence: float,
        group_scores: Dict[str, float],
    ) -> GroupMomentumResult:
        """Evaluate whether a trade aligns with group momentum.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            direction: Trade direction ("LONG" or "SHORT")
            confidence: Trade confidence (0-1)
            group_scores: Group momentum scores from compute_group_momentum()

        Returns:
            GroupMomentumResult with action, reason, and position modifier
        """
        currencies = PAIR_CURRENCIES.get(pair)
        if not currencies:
            parts = pair.split("_")
            if len(parts) == 2:
                currencies = (parts[0], parts[1])
            else:
                return GroupMomentumResult(
                    action="allow",
                    reason=f"Unknown pair {pair} — no group data",
                    group_scores=group_scores,
                )

        base_ccy, quote_ccy = currencies
        base_score = group_scores.get(base_ccy, 0.0)
        quote_score = group_scores.get(quote_ccy, 0.0)

        # Compute alignment: for LONG EUR_USD, we want EUR bullish (positive)
        # and USD bearish (negative). Alignment = base_sign - quote_sign
        sign = DIRECTION_SIGN.get(direction.upper(), 0.0)
        if sign == 0.0:
            return GroupMomentumResult(
                action="allow",
                reason=f"Invalid direction {direction}",
                group_scores=group_scores,
                base_score=base_score,
                quote_score=quote_score,
            )

        # Alignment: how well group momentum supports the trade direction
        # LONG: alignment = base_momentum - quote_momentum (want base up, quote down)
        # SHORT: alignment = -base_momentum + quote_momentum (want base down, quote up)
        alignment = sign * (base_score - quote_score) / 2.0
        alignment = max(-1.0, min(1.0, round(alignment, 4)))

        # Decision logic
        if confidence > 0.55 and alignment < self.conflict_threshold:
            # High-confidence trade conflicts with group direction
            return GroupMomentumResult(
                action="block",
                reason=(
                    f"{pair} {direction} conflicts with group momentum: "
                    f"{base_ccy}={base_score:+.2f}, {quote_ccy}={quote_score:+.2f}, "
                    f"alignment={alignment:+.3f} < {self.conflict_threshold}"
                ),
                group_scores=group_scores,
                base_score=base_score,
                quote_score=quote_score,
                alignment=alignment,
                position_modifier=1.0,
            )

        if alignment > self.boost_threshold:
            # Strong group alignment — boost position
            return GroupMomentumResult(
                action="boost",
                reason=(
                    f"{pair} {direction} strongly aligned with group momentum: "
                    f"{base_ccy}={base_score:+.2f}, {quote_ccy}={quote_score:+.2f}, "
                    f"alignment={alignment:+.3f} > {self.boost_threshold}"
                ),
                group_scores=group_scores,
                base_score=base_score,
                quote_score=quote_score,
                alignment=alignment,
                position_modifier=self.boost_modifier,
            )

        # Neutral — allow without modification
        return GroupMomentumResult(
            action="allow",
            reason=(
                f"{pair} {direction} neutral group momentum: "
                f"{base_ccy}={base_score:+.2f}, {quote_ccy}={quote_score:+.2f}, "
                f"alignment={alignment:+.3f}"
            ),
            group_scores=group_scores,
            base_score=base_score,
            quote_score=quote_score,
            alignment=alignment,
            position_modifier=1.0,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return aggregator configuration status."""
        return {
            "conflict_threshold": self.conflict_threshold,
            "boost_threshold": self.boost_threshold,
            "boost_modifier": self.boost_modifier,
            "min_pairs_for_group": self.min_pairs_for_group,
            "known_pairs": len(PAIR_CURRENCIES),
        }
