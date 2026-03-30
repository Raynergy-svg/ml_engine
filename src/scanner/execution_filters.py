"""Pre-execution filter chain for execute_trade().

Allows new filters to be registered without modifying execute_trade() directly.
Each filter is a callable that takes a context dict and returns a FilterResult.
Filters run in registration order. First BLOCK short-circuits. REDUCE_SIZE multipliers compound.

Phase 50 (US-310): Extensible filter chain for gate modularization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of a single filter execution.

    Attributes:
        passed: True if filter allows trade, False otherwise.
        action: One of "PASS", "BLOCK", "REDUCE_SIZE".
        reason: Human-readable explanation of the decision.
        size_multiplier: Applied only if action == "REDUCE_SIZE". Compounds across filters.
        filter_name: Name of the filter that produced this result.
    """

    passed: bool
    action: str  # "PASS", "BLOCK", "REDUCE_SIZE"
    reason: str
    size_multiplier: float = 1.0
    filter_name: str = ""

    def __post_init__(self) -> None:
        """Validate action field."""
        if self.action not in ("PASS", "BLOCK", "REDUCE_SIZE"):
            raise ValueError(f"Invalid action: {self.action}. Must be PASS, BLOCK, or REDUCE_SIZE.")
        if self.size_multiplier < 0.01 or self.size_multiplier > 1.0:
            logger.warning(
                "Filter %s: size_multiplier=%.3f outside [0.01, 1.0]",
                self.filter_name,
                self.size_multiplier,
            )


class FilterChain:
    """Pluggable filter chain for pre-execution gates.

    Allows external modules to register filters without modifying execute_trade() directly.
    Filters run in registration order. The chain fails open (exception = PASS).

    Usage:
        chain = FilterChain()
        chain.register("spread_check", my_spread_filter_fn)
        chain.register("calendar_check", my_calendar_filter_fn)
        result = chain.apply(context_dict)
    """

    def __init__(self) -> None:
        """Initialize empty filter chain."""
        self._filters: List[Tuple[str, Callable[[Dict[str, Any]], FilterResult]]] = []

    def register(
        self,
        name: str,
        filter_fn: Callable[[Dict[str, Any]], FilterResult],
    ) -> None:
        """Register a filter.

        Filters run in registration order. Name is logged and returned in FilterResult.

        Args:
            name: Unique name for this filter (logged in results).
            filter_fn: Callable that takes context dict and returns FilterResult.
                Should be tolerant to missing keys in context.
        """
        if not callable(filter_fn):
            raise TypeError(f"filter_fn must be callable, got {type(filter_fn)}")
        self._filters.append((name, filter_fn))
        logger.debug(f"FilterChain: Registered filter '{name}' (total: {len(self._filters)})")

    def apply(self, context: Dict[str, Any]) -> FilterResult:
        """Run all filters in sequence.

        Args:
            context: Trade execution context dict. May be incomplete or missing keys.

        Returns:
            FilterResult representing final decision. First BLOCK short-circuits.
            If any filter exceptions, that filter is treated as PASS (fail-open).
            Size multipliers from REDUCE_SIZE filters are compounded.
        """
        if not self._filters:
            logger.debug("FilterChain: No filters registered, passing")
            return FilterResult(
                passed=True,
                action="PASS",
                reason="No filters registered",
                filter_name="FilterChain",
            )

        compound_size_mult = 1.0
        all_reasons = []

        for filter_name, filter_fn in self._filters:
            try:
                result = filter_fn(context)
                result.filter_name = filter_name
                logger.debug(
                    "FilterChain: %s → %s (%.3fx) [%s]",
                    filter_name,
                    result.action,
                    result.size_multiplier,
                    result.reason,
                )
                all_reasons.append(f"{filter_name}: {result.reason}")

                # Short-circuit on BLOCK
                if result.action == "BLOCK" and not result.passed:
                    return FilterResult(
                        passed=False,
                        action="BLOCK",
                        reason=result.reason,
                        filter_name=filter_name,
                    )

                # Compound size multipliers
                if result.action == "REDUCE_SIZE" and result.size_multiplier < 1.0:
                    compound_size_mult *= result.size_multiplier

            except Exception as e:
                logger.debug(f"FilterChain: Exception in {filter_name}: {e}")
                # Fail open: treat exception as PASS
                all_reasons.append(f"{filter_name}: exception (treated as PASS)")
                continue

        # All filters passed (or reduced size)
        return FilterResult(
            passed=True,
            action="REDUCE_SIZE" if compound_size_mult < 1.0 else "PASS",
            reason=" → ".join(all_reasons) if all_reasons else "all filters passed",
            size_multiplier=compound_size_mult,
            filter_name="FilterChain",
        )

    def get_filter_names(self) -> List[str]:
        """Return list of registered filter names."""
        return [name for name, _ in self._filters]

    def clear(self) -> None:
        """Clear all registered filters (useful for testing)."""
        self._filters.clear()
        logger.debug("FilterChain: Cleared all filters")
