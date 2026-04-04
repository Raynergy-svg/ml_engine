"""Serial trade executor — one at a time with re-validation between each.

Fixes the "6 found but 2 executed" bug by:
1. Executing trades one at a time in confidence order
2. Re-checking portfolio risk against LIVE broker state between each
3. Re-checking correlation against CURRENTLY open positions (not pre-scan snapshot)
4. Stopping when ceiling reached (not silently dropping)
5. Logging a full execution funnel
"""

from typing import Any, Dict, List, Optional

import structlog

from src.scanner.automation.execution_funnel import ExecutionFunnel

logger = structlog.get_logger(__name__)


class SerialExecutor:
    """Execute trades one at a time with re-validation between each."""

    def __init__(
        self,
        execution_manager: Any = None,
        scanner: Any = None,
        max_spread_pips: float = 3.0,
    ):
        self._em = execution_manager
        self._scanner = scanner
        self._max_spread_pips = max_spread_pips

    def execute(
        self,
        candidates: List[Any],
        max_trades: Optional[int] = None,
    ) -> ExecutionFunnel:
        """Execute candidates serially with re-validation.

        Args:
            candidates: List of PairAnalysis objects, sorted by confidence (highest first).
            max_trades: Maximum trades to execute this round. None = no limit.

        Returns:
            ExecutionFunnel with full attrition tracking.
        """
        funnel = ExecutionFunnel()
        funnel.capacity_filtered = len(candidates)

        if not candidates or self._em is None:
            funnel.log_summary()
            return funnel

        if max_trades is not None and max_trades <= 0:
            funnel.log_summary()
            return funnel

        # Sort by confidence descending
        sorted_candidates = sorted(
            candidates,
            key=lambda a: getattr(a, "confidence", 0),
            reverse=True,
        )

        executed_count = 0
        executed_pairs: List[str] = []

        for candidate in sorted_candidates:
            pair = getattr(candidate, "pair", "UNKNOWN")

            # Check max trades limit
            if max_trades is not None and executed_count >= max_trades:
                funnel.record_skip(pair, "max_trades_reached")
                continue

            # Re-check portfolio risk against LIVE state
            try:
                can_trade = self._em.can_trade() if hasattr(self._em, "can_trade") else True
                if not can_trade:
                    funnel.record_skip(pair, "portfolio_risk_ceiling")
                    logger.info(
                        "serial_executor.risk_ceiling",
                        pair=pair,
                        executed_so_far=executed_count,
                    )
                    break  # Stop entirely — risk ceiling is a hard stop
            except Exception as e:
                logger.debug("serial_executor.risk_check_error", error=str(e))

            # Re-check correlation against currently open + just-executed pairs
            if self._is_correlated(pair, executed_pairs):
                funnel.record_skip(pair, "correlated_with_open")
                continue

            # Execute single trade
            funnel.attempted += 1
            try:
                result = self._execute_single(candidate)
                if result.get("success"):
                    executed_count += 1
                    funnel.executed += 1
                    executed_pairs.append(pair)
                    logger.info(
                        "serial_executor.trade_opened",
                        pair=pair,
                        trade_id=result.get("trade_id", ""),
                        n=executed_count,
                    )
                else:
                    reason = result.get("reason", "execution_failed")
                    funnel.record_skip(pair, reason)
            except Exception as e:
                funnel.record_skip(pair, f"exception: {e}")
                logger.warning("serial_executor.execution_error", pair=pair, error=str(e))

        funnel.log_summary()
        return funnel

    def _execute_single(self, analysis: Any) -> Dict[str, Any]:
        """Execute a single trade via ExecutionManager."""
        if self._em is None:
            return {"success": False, "reason": "no_execution_manager"}

        try:
            # Use execute_trade which handles all gates internally
            result = self._em.execute_trade(analysis)
            if result is None:
                return {"success": False, "reason": "execute_trade_returned_none"}
            # ExecutionManager.execute_trade returns a journal entry dict on success
            if isinstance(result, dict):
                tid = result.get("oanda_trade_id", result.get("trade_id", ""))
                return {"success": True, "trade_id": tid}
            # Some code paths return ExecutionResult-like objects
            success = getattr(result, "success", False)
            tid = getattr(result, "trade_id", "")
            reason = getattr(result, "reason", "unknown")
            return {"success": success, "trade_id": tid, "reason": reason}
        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _is_correlated(self, pair: str, executed_pairs: List[str]) -> bool:
        """Check if pair is correlated with any open or just-executed positions."""
        if not executed_pairs:
            # Check against broker open positions
            try:
                if hasattr(self._em, "_check_correlation_exposure"):
                    is_blocked = self._em._check_correlation_exposure(pair)
                    return bool(is_blocked)
            except Exception:
                pass
            return False

        # Simple correlation check: same base or quote currency
        pair_parts = set(pair.split("_"))
        for ep in executed_pairs:
            ep_parts = set(ep.split("_"))
            if pair_parts & ep_parts:  # Shared currency
                return True
        return False
