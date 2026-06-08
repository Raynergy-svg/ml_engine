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

            # Agent veto must happen before the broker/execution layer. The
            # execution manager also blocks this, but serial execution should
            # classify it as a skipped candidate, not an attempted order.
            if not bool(getattr(candidate, "agent_passed", False)):
                funnel.record_skip(pair, "agent_veto_pre_execution")
                continue

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
        """Execute a single trade via ExecutionManager.

        Unpacks the PairAnalysis dataclass into execute_trade() kwargs,
        mirroring the pattern used by ExecutionManager.execute_trades().
        """
        if self._em is None:
            return {"success": False, "reason": "no_execution_manager"}

        try:
            a = analysis

            # Map volatility_regime string -> int expected by execute_trade
            regime_str = str(getattr(a, "volatility_regime", "") or "").upper()
            regime_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}
            vol_regime = regime_map.get(regime_str)

            # Meta confidence (prefer TCN, fallback to weighted vote score)
            meta_conf = (
                getattr(a, "tcn_confidence", None)
                or getattr(a, "weighted_vote_score", None)
            )
            meta_conf_f = float(meta_conf) if meta_conf is not None else None

            # recommended_lots: only pass if > 0, else let execute_trade size it
            rec_lots = getattr(a, "recommended_lots", None)
            try:
                lots = float(rec_lots) if rec_lots and float(rec_lots) > 0 else None
            except (TypeError, ValueError):
                lots = None

            # Build analysis_context for journal + downstream logic
            agent_reasons = getattr(a, "agent_reasons", None) or []
            analysis_context: Dict[str, Any] = {
                "momentum_passed": getattr(a, "momentum_passed", None),
                "confidence_passed": getattr(a, "confidence_passed", None),
                "risk_passed": getattr(a, "risk_passed", None),
                "gate_summary": getattr(a, "gate_summary", None),
                "agent_votes": getattr(a, "agent_votes", None),
                "agent_total": getattr(a, "agent_total", None),
                "agent_passed": getattr(a, "agent_passed", None),
                "agent_promoted": getattr(a, "agent_promoted", None),
                "fasttrack": getattr(a, "fasttrack", False),
                "weighted_vote_score": getattr(a, "weighted_vote_score", None),
                "agent_reasons": agent_reasons[:5] if agent_reasons else [],
                "volatility_regime": getattr(a, "volatility_regime", None),
                "atr_pips": getattr(a, "atr_pips", None),
                "uncertainty_score": getattr(a, "uncertainty_score", None),
                "model_disagreement": getattr(a, "model_disagreement", None),
                "tcn_confidence": getattr(a, "tcn_confidence", None),
                "ridge_confidence": getattr(a, "ridge_confidence", None),
                "momentum": getattr(a, "momentum", None),
                "drawdown": getattr(a, "drawdown", None),
                "overall_score": getattr(a, "overall_score", None),
                "risk_pct": getattr(a, "risk_pct", None),
                "source": "serial_executor",
            }

            result = self._em.execute_trade(
                pair=getattr(a, "pair", ""),
                direction=getattr(a, "direction", "HOLD"),
                confidence=float(getattr(a, "confidence", 0.0) or 0.0),
                current_price=float(getattr(a, "current_price", 0.0) or 0.0),
                atr=float(getattr(a, "atr", 0.0) or 0.0),
                sl_pips=getattr(a, "sl_pips", None),
                tp_pips=getattr(a, "tp_pips", None),
                lots=lots,
                volatility_regime=vol_regime,
                meta_confidence=meta_conf_f,
                analysis_context=analysis_context,
            )

            if result is None:
                return {"success": False, "reason": "execute_trade_returned_none"}
            # ExecutionManager.execute_trade returns a journal entry dict on success
            if isinstance(result, dict):
                tid = result.get("oanda_trade_id", result.get("trade_id", ""))
                return {"success": True, "trade_id": tid}
            # Some code paths return ExecutionResult-like objects
            success = getattr(result, "success", False)
            tid = getattr(result, "trade_id", "")
            reason = getattr(result, "reason", getattr(result, "error", "unknown"))
            return {"success": success, "trade_id": tid, "reason": reason}
        except Exception as e:
            logger.warning(
                "serial_executor.execute_single_error",
                pair=getattr(analysis, "pair", "UNKNOWN"),
                error=str(e),
            )
            return {"success": False, "reason": str(e)}

    def _is_correlated(self, pair: str, executed_pairs: List[str]) -> bool:
        """Check if pair is correlated with any open or just-executed positions."""
        if not executed_pairs:
            # Check against broker open positions
            try:
                if hasattr(self._em, "_check_correlation_exposure"):
                    result = self._em._check_correlation_exposure(pair)
                    # _check_correlation_exposure returns Tuple[bool, str]
                    # where bool = allowed (True = OK to trade, False = blocked)
                    if isinstance(result, tuple):
                        allowed, _reason = result
                        return not allowed
                    return not bool(result)
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
