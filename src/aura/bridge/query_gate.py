"""AuraQueryGate — pre-execution co-cognition check.

Buddy calls this before high-risk trades. If Aura is available and responsive,
the gate asks about readiness. If Aura is offline or doesn't respond within
the timeout, the gate passes (non-blocking by design).

This is additive to existing execution gates — it never replaces them.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from .signals import AuraQuery, AuraQueryResponse, FeedbackBridge

logger = logging.getLogger(__name__)

# Thresholds that trigger an Aura query
QUERY_CONFIDENCE_THRESHOLD = 0.60    # Below this → ask Aura
QUERY_VOTE_THRESHOLD = 0.65          # Below this → ask Aura
QUERY_PORTFOLIO_RISK_THRESHOLD = 0.08  # Above this → ask Aura
QUERY_DRAWDOWN_THRESHOLD = 0.10      # Above this → ask Aura
VOLATILE_REGIMES = {"VOLATILE", "TRENDING_FAST", "CRISIS"}


class AuraQueryGate:
    """Pre-execution gate: queries Aura before borderline/high-risk trades.

    Usage:
        gate = AuraQueryGate(bridge, timeout_sec=25.0)
        result = gate.evaluate(trade_context)
        if result["hard_veto"]:
            # block trade
        elif result["confidence_modifier"] != 0.0:
            # adjust confidence
    """

    def __init__(
        self,
        bridge: Optional[FeedbackBridge] = None,
        timeout_sec: float = 25.0,
    ):
        self._bridge = bridge or FeedbackBridge()
        self._timeout = timeout_sec
        self._queries_sent: int = 0
        self._vetoes_received: int = 0
        self._timeouts: int = 0

    def should_query(self, trade_context: Dict[str, Any]) -> bool:
        """Return True if this trade warrants an Aura query."""
        confidence = float(trade_context.get("confidence", 1.0))
        vote_score = float(trade_context.get("weighted_vote_score", 1.0))
        portfolio_risk = float(trade_context.get("portfolio_risk_pct", 0.0))
        drawdown = float(trade_context.get("drawdown_today_pct", 0.0))
        regime = str(trade_context.get("regime", "NORMAL")).upper()

        if confidence < QUERY_CONFIDENCE_THRESHOLD:
            return True
        if vote_score < QUERY_VOTE_THRESHOLD:
            return True
        if portfolio_risk > QUERY_PORTFOLIO_RISK_THRESHOLD:
            return True
        if drawdown > QUERY_DRAWDOWN_THRESHOLD:
            return True
        if regime in VOLATILE_REGIMES:
            return True
        return False

    def evaluate(self, trade_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the gate. Returns result dict.

        Returns:
            {
                "gate_passed": bool,
                "confidence_modifier": float,
                "hard_veto": bool,
                "aura_available": bool,
                "response_received": bool,
                "query_id": str | None,
                "message": str,
            }
        """
        default_pass = {
            "gate_passed": True,
            "confidence_modifier": 0.0,
            "hard_veto": False,
            "aura_available": False,
            "response_received": False,
            "query_id": None,
            "message": "AuraQueryGate: Aura not queried (conditions not met or Aura offline)",
        }

        if not self.should_query(trade_context):
            default_pass["message"] = "AuraQueryGate: query not warranted for this trade"
            return default_pass

        # Check if Aura bridge is live (readiness file must exist)
        try:
            readiness = self._bridge.read_readiness()
            if readiness is None:
                default_pass["message"] = "AuraQueryGate: Aura bridge offline — proceeding without query"
                return default_pass
        except Exception as e:
            logger.warning("AuraQueryGate: bridge check failed: %s", e)
            return default_pass

        # Submit query
        query_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=self._timeout + 5)
        query = AuraQuery(
            query_id=query_id,
            query_type="readiness_check",
            context=trade_context,
            timestamp=now.isoformat(),
            expires_at=expires.isoformat(),
        )

        try:
            self._bridge.submit_query(query)
            self._queries_sent += 1
        except Exception as e:
            logger.warning("AuraQueryGate: failed to submit query: %s", e)
            default_pass["message"] = f"AuraQueryGate: query submission failed — proceeding: {e}"
            return default_pass

        # Wait for response
        response = self._bridge.wait_for_response(query_id, timeout_sec=self._timeout)

        if response is None:
            self._timeouts += 1
            return {
                "gate_passed": True,
                "confidence_modifier": 0.0,
                "hard_veto": False,
                "aura_available": True,
                "response_received": False,
                "query_id": query_id,
                "message": f"AuraQueryGate: timeout after {self._timeout}s — proceeding",
            }

        # Process response
        modifier = max(-0.3, min(0.1, response.confidence_modifier))
        gate_passed = not response.hard_veto

        if response.hard_veto:
            self._vetoes_received += 1
            logger.warning(
                "AuraQueryGate: HARD VETO from Aura (readiness=%.2f): %s",
                response.readiness_score, response.message,
            )
        elif not response.proceed:
            logger.info(
                "AuraQueryGate: Aura recommends caution (modifier=%.2f): %s",
                modifier, response.message,
            )
        else:
            logger.info(
                "AuraQueryGate: Aura clears trade (readiness=%.2f, modifier=%.2f)",
                response.readiness_score, modifier,
            )

        return {
            "gate_passed": gate_passed,
            "confidence_modifier": modifier,
            "hard_veto": response.hard_veto,
            "aura_available": True,
            "response_received": True,
            "query_id": query_id,
            "message": response.message,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "queries_sent": self._queries_sent,
            "vetoes_received": self._vetoes_received,
            "timeouts": self._timeouts,
        }
