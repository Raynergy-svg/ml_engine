"""
Cluster-Aware Confidence Gate — Phase 54 (US-337).

At execution time, classifies incoming trades against the stored cluster registry
(from TradeClusterAnalyzer) and adjusts confidence accordingly.

High-win cluster match: +0.05 boost
Low-win cluster match: -0.10 penalty
No match: pass through unchanged

Wires in AFTER setup quality gate, BEFORE adaptive R:R.

Usage:
    gate = ClusterConfidenceGate()
    result = gate.evaluate(trade_features, confidence)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClusterGateResult:
    """Result of cluster gate evaluation."""
    original_confidence: float = 0.0
    adjusted_confidence: float = 0.0
    cluster_matched: bool = False
    cluster_id: int = -1
    cluster_label: str = "none"
    cluster_win_rate: float = 0.0
    confidence_adjustment: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_confidence": round(self.original_confidence, 4),
            "adjusted_confidence": round(self.adjusted_confidence, 4),
            "cluster_matched": self.cluster_matched,
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "cluster_win_rate": round(self.cluster_win_rate, 4),
            "confidence_adjustment": round(self.confidence_adjustment, 4),
        }


class ClusterConfidenceGate:
    """Adjusts trade confidence based on cluster membership.

    Uses TradeClusterAnalyzer's nearest-centroid matching to boost
    or penalize confidence based on historical cluster win rates.
    """

    def __init__(self, cluster_analyzer=None, max_boost: float = 0.10,
                 max_penalty: float = -0.10):
        """
        Args:
            cluster_analyzer: TradeClusterAnalyzer instance (lazy-loaded if None)
            max_boost: Maximum confidence boost (default +0.10)
            max_penalty: Maximum confidence penalty (default -0.10)
        """
        self._analyzer = cluster_analyzer
        self._max_boost = max_boost
        self._max_penalty = max_penalty
        self._match_count: int = 0
        self._total_evaluations: int = 0

        if self._analyzer is None:
            try:
                from src.scanner.trade_clusters import TradeClusterAnalyzer
                self._analyzer = TradeClusterAnalyzer()
            except Exception as e:
                logger.debug(f"Phase 54 (US-337): Cluster analyzer not available: {e}")

    def evaluate(self, trade: Dict[str, Any], confidence: float) -> ClusterGateResult:
        """Evaluate a trade against cluster registry and adjust confidence.

        Args:
            trade: Trade features dict
            confidence: Current confidence score

        Returns:
            ClusterGateResult with adjustment details
        """
        self._total_evaluations += 1
        result = ClusterGateResult(
            original_confidence=confidence,
            adjusted_confidence=confidence,
        )

        if self._analyzer is None:
            return result

        cluster = self._analyzer.classify_trade(trade)
        if cluster is None:
            return result

        result.cluster_matched = True
        result.cluster_id = cluster.cluster_id
        result.cluster_label = cluster.label
        result.cluster_win_rate = cluster.win_rate
        self._match_count += 1

        # Apply adjustment capped to bounds
        adj = cluster.confidence_adjustment
        adj = max(self._max_penalty, min(self._max_boost, adj))
        result.confidence_adjustment = adj
        result.adjusted_confidence = max(0.0, min(1.0, confidence + adj))

        if adj != 0.0:
            logger.info(
                "Phase 54 (US-337): Cluster gate — cluster=%d (%s, wr=%.3f) "
                "confidence %.3f → %.3f (%+.3f)",
                cluster.cluster_id, cluster.label, cluster.win_rate,
                confidence, result.adjusted_confidence, adj,
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        return {
            "total_evaluations": self._total_evaluations,
            "match_count": self._match_count,
            "match_rate": (
                self._match_count / self._total_evaluations
                if self._total_evaluations > 0 else 0.0
            ),
            "has_analyzer": self._analyzer is not None,
            "registry_size": (
                len(self._analyzer.get_cluster_registry())
                if self._analyzer is not None else 0
            ),
        }
