"""Drift Proxy Guard — Phase 57 US-355.

Prevents the concept drift detector from reading penalized confidence
as its input proxy, which creates a self-reinforcing feedback loop.

The problem:
    engine.py concept drift proxies:
        prediction_error = abs(confidence - 0.5)   # distance from uncertainty
        agent_disagreement = 1.0 - confidence       # proxy for disagreement

    When confidence is already penalized by overconfidence (-3pt) or other
    systems, these proxies are computed from reduced confidence, not from
    the raw model signal. Example:
        raw confidence = 0.70  → prediction_error = 0.20, disagreement = 0.30
        after -3pt penalty: confidence = 0.67 (on 0-100 → 0.67 on 0-1 scale)
        But engine uses confidence in 0-100 scale before normalisation...

    Regardless of scale, the circular reference is: penalties reduce confidence
    → drift proxies see lower confidence → drift fires → multiplier further
    reduces confidence → next scan has even lower confidence proxies.

Solution:
    Capture pre_penalty_confidence BEFORE any penalty system runs.
    Pass this pre-penalty value to the drift detector update().
    The drift detector then measures drift in the raw signal, not in the
    penalty-corrected output.

Usage:
    guard = DriftProxyGuard()
    # Before any penalty is applied:
    pre_penalty_conf = confidence   # raw model output (0-1 scale)
    # ... apply overconfidence, ensemble disagreement, etc ...
    # Now update drift with the pre-penalty proxies:
    proxies = guard.compute_proxies(pre_penalty_conf, confidence)
    drift_detector.update(
        prediction_error=proxies.prediction_error,
        agent_disagreement=proxies.agent_disagreement,
        micro_anomaly_score=micro_score,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Threshold: if post-penalty confidence deviates more than this from
# pre-penalty, the reference is considered self-referential
SELF_REFERENCE_THRESHOLD = 0.10  # On 0-1 or proportional scale


@dataclass
class DriftProxies:
    """Computed drift proxy values."""
    prediction_error: float   # abs(confidence - 0.5)
    agent_disagreement: float  # 1.0 - confidence
    source_confidence: float   # which confidence value was used
    is_self_referential: bool  # True if penalty delta exceeded threshold


class DriftProxyGuard:
    """Computes concept drift proxies from pre-penalty confidence to break the feedback loop.

    The drift detector uses prediction_error and agent_disagreement as proxies
    for model performance. When these are computed from penalized confidence,
    normal penalty application looks like drift. This guard uses the pre-penalty
    raw confidence as the drift signal source.

    Usage:
        guard = DriftProxyGuard()
        proxies = guard.compute_proxies(pre_penalty_confidence=0.72, post_penalty_confidence=0.65)
        # Use proxies.prediction_error, proxies.agent_disagreement in drift update()
    """

    def __init__(self, self_reference_threshold: float = SELF_REFERENCE_THRESHOLD):
        self.self_reference_threshold = float(self_reference_threshold)
        self._total_checks: int = 0
        self._self_referential_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def compute_proxies(
        self,
        pre_penalty_confidence: float,
        post_penalty_confidence: float,
        pair: str = "",
    ) -> DriftProxies:
        """Compute drift proxies from pre-penalty confidence.

        Args:
            pre_penalty_confidence: Raw confidence BEFORE any penalty system ran (0-1 scale).
            post_penalty_confidence: Confidence AFTER all penalties (0-1 scale).
            pair: Pair name for logging.

        Returns:
            DriftProxies with prediction_error and agent_disagreement computed
            from pre_penalty_confidence to avoid the circular reference.
        """
        self._total_checks += 1
        pre = max(0.0, min(1.0, float(pre_penalty_confidence)))
        post = max(0.0, min(1.0, float(post_penalty_confidence)))

        is_self_ref = self.is_self_referential(pre, post)
        if is_self_ref:
            self._self_referential_count += 1
            logger.debug(
                "%s: DriftProxyGuard: self-referential detected pre=%.3f post=%.3f delta=%.3f — "
                "using pre-penalty confidence for drift proxies",
                pair or "?", pre, post, abs(pre - post),
            )

        # Always use pre-penalty confidence as drift proxy source
        prediction_error = abs(pre - 0.5)
        agent_disagreement = 1.0 - pre

        return DriftProxies(
            prediction_error=round(prediction_error, 6),
            agent_disagreement=round(agent_disagreement, 6),
            source_confidence=pre,
            is_self_referential=is_self_ref,
        )

    def is_self_referential(
        self,
        pre_penalty_confidence: float,
        post_penalty_confidence: float,
    ) -> bool:
        """Return True when penalty has significantly reduced confidence.

        If the gap between pre-penalty and post-penalty exceeds the threshold,
        using post-penalty confidence as the drift proxy would be self-referential.

        Args:
            pre_penalty_confidence: Pre-penalty confidence (0-1).
            post_penalty_confidence: Post-penalty confidence (0-1).

        Returns:
            True when abs(pre - post) > self_reference_threshold.
        """
        delta = abs(float(pre_penalty_confidence) - float(post_penalty_confidence))
        return delta > self.self_reference_threshold

    def get_stats(self) -> dict:
        """Return guard statistics."""
        rate = (
            self._self_referential_count / self._total_checks
            if self._total_checks > 0 else 0.0
        )
        return {
            "total_checks": self._total_checks,
            "self_referential_count": self._self_referential_count,
            "self_referential_rate": round(rate, 4),
            "self_reference_threshold": self.self_reference_threshold,
        }
