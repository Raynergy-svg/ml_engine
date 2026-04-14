"""Adaptive supervision policy engine.

Manages time-decaying policy overrides that dynamically tighten gates,
pause pairs, or adjust sizing based on recent close diagnoses.

Separate from the Tier 7 PolicyEngine (action gating) — this handles
adaptive supervision responses.
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)


class PolicyType(Enum):
    PAIR_COOLDOWN = "pair_cooldown"
    GATE_TIGHTEN = "gate_tighten"
    GATE_LOOSEN = "gate_loosen"
    SIZING_ADJUST = "sizing_adjust"
    REGIME_OVERRIDE = "regime_override"
    ENTRY_BLOCK = "entry_block"


@dataclass
class AdaptivePolicy:
    policy_type: PolicyType
    parameter: str
    adjustment: float
    reason: str
    expires_at: float
    pair: Optional[str] = None
    source_diagnosis_ids: List[str] = field(default_factory=list)
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        d = asdict(self)
        d["policy_type"] = self.policy_type.value
        return d


class SupervisionPolicyEngine:
    """Manages active adaptive policies with time-based expiration."""

    def __init__(self):
        self._policies: List[AdaptivePolicy] = []
        logger.info("supervision_policy_engine.initialized")

    def add_policy(self, policy: AdaptivePolicy) -> None:
        self._policies.append(policy)
        logger.info(
            "supervision_policy.added",
            policy_id=policy.policy_id,
            policy_type=policy.policy_type.value,
            pair=policy.pair,
            parameter=policy.parameter,
            adjustment=policy.adjustment,
            reason=policy.reason,
            expires_at=policy.expires_at,
        )

    def get_active_policies(
        self,
        pair: Optional[str] = None,
        policy_type: Optional[PolicyType] = None,
    ) -> List[AdaptivePolicy]:
        now = time.time()
        result = []
        for p in self._policies:
            if now > p.expires_at:
                continue
            if pair is not None and p.pair is not None and p.pair != pair:
                continue
            if policy_type is not None and p.policy_type != policy_type:
                continue
            result.append(p)
        return result

    def get_effective_adjustment(self, pair: str, parameter: str) -> float:
        now = time.time()
        total = 0.0
        for p in self._policies:
            if now > p.expires_at:
                continue
            if p.parameter != parameter:
                continue
            if p.pair is not None and p.pair != pair:
                continue
            total += p.adjustment
        return total

    def expire_stale(self) -> int:
        now = time.time()
        kept = []
        removed = 0
        for p in self._policies:
            if now > p.expires_at:
                logger.info(
                    "supervision_policy.expired",
                    policy_id=p.policy_id,
                    policy_type=p.policy_type.value,
                    pair=p.pair,
                    parameter=p.parameter,
                )
                removed += 1
            else:
                kept.append(p)
        self._policies = kept
        return removed

    def has_active_block(self, pair: str) -> bool:
        now = time.time()
        for p in self._policies:
            if now > p.expires_at:
                continue
            if p.policy_type not in (PolicyType.ENTRY_BLOCK, PolicyType.PAIR_COOLDOWN):
                continue
            if p.pair is None or p.pair == pair:
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "active_policies": [p.to_dict() for p in self._policies if not p.is_expired()],
            "total_count": len(self._policies),
        }

    def clear(self) -> None:
        count = len(self._policies)
        self._policies = []
        logger.info("supervision_policy.cleared", removed=count)
