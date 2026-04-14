"""SupervisionCore — the main brain of the supervise-first loop.

Watches open positions, processes close events through the diagnostician,
applies adaptive policies, and fires scan triggers.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.scanner.automation.close_diagnostician import CloseDiagnosis, CloseDiagnostician, CloseReason
from src.scanner.automation.supervision_policy_engine import (
    AdaptivePolicy,
    PolicyType,
    SupervisionPolicyEngine,
)
from src.scanner.automation.trade_slots import SlotManager, TradeSlotState
from src.scanner.automation.trigger_registry import ScanTrigger, TriggerRegistry, TriggerType

_DEFAULT_CONFIG = {
    "max_concurrent": 3,
    "cooldown_seconds": 3600,
    "consecutive_loss_cooldown": 2,
    "supervision_tiers": {15: "tick", 120: "1m", 480: "5m"},
    "sweep_interval_minutes": 30.0,
}


class SupervisionCore:
    """Central supervisor that replaces the scan-first loop with supervise-first."""

    def __init__(
        self,
        slot_manager: SlotManager,
        diagnostician: CloseDiagnostician,
        policy_engine: SupervisionPolicyEngine,
        trigger_registry: TriggerRegistry,
        config: Optional[dict] = None,
    ) -> None:
        self.slot_manager = slot_manager
        self.diagnostician = diagnostician
        self.policy_engine = policy_engine
        self.trigger_registry = trigger_registry
        self.config: dict = {**_DEFAULT_CONFIG, **(config or {})}
        self._consecutive_losses: Dict[str, int] = {}

    # ── Public API ──────────────────────────────────────────────

    def process_close_event(self, trade_outcome: dict) -> CloseDiagnosis:
        """Diagnose a closed trade, adapt policies, and manage slot lifecycle."""
        diagnosis = self.diagnostician.diagnose(trade_outcome)
        pair = diagnosis.pair

        # Transition slot through diagnostic states
        self.slot_manager.transition(pair, TradeSlotState.DIAGNOSING)
        self.slot_manager.transition(pair, TradeSlotState.ADAPTING)

        # Track consecutive losses
        trade_won = diagnosis.realized_pl >= 0
        if trade_won:
            self._consecutive_losses[pair] = 0
        else:
            self._consecutive_losses[pair] = self._consecutive_losses.get(pair, 0) + 1

        # Apply adaptive response based on diagnosis
        self._apply_adaptive_response(diagnosis)

        # Determine terminal slot state
        if self._consecutive_losses.get(pair, 0) >= self.config["consecutive_loss_cooldown"]:
            # set_cooldown transitions to COOLDOWN internally
            self.slot_manager.set_cooldown(pair, self.config["cooldown_seconds"])
        else:
            self.slot_manager.transition(pair, TradeSlotState.IDLE)

        # Fire trigger so scanner knows a slot opened
        if self.slot_manager.get_state(pair) == TradeSlotState.IDLE:
            self.trigger_registry.fire(
                ScanTrigger(
                    trigger_type=TriggerType.SLOT_OPENED,
                    pairs=[pair],
                    source="supervision_core",
                )
            )

        return diagnosis

    def get_supervision_interval(self, trade_age_minutes: float) -> str:
        """Return the check frequency string from supervision_tiers based on trade age."""
        tiers = self.config["supervision_tiers"]
        # Tiers are keyed by minute thresholds — find the highest threshold <= trade_age
        result = "tick"  # default for youngest trades
        for threshold in sorted(tiers.keys()):
            if trade_age_minutes >= threshold:
                result = tiers[threshold]
        return result

    def tick(self) -> None:
        """Periodic housekeeping: expire policies, check cooldowns, sweep."""
        self.policy_engine.expire_stale()

        freed_pairs = self.slot_manager.check_cooldowns()
        for pair in freed_pairs:
            self.trigger_registry.fire(
                ScanTrigger(
                    trigger_type=TriggerType.COOLDOWN_EXPIRED,
                    pairs=[pair],
                    source="supervision_core.tick",
                )
            )

        if self.trigger_registry.should_sweep():
            self.trigger_registry.fire(
                ScanTrigger(
                    trigger_type=TriggerType.SCHEDULED_SWEEP,
                    source="supervision_core.tick",
                )
            )

    def get_status(self) -> dict:
        """Snapshot of supervision state."""
        return {
            "open_positions": len(self.slot_manager.open_positions()),
            "available_slots": self.slot_manager.available_slots(),
            "active_policies": len(self.policy_engine.get_active_policies()),
            "pending_triggers": len(self.trigger_registry.drain()) if self.trigger_registry.has_pending() else 0,
            "consecutive_losses": dict(self._consecutive_losses),
        }

    # ── Private ─────────────────────────────────────────────────

    def _apply_adaptive_response(self, diagnosis: CloseDiagnosis) -> None:
        """Add adaptive policies based on the diagnosis."""
        pair = diagnosis.pair
        trade_won = diagnosis.realized_pl >= 0
        now = time.time()
        consecutive = self._consecutive_losses.get(pair, 0)

        # Consecutive-loss cooldown policy
        if consecutive >= self.config["consecutive_loss_cooldown"]:
            self.policy_engine.add_policy(
                AdaptivePolicy(
                    policy_type=PolicyType.PAIR_COOLDOWN,
                    parameter="pair_block",
                    adjustment=0.0,
                    reason=f"{consecutive} consecutive losses on {pair}",
                    expires_at=now + self.config["cooldown_seconds"],
                    pair=pair,
                    source_diagnosis_ids=[diagnosis.trade_id],
                )
            )

        # Regime change on a loss — tighten gates
        if diagnosis.regime_changed and not trade_won:
            self.policy_engine.add_policy(
                AdaptivePolicy(
                    policy_type=PolicyType.GATE_TIGHTEN,
                    parameter="min_confidence",
                    adjustment=0.05,
                    reason=f"Regime changed during losing trade {diagnosis.trade_id}",
                    expires_at=now + 7200,  # 2 hours
                    pair=pair,
                    source_diagnosis_ids=[diagnosis.trade_id],
                )
            )

        # Quick TP hit on a win — widen TP
        if (
            trade_won
            and diagnosis.close_reason == CloseReason.TP_HIT
            and diagnosis.duration_minutes < 30
        ):
            self.policy_engine.add_policy(
                AdaptivePolicy(
                    policy_type=PolicyType.SIZING_ADJUST,
                    parameter="atr_tp_multiplier",
                    adjustment=0.1,
                    reason=f"Quick TP hit ({diagnosis.duration_minutes:.0f}m) on {pair}",
                    expires_at=now + 14400,  # 4 hours
                    pair=pair,
                    source_diagnosis_ids=[diagnosis.trade_id],
                )
            )
