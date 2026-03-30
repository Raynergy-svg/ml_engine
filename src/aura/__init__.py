"""
Aura bridge — Buddy's consumer interface to the Aura/P90 system.

This package contains only the bridge protocol. Aura's brain lives in P90.
Buddy reads signals written by P90 and writes outcome signals back.

Bridge files (.aura/bridge/):
  readiness_signal.json   <- P90 writes, Buddy reads
  outcome_signal.json     <- Buddy writes, P90 reads
  active_rules.json       <- P90 writes, Buddy reads
  override_events.jsonl   <- Both append
"""

from src.aura.bridge.signals import FeedbackBridge, OutcomeSignal
from src.aura.bridge.rules_engine import BridgeRulesEngine

__all__ = ["FeedbackBridge", "OutcomeSignal", "BridgeRulesEngine"]
