"""Deterministic Surgeon — translates PostTradeDiagnostics action list
into a ChangePackage config_delta when meta_manager_use_llm=False.

Rationale (Mythos audit 2026-04-28):
    With use_llm=False the production singleton wires a noop
    specialist_invoker — the LLM Code Surgeon's input arrives as the
    empty string and its YAML parser produces an empty config_delta.
    The ChangePackage stays empty, request_approval enqueues a black-
    hole proposal, and nothing ever actuates through the meta-pipeline.
    The orchestrator's direct SelfHeal path keeps actuating in
    parallel, but the meta-pipeline produces no provenance, no
    revertable change_id, and no Constitution attestation for the
    operator inbox.

    The Software Architect subagent's design recommended Option A:
    activate a deterministic translator at the same hook point the
    LLM Surgeon uses, so the pipeline finally produces actionable
    proposals under use_llm=False. Constitution + scorecard remain
    the governance gate — this surgeon is a translator, not an
    actuator. Option C (dual-write coordination with the
    orchestrator's SelfHeal path) is deliberately deferred to a
    follow-up PR.

Phase-1 verb support (CONFIG kind only):
    - reset_gate_threshold_to_default:<gate>
    - tighten_gate_threshold:<gate>
    - reduce_risk_per_trade_pct

Phase-2 verbs (deferred):
    - retrain_gates                  (MODEL kind — needs marker file)
    - soft_reset_agent_weight:<X>    (MODEL kind — agent_weights.json)
    - retrain_rl_position_sizer      (MODEL kind — retrain_request)

The deferred verbs are dropped silently with a list-entry so the
operator sees "this action wasn't translated" rather than thinking it
ran. StagedDeployer doesn't yet handle ChangeKind.MODEL deploys, so
emitting them as proposals would just queue no-op packages.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


# Probability-shaped thresholds in [0, 1] are clamped at this ceiling
# during tighten — same invariant the SelfHeal handler enforces.
_PROBABILITY_CLAMP: float = 1.0
_GATE_TIGHTEN_FACTOR: float = 1.10
_RISK_REDUCTION_MULTIPLIER: float = 0.75


class DeterministicSurgeon:
    """Translate PostTradeDiagnostics recommended_actions → config_delta.

    Pure-Python translator. No I/O beyond reading ScannerConfig
    dataclass defaults via the same introspection helper the LLM
    Surgeon path uses. ``translate()`` never raises — failures are
    captured in the returned ``dropped`` list so the caller can log
    them without the whole pipeline aborting.
    """

    GATE_TIGHTEN_FACTOR: float = _GATE_TIGHTEN_FACTOR
    RISK_REDUCTION_MULTIPLIER: float = _RISK_REDUCTION_MULTIPLIER
    PROBABILITY_CLAMP: float = _PROBABILITY_CLAMP

    # Diagnostic's gate_firing_rate check emits *_passed names from the
    # journal schema; ScannerConfig stores threshold field names. The
    # SelfHeal handler maps them implicitly via its YAML paths_by_gate
    # table — we mirror that here so the same diagnostic-emitted action
    # produces the same field mutation regardless of which surgeon path
    # runs (LLM or deterministic).
    GATE_NAME_ALIASES: Dict[str, str] = {
        "momentum_passed": "min_momentum",
        "confidence_passed": "min_confidence",
        "risk_passed": "min_risk_reward_ratio",
    }

    # Phase-2 deferred verbs — recognized so we can record them in
    # `dropped` rather than treating them as "unknown". When MODEL-kind
    # support lands, move these out of the deferred set into per-verb
    # handlers.
    _DEFERRED_VERBS: Set[str] = frozenset({
        "retrain_gates",
        "soft_reset_agent_weight",
        "retrain_rl_position_sizer",
    })

    def __init__(self, *, valid_config_keys: Optional[Set[str]] = None) -> None:
        """Optional valid_config_keys override for tests; production
        path uses dataclass introspection."""
        self._valid_keys = valid_config_keys

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def translate(
        self, incident: Any
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Map an incident's diagnostic actions to a config_delta.

        Returns ``(delta, dropped)`` where ``delta`` is a dict of
        ScannerConfig field → new value, and ``dropped`` is the list
        of action strings that produced no delta (unknown verbs,
        malformed args, orphan keys, deferred MODEL-kind verbs).

        Never raises.
        """
        actions = self._extract_actions(incident)
        if not actions:
            return {}, []

        delta: Dict[str, Any] = {}
        dropped: List[str] = []

        for action in actions:
            try:
                pair = self._dispatch(action)
            except Exception as err:  # noqa: BLE001 — translator must not raise
                logger.warning(
                    "deterministic_surgeon.dispatch_failed action=%r err=%r",
                    action, err,
                )
                dropped.append(action)
                continue

            if pair is None:
                dropped.append(action)
                continue

            key, value = pair
            if not self._is_valid_config_key(key):
                logger.debug(
                    "deterministic_surgeon.orphan_key action=%r key=%s",
                    action, key,
                )
                dropped.append(action)
                continue

            delta[key] = value

        return delta, dropped

    # ──────────────────────────────────────────────────────────────
    # Verb dispatch
    # ──────────────────────────────────────────────────────────────

    def _dispatch(self, action: str) -> Optional[Tuple[str, float]]:
        """Parse one action and return (key, value) or None."""
        if not isinstance(action, str):
            return None
        verb, sep, arg = action.partition(":")
        verb = verb.strip()
        arg = arg.strip()

        if verb in self._DEFERRED_VERBS:
            # Recognized but Phase-2 — return None so caller records
            # the action in `dropped`.
            return None

        if verb == "reset_gate_threshold_to_default":
            return self._handle_reset(arg)
        if verb == "tighten_gate_threshold":
            return self._handle_tighten(arg)
        if verb == "reduce_risk_per_trade_pct":
            # No argument expected.
            return self._handle_reduce_risk()
        return None

    def _handle_reset(self, gate: str) -> Optional[Tuple[str, float]]:
        if not gate:
            return None
        gate = self.GATE_NAME_ALIASES.get(gate, gate)
        default = self._scanner_config_default(gate)
        if default is None:
            return None
        return (gate, default)

    def _handle_tighten(self, gate: str) -> Optional[Tuple[str, float]]:
        if not gate:
            return None
        gate = self.GATE_NAME_ALIASES.get(gate, gate)
        current = self._scanner_config_default(gate)
        if current is None:
            return None
        new_value = self._tighten_value(
            current=current,
            is_probability=(0.0 < current <= 1.0),
        )
        return (gate, new_value)

    def _handle_reduce_risk(self) -> Optional[Tuple[str, float]]:
        current = self._scanner_config_default("risk_per_trade_pct")
        if current is None:
            return None
        return ("risk_per_trade_pct", float(current) * self.RISK_REDUCTION_MULTIPLIER)

    @classmethod
    def _tighten_value(cls, *, current: float, is_probability: bool) -> float:
        """Apply tighten factor, clamping probability-shaped values at 1.0."""
        new_value = float(current) * cls.GATE_TIGHTEN_FACTOR
        if is_probability and new_value > cls.PROBABILITY_CLAMP:
            return cls.PROBABILITY_CLAMP
        return new_value

    # ──────────────────────────────────────────────────────────────
    # ScannerConfig introspection (mirrors meta_manager._validate_config_delta)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _scanner_config_default(field: str) -> Optional[float]:
        """Return the ScannerConfig dataclass default for ``field`` or None.

        Mirrors SelfHeal._scanner_config_default — keep these two in
        sync so the deterministic-surgeon-emitted delta matches the
        orchestrator's direct SelfHeal write for the same verb.
        """
        try:
            from src.scanner.config import ScannerConfig  # noqa: WPS433
            cfg = ScannerConfig()
        except Exception as err:
            logger.debug(
                "deterministic_surgeon.scanner_config_import_failed err=%r", err,
            )
            return None
        if not hasattr(cfg, field):
            return None
        try:
            return float(getattr(cfg, field))
        except (TypeError, ValueError):
            return None

    def _is_valid_config_key(self, key: str) -> bool:
        if self._valid_keys is not None:
            return key in self._valid_keys
        try:
            import dataclasses  # noqa: WPS433
            from src.scanner.config import ScannerConfig  # noqa: WPS433
            valid = {f.name for f in dataclasses.fields(ScannerConfig)}
            return key in valid
        except Exception as err:
            logger.debug(
                "deterministic_surgeon.config_keys_introspection_failed err=%r", err,
            )
            # Fail-OPEN like _validate_config_delta — better to let the
            # delta through and have Constitution catch it than to drop
            # a legitimate change because introspection broke.
            return True

    # ──────────────────────────────────────────────────────────────
    # Incident-shape helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_actions(incident: Any) -> List[str]:
        """Pull recommended_actions from incident["diag"] safely."""
        if not isinstance(incident, dict):
            return []
        diag = incident.get("diag")
        if not isinstance(diag, dict):
            return []
        actions = diag.get("recommended_actions")
        if not isinstance(actions, list):
            return []
        return [
            a.strip() for a in actions
            if isinstance(a, str) and a.strip()
        ]


__all__ = ["DeterministicSurgeon"]
