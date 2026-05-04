"""Mythos audit 2026-04-28 — Deterministic Surgeon (Phase 1 / Option A).

Background: when meta_manager_use_llm=False the production singleton wires
the Code Surgeon's specialist_invoker to a noop lambda (returns ""). The
LLM-driven proposer then builds an empty config_delta, the ChangePackage
stays empty, and the meta-pipeline is a black hole — every incident
reaches AWAITING_APPROVAL with nothing actionable. Operator complaint:
"buddy is still invoking LLM for self-healing and not using the
cybernetic meta — there is a pipeline bug with meta_manager [maybe]."

Software Architect's design: a deterministic translator that maps
PostTradeDiagnostics.recommended_actions → config_delta when use_llm is
False, so the meta-pipeline produces a real Proposal that flows through
the unchanged Auditor + Constitution + scorecard chain. The Constitution
remains the source-of-truth governance gate — the deterministic surgeon
is a translator, not an actuator.

Phase 1 scope: 3 CONFIG-kind verbs (reset/tighten/reduce_risk). The 3
MODEL-kind verbs (retrain_gates / soft_reset_agent_weight /
retrain_rl_position_sizer) are deferred — StagedDeployer doesn't yet
handle ChangeKind.MODEL deploys, so emitting them would just queue
no-op proposals.
"""
from __future__ import annotations

import pytest

from src.scanner.automation.deterministic_surgeon import DeterministicSurgeon
from src.scanner.config import ScannerConfig


# Trap-recovery gate names from PostTradeDiagnostics._QUIET_STREAK_RECOVERY_ORDER.
# Each is a real ScannerConfig field; the surgeon must reset to dataclass default.
_TRAP_GATES = (
    "min_confidence",
    "weighted_vote_threshold",
    "max_model_disagreement",
    "max_uncertainty_score",
    "atr_sl_multiplier",
    "min_momentum",
    "min_risk_reward_ratio",
)


def _diag_with_actions(*actions: str) -> dict:
    """Wrap an action list in the incident envelope cycle_autonomy uses."""
    return {
        "kind": "self_heal",
        "diag": {
            "status": "CRITICAL",
            "issues": [{"check": "quiet_streak", "severity": "CRITICAL"}],
            "recommended_actions": list(actions),
        },
    }


# ────────────────────────────────────────────────────────────────────
# T-1: reset verb resolves every trap-recovery gate
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("gate", _TRAP_GATES)
def test_T1_reset_resolves_every_trap_recovery_gate(gate):
    surgeon = DeterministicSurgeon()
    expected = getattr(ScannerConfig(), gate)

    delta, dropped = surgeon.translate(
        _diag_with_actions(f"reset_gate_threshold_to_default:{gate}")
    )
    assert dropped == [], f"Unexpected drops for gate={gate}: {dropped}"
    assert gate in delta
    assert delta[gate] == pytest.approx(expected)


# ────────────────────────────────────────────────────────────────────
# T-2: tighten verb resolves via gate-name aliases + clamps probability
# ────────────────────────────────────────────────────────────────────


def test_T2_tighten_aliases_passed_name_to_threshold_field():
    """Diagnostic's gate_firing_rate check emits 'momentum_passed' /
    'confidence_passed' / 'risk_passed'. Surgeon must alias to the real
    ScannerConfig field names (min_momentum / min_confidence /
    min_risk_reward_ratio)."""
    surgeon = DeterministicSurgeon()
    cfg = ScannerConfig()

    delta, dropped = surgeon.translate(
        _diag_with_actions("tighten_gate_threshold:momentum_passed")
    )
    assert dropped == []
    assert "min_momentum" in delta
    assert delta["min_momentum"] == pytest.approx(cfg.min_momentum * 1.10)


def test_T2b_tighten_clamps_probability_shaped_thresholds_to_one():
    """Probability-shaped thresholds in (0, 1] must NOT exceed 1.0 after
    tightening."""
    surgeon = DeterministicSurgeon()
    # max_uncertainty_score default is 0.4; * 1.10 = 0.44 — still in (0,1]
    # so we hand-craft a threshold that would overshoot. Use a custom
    # current value via the public probe.
    new = surgeon._tighten_value(current=0.95, is_probability=True)
    assert new == pytest.approx(1.0)
    new2 = surgeon._tighten_value(current=0.5, is_probability=True)
    assert new2 == pytest.approx(0.55)


# ────────────────────────────────────────────────────────────────────
# T-3: reduce_risk verb scales risk_per_trade_pct by 0.75
# ────────────────────────────────────────────────────────────────────


def test_T3_reduce_risk_scales_by_multiplier():
    surgeon = DeterministicSurgeon()
    cfg = ScannerConfig()
    delta, dropped = surgeon.translate(_diag_with_actions("reduce_risk_per_trade_pct"))
    assert dropped == []
    assert "risk_per_trade_pct" in delta
    assert delta["risk_per_trade_pct"] == pytest.approx(cfg.risk_per_trade_pct * 0.75)


# ────────────────────────────────────────────────────────────────────
# T-7/8/9: malformed / unknown / orphan verbs are dropped, not raised
# ────────────────────────────────────────────────────────────────────


def test_T7_unknown_verb_is_dropped_not_raised():
    surgeon = DeterministicSurgeon()
    delta, dropped = surgeon.translate(_diag_with_actions("do_something_weird:foo"))
    assert delta == {}
    assert "do_something_weird:foo" in dropped


def test_T8_malformed_argument_is_dropped():
    surgeon = DeterministicSurgeon()
    delta, dropped = surgeon.translate(_diag_with_actions("tighten_gate_threshold:"))
    assert delta == {}
    assert "tighten_gate_threshold:" in dropped


def test_T9_orphan_scanner_config_field_is_dropped():
    """Re-uses the same orphan-key safety the LLM Surgeon path uses
    (the 2026-04-16 $3,527 bug pattern)."""
    surgeon = DeterministicSurgeon()
    delta, dropped = surgeon.translate(
        _diag_with_actions("reset_gate_threshold_to_default:not_a_real_field_xyz")
    )
    assert delta == {}
    assert "reset_gate_threshold_to_default:not_a_real_field_xyz" in dropped


# ────────────────────────────────────────────────────────────────────
# T-10/11: defensive on missing or empty diag
# ────────────────────────────────────────────────────────────────────


def test_T10_missing_diag_returns_empty():
    surgeon = DeterministicSurgeon()
    delta, dropped = surgeon.translate({"kind": "self_heal"})  # no diag
    assert delta == {}
    assert dropped == []


def test_T11_empty_recommended_actions_returns_empty():
    surgeon = DeterministicSurgeon()
    delta, dropped = surgeon.translate(
        {"kind": "self_heal", "diag": {"recommended_actions": []}}
    )
    assert delta == {}
    assert dropped == []


def test_T11b_non_dict_incident_returns_empty():
    surgeon = DeterministicSurgeon()
    delta, dropped = surgeon.translate("not a dict")  # type: ignore[arg-type]
    assert delta == {}
    assert dropped == []


# ────────────────────────────────────────────────────────────────────
# T-12: mixed supported + unsupported verbs — supported ones still emit
# ────────────────────────────────────────────────────────────────────


def test_T12_mixed_actions_emit_partial_delta_with_drops():
    surgeon = DeterministicSurgeon()
    cfg = ScannerConfig()
    delta, dropped = surgeon.translate(
        _diag_with_actions(
            "reset_gate_threshold_to_default:min_confidence",
            "soft_reset_agent_weight:trend",  # MODEL kind — deferred to Phase 2
            "reduce_risk_per_trade_pct",
            "garbage_verb",
        )
    )
    assert delta["min_confidence"] == pytest.approx(cfg.min_confidence)
    assert delta["risk_per_trade_pct"] == pytest.approx(cfg.risk_per_trade_pct * 0.75)
    assert "soft_reset_agent_weight:trend" in dropped
    assert "garbage_verb" in dropped


# ────────────────────────────────────────────────────────────────────
# T-Integration: MetaManager(use_llm=False) produces non-empty proposal
# end-to-end via the production proposer hook.
# ────────────────────────────────────────────────────────────────────


def test_I1_meta_manager_use_llm_false_produces_real_proposal(tmp_path):
    """The whole point: with use_llm=False, the meta-pipeline must
    produce an actionable Proposal instead of a black-hole empty one.
    """
    from src.scanner.automation.meta_manager import MetaManager

    mgr = MetaManager(
        use_llm=False,
        specialist_invoker=lambda mode, prompt: "",
        changes_dir=tmp_path / "changes",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    incident = _diag_with_actions(
        "reset_gate_threshold_to_default:min_confidence",
        "reduce_risk_per_trade_pct",
    )

    pkg = mgr.intake(incident)
    pkg = mgr.propose(pkg)

    assert pkg.proposal is not None
    assert pkg.proposal.config_delta, (
        f"Proposal config_delta is empty under use_llm=False — meta is "
        f"still a black hole. proposal={pkg.proposal!r}"
    )
    assert "min_confidence" in pkg.proposal.config_delta
    assert "risk_per_trade_pct" in pkg.proposal.config_delta


def test_I5_meta_manager_use_llm_True_explicit_opt_in_path_unchanged(tmp_path):
    """Operator-explicit use_llm=True opt-in path: LLM specialist invoker
    returns empty → empty delta (deterministic surgeon does NOT step in).

    Mythos audit 2026-05-04 — the prior MetaManager default was use_llm=True
    which let any direct construction silently fire Claude. Default flipped
    to False (matches CLAUDE.md "runtime is Claude-free"). This test now
    asserts the OPT-IN path: when an operator explicitly passes
    use_llm=True, the LLM-only contract is preserved (empty invoker output
    → empty config_delta — surgeon does NOT compensate).
    """
    from src.scanner.automation.meta_manager import MetaManager

    mgr = MetaManager(
        use_llm=True,  # Operator-explicit opt-in (no longer the default).
        specialist_invoker=lambda mode, prompt: "",
        changes_dir=tmp_path / "changes",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    incident = _diag_with_actions(
        "reset_gate_threshold_to_default:min_confidence"
    )

    pkg = mgr.intake(incident)
    pkg = mgr.propose(pkg)

    # use_llm=True path: LLM specialist invoker returns "" → empty delta.
    # Deterministic surgeon must NOT activate when use_llm=True.
    assert pkg.proposal is not None
    assert pkg.proposal.config_delta == {}, (
        "Deterministic surgeon activated under explicit use_llm=True — "
        "the LLM-only contract was broken."
    )
