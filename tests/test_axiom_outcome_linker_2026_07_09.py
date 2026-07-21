from __future__ import annotations

import json

from src.agent_runtime.policy import compute_proposal_id
from src.axiom_training.outcome_linker import (
    build_outcome_links,
    link_delayed_outcomes,
    outcome_link_summary,
    proposal_refs_from_episode,
    read_outcome_links,
)


def _episode(**overrides):
    action = "change_strategy_or_code"
    params = {"target": "trained_data/models/agent_weights.json", "change": "remove orphan key"}
    base = {
        "episode_id": "axiom-episode-cycle-link-test",
        "cycle_id": "cycle-link-test",
        "actions": [
            {
                "action": action,
                "tier": "escalation",
                "params": params,
                "rationale": "operator gated cleanup",
            }
        ],
        "outcome": {
            "immediate_actions": [
                {
                    "action": action,
                    "tier": "escalation",
                    "proposal": True,
                    "shadow": False,
                    "executed": False,
                    "denied": False,
                    "detail": {"action": action, "params": params},
                }
            ],
            "needs_delayed_outcome": True,
        },
        "reward": {"pending": True},
        "next_training_use": "needs_operator_disposition",
    }
    base.update(overrides)
    return base


def test_proposal_refs_reconstruct_stable_proposal_id() -> None:
    episode = _episode()
    ref = proposal_refs_from_episode(episode)[0]

    assert ref["proposal_id"] == compute_proposal_id(ref["action"], ref["params"])
    assert ref["action"] == "change_strategy_or_code"


def test_link_delayed_outcomes_appends_operator_disposition_idempotently(tmp_path) -> None:
    episodes_path = tmp_path / "episodes.jsonl"
    links_path = tmp_path / "episode_outcomes.jsonl"
    dispositions_path = tmp_path / "proposal_dispositions.json"
    episode = _episode()
    proposal_id = proposal_refs_from_episode(episode)[0]["proposal_id"]
    episodes_path.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")
    dispositions_path.write_text(
        json.dumps({
            proposal_id: {
                "proposal_id": proposal_id,
                "status": "denied",
                "actor": "test-operator",
                "reason": "operator denied via Activity panel",
                "detail": {"action": "change_strategy_or_code"},
                "decided_at": "2026-07-09T16:00:00+00:00",
            }
        }),
        encoding="utf-8",
    )

    first = link_delayed_outcomes(
        episodes_path=episodes_path,
        outcome_links_path=links_path,
        dispositions_path=dispositions_path,
    )
    second = link_delayed_outcomes(
        episodes_path=episodes_path,
        outcome_links_path=links_path,
        dispositions_path=dispositions_path,
    )
    rows = read_outcome_links(links_path)

    assert first["candidate_links"] == 1
    assert first["links_written"] == 1
    assert second["links_written"] == 0
    assert second["links_skipped_existing"] == 1
    assert len(rows) == 1
    assert rows[0]["outcome_type"] == "operator_denied_proposal"
    assert rows[0]["training_use"] == "train_escalation_boundary"
    assert rows[0]["reference_id"] == proposal_id
    assert outcome_link_summary(outcome_links_path=links_path)["linked_episode_count"] == 1


def test_build_outcome_links_noops_when_shadow_episode_has_no_proposal() -> None:
    built = build_outcome_links([
        _episode(
            actions=[{"action": "gate_health", "tier": "operational", "params": {}}],
            outcome={
                "immediate_actions": [
                    {"action": "gate_health", "proposal": False, "shadow": True, "executed": False}
                ],
                "needs_delayed_outcome": True,
            },
            next_training_use="train_shadow_policy",
        )
    ], dispositions={})

    assert built["proposal_refs_seen"] == 0
    assert built["unresolved_proposal_refs"] == 0
    assert built["links"] == []
