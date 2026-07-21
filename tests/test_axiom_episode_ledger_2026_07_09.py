from __future__ import annotations

import json

from src.axiom_training.episode_ledger import (
    append_episode_from_cycle,
    backfill_episodes_from_cycles,
    closure_summary,
    episode_from_cycle,
    read_episodes,
)


def _cycle(**overrides):
    base = {
        "cycle_id": "cycle-episode-test",
        "ts": "2026-07-09T12:00:00+00:00",
        "actor": "test-agent",
        "cli_available": True,
        "autonomy_enabled": False,
        "observations": {
            "health_check": {"ok": True, "data": {"running": True}},
            "trade_feedback": {"ok": True, "data": {"count": 3}},
            "shadow_ledger:crypto_momentum": {"ok": False, "error": "missing"},
        },
        "diagnosis": {
            "beliefs": "trend lane alive; crypto shadow missing",
            "reasoning": "Observe, then keep shadow lane gated.",
        },
        "proposed_actions": [
            {"action": "keep_shadow", "params": {}, "rationale": "not verified"}
        ],
        "outcomes": [
            {
                "action": "keep_shadow",
                "tier": "operational",
                "executed": False,
                "shadow": True,
                "proposal": False,
                "denied": False,
            }
        ],
        "verified": True,
        "verify_detail": {"practice_pin_ok": True},
    }
    base.update(overrides)
    return base


def test_episode_from_cycle_preserves_state_belief_actions_and_post_mortem() -> None:
    episode = episode_from_cycle(_cycle())

    assert episode.episode_id == "axiom-episode-cycle-episode-test"
    assert episode.state["observations"]["health_check"]["data"]["running"] is True
    assert episode.belief["beliefs"] == "trend lane alive; crypto shadow missing"
    assert episode.tools_used == ["health_check", "shadow_ledger:crypto_momentum:error", "trade_feedback"]
    assert episode.actions[0]["action"] == "keep_shadow"
    assert episode.outcome["labels"] == ["safe_shadow"]
    assert episode.reward.immediate == 0.2
    assert episode.reward.pending is True
    assert episode.next_training_use == "train_shadow_policy"


def test_append_episode_is_idempotent_by_cycle_id(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"

    assert append_episode_from_cycle(_cycle(), path=path) is True
    assert append_episode_from_cycle(_cycle(), path=path) is False

    rows = read_episodes(path)
    assert len(rows) == 1
    assert rows[0]["cycle_id"] == "cycle-episode-test"


def test_backfill_and_closure_summary_report_open_runtime_loop(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"
    cycles = [
        _cycle(cycle_id="cycle-a"),
        _cycle(
            cycle_id="cycle-b",
            outcomes=[],
            proposed_actions=[],
            diagnosis={"beliefs": "nothing to do", "reasoning": "observe only"},
        ),
        _cycle(cycle_id="cycle-c", verified=False),
    ]

    result = backfill_episodes_from_cycles(cycles, path=path)
    summary = closure_summary(episodes_path=path)

    assert result == {"written": 3, "skipped_existing": 0}
    assert summary["episodes"] == 3
    assert summary["closed"] == 1
    assert summary["pending_delayed_outcome"] == 2
    assert summary["rejected_or_unverified"] == 1
    assert summary["runtime_consumes_episode_model"] is False

    # Stays JSONL-serializable for shell tooling and future offline trainers.
    for line in path.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["schema_version"] == "2026-07-09-v1"
