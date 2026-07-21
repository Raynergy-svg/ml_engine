"""Canonical AXIOM episode ledger.

An episode is the durable unit that closes the Tier-7 loop:

    observed state -> AXIOM belief -> proposed/tool actions -> gate outcomes
    -> immediate post-mortem -> future training use.

This module is intentionally data-only. It does not execute tools, alter
controls, promote models, or make runtime decisions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.axiom_operator.session import utc_now

REPO_ROOT = Path(__file__).resolve().parents[2]
AXIOM_DIR = REPO_ROOT / "trained_data" / "axiom"
EPISODES_PATH = AXIOM_DIR / "episodes.jsonl"


@dataclass(frozen=True)
class EpisodeReward:
    immediate: float
    components: Dict[str, float] = field(default_factory=dict)
    pending: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AxiomEpisode:
    episode_id: str
    cycle_id: str
    ts: str
    actor: str
    source: str
    state: Dict[str, Any]
    belief: Dict[str, Any]
    tools_used: List[str]
    actions: List[Dict[str, Any]]
    safety: Dict[str, Any]
    outcome: Dict[str, Any]
    post_mortem: Dict[str, Any]
    reward: EpisodeReward
    next_training_use: str
    schema_version: str = "2026-07-09-v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _tool_names(observations: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    for name, payload in observations.items():
        if not isinstance(payload, Mapping):
            names.append(str(name))
            continue
        if payload.get("ok") is False:
            names.append(f"{name}:error")
        else:
            names.append(str(name))
    return sorted(names)


def _outcome_label(outcome: Mapping[str, Any], *, verified: bool) -> tuple[str, float]:
    if outcome.get("denied"):
        return "invalid_or_denied", -1.0
    if outcome.get("proposal"):
        return ("good_escalation", 0.5) if verified else ("unsafe_or_unverified_proposal", -1.0)
    if outcome.get("shadow"):
        return ("safe_shadow", 0.2) if verified else ("unverified_shadow", -0.2)
    if outcome.get("executed"):
        return ("safe_executed", 1.0) if verified else ("unsafe_execution", -1.0)
    return ("observe_only", 0.25) if verified else ("unverified_observe", -0.2)


def _training_use(*, verified: bool, labels: List[str], outcomes: List[Mapping[str, Any]]) -> str:
    if not verified:
        return "reject_until_verified"
    if any(outcome.get("proposal") for outcome in outcomes):
        return "needs_operator_disposition"
    if any(outcome.get("shadow") for outcome in outcomes):
        return "train_shadow_policy"
    if any(outcome.get("executed") for outcome in outcomes):
        return "train_runtime_policy"
    if labels == ["observe_only"] or not outcomes:
        return "train_observation_policy"
    return "review"


def episode_from_cycle(cycle: Mapping[str, Any]) -> AxiomEpisode:
    """Build one canonical episode from a resident-loop cycle row."""
    cycle_id = str(cycle.get("cycle_id") or "")
    if not cycle_id:
        cycle_id = f"cycle-missing-{abs(hash(json.dumps(cycle, sort_keys=True, default=str)))}"
    observations = _as_dict(cycle.get("observations"))
    diagnosis = _as_dict(cycle.get("diagnosis"))
    outcomes_raw = [o for o in _as_list(cycle.get("outcomes")) if isinstance(o, Mapping)]
    proposed_raw = [p for p in _as_list(cycle.get("proposed_actions")) if isinstance(p, Mapping)]
    verified = bool(cycle.get("verified"))

    labels: List[str] = []
    rewards: List[float] = []
    for outcome in outcomes_raw:
        label, reward = _outcome_label(outcome, verified=verified)
        labels.append(label)
        rewards.append(reward)
    if not outcomes_raw:
        label, reward = _outcome_label({}, verified=verified)
        labels.append(label)
        rewards.append(reward)

    counts = {
        "proposed": float(len(proposed_raw)),
        "outcomes": float(len(outcomes_raw)),
        "executed": float(sum(1 for o in outcomes_raw if o.get("executed"))),
        "shadow": float(sum(1 for o in outcomes_raw if o.get("shadow"))),
        "proposal": float(sum(1 for o in outcomes_raw if o.get("proposal"))),
        "denied": float(sum(1 for o in outcomes_raw if o.get("denied"))),
        "verified": 1.0 if verified else 0.0,
    }
    immediate = round(sum(rewards) / max(1, len(rewards)), 4)

    post_mortem = {
        "labels": labels,
        "summary": (
            "verified cycle"
            if verified
            else "cycle failed verification; do not train as positive behavior"
        ),
        "missing_closure": [
            item for item, missing in {
                "no_outcome": not outcomes_raw,
                "no_proposed_actions": not proposed_raw,
                "unverified": not verified,
            }.items() if missing
        ],
    }

    return AxiomEpisode(
        episode_id=f"axiom-episode-{cycle_id}",
        cycle_id=cycle_id,
        ts=str(cycle.get("ts") or utc_now()),
        actor=str(cycle.get("actor") or ""),
        source="resident_loop",
        state={
            "observations": observations,
            "autonomy_enabled": bool(cycle.get("autonomy_enabled")),
            "cli_available": bool(cycle.get("cli_available")),
        },
        belief={
            "reasoning": str(diagnosis.get("reasoning") or ""),
            "beliefs": str(diagnosis.get("beliefs") or ""),
            "raw": diagnosis,
        },
        tools_used=_tool_names(observations),
        actions=[dict(action) for action in proposed_raw],
        safety={
            "verified": verified,
            "verify_detail": _as_dict(cycle.get("verify_detail")),
            "outcome_counts": counts,
        },
        outcome={
            "immediate_actions": [dict(outcome) for outcome in outcomes_raw],
            "labels": labels,
            "closed": verified and bool(outcomes_raw),
            "needs_delayed_outcome": any(o.get("shadow") or o.get("proposal") for o in outcomes_raw),
        },
        post_mortem=post_mortem,
        reward=EpisodeReward(
            immediate=immediate,
            components=counts,
            pending=any(o.get("shadow") or o.get("proposal") for o in outcomes_raw),
            notes=list(post_mortem["missing_closure"]),
        ),
        next_training_use=_training_use(verified=verified, labels=labels, outcomes=outcomes_raw),
    )


def episode_to_dict(episode: AxiomEpisode) -> Dict[str, Any]:
    return asdict(episode)


def read_episodes(path: Path = EPISODES_PATH, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []
    if limit is not None:
        lines = lines[-int(limit):]
    rows: List[Dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _existing_episode_ids(path: Path) -> set[str]:
    return {str(row.get("episode_id")) for row in read_episodes(path)}


def append_episode(episode: AxiomEpisode, *, path: Path = EPISODES_PATH) -> bool:
    """Append an episode if its id has not already been written."""
    path = Path(path)
    if episode.episode_id in _existing_episode_ids(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(episode_to_dict(episode), sort_keys=True, default=str) + "\n")
    return True


def append_episode_from_cycle(cycle: Mapping[str, Any], *, path: Path = EPISODES_PATH) -> bool:
    return append_episode(episode_from_cycle(cycle), path=path)


def backfill_episodes_from_cycles(
    cycles: Iterable[Mapping[str, Any]], *, path: Path = EPISODES_PATH,
) -> Dict[str, int]:
    path = Path(path)
    existing = _existing_episode_ids(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with path.open("a", encoding="utf-8") as fh:
        for cycle in cycles:
            episode = episode_from_cycle(cycle)
            if episode.episode_id in existing:
                skipped += 1
                continue
            fh.write(json.dumps(episode_to_dict(episode), sort_keys=True, default=str) + "\n")
            existing.add(episode.episode_id)
            written += 1
    return {"written": written, "skipped_existing": skipped}


def closure_summary(*, episodes_path: Path = EPISODES_PATH) -> Dict[str, Any]:
    episodes = read_episodes(episodes_path)
    by_use: Dict[str, int] = {}
    closed = 0
    pending = 0
    rejected = 0
    for episode in episodes:
        use = str(episode.get("next_training_use") or "unknown")
        by_use[use] = by_use.get(use, 0) + 1
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        reward = episode.get("reward") if isinstance(episode.get("reward"), dict) else {}
        if outcome.get("closed"):
            closed += 1
        if reward.get("pending"):
            pending += 1
        if use.startswith("reject"):
            rejected += 1
    return {
        "episodes": len(episodes),
        "closed": closed,
        "pending_delayed_outcome": pending,
        "rejected_or_unverified": rejected,
        "by_training_use": by_use,
        "runtime_consumes_episode_model": False,
    }


__all__ = [
    "AxiomEpisode",
    "EPISODES_PATH",
    "EpisodeReward",
    "append_episode",
    "append_episode_from_cycle",
    "backfill_episodes_from_cycles",
    "closure_summary",
    "episode_from_cycle",
    "episode_to_dict",
    "read_episodes",
]
