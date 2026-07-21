"""Delayed outcome links for AXIOM episodes.

Episodes are append-only observations of what AXIOM saw, believed, and
proposed. Outcome links are append-only later evidence that closes a pending
branch, such as the operator accepting or denying an escalation proposal.

This module does not execute actions or mutate episode rows.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.agent_runtime.policy import compute_proposal_id
from src.agent_runtime.proposal_store import DISPOSITIONS_PATH, list_dispositions
from src.axiom_operator.session import utc_now
from src.axiom_training.episode_ledger import AXIOM_DIR, EPISODES_PATH, read_episodes

OUTCOME_LINKS_PATH = AXIOM_DIR / "episode_outcomes.jsonl"


@dataclass(frozen=True)
class EpisodeOutcomeLink:
    link_id: str
    episode_id: str
    cycle_id: str
    linked_at: str
    source: str
    reference_id: str
    outcome_type: str
    status: str
    reward_delta: float
    training_use: str
    detail: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2026-07-09-v1"


def _read_jsonl(path: Path, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
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


def read_outcome_links(path: Path = OUTCOME_LINKS_PATH, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    return _read_jsonl(Path(path), limit=limit)


def _existing_link_ids(path: Path) -> set[str]:
    return {str(row.get("link_id")) for row in read_outcome_links(path)}


def _stable_link_id(*, episode_id: str, source: str, reference_id: str, status: str, decided_at: str) -> str:
    payload = {
        "decided_at": decided_at,
        "episode_id": episode_id,
        "reference_id": reference_id,
        "source": source,
        "status": status,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return f"axiom-link-{digest}"


def _proposal_ref_from_mapping(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    action = str(payload.get("action") or "")
    detail = payload.get("detail") if isinstance(payload.get("detail"), Mapping) else {}
    params = payload.get("params") if isinstance(payload.get("params"), Mapping) else None
    if params is None:
        params = detail.get("params") if isinstance(detail.get("params"), Mapping) else {}
    if not action:
        action = str(detail.get("action") or "")
    if not action:
        return None
    proposal_id = str(payload.get("proposal_id") or detail.get("proposal_id") or "")
    if not proposal_id:
        proposal_id = compute_proposal_id(action, dict(params))
    return {
        "proposal_id": proposal_id,
        "action": action,
        "params": dict(params),
    }


def proposal_refs_from_episode(episode: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return escalation proposal references recoverable from an episode."""
    refs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    outcome = episode.get("outcome") if isinstance(episode.get("outcome"), Mapping) else {}
    immediate = outcome.get("immediate_actions") if isinstance(outcome.get("immediate_actions"), list) else []
    for row in immediate:
        if not isinstance(row, Mapping) or not row.get("proposal"):
            continue
        ref = _proposal_ref_from_mapping(row)
        if ref is None:
            continue
        key = str(ref["proposal_id"])
        if key not in seen:
            refs.append(ref)
            seen.add(key)

    actions = episode.get("actions") if isinstance(episode.get("actions"), list) else []
    for row in actions:
        if not isinstance(row, Mapping) or str(row.get("tier") or "").lower() != "escalation":
            continue
        ref = _proposal_ref_from_mapping(row)
        if ref is None:
            continue
        key = str(ref["proposal_id"])
        if key not in seen:
            refs.append(ref)
            seen.add(key)

    return refs


def _link_from_disposition(
    episode: Mapping[str, Any], ref: Mapping[str, Any], disposition: Mapping[str, Any],
) -> EpisodeOutcomeLink:
    status = str(disposition.get("status") or "unknown")
    decided_at = str(disposition.get("decided_at") or "")
    proposal_id = str(ref.get("proposal_id") or disposition.get("proposal_id") or "")
    accepted = status == "accepted"
    return EpisodeOutcomeLink(
        link_id=_stable_link_id(
            episode_id=str(episode.get("episode_id") or ""),
            source="proposal_disposition",
            reference_id=proposal_id,
            status=status,
            decided_at=decided_at,
        ),
        episode_id=str(episode.get("episode_id") or ""),
        cycle_id=str(episode.get("cycle_id") or ""),
        linked_at=utc_now(),
        source="proposal_disposition",
        reference_id=proposal_id,
        outcome_type="operator_accepted_proposal" if accepted else "operator_denied_proposal",
        status=status,
        reward_delta=1.0 if accepted else -0.75,
        training_use=(
            "train_escalation_that_operator_accepted"
            if accepted
            else "train_escalation_boundary"
        ),
        detail={
            "action": ref.get("action"),
            "params": ref.get("params") if isinstance(ref.get("params"), Mapping) else {},
            "disposition": dict(disposition),
        },
    )


def build_outcome_links(
    episodes: Iterable[Mapping[str, Any]],
    *,
    dispositions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    disposition_rows = dict(dispositions or {})
    links: List[EpisodeOutcomeLink] = []
    proposal_refs_seen = 0
    unresolved_refs = 0
    linked_episode_ids: set[str] = set()

    for episode in episodes:
        refs = proposal_refs_from_episode(episode)
        proposal_refs_seen += len(refs)
        for ref in refs:
            proposal_id = str(ref.get("proposal_id") or "")
            disposition = disposition_rows.get(proposal_id)
            if not isinstance(disposition, Mapping):
                unresolved_refs += 1
                continue
            link = _link_from_disposition(episode, ref, disposition)
            links.append(link)
            linked_episode_ids.add(str(episode.get("episode_id") or ""))

    return {
        "links": links,
        "linked_episode_ids": linked_episode_ids,
        "proposal_refs_seen": proposal_refs_seen,
        "unresolved_proposal_refs": unresolved_refs,
    }


def append_outcome_links(links: Iterable[EpisodeOutcomeLink], *, path: Path = OUTCOME_LINKS_PATH) -> Dict[str, int]:
    path = Path(path)
    existing = _existing_link_ids(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with path.open("a", encoding="utf-8") as fh:
        for link in links:
            if link.link_id in existing:
                skipped += 1
                continue
            fh.write(json.dumps(asdict(link), sort_keys=True, default=str) + "\n")
            existing.add(link.link_id)
            written += 1
    return {"written": written, "skipped_existing": skipped}


def link_delayed_outcomes(
    *,
    episodes_path: Path = EPISODES_PATH,
    outcome_links_path: Path = OUTCOME_LINKS_PATH,
    dispositions_path: Path = DISPOSITIONS_PATH,
    write: bool = True,
) -> Dict[str, Any]:
    episodes = read_episodes(episodes_path)
    dispositions = list_dispositions(path=dispositions_path)
    built = build_outcome_links(episodes, dispositions=dispositions)
    links = built["links"]
    append_result = append_outcome_links(links, path=outcome_links_path) if write else {
        "written": 0,
        "skipped_existing": 0,
    }
    existing_after = read_outcome_links(outcome_links_path) if Path(outcome_links_path).exists() else []
    linked_after = {str(row.get("episode_id")) for row in existing_after if row.get("episode_id")}
    if not write:
        linked_after = set(built["linked_episode_ids"])
    return {
        "episodes_seen": len(episodes),
        "dispositions_seen": len(dispositions),
        "proposal_refs_seen": int(built["proposal_refs_seen"]),
        "unresolved_proposal_refs": int(built["unresolved_proposal_refs"]),
        "candidate_links": len(links),
        "links_written": append_result["written"],
        "links_skipped_existing": append_result["skipped_existing"],
        "outcome_links_total": len(existing_after),
        "linked_episode_count": len(linked_after),
    }


def outcome_link_summary(*, outcome_links_path: Path = OUTCOME_LINKS_PATH) -> Dict[str, Any]:
    links = read_outcome_links(outcome_links_path)
    by_type: Dict[str, int] = {}
    by_training_use: Dict[str, int] = {}
    linked_episode_ids: set[str] = set()
    for link in links:
        linked_episode_ids.add(str(link.get("episode_id") or ""))
        outcome_type = str(link.get("outcome_type") or "unknown")
        training_use = str(link.get("training_use") or "unknown")
        by_type[outcome_type] = by_type.get(outcome_type, 0) + 1
        by_training_use[training_use] = by_training_use.get(training_use, 0) + 1
    linked_episode_ids.discard("")
    return {
        "links": len(links),
        "linked_episode_count": len(linked_episode_ids),
        "by_outcome_type": by_type,
        "by_training_use": by_training_use,
    }


__all__ = [
    "EpisodeOutcomeLink",
    "OUTCOME_LINKS_PATH",
    "append_outcome_links",
    "build_outcome_links",
    "link_delayed_outcomes",
    "outcome_link_summary",
    "proposal_refs_from_episode",
    "read_outcome_links",
]
