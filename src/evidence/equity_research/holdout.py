"""Operator-declared, single-use holdout evidence; never a holdout-data reader.

The authorized evaluator must durably open before accessing sealed data and
record the resulting hash afterward. Crashes leave OPENED evidence burned.
Dataset authenticity, sealed bytes/ACLs and evaluator-result attestation belong
to the later evaluator/deployment boundary; a supplied digest alone proves none.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from src.axiom2.data.temporal import _as_utc
from src.evidence.signing import Ed25519Signer
from src.evidence.store import EvidenceStore
from .experiment_registry import ExperimentRegistry, HoldoutRecord, ResearchEvent, ResearchState, _body, _digest, _text


def apply_holdout_event(
    event: ResearchEvent,
    digest: str,
    signer_key_id: str,
    data: dict,
    experiments: dict,
    candidates: dict,
    holdouts: dict,
) -> None:
    """Replay validated holdout transitions; called by the store's research reducer."""
    if event.kind == "HOLDOUT_DECLARED":
        _body(data, {"holdout_id", "dataset_digest", "coverage_start", "coverage_end"})
        identifier = _text(data["holdout_id"], "holdout_id")
        dataset = _digest(data["dataset_digest"], "dataset_digest")
        start = _as_utc(datetime.fromisoformat(data["coverage_start"]), "coverage_start")
        end = _as_utc(datetime.fromisoformat(data["coverage_end"]), "coverage_end")
        if end <= start:
            raise ValueError("holdout coverage_end must follow coverage_start")
        if identifier in holdouts:
            raise ValueError("holdout ID already declared")
        for previous in holdouts.values():
            if previous.dataset_digest == dataset or (start < previous.coverage_end and previous.coverage_start < end):
                raise ValueError("holdout dataset or coverage overlaps previously declared evidence")
        holdouts[identifier] = HoldoutRecord(identifier, dataset, start, end, digest, event.sequence)
        return
    expected = (
        {"candidate_id", "holdout_id", "attempt_count"}
        if event.kind == "HOLDOUT_OPENED"
        else {"candidate_id", "holdout_id", "result_hash"}
    )
    _body(data, expected)
    identifier, candidate_id = _text(data["holdout_id"], "holdout_id"), _digest(data["candidate_id"], "candidate_id")
    held, candidate = holdouts.get(identifier), candidates.get(candidate_id)
    if held is None or candidate is None:
        raise ValueError("a declared holdout and a frozen candidate are required")
    if experiments[candidate.experiment_id].failure_reason is not None:
        raise ValueError("failed candidate cannot use holdout evidence")
    if event.actor_id == candidate.actor_id or signer_key_id == candidate.signer_key_id:
        raise ValueError("holdout evaluator must be independent of candidate freezer")
    if held.declared_sequence >= candidate.freeze_sequence:
        raise ValueError("holdout must be declared before candidate freeze")
    if event.kind == "HOLDOUT_OPENED":
        if held.status != "UNTOUCHED":
            raise ValueError("holdout is no longer untouched")
        if any(value.candidate_id == candidate_id for value in holdouts.values()):
            raise ValueError("candidate already opened a final holdout")
        if type(data["attempt_count"]) is not int or data["attempt_count"] != len(experiments):
            raise ValueError("attempt_count does not include all registered attempts")
        holdouts[identifier] = replace(
            held,
            candidate_id=candidate_id,
            opened_digest=digest,
            opened_by=event.actor_id,
            attempt_count=len(experiments),
        )
    elif event.kind == "HOLDOUT_CONSUMED":
        if held.status == "CONSUMED":
            raise ValueError("holdout is already consumed")
        if held.status != "OPENED":
            raise ValueError("holdout must be durably opened before consumption")
        if held.candidate_id != candidate_id or held.opened_by != event.actor_id:
            raise ValueError("holdout reservation belongs to another candidate or evaluator")
        holdouts[identifier] = replace(
            held, result_hash=_digest(data["result_hash"], "result_hash"), consumption_digest=digest
        )
    else:
        raise ValueError("unknown holdout transition")


class HoldoutAuthority:
    """Control-plane capability, configured separately from a research producer."""

    def __init__(self, store: EvidenceStore, *, signer: Ed25519Signer, actor_id: str) -> None:
        self._registry = ExperimentRegistry(store, signer=signer, actor_id=actor_id)

    def get_holdout(self, holdout_id: str) -> HoldoutRecord:
        held = self._registry.snapshot().holdouts.get(_text(holdout_id, "holdout_id"))
        if held is None:
            raise ValueError("holdout must be explicitly declared")
        return held

    def assert_untouched(self, holdout_id: str) -> None:
        """Status check only, not a reservation; open_holdout is the atomic claim."""
        if self.get_holdout(holdout_id).status != "UNTOUCHED":
            raise ValueError("holdout is no longer untouched")

    def register_holdout(
        self, holdout_id: str, dataset_digest: str, coverage_start: datetime, coverage_end: datetime
    ) -> HoldoutRecord:
        identifier, dataset = _text(holdout_id, "holdout_id"), _digest(dataset_digest, "dataset_digest")
        start, end = _as_utc(coverage_start, "coverage_start"), _as_utc(coverage_end, "coverage_end")

        def build(state: ResearchState) -> dict | None:
            previous = state.holdouts.get(identifier)
            if previous is not None:
                if (previous.dataset_digest, previous.coverage_start, previous.coverage_end) != (dataset, start, end):
                    raise ValueError("declared holdout identity is immutable")
                return None
            return {
                "holdout_id": identifier,
                "dataset_digest": dataset,
                "coverage_start": start.isoformat(),
                "coverage_end": end.isoformat(),
            }

        return self._registry._record("HOLDOUT_DECLARED", build).holdouts[identifier]

    def open_holdout(self, candidate_id: str, holdout_id: str) -> HoldoutRecord:
        """Irreversibly mark used BEFORE evaluation; even exact retries fail closed."""
        candidate, identifier = _digest(candidate_id, "candidate_id"), _text(holdout_id, "holdout_id")
        return self._registry._record(
            "HOLDOUT_OPENED",
            lambda state: {"candidate_id": candidate, "holdout_id": identifier, "attempt_count": state.attempt_count},
        ).holdouts[identifier]

    def consume_holdout(self, candidate_id: str, holdout_id: str, result_hash: str) -> HoldoutRecord:
        """Attach one immutable result to the existing reservation; never reopens it."""
        candidate, identifier = _digest(candidate_id, "candidate_id"), _text(holdout_id, "holdout_id")
        result = _digest(result_hash, "result_hash")
        return self._registry._record(
            "HOLDOUT_CONSUMED", lambda _: {"candidate_id": candidate, "holdout_id": identifier, "result_hash": result}
        ).holdouts[identifier]


def consume_holdout(
    candidate_id: str, holdout_id: str, result_hash: str, *, authority: HoldoutAuthority
) -> HoldoutRecord:
    """Explicit-context entrypoint; no researcher-owned default signing authority."""
    return authority.consume_holdout(candidate_id, holdout_id, result_hash)
