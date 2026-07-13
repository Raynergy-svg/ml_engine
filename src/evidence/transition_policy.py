"""Fail-closed disposition transitions and transition authority."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .contracts import AuthorityRole, DispositionEvent, DispositionState


class TransitionPolicyError(ValueError):
    """Raised when state, actor, signer, or authority violates local policy."""


_TRANSITION_AUTHORITIES: dict[
    tuple[DispositionState | None, DispositionState], frozenset[AuthorityRole]
] = {
    (None, DispositionState.CREATED): frozenset({AuthorityRole.PRODUCER}),
    (DispositionState.CREATED, DispositionState.RECEIVED): frozenset({AuthorityRole.LOCAL_IMPORTER}),
    (DispositionState.RECEIVED, DispositionState.HASH_VERIFIED): frozenset({AuthorityRole.LOCAL_IMPORTER}),
    (DispositionState.HASH_VERIFIED, DispositionState.POLICY_VERIFIED): frozenset(
        {AuthorityRole.LOCAL_IMPORTER}
    ),
    (DispositionState.POLICY_VERIFIED, DispositionState.METRIC_REPLAYED): frozenset(
        {AuthorityRole.INDEPENDENT_VERIFIER}
    ),
    (DispositionState.METRIC_REPLAYED, DispositionState.QUARANTINED): frozenset(
        {AuthorityRole.LOCAL_IMPORTER}
    ),
    (DispositionState.QUARANTINED, DispositionState.SHADOW): frozenset({AuthorityRole.OPERATOR}),
    (DispositionState.SHADOW, DispositionState.OPERATOR_APPROVED): frozenset({AuthorityRole.OPERATOR}),
    (DispositionState.OPERATOR_APPROVED, DispositionState.CHAMPION): frozenset(
        {AuthorityRole.PROMOTION_SERVICE}
    ),
    (DispositionState.QUARANTINED, DispositionState.RETIRED): frozenset({AuthorityRole.OPERATOR}),
    (DispositionState.SHADOW, DispositionState.RETIRED): frozenset({AuthorityRole.OPERATOR}),
    (DispositionState.OPERATOR_APPROVED, DispositionState.RETIRED): frozenset({AuthorityRole.OPERATOR}),
    (DispositionState.CHAMPION, DispositionState.RETIRED): frozenset(
        {AuthorityRole.OPERATOR, AuthorityRole.SAFETY_MONITOR}
    ),
}

for _state in (
    DispositionState.RECEIVED,
    DispositionState.HASH_VERIFIED,
    DispositionState.POLICY_VERIFIED,
    DispositionState.METRIC_REPLAYED,
):
    _TRANSITION_AUTHORITIES[(_state, DispositionState.REJECTED)] = frozenset(
        {AuthorityRole.LOCAL_IMPORTER}
    )


@dataclass
class AuthorityRegistry:
    """Bind declared actor roles to trusted signer key IDs.

    Multiple keys may overlap for one actor during rotation. Merely writing a
    privileged role into an event never grants that role.
    """

    _bindings: dict[tuple[str, AuthorityRole], set[str]] = field(default_factory=dict)

    def register(self, *, actor_id: str, role: AuthorityRole, key_ids: Iterable[str]) -> None:
        keys = set(key_ids)
        if not actor_id or not keys or any(not key for key in keys):
            raise ValueError("actor_id and at least one non-empty key_id are required")
        self._bindings.setdefault((actor_id, role), set()).update(keys)

    def revoke(self, *, actor_id: str, role: AuthorityRole, key_id: str) -> None:
        binding = self._bindings.get((actor_id, role))
        if binding is None or key_id not in binding:
            raise KeyError((actor_id, role, key_id))
        binding.remove(key_id)

    def authorize(self, event: DispositionEvent, envelope_signer_key_id: str) -> None:
        if event.signer_key_id != envelope_signer_key_id:
            raise TransitionPolicyError("event signer_key_id does not match envelope signer")
        self.authorize_identity(
            actor_id=event.actor_id,
            role=event.authority,
            key_id=event.signer_key_id,
        )

    def authorize_identity(self, *, actor_id: str, role: AuthorityRole, key_id: str) -> None:
        permitted_keys = self._bindings.get((actor_id, role), set())
        if key_id not in permitted_keys:
            raise TransitionPolicyError(
                f"signer {key_id!r} is not registered for "
                f"{actor_id!r} as {role.value!r}"
            )


def allowed_authorities(
    from_state: DispositionState | None,
    to_state: DispositionState,
) -> frozenset[AuthorityRole]:
    return _TRANSITION_AUTHORITIES.get((from_state, to_state), frozenset())


def enforce_transition(event: DispositionEvent, authorities: AuthorityRegistry, signer_key_id: str) -> None:
    permitted = allowed_authorities(event.from_state, event.to_state)
    if not permitted:
        raise TransitionPolicyError(
            f"transition {event.from_state!s} -> {event.to_state.value} is forbidden"
        )
    if event.authority not in permitted:
        roles = ", ".join(sorted(role.value for role in permitted))
        raise TransitionPolicyError(
            f"{event.authority.value} cannot authorize {event.from_state!s} -> "
            f"{event.to_state.value}; required one of: {roles}"
        )
    authorities.authorize(event, signer_key_id)


_SEPARATED_DUTIES = (
    AuthorityRole.PRODUCER,
    AuthorityRole.LOCAL_IMPORTER,
    AuthorityRole.INDEPENDENT_VERIFIER,
    AuthorityRole.OPERATOR,
    AuthorityRole.PROMOTION_SERVICE,
)


def enforce_separation_of_duties(events: Iterable[DispositionEvent]) -> None:
    """Require cryptographic and actor independence across privileged duties."""
    actors = {role: set() for role in _SEPARATED_DUTIES}
    keys = {role: set() for role in _SEPARATED_DUTIES}
    present: set[AuthorityRole] = set()
    final_state: DispositionState | None = None
    for event in events:
        final_state = event.to_state
        if event.authority in actors:
            present.add(event.authority)
            actors[event.authority].add(event.actor_id)
            keys[event.authority].add(event.signer_key_id)

    for index, left in enumerate(_SEPARATED_DUTIES):
        for right in _SEPARATED_DUTIES[index + 1 :]:
            if actors[left] & actors[right]:
                raise TransitionPolicyError(
                    f"separation of duties violation: actor identity shared by {left.value} and {right.value}"
                )
            if keys[left] & keys[right]:
                raise TransitionPolicyError(
                    f"separation of duties violation: signer key shared by {left.value} and {right.value}"
                )

    if final_state in {
        DispositionState.OPERATOR_APPROVED,
        DispositionState.CHAMPION,
        DispositionState.RETIRED,
    }:
        required = {
            AuthorityRole.PRODUCER,
            AuthorityRole.INDEPENDENT_VERIFIER,
            AuthorityRole.OPERATOR,
        }
        if final_state in {DispositionState.CHAMPION, DispositionState.RETIRED}:
            required.add(AuthorityRole.PROMOTION_SERVICE)
        missing = required - present
        if missing:
            names = ", ".join(sorted(role.value for role in missing))
            raise TransitionPolicyError(f"promotion ledger is missing required duties: {names}")
