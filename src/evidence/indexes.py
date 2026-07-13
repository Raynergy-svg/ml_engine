"""Rebuildable, deterministic projections over disposition ledgers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .contracts import DispositionReceipt
from .event_store import ReconstructedDisposition, reconstruct_disposition
from .signing import TrustStore
from .transition_policy import AuthorityRegistry


@dataclass(frozen=True)
class EvidenceIndex:
    entries: tuple[ReconstructedDisposition, ...]

    def by_package(self) -> dict[str, ReconstructedDisposition]:
        return {entry.package_digest: entry for entry in self.entries}


def rebuild_index(
    ledgers: Mapping[str, Iterable[DispositionReceipt]],
    *,
    trust_store: TrustStore,
    authorities: AuthorityRegistry,
) -> EvidenceIndex:
    """Rebuild current package state solely from signed disposition history."""
    entries: list[ReconstructedDisposition] = []
    for declared_package, ledger in ledgers.items():
        state = reconstruct_disposition(ledger, trust_store=trust_store, authorities=authorities)
        if state.package_digest != declared_package:
            raise ValueError(
                f"ledger key {declared_package!r} does not match package {state.package_digest!r}"
            )
        entries.append(state)
    return EvidenceIndex(entries=tuple(sorted(entries, key=lambda item: item.package_digest)))
