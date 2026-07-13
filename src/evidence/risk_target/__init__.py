"""Phase I — risk-target end-to-end evidence vertical slice.

Wires the existing offline risk-target trainer through the full AXIOM evidence
lifecycle described in the training roadmap §12:

    FX daily snapshot
    -> DatasetManifest
    -> risk-target feature snapshot
    -> signed JobManifest
    -> isolated (authority-free) training
    -> per-head EvaluationReport
    -> EvidencePackage
    -> remote CREATED event
    -> local RECEIVED event
    -> hash verification
    -> policy verification
    -> local metric replay
    -> LocalImportVerdict
    -> QUARANTINED or REJECTED

The producer half (:mod:`worker`, :mod:`evaluation`, :mod:`manifests`) is
structurally authority-free: it never imports the broker path, the trade
execution/gate path, or the state engine, and it is handed only its declared
dataset partitions as bytes — it can neither read ``.claude/state.json`` nor
place an order. The local half (:mod:`local_import`) independently reproduces
the producer's metrics before it will accept a package for quarantine, so the
process that produced a candidate is never the sole process that approves it.

A candidate never overwrites an incumbent: this slice stops at QUARANTINED, and
champion promotion belongs to a separate operator-authorized service.
"""

from __future__ import annotations

from .models import HeadResult, HeadImportOutcome, PackagedHead, WorkerOutput

__all__ = [
    "HeadResult",
    "HeadImportOutcome",
    "PackagedHead",
    "WorkerOutput",
]
