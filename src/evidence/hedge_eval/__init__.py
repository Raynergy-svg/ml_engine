"""Phase J6 — hedge evaluation end-to-end evidence vertical slice.

Brings the AXIOM hedge/exposure lane through the full evidence lifecycle
described in the training roadmap §12, reusing the risk-target slice's patterns:

    raw-vs-hedged shadow ledger (per strategy)
    -> DatasetManifest
    -> StrategyManifest (workload spec)
    -> signed JobManifest
    -> isolated (authority-free) scoring
    -> per-strategy EvaluationReport
    -> EvidencePackage
    -> remote CREATED event
    -> local RECEIVED event
    -> hash verification
    -> policy verification
    -> local scorecard replay
    -> LocalImportVerdict
    -> QUARANTINED or REJECTED

The unit of evaluation is one *strategy*: its logged raw-vs-hedged returns are
scored into an alpha-vs-beta verdict (via the existing analysis-only
:mod:`src.hedge.hedge_scorecard`) and gated on its own after-cost admissibility,
so a strategy whose return dies once hedged (``return_was_beta_not_alpha``) is
REJECTED while a strategy whose hedged residual is genuine alpha is QUARANTINED
— fully independently, exactly like the risk-target vol/drawdown heads.

The producer half (:mod:`worker`, :mod:`evaluation`, :mod:`manifests`,
:mod:`models`) is structurally authority-free: it never imports the broker path,
the trade execution/gate path, or the state engine, and it is handed only its
declared ledger partitions as bytes. The local half (:mod:`local_import`)
independently reproduces the producer's scorecard before it will accept a
package for quarantine. A candidate never overwrites an incumbent: this slice
stops at QUARANTINED, and champion promotion belongs to a separate
operator-authorized service.
"""

from __future__ import annotations

from .models import (
    HedgeHeadResult,
    HedgeImportOutcome,
    HedgeWorkerOutput,
    PackagedHedgeHead,
    lane_id_for_strategy,
)

__all__ = [
    "HedgeHeadResult",
    "HedgeImportOutcome",
    "HedgeWorkerOutput",
    "PackagedHedgeHead",
    "lane_id_for_strategy",
]
