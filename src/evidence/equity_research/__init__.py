"""Phase J2 — equity research workers end-to-end evidence vertical slice.

Brings the equity cross-sectional research-alpha lane through the full evidence
lifecycle described in the training roadmap §12, reusing the hedge / risk-target
slice patterns:

    per-fold research cross-sections (one partition per fold)
    -> DatasetManifest
    -> StrategyManifest (workload spec, declaring the expected fold set per tier)
    -> signed JobManifest
    -> isolated (authority-free) fold scoring + deterministic tier aggregation
    -> per-tier EvaluationReport
    -> EvidencePackage
    -> remote CREATED event
    -> local RECEIVED event
    -> hash verification
    -> policy verification
    -> local rank-IC aggregation replay
    -> LocalImportVerdict
    -> QUARANTINED or REJECTED

The Phase-H parallel unit is ``hypothesis × universe-snapshot × fold``: many
per-fold cross-sectional rank-IC scorecards are combined by a DETERMINISTIC
AGGREGATOR that refuses an incomplete or duplicated fold set. The unit of
disposition is one *universe tier* (curated / wide survivorship-corrected /
full_history); each tier is scored, gated, packaged and dispositioned fully
independently, so the attractive curated result and the defensible wide result
are reported separately and preserved side by side even when they disagree — a
tier whose out-of-sample rank-IC fades to insignificance as the universe widens
is REJECTED as a first-class honest negative, never rescued.

The producer half (:mod:`worker`, :mod:`evaluation`, :mod:`manifests`,
:mod:`models`) is structurally authority-free: it never imports the broker path,
the trade execution/gate path, or the state engine, and it is handed only its
declared fold partitions as bytes. The local half (:mod:`local_import`)
independently reproduces the producer's aggregation before it will accept a
package for quarantine. A candidate never overwrites an incumbent: this slice
stops at QUARANTINED, and champion promotion belongs to a separate
operator-authorized service.
"""

from __future__ import annotations

from .models import (
    EquityResearchWorkerOutput,
    PackagedTierHead,
    TierHeadResult,
    TierImportOutcome,
    lane_id_for_tier,
)

__all__ = [
    "EquityResearchWorkerOutput",
    "PackagedTierHead",
    "TierHeadResult",
    "TierImportOutcome",
    "lane_id_for_tier",
]
