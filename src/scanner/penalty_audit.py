"""Penalty Auditor — Phase 57 US-356.

Surfaces a structured penalty breakdown report when the engine is stalled.
Consumes the Phase 56 ConfidencePenaltyMonitor and GateRejectionTracker
to answer: "what is blocking signals, and what should the operator do?"

Output recommendations:
    CEILING_NEEDED          — breach_rate > 0.80: stacking penalties are floor-suppressing
    CALIBRATION_GUARD_NEEDED — most common source is 'overconfidence' with < 20 predictions
    THRESHOLD_RELAXATION_NEEDED — gate_pass_rate_pct < 5%: min_confidence threshold too high
    OK                      — no actionable finding

Usage:
    auditor = PenaltyAuditor()
    audit = auditor.run_audit(
        confidence_penalty_monitor=cpm,
        gate_rejection_tracker=grt,
    )
    auditor.save_audit(audit, path)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_AUDIT_PATH = _PROJECT_ROOT / "trained_data" / "penalty_audit_report.json"

# Thresholds for recommendations
BREACH_RATE_CEILING_THRESHOLD = 0.50   # breach_rate > 50% → CEILING_NEEDED
GATE_PASS_RATE_THRESHOLD = 5.0         # gate_pass_rate_pct < 5% → THRESHOLD_RELAXATION


class PenaltyAuditor:
    """Generates actionable penalty audit reports when the engine is stalled.

    Integrates Phase 56 diagnostic data (ConfidencePenaltyMonitor,
    GateRejectionTracker) to produce operator-readable recommendations.

    Usage:
        auditor = PenaltyAuditor()
        audit = auditor.run_audit(cpm, grt)
        auditor.save_audit(audit)
    """

    def __init__(self, audit_path: Optional[Path] = None):
        self.audit_path = Path(audit_path or _DEFAULT_AUDIT_PATH)

    # ── Public API ─────────────────────────────────────────────────────────

    def run_audit(
        self,
        confidence_penalty_monitor=None,
        gate_rejection_tracker=None,
        calibration_n_predictions: int = 0,
        scan_diagnostics_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a structured penalty audit report.

        Args:
            confidence_penalty_monitor: ConfidencePenaltyMonitor instance (Phase 56).
            gate_rejection_tracker: GateRejectionTracker instance (Phase 56).
            calibration_n_predictions: n_predictions from ModelCalibrationTracker (for guard check).
            scan_diagnostics_report: Latest ScanDiagnosticsReporter report dict.

        Returns:
            Audit dict with: top_penalty_source, breach_rate_pct, avg_cumulative_penalty,
            pairs_above_ceiling, recommendation, top_blocking_gate, gate_pass_rate_pct,
            timestamp.
        """
        # Collect penalty data
        breach_rate = 0.0
        avg_penalty = 0.0
        top_source = "unknown"
        pairs_above_ceiling = 0

        if confidence_penalty_monitor is not None:
            try:
                breach_rate = float(confidence_penalty_monitor.get_breach_rate())
                cpm_report = confidence_penalty_monitor.generate_report()

                # Aggregate across all pairs
                all_sources: Dict[str, float] = {}
                total_penalty = 0.0
                n_pairs = 0
                n_above_ceiling = 0
                CEILING_FLOOR = 0.20  # pairs_with_total_penalty > 0.20 = above ceiling concern

                for pair_data in cpm_report.values():
                    if not isinstance(pair_data, dict):
                        continue
                    n_pairs += 1
                    pair_total = float(pair_data.get("total_penalty", 0.0))
                    total_penalty += pair_total
                    if pair_total > CEILING_FLOOR:
                        n_above_ceiling += 1
                    for src, val in pair_data.get("sources", {}).items():
                        all_sources[src] = all_sources.get(src, 0.0) + float(val)

                avg_penalty = total_penalty / max(n_pairs, 1)
                pairs_above_ceiling = n_above_ceiling
                if all_sources:
                    top_source = max(all_sources, key=all_sources.get)
            except Exception as e:
                logger.debug("PenaltyAuditor: CPM data extraction failed: %s", e)

        # Collect gate rejection data
        top_blocking_gate = "unknown"
        gate_pass_rate_pct = 100.0

        if gate_rejection_tracker is not None:
            try:
                tg = gate_rejection_tracker.get_top_blocking_gate()
                if tg:
                    top_blocking_gate = tg
            except Exception as e:
                logger.debug("PenaltyAuditor: GRT data extraction failed: %s", e)

        if scan_diagnostics_report is not None:
            gate_pass_rate_pct = float(
                scan_diagnostics_report.get("gate_pass_rate_pct", 100.0)
            )

        # Determine recommendation
        recommendation = self._recommend(
            breach_rate=breach_rate,
            top_source=top_source,
            calibration_n_predictions=calibration_n_predictions,
            gate_pass_rate_pct=gate_pass_rate_pct,
        )

        audit = {
            "top_penalty_source": top_source,
            "breach_rate_pct": round(breach_rate * 100.0, 1),
            "avg_cumulative_penalty": round(avg_penalty, 4),
            "pairs_above_ceiling": pairs_above_ceiling,
            "top_blocking_gate": top_blocking_gate,
            "gate_pass_rate_pct": gate_pass_rate_pct,
            "calibration_n_predictions": calibration_n_predictions,
            "recommendation": recommendation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.warning(
            "PenaltyAuditor: breach_rate=%.1f%% top_source=%s top_gate=%s "
            "gate_pass=%.1f%% recommendation=%s",
            breach_rate * 100.0, top_source, top_blocking_gate,
            gate_pass_rate_pct, recommendation,
        )
        return audit

    def save_audit(
        self,
        audit: Dict[str, Any],
        path: Optional[Path] = None,
    ) -> None:
        """Atomically write audit report to disk."""
        target = Path(path or self.audit_path)
        self._atomic_write(target, audit)

    # ── Internal ────────────────────────────────────────────────────────────

    def _recommend(
        self,
        breach_rate: float,
        top_source: str,
        calibration_n_predictions: int,
        gate_pass_rate_pct: float,
    ) -> str:
        if breach_rate > BREACH_RATE_CEILING_THRESHOLD:
            return "CEILING_NEEDED"
        if top_source == "overconfidence" and calibration_n_predictions < 20:
            return "CALIBRATION_GUARD_NEEDED"
        if gate_pass_rate_pct < GATE_PASS_RATE_THRESHOLD:
            return "THRESHOLD_RELAXATION_NEEDED"
        return "OK"

    def _atomic_write(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, path)
        except Exception as e:
            logger.warning("PenaltyAuditor: Write to %s failed: %s", path, e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
