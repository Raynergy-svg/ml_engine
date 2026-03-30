"""Phase 60 US-367/US-368 + Phase 65 US-385: ScanHealthSynthesizer + BottleneckRanker.

Reads from all Phase 56-59 observability modules and produces a single unified
health report per synthesize() call. Purely read-only — never modifies any
other module.

Phase 65 upgrade: consumes Phase 61-64 momentum and risk diagnostic signals.
MOMENTUM_KILL and RISK_KILL bottlenecks now carry actual classification data
(NEAR_THRESHOLD/MODERATE/FAR_BELOW/FAR_ABOVE), near_miss_rate_pct, and adaptive
floor status. Two new bottleneck types:
  MOMENTUM_NEAR_MISS (0.68) — momentum gate NEAR_THRESHOLD, micro-adj possible
  RISK_NEAR_MISS    (0.63) — risk gate NEAR_THRESHOLD, ceiling raise considered

Health status tiers:
  CRITICAL  — stall active AND (near-miss rate < 30% OR ensemble divergent)
               → intervention required, likely external (retrain or manual review)
  DEGRADED  — stall active OR dominant kill != 'none' OR near-miss rate > 50%
               → automated adjustments may help, needs monitoring
  HEALTHY   — no significant bottlenecks detected

Bottleneck impact scores (0.0-1.0, sorted descending):
  SYSTEM_STALL        = 1.00  (always highest — existential)
  ENSEMBLE_DIVERGENCE = 0.85  (structural model disagreement — requires retrain)
  CONFIDENCE_TOO_HIGH = 0.75 * min(gap_at_p75/20.0, 1.0)  (config-level fix)
  MOMENTUM_KILL       = 0.70  (gate tuning needed)
  MOMENTUM_NEAR_MISS  = 0.68  (Phase 65: NEAR_THRESHOLD momentum — micro-adj)
  NEAR_MISS_CLUSTER   = 0.65 * (near_miss_rate_pct/100.0)  (threshold micro-adj)
  RISK_NEAR_MISS      = 0.63  (Phase 65: NEAR_THRESHOLD risk — ceiling micro-adj)
  RISK_KILL           = 0.60  (gate tuning needed)

Design: wired into continuous.py, synthesize() called every 10 scan cycles.
All Phase 65 additions are backward-compatible: existing synthesize() works with
no momentum_recorder or risk_recorder (safe defaults throughout).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── health status constants ──────────────────────────────────────────────────

STATUS_HEALTHY = "HEALTHY"
STATUS_DEGRADED = "DEGRADED"
STATUS_CRITICAL = "CRITICAL"

# ─── bottleneck name constants ────────────────────────────────────────────────

BN_SYSTEM_STALL = "SYSTEM_STALL"
BN_ENSEMBLE_DIVERGENCE = "ENSEMBLE_DIVERGENCE"
BN_CONFIDENCE_TOO_HIGH = "CONFIDENCE_TOO_HIGH"
BN_MOMENTUM_KILL = "MOMENTUM_KILL"
BN_MOMENTUM_NEAR_MISS = "MOMENTUM_NEAR_MISS"   # Phase 65
BN_NEAR_MISS_CLUSTER = "NEAR_MISS_CLUSTER"
BN_RISK_NEAR_MISS = "RISK_NEAR_MISS"           # Phase 65
BN_RISK_KILL = "RISK_KILL"
BN_VOTE_NEAR_MISS = "VOTE_NEAR_MISS"           # Phase 66
BN_VOTE_KILL = "VOTE_KILL"                      # Phase 66

# ─── ensemble divergence recommendation constant (matches Phase 59) ───────────

_REC_HIGH_DISAGREEMENT = "HIGH_ENSEMBLE_DISAGREEMENT"

# ─── Phase 61-64 classification constants ────────────────────────────────────

_CLS_NEAR_THRESHOLD = "NEAR_THRESHOLD"


def rank_bottlenecks(signals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify and rank bottlenecks by impact score from synthesized signals.

    Args:
        signals: dict with keys (Phase 60 baseline):
            gap_at_p75 (float)          — confidence gap at 75th percentile
            near_miss_rate_pct (float)  — % of conf-rejects that are near-misses
            dominant_kill_stage (str)   — 'confidence'/'momentum'/'risk'/'none'
            ensemble_recommendation (str) — 'HIGH_ENSEMBLE_DISAGREEMENT'/'OK'
            stall_active (bool)         — whether system is currently stalled
            tradeable_pct (float)       — % of pairs that are tradeable

            Phase 65 additional keys (optional — all have safe defaults):
            momentum_classification (str)       — NEAR_THRESHOLD/MODERATE/FAR_BELOW/INSUFFICIENT_DATA
            momentum_gap_at_p50 (float)         — momentum gap at 50th percentile
            momentum_near_miss_rate_pct (float) — % of momentum rejects near threshold
            momentum_pass_rate (float)          — momentum gate pass rate (0-1)
            risk_classification (str)           — NEAR_THRESHOLD/MODERATE/FAR_ABOVE/INSUFFICIENT_DATA
            risk_gap_at_p50 (float)             — risk gap at 50th percentile
            risk_near_miss_rate_pct (float)     — % of risk rejects near threshold
            risk_pass_rate (float)              — risk gate pass rate (0-1)
            risk_drawdown_kill_pct (float)      — % of risk kills from drawdown (vs streak)

            Phase 66 additional keys (optional — all have safe defaults):
            vote_classification (str)           — NEAR_THRESHOLD/MODERATE/FAR_BELOW/INSUFFICIENT_DATA
            vote_gap_at_p50 (float)             — vote gap at 50th percentile
            vote_near_miss_rate_pct (float)     — % of vote rejects near threshold
            vote_pass_rate (float)              — vote gate pass rate (0-1)

    Returns:
        List of bottleneck dicts sorted by impact_score descending.
        Each dict: {name, impact_score, description, action}
    """
    bottlenecks: List[Dict[str, Any]] = []

    # ── Phase 60 baseline signals ────────────────────────────────────────────
    gap_at_p75 = float(signals.get("gap_at_p75", 0.0))
    near_miss_rate_pct = float(signals.get("near_miss_rate_pct", 0.0))
    dominant_kill = str(signals.get("dominant_kill_stage", "none")).lower()
    ensemble_rec = str(signals.get("ensemble_recommendation", "OK"))
    stall_active = bool(signals.get("stall_active", False))
    tradeable_pct = float(signals.get("tradeable_pct", 100.0))

    # ── Phase 65 momentum signals ─────────────────────────────────────────────
    momentum_cls = str(signals.get("momentum_classification", ""))
    momentum_near_miss_pct = float(signals.get("momentum_near_miss_rate_pct", 0.0))

    # ── Phase 65 risk signals ─────────────────────────────────────────────────
    risk_cls = str(signals.get("risk_classification", ""))
    risk_drawdown_kill_pct = float(signals.get("risk_drawdown_kill_pct", 0.0))
    risk_near_miss_pct = float(signals.get("risk_near_miss_rate_pct", 0.0))

    # ── Phase 66 vote signals ──────────────────────────────────────────────
    vote_cls = str(signals.get("vote_classification", ""))
    vote_near_miss_pct = float(signals.get("vote_near_miss_rate_pct", 0.0))

    # SYSTEM_STALL — highest impact, always first when active
    if stall_active:
        bottlenecks.append({
            "name": BN_SYSTEM_STALL,
            "impact_score": 1.00,
            "description": "System is stalled — no tradeable pairs detected over multiple scans",
            "action": "Check penalty auditor recommendation; verify raw confidence distribution vs threshold",
        })

    # ENSEMBLE_DIVERGENCE — structural model disagreement, requires retraining
    if ensemble_rec == _REC_HIGH_DISAGREEMENT:
        bottlenecks.append({
            "name": BN_ENSEMBLE_DIVERGENCE,
            "impact_score": 0.85,
            "description": "TCN and Ridge models are structurally divergent (>30% of snapshots show >20pp gap)",
            "action": "Trigger model retraining; investigate feature drift between TCN and Ridge inputs",
        })

    # CONFIDENCE_TOO_HIGH — threshold is above the distribution's 75th percentile
    if gap_at_p75 > 5.0:
        score = 0.75 * min(gap_at_p75 / 20.0, 1.0)
        bottlenecks.append({
            "name": BN_CONFIDENCE_TOO_HIGH,
            "impact_score": round(score, 4),
            "description": f"Confidence threshold {gap_at_p75:.1f}pt above 75th-percentile of raw scores",
            "action": f"Consider adaptive floor reduction; ThresholdGapAnalyzer gap_at_p75={gap_at_p75:.1f}pt suggests threshold too high",
        })

    # MOMENTUM_KILL — pairs clearing confidence but dying at momentum gate
    # Phase 65: enrich action with actual classification if available
    if dominant_kill == "momentum" and tradeable_pct < 5.0:
        if momentum_cls and momentum_cls != _CLS_NEAR_THRESHOLD:
            # Has Phase 65 data and it's NOT near-threshold → standard MOMENTUM_KILL
            action = (
                f"Momentum gate classification={momentum_cls} "
                f"near_miss_rate={momentum_near_miss_pct:.1f}% — "
            )
            if momentum_cls == "FAR_BELOW":
                action += "momentum scores far below threshold; trigger XGBoost retrain"
            elif momentum_cls == "MODERATE":
                action += "moderate gap; review XGBoost momentum signal quality"
            else:
                action += "review momentum gate threshold; check XGBoost momentum signal distribution"
        else:
            action = "Investigate momentum gate threshold; review XGBoost momentum signal distribution"

        bottlenecks.append({
            "name": BN_MOMENTUM_KILL,
            "impact_score": 0.70,
            "description": "Momentum gate is dominant kill stage — pairs clear confidence but fail momentum",
            "action": action,
        })

    # MOMENTUM_NEAR_MISS — Phase 65: momentum dominant AND NEAR_THRESHOLD
    # Sits at 0.68 — between MOMENTUM_KILL (0.70) and NEAR_MISS_CLUSTER (0.65)
    if dominant_kill == "momentum" and momentum_cls == _CLS_NEAR_THRESHOLD:
        bottlenecks.append({
            "name": BN_MOMENTUM_NEAR_MISS,
            "impact_score": 0.68,
            "description": (
                f"Momentum gate NEAR_THRESHOLD — {momentum_near_miss_pct:.1f}% of rejects "
                f"are within 0.05 of threshold; micro-adjustment may unlock trades"
            ),
            "action": (
                f"Momentum classification=NEAR_THRESHOLD near_miss_rate={momentum_near_miss_pct:.1f}% — "
                "lower min_momentum by 0.02 via AdaptiveMomentumFloor or manual config adjustment"
            ),
        })

    # NEAR_MISS_CLUSTER — many confidence rejects are within 1-3pt of threshold
    if near_miss_rate_pct > 50.0:
        score = 0.65 * (near_miss_rate_pct / 100.0)
        bottlenecks.append({
            "name": BN_NEAR_MISS_CLUSTER,
            "impact_score": round(score, 4),
            "description": f"{near_miss_rate_pct:.1f}% of confidence rejects are near-misses (gap ≤3pt)",
            "action": "1-2pt threshold reduction would likely unblock these pairs; use GateProximityReporter.get_actionable_threshold()",
        })

    # RISK_NEAR_MISS — Phase 65: risk dominant AND NEAR_THRESHOLD
    # Sits at 0.63 — between NEAR_MISS_CLUSTER (0.65*rate) and RISK_KILL (0.60)
    if dominant_kill == "risk" and risk_cls == _CLS_NEAR_THRESHOLD:
        bottlenecks.append({
            "name": BN_RISK_NEAR_MISS,
            "impact_score": 0.63,
            "description": (
                f"Risk gate NEAR_THRESHOLD — {risk_near_miss_pct:.1f}% of rejects "
                f"are within 0.005 of max_drawdown threshold; ceiling raise may help"
            ),
            "action": (
                f"Risk classification=NEAR_THRESHOLD near_miss_rate={risk_near_miss_pct:.1f}% — "
                "raise max_drawdown_pct by 0.001 via AdaptiveDrawdownCeiling or manual config"
            ),
        })

    # RISK_KILL — pairs clearing confidence + momentum but dying at risk gate
    # Phase 65: enrich action with actual classification if available
    if dominant_kill == "risk" and tradeable_pct < 5.0:
        if risk_cls and risk_cls != _CLS_NEAR_THRESHOLD:
            action = (
                f"Risk gate classification={risk_cls} "
                f"drawdown_kill_pct={risk_drawdown_kill_pct:.1f}% — "
            )
            if risk_cls == "FAR_ABOVE":
                action += "predicted drawdown far above limit; verify market regime or retrain RF"
            elif risk_cls == "MODERATE":
                action += "moderate drawdown gap; check RF calibration quality"
            else:
                action += "review risk gate parameters; check ATR-based SL/TP and R:R ratio gate"
        else:
            action = "Review risk gate parameters; check ATR-based SL/TP calculation and R:R ratio gate"

        bottlenecks.append({
            "name": BN_RISK_KILL,
            "impact_score": 0.60,
            "description": "Risk gate is dominant kill stage — pairs clear confidence and momentum but fail risk",
            "action": action,
        })

    # VOTE_NEAR_MISS — Phase 66: agent consensus dominant AND NEAR_THRESHOLD
    # Sits at 0.62 — between RISK_NEAR_MISS (0.63) and RISK_KILL (0.60)
    if dominant_kill == "agent_consensus" and vote_cls == _CLS_NEAR_THRESHOLD:
        bottlenecks.append({
            "name": BN_VOTE_NEAR_MISS,
            "impact_score": 0.62,
            "description": (
                f"Agent consensus NEAR_THRESHOLD — {vote_near_miss_pct:.1f}% of rejects "
                f"are within 0.05 of vote threshold; threshold reduction may help"
            ),
            "action": (
                f"Vote classification=NEAR_THRESHOLD near_miss_rate={vote_near_miss_pct:.1f}% — "
                "reduce weighted_vote_threshold by 0.02 via AdaptiveVoteThreshold or manual config"
            ),
        })

    # VOTE_KILL — agent consensus is dominant kill and not NEAR_THRESHOLD
    if dominant_kill == "agent_consensus" and tradeable_pct < 5.0:
        if vote_cls and vote_cls != _CLS_NEAR_THRESHOLD:
            action = (
                f"Vote classification={vote_cls} — "
            )
            if vote_cls == "FAR_BELOW":
                action += "agent scores structurally weak; review agent weights, sub-inference pipeline, and RL updates"
            elif vote_cls == "MODERATE":
                action += "moderate gap; check agent weight distribution and recent RL sync quality"
            else:
                action += "review weighted_vote_threshold and agent team configuration"
        else:
            action = "Review weighted_vote_threshold and agent team consensus pipeline"

        bottlenecks.append({
            "name": BN_VOTE_KILL,
            "impact_score": 0.58,
            "description": "Agent consensus is dominant kill stage — pairs clear other gates but fail vote threshold",
            "action": action,
        })

    # Sort by impact_score descending
    bottlenecks.sort(key=lambda b: b["impact_score"], reverse=True)
    return bottlenecks


class ScanHealthSynthesizer:
    """Unified health synthesizer — reads from all Phase 56-66 modules.

    Phase 65 upgrade: momentum_recorder and risk_recorder for enriched diagnostics.
    Phase 66 upgrade: vote_recorder for agent consensus gate diagnostics.

    Fully backward-compatible: omitting any recorder preserves prior behavior.

    Usage:
        synthesizer = ScanHealthSynthesizer(
            signal_funnel=self._signal_funnel,
            gate_proximity=self._gate_proximity,
            ensemble_divergence=self._ensemble_divergence,
            raw_conf_recorder=getattr(self._scanner, '_raw_conf_recorder', None),
            scan_diagnostics=self._scan_diagnostics,
            # Phase 65 additions:
            momentum_recorder=getattr(self._scanner, '_momentum_recorder', None),
            risk_recorder=getattr(self._scanner, '_risk_recorder', None),
            momentum_max_threshold=getattr(self._scanner.config, 'min_momentum', 0.45),
            risk_max_drawdown=getattr(self._scanner.config, 'max_drawdown_pct', 0.025),
        )
        if scan_cycle % 10 == 0:
            report = synthesizer.synthesize()
    """

    def __init__(
        self,
        signal_funnel=None,
        gate_proximity=None,
        ensemble_divergence=None,
        raw_conf_recorder=None,
        scan_diagnostics=None,
        # Phase 65 additions
        momentum_recorder=None,
        risk_recorder=None,
        momentum_max_threshold: float = 0.45,
        risk_max_drawdown: float = 0.025,
        # Phase 66 additions
        vote_recorder=None,
        weighted_vote_threshold: float = 0.55,
    ) -> None:
        self._signal_funnel = signal_funnel
        self._gate_proximity = gate_proximity
        self._ensemble_divergence = ensemble_divergence
        self._raw_conf_recorder = raw_conf_recorder
        self._scan_diagnostics = scan_diagnostics
        # Phase 65
        self._momentum_recorder = momentum_recorder
        self._risk_recorder = risk_recorder
        self._momentum_max_threshold = float(momentum_max_threshold)
        self._risk_max_drawdown = float(risk_max_drawdown)
        # Phase 66
        self._vote_recorder = vote_recorder
        self._weighted_vote_threshold = float(weighted_vote_threshold)

    # ------------------------------------------------------------------ #
    # Core synthesis                                                        #
    # ------------------------------------------------------------------ #

    def synthesize(self) -> Dict[str, Any]:
        """Read from all available modules and return unified health report.

        Returns a dict with keys:
            health_status (str)           — HEALTHY / DEGRADED / CRITICAL
            bottlenecks (list)            — sorted by impact_score desc
            top_action (str)              — highest-priority action
            confidence_floor_gap (float)  — gap_at_p75 from ThresholdGapAnalyzer
            near_miss_rate_pct (float)    — from GateProximityReporter
            funnel_dominant_kill (str)    — from SignalFunnelTracker
            ensemble_recommendation (str) — from EnsembleDivergenceMonitor
            stall_active (bool)           — from ScanDiagnosticsReporter
            tradeable_pct (float)         — from SignalFunnelTracker
            summary (str)                 — one-line human-readable
            signals (dict)                — full raw signals for debugging
        """
        try:
            signals = self._collect_signals()
            bottlenecks = rank_bottlenecks(signals)
            health_status = self._compute_health_status(signals, bottlenecks)
            top_action = bottlenecks[0]["action"] if bottlenecks else "No action needed — system healthy"
            summary = self._build_summary(health_status, bottlenecks, signals)

            return {
                "health_status": health_status,
                "bottlenecks": bottlenecks,
                "top_action": top_action,
                "confidence_floor_gap": signals.get("gap_at_p75", 0.0),
                "near_miss_rate_pct": signals.get("near_miss_rate_pct", 0.0),
                "funnel_dominant_kill": signals.get("dominant_kill_stage", "none"),
                "ensemble_recommendation": signals.get("ensemble_recommendation", "OK"),
                "stall_active": signals.get("stall_active", False),
                "tradeable_pct": signals.get("tradeable_pct", 0.0),
                "summary": summary,
                "signals": signals,
            }
        except Exception as exc:
            logger.error("ScanHealthSynthesizer.synthesize() failed: %s", exc)
            return {
                "health_status": STATUS_HEALTHY,
                "bottlenecks": [],
                "top_action": "Synthesis error — check logs",
                "confidence_floor_gap": 0.0,
                "near_miss_rate_pct": 0.0,
                "funnel_dominant_kill": "none",
                "ensemble_recommendation": "OK",
                "stall_active": False,
                "tradeable_pct": 0.0,
                "summary": f"Health synthesis failed: {exc}",
                "signals": {},
            }

    # ------------------------------------------------------------------ #
    # Signal collection                                                     #
    # ------------------------------------------------------------------ #

    def _collect_signals(self) -> Dict[str, Any]:
        """Pull current data from each available module.

        Phase 65: also queries MomentumGapAnalyzer and RiskGateAnalyzer
        if momentum_recorder / risk_recorder are wired in.
        """
        signals: Dict[str, Any] = {
            # Phase 60 baseline
            "gap_at_p75": 0.0,
            "near_miss_rate_pct": 0.0,
            "dominant_kill_stage": "none",
            "ensemble_recommendation": "OK",
            "stall_active": False,
            "tradeable_pct": 100.0,
            # Phase 65 momentum defaults
            "momentum_classification": "",
            "momentum_gap_at_p50": 0.0,
            "momentum_near_miss_rate_pct": 0.0,
            "momentum_pass_rate": 1.0,
            # Phase 65 risk defaults
            "risk_classification": "",
            "risk_gap_at_p50": 0.0,
            "risk_near_miss_rate_pct": 0.0,
            "risk_pass_rate": 1.0,
            "risk_drawdown_kill_pct": 0.0,
            # Phase 66 vote defaults
            "vote_classification": "",
            "vote_gap_at_p50": 0.0,
            "vote_near_miss_rate_pct": 0.0,
            "vote_pass_rate": 1.0,
        }

        # ScanDiagnosticsReporter (Phase 56 US-350)
        if self._scan_diagnostics is not None:
            try:
                diag = self._scan_diagnostics.generate_report()
                signals["stall_active"] = bool(diag.get("stalled", False))
            except Exception as exc:
                logger.debug("ScanHealthSynthesizer: scan_diagnostics read failed: %s", exc)

        # SignalFunnelTracker (Phase 59 US-363)
        if self._signal_funnel is not None:
            try:
                funnel = self._signal_funnel.get_funnel_summary()
                signals["dominant_kill_stage"] = funnel.get("dominant_kill_stage", "none")
                signals["tradeable_pct"] = float(funnel.get("tradeable_pct", 100.0))
            except Exception as exc:
                logger.debug("ScanHealthSynthesizer: signal_funnel read failed: %s", exc)

        # GateProximityReporter (Phase 59 US-364)
        if self._gate_proximity is not None:
            try:
                proximity = self._gate_proximity.get_proximity_summary()
                signals["near_miss_rate_pct"] = float(proximity.get("near_miss_rate_pct", 0.0))
            except Exception as exc:
                logger.debug("ScanHealthSynthesizer: gate_proximity read failed: %s", exc)

        # EnsembleDivergenceMonitor (Phase 59 US-365)
        if self._ensemble_divergence is not None:
            try:
                div = self._ensemble_divergence.get_divergence_summary()
                signals["ensemble_recommendation"] = div.get("recommendation", "OK")
            except Exception as exc:
                logger.debug("ScanHealthSynthesizer: ensemble_divergence read failed: %s", exc)

        # RawConfidenceRecorder + ThresholdGapAnalyzer (Phase 58 US-358/US-359)
        if self._raw_conf_recorder is not None:
            try:
                from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
                _analyzer = ThresholdGapAnalyzer()
                _threshold = float(getattr(self._raw_conf_recorder, '_threshold', 58.0))
                gap_report = _analyzer.analyze(self._raw_conf_recorder, _threshold)
                signals["gap_at_p75"] = float(gap_report.get("gap_at_p75", 0.0))
            except Exception as exc:
                logger.debug("ScanHealthSynthesizer: raw_conf_recorder/gap_analyzer read failed: %s", exc)

        # ── Phase 65: MomentumGapAnalyzer (Phase 61 US-372) ──────────────────
        if self._momentum_recorder is not None:
            try:
                from src.scanner.momentum_gap_analyzer import analyze as _mga_analyze
                mga_result = _mga_analyze(self._momentum_recorder, self._momentum_max_threshold)
                signals["momentum_classification"] = str(mga_result.get("classification", ""))
                signals["momentum_gap_at_p50"] = float(mga_result.get("gap_at_p50", 0.0))
                signals["momentum_near_miss_rate_pct"] = float(mga_result.get("near_miss_rate_pct", 0.0))
                signals["momentum_pass_rate"] = float(mga_result.get("pass_rate", 1.0))
            except Exception as exc:
                logger.debug("ScanHealthSynthesizer: momentum_recorder/gap_analyzer read failed: %s", exc)

        # ── Phase 65: RiskGateAnalyzer (Phase 63 US-379) ─────────────────────
        if self._risk_recorder is not None:
            try:
                from src.scanner.risk_gate_analyzer import get_risk_signals as _get_risk_signals
                risk_sigs = _get_risk_signals(self._risk_recorder, self._risk_max_drawdown)
                signals["risk_classification"] = str(risk_sigs.get("risk_classification", ""))
                signals["risk_gap_at_p50"] = float(risk_sigs.get("risk_gap_at_p50", 0.0))
                signals["risk_near_miss_rate_pct"] = float(risk_sigs.get("risk_near_miss_rate_pct", 0.0))
                signals["risk_pass_rate"] = float(risk_sigs.get("risk_pass_rate", 1.0))
                signals["risk_drawdown_kill_pct"] = float(risk_sigs.get("risk_drawdown_kill_pct", 0.0))
            except Exception as exc:
                logger.debug("ScanHealthSynthesizer: risk_recorder/gate_analyzer read failed: %s", exc)

        # ── Phase 66: VoteGapAnalyzer (Phase 66 US-389) ────────────────────
        if self._vote_recorder is not None:
            try:
                from src.scanner.vote_gap_analyzer import analyze as _vga_analyze
                vga_result = _vga_analyze(self._vote_recorder, self._weighted_vote_threshold)
                signals["vote_classification"] = str(vga_result.get("classification", ""))
                signals["vote_gap_at_p50"] = float(vga_result.get("gap_at_p50", 0.0))
                signals["vote_near_miss_rate_pct"] = float(vga_result.get("near_miss_rate_pct", 0.0))
                signals["vote_pass_rate"] = float(vga_result.get("pass_rate", 1.0))
            except Exception as exc:
                logger.debug("ScanHealthSynthesizer: vote_recorder/gap_analyzer read failed: %s", exc)

        return signals

    # ------------------------------------------------------------------ #
    # Health status computation                                             #
    # ------------------------------------------------------------------ #

    def _compute_health_status(
        self,
        signals: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]],
    ) -> str:
        """Determine health status from signals and ranked bottlenecks."""
        stall_active = signals.get("stall_active", False)
        near_miss_rate = signals.get("near_miss_rate_pct", 0.0)
        ensemble_rec = signals.get("ensemble_recommendation", "OK")
        dominant_kill = signals.get("dominant_kill_stage", "none")

        # CRITICAL: stall active AND (near-miss cluster absent OR ensemble divergent)
        # meaning: system is stalled and the problem is NOT just a small threshold adj
        if stall_active and (near_miss_rate < 30.0 or ensemble_rec == _REC_HIGH_DISAGREEMENT):
            return STATUS_CRITICAL

        # DEGRADED: any significant signal present
        if (
            stall_active
            or (dominant_kill not in ("none", ""))
            or near_miss_rate > 50.0
            or ensemble_rec == _REC_HIGH_DISAGREEMENT
        ):
            return STATUS_DEGRADED

        return STATUS_HEALTHY

    # ------------------------------------------------------------------ #
    # Summary                                                               #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        health_status: str,
        bottlenecks: List[Dict[str, Any]],
        signals: Dict[str, Any],
    ) -> str:
        if not bottlenecks:
            return f"[{health_status}] All signals nominal — no bottlenecks detected"
        top = bottlenecks[0]
        return (
            f"[{health_status}] Top bottleneck: {top['name']} "
            f"(impact={top['impact_score']:.2f}) | Action: {top['action']}"
        )
