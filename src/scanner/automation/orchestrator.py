"""ML Agent Orchestrator — full scan→trade→learn→tune loop.

Coordinates all automation modules into a single improvement cycle.
This is the brain of Buddy's autonomous evolution.

Architecture:
    Scanner → Agents → Gates → Execution → OANDA
        ↑                                      ↓
        └── Config Tuner ← Rules ← Learnings ← RL Feedback ← Trade Outcomes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of a single orchestration cycle."""

    cycle_id: str = ""
    timestamp: str = ""
    scan_duration_secs: float = 0.0
    pairs_scanned: int = 0
    tradeable_setups: int = 0
    trades_executed: int = 0
    learnings_extracted: int = 0
    rules_promoted: int = 0
    config_adjustments: int = 0
    observations_logged: int = 0
    agent_health_report: Optional[Dict[str, Any]] = None
    stress_state: Optional[Dict[str, Any]] = None         # US-070
    online_rl_update: Optional[Dict[str, Any]] = None     # US-071
    system_health_report: Optional[Dict[str, Any]] = None  # US-072
    aura_patterns_detected: int = 0                          # Phase 2 cross-domain
    observation_patterns: int = 0                              # Phase 29 (US-177)
    observation_recommendations: int = 0                       # Phase 29 (US-177)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "scan_duration_secs": self.scan_duration_secs,
            "pairs_scanned": self.pairs_scanned,
            "tradeable_setups": self.tradeable_setups,
            "trades_executed": self.trades_executed,
            "learnings_extracted": self.learnings_extracted,
            "rules_promoted": self.rules_promoted,
            "config_adjustments": self.config_adjustments,
            "observations_logged": self.observations_logged,
            "agent_health_report": self.agent_health_report,
            "stress_state": self.stress_state,
            "online_rl_update": self.online_rl_update,
            "system_health_report": self.system_health_report,
            "aura_patterns_detected": self.aura_patterns_detected,
            "observation_patterns": self.observation_patterns,
            "observation_recommendations": self.observation_recommendations,
            "errors": self.errors,
        }


class Orchestrator:
    """Coordinates the full scan→trade→learn→tune improvement cycle.

    Each call to `run_cycle()` performs:
    1. SCAN  — Run multi-pair scanner with agent consensus
    2. LOG   — Record observations from scan (even non-tradeable)
    3. TRADE — Execute qualifying setups via OANDA (if auto_execute)
    4. SYNC  — Sync closed trades from OANDA + run RL weight updates
    5. LEARN — Extract learnings from trade outcomes
    6. PROMOTE — Check if any learnings should become rules
    7. TUNE  — Apply rules to config parameters
    8. STATE — Update session state
    9. TRACK — Record improvement metrics
    """

    def __init__(self, project_root: Optional[str] = None, auto_execute: bool = False):
        self._root = Path(project_root) if project_root else Path.cwd()
        self._auto_execute = auto_execute
        self._cycle_count = 0
        self._session_start = datetime.utcnow()

        # Initialize all automation modules (lazy — they handle missing deps gracefully)
        self._state = None
        self._learner = None
        self._tuner = None
        self._tracker = None
        self._observer = None
        self._agent_health = None
        self._macro_stress = None       # US-070
        self._online_rl = None          # US-071
        self._qa_pipeline = None        # US-072
        self._gate_health = None        # US-072
        self._dynamic_hedge = None      # US-077
        self._config_adjuster = None    # Phase 22 (US-137)
        self._observation_consumer = None  # Phase 29 (US-177)

    def _init_modules(self):
        """Lazy-initialize automation modules."""
        if self._state is not None:
            return  # Already initialized

        root = self._root

        try:
            from src.scanner.automation.state_engine import StateEngine
            self._state = StateEngine(state_path=root / ".claude" / "state.json")
        except Exception as e:
            logger.warning(f"StateEngine init failed: {e}")
            self._state = None

        try:
            from src.scanner.automation.learning_engine import LearningEngine
            self._learner = LearningEngine(
                learnings_path=root / ".claude" / "learnings.md",
                rules_path=root / ".claude" / "rules" / "trading.md",
            )
        except Exception as e:
            logger.warning(f"LearningEngine init failed: {e}")
            self._learner = None

        try:
            from src.scanner.automation.config_tuner import ConfigTuner
            self._tuner = ConfigTuner(
                rules_path=root / ".claude" / "rules" / "trading.md",
                adjustments_path=root / ".claude" / "config_adjustments.json",
                applied_rules_path=root / ".claude" / "config_applied_rules.json",
            )
        except Exception as e:
            logger.warning(f"ConfigTuner init failed: {e}")
            self._tuner = None

        try:
            from src.scanner.automation.improvement_tracker import ImprovementTracker
            self._tracker = ImprovementTracker(
                log_path=root / "trained_data" / "improvement_log.jsonl",
            )
        except Exception as e:
            logger.warning(f"ImprovementTracker init failed: {e}")
            self._tracker = None

        try:
            from src.scanner.automation.observation_log import ObservationLog
            self._observer = ObservationLog(
                log_path=root / "trained_data" / "observations.jsonl",
            )
        except Exception as e:
            logger.warning(f"ObservationLog init failed: {e}")
            self._observer = None

        try:
            from src.scanner.automation.agent_health import AgentHealthMonitor
            self._agent_health = AgentHealthMonitor()
        except Exception as e:
            logger.warning(f"AgentHealthMonitor init failed: {e}")
            self._agent_health = None

        try:
            from src.scanner.automation.macro_stress import MacroStressDetector
            self._macro_stress = MacroStressDetector()
        except Exception as e:
            logger.warning(f"MacroStressDetector init failed: {e}")
            self._macro_stress = None

        try:
            from src.scanner.automation.online_rl import OnlineWeightUpdater
            self._online_rl = OnlineWeightUpdater()
        except Exception as e:
            logger.warning(f"OnlineWeightUpdater init failed: {e}")
            self._online_rl = None

        try:
            from src.scanner.automation.qa_pipeline import QAPipeline
            self._qa_pipeline = QAPipeline()
        except Exception as e:
            logger.warning(f"QAPipeline init failed: {e}")
            self._qa_pipeline = None

        try:
            from src.scanner.automation.gate_health import GateHealthTracker
            self._gate_health = GateHealthTracker()
        except Exception as e:
            logger.warning(f"GateHealthTracker init failed: {e}")
            self._gate_health = None

        # _config may not be set — guard with getattr on self
        _cfg = getattr(self, "_config", None)
        if getattr(_cfg, "enable_dynamic_hedging", False):
            try:
                from src.scanner.automation.dynamic_hedging import DynamicHedgeManager
                self._dynamic_hedge = DynamicHedgeManager(
                    min_correlation=getattr(_cfg, "hedge_min_correlation", 0.65),
                )
            except Exception as e:
                logger.warning(f"DynamicHedgeManager init failed: {e}")
                self._dynamic_hedge = None

        # Aura feedback bridge (writes outcome signals for human engine)
        self._aura_bridge = None
        try:
            from src.aura.bridge.signals import FeedbackBridge
            self._aura_bridge = FeedbackBridge()
            logger.info("Aura feedback bridge initialized")
        except Exception as e:
            logger.debug(f"Aura bridge not available: {e}")

        # Bridge-domain recursive learner (override patterns → rule promotion)
        self._bridge_learner = None
        try:
            from src.recursive_intelligence.learner import RecursiveLearner
            from src.aura.patterns.override_extractor import OverridePatternExtractor
            self._bridge_learner = RecursiveLearner(
                domain="bridge",
                pattern_types=[
                    "override_win", "override_loss", "emotional_override",
                    "cognitive_override", "confidence_mismatch",
                ],
                promotion_threshold=3,
                learnings_path=root / ".aura" / "learnings_bridge.json",
                rules_path=root / ".aura" / "rules_bridge.json",
                extractors={"override": OverridePatternExtractor()},
            )
            logger.info("Bridge-domain recursive learner initialized")
        except Exception as e:
            logger.debug(f"Bridge learner not available: {e}")

        # Phase 4: Bridge rules engine, rule promoter, self-model validator
        self._bridge_rules = None
        self._rule_promoter = None
        self._self_model_validator = None
        try:
            from src.aura.bridge.rules_engine import BridgeRulesEngine
            self._bridge_rules = BridgeRulesEngine(
                rules_path=root / ".aura" / "bridge" / "active_rules.json"
            )
            logger.info("Bridge rules engine initialized")
        except Exception as e:
            logger.debug(f"Bridge rules engine not available: {e}")

        try:
            from src.aura.patterns.rule_promoter import AuraRulePromoter
            if self._bridge_rules:
                self._rule_promoter = AuraRulePromoter(
                    rules_engine=self._bridge_rules,
                    promotion_log_path=root / ".aura" / "promotion_log.jsonl",
                )
                logger.info("Aura rule promoter initialized")
        except Exception as e:
            logger.debug(f"Aura rule promoter not available: {e}")

        try:
            from src.aura.core.self_model_validator import SelfModelValidator
            self._self_model_validator = SelfModelValidator(
                auto_remediate=True,
                report_dir=root / ".aura" / "validation_reports",
            )
            logger.info("Self-model validator initialized")
        except Exception as e:
            logger.debug(f"Self-model validator not available: {e}")

        # Phase 3: Override predictor + Readiness v2
        self._override_predictor = None
        self._readiness_v2 = None
        try:
            from src.aura.prediction.override_predictor import OverridePredictor
            self._override_predictor = OverridePredictor(
                model_path=root / ".aura" / "models" / "override_predictor.json"
            )
            logger.info("Override predictor initialized")
        except Exception as e:
            logger.debug(f"Override predictor not available: {e}")

        try:
            from src.aura.prediction.readiness_v2 import ReadinessModelV2
            self._readiness_v2 = ReadinessModelV2(
                model_path=root / ".aura" / "models" / "readiness_v2.json"
            )
            logger.info("Readiness v2 model initialized")
        except Exception as e:
            logger.debug(f"Readiness v2 not available: {e}")

        # Aura pattern engine (cross-domain correlations — Phase 2)
        self._aura_patterns = None
        try:
            from src.aura.patterns.engine import PatternEngine
            self._aura_patterns = PatternEngine(
                trade_journal_path=root / "trained_data" / "trade_journal_rl.json",
            )
            logger.info("Aura pattern engine initialized")
        except Exception as e:
            logger.debug(f"Aura pattern engine not available: {e}")

        # Phase 29 (US-177): Observation consumer for pattern detection in run_cycle
        try:
            from src.scanner.automation.observation_consumer import ObservationConsumer
            self._observation_consumer = ObservationConsumer(
                observations_path=root / "trained_data" / "observations.jsonl",
            )
            logger.info("ObservationConsumer initialized in orchestrator")
        except Exception as e:
            logger.debug(f"ObservationConsumer not available: {e}")
            self._observation_consumer = None

        # Phase 22 (US-137): Central config adjustment manager
        try:
            from src.scanner.automation.config_adjuster import ConfigAdjuster
            self._config_adjuster = ConfigAdjuster()
            logger.info("ConfigAdjuster initialized in orchestrator")
        except Exception as e:
            logger.debug(f"ConfigAdjuster not available: {e}")
            self._config_adjuster = None

        # Phase 28 (US-169): PRD Agent Chain — event-driven PRD completion watcher
        # Runs as a background listener: PRD complete → gap wirer → code reviewer
        self._prd_chain = None
        try:
            from src.scanner.automation.prd_agent_chain import PRDAgentChain
            self._prd_chain = PRDAgentChain(
                project_root=root,
                ralph_dir=root / ".claude" / "ralph",
                auto_start=False,  # Start explicitly when watch mode begins
            )
            logger.info("PRDAgentChain initialized in orchestrator")
        except Exception as e:
            logger.debug(f"PRDAgentChain not available: {e}")

    def run_cycle(
        self,
        profile: str = "balanced",
        pairs: Optional[List[str]] = None,
        granularity: str = "H1",
    ) -> OrchestrationResult:
        """Execute a single full improvement cycle.

        Args:
            profile: Scan profile (balanced, conservative, aggressive, smart)
            pairs: Currency pairs to scan (None = all defaults)
            granularity: OANDA granularity (H1, M15, H4, D)

        Returns:
            OrchestrationResult with metrics from the cycle.
        """
        self._init_modules()
        self._cycle_count += 1
        result = OrchestrationResult(
            cycle_id=f"cycle_{self._cycle_count}_{datetime.utcnow().strftime('%H%M%S')}",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        # ── Step 1: SCAN ─────────────────────────────────────────
        scan_result = None
        scanner = None  # Retained for Phase 18 outcome feedback
        t0 = time.time()
        try:
            from src.scanner.config import ScannerConfig
            from src.scanner.engine import Scanner

            config = ScannerConfig(profile=profile)
            config.granularity = granularity

            # Apply any tuner adjustments
            if self._tuner:
                try:
                    adjustments = self._tuner.apply_to_config(config)
                    result.config_adjustments = len(adjustments)
                except Exception as e:
                    logger.debug(f"Config tuner skipped: {e}")

            scanner = Scanner(config)
            scan_result = scanner.scan(pairs=pairs)
            result.scan_duration_secs = round(time.time() - t0, 1)

            if scan_result and scan_result.analyses:
                result.pairs_scanned = len(scan_result.analyses)
                result.tradeable_setups = sum(
                    1 for a in scan_result.analyses
                    if getattr(a, "gates_passed", False) and getattr(a, "is_tradeable", False)
                )
        except Exception as e:
            error_class = type(e).__name__
            result.errors.append(f"scan_failed ({error_class}): {e}")
            logger.error(
                "Scan failed [%s]: %s — check model compatibility and data pipeline",
                error_class, e,
            )

        # ── Step 2: LOG OBSERVATIONS ─────────────────────────────
        if scan_result and self._observer:
            try:
                for analysis in scan_result.analyses:
                    result.observations_logged += self._observer.log_from_analysis(analysis)
            except Exception as e:
                result.errors.append(f"observation_log_failed: {e}")

        # ── Step 2b: MACRO STRESS UPDATE (US-070) ─────────────
        if self._macro_stress and scan_result:
            try:
                # Extract dominant regime from scan analyses
                from collections import Counter as _Counter
                regimes = [
                    str(getattr(a, "volatility_regime", "NORMAL") or "NORMAL").upper()
                    for a in (scan_result.analyses or [])
                ]
                dominant_regime = _Counter(regimes).most_common(1)[0][0] if regimes else "NORMAL"

                # Average spread ratio from analysis contexts
                spread_ratios = [
                    getattr(a, "spread_pips", 0) / 2.0
                    for a in (scan_result.analyses or [])
                    if getattr(a, "spread_pips", 0) > 0
                ]
                avg_spread_ratio = (sum(spread_ratios) / len(spread_ratios)) if spread_ratios else 1.0

                self._macro_stress.update(
                    regime=dominant_regime,
                    avg_spread_ratio=avg_spread_ratio,
                )
                stress_mod = self._macro_stress.get_stress_modifier()
                result.stress_state = {
                    "stress_score": getattr(self._macro_stress, "_stress_score", 0),
                    "stress_modifier": stress_mod,
                    "regime": dominant_regime,
                }
                logger.debug("Macro stress updated: modifier=%.2f", stress_mod)
            except Exception as e:
                logger.debug(f"Macro stress update skipped: {e}")

        # ── Step 2d: DYNAMIC HEDGE EVALUATION (US-077) ──────────
        if self._dynamic_hedge and result.stress_state:
            try:
                stress_mod = result.stress_state.get("stress_modifier", 1.0)
                stress_score = result.stress_state.get("stress_score", 0.0)
                # Get open positions from execution manager if available
                _open_positions: list = []
                try:
                    if hasattr(self._scanner, "_executor") and self._scanner._executor:
                        _acct = self._scanner._executor.get_account_status()
                        # Parse open positions from account status
                        if isinstance(_acct, tuple) and len(_acct) >= 1:
                            _open_positions = _acct[0] if isinstance(_acct[0], list) else []
                except Exception:
                    pass
                hedge_status = self._dynamic_hedge.evaluate(
                    open_positions=_open_positions,
                    stress_modifier=float(stress_mod),
                    stress_score=float(stress_score),
                )
                if hedge_status.is_hedging_active or hedge_status.hedges_to_close:
                    logger.info(
                        "DynamicHedge: active=%s, candidates=%d, to_close=%d",
                        hedge_status.is_hedging_active,
                        len(hedge_status.candidates),
                        len(hedge_status.hedges_to_close),
                    )
            except Exception as e:
                logger.debug(f"Dynamic hedge evaluation skipped: {e}")

        # ── Step 2c: ONLINE RL MICRO-UPDATE (US-071) ──────────
        if self._online_rl:
            try:
                adjustments = self._online_rl.maybe_update(self._cycle_count)
                if adjustments:
                    from dataclasses import asdict as _asdict
                    result.online_rl_update = {
                        "cycle": self._cycle_count,
                        "adjustments": [_asdict(a) for a in adjustments],
                        "count": len(adjustments),
                    }
                    logger.debug("Online RL micro-update: %d adjustments at cycle %d", len(adjustments), self._cycle_count)
            except Exception as e:
                logger.debug(f"Online RL update skipped: {e}")

        # ── Step 3: RECORD SCAN TO STATE ─────────────────────────
        if self._state:
            try:
                self._state.increment_scan_cycle()
            except Exception as e:
                logger.debug(f"State scan record skipped: {e}")

        # ── Step 4: SYNC CLOSED TRADES + RL ──────────────────────
        closed_trades = []
        try:
            from src.scanner.execution import ExecutionManager
            em = ExecutionManager()
            sync_result = em.sync_closed_trades_rl(scanner=scanner) or {}
            # sync_closed_trades_rl returns a dict with metadata, not a list.
            # Read the journal directly to get trade dicts with outcomes for learning.
            if isinstance(sync_result, dict) and sync_result.get("trades_synced", 0) > 0:
                import json
                journal_path = self._root / "trained_data" / "trade_journal_rl.json"
                try:
                    journal = json.loads(journal_path.read_text())
                    closed_trades = [t for t in journal if isinstance(t, dict) and t.get("outcome")]
                except Exception:
                    pass
            logger.info(f"RL sync result: {sync_result}")
        except Exception as e:
            logger.debug(f"RL sync skipped: {e}")

        # ── Step 4b: AGENT HEALTH ATTRIBUTION (US-068) ─────────
        if closed_trades and self._agent_health:
            for trade in closed_trades:
                try:
                    outcome = trade.get("outcome", {})
                    trade_won = outcome.get("trade_won", False)
                    pair = trade.get("pair", "")
                    agents_data = trade.get("agents", {})
                    agent_verdicts = []
                    if isinstance(agents_data, dict):
                        agent_verdicts = agents_data.get("agent_reasons", [])
                    self._agent_health.record_outcome(
                        agent_verdicts=agent_verdicts,
                        trade_won=trade_won,
                        pair=pair,
                    )
                except Exception as e:
                    logger.debug(f"Agent health recording failed: {e}")
            try:
                result.agent_health_report = self._agent_health.get_health_report()
            except Exception as e:
                logger.debug(f"Agent health report failed: {e}")

        # ── Step 5: EXTRACT LEARNINGS ────────────────────────────
        if closed_trades and self._learner:
            for trade in closed_trades:
                try:
                    insights = self._learner.analyze_trade(trade)
                    if insights:
                        self._learner.append_to_learnings(insights)
                        result.learnings_extracted += len(insights)
                except Exception as e:
                    result.errors.append(f"learning_extraction_failed: {e}")

        # ── Step 5a: OVERRIDE PATTERN EXTRACTION (PRD §7.4) ────
        # Analyze resolved override events through the learning engine
        if self._learner and self._aura_bridge:
            try:
                overrides = self._aura_bridge.get_recent_overrides(limit=50)
                resolved_overrides = [
                    o.to_dict() for o in overrides if o.outcome is not None
                ]
                if resolved_overrides:
                    override_entries = self._learner.analyze_overrides_batch(
                        resolved_overrides
                    )
                    if override_entries:
                        self._learner.append_to_learnings(override_entries)
                        result.learnings_extracted += len(override_entries)
                        logger.info(
                            "Override patterns extracted: %d from %d events",
                            len(override_entries), len(resolved_overrides),
                        )
            except Exception as e:
                logger.debug(f"Override pattern extraction skipped: {e}")

        # ── Step 5a2: BRIDGE RECURSIVE LEARNER — override → promote ──
        # Feed resolved overrides into the domain-agnostic recursive learner
        # for independent promotion tracking (separate from Buddy's trading rules)
        if self._bridge_learner and self._aura_bridge:
            try:
                overrides = self._aura_bridge.get_recent_overrides(limit=50)
                for override in overrides:
                    if override.outcome is not None:
                        self._bridge_learner.observe(override.to_dict())
                # Check if any override patterns should become rules
                promoted = self._bridge_learner.check_promotions()
                if promoted:
                    logger.info(
                        "Bridge learner promoted %d override patterns to rules",
                        len(promoted),
                    )
            except Exception as e:
                logger.debug(f"Bridge recursive learner skipped: {e}")

        # ── Step 5b: EXIT REASON PATTERN EXTRACTION (US-067) ───
        if self._learner:
            try:
                exit_patterns = self._learner.extract_exit_reason_patterns(
                    journal_path=self._root / "trained_data" / "trade_journal_rl.json",
                )
                if exit_patterns:
                    self._learner.append_to_learnings(exit_patterns)
                    result.learnings_extracted += len(exit_patterns)
                    logger.info("Exit reason patterns extracted: %d", len(exit_patterns))
            except Exception as e:
                logger.debug(f"Exit reason pattern extraction skipped: {e}")

        # ── Step 6: CHECK RULE PROMOTIONS ────────────────────────
        if self._learner and result.learnings_extracted > 0:
            try:
                promotions = self._learner.check_promotions()
                result.rules_promoted = len(promotions)
            except Exception as e:
                logger.debug(f"Promotion check skipped: {e}")

        # ── Step 7: CONSOLIDATE IF NEEDED ────────────────────────
        if self._learner:
            try:
                audit = self._learner.audit()
                if audit.get("total_learnings", 0) > 30:
                    self._learner.consolidate()
            except Exception as e:
                logger.debug(f"Consolidation skipped: {e}")

        # ── Step 8: UPDATE STATE ─────────────────────────────────
        if self._state:
            try:
                self._state.save_state(
                    goal="autonomous_orchestration",
                    status="running" if self._auto_execute else "scanning",
                    done=[result.cycle_id],
                    next_action="next_scan_cycle",
                )
            except Exception as e:
                logger.debug(f"State update skipped: {e}")

        # ── Step 8b: SYSTEM HEALTH DIAGNOSTICS (US-072) ─────────
        # Run every 10 cycles to avoid overhead
        if self._cycle_count % 10 == 0:
            try:
                result.system_health_report = self._get_system_health_report()
                health_score = (result.system_health_report or {}).get("overall_score", 1.0)
                if health_score < 0.6:
                    logger.warning(
                        "System health degraded: score=%.2f — check diagnostics",
                        health_score,
                    )
            except Exception as e:
                logger.debug(f"System health report skipped: {e}")

        # ── Step 8c: APPLY CONFIG ADJUSTMENTS (Phase 22, US-137) ──
        if self._config_adjuster and scanner is not None:
            try:
                applied = self._config_adjuster.apply_adjustments(
                    scanner.config, self._cycle_count
                )
                if applied:
                    result.config_adjustments = (result.config_adjustments or 0) + len(applied)
                    logger.info(
                        "ConfigAdjuster applied %d adjustments: %s",
                        len(applied),
                        ", ".join(a["key"] for a in applied),
                    )
                self._config_adjuster.save_state()
            except Exception as e:
                logger.debug(f"ConfigAdjuster apply skipped: {e}")

        # ── Step 8d: OBSERVATION CONSUMER — pattern detection (Phase 29, US-177) ──
        # Run every 5 cycles to consume observations and detect actionable patterns
        if self._observation_consumer and self._cycle_count % 5 == 0:
            try:
                _oc = self._observation_consumer
                _oc.consume_observations()
                _patterns = _oc.detect_patterns()
                _recommendations = _oc.recommend_adjustments()
                _alerts = _oc.check_alerts(window_hours=6)

                result.observation_patterns = len(_patterns) if _patterns else 0
                result.observation_recommendations = len(_recommendations) if _recommendations else 0

                if _alerts:
                    for _alert in _alerts:
                        logger.warning(
                            "ObservationConsumer alert: category=%s count=%d in_window=%s",
                            _alert.get("category", "unknown"),
                            _alert.get("count", 0),
                            _alert.get("window_hours", "?"),
                        )

                if _recommendations:
                    logger.info(
                        "ObservationConsumer: %d patterns, %d recommendations",
                        result.observation_patterns,
                        result.observation_recommendations,
                    )

                _oc.save_state()
            except Exception as e:
                logger.debug(f"ObservationConsumer skipped: {e}")

        # ── Step 8d: OBSERVATION CONSUMER PATTERN DETECTION (Phase 29, US-177) ──
        # Run every 5 cycles to detect recurring patterns and generate recommendations.
        if self._observation_consumer and self._cycle_count % 5 == 0:
            try:
                consumed = self._observation_consumer.consume_observations()
                patterns = self._observation_consumer.detect_patterns()
                result.observation_patterns = len(patterns)
                # Generate recommendations from detected patterns
                recommendations = []
                for p in patterns:
                    if p.get("confidence", 0) > 0.6:
                        recommendations.append(p)
                result.observation_recommendations = len(recommendations)
                # Log spike alerts
                for p in patterns:
                    if p.get("type") == "spike" or p.get("severity") == "high":
                        logger.warning(
                            "Observation spike: %s (count=%d, confidence=%.2f)",
                            p.get("pattern", "unknown"),
                            p.get("count", 0),
                            p.get("confidence", 0),
                        )
                if consumed > 0 or patterns:
                    logger.info(
                        "ObservationConsumer: consumed=%d, patterns=%d, recommendations=%d",
                        consumed, len(patterns), len(recommendations),
                    )
            except Exception as e:
                logger.debug(f"Observation consumer pattern detection skipped: {e}")

        # ── Step 9: TRACK IMPROVEMENT ────────────────────────────
        if self._tracker:
            try:
                self._tracker.record_session(
                    trades=closed_trades,
                    learnings_added=result.learnings_extracted,
                    rules_promoted=result.rules_promoted,
                )
            except Exception as e:
                logger.debug(f"Improvement tracking skipped: {e}")

        # ── Step 10: AURA BRIDGE — Write outcome signal ─────────
        if self._aura_bridge:
            try:
                from src.aura.bridge.signals import OutcomeSignal

                # Compute 7-day win rate from trade journal
                win_rate_7d = 0.5
                try:
                    journal_path = self._root / "trained_data" / "trade_journal_rl.json"
                    if journal_path.exists():
                        import json as _json
                        journal = _json.loads(journal_path.read_text())
                        recent = journal[-50:] if isinstance(journal, list) else []
                        outcomes = [t for t in recent if t.get("outcome")]
                        if outcomes:
                            wins = sum(1 for t in outcomes if t.get("outcome", {}).get("trade_won"))
                            win_rate_7d = wins / len(outcomes)
                except Exception:
                    pass

                # Determine streak
                streak = "neutral"
                if closed_trades:
                    last_3 = closed_trades[-3:]
                    if all(t.get("outcome", {}).get("trade_won") for t in last_3):
                        streak = "winning"
                    elif all(not t.get("outcome", {}).get("trade_won") for t in last_3):
                        streak = "losing"

                dominant_regime = "NORMAL"
                if result.stress_state:
                    dominant_regime = result.stress_state.get("regime", "NORMAL")

                # Phase 27 (US-168): Detect manual trade overrides
                _override_events = []
                try:
                    # Compare OANDA trade IDs vs journal trade IDs
                    _journal_tids = set()
                    if isinstance(journal, list):
                        _journal_tids = {
                            str(t.get("trade_id", ""))
                            for t in journal if t.get("trade_id")
                        }
                    if self.scanner is not None:
                        _em = getattr(self.scanner, "_execution_manager", None)
                        if _em is not None:
                            _open = _em.monitor_open_trades()
                            for _ot in (_open or []):
                                _tid = str(_ot.get("id", _ot.get("trade_id", "")))
                                if _tid and _tid not in _journal_tids:
                                    _override_events.append({
                                        "type": "manual_trade",
                                        "trade_id": _tid,
                                        "pair": _ot.get("pair", _ot.get("instrument", "")),
                                        "units": _ot.get("units", 0),
                                    })
                    if _override_events:
                        logger.info(
                            "US-168: Detected %d manual override trades: %s",
                            len(_override_events),
                            [e["pair"] for e in _override_events],
                        )
                        try:
                            from src.scanner.automation.observation_log import ObservationLog
                            ObservationLog().log_observation(
                                pair="SYSTEM",
                                category="manual_override_detected",
                                description=f"US-168: {len(_override_events)} manual trade(s) detected",
                                metadata={"overrides": _override_events},
                            )
                        except Exception:
                            pass
                except Exception as _ovr_err:
                    logger.debug(f"US-168: Override detection error: {_ovr_err}")

                outcome_signal = OutcomeSignal(
                    pnl_today=sum(
                        t.get("outcome", {}).get("realized_pl", 0)
                        for t in closed_trades
                    ),
                    win_rate_7d=win_rate_7d,
                    override_events=_override_events,
                    regime=dominant_regime,
                    streak=streak,
                    trades_today=result.trades_executed,
                    open_positions=0,
                    max_drawdown_today=0.0,
                )
                self._aura_bridge.write_outcome(outcome_signal)
                logger.debug("Aura bridge: outcome signal written")
            except Exception as e:
                logger.debug(f"Aura bridge outcome write skipped: {e}")

        # ── Step 11: AURA PATTERN ENGINE — Multi-tier pattern detection ──
        # T1+T2 run every 5th cycle; T3 (narrative arcs) every 30th cycle
        if self._aura_patterns and self._cycle_count % 5 == 0:
            try:
                from src.aura.core.self_model import SelfModelGraph
                _graph = SelfModelGraph()
                _convs = _graph.get_recent_conversations(limit=50)
                _readiness = _graph.get_readiness_history(limit=50)
                _graph.close()

                # T1 + T2 every 5th cycle
                t1_patterns = self._aura_patterns.run_t1(_convs, _readiness)
                t2_patterns = self._aura_patterns.run_t2(_convs, _readiness)
                t1_count = len(t1_patterns)
                t2_count = len(t2_patterns)
                t3_count = 0

                # T3 (narrative arcs) every 30th cycle — monthly cadence
                if self._cycle_count % 30 == 0:
                    # T3 needs broader history for arc detection
                    _graph2 = SelfModelGraph()
                    _convs_full = _graph2.get_recent_conversations(limit=500)
                    _readiness_full = _graph2.get_readiness_history(limit=500)
                    _graph2.close()

                    t3_patterns = self._aura_patterns.run_t3(
                        _convs_full, _readiness_full
                    )
                    t3_count = len(t3_patterns)

                result.aura_patterns_detected = t1_count + t2_count + t3_count

                if result.aura_patterns_detected > 0:
                    logger.info(
                        "Aura patterns: %d T1 + %d T2 + %d T3 patterns detected",
                        t1_count, t2_count, t3_count,
                    )
            except Exception as e:
                logger.debug(f"Aura pattern engine skipped: {e}")

        # ── Step 11b: AURA RULE PROMOTER — Pattern → Bridge Rule pipeline ──
        if self._rule_promoter and self._aura_patterns:
            try:
                all_patterns = self._aura_patterns.get_all_active_patterns()
                promotions = self._rule_promoter.scan_and_promote(all_patterns)
                if promotions:
                    logger.info(
                        "Aura rule promoter: %d patterns promoted to bridge rules",
                        len(promotions),
                    )
            except Exception as e:
                logger.debug(f"Aura rule promotion skipped: {e}")

        # ── Step 11c: BRIDGE RULES — Expire stale rules ──────────────
        if self._bridge_rules:
            try:
                expired = self._bridge_rules.expire_stale_rules()
                if expired:
                    logger.info("Bridge rules: %d rules expired", expired)
            except Exception as e:
                logger.debug(f"Bridge rule expiry skipped: {e}")

        # ── Step 11d: SELF-MODEL VALIDATION — Graph integrity (every 30th cycle) ──
        if self._self_model_validator and self._cycle_count % 30 == 0:
            try:
                from src.aura.core.self_model import SelfModelGraph
                _vgraph = SelfModelGraph()
                val_report = self._self_model_validator.validate(graph=_vgraph)
                _vgraph.close()
                logger.info(
                    "Self-model validation: health=%.0f/100, issues=%d, auto-fixes=%d",
                    val_report.health_score,
                    len(val_report.issues),
                    val_report.auto_remediations,
                )
            except Exception as e:
                logger.debug(f"Self-model validation skipped: {e}")

        # ── Step 12: PHASE 3 — Train/update prediction models ────────
        # Override predictor: retrain every 10th cycle on all resolved overrides
        if self._override_predictor and self._aura_bridge and self._cycle_count % 10 == 0:
            try:
                overrides = self._aura_bridge.get_recent_overrides(limit=200)
                resolved = [o.to_dict() for o in overrides if o.outcome is not None]
                if len(resolved) >= 5:
                    metrics = self._override_predictor.fit(resolved)
                    logger.info(
                        "Override predictor retrained: %d samples, %.1f%% accuracy",
                        metrics.get("samples", 0),
                        metrics.get("accuracy", 0) * 100,
                    )
            except Exception as e:
                logger.debug(f"Override predictor training skipped: {e}")

        # Readiness v2: feed today's readiness+outcome into training buffer
        if self._readiness_v2 and self._aura_bridge and closed_trades:
            try:
                readiness_signal = self._aura_bridge.read_readiness()
                if readiness_signal:
                    components = readiness_signal.get("components", {})
                    # Compute outcome quality from today's trades
                    wins = sum(1 for t in closed_trades if t.get("outcome", {}).get("trade_won"))
                    total = len(closed_trades)
                    win_rate = wins / total if total > 0 else 0.5
                    total_pnl = sum(t.get("outcome", {}).get("realized_pl", 0) for t in closed_trades)
                    # Normalize PnL to 0-1 range (±$500 maps to 0-1)
                    pnl_score = max(0.0, min(1.0, (total_pnl + 500) / 1000))
                    outcome_quality = win_rate * 0.6 + pnl_score * 0.4

                    self._readiness_v2.add_training_sample(
                        readiness_components=components,
                        trading_outcome_quality=outcome_quality,
                    )
                    logger.debug(
                        "Readiness v2: buffered sample (quality=%.2f, wr=%.0f%%, pnl=$%.0f)",
                        outcome_quality, win_rate * 100, total_pnl,
                    )
            except Exception as e:
                logger.debug(f"Readiness v2 training sample skipped: {e}")

        # ── Step 13: Phase 25 (US-154): Learnings consolidation ──────
        if self._cycle_count % 50 == 0:
            try:
                _learnings_path = self._root / ".claude" / "learnings.md"
                _config_adj_path = self._root / ".claude" / "config_adjustments.json"
                _archive_path = self._root / ".claude" / "learnings_archive.md"

                # Consolidate learnings.md if > 30 entries
                if _learnings_path.exists():
                    _lines = _learnings_path.read_text(encoding="utf-8").strip().split("\n")
                    # Count entry lines (start with "- " or "## ")
                    _entries = [l for l in _lines if l.strip().startswith("- ") or l.strip().startswith("## 20")]
                    if len(_entries) > 30:
                        # Archive entries older than 30 days
                        from datetime import datetime as _dt154, timedelta as _td154
                        _cutoff = (_dt154.now() - _td154(days=30)).strftime("%Y-%m-%d")
                        _keep = []
                        _archive = []
                        for l in _lines:
                            # Lines with dates before cutoff go to archive
                            if any(d in l for d in [f"20{y}" for y in range(24, 27)] if d < _cutoff):
                                _archive.append(l)
                            else:
                                _keep.append(l)
                        if _archive:
                            # Append to archive file
                            with open(_archive_path, "a", encoding="utf-8") as _af:
                                _af.write(f"\n# Archived {_dt154.now().strftime('%Y-%m-%d')}\n")
                                _af.write("\n".join(_archive) + "\n")
                            # Rewrite learnings with kept entries
                            _learnings_path.write_text("\n".join(_keep) + "\n", encoding="utf-8")
                            logger.info(
                                f"US-154: Archived {len(_archive)} old learnings "
                                f"(kept {len(_keep)})"
                            )

                # Prune config_adjustments.json if > 100 entries
                if _config_adj_path.exists():
                    import json as _j154
                    try:
                        _adj_data = _j154.loads(_config_adj_path.read_text(encoding="utf-8"))
                        if isinstance(_adj_data, list) and len(_adj_data) > 100:
                            from datetime import datetime as _dt154b, timedelta as _td154b
                            _cutoff_ts = (_dt154b.now() - _td154b(days=30)).isoformat()
                            _pruned = [
                                a for a in _adj_data
                                if a.get("timestamp", "") > _cutoff_ts
                            ]
                            _removed = len(_adj_data) - len(_pruned)
                            if _removed > 0:
                                _config_adj_path.write_text(
                                    _j154.dumps(_pruned, indent=2), encoding="utf-8"
                                )
                                logger.info(
                                    f"US-154: Pruned {_removed} old config adjustments "
                                    f"(kept {len(_pruned)})"
                                )
                    except Exception:
                        pass
            except Exception as _consol_err:
                logger.debug(f"US-154: Consolidation skipped: {_consol_err}")

        return result

    def _get_system_health_report(self) -> Dict[str, Any]:
        """Combine QA score + gate health + agent health into a unified report.

        US-072: Runs periodically (every 10 cycles) to surface system degradation.
        """
        report: Dict[str, Any] = {"timestamp": datetime.utcnow().isoformat() + "Z"}
        scores: List[float] = []

        # QA pipeline score (0-100 scale → normalize to 0-1)
        if self._qa_pipeline:
            try:
                qa_report = self._qa_pipeline.run_full_audit()
                qa_dict = qa_report.to_dict() if hasattr(qa_report, "to_dict") else {"score": 100.0}
                qa_score = min(qa_dict.get("score", 100.0) / 100.0, 1.0)
                report["qa_pipeline"] = qa_dict
                scores.append(qa_score)
            except Exception as e:
                report["qa_pipeline"] = {"error": str(e)}
                scores.append(0.5)

        # Gate health
        if self._gate_health:
            try:
                gate_report = self._gate_health.get_gate_health()
                report["gate_health"] = gate_report
                # Score based on fraction of non-degraded gates
                if isinstance(gate_report, dict):
                    total = len(gate_report)
                    degraded = sum(
                        1 for g in gate_report.values()
                        if isinstance(g, dict) and g.get("degraded", False)
                    )
                    gate_score = (total - degraded) / total if total > 0 else 1.0
                    scores.append(gate_score)
            except Exception as e:
                report["gate_health"] = {"error": str(e)}

        # Agent health summary
        if self._agent_health:
            try:
                health = self._agent_health.get_health_report()
                report["agent_health"] = health
                summary = health.get("summary", {}) if isinstance(health, dict) else {}
                total_agents = summary.get("total_tracked", 0)
                stale_agents = summary.get("stale_agents", 0)
                agent_score = (total_agents - stale_agents) / total_agents if total_agents > 0 else 1.0
                scores.append(agent_score)
            except Exception as e:
                report["agent_health"] = {"error": str(e)}

        # Overall composite score
        report["overall_score"] = round(sum(scores) / len(scores), 3) if scores else 1.0

        return report

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for the dashboard.

        Returns a dict suitable for JSON serialization with:
        - Module availability (which components are active)
        - Last cycle results
        - Improvement trends
        - Learning stats
        """
        self._init_modules()

        status: Dict[str, Any] = {
            "modules": {
                "state_engine": self._state is not None,
                "learning_engine": self._learner is not None,
                "config_tuner": self._tuner is not None,
                "improvement_tracker": self._tracker is not None,
                "observation_log": self._observer is not None,
                "agent_health": self._agent_health is not None,
                "macro_stress": self._macro_stress is not None,
                "online_rl": self._online_rl is not None,
                "qa_pipeline": self._qa_pipeline is not None,
                "gate_health": self._gate_health is not None,
                "dynamic_hedge": self._dynamic_hedge is not None,
                "bridge_learner": self._bridge_learner is not None,
                "config_adjuster": self._config_adjuster is not None,
                "aura_bridge": self._aura_bridge is not None,
                "aura_patterns": self._aura_patterns is not None,
                "override_predictor": self._override_predictor is not None,
                "readiness_v2": self._readiness_v2 is not None,
                "bridge_rules_engine": self._bridge_rules is not None,
                "rule_promoter": self._rule_promoter is not None,
                "self_model_validator": self._self_model_validator is not None,
                "prd_agent_chain": self._prd_chain is not None,
                "observation_consumer": self._observation_consumer is not None,  # Phase 29 (US-177)
            },
            "session": {
                "started": self._session_start.isoformat() + "Z",
                "cycles_completed": self._cycle_count,
                "auto_execute": self._auto_execute,
            },
        }

        # Add state info
        if self._state:
            try:
                status["state"] = self._state.load_state()
            except Exception:
                status["state"] = {}

        # Add improvement trend
        if self._tracker:
            try:
                status["improvement_trend"] = self._tracker.get_trend()
            except Exception:
                status["improvement_trend"] = {}

        # Add learning audit
        if self._learner:
            try:
                status["learning_audit"] = self._learner.audit()
            except Exception:
                status["learning_audit"] = {}

        # Add config adjuster status (Phase 22, US-137)
        if self._config_adjuster:
            try:
                status["config_adjuster"] = {
                    "pending": len(getattr(self._config_adjuster, "_pending", {})),
                    "history_count": len(getattr(self._config_adjuster, "_history", [])),
                }
            except Exception:
                status["config_adjuster"] = {}

        # Add pattern engine status (T1+T2+T3 + cloud synthesis)
        if self._aura_patterns:
            try:
                status["pattern_engine"] = self._aura_patterns.get_status()
            except Exception:
                status["pattern_engine"] = {}

        # Add bridge rules status
        if self._bridge_rules:
            try:
                status["bridge_rules"] = self._bridge_rules.get_rules_summary()
            except Exception:
                status["bridge_rules"] = {}

        # Add self-model validation status
        if self._self_model_validator:
            try:
                latest = self._self_model_validator.get_latest_report()
                if latest:
                    status["self_model_health"] = {
                        "health_score": latest.health_score,
                        "issues_count": len(latest.issues),
                        "auto_remediations": latest.auto_remediations,
                        "last_validated": latest.timestamp,
                    }
            except Exception:
                pass

        # Phase 27 (US-164): Include health registry scores when available
        if self.scanner is not None:
            _hr = getattr(self.scanner, "_health_registry", None)
            if _hr is not None:
                try:
                    _health_score = _hr.get_system_health_score()
                    _module_scores = {}
                    for _mod_name in getattr(_hr, "_modules", {}):
                        try:
                            _ms = _hr.get_module_health(_mod_name)
                            if _ms is not None:
                                _module_scores[_mod_name] = round(_ms, 3)
                        except Exception:
                            pass
                    status["health_registry"] = {
                        "system_health_score": round(_health_score, 3),
                        "module_scores": _module_scores,
                    }
                    # Alert on degradation
                    if _health_score < 0.70:
                        logger.warning(
                            "US-164: System health degraded — score=%.2f",
                            _health_score,
                        )
                        try:
                            from src.scanner.automation.observation_log import ObservationLog
                            ObservationLog().log_observation(
                                pair="SYSTEM",
                                category="health_degradation",
                                description=(
                                    f"US-164: System health score {_health_score:.2f} "
                                    f"below threshold 0.70"
                                ),
                                metadata={
                                    "health_score": round(_health_score, 3),
                                    "degraded_modules": {
                                        k: v for k, v in _module_scores.items()
                                        if v < 0.70
                                    },
                                },
                            )
                        except Exception:
                            pass
                except Exception as _hr_err:
                    logger.debug(f"US-164: Health registry status error: {_hr_err}")

        # Phase 28 (US-169): PRD agent chain status
        if self._prd_chain is not None:
            try:
                status["prd_agent_chain"] = self._prd_chain.get_status()
            except Exception as e:
                logger.debug(f"PRDAgentChain status error: {e}")

        return status

    def get_improvement_report(self) -> str:
        """Generate a human-readable improvement report."""
        self._init_modules()
        if self._tracker:
            try:
                return self._tracker.generate_report()
            except Exception:
                pass
        return "Improvement tracker not available."
