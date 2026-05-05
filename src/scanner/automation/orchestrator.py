"""ML Agent Orchestrator — full scan→trade→learn→tune loop.

Coordinates all automation modules into a single improvement cycle.
This is the brain of Buddy's autonomous evolution.

Architecture:
    Scanner → Agents → Gates → Execution → OANDA
        ↑                                      ↓
        └── Config Tuner ← Rules ← Learnings ← RL Feedback ← Trade Outcomes
"""

from __future__ import annotations

import atexit
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.observability import init_weave, op as weave_op, thread as weave_thread

logger = logging.getLogger(__name__)


class _LastKnownRegimeProbe:
    """Thin adapter so StagedDeployer can snapshot the regime at deploy time.

    The orchestrator updates `_regime` from the dominant per-scan regime
    every cycle (see `_macro_stress_update_dispatch`). StagedDeployer's
    Phase-1 Task-7 contract calls `regime_detector.current()` once at
    promotion-to-shadow and stores the result on `DeploymentRecord.regime_at_deploy`.
    Decoupled from `MarketRegimeDetector` to avoid forcing a fresh detection
    inside the deploy critical section — we use whatever regime the most
    recent scan settled on.
    """

    __slots__ = ("_regime",)

    def __init__(self) -> None:
        self._regime: Optional[str] = None

    def update(self, regime: Optional[str]) -> None:
        if regime is None:
            return
        # Normalise to upper-case to match the LOW/MED/HIGH/NORMAL convention
        # used by the agent team's volatility_regime field and the trading
        # rules in .claude/rules/trading.md.
        self._regime = str(regime).upper() or None

    def current(self) -> Optional[str]:
        return self._regime


@dataclass
class DispatchStep:
    """A single registered step in the orchestration dispatch table."""
    name: str
    callable: Callable[[], Any]
    condition: Callable[[], bool]
    interval: int = 1
    critical: bool = False
    result_key: Optional[str] = None


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
        # Observability: initialize Weave once. No-op unless BUDDY_WEAVE_ENABLED=1.
        init_weave()
        self._root = Path(project_root) if project_root else Path.cwd()
        self._auto_execute = auto_execute
        self._cycle_count = 0
        self._remediation_cycle_counter = 0  # Track cycles for drift remediator check
        self._perf_prd_cycle_counter = 0  # Track cycles for auto PRD generation check
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

        # Dispatch table for module calls — populated in _build_dispatch_table()
        self._dispatch_table: List[DispatchStep] = []
        self._dynamic_hedge = None      # US-077
        self._config_adjuster = None    # Phase 22 (US-137)
        self._observation_consumer = None  # Phase 29 (US-177)
        self._drift_remediator = None   # Drift auto-remediation loop
        self._perf_prd_gen = None       # Autonomous performance-driven PRD generator
        self._policy_engine = None      # Tier 7: Policy engine for action gates
        atexit.register(self.close)

    def _init_modules(self):
        """Lazy-initialize automation modules."""
        if self._state is not None:
            return  # Already initialized

        root = self._root

        try:
            from src.scanner.automation.state_engine import StateEngine
            self._state = StateEngine(state_path=root / ".claude" / "state.json")
        except Exception as e:
            logger.warning("StateEngine init failed: %s", e)
            self._state = None

        try:
            from src.scanner.automation.learning_engine import LearningEngine
            self._learner = LearningEngine(
                learnings_path=root / ".claude" / "learnings.md",
                rules_path=root / ".claude" / "rules" / "trading.md",
            )
        except Exception as e:
            logger.warning("LearningEngine init failed: %s", e)
            self._learner = None

        try:
            from src.scanner.automation.config_tuner import ConfigTuner
            self._tuner = ConfigTuner(
                rules_path=root / ".claude" / "rules" / "trading.md",
                adjustments_path=root / ".claude" / "config_adjustments.json",
                applied_rules_path=root / ".claude" / "config_applied_rules.json",
            )
        except Exception as e:
            logger.warning("ConfigTuner init failed: %s", e)
            self._tuner = None

        try:
            from src.scanner.automation.improvement_tracker import ImprovementTracker
            self._tracker = ImprovementTracker(
                log_path=root / "trained_data" / "improvement_log.jsonl",
            )
        except Exception as e:
            logger.warning("ImprovementTracker init failed: %s", e)
            self._tracker = None

        try:
            from src.scanner.automation.observation_log import ObservationLog
            self._observer = ObservationLog(
                log_path=root / "trained_data" / "observations.jsonl",
            )
        except Exception as e:
            logger.warning("ObservationLog init failed: %s", e)
            self._observer = None

        try:
            from src.scanner.automation.agent_health import AgentHealthMonitor
            self._agent_health = AgentHealthMonitor()
        except Exception as e:
            logger.warning("AgentHealthMonitor init failed: %s", e)
            self._agent_health = None

        try:
            from src.scanner.automation.macro_stress import MacroStressDetector
            self._macro_stress = MacroStressDetector()
        except Exception as e:
            logger.warning("MacroStressDetector init failed: %s", e)
            self._macro_stress = None

        try:
            from src.scanner.automation.online_rl import OnlineWeightUpdater
            self._online_rl = OnlineWeightUpdater()
        except Exception as e:
            logger.warning("OnlineWeightUpdater init failed: %s", e)
            self._online_rl = None

        try:
            from src.scanner.automation.qa_pipeline import QAPipeline
            self._qa_pipeline = QAPipeline()
        except Exception as e:
            logger.warning("QAPipeline init failed: %s", e)
            self._qa_pipeline = None

        try:
            from src.scanner.automation.gate_health import GateHealthTracker
            self._gate_health = GateHealthTracker()
        except Exception as e:
            logger.warning("GateHealthTracker init failed: %s", e)
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
                logger.warning("DynamicHedgeManager init failed: %s", e)
                self._dynamic_hedge = None

        # Aura feedback bridge (writes outcome signals for human engine)
        self._aura_bridge = None
        try:
            from src.aura.bridge.signals import FeedbackBridge
            self._aura_bridge = FeedbackBridge()
            logger.info("Aura feedback bridge initialized")
        except Exception as e:
            logger.debug("Aura bridge not available: %s", e)

        # Bridge-domain recursive learner (override patterns → rule promotion)
        self._bridge_learner = None
        try:
            from src.recursive_intelligence.learner import RecursiveLearner
            self._bridge_learner = RecursiveLearner(
                domain="bridge",
                pattern_types=[
                    "override_win", "override_loss", "emotional_override",
                    "cognitive_override", "confidence_mismatch",
                ],
                promotion_threshold=3,
                learnings_path=root / ".aura" / "learnings_bridge.json",
                rules_path=root / ".aura" / "rules_bridge.json",
                extractors={},
            )
            logger.info("Bridge-domain recursive learner initialized")
        except Exception as e:
            logger.debug("Bridge learner not available: %s", e)

        # Phase 4: Bridge rules engine (rule_promoter and self_model_validator deleted)
        self._bridge_rules = None
        try:
            from src.aura.bridge.rules_engine import BridgeRulesEngine
            self._bridge_rules = BridgeRulesEngine(
                rules_path=root / ".aura" / "bridge" / "active_rules.json"
            )
            logger.info("Bridge rules engine initialized")
        except Exception as e:
            logger.debug("Bridge rules engine not available: %s", e)



        # Phase 29 (US-177): Observation consumer for pattern detection in run_cycle
        try:
            from src.scanner.automation.observation_consumer import ObservationConsumer
            self._observation_consumer = ObservationConsumer(
                observations_path=root / "trained_data" / "observations.jsonl",
            )
            logger.info("ObservationConsumer initialized in orchestrator")
        except Exception as e:
            logger.debug("ObservationConsumer not available: %s", e)
            self._observation_consumer = None

        # Phase 22 (US-137): Central config adjustment manager
        try:
            from src.scanner.automation.config_adjuster import ConfigAdjuster
            self._config_adjuster = ConfigAdjuster()
            logger.info("ConfigAdjuster initialized in orchestrator")
        except Exception as e:
            logger.debug("ConfigAdjuster not available: %s", e)
            self._config_adjuster = None

        # Drift Auto-Remediation Loop
        try:
            from src.scanner.automation.drift_remediator import DriftRemediator
            self._drift_remediator = DriftRemediator(
                request_path=root / "trained_data" / "retrain_request.json",
                remediation_log_path=root / "trained_data" / "drift_remediation_log.jsonl",
                retrain_summary_path=root / "trained_data" / "retrain_all_summary.json",
                agent_weights_path=root / "trained_data" / "models" / "agent_weights.json",
                project_root=root,
            )
            logger.info("DriftRemediator initialized in orchestrator")
        except Exception as e:
            logger.debug("DriftRemediator not available: %s", e)
            self._drift_remediator = None

        # Autonomous Performance-Driven PRD Generator
        try:
            from src.scanner.automation.performance_prd_generator import PerformancePRDGenerator
            self._perf_prd_gen = PerformancePRDGenerator(project_root=root)
            logger.info("PerformancePRDGenerator initialized in orchestrator")
        except Exception as e:
            logger.debug("PerformancePRDGenerator not available: %s", e)
            self._perf_prd_gen = None

        # Phase 28 (US-169): PRD Agent Chain — event-driven PRD completion watcher
        # Runs as a background listener: PRD complete → gap wirer → code reviewer
        self._prd_chain = None
        try:
            from src.scanner.automation.prd_agent_chain import PRDAgentChain
            self._prd_chain = PRDAgentChain(
                project_root=root,
                ralph_dir=root / ".claude" / "ralph",
                ralph_tool="claude",
            )
            self._prd_chain.start()  # Start watcher immediately — PRD complete → chain fires
            logger.info("PRDAgentChain initialized and watcher started in orchestrator")
        except Exception as e:
            logger.debug("PRDAgentChain not available: %s", e)

        # Tier 7: Policy Engine
        try:
            from src.scanner.automation.policy_engine import get_policy_engine
            self._policy_engine = get_policy_engine()
            logger.info("Tier 7: PolicyEngine initialized in orchestrator")
        except Exception as e:
            logger.debug("Tier 7: PolicyEngine not available: %s", e)

        # Meta-cybernetic change pipeline (gated by config.enable_meta_manager).
        # Lazy: when disabled, no Claude subprocess and no approval queue work
        # ever fires.
        self._meta_manager = None
        try:
            _cfg = getattr(self, "_config", None)
            _meta_enabled = bool(getattr(_cfg, "enable_meta_manager", False))
            # Surface the flag via env so cheap callers (cycle_autonomy,
            # prd_agent_chain) can route via meta_manager.is_enabled() without
            # importing the heavy config singleton on every signal.
            import os
            os.environ["BUDDY_META_MANAGER_ENABLED"] = "1" if _meta_enabled else "0"
            if _meta_enabled:
                from src.scanner.automation.meta_manager import MetaManager
                from src.scanner.automation.staged_deployer import StagedDeployer
                from src.scanner.automation.change_eval import ChangeEvalHarness
                from src.scanner.automation.adjustment_approver import AdjustmentApprover
                eval_harness = None
                try:
                    from src.scanner.automation.replay_validator import ReplayValidator
                    eval_harness = ChangeEvalHarness(
                        replay_validator=ReplayValidator(),
                        qa_pipeline=self._qa_pipeline,
                        baseline_config=dict(getattr(_cfg, "__dict__", {}) or {}),
                    )
                except Exception as eh_err:
                    logger.warning(
                        "MetaManager: eval harness init failed: %s — "
                        "packages will abort at eval stage", eh_err,
                    )
                # Approver is required so canary deploys can move their own
                # proposals from pending → approved-history without tripping
                # ConfigAdjuster.apply_adjustments's BypassAttempt write-guard
                # (US-508). The meta pipeline is its own approval authority
                # for proposals it generated; this does NOT auto-approve
                # proposals from any other source.
                meta_approver = AdjustmentApprover()
                # Phase-1 Task-7 wiring: regime probe is updated once per scan
                # in _macro_stress_update_dispatch (the dominant regime is already
                # computed there). StagedDeployer.current() reads the last value
                # at promote-to-shadow time and stores it on DeploymentRecord.
                self._regime_probe = _LastKnownRegimeProbe()
                deployer = StagedDeployer(
                    config=_cfg,
                    config_adjuster=self._config_adjuster,
                    adjustment_approver=meta_approver,
                    shadow_cycles=getattr(_cfg, "staged_deploy_shadow_cycles", 20),
                    canary_trades=getattr(_cfg, "staged_deploy_canary_trades", 10),
                    regime_detector=self._regime_probe,
                )
                # Honor the project rule that Buddy's runtime is Claude-free.
                # When meta_manager_use_llm=False (default), inject a no-op
                # specialist_invoker so the 9-stage pipeline operates on
                # pure-Python policy alone (constitution + scorecard +
                # revert_by_id all gate deterministically). Specialists
                # enrich the change ledger; they don't gate anything.
                _meta_use_llm = bool(getattr(_cfg, "meta_manager_use_llm", False))
                _invoker = None  # MetaManager picks _default_specialist_invoker (calls Claude)
                if not _meta_use_llm:
                    _invoker = lambda mode, prompt: ""  # noqa: E731
                    logger.info(
                        "MetaManager: LLM disabled (meta_manager_use_llm=False) — "
                        "specialists return empty enrichment; constitution + "
                        "scorecard still gate."
                    )
                # Phase-1 Task-2/Task-4 wiring: episodic memory feeds the G3
                # intake-side query and the G7 historical_loss_rate constitution
                # clause. The agent team writes outcomes to the same backing
                # file (trained_data/episodic_memory.json) so this instance
                # naturally observes real trade history. Default constructor
                # reuses the canonical persistence path; concurrent writes
                # are safe under the GIL for the bounded per-trade rate.
                _meta_episodic = None
                try:
                    from src.scanner.automation.episodic_memory import EpisodicMemory
                    _meta_episodic = EpisodicMemory()
                except Exception as ep_err:
                    logger.warning(
                        "MetaManager: episodic_memory init failed: %s — "
                        "G3 intake query and G7 historical_loss_rate veto "
                        "will be no-ops in this session", ep_err,
                    )
                self._meta_manager = MetaManager(
                    config=_cfg,
                    eval_harness=eval_harness,
                    staged_deployer=deployer,
                    specialist_invoker=_invoker,
                    max_concurrent=getattr(_cfg, "meta_manager_max_concurrent", 1),
                    episodic_memory=_meta_episodic,
                )
                logger.info(
                    "MetaManager initialized in orchestrator "
                    "(episodic_memory=%s, regime_probe=wired)",
                    "wired" if _meta_episodic is not None else "DISABLED",
                )
        except Exception as e:
            logger.warning("MetaManager init failed: %s", e)
            self._meta_manager = None

        # Build dispatch table from initialized modules

        try:
            from src.scanner.automation.module_dispatcher import ModuleDispatcher
            self._module_dispatcher = ModuleDispatcher()
            logger.info("ModuleDispatcher initialized in orchestrator")
        except Exception as e:
            logger.debug("ModuleDispatcher not available: %s", e)
            self._module_dispatcher = None

        self._build_dispatch_table()

    def _build_dispatch_table(self):
        """Build the registry of module dispatch steps from initialized modules.

        Each step encapsulates a module call: its callable, condition, and interval.
        Steps are executed in order every run_cycle() with proper error isolation.
        """
        self._dispatch_table = []

        # Step 1: Log observations (every cycle if available)
        self._dispatch_table.append(DispatchStep(
            name="log_observations",
            callable=lambda: self._log_observations_dispatch(),
            condition=lambda: self._observer is not None,
            interval=1,
            critical=False,
        ))

        # Step 2: Macro stress update (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="macro_stress_update",
            callable=lambda: self._macro_stress_update_dispatch(),
            condition=lambda: self._macro_stress is not None,
            interval=1,
            critical=False,
        ))

        # Step 2d: Dynamic hedge evaluation (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="dynamic_hedge_evaluation",
            callable=lambda: self._dynamic_hedge_evaluation_dispatch(),
            condition=lambda: self._dynamic_hedge is not None,
            interval=1,
            critical=False,
        ))

        # Step 2c: Online RL micro-update (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="online_rl_update",
            callable=lambda: self._online_rl_update_dispatch(),
            condition=lambda: self._online_rl is not None,
            interval=1,
            critical=False,
        ))

        # Step 3: Record scan to state (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="state_scan_record",
            callable=lambda: self._state_scan_record_dispatch(),
            condition=lambda: self._state is not None,
            interval=1,
            critical=False,
        ))

        # Step 4: Sync closed trades + RL (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="sync_closed_trades_rl",
            callable=lambda: self._sync_closed_trades_rl_dispatch(),
            condition=lambda: True,
            interval=1,
            critical=False,
            result_key="closed_trades",
        ))

        # Step 4b: Agent health attribution (every cycle, depends on closed_trades)
        self._dispatch_table.append(DispatchStep(
            name="agent_health_attribution",
            callable=lambda: self._agent_health_attribution_dispatch(),
            condition=lambda: self._agent_health is not None,
            interval=1,
            critical=False,
        ))

        # Step 4c: Post-trade diagnostics + self-heal (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="post_trade_diagnostics",
            callable=lambda: self._post_trade_diagnostics_dispatch(),
            condition=lambda: True,
            interval=1,
            critical=False,
            result_key="post_trade_diag",
        ))

        # Step 5: Extract learnings (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="extract_learnings",
            callable=lambda: self._extract_learnings_dispatch(),
            condition=lambda: self._learner is not None,
            interval=1,
            critical=False,
        ))

        # Step 5a: Override pattern extraction (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="override_pattern_extraction",
            callable=lambda: self._override_pattern_extraction_dispatch(),
            condition=lambda: self._learner is not None and self._aura_bridge is not None,
            interval=1,
            critical=False,
        ))

        # Step 5a2: Bridge recursive learner (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="bridge_recursive_learner",
            callable=lambda: self._bridge_recursive_learner_dispatch(),
            condition=lambda: self._bridge_learner is not None and self._aura_bridge is not None,
            interval=1,
            critical=False,
        ))

        # Step 5b: Exit reason pattern extraction (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="exit_reason_pattern_extraction",
            callable=lambda: self._exit_reason_pattern_extraction_dispatch(),
            condition=lambda: self._learner is not None,
            interval=1,
            critical=False,
        ))

        # Step 6: Check rule promotions (every cycle if learnings extracted)
        self._dispatch_table.append(DispatchStep(
            name="check_rule_promotions",
            callable=lambda: self._check_rule_promotions_dispatch(),
            condition=lambda: self._learner is not None,
            interval=1,
            critical=False,
        ))

        # Step 7: Consolidate learnings (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="consolidate_learnings",
            callable=lambda: self._consolidate_learnings_dispatch(),
            condition=lambda: self._learner is not None,
            interval=1,
            critical=False,
        ))

        # Step 8: Update state (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="update_state",
            callable=lambda: self._update_state_dispatch(),
            condition=lambda: self._state is not None,
            interval=1,
            critical=False,
        ))

        # Step 8b: System health diagnostics (every 10 cycles)
        self._dispatch_table.append(DispatchStep(
            name="system_health_diagnostics",
            callable=lambda: self._system_health_diagnostics_dispatch(),
            condition=lambda: True,
            interval=10,
            critical=False,
        ))

        # Step 8c: Apply config adjustments (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="apply_config_adjustments",
            callable=lambda: self._apply_config_adjustments_dispatch(),
            condition=lambda: self._config_adjuster is not None,
            interval=1,
            critical=False,
        ))

        # Step 8d: Observation consumer pattern detection (every 5 cycles)
        self._dispatch_table.append(DispatchStep(
            name="observation_consumer_pattern_detection",
            callable=lambda: self._observation_consumer_pattern_detection_dispatch(),
            condition=lambda: self._observation_consumer is not None,
            interval=5,
            critical=False,
        ))

        # Step 9: Track improvement (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="track_improvement",
            callable=lambda: self._track_improvement_dispatch(),
            condition=lambda: self._tracker is not None,
            interval=1,
            critical=False,
        ))

        # Step 10: Aura bridge outcome signal (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="aura_bridge_outcome_signal",
            callable=lambda: self._aura_bridge_outcome_signal_dispatch(),
            condition=lambda: self._aura_bridge is not None,
            interval=1,
            critical=False,
        ))


        # Step 11c: Bridge rules expiry (every cycle)
        self._dispatch_table.append(DispatchStep(
            name="bridge_rules_expiry",
            callable=lambda: self._bridge_rules_expiry_dispatch(),
            condition=lambda: self._bridge_rules is not None,
            interval=1,
            critical=False,
        ))


        # Step 13: Learnings consolidation (every 50 cycles)
        self._dispatch_table.append(DispatchStep(
            name="learnings_consolidation",
            callable=lambda: self._learnings_consolidation_dispatch(),
            condition=lambda: True,
            interval=50,
            critical=False,
        ))

        # Step 13: Drift auto-remediation (configurable interval, default 20)
        self._dispatch_table.append(DispatchStep(
            name="drift_auto_remediation",
            callable=lambda: self._drift_auto_remediation_dispatch(),
            condition=lambda: self._drift_remediator is not None,
            interval=20,
            critical=False,
        ))

        # Step 14: Autonomous performance-driven PRD generation (every 50 cycles)
        self._dispatch_table.append(DispatchStep(
            name="performance_prd_generation",
            callable=lambda: self._performance_prd_generation_dispatch(),
            condition=lambda: self._perf_prd_gen is not None,
            interval=50,
            critical=False,
        ))

        # Step 15 (Tier 7): Policy engine environment audit (every 5 cycles)
        self._dispatch_table.append(DispatchStep(
            name="policy_engine_audit",
            callable=lambda: self._policy_engine_audit_dispatch(),
            condition=lambda: self._policy_engine is not None,
            interval=5,
            critical=False,
        ))

        # Step 16: Model freshness check (every 10 cycles)
        self._dispatch_table.append(DispatchStep(
            name="model_freshness_check",
            callable=lambda: self._model_freshness_check_dispatch(),
            condition=lambda: True,
            interval=10,
            critical=False,
        ))

        # Step 17: Meta-cybernetic change pipeline drain (every 5 cycles when enabled)
        self._dispatch_table.append(DispatchStep(
            name="meta_pipeline_drain",
            callable=lambda: self._meta_pipeline_drain_dispatch(),
            condition=lambda: self._meta_manager is not None,
            interval=5,
            critical=False,
        ))

    def register_module(
        self,
        name: str,
        callable: Callable[[], Any],
        condition: Callable[[], bool],
        interval: int = 1,
        critical: bool = False,
        result_key: Optional[str] = None,
    ) -> None:
        """Register a new module step in the dispatch table.

        Args:
            name: Unique name for the step (for logging)
            callable: No-arg function that executes the step (use lambda: ...)
            condition: No-arg function returning bool (when True, step runs)
            interval: Run every N cycles (1 = every cycle)
            critical: If True, step failure stops the cycle
            result_key: If set, stores return value in results dict under this key
        """
        self._dispatch_table.append(DispatchStep(
            name=name,
            callable=callable,
            condition=condition,
            interval=interval,
            critical=critical,
            result_key=result_key,
        ))
        logger.debug("Orchestrator: registered dispatch step '%s'", name)

    @weave_op
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

        # Store context for dispatch step helper methods
        self._current_profile = profile
        self._current_pairs = pairs
        self._current_granularity = granularity
        self._current_result = result
        self._dispatch_results: Dict[str, Any] = {}

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
                    logger.warning("Config tuner skipped: %s", e)

            scanner = Scanner(config)

            # Tier 3: Apply any LLM-proposed agent weight deltas (clamped + blended).
            # ClaudeReflectionHandler writes .claude/proposed_weights.json; we
            # validate + apply here so the next scan uses the updated weights.
            try:
                team = getattr(scanner, "agent_team", None) or getattr(scanner, "_agent_team", None)
                if team is not None and hasattr(team, "apply_proposed_weights"):
                    _pw_result = team.apply_proposed_weights()
                    if _pw_result.get("applied"):
                        logger.info(
                            "Proposed weights applied: %d changes in %s",
                            len(_pw_result.get("changes", {})),
                            _pw_result.get("changes"),
                        )
            except Exception as _pw_err:
                logger.warning("Proposed weights apply skipped: %s", _pw_err)
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

        # Store scan results in dispatch context
        self._current_scan_result = scan_result
        self._current_scanner = scanner

        # ── DISPATCH LOOP: Run all registered module steps ───────────────
        for step in self._dispatch_table:
            # Check interval: only run if cycle_count is divisible by interval
            if self._cycle_count % step.interval != 0:
                continue

            # Check condition: skip if condition returns False
            if not step.condition():
                continue

            # Execute the step with error isolation
            try:
                step_result = step.callable()
                if step.result_key:
                    self._dispatch_results[step.result_key] = step_result
            except Exception as e:
                logger.warning("Dispatch step '%s' failed: %s", step.name, e)
                if step.critical:
                    raise

        # ── OBSERVATION LOG (Old Step 2, kept inline for initial compatibility)
        if scan_result and self._observer:
            try:
                for analysis in scan_result.analyses:
                    result.observations_logged += self._observer.log_from_analysis(analysis)
            except Exception as e:
                result.errors.append(f"observation_log_failed: {e}")

        # ── TRADE EXECUTION (Step 3) ─────────────────────────────────────
        # 2026-04-20: Wire `auto_execute` flag — previously stored but never consumed.
        # Executes only when:
        #   • self._auto_execute was set True by the caller (e.g. main.py --execute)
        #   • Scanner config has enable_execution=True (double opt-in)
        #   • scan produced tradeable analyses
        # All defense-in-depth gates (agent_passed, circuit breakers, R:R ≥ 1.2,
        # disagreement/regime/confidence) are re-enforced inside Scanner.execute_trades →
        # ExecutionManager.execute_trade, so this line is a dispatch hook, not a bypass.
        if self._auto_execute and scan_result and scan_result.analyses:
            try:
                if getattr(self._current_scanner.config, "enable_execution", False):
                    exec_results = self._current_scanner.execute_trades(scan_result.analyses)
                    successful = [r for r in (exec_results or []) if getattr(r, "success", False)]
                    result.trades_executed = len(successful)
                    if exec_results:
                        logger.info(
                            "Auto-execute: attempted=%d, successful=%d",
                            len(exec_results), len(successful),
                        )
                else:
                    logger.info(
                        "Auto-execute requested but scanner.config.enable_execution=False — skipping"
                    )
            except Exception as e:
                error_class = type(e).__name__
                result.errors.append(f"auto_execute_failed ({error_class}): {e}")
                logger.error("Auto-execute failed [%s]: %s", error_class, e)

        return result

    # ── Dispatch Step Helper Methods ───────────────────────────────
    # These methods are called by the dispatch table and encapsulate each module's logic.

    def _log_observations_dispatch(self) -> None:
        """Log observations from scan analyses."""
        scan_result = self._current_scan_result
        result = self._current_result
        if scan_result and scan_result.analyses:
            try:
                for analysis in scan_result.analyses:
                    result.observations_logged += self._observer.log_from_analysis(analysis)
            except Exception as e:
                result.errors.append(f"observation_log_failed: {e}")

    def _macro_stress_update_dispatch(self) -> None:
        """Update macro stress detector."""
        scan_result = self._current_scan_result
        result = self._current_result
        if scan_result:
            try:
                from collections import Counter as _Counter
                regimes = [
                    str(getattr(a, "volatility_regime", "NORMAL") or "NORMAL").upper()
                    for a in (scan_result.analyses or [])
                ]
                dominant_regime = _Counter(regimes).most_common(1)[0][0] if regimes else "NORMAL"
                # Phase-1 Task-7: feed the regime probe so StagedDeployer's
                # promote-to-shadow capture has a real value to snapshot.
                # Falls through silently when the probe wasn't created (meta
                # disabled).
                _probe = getattr(self, "_regime_probe", None)
                if _probe is not None:
                    _probe.update(dominant_regime)
                spread_ratios = [
                    getattr(a, "spread_pips", 0) / 2.0
                    for a in (scan_result.analyses or [])
                    if getattr(a, "spread_pips", 0) > 0
                ]
                avg_spread_ratio = (sum(spread_ratios) / len(spread_ratios)) if spread_ratios else 1.0
                self._macro_stress.update(regime=dominant_regime, avg_spread_ratio=avg_spread_ratio)
                stress_mod = self._macro_stress.get_stress_modifier()
                result.stress_state = {
                    "stress_score": getattr(self._macro_stress, "_stress_score", 0),
                    "stress_modifier": stress_mod,
                    "regime": dominant_regime,
                }
                logger.debug("Macro stress updated: modifier=%.2f", stress_mod)
            except Exception as e:
                logger.debug("Macro stress update skipped: %s", e)

    def _dynamic_hedge_evaluation_dispatch(self) -> None:
        """Evaluate dynamic hedging."""
        result = self._current_result
        if result.stress_state:
            try:
                stress_mod = result.stress_state.get("stress_modifier", 1.0)
                stress_score = result.stress_state.get("stress_score", 0.0)
                _open_positions: list = []
                try:
                    if hasattr(self, "_scanner") and self._scanner:
                        _acct = self._scanner._executor.get_account_status()
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
                logger.debug("Dynamic hedge evaluation skipped: %s", e)

    def _online_rl_update_dispatch(self) -> None:
        """Run online RL micro-update."""
        result = self._current_result
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
            logger.warning("Online RL update skipped: %s", e)

    def _state_scan_record_dispatch(self) -> None:
        """Record scan cycle in state."""
        try:
            self._state.increment_scan_cycle()
        except Exception as e:
            logger.warning("State scan record skipped: %s", e)

    def _sync_closed_trades_rl_dispatch(self) -> list:
        """Sync closed trades and run RL."""
        result = self._current_result
        scanner = self._current_scanner
        closed_trades = []
        try:
            from src.scanner.execution import ExecutionManager
            em = ExecutionManager()
            sync_result = em.sync_closed_trades_rl(scanner=scanner) or {}
            if isinstance(sync_result, dict) and sync_result.get("trades_synced", 0) > 0:
                import json
                journal_path = self._root / "trained_data" / "trade_journal_rl.json"
                try:
                    journal = json.loads(journal_path.read_text())
                    closed_trades = [t for t in journal if isinstance(t, dict) and t.get("outcome")]
                except Exception:
                    pass
            logger.info("RL sync result: %s", sync_result)
        except Exception as e:
            logger.debug("RL sync skipped: %s", e)
        return closed_trades

    def _post_trade_diagnostics_dispatch(self) -> dict:
        """Run post-trade diagnostics, apply SelfHeal, and route to meta.

        Three layers run per cycle when meta is enabled:

          1. PostTradeDiagnostics.run() — emit recommended_actions + status.

          2. SelfHeal.apply(diag) — DIRECT actuation: the orchestrator's
             low-latency emergency response. The 04-28 trap-recovery
             commit (802c6ca) made this path actually reset gates instead
             of silent dead-writing.

          3. NEW (Mythos audit 2026-04-29): when enable_meta_manager=True
             AND a non-HEALTHY diagnostic surfaces, route the same
             incident through MetaManager.route_incident. The orchestrator
             already actuated via SelfHeal; the meta-pipeline produces an
             auditable ChangePackage with change_id + Constitution
             attestation + scorecard so the operator's F2 inbox can see
             every cybernetic event. Packages sit in AWAITING_APPROVAL
             (no auto-deploy from process()) — operator reviews via
             inbox; revert_by_id targets the change_id if needed.

             Dedup signature throttles routing: only call route_incident
             when (status, sorted recommended_actions) changed from the
             last cycle. Without this, every scan cycle would create a
             new package even when the diagnostic is reporting the same
             unresolved condition. The meta-pipeline's own G8 dedup
             provides a second safety net (in-flight package hash match
             aborts intake silently).
        """
        try:
            from src.scanner.feedback.diagnostics import PostTradeDiagnostics
            diag = PostTradeDiagnostics().run()
            heal_result: Optional[Dict[str, Any]] = None
            if isinstance(diag, dict) and diag.get("status") not in ("HEALTHY", None):
                try:
                    from src.scanner.feedback.self_heal import SelfHeal
                    heal_result = SelfHeal().apply(diag)
                    logger.info(
                        "Post-trade diagnostics: %s, heal: %s",
                        diag.get("status"), heal_result.get("status"),
                    )
                except Exception as heal_err:
                    logger.debug("Self-heal skipped: %s", heal_err)

                self._maybe_route_to_meta(diag, heal_result)
            return {"diagnostics": diag, "heal": heal_result}
        except Exception as e:
            logger.debug("Post-trade diagnostics skipped: %s", e)
            return {}

    def _maybe_route_to_meta(
        self,
        diag: Dict[str, Any],
        heal_result: Optional[Dict[str, Any]],
    ) -> None:
        """Route a non-HEALTHY diagnostic through the meta-pipeline once
        per (status, recommended_actions) signature change.

        Fail-tolerant — every error path returns silently after a debug
        log. Never aborts the parent dispatch, even when the meta
        module is missing entirely.
        """
        try:
            from src.scanner.automation.meta_manager import (
                is_enabled as _meta_enabled,
                route_incident,
            )
        except ImportError:
            return
        if not _meta_enabled():
            return

        actions = diag.get("recommended_actions") or []
        if not isinstance(actions, list) or not actions:
            return
        try:
            sig = (
                str(diag.get("status", "")),
                tuple(sorted(str(a) for a in actions if isinstance(a, str))),
            )
        except Exception:
            return

        last_sig = getattr(self, "_last_meta_route_sig", None)
        if sig == last_sig:
            return  # signature unchanged — meta-pipeline G8 would dedup anyway,
                    # but skipping the call entirely avoids changes_dir scan.
        self._last_meta_route_sig = sig

        try:
            routed = route_incident({
                "kind": "self_heal",
                "source": "orchestrator_post_trade_diagnostics",
                "diag": diag,
                "heal": heal_result,
            })
            logger.info(
                "orchestrator.meta_route status=%s actions=%d routed=%s",
                diag.get("status"), len(actions), routed,
            )
        except Exception as e:
            logger.debug("orchestrator.meta_route_failed err=%s", e)

    def _agent_health_attribution_dispatch(self) -> None:
        """Record outcomes for agent health tracking."""
        result = self._current_result
        closed_trades = self._dispatch_results.get("closed_trades", [])
        if closed_trades:
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
                    logger.warning("Agent health recording failed: %s", e)
            try:
                result.agent_health_report = self._agent_health.get_health_report()
            except Exception as e:
                logger.debug("Agent health report failed: %s", e)

    def _extract_learnings_dispatch(self) -> None:
        """Extract learnings from closed trades."""
        result = self._current_result
        closed_trades = self._dispatch_results.get("closed_trades", [])
        if closed_trades:
            for trade in closed_trades:
                try:
                    insights = self._learner.analyze_trade(trade)
                    if insights:
                        self._learner.append_to_learnings(insights)
                        result.learnings_extracted += len(insights)
                except Exception as e:
                    result.errors.append(f"learning_extraction_failed: {e}")

    def _override_pattern_extraction_dispatch(self) -> None:
        """Extract patterns from override events."""
        result = self._current_result
        try:
            overrides = self._aura_bridge.get_recent_overrides(limit=50)
            resolved_overrides = [o.to_dict() for o in overrides if o.outcome is not None]
            if resolved_overrides:
                override_entries = self._learner.analyze_overrides_batch(resolved_overrides)
                if override_entries:
                    self._learner.append_to_learnings(override_entries)
                    result.learnings_extracted += len(override_entries)
                    logger.info(
                        "Override patterns extracted: %d from %d events",
                        len(override_entries), len(resolved_overrides),
                    )
        except Exception as e:
            logger.debug("Override pattern extraction skipped: %s", e)

    def _bridge_recursive_learner_dispatch(self) -> None:
        """Feed overrides to bridge recursive learner."""
        try:
            overrides = self._aura_bridge.get_recent_overrides(limit=50)
            for override in overrides:
                if override.outcome is not None:
                    self._bridge_learner.observe(override.to_dict())
            promoted = self._bridge_learner.check_promotions()
            if promoted:
                logger.info("Bridge learner promoted %d override patterns to rules", len(promoted))
        except Exception as e:
            logger.debug("Bridge recursive learner skipped: %s", e)

    def _exit_reason_pattern_extraction_dispatch(self) -> None:
        """Extract exit reason patterns."""
        result = self._current_result
        try:
            exit_patterns = self._learner.extract_exit_reason_patterns(
                journal_path=self._root / "trained_data" / "trade_journal_rl.json",
            )
            if exit_patterns:
                self._learner.append_to_learnings(exit_patterns)
                result.learnings_extracted += len(exit_patterns)
                logger.info("Exit reason patterns extracted: %d", len(exit_patterns))
        except Exception as e:
            logger.debug("Exit reason pattern extraction skipped: %s", e)

    def _check_rule_promotions_dispatch(self) -> None:
        """Check for learnings that should become rules."""
        result = self._current_result
        try:
            promotions = self._learner.check_promotions()
            result.rules_promoted = len(promotions)
        except Exception as e:
            logger.warning("Promotion check skipped: %s", e)

    def _consolidate_learnings_dispatch(self) -> None:
        """Consolidate learnings if needed."""
        try:
            audit = self._learner.audit()
            if audit.get("total_learnings", 0) > 30:
                self._learner.consolidate()
        except Exception as e:
            logger.debug("Consolidation skipped: %s", e)

    def _update_state_dispatch(self) -> None:
        """Update session state."""
        result = self._current_result
        try:
            self._state.save_state(
                goal="autonomous_orchestration",
                status="running" if self._auto_execute else "scanning",
                done=[result.cycle_id],
                next_action="next_scan_cycle",
            )
        except Exception as e:
            logger.debug("State update skipped: %s", e)

    def _system_health_diagnostics_dispatch(self) -> None:
        """Run system health diagnostics."""
        result = self._current_result
        try:
            result.system_health_report = self._get_system_health_report()
            health_score = (result.system_health_report or {}).get("overall_score", 1.0)
            if health_score < 0.6:
                logger.warning("System health degraded: score=%.2f — check diagnostics", health_score)
        except Exception as e:
            logger.debug("System health report skipped: %s", e)

    def _apply_config_adjustments_dispatch(self) -> None:
        """Apply config adjustments."""
        result = self._current_result
        scanner = self._current_scanner
        if scanner is not None:
            try:
                applied = self._config_adjuster.apply_adjustments(scanner.config, self._cycle_count)
                if applied:
                    result.config_adjustments = (result.config_adjustments or 0) + len(applied)
                    logger.info(
                        "ConfigAdjuster applied %d adjustments: %s",
                        len(applied),
                        ", ".join(a["key"] for a in applied),
                    )
                self._config_adjuster.save_state()
            except Exception as e:
                logger.debug("ConfigAdjuster apply skipped: %s", e)

    def _observation_consumer_pattern_detection_dispatch(self) -> None:
        """Detect patterns from observations."""
        result = self._current_result
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
            logger.debug("ObservationConsumer skipped: %s", e)

    def _track_improvement_dispatch(self) -> None:
        """Track improvement metrics."""
        result = self._current_result
        closed_trades = self._dispatch_results.get("closed_trades", [])
        try:
            self._tracker.record_session(
                trades=closed_trades,
                learnings_added=result.learnings_extracted,
                rules_promoted=result.rules_promoted,
            )
        except Exception as e:
            logger.debug("Improvement tracking skipped: %s", e)

    def _aura_bridge_outcome_signal_dispatch(self) -> None:
        """Write outcome signal to aura bridge."""
        result = self._current_result
        closed_trades = self._dispatch_results.get("closed_trades", [])
        try:
            from src.aura.bridge.signals import OutcomeSignal
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
            outcome_signal = OutcomeSignal(
                pnl_today=sum(t.get("outcome", {}).get("realized_pl", 0) for t in closed_trades),
                win_rate_7d=win_rate_7d,
                override_events=[],
                regime=dominant_regime,
                streak=streak,
                trades_today=result.trades_executed,
                open_positions=0,
                max_drawdown_today=0.0,
            )
            self._aura_bridge.write_outcome(outcome_signal)
            logger.debug("Aura bridge: outcome signal written")
        except Exception as e:
            logger.debug("Aura bridge outcome write skipped: %s", e)


    def _bridge_rules_expiry_dispatch(self) -> None:
        """Expire stale bridge rules."""
        try:
            expired = self._bridge_rules.expire_stale_rules()
            if expired:
                logger.info("Bridge rules: %d rules expired", expired)
        except Exception as e:
            logger.debug("Bridge rule expiry skipped: %s", e)


    def _learnings_consolidation_dispatch(self) -> None:
        """Consolidate learnings and config adjustments (every 50 cycles)."""
        try:
            _learnings_path = self._root / ".claude" / "learnings.md"
            _config_adj_path = self._root / ".claude" / "config_adjustments.json"
            _archive_path = self._root / ".claude" / "learnings_archive.md"
            if _learnings_path.exists():
                _lines = _learnings_path.read_text(encoding="utf-8").strip().split("\n")
                _entries = [l for l in _lines if l.strip().startswith("- ") or l.strip().startswith("## 20")]
                if len(_entries) > 30:
                    from datetime import datetime as _dt154, timedelta as _td154
                    _cutoff = (_dt154.now() - _td154(days=30)).strftime("%Y-%m-%d")
                    _keep = []
                    _archive = []
                    for l in _lines:
                        if any(d in l for d in [f"20{y}" for y in range(24, 27)] if d < _cutoff):
                            _archive.append(l)
                        else:
                            _keep.append(l)
                    if _archive:
                        with open(_archive_path, "a", encoding="utf-8") as _af:
                            _af.write(f"\n# Archived {_dt154.now().strftime('%Y-%m-%d')}\n")
                            _af.write("\n".join(_archive) + "\n")
                        _learnings_path.write_text("\n".join(_keep) + "\n", encoding="utf-8")
                        logger.info("US-154: Archived %d old learnings (kept %d)", len(_archive), len(_keep))
            if _config_adj_path.exists():
                import json as _j154
                try:
                    _adj_data = _j154.loads(_config_adj_path.read_text(encoding="utf-8"))
                    if isinstance(_adj_data, list) and len(_adj_data) > 100:
                        from datetime import datetime as _dt154b, timedelta as _td154b
                        _cutoff_ts = (_dt154b.now() - _td154b(days=30)).isoformat()
                        _pruned = [a for a in _adj_data if a.get("timestamp", "") > _cutoff_ts]
                        _removed = len(_adj_data) - len(_pruned)
                        if _removed > 0:
                            _config_adj_path.write_text(_j154.dumps(_pruned, indent=2), encoding="utf-8")
                            logger.info("US-154: Pruned %d old config adjustments (kept %d)", _removed, len(_pruned))
                except Exception:
                    pass
        except Exception as _consol_err:
            logger.debug("US-154: Consolidation skipped: %s", _consol_err)

    def _drift_auto_remediation_dispatch(self) -> None:
        """Check and remediate drift in model performance (interval-gated)."""
        try:
            _cfg = getattr(self, "_config", None)
            _check_interval = getattr(_cfg, "drift_remediation_check_interval", 20) if _cfg else 20
            _remediation_enabled = getattr(_cfg, "enable_drift_remediator", True) if _cfg else True
            self._remediation_cycle_counter += 1
            if _remediation_enabled and (self._remediation_cycle_counter % _check_interval == 0):
                rem_result = self._drift_remediator.check_and_remediate()
                logger.info(
                    f"DriftRemediator: {rem_result.action_taken} — {rem_result.reason} "
                    f"(duration={rem_result.duration_seconds:.1f}s)"
                )
            else:
                logger.debug(
                    f"DriftRemediator: skipping this cycle "
                    f"(counter={self._remediation_cycle_counter}, interval={_check_interval})"
                )
        except Exception as e:
            logger.debug("Drift remediator check failed: %s", e)

    def _performance_prd_generation_dispatch(self) -> None:
        """Auto-generate PRD stories based on performance gaps."""
        try:
            _cfg = getattr(self, "_config", None)
            _prd_enabled = getattr(_cfg, "enable_auto_ralph", True) if _cfg else True
            if _prd_enabled:
                prd_result = self._perf_prd_gen.run()
                logger.info(
                    f"PerformancePRDGenerator: triggered={prd_result.run_triggered}, "
                    f"gaps={prd_result.gaps_found}, stories={prd_result.stories_generated}, "
                    f"ralph_spawned={prd_result.ralph_spawned} — {prd_result.reason}"
                )
        except Exception as e:
            logger.debug("Performance PRD generation check failed: %s", e)

    def _policy_engine_audit_dispatch(self) -> None:
        """Tier 7: Audit policy rules and log environment snapshot."""
        try:
            env = self._policy_engine.get_environment()
            issues = self._policy_engine.lint_rules()
            if issues:
                logger.warning("Tier 7: Policy lint issues: %s", issues)
            else:
                logger.debug("Tier 7: Policy audit clean (env=%s)", env.get("account_mode", "unknown"))
        except Exception as e:
            logger.debug("Tier 7: Policy audit failed: %s", e)

    def _meta_pipeline_drain_dispatch(self) -> None:
        """Walk every in-flight ChangePackage one stage forward.

        Reads the approval queue, advances approved packages to their target
        deploy stage, and routes already-deployed packages through the
        post-deploy critic for promotion or rollback. Pure delegation —
        all decision logic lives in MetaManager.drain.
        """
        try:
            counters = self._meta_manager.drain(current_cycle=self._cycle_count)
            if any(v for v in counters.values()):
                logger.info("meta_pipeline.drain %s", counters)
        except Exception as e:
            logger.warning("meta_pipeline.drain_failed err=%s", e)

    def _model_freshness_check_dispatch(self) -> None:
        """Check model freshness and warn/trigger retrain on staleness.

        Runs every 10 cycles. On CRITICAL or STALE status, logs a warning.
        On CRITICAL, attempts to trigger the drift remediator if available,
        since stale models are functionally equivalent to drifted models.
        """
        try:
            from src.scanner.automation.model_freshness import get_model_freshness

            freshness = get_model_freshness()
            status = freshness.get("status", "UNKNOWN")
            oldest = freshness.get("oldest_age_days")
            stale_list = freshness.get("stale_models", [])

            if status == "FRESH":
                logger.debug(
                    "Model freshness: FRESH (oldest=%.1fd)", oldest or 0
                )
                return

            if status == "AGING":
                logger.info(
                    "Model freshness: AGING (oldest=%.1fd) — approaching staleness",
                    oldest or 0,
                )
                return

            # STALE or CRITICAL — warn
            logger.warning(
                "Model freshness: %s (oldest=%.1fd) — stale models: %s",
                status,
                oldest or 0,
                ", ".join(stale_list) if stale_list else "none",
            )

            # On CRITICAL, attempt to trigger retraining via drift remediator
            if status == "CRITICAL" and self._drift_remediator is not None:
                try:
                    logger.info(
                        "Model freshness CRITICAL — triggering drift remediator "
                        "as staleness proxy (oldest=%.1fd)",
                        oldest or 0,
                    )
                    rem_result = self._drift_remediator.check_and_remediate()
                    logger.info(
                        "Freshness-triggered remediation: %s — %s (%.1fs)",
                        rem_result.action_taken,
                        rem_result.reason,
                        rem_result.duration_seconds,
                    )
                except Exception as retrain_err:
                    logger.warning(
                        "Freshness-triggered retrain failed: %s", retrain_err
                    )
        except Exception as e:
            logger.debug("Model freshness check failed: %s", e)

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

    def close(self) -> None:
        # Phase 100: Stop PRDAgentChain daemon thread first
        if getattr(self, "_prd_chain", None) is not None:
            try:
                self._prd_chain.stop()
                logger.info("Shutdown: PRDAgentChain stopped")
            except Exception as _e:
                logger.debug("Shutdown: PRDAgentChain stop failed: %s", _e)

        # US-006: Graceful shutdown. Orchestrator had no shutdown hook before this fix.
        _modules_to_save = [
            ("_state", "state_engine"),
            ("_config_adjuster", "config_adjuster"),
            ("_observation_consumer", "observation_consumer"),
            ("_drift_remediator", "drift_remediator"),
            ("_online_rl", "online_rl"),
            ("_gate_health", "gate_health"),
            ("_agent_health", "agent_health"),
            ("_module_dispatcher", "module_dispatcher"),
        ]
        for attr, label in _modules_to_save:
            mod = getattr(self, attr, None)
            if mod is None:
                continue
            for method in ("save_state", "save_log", "save"):
                if hasattr(mod, method):
                    try:
                        getattr(mod, method)()
                        logger.info("Shutdown: %s via %s()", label, method)
                    except Exception as _e:
                        logger.debug("Shutdown: %s failed: %s", label, _e)
                    break

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
                "bridge_rules_engine": self._bridge_rules is not None,
                "prd_agent_chain": self._prd_chain is not None,
                "observation_consumer": self._observation_consumer is not None,  # Phase 29 (US-177)
                "policy_engine": self._policy_engine is not None,  # Tier 7
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

        # Add bridge rules status
        if self._bridge_rules:
            try:
                status["bridge_rules"] = self._bridge_rules.get_rules_summary()
            except Exception:
                status["bridge_rules"] = {}

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
                    logger.debug("US-164: Health registry status error: %s", _hr_err)

        # Phase 28 (US-169): PRD agent chain status
        if self._prd_chain is not None:
            try:
                status["prd_agent_chain"] = self._prd_chain.get_status()
            except Exception as e:
                logger.debug("PRDAgentChain status error: %s", e)

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
