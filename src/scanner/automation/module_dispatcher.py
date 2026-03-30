"""Module activation dispatcher with cycle-based scheduling.

US-124: Central dispatcher that controls which modules run each cycle
based on their registered frequency via ModuleActivationMap. Expensive
modules (replay_validator, model_calibration) run less frequently;
lightweight modules (memory_manager, health_registry) run every cycle.

Approach:
1. Register all enabled modules with appropriate run frequencies
2. Each cycle, ask ModuleActivationMap which modules should run
3. Execute runnable modules in dependency order
4. Track orphaned modules and dispatch metrics
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "module_dispatcher_log.json"

# Module run frequencies (cycles between runs)
# Lower number = runs more often. 1 = every cycle.
DEFAULT_FREQUENCIES: Dict[str, int] = {
    # Phase 19 — production hardening (lightweight → every cycle)
    "memory_manager": 1,
    "health_registry": 1,
    "module_activation": 5,
    "agent_lifecycle": 1,
    "observation_consumer": 10,
    "replay_validator": 20,
    # Phase 18 — adaptive quality (medium frequency)
    "execution_quality_optimizer": 5,
    "threshold_optimizer": 5,
    "dynamic_risk_allocator": 3,
    "model_calibration": 10,
    "pair_regime_agent_matrix": 5,
    "signal_timing": 10,
    # Phase 17 — signal consumption (medium)
    "position_manager": 1,
    "affinity_portfolio": 10,
    "execution_router": 1,
    "model_router": 10,
    "training_augmenter": 20,
    "attention_feedback": 5,
    # Phase 16 — execution metalearning (medium)
    "entropy_sizer": 1,
    "exec_quality_tracker": 1,
    "regime_broadcaster": 1,
    "agent_accuracy_matrix": 5,
    "temporal_attention": 5,
    "pair_transfer": 10,
    # Phase 29 (US-176): Additional module frequencies
    "drift_monitor": 20,
    "config_adjuster": 10,
    "dynamic_sl_tp": 3,
    "trade_explainer": 10,
    "multi_horizon_fusion": 5,
    "kelly_portfolio": 10,
    "ensemble_disagreement": 3,
    "feature_attention": 10,
    "adversarial_trainer": 20,
    "regime_reward": 5,
    "adaptive_lr": 10,
    "market_impact": 5,
    "model_bandit": 10,
    "causal_filter": 10,
    "confidence_calibrator": 10,
    "concept_drift": 15,
    "lead_lag_detector": 10,
    "position_timeout": 3,
    "dynamic_hedging": 5,
    "trade_outcome_predictor": 10,
    # Phase 33 (US-215): Missing dispatchable modules
    "macro_stress": 20,             # Runs infrequently — stress testing
    "microstructure_regime": 5,     # Medium frequency — regime detection
    "regime_tracker": 3,            # Frequent — tracks regime transitions
    # Phase 50: Previously orphaned modules
    "retrain_trigger": 30,          # Training — check drift infrequently
    "pair_model_selector": 10,      # Optimization — model selection per pair
    "portfolio_optimizer": 10,      # Optimization — pair ranking by Sharpe
    "session_snapshot": 5,          # Monitoring — session state capture
    "alert_manager": 2,             # Monitoring — check alerts frequently
    "dynamic_drawdown": 2,          # Monitoring — drawdown threshold adaptation
    "smart_execution": 1,           # Execution — order slicing every cycle
    "accuracy_gate": 5,             # Monitoring — per-pair accuracy gating
    "pair_affinity": 10,            # Optimization — co-occurrence tracking
    "hierarchical_risk_parity": 10, # Optimization — HRP weight allocation
    "observational_learning": 20,   # Training — synthetic pattern extraction
    "model_manager": 20,            # Training — version management and promotion
    "crisis_generator": 50,         # Crisis/stress — synthetic scenario generation
    "api_retry": 1,                 # Monitoring — circuit breaker health every cycle
    # US-006–US-019: Previously unregistered modules
    "group_momentum": 3,            # Aggregation — group momentum every few cycles
    "seasonality": 10,              # Analytics — hour-of-day modifiers
    "counterfactual_learner": 20,   # Learning — counterfactual trade analysis
    "chain_memory": 10,             # Memory — PRD agent chain context
    "drift_projector": 20,          # Analytics — drift projection caching
    "episodic_memory": 10,          # Learning — episodic trade pattern recall
    "self_model_state": 10,         # Analytics — self-model projection state
    "causal_counterfactual": 20,    # Analytics — causal counterfactual engine
}


class ModuleDispatcher:
    """Central dispatcher that schedules module execution by cycle frequency.

    Bridges ModuleActivationMap (which tracks state) with actual module
    execution. Registers all enabled modules from Scanner, then each cycle
    determines which subset should run.

    Usage:
        dispatcher = ModuleDispatcher(scanner)
        dispatcher.register_all_modules()

        # In scan loop:
        runnable = dispatcher.get_runnable_modules(cycle_number)
        dispatcher.execute_cycle(cycle_number)
    """

    def __init__(
        self,
        scanner: Any = None,
        persistence_path: Optional[Path] = None,
    ):
        self._scanner = scanner
        self.persistence_path = persistence_path or _LOG_PATH

        # Module registry: name → (callable, frequency)
        self._modules: Dict[str, Tuple[Callable, int]] = {}
        self._cycle_count: int = 0
        self._total_dispatches: int = 0
        self._dispatch_errors: int = 0
        self._dispatch_log: List[Dict[str, Any]] = []

    def register_module(
        self,
        name: str,
        run_fn: Callable[[], None],
        run_every_n: int = 1,
    ) -> None:
        """Register a module with its run function and cycle frequency.

        Args:
            name: Module identifier.
            run_fn: Callable to execute the module's per-cycle work.
            run_every_n: Run every N cycles (1 = every cycle).
        """
        self._modules[name] = (run_fn, max(1, run_every_n))

    def register_all_modules(self) -> int:
        """Auto-register all enabled modules from the Scanner instance.

        Returns:
            Number of modules registered.
        """
        if self._scanner is None:
            return 0

        registered = 0

        # Map attribute names → module names and their save_state/run methods
        module_map: Dict[str, str] = {
            "_memory_manager": "memory_manager",
            "_health_registry": "health_registry",
            "_module_activation": "module_activation",
            "_agent_lifecycle": "agent_lifecycle",
            "_observation_consumer": "observation_consumer",
            "_replay_validator": "replay_validator",
            "_execution_quality_optimizer": "execution_quality_optimizer",
            "_threshold_optimizer": "threshold_optimizer",
            "_dynamic_risk_allocator": "dynamic_risk_allocator",
            "_model_calibration": "model_calibration",
            "_pair_regime_agent_matrix": "pair_regime_agent_matrix",
            "_signal_timing": "signal_timing",
            "_position_manager": "position_manager",
            "_affinity_portfolio": "affinity_portfolio",
            "_execution_router": "execution_router",
            "_model_router": "model_router",
            "_training_augmenter": "training_augmenter",
            "_attention_feedback": "attention_feedback",
            "_entropy_sizer": "entropy_sizer",
            "_exec_quality_tracker": "exec_quality_tracker",
            "_regime_broadcaster": "regime_broadcaster",
            "_agent_accuracy_matrix": "agent_accuracy_matrix",
            "_temporal_attention": "temporal_attention",
            "_pair_transfer": "pair_transfer",
            # Phase 29 (US-176): Additional modules
            "_drift_monitor": "drift_monitor",
            "_config_adjuster": "config_adjuster",
            "_dynamic_sl_tp": "dynamic_sl_tp",
            "_trade_explainer": "trade_explainer",
            "_multi_horizon_fusion": "multi_horizon_fusion",
            "_kelly_optimizer": "kelly_portfolio",
            "_ensemble_disagreement": "ensemble_disagreement",
            "_feature_attention": "feature_attention",
            "_adversarial_trainer": "adversarial_trainer",
            "_regime_reward": "regime_reward",
            "_adaptive_lr": "adaptive_lr",
            "_market_impact": "market_impact",
            "_model_bandit": "model_bandit",
            "_causal_filter": "causal_filter",
            "_confidence_calibrator": "confidence_calibrator",
            "_concept_drift": "concept_drift",
            "_lead_lag_detector": "lead_lag_detector",
            "_position_timeout": "position_timeout",
            "_dynamic_hedging": "dynamic_hedging",
            "_trade_outcome_predictor": "trade_outcome_predictor",
            # Phase 50: Previously orphaned modules
            "_retrain_trigger": "retrain_trigger",
            "_pair_model_selector": "pair_model_selector",
            "_portfolio_optimizer": "portfolio_optimizer",
            "_session_snapshot": "session_snapshot",
            "_alert_manager": "alert_manager",
            "_dynamic_drawdown": "dynamic_drawdown",
            "_smart_execution": "smart_execution",
            "_accuracy_gate": "accuracy_gate",
            "_pair_affinity": "pair_affinity",
            "_hrp_optimizer": "hierarchical_risk_parity",
            "_observational_learner": "observational_learning",
            "_model_manager": "model_manager",
            "_crisis_generator": "crisis_generator",
            "_api_retry": "api_retry",
            # US-006–US-019: Previously unregistered modules
            "_group_momentum": "group_momentum",
            "_seasonality": "seasonality",
            "_counterfactual_learner": "counterfactual_learner",
            "_chain_memory": "chain_memory",
            "_drift_projector": "drift_projector",
            "_episodic_memory": "episodic_memory",
            "_self_model_state": "self_model_state",
            "_causal_counterfactual": "causal_counterfactual",
            # Gap resolution: modules in DEFAULT_FREQUENCIES but missing from module_map
            "_microstructure_detector": "microstructure_regime",
            "_regime_tracker": "regime_tracker",
            "_macro_stress": "macro_stress",
        }

        for attr, name in module_map.items():
            module = getattr(self._scanner, attr, None)
            if module is not None:
                freq = DEFAULT_FREQUENCIES.get(name, 5)
                # Create a save_state wrapper as the default dispatch action
                # Fallback to save_log for modules that persist via save_log()
                # Find best dispatch method: save_state > save_log > save > save_projections > save_analysis > noop
                _dispatch_fn = None
                for _method_name in ("save_state", "save_log", "save", "save_projections", "save_analysis"):
                    if hasattr(module, _method_name):
                        _dispatch_fn = getattr(module, _method_name)
                        break
                if _dispatch_fn is None:
                    # Read-only module — register with noop so it appears in dispatch tracking
                    _dispatch_fn = lambda: None  # noqa: E731
                self.register_module(
                    name,
                    run_fn=_dispatch_fn,
                    run_every_n=freq,
                )
                registered += 1

        # Also register with ModuleActivationMap if available
        activation_map = getattr(self._scanner, "_module_activation", None)
        if activation_map is not None:
            for name, (_, freq) in self._modules.items():
                try:
                    mod_instance = None
                    for attr, mod_name in module_map.items():
                        if mod_name == name:
                            mod_instance = getattr(self._scanner, attr, None)
                            break
                    activation_map.register_module(
                        name=name,
                        instance=mod_instance,
                        run_every_n_cycles=freq,
                    )
                except Exception:
                    pass

        logger.info("ModuleDispatcher: %d modules registered", registered)
        return registered

    def get_runnable_modules(self, cycle: int) -> List[str]:
        """Get list of modules that should run this cycle.

        Args:
            cycle: Current cycle number.

        Returns:
            List of module names scheduled for this cycle.
        """
        runnable = []
        for name, (_, freq) in self._modules.items():
            if cycle % freq == 0:
                runnable.append(name)
        return runnable

    def execute_cycle(self, cycle: int) -> Dict[str, bool]:
        """Execute all modules scheduled for this cycle.

        Args:
            cycle: Current cycle number.

        Returns:
            Dict mapping module name → success/failure.
        """
        self._cycle_count = cycle
        runnable = self.get_runnable_modules(cycle)
        results: Dict[str, bool] = {}

        for name in runnable:
            run_fn, _ = self._modules[name]
            try:
                run_fn()
                results[name] = True
                self._total_dispatches += 1
            except Exception as e:
                results[name] = False
                self._dispatch_errors += 1
                logger.debug(f"ModuleDispatcher: {name} failed: {e}")

        # Update ModuleActivationMap if available
        activation_map = getattr(self._scanner, "_module_activation", None)
        if activation_map is not None:
            try:
                activation_map.complete_cycle(cycle)
            except Exception:
                pass

        # Log dispatch summary periodically
        if cycle % 10 == 0:
            self._dispatch_log.append({
                "cycle": cycle,
                "dispatched": len(runnable),
                "succeeded": sum(1 for v in results.values() if v),
                "failed": sum(1 for v in results.values() if not v),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self._dispatch_log) > 100:
                self._dispatch_log = self._dispatch_log[-50:]

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get dispatcher status for observability."""
        return {
            "registered_modules": len(self._modules),
            "total_dispatches": self._total_dispatches,
            "dispatch_errors": self._dispatch_errors,
            "cycle_count": self._cycle_count,
            "module_frequencies": {
                name: freq for name, (_, freq) in self._modules.items()
            },
            "recent_log": self._dispatch_log[-5:],
        }

    def save_state(self) -> None:
        """Persist dispatcher state."""
        try:
            data = {
                "version": 1,
                "total_dispatches": self._total_dispatches,
                "dispatch_errors": self._dispatch_errors,
                "cycle_count": self._cycle_count,
                "dispatch_log": self._dispatch_log[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ModuleDispatcher: Failed to save: {e}")
