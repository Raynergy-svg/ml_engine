"""Post-trade event handlers extracted from ExecutionManager.sync_closed_trades_rl().

Each handler wraps a single module callback that previously ran inline after
trade-close journal write. Handlers receive a TradingEvent + SyncContext and
execute independently via the EventQueue.

Design rules:
  - Handlers may READ the canonical journal but NEVER write to it.
  - Handlers are idempotent — re-processing the same event is safe.
  - Handler failures are isolated — one crash does not block others.
  - Two composite handlers preserve internal dependency chains:
    * AttributionPipelineHandler (attribution → gate threshold)
    * WalkForwardPipelineHandler (retrainer ↔ WFE monitor)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import structlog

logger = structlog.get_logger(__name__)


# ── Context ────────────────────────────────────────────────────────────

@dataclass
class SyncContext:
    """Runtime context passed to handlers alongside the TradingEvent.

    Contains mutable state and references that handlers need but which
    should NOT live in the serializable event payload.
    """
    entry: Dict[str, Any]              # The journal entry being processed
    outcome_data: Dict[str, Any]       # Computed outcome dict
    closed_trades: Dict[str, Any]      # OANDA closed trade snapshots {tid: trade}
    entries: List[Dict[str, Any]]      # Full journal (read-only for handlers)
    synced_entries: List[Dict[str, Any]]  # Entries synced this cycle
    rl_updates: List[Dict[str, Any]]   # RL update dicts for weight feedback
    scanner: Any = None                # Scanner instance (for getattr module access)
    execution_manager: Any = None      # ExecutionManager instance
    synced_count: int = 0              # Number of trades synced this cycle
    trade_id: str = ""                 # Current trade ID
    pair: str = ""                     # Current pair
    trade_won: bool = False
    realized_pl: float = 0.0
    pnl_pips: float = 0.0


# ── Handler Protocol ───────────────────────────────────────────────────

class EventHandler(Protocol):
    """Protocol for post-trade event handlers."""
    name: str
    priority: int

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        """Process the event. Returns True on success."""
        ...


# ── Individual Handlers ────────────────────────────────────────────────

class RLReplayHandler:
    """Feed trade outcome to RL replay buffer."""
    name = "rl_replay"
    priority = 10

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None:
            return False
        em._sync_trade_outcome_to_rl(context.entry, context.outcome_data)
        return True


class EpisodicMemoryHandler:
    """Record outcome to episodic memory for pattern suppression."""
    name = "episodic_memory"
    priority = 15

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._episodic_memory is None:
            return False
        episode_id = context.entry.get("episode_id", "")
        if not episode_id:
            return True  # No episode to record — not an error
        outcome_str = "WIN" if context.trade_won else "LOSS"
        em._episodic_memory.record_outcome(
            episode_id=episode_id,
            outcome=outcome_str,
            pnl_pips=float(context.pnl_pips),
        )
        return True


class PositionTimeoutHandler:
    """Deregister closed trade from PositionTimeoutManager."""
    name = "position_timeout"
    priority = 15

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._position_timeout_mgr is None:
            return False
        em._position_timeout_mgr.close_trade(context.trade_id)
        return True


class AccuracyGateHandler:
    """Record outcome to per-pair rolling accuracy gate."""
    name = "accuracy_gate"
    priority = 20

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        ag = getattr(context.scanner, "_accuracy_gate", None)
        if ag is None:
            return False
        direction = context.entry.get("direction", "LONG")
        conf = float(context.entry.get("confidence", 0.5))
        ag.record_outcome(
            pair=context.pair,
            predicted_direction=direction,
            actual_outcome=context.trade_won,
            confidence=conf,
            trade_id=context.trade_id,
        )
        return True


class RetrainTriggerHandler:
    """Record prediction to retrain trigger for drift detection."""
    name = "retrain_trigger"
    priority = 20

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        rt = getattr(context.scanner, "_retrain_trigger", None)
        if rt is None:
            return False
        rt.record_prediction(pair=context.pair, correct=context.trade_won)
        return True


class MetaLearnerHandler:
    """Update meta-learner hyperparameters on trade close."""
    name = "meta_learner"
    priority = 30

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        ml = getattr(context.scanner, "_meta_learner", None)
        if ml is None:
            return False

        entry = context.entry
        agents_info = entry.get("agents", {})
        agent_reasons = agents_info.get("agent_reasons", [])
        agent_scores = {}
        agent_weights = {}
        for ar in agent_reasons:
            name = ar.get("name", "")
            if name:
                agent_scores[name] = float(ar.get("score", 0.5))
                agent_weights[name] = float(ar.get("weight", 1.0))

        if not agent_scores:
            return True

        regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
        entry_price = float(entry.get("entry_price", 0))
        ct = context.closed_trades.get(context.trade_id, {})
        notional = abs(entry_price * float(ct.get("initialUnits", 1)))
        pnl_pct = context.realized_pl / max(notional, 1.0)

        ml.on_trade_close(
            pair=entry.get("pair", ""),
            regime=regime,
            trade_result={"pnl_pct": pnl_pct, "outcome": "win" if context.trade_won else "loss"},
            agent_scores=agent_scores,
            agent_weights=agent_weights,
        )
        return True


class EnsembleWeighterHandler:
    """Record outcome to ensemble weighter for learned model weights."""
    name = "ensemble_weighter"
    priority = 30

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        ew = getattr(context.scanner, "_ensemble_weighter", None)
        if ew is None:
            return False

        entry = context.entry
        gate = entry.get("gate_details", {})
        scores = gate.get("scores", {})
        dir_score = float(scores.get("direction_score", 0.5))
        conf_score = float(scores.get("confidence_score", 0.5))
        mom_score = float(scores.get("momentum_score", 0.5))
        risk_score = float(scores.get("risk_score", 0.5))
        regime = entry.get("regime", {})
        vol_id = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}.get(
            str(regime.get("volatility_regime", "NORMAL")), 1
        )
        vol_conf = float(regime.get("volatility_regime_confidence", 0.5))
        outcome = context.pnl_pips / 100.0

        ew.record(
            context_features=[vol_id / 3.0, vol_conf],
            model_predictions=[dir_score, conf_score, mom_score, risk_score],
            realized_outcome=outcome,
        )
        return True


class MAMLHandler:
    """Record outcome to MAML Ridge + benchmark."""
    name = "maml_ridge"
    priority = 30

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        maml = getattr(context.scanner, "_maml_ridge", None)
        if maml is None:
            return False

        import numpy as np
        entry = context.entry
        regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
        if isinstance(regime, int):
            regime = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}.get(regime, "NORMAL")

        ridge_conf = float(entry.get("regime", {}).get("ridge_confidence",
                           entry.get("agents", {}).get("ridge_confidence", 50)))
        if context.trade_won:
            target = min(100.0, ridge_conf + abs(context.pnl_pips) * 0.5)
        else:
            target = max(0.0, ridge_conf - abs(context.pnl_pips) * 0.5)
        target = max(0.0, min(100.0, target))

        stored_features = entry.get("ridge_features")
        if stored_features is not None and len(stored_features) >= 14:
            features = np.array(stored_features[:14], dtype=np.float32)
        else:
            gate = entry.get("gate_details", {})
            scores = gate.get("scores", {})
            features = np.array([
                float(scores.get("direction_score", 0.5)),
                float(scores.get("confidence_score", 0.5)),
                float(scores.get("momentum_score", 0.5)),
                float(scores.get("risk_score", 0.5)),
            ] + [0.0] * 10, dtype=np.float32) if scores else None

        if features is None:
            return True

        maml.record_outcome(features=features, regime=str(regime), target_confidence=target)

        # Benchmark recording
        bench = getattr(context.scanner, "_maml_benchmark", None)
        if bench is not None:
            shadow = entry.get("maml_shadow", {})
            bench_conf = float(shadow.get("maml_confidence", target))
            bench.record_outcome(
                trade_id=entry.get("oanda_trade_id", entry.get("pair", "unknown")),
                regime=str(regime),
                maml_confidence=bench_conf,
                ridge_confidence=ridge_conf,
                realized_confidence=target,
                pnl_pips=float(context.pnl_pips),
                won=bool(context.trade_won),
                exit_reason=str(context.outcome_data.get("exit_reason", "")),
                duration_minutes=context.outcome_data.get("duration_minutes"),
                regime_at_exit=str(context.outcome_data.get("regime_at_exit", "")),
            )
        return True


class ExpectancyHandler:
    """Record trade outcomes to expectancy tracker."""
    name = "expectancy_tracker"
    priority = 40

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None:
            return False
        # Use synced_entries (post-sync handler gets the full list)
        pending = [e for e in context.synced_entries if e.get("trade_id") in context.closed_trades]
        em._record_expectancy_from_trades(pending, context.closed_trades)
        return True


class PairPerformanceHandler:
    """Update pair tracker for auto-blacklisting."""
    name = "pair_performance"
    priority = 40

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._pair_tracker is None:
            return False
        ct = context.closed_trades.get(context.trade_id)
        if ct is None:
            return True
        pl = float(ct.get("realizedPL", 0))
        em._pair_tracker.update_from_trade(pair=context.pair, pnl=pl, won=pl > 0)
        em._pair_tracker.save_state()
        return True


class AttributionPipelineHandler:
    """Composite: Attribution → Gate Threshold feedback → Gate Threshold optimize.

    These three steps have a strict dependency chain and must run in order.
    """
    name = "attribution_pipeline"
    priority = 50

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._attribution_engine is None:
            return False
        if context.trade_id not in context.closed_trades:
            return True

        entry = context.entry
        ct = context.closed_trades[context.trade_id]
        pair = context.pair
        direction = entry.get("direction", "BUY").upper()
        realized_pl = float(ct.get("realizedPL", 0))
        won = realized_pl > 0
        outcome_str = "WIN" if won else "LOSS"

        agent_reasons = entry.get("agents", {}).get("agent_reasons", [])
        verdicts = {}
        for ar in agent_reasons:
            name = ar.get("name", "")
            if name:
                verdict = "LONG" if ar.get("passed", False) else "SHORT"
                weight = float(ar.get("weight", 1.0))
                conf = float(ar.get("confidence", 0.5))
                verdicts[name] = (verdict, conf, weight)

        if not verdicts:
            return True

        # Step 1: Attribute trade
        attribution = em._attribution_engine.attribute_trade(
            trade_id=context.trade_id,
            pair=pair,
            direction=direction,
            outcome=outcome_str,
            pnl=realized_pl,
            agent_verdicts=verdicts,
        )
        if entry.get("outcome") is None:
            entry["outcome"] = {}
        entry["outcome"]["attribution"] = {
            "alignment_ratio": round(attribution.alignment_ratio, 3),
            "aligned": [c.agent_name for c in attribution.aligned_agents],
            "misaligned": [c.agent_name for c in attribution.misaligned_agents],
        }
        em._attribution_engine.save_state()

        # Step 2: Pipe weight recommendations to gate threshold optimizer
        weight_recs = em._attribution_engine.get_weight_recommendations()
        if em._gate_threshold_optimizer is not None and weight_recs:
            try:
                for agent_name, weight_delta in weight_recs.items():
                    em._gate_threshold_optimizer.record_agent_feedback(
                        agent_name=agent_name,
                        weight_delta=weight_delta,
                        source="trade_attribution",
                    )
            except AttributeError:
                pass  # method doesn't exist yet

        # Step 3: Compute and apply gate threshold adjustments
        if em._gate_threshold_optimizer is not None:
            report = em._attribution_engine.generate_report()
            if report:
                recs = em._gate_threshold_optimizer.compute_recommendations(report)
                if recs:
                    applied = em._gate_threshold_optimizer.apply_recommendations(recs)
                    if applied:
                        em._gate_threshold_optimizer.save_state()
                        logger.info(
                            "attribution_pipeline.gate_adjusted",
                            count=len(applied),
                        )

        return True


class DriftDetectorHandler:
    """Record outcomes and check for feature drift."""
    name = "drift_detector"
    priority = 50

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._feature_drift_detector is None:
            return False
        if context.trade_id not in context.closed_trades:
            return True

        won = float(context.closed_trades[context.trade_id].get("realizedPL", 0)) > 0
        em._feature_drift_detector.record_trade(won=won)

        if em._feature_drift_detector.should_check():
            all_with_outcomes = [e for e in context.entries if e.get("outcome") is not None]
            cfg = em._feature_drift_detector.config
            recent_n = min(cfg.recent_window, len(all_with_outcomes))
            baseline_n = min(cfg.baseline_window, len(all_with_outcomes) - recent_n)
            if recent_n >= 50 and baseline_n >= 50:
                recent = all_with_outcomes[-recent_n:]
                baseline = all_with_outcomes[-(recent_n + baseline_n):-recent_n]
                result = em._feature_drift_detector.check_drift(recent, baseline)
                em._feature_drift_detector.save_state()
                if result.drift_detected:
                    logger.warning("drift_detector.drift_found", reason=result.reason)
        return True


class FeatureHealthHandler:
    """Monitor per-feature PSI every 50 trades."""
    name = "feature_health"
    priority = 50

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._feature_health_monitor is None:
            return False

        em._feature_health_monitor.record_trade()

        if em._feature_health_monitor.should_check():
            all_with_outcomes = [e for e in context.entries if e.get("outcome") is not None]
            recent_n = min(em._feature_health_monitor.recent_window, len(all_with_outcomes))
            baseline_n = min(em._feature_health_monitor.baseline_window, len(all_with_outcomes) - recent_n)
            if recent_n >= 50 and baseline_n >= 50:
                recent = all_with_outcomes[-recent_n:]
                baseline = all_with_outcomes[-(recent_n + baseline_n):-recent_n]
                result = em._feature_health_monitor.check_health(
                    recent_trades=recent,
                    baseline_trades=baseline,
                    wfe_monitor=getattr(em, "_wfe_monitor", None),
                )
                em._feature_health_monitor.save_state()
                if result.alert_triggered:
                    logger.warning(
                        "feature_health.alert",
                        score=result.feature_health_score,
                        drifted=result.significant_drift_features,
                    )
        return True


class ClusterAnalyzerHandler:
    """Recluster trade patterns when interval reached."""
    name = "cluster_analyzer"
    priority = 60

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._trade_cluster_analyzer is None:
            return False
        em._trade_cluster_analyzer.record_trade()
        if em._trade_cluster_analyzer.should_recluster():
            all_with_outcomes = [e for e in context.entries if e.get("outcome") is not None]
            if len(all_with_outcomes) >= 50:
                em._trade_cluster_analyzer.fit(all_with_outcomes)
                em._trade_cluster_analyzer.save_state()
        return True


class WalkForwardOptimizerHandler:
    """Record outcomes and trigger walk-forward optimization."""
    name = "walkforward_optimizer"
    priority = 60

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._walkforward_optimizer is None:
            return False
        if context.trade_id not in context.closed_trades:
            return True

        import time as _time_mod
        from datetime import datetime

        entry = context.entry
        ct = context.closed_trades[context.trade_id]
        realized_pl = float(ct.get("realizedPL", 0))
        won = realized_pl > 0
        entry_ts = entry.get("timestamp", "")
        ts = 0.0
        if entry_ts:
            try:
                ts = datetime.fromisoformat(entry_ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = _time_mod.time()
        else:
            ts = _time_mod.time()

        from src.scanner.walkforward_optimizer import TradeOutcome as WFTradeOutcome
        em._walkforward_optimizer.record_outcome(WFTradeOutcome(
            pair=context.pair,
            direction=entry.get("direction", "BUY").upper(),
            pnl=realized_pl,
            confidence=float(entry.get("confidence", 0.5)),
            momentum_score=float(entry.get("agents", {}).get("weighted_vote_score", 0.5)),
            atr_sl_pips=float(entry.get("sl_pips", 0)),
            atr_tp_pips=float(entry.get("tp_pips", 0)),
            regime=entry.get("regime", {}).get("volatility_regime", "NORMAL"),
            won=won,
            timestamp=ts,
        ))

        if em._walkforward_optimizer.should_optimize():
            result = em._walkforward_optimizer.run_optimization()
            em._walkforward_optimizer.save_state()
            logger.info(
                "walkforward_optimizer.optimized",
                wfe=result.wfe,
                recommendation=result.recommendation,
            )
        return True


class WalkForwardPipelineHandler:
    """Composite: Walk-forward retrainer ↔ WFE monitor.

    These are interleaved — the WFE monitor controls whether retraining proceeds.
    Only runs when synced >= 5 trades.
    """
    name = "walkforward_pipeline"
    priority = 60

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._walkforward_retrainer is None:
            return False
        if context.synced_count < 5:
            return True

        from pathlib import Path
        journal_path = str(Path("trained_data/trade_journal_rl.json"))
        epochs = em._walkforward_retrainer.run(
            journal_path=journal_path,
            current_weights_path="trained_data/models/agent_weights.json",
            output_path="trained_data/models/walkforward_epochs.json",
        )
        if not epochs:
            return True

        n_epochs = len(epochs)
        accepted = sum(1 for e in epochs if e.accepted)
        latest_wfe = epochs[-1].wfe if epochs else 0.0

        # WFE monitor gate
        allow_retrain = True
        if em._wfe_monitor is not None:
            em._wfe_monitor.record_wfe(
                wfe=latest_wfe, epoch_count=n_epochs, accepted_count=accepted,
            )
            allow_retrain = em._wfe_monitor.should_retrain()
            if not allow_retrain:
                logger.warning("walkforward_pipeline.wfe_paused")
            em._wfe_monitor.save_state()

        if accepted > 0 and allow_retrain:
            new_weights = em._walkforward_retrainer.get_current_weights()
            if new_weights:
                em._walkforward_retrainer.save_weights("trained_data/models/agent_weights.json")
                if em._wfe_monitor is not None:
                    try:
                        em._wfe_monitor.freeze_weights(new_weights)
                    except Exception:
                        pass
                logger.info(
                    "walkforward_pipeline.weights_saved",
                    accepted=accepted,
                    wfe=latest_wfe,
                )
        return True


class TrancheHandler:
    """Remove closed trades from tranche tracker."""
    name = "tranche_cleanup"
    priority = 60

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._tranche_tracker is None:
            return False
        if context.trade_id in context.closed_trades:
            em._tranche_tracker.remove_trade(context.trade_id)
        return True


class AdaptiveRRHandler:
    """Record R:R target vs actual for learning."""
    name = "adaptive_rr"
    priority = 60

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._adaptive_rr is None:
            return False
        if context.trade_id not in context.closed_trades:
            return True

        entry = context.entry
        sl = float(entry.get("sl_pips", 0))
        tp = float(entry.get("tp_pips", 0))
        regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
        realized = float(context.closed_trades[context.trade_id].get("realizedPL", 0))
        rr_target = tp / sl if sl > 0 else 1.5
        actual_rr = abs(realized) / sl if sl > 0 and realized != 0 else 0.0

        em._adaptive_rr.record_outcome(
            rr_target=rr_target, actual_rr=actual_rr,
            won=realized > 0, regime=str(regime),
        )
        return True


class AdaptiveSizerHandler:
    """Feed trade outcomes to adaptive position sizer."""
    name = "adaptive_sizer"
    priority = 70

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._adaptive_position_sizer is None:
            return False
        if context.trade_id not in context.closed_trades:
            return True
        ct = context.closed_trades[context.trade_id]
        pl = float(ct.get("realizedPL", 0))
        em._adaptive_position_sizer.record_trade(pnl=pl, win=pl > 0)
        return True


class BayesianWeightsHandler:
    """Update pair-specific Bayesian agent weights."""
    name = "bayesian_weights"
    priority = 70

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None or em._pair_bayesian_weights is None:
            return False

        for upd in context.rl_updates:
            pair = upd.get("pair", "")
            verdicts = upd.get("agent_verdicts", {})
            won = upd.get("trade_won", False)
            if not pair or not verdicts:
                continue
            scores = {}
            for agent_name, verdict in verdicts.items():
                if isinstance(verdict, dict):
                    scores[agent_name] = float(verdict.get("score", 0.5))
                elif isinstance(verdict, (int, float)):
                    scores[agent_name] = float(verdict)
            if scores:
                em._pair_bayesian_weights.update(
                    pair=pair, agent_scores=scores,
                    outcome="win" if won else "loss",
                )
        em._pair_bayesian_weights.save_state()
        return True


class AgentAccuracyMatrixHandler:
    """Record agent verdicts to per-pair accuracy matrix."""
    name = "agent_accuracy_matrix"
    priority = 70

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        matrix = getattr(context.scanner, "_agent_accuracy_matrix", None)
        if matrix is None:
            return False

        recorded = 0
        for entry in context.synced_entries:
            outcome = entry.get("outcome")
            if outcome is None:
                continue
            pair = entry.get("pair", "")
            regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
            direction = entry.get("direction", "BUY").upper()
            won = bool(outcome.get("trade_won", False))
            actual = "win" if won else "loss"
            for ar in entry.get("agents", {}).get("agent_reasons", []):
                name = ar.get("name", "")
                if name and ar.get("passed", False):
                    matrix.record_verdict(
                        agent_name=name, pair=pair, regime=regime,
                        predicted_direction=direction, actual_outcome=actual,
                    )
                    recorded += 1

        if recorded > 0:
            matrix.save_state()
        return True


class PairRegimeMatrixHandler:
    """Feed outcomes to pair_regime_agent_matrix."""
    name = "pair_regime_matrix"
    priority = 70

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        pram = getattr(context.scanner, "_pair_regime_agent_matrix", None)
        if pram is None:
            return False

        recorded = 0
        for entry in context.synced_entries:
            outcome = entry.get("outcome")
            if outcome is None:
                continue
            pair = entry.get("pair", "")
            regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
            direction = entry.get("direction", "BUY").upper()
            won = bool(outcome.get("trade_won", False))
            for ar in entry.get("agents", {}).get("agent_reasons", []):
                name = ar.get("name", "")
                if name and ar.get("passed", False):
                    pram.record_vote(
                        pair=pair, regime=regime, agent=name,
                        vote=direction, won=won,
                    )
                    recorded += 1

        if recorded > 0:
            pram.save_state()
        return True


class FastTrackHandler:
    """Track fast-track trade A/B performance."""
    name = "fasttrack_ab"
    priority = 70

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None:
            return False

        for entry in context.synced_entries:
            outcome = entry.get("outcome")
            if outcome is None:
                continue
            ctx = entry.get("analysis_context", entry.get("agents", {}))
            if ctx.get("fasttrack", False):
                em._fasttrack_stats["total"] += 1
                if outcome.get("trade_won", False):
                    em._fasttrack_stats["wins"] += 1
                else:
                    em._fasttrack_stats["losses"] += 1

        total = em._fasttrack_stats["total"]
        if total > 0 and total % 5 == 0:
            wr = em._fasttrack_stats["wins"] / total
            logger.info(
                "fasttrack.ab_report",
                total=total, win_rate=round(wr, 3),
            )
        return True


class CalibrationHandler:
    """Feed outcomes to Phase 18 calibration modules via Scanner."""
    name = "calibration_phase18"
    priority = 70

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        method = getattr(context.scanner, "record_trade_outcome_phase18", None)
        if method is None:
            return False

        for entry in context.synced_entries:
            outcome = entry.get("outcome")
            if outcome is None:
                continue
            agents_info = entry.get("agents", {})
            gate_vals = None
            if agents_info:
                gate_vals = {
                    "confidence": float(entry.get("confidence", 0)),
                    "momentum": float(agents_info.get("weighted_vote_score", 0)),
                    "rr_ratio": (
                        float(entry.get("tp_pips", 0)) / max(float(entry.get("sl_pips", 1)), 0.1)
                    ),
                }
            method(
                pair=entry.get("pair", ""),
                regime=entry.get("regime", {}).get("volatility_regime", "NORMAL"),
                pnl_pips=float(outcome.get("pnl_pips", 0)),
                trade_won=bool(outcome.get("trade_won", False)),
                duration_bars=int(outcome.get("duration_minutes", 0) / 5),
                gate_values=gate_vals,
                agent_verdicts=agents_info.get("agent_reasons"),
                model=entry.get("model", "ensemble"),
                episode_id=entry.get("episode_id", ""),
            )
        return True


class PairPerformanceSummaryHandler:
    """Update per-pair performance summary."""
    name = "pair_performance_summary"
    priority = 80

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None:
            return False
        em._update_pair_performance()
        return True


# ── Post-Journal Handlers (run ONCE after all per-trade handlers) ──────

class AccuracyRebuildHandler:
    """Rebuild accuracy gate from journal to prevent duplicate inflation."""
    name = "accuracy_rebuild"
    priority = 85

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        ag = getattr(context.scanner, "_accuracy_gate", None)
        if ag is None:
            return False
        ag.rebuild_from_journal(str("trained_data/trade_journal_rl.json"))
        return True


class AgentVoteIndexHandler:
    """Bucket D2: keep the agent-vote audit index warm post-sync.

    Reads trade_journal_rl.json incrementally via watermark and upserts rows
    into trained_data/agent_vote_index.db. Cheap (~ms per added trade) and
    swallows DB errors so it never blocks the post-sync pipeline.
    """
    name = "agent_vote_index"
    priority = 95

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        import sqlite3
        try:
            from src.tui.agent_vote_index import AgentVoteIndex
        except ImportError as e:
            logger.debug("AgentVoteIndexHandler: import failed: %s", e)
            return False
        try:
            with AgentVoteIndex.open() as idx:
                stats = idx.sync_from_journal()
            if stats.get("added"):
                logger.info(
                    "agent_vote_index: synced added=%d updated=%d skipped=%d",
                    stats.get("added", 0), stats.get("updated", 0), stats.get("skipped", 0),
                )
            return True
        except (sqlite3.DatabaseError, OSError) as e:
            logger.warning("AgentVoteIndexHandler: DB error: %s", e)
            return False


class RetrainDriftCheckHandler:
    """Check drift and spawn background retrain if needed."""
    name = "retrain_drift_check"
    priority = 85

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        rt = getattr(context.scanner, "_retrain_trigger", None)
        if rt is None:
            return False
        em = context.execution_manager
        drift_request = rt.check_drift()
        if drift_request is not None and em is not None:
            logger.warning(
                "retrain_drift_check.drift_detected",
                pairs=getattr(drift_request, "pairs", []),
            )
            em._spawn_background_retrain(getattr(drift_request, "pairs", []))
        return True


class PredictorTrainingHandler:
    """Auto-train TradeOutcomePredictor at 50-trade threshold."""
    name = "predictor_training"
    priority = 85

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        top = getattr(context.scanner, "_trade_outcome_predictor", None)
        if top is None:
            return False

        complete_trades = [
            e for e in context.entries
            if isinstance(e.get("outcome"), dict)
            and e["outcome"].get("pnl_pips") is not None
            and e["outcome"].get("mae_pips") is not None
        ]
        n = len(complete_trades)
        should_train = n >= 50 and (n % 10 == 0 or not top._is_trained)
        if should_train:
            trained = top.fit_from_journal()
            if trained:
                logger.info("predictor_training.trained", n_trades=n)
        return True


class AffinityHandler:
    """Update pair co-occurrence affinity matrix."""
    name = "pair_affinity"
    priority = 85

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        pa = getattr(context.scanner, "_pair_affinity", None)
        if pa is None:
            return False
        pa.update_from_journal()
        return True


class CounterfactualHandler:
    """Post-loss counterfactual analysis (capped at 5 per sync)."""
    name = "counterfactual"
    priority = 90

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if context.scanner is None:
            return False
        cc = getattr(context.scanner, "_causal_counterfactual", None)
        if cc is None:
            return False

        analyzed = 0
        cap = 5
        for entry in context.synced_entries:
            if analyzed >= cap:
                break
            tid = str(entry.get("trade_id", ""))
            if tid not in context.closed_trades:
                continue
            ct = context.closed_trades[tid]
            pl = float(ct.get("realizedPL", 0))
            if pl >= 0:
                continue  # Only losses
            features = {
                "atr_14": float(entry.get("regime", {}).get("atr_pips", 0.001)),
                "sma_trend": float(entry.get("regime", {}).get("uncertainty_score", 0.5)),
                "momentum_signal": float(entry.get("confidence", 0.5)),
            }
            trade_ctx = {
                "pair": entry.get("pair", ""),
                "features": features,
                "regime": str(entry.get("regime", {}).get("volatility_regime", "NORMAL")),
                "outcome": 0.0,
            }
            scenarios = cc.generate_standard_scenarios(trade_ctx)
            results = cc.batch_analyze(trade_ctx, scenarios)
            cc.save_analysis(tid, results)
            analyzed += 1

        if analyzed > 0:
            logger.info("counterfactual.analyzed", count=analyzed)
        return True


class StatePersistenceHandler:
    """Persist all module states after processing (runs last)."""
    name = "state_persistence"
    priority = 95

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        em = context.execution_manager
        if em is None:
            return False

        if em._adaptive_position_sizer is not None:
            try:
                em._adaptive_position_sizer.save_state("trained_data/adaptive_sizer_state.json")
            except Exception as e:
                logger.debug("state_persistence.adaptive_sizer_failed", error=str(e))

        if em._ewma_correlation is not None:
            try:
                em._ewma_correlation.save_state("trained_data/ewma_correlation_state.json")
            except Exception as e:
                logger.debug("state_persistence.ewma_failed", error=str(e))

        if em._gate_attribution is not None:
            try:
                em._gate_attribution.save_state()
            except Exception as e:
                logger.debug("state_persistence.gate_attribution_failed", error=str(e))

        if em._regime_reward is not None:
            try:
                em._regime_reward.save_log()
            except Exception as e:
                logger.debug("state_persistence.regime_reward_failed", error=str(e))

        return True


class ClaudeReflectionHandler:
    """Spawn headless Claude CLI on trade close to analyze outcome and
    write self-improvement artifacts (learnings, proposed weights, rules).

    Hybrid cadence:
      - Lightweight mode (every close): 60s timeout, ≤500 output tokens
      - Deep mode (loss > $50 OR win > 3R OR 3 consecutive losses): 300s, full MCP

    Priority 100 — runs AFTER state persistence (priority 95) so all mechanical
    RL updates have already flushed to disk before Claude reads the journal.

    Safety:
      - SingleFlightLock prevents parallel spawns when batch sync closes N trades
      - ReflectionBudget enforces daily $ cap ($5/day default)
      - Never raises — all failures logged and return False (handler chain isolation)
    """
    name = "claude_reflection"
    priority = 100

    # Class-level caches — shared across all handler instances within a process
    _recent_outcomes: List[bool] = []  # True=win, False=loss (last 10)
    _RECENT_WINDOW = 10

    def __init__(self, enabled: bool = True, max_journal_tail: int = 20):
        self._enabled = enabled
        self._max_journal_tail = max_journal_tail
        # Lazy import to avoid circular dependency at module load time
        self._invoker = None
        self._budget = None
        self._lock_cls = None

    def _ensure_deps(self) -> None:
        if self._invoker is None:
            from src.scanner.automation.claude_subprocess import (
                invoke_claude_reflection,
                ReflectionBudget,
                SingleFlightLock,
            )
            self._invoker = invoke_claude_reflection
            self._budget = ReflectionBudget()
            self._lock_cls = SingleFlightLock

    def _decide_mode(self, context: SyncContext) -> str:
        """Lightweight by default. Deep when loss > $50, win > 3R, or 3 consecutive losses."""
        realized_pl = float(context.realized_pl or 0.0)
        if realized_pl <= -50.0:
            return "deep"
        # Big win = win with R-multiple > 3. R = tp_pips/sl_pips approximation via
        # pnl_pips vs sl_pips if available.
        entry = context.entry or {}
        sl_pips = float(entry.get("sl_pips", 0) or 0)
        pnl_pips = float(context.pnl_pips or 0)
        if context.trade_won and sl_pips > 0 and pnl_pips >= 3.0 * sl_pips:
            return "deep"
        # 3 consecutive losses
        recent = ClaudeReflectionHandler._recent_outcomes[-3:]
        if len(recent) >= 3 and not any(recent):
            return "deep"
        return "lightweight"

    def _build_prompt(self, context: SyncContext, mode: str) -> str:
        """Construct the trade-reflection prompt. Keep it compact to stay in budget."""
        entry = context.entry or {}
        outcome = context.outcome_data or {}
        # Journal tail — last N entries for pattern reference (skip the current one)
        all_entries = context.entries or []
        tail = all_entries[-self._max_journal_tail :] if all_entries else []

        # Minimal-but-grounded snapshot — NEVER invent fields that aren't here
        entry_snapshot = {
            "pair": entry.get("pair"),
            "direction": entry.get("direction"),
            "confidence": entry.get("confidence"),
            "sl_pips": entry.get("sl_pips"),
            "tp_pips": entry.get("tp_pips"),
            "rr_ratio": entry.get("rr_ratio"),
            "gates": entry.get("gates"),
            "agents": entry.get("agents", {}).get("agent_reasons", []),
            "weighted_vote_score": entry.get("agents", {}).get("weighted_vote_score"),
            "regime": entry.get("regime"),
        }
        import json as _json

        prompt_lines = [
            "Use the trade-reflection skill to analyze this just-closed trade.",
            "",
            f"MODE: {mode}",
            f"TRADE_ID: {context.trade_id}",
            f"PAIR: {context.pair}",
            f"DIRECTION: {entry.get('direction', '')}",
            f"CONFIDENCE: {entry.get('confidence', '')}",
            "",
            "OUTCOME:",
            f"  won: {context.trade_won}",
            f"  realized_pl: {context.realized_pl}",
            f"  pnl_pips: {context.pnl_pips}",
            f"  exit_reason: {outcome.get('exit_reason', '')}",
            f"  duration_minutes: {outcome.get('duration_minutes', '')}",
            "",
            "ENTRY_SNAPSHOT:",
            _json.dumps(entry_snapshot, indent=2, default=str),
            "",
            "JOURNAL_TAIL (last entries for pattern reference, read-only):",
            _json.dumps(
                [
                    {
                        "trade_id": e.get("trade_id"),
                        "pair": e.get("pair"),
                        "direction": e.get("direction"),
                        "confidence": e.get("confidence"),
                        "outcome": e.get("outcome"),
                    }
                    for e in tail
                ],
                indent=2,
                default=str,
            ),
            "",
            "Follow the trade-reflection skill's SAFETY RULES and OUTPUT FORMAT exactly.",
            "End with <reflection-result>...</reflection-result> then <promise>REFLECTION_COMPLETE</promise>.",
        ]
        return "\n".join(prompt_lines)

    def handle(self, event: Dict[str, Any], context: SyncContext) -> bool:
        if not self._enabled:
            return True

        # Track outcome for consecutive-loss detection (class-level)
        try:
            ClaudeReflectionHandler._recent_outcomes.append(bool(context.trade_won))
            if len(ClaudeReflectionHandler._recent_outcomes) > ClaudeReflectionHandler._RECENT_WINDOW:
                ClaudeReflectionHandler._recent_outcomes.pop(0)
        except Exception as e:
            logger.debug("claude_reflection.outcome_track_failed", error=str(e))

        try:
            self._ensure_deps()
        except ImportError as e:
            # Missing claude_subprocess module — handler is opt-in, don't break chain
            logger.warning("claude_reflection.deps_missing", error=str(e))
            return True

        # Budget check
        mode = self._decide_mode(context)
        if not self._budget.allows(mode):
            logger.info(
                "claude_reflection.skipped_budget",
                trade_id=context.trade_id,
                mode=mode,
            )
            return True  # Not an error — just out of budget

        # Single-flight — if another reflection is running, skip this trade.
        # It will be picked up in the next batch sync via journal_tail context.
        lock = self._lock_cls()
        if not lock.acquire():
            logger.info(
                "claude_reflection.skipped_inflight",
                trade_id=context.trade_id,
                mode=mode,
            )
            return True

        try:
            timeout = 420 if mode == "deep" else 90
            prompt = self._build_prompt(context, mode)
            result = self._invoker(
                prompt=prompt,
                trade_id=str(context.trade_id),
                mode=mode,
                timeout_seconds=timeout,
                cwd=Path.cwd(),
            )

            # Record spend regardless of success (we paid for the spawn)
            try:
                self._budget.record(cost_usd=float(result.cost_usd or 0.0), mode=mode)
            except Exception as e:
                logger.debug("claude_reflection.budget_record_failed", error=str(e))

            if result.success:
                logger.info(
                    "claude_reflection.completed",
                    trade_id=context.trade_id,
                    mode=mode,
                    duration=result.duration_seconds,
                    artifacts=result.artifacts_written,
                    hypothesis=result.hypothesis,
                )
                # Tier 2: git commit + push the artifacts (wired below if git_sync exists)
                self._maybe_git_sync(result)
            else:
                logger.warning(
                    "claude_reflection.failed",
                    trade_id=context.trade_id,
                    mode=mode,
                    error=result.error,
                    returncode=result.returncode,
                )
        finally:
            lock.release()

        # Always return True — reflection failure must not poison the handler chain.
        # The error is logged + persisted in logs/reflection_log.jsonl for audit.
        return True

    def _maybe_git_sync(self, result) -> None:
        """If git_sync module present, commit+push whitelisted artifacts.

        Tier 2 wiring — graceful no-op when git_sync not yet installed.
        """
        try:
            from src.scanner.automation.git_sync import commit_and_push_self_improvements
        except ImportError:
            return  # Tier 1 deployment — git sync not yet available
        try:
            commit_and_push_self_improvements(
                artifacts=result.artifacts_written,
                trade_id=result.trade_id,
                hypothesis=result.hypothesis or "",
                mode=result.mode,
            )
        except Exception as e:
            # Git failures must not crash the handler chain.
            # commit_and_push_self_improvements internally logs with context.
            logger.warning("claude_reflection.git_sync_failed", error=str(e))


# ── Registry ───────────────────────────────────────────────────────────

# All per-trade handlers (run once per closed trade)
PER_TRADE_HANDLERS: List[type] = [
    RLReplayHandler,
    EpisodicMemoryHandler,
    PositionTimeoutHandler,
    AccuracyGateHandler,
    RetrainTriggerHandler,
    MetaLearnerHandler,
    EnsembleWeighterHandler,
    MAMLHandler,
    PairPerformanceHandler,
    AttributionPipelineHandler,
    DriftDetectorHandler,
    FeatureHealthHandler,
    ClusterAnalyzerHandler,
    WalkForwardOptimizerHandler,
    TrancheHandler,
    AdaptiveRRHandler,
    AdaptiveSizerHandler,
    ClaudeReflectionHandler,
]

# Post-sync handlers (run once after ALL trades processed)
POST_SYNC_HANDLERS: List[type] = [
    ExpectancyHandler,
    BayesianWeightsHandler,
    AgentAccuracyMatrixHandler,
    PairRegimeMatrixHandler,
    FastTrackHandler,
    CalibrationHandler,
    PairPerformanceSummaryHandler,
    AccuracyRebuildHandler,
    RetrainDriftCheckHandler,
    PredictorTrainingHandler,
    AffinityHandler,
    CounterfactualHandler,
    WalkForwardPipelineHandler,
    StatePersistenceHandler,
    AgentVoteIndexHandler,
]


def get_all_handler_instances() -> List[Any]:
    """Instantiate all handlers, sorted by priority."""
    all_classes = PER_TRADE_HANDLERS + POST_SYNC_HANDLERS
    instances = [cls() for cls in all_classes]
    instances.sort(key=lambda h: h.priority)
    return instances


def get_handler_count() -> int:
    """Total number of registered handler classes."""
    return len(PER_TRADE_HANDLERS) + len(POST_SYNC_HANDLERS)
