"""MAML Ridge Phase 2 Benchmarking — Shadow Mode Metrics & Analysis.

Collects, persists, and analyzes shadow-mode comparison data to determine
whether MAML Ridge should be promoted from shadow to production weight.

Key metrics:
    1. Shadow MAE — mean absolute error between MAML predictions and realized outcomes
    2. Ridge MAE — same for Ridge (baseline comparison)
    3. Adaptation speed — trades until MAML converges after regime transition
    4. Regime divergence — do MAML predictions meaningfully differ across regimes
    5. Meta-train loss curve — is the MAML outer loop actually improving
    6. Calibration — how well do predicted confidences match actual win rates

Usage:
    from src.recursive_intelligence.maml_benchmark import MAMLBenchmark

    bench = MAMLBenchmark()

    # On each shadow prediction:
    bench.record_prediction(regime, maml_conf, ridge_conf)

    # On trade close:
    bench.record_outcome(trade_id, regime, maml_conf, ridge_conf, realized_conf, pnl_pips, won)

    # On regime transition:
    bench.record_regime_transition(old_regime, new_regime)

    # On meta-train:
    bench.record_meta_train(train_stats)

    # Generate report:
    report = bench.generate_report()
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_BENCHMARK_DIR = Path("trained_data/maml_benchmark")
_REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark collection and analysis."""
    # Persistence
    max_predictions: int = 5000       # Max prediction records (sliding window)
    max_outcomes: int = 2000          # Max outcome records (sliding window)
    save_interval: int = 10           # Persist every N new outcomes

    # Convergence detection
    convergence_window: int = 10      # Rolling window for convergence MAE
    convergence_threshold: float = 5.0  # MAE below this = "converged" (0-100 scale)

    # Report thresholds
    min_outcomes_for_report: int = 20  # Min outcomes before generating meaningful report
    promotion_mae_advantage: float = 2.0  # MAML must beat Ridge MAE by this much to promote
    min_regime_samples: int = 5       # Min samples per regime for regime analysis


@dataclass
class PredictionRecord:
    """A single shadow prediction comparison."""
    timestamp: float
    regime: str
    maml_confidence: float
    ridge_confidence: float
    scan_cycle: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": round(self.timestamp, 3),
            "regime": self.regime,
            "maml": round(self.maml_confidence, 2),
            "ridge": round(self.ridge_confidence, 2),
            "cycle": self.scan_cycle,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionRecord":
        return cls(
            timestamp=float(d["ts"]),
            regime=str(d["regime"]),
            maml_confidence=float(d["maml"]),
            ridge_confidence=float(d["ridge"]),
            scan_cycle=int(d.get("cycle", 0)),
        )


@dataclass
class OutcomeRecord:
    """A trade outcome with both MAML and Ridge predictions."""
    timestamp: float
    trade_id: str
    regime: str
    maml_confidence: float
    ridge_confidence: float
    realized_confidence: float  # Ground truth derived from outcome
    pnl_pips: float
    won: bool
    regime_age: int = 0  # Trades since last regime transition
    # Phase 2 enriched fields
    exit_reason: str = ""           # tp|sl|ts|manual|timeout
    duration_minutes: Optional[float] = None
    regime_at_exit: str = ""        # May differ from entry regime

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ts": round(self.timestamp, 3),
            "trade_id": self.trade_id,
            "regime": self.regime,
            "maml": round(self.maml_confidence, 2),
            "ridge": round(self.ridge_confidence, 2),
            "realized": round(self.realized_confidence, 2),
            "pnl": round(self.pnl_pips, 2),
            "won": self.won,
            "regime_age": self.regime_age,
        }
        if self.exit_reason:
            d["exit_reason"] = self.exit_reason
        if self.duration_minutes is not None:
            d["duration"] = round(self.duration_minutes, 1)
        if self.regime_at_exit:
            d["exit_regime"] = self.regime_at_exit
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutcomeRecord":
        return cls(
            timestamp=float(d["ts"]),
            trade_id=str(d["trade_id"]),
            regime=str(d["regime"]),
            maml_confidence=float(d["maml"]),
            ridge_confidence=float(d["ridge"]),
            realized_confidence=float(d["realized"]),
            pnl_pips=float(d["pnl"]),
            won=bool(d["won"]),
            regime_age=int(d.get("regime_age", 0)),
            exit_reason=str(d.get("exit_reason", "")),
            duration_minutes=float(d["duration"]) if "duration" in d else None,
            regime_at_exit=str(d.get("exit_regime", "")),
        )


@dataclass
class RegimeTransition:
    """Records a regime change for adaptation speed measurement."""
    timestamp: float
    old_regime: str
    new_regime: str
    trades_since: int = 0  # Trades counted since this transition
    converged_after: Optional[int] = None  # Trades until convergence (None = not yet)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": round(self.timestamp, 3),
            "old": self.old_regime,
            "new": self.new_regime,
            "trades": self.trades_since,
            "converged": self.converged_after,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeTransition":
        return cls(
            timestamp=float(d["ts"]),
            old_regime=str(d["old"]),
            new_regime=str(d["new"]),
            trades_since=int(d.get("trades", 0)),
            converged_after=d.get("converged"),
        )


@dataclass
class MetaTrainRecord:
    """Records a meta-training event."""
    timestamp: float
    train_number: int
    outer_loss: float
    regime_losses: Dict[str, float]
    regimes_used: List[str]
    # Phase 2 diagnostics
    inner_losses: Dict[str, List[float]] = field(default_factory=dict)  # Per-regime inner loop trajectory
    support_losses: Dict[str, float] = field(default_factory=dict)      # Final support loss per regime
    param_drift: Dict[str, float] = field(default_factory=dict)         # L2 param change per regime

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ts": round(self.timestamp, 3),
            "n": self.train_number,
            "loss": round(self.outer_loss, 6),
            "regime_losses": {k: round(v, 6) for k, v in self.regime_losses.items()},
            "regimes": self.regimes_used,
        }
        if self.inner_losses:
            d["inner_losses"] = {
                k: [round(v, 6) for v in losses]
                for k, losses in self.inner_losses.items()
            }
        if self.support_losses:
            d["support_losses"] = {k: round(v, 6) for k, v in self.support_losses.items()}
        if self.param_drift:
            d["param_drift"] = {k: round(v, 6) for k, v in self.param_drift.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetaTrainRecord":
        return cls(
            timestamp=float(d["ts"]),
            train_number=int(d["n"]),
            outer_loss=float(d["loss"]),
            regime_losses={k: float(v) for k, v in d.get("regime_losses", {}).items()},
            regimes_used=list(d.get("regimes", [])),
            inner_losses={
                k: [float(v) for v in losses]
                for k, losses in d.get("inner_losses", {}).items()
            },
            support_losses={k: float(v) for k, v in d.get("support_losses", {}).items()},
            param_drift={k: float(v) for k, v in d.get("param_drift", {}).items()},
        )


class MAMLBenchmark:
    """Collects and analyzes MAML shadow mode benchmarking data.

    Tracks three streams of data:
    1. Predictions — every shadow comparison (MAML vs Ridge)
    2. Outcomes — trade results with ground truth
    3. Regime transitions — for adaptation speed measurement

    Analysis capabilities:
    - MAE comparison (MAML vs Ridge vs realized)
    - Adaptation speed (convergence after regime transition)
    - Regime divergence (MAML specialization across regimes)
    - Meta-train loss curve (is learning happening)
    - Promotion readiness (should MAML replace Ridge)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

        # Data stores
        self._predictions: List[PredictionRecord] = []
        self._outcomes: List[OutcomeRecord] = []
        self._transitions: List[RegimeTransition] = []
        self._meta_trains: List[MetaTrainRecord] = []

        # Live tracking
        self._current_regime: str = "NORMAL"
        self._trades_since_transition: int = 0
        self._scan_cycle: int = 0
        self._outcomes_since_save: int = 0

        # Load persisted state
        self._load_state()

        logger.info(
            "MAML Benchmark initialized (predictions=%d, outcomes=%d, transitions=%d)",
            len(self._predictions),
            len(self._outcomes),
            len(self._transitions),
        )

    # ── Recording Methods ───────────────────────────────────────────────

    def record_prediction(
        self,
        regime: str,
        maml_confidence: float,
        ridge_confidence: float,
    ) -> None:
        """Record a shadow prediction comparison."""
        regime = (regime or "NORMAL").upper()

        # Auto-detect regime transitions
        if regime != self._current_regime:
            self.record_regime_transition(self._current_regime, regime)

        rec = PredictionRecord(
            timestamp=time.time(),
            regime=regime,
            maml_confidence=float(maml_confidence),
            ridge_confidence=float(ridge_confidence),
            scan_cycle=self._scan_cycle,
        )
        self._predictions.append(rec)

        # Sliding window
        if len(self._predictions) > self.config.max_predictions:
            self._predictions = self._predictions[-self.config.max_predictions:]

    def record_outcome(
        self,
        trade_id: str,
        regime: str,
        maml_confidence: float,
        ridge_confidence: float,
        realized_confidence: float,
        pnl_pips: float,
        won: bool,
        exit_reason: str = "",
        duration_minutes: Optional[float] = None,
        regime_at_exit: str = "",
    ) -> None:
        """Record a trade outcome with both model predictions."""
        regime = (regime or "NORMAL").upper()

        rec = OutcomeRecord(
            timestamp=time.time(),
            trade_id=str(trade_id),
            regime=regime,
            maml_confidence=float(maml_confidence),
            ridge_confidence=float(ridge_confidence),
            realized_confidence=float(realized_confidence),
            pnl_pips=float(pnl_pips),
            won=bool(won),
            regime_age=self._trades_since_transition,
            exit_reason=str(exit_reason),
            duration_minutes=float(duration_minutes) if duration_minutes is not None else None,
            regime_at_exit=str(regime_at_exit).upper() if regime_at_exit else "",
        )
        self._outcomes.append(rec)
        self._trades_since_transition += 1

        # Sliding window
        if len(self._outcomes) > self.config.max_outcomes:
            self._outcomes = self._outcomes[-self.config.max_outcomes:]

        # Update convergence tracking on active transitions
        self._update_convergence(rec)

        # Auto-persist
        self._outcomes_since_save += 1
        if self._outcomes_since_save >= self.config.save_interval:
            self._save_state()
            self._outcomes_since_save = 0

    def record_regime_transition(self, old_regime: str, new_regime: str) -> None:
        """Record a regime change for adaptation speed tracking."""
        old_regime = (old_regime or "NORMAL").upper()
        new_regime = (new_regime or "NORMAL").upper()

        if old_regime == new_regime:
            return  # Not a real transition

        trans = RegimeTransition(
            timestamp=time.time(),
            old_regime=old_regime,
            new_regime=new_regime,
        )
        self._transitions.append(trans)
        self._trades_since_transition = 0
        self._current_regime = new_regime

        logger.info(
            "MAML Benchmark: regime transition %s → %s",
            old_regime, new_regime,
        )

    def record_meta_train(self, train_stats: Dict[str, Any]) -> None:
        """Record a meta-training event from MAMLRidge.meta_train()."""
        if train_stats.get("status") != "trained":
            return

        rec = MetaTrainRecord(
            timestamp=time.time(),
            train_number=int(train_stats.get("meta_trains", 0)),
            outer_loss=float(train_stats.get("outer_loss", 0.0)),
            regime_losses={k: float(v) for k, v in train_stats.get("regime_losses", {}).items()},
            regimes_used=list(train_stats.get("regimes_used", [])),
            inner_losses={
                k: [float(v) for v in losses]
                for k, losses in train_stats.get("inner_losses", {}).items()
            },
            support_losses={k: float(v) for k, v in train_stats.get("support_losses", {}).items()},
            param_drift={k: float(v) for k, v in train_stats.get("param_drift", {}).items()},
        )
        self._meta_trains.append(rec)

    def set_scan_cycle(self, cycle: int) -> None:
        """Update the current scan cycle number."""
        self._scan_cycle = cycle

    # ── Analysis Methods ────────────────────────────────────────────────

    def compute_mae(
        self,
        model: str = "maml",
        regime: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> Optional[float]:
        """Compute Mean Absolute Error for a model against realized outcomes.

        Args:
            model: "maml" or "ridge"
            regime: Optional regime filter
            last_n: Optional — only use last N outcomes

        Returns:
            MAE or None if insufficient data
        """
        outcomes = self._filter_outcomes(regime=regime, last_n=last_n)
        if not outcomes:
            return None

        errors = []
        for o in outcomes:
            predicted = o.maml_confidence if model == "maml" else o.ridge_confidence
            errors.append(abs(predicted - o.realized_confidence))

        return round(float(np.mean(errors)), 4)

    def compute_regime_mae(self, model: str = "maml") -> Dict[str, Optional[float]]:
        """Compute MAE per regime."""
        result = {}
        for regime in _REGIME_NAMES:
            result[regime] = self.compute_mae(model=model, regime=regime)
        return result

    def compute_adaptation_speed(self) -> Dict[str, Any]:
        """Analyze adaptation speed after regime transitions.

        Returns:
            Dict with convergence stats:
            - "transitions": total transitions observed
            - "converged": number that reached convergence
            - "avg_trades_to_converge": mean trades until convergence
            - "per_transition": details of each transition
        """
        if not self._transitions:
            return {"transitions": 0, "converged": 0, "avg_trades_to_converge": None}

        converged = [t for t in self._transitions if t.converged_after is not None]
        per_transition = []
        for t in self._transitions:
            per_transition.append({
                "from": t.old_regime,
                "to": t.new_regime,
                "trades_since": t.trades_since,
                "converged_after": t.converged_after,
                "timestamp": t.timestamp,
            })

        avg_converge = None
        if converged:
            avg_converge = round(
                float(np.mean([t.converged_after for t in converged])),
                1,
            )

        return {
            "transitions": len(self._transitions),
            "converged": len(converged),
            "avg_trades_to_converge": avg_converge,
            "per_transition": per_transition,
        }

    def compute_regime_divergence(self) -> Dict[str, Any]:
        """Measure how much MAML predictions diverge across regimes.

        If MAML is learning regime-specific patterns, its predictions should
        differ meaningfully across regimes. If all regimes produce similar
        predictions, the adaptation isn't working.

        Returns:
            Dict with divergence analysis:
            - "regime_means": mean MAML prediction per regime
            - "regime_stds": std of MAML predictions per regime
            - "cross_regime_std": std of regime means (higher = more divergence)
            - "divergence_score": normalized 0-1 score (>0.3 is meaningful)
            - "verdict": human-readable assessment
        """
        regime_predictions: Dict[str, List[float]] = defaultdict(list)

        for o in self._outcomes:
            regime_predictions[o.regime].append(o.maml_confidence)

        # Also use prediction records for higher sample count
        for p in self._predictions:
            regime_predictions[p.regime].append(p.maml_confidence)

        regime_means = {}
        regime_stds = {}
        for regime in _REGIME_NAMES:
            preds = regime_predictions.get(regime, [])
            if len(preds) >= self.config.min_regime_samples:
                regime_means[regime] = round(float(np.mean(preds)), 2)
                regime_stds[regime] = round(float(np.std(preds)), 2)

        # Cross-regime divergence
        if len(regime_means) < 2:
            return {
                "regime_means": regime_means,
                "regime_stds": regime_stds,
                "cross_regime_std": None,
                "divergence_score": None,
                "verdict": "insufficient_data",
            }

        means_array = np.array(list(regime_means.values()))
        cross_std = float(np.std(means_array))

        # Normalize to 0-1 (divide by typical confidence range ~50)
        divergence_score = min(1.0, cross_std / 50.0)

        if divergence_score > 0.3:
            verdict = "strong_divergence"
        elif divergence_score > 0.1:
            verdict = "moderate_divergence"
        elif divergence_score > 0.03:
            verdict = "weak_divergence"
        else:
            verdict = "no_divergence"

        return {
            "regime_means": regime_means,
            "regime_stds": regime_stds,
            "cross_regime_std": round(cross_std, 4),
            "divergence_score": round(divergence_score, 4),
            "verdict": verdict,
        }

    def compute_meta_train_curve(self) -> Dict[str, Any]:
        """Analyze meta-training loss curve for learning progress.

        Returns:
            Dict with learning curve analysis:
            - "total_trains": number of meta-training rounds
            - "losses": list of outer losses
            - "trend": "improving" | "flat" | "degrading"
            - "improvement_pct": % improvement from first to last 3 trains
        """
        if not self._meta_trains:
            return {"total_trains": 0, "losses": [], "trend": "no_data"}

        losses = [t.outer_loss for t in self._meta_trains]

        result: Dict[str, Any] = {
            "total_trains": len(self._meta_trains),
            "losses": [round(l, 6) for l in losses],
            "first_loss": round(losses[0], 6),
            "last_loss": round(losses[-1], 6),
        }

        if len(losses) < 3:
            result["trend"] = "insufficient_data"
            result["improvement_pct"] = None
            return result

        # Compare first 3 vs last 3
        first_avg = float(np.mean(losses[:3]))
        last_avg = float(np.mean(losses[-3:]))

        if first_avg > 0:
            improvement = (first_avg - last_avg) / first_avg * 100
        else:
            improvement = 0.0

        result["improvement_pct"] = round(improvement, 2)

        if improvement > 10:
            result["trend"] = "improving"
        elif improvement < -10:
            result["trend"] = "degrading"
        else:
            result["trend"] = "flat"

        # Per-regime loss curves
        regime_curves: Dict[str, List[float]] = defaultdict(list)
        for t in self._meta_trains:
            for regime, loss in t.regime_losses.items():
                regime_curves[regime].append(loss)
        result["regime_curves"] = {
            r: [round(l, 6) for l in losses_list]
            for r, losses_list in regime_curves.items()
        }

        return result

    def compute_calibration(self, bins: int = 5) -> Dict[str, Any]:
        """Measure prediction calibration — do confidence levels match win rates.

        Bins outcomes by predicted confidence and checks if higher confidence
        actually corresponds to higher win rates.

        Args:
            bins: Number of confidence bins (default 5: 0-20, 20-40, 40-60, 60-80, 80-100)

        Returns:
            Dict with calibration analysis for both MAML and Ridge
        """
        if len(self._outcomes) < self.config.min_outcomes_for_report:
            return {"status": "insufficient_data", "count": len(self._outcomes)}

        edges = np.linspace(0, 100, bins + 1)

        def _calibrate(model: str) -> List[Dict[str, Any]]:
            bin_results = []
            for i in range(bins):
                lo, hi = float(edges[i]), float(edges[i + 1])
                in_bin = [
                    o for o in self._outcomes
                    if lo <= (o.maml_confidence if model == "maml" else o.ridge_confidence) < hi
                    or (i == bins - 1 and (o.maml_confidence if model == "maml" else o.ridge_confidence) == hi)
                ]
                if in_bin:
                    win_rate = sum(1 for o in in_bin if o.won) / len(in_bin)
                    avg_pred = float(np.mean([
                        o.maml_confidence if model == "maml" else o.ridge_confidence
                        for o in in_bin
                    ]))
                    avg_pnl = float(np.mean([o.pnl_pips for o in in_bin]))
                else:
                    win_rate = None
                    avg_pred = None
                    avg_pnl = None

                bin_results.append({
                    "range": f"{lo:.0f}-{hi:.0f}",
                    "count": len(in_bin),
                    "win_rate": round(win_rate, 4) if win_rate is not None else None,
                    "avg_predicted": round(avg_pred, 2) if avg_pred is not None else None,
                    "avg_pnl": round(avg_pnl, 2) if avg_pnl is not None else None,
                })
            return bin_results

        return {
            "status": "ok",
            "count": len(self._outcomes),
            "maml": _calibrate("maml"),
            "ridge": _calibrate("ridge"),
        }

    def compute_win_rate_by_model(self, regime: Optional[str] = None) -> Dict[str, Any]:
        """Compare win rates for trades where MAML vs Ridge was more confident.

        For each trade, determine which model was more confident.
        Then compare win rates between the two groups.
        """
        outcomes = self._filter_outcomes(regime=regime)
        if len(outcomes) < self.config.min_outcomes_for_report:
            return {"status": "insufficient_data", "count": len(outcomes)}

        maml_more_confident = [o for o in outcomes if o.maml_confidence > o.ridge_confidence]
        ridge_more_confident = [o for o in outcomes if o.ridge_confidence > o.maml_confidence]
        equal = [o for o in outcomes if abs(o.maml_confidence - o.ridge_confidence) < 0.5]

        def _wr(group: List[OutcomeRecord]) -> Optional[float]:
            if not group:
                return None
            return round(sum(1 for o in group if o.won) / len(group), 4)

        return {
            "status": "ok",
            "total_trades": len(outcomes),
            "maml_more_confident": {
                "count": len(maml_more_confident),
                "win_rate": _wr(maml_more_confident),
            },
            "ridge_more_confident": {
                "count": len(ridge_more_confident),
                "win_rate": _wr(ridge_more_confident),
            },
            "equal": {
                "count": len(equal),
                "win_rate": _wr(equal),
            },
        }

    # ── Statistical Analysis ───────────────────────────────────────────

    def compute_significance_test(self) -> Dict[str, Any]:
        """Paired statistical test: is MAML significantly better than Ridge?

        Uses paired t-test (same trades) and Wilcoxon signed-rank (non-parametric).
        Reports effect size (Cohen's d) for practical significance.

        Returns:
            Dict with test results, p-values, and interpretation
        """
        outcomes = self._outcomes
        if len(outcomes) < 30:
            return {"status": "insufficient_data", "samples": len(outcomes), "needed": 30}

        maml_errors = np.array([abs(o.maml_confidence - o.realized_confidence) for o in outcomes])
        ridge_errors = np.array([abs(o.ridge_confidence - o.realized_confidence) for o in outcomes])
        differences = maml_errors - ridge_errors  # Negative = MAML better

        try:
            from scipy.stats import ttest_rel, wilcoxon
        except ImportError:
            return {"status": "scipy_unavailable"}

        # Paired t-test
        t_stat, t_pval = ttest_rel(maml_errors, ridge_errors)

        # Wilcoxon signed-rank (non-parametric — robust to non-normal distributions)
        try:
            w_stat, w_pval = wilcoxon(maml_errors, ridge_errors)
        except ValueError:
            # All differences are zero
            w_stat, w_pval = 0.0, 1.0

        # Effect size: Cohen's d for paired samples
        mean_diff = float(np.mean(differences))
        std_diff = float(np.std(differences, ddof=1)) if len(differences) > 1 else 1.0
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        # Bootstrap 95% CI on MAE difference
        ci_low, ci_high = self._bootstrap_ci(differences, n_bootstrap=1000)

        # Interpret
        if abs(cohens_d) >= 0.8:
            effect_label = "large"
        elif abs(cohens_d) >= 0.5:
            effect_label = "medium"
        elif abs(cohens_d) >= 0.2:
            effect_label = "small"
        else:
            effect_label = "negligible"

        maml_better = mean_diff < 0  # Negative = MAML has lower error
        significant = t_pval < 0.05

        if significant and abs(cohens_d) >= 0.5:
            interpretation = f"{'MAML' if maml_better else 'Ridge'} is significantly better with {effect_label} effect"
        elif significant:
            interpretation = f"Statistically significant but {effect_label} effect — may not be practically meaningful"
        else:
            interpretation = "No statistically significant difference between models"

        return {
            "status": "ok",
            "samples": len(outcomes),
            "maml_mean_error": round(float(np.mean(maml_errors)), 4),
            "ridge_mean_error": round(float(np.mean(ridge_errors)), 4),
            "mean_difference": round(mean_diff, 4),
            "t_statistic": round(float(t_stat), 4),
            "t_pvalue": round(float(t_pval), 6),
            "wilcoxon_pvalue": round(float(w_pval), 6),
            "cohens_d": round(cohens_d, 4),
            "effect_size": effect_label,
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "significant_at_005": significant,
            "maml_better": maml_better,
            "interpretation": interpretation,
        }

    def compute_calibration_metrics(self) -> Dict[str, Any]:
        """Compute ECE, MCE, and Brier score for both models.

        ECE (Expected Calibration Error): weighted average gap between
            predicted confidence and observed win rate across bins.
        MCE (Max Calibration Error): worst-case miscalibration.
        Brier Score: MSE between predicted probability and binary outcome.
        """
        if len(self._outcomes) < 20:
            return {"status": "insufficient_data", "count": len(self._outcomes)}

        outcomes_arr = np.array([1.0 if o.won else 0.0 for o in self._outcomes])

        def _metrics(confs_raw: np.ndarray, name: str) -> Dict[str, Any]:
            confs = confs_raw / 100.0  # Normalize to 0-1 probability scale

            # Brier Score: MSE between predicted prob and binary outcome
            brier = float(np.mean((confs - outcomes_arr) ** 2))

            # ECE and MCE via binning
            n_bins = 10
            ece = 0.0
            mce = 0.0
            bins = []

            for i in range(n_bins):
                lo = i / n_bins
                hi = (i + 1) / n_bins
                mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 else (confs >= lo) & (confs <= hi)
                count = int(mask.sum())

                if count == 0:
                    continue

                avg_conf = float(confs[mask].mean())
                actual_wr = float(outcomes_arr[mask].mean())
                gap = abs(avg_conf - actual_wr)
                weight = count / len(confs)

                ece += weight * gap
                mce = max(mce, gap)

                bins.append({
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "count": count,
                    "predicted": round(avg_conf, 4),
                    "actual": round(actual_wr, 4),
                    "gap": round(gap, 4),
                })

            return {
                "name": name,
                "brier_score": round(brier, 6),
                "ece": round(ece, 6),
                "mce": round(mce, 6),
                "bins": bins,
            }

        maml_confs = np.array([o.maml_confidence for o in self._outcomes])
        ridge_confs = np.array([o.ridge_confidence for o in self._outcomes])

        return {
            "status": "ok",
            "count": len(self._outcomes),
            "maml": _metrics(maml_confs, "MAML"),
            "ridge": _metrics(ridge_confs, "Ridge"),
            "reference": {
                "brier": "Lower is better. 0=perfect, 0.25=random on 50/50",
                "ece": "Lower is better. <0.05=well calibrated, >0.15=poor",
                "mce": "Lower is better. Worst single bin gap",
            },
        }

    def compute_error_time_series(self) -> Dict[str, Any]:
        """Analyze error trajectory over time — trend, velocity, autocorrelation.

        Detects whether prediction quality is improving, degrading, or oscillating.
        """
        if len(self._outcomes) < 15:
            return {"status": "insufficient_data", "count": len(self._outcomes)}

        maml_errors = np.array([abs(o.maml_confidence - o.realized_confidence) for o in self._outcomes])
        ridge_errors = np.array([abs(o.ridge_confidence - o.realized_confidence) for o in self._outcomes])

        def _analyze(errors: np.ndarray, name: str) -> Dict[str, Any]:
            n = len(errors)

            # Linear trend (slope via least squares)
            x = np.arange(n, dtype=np.float64)
            slope, intercept = np.polyfit(x, errors, 1)

            # EWMA (exponentially weighted moving average) — recent errors weighted more
            alpha = 2 / (min(n, 20) + 1)
            ewma = np.zeros(n)
            ewma[0] = errors[0]
            for i in range(1, n):
                ewma[i] = alpha * errors[i] + (1 - alpha) * ewma[i - 1]

            # Autocorrelation at lag 1 (are errors clustered?)
            if n > 5:
                mean_e = errors.mean()
                denom = np.sum((errors - mean_e) ** 2)
                if denom > 0:
                    autocorr_1 = float(np.sum((errors[:-1] - mean_e) * (errors[1:] - mean_e)) / denom)
                else:
                    autocorr_1 = 0.0
            else:
                autocorr_1 = None

            # Error velocity: how fast is error changing (slope relative to mean)
            mean_error = float(errors.mean())
            velocity_pct = (float(slope) / mean_error * 100) if mean_error > 0 else 0.0

            if velocity_pct < -5:
                trend = "improving"
            elif velocity_pct > 5:
                trend = "degrading"
            else:
                trend = "stable"

            return {
                "name": name,
                "trend": trend,
                "slope_per_trade": round(float(slope), 6),
                "velocity_pct": round(velocity_pct, 2),
                "current_ewma": round(float(ewma[-1]), 4),
                "initial_ewma": round(float(ewma[min(5, n - 1)]), 4),
                "autocorrelation_lag1": round(autocorr_1, 4) if autocorr_1 is not None else None,
                "mean_error": round(mean_error, 4),
                "std_error": round(float(errors.std()), 4),
                "min_error": round(float(errors.min()), 4),
                "max_error": round(float(errors.max()), 4),
            }

        return {
            "status": "ok",
            "count": len(self._outcomes),
            "maml": _analyze(maml_errors, "MAML"),
            "ridge": _analyze(ridge_errors, "Ridge"),
        }

    def compute_inner_loop_diagnostics(self) -> Dict[str, Any]:
        """Analyze inner-loop convergence from meta-training records.

        Checks:
        - Do inner losses decrease (convergence)?
        - Is convergence rate consistent across regimes?
        - Are there oscillations or instability?
        """
        trains_with_inner = [t for t in self._meta_trains if t.inner_losses]
        if not trains_with_inner:
            return {"status": "no_inner_loss_data", "total_trains": len(self._meta_trains)}

        regime_convergence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for train in trains_with_inner:
            for regime, losses in train.inner_losses.items():
                if len(losses) < 2:
                    continue

                first = losses[0]
                last = losses[-1]
                reduction = (first - last) / first * 100 if first > 0 else 0.0
                monotonic = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
                oscillating = not monotonic and reduction > 0  # Decreased overall but not monotonically

                regime_convergence[regime].append({
                    "train_n": train.train_number,
                    "losses": [round(l, 6) for l in losses],
                    "reduction_pct": round(reduction, 2),
                    "monotonic": monotonic,
                    "oscillating": oscillating,
                })

        # Aggregate per regime
        regime_summary = {}
        for regime, records in regime_convergence.items():
            reductions = [r["reduction_pct"] for r in records]
            monotonic_pct = sum(1 for r in records if r["monotonic"]) / len(records) * 100

            regime_summary[regime] = {
                "train_count": len(records),
                "avg_reduction_pct": round(float(np.mean(reductions)), 2),
                "monotonic_pct": round(monotonic_pct, 1),
                "healthy": float(np.mean(reductions)) > 10 and monotonic_pct > 50,
            }

        return {
            "status": "ok",
            "total_trains_with_data": len(trains_with_inner),
            "regime_summary": regime_summary,
            "per_regime_details": {k: v for k, v in regime_convergence.items()},
        }

    def compute_param_drift_analysis(self) -> Dict[str, Any]:
        """Analyze parameter drift magnitude across meta-training rounds.

        Tracks how much meta-parameters change per training round.
        Decreasing drift → convergence. Increasing drift → instability.
        """
        trains_with_drift = [t for t in self._meta_trains if t.param_drift]
        if not trains_with_drift:
            return {"status": "no_drift_data", "total_trains": len(self._meta_trains)}

        # Total drift per training round
        total_drifts = []
        per_regime_drifts: Dict[str, List[float]] = defaultdict(list)

        for train in trains_with_drift:
            total = sum(train.param_drift.values())
            total_drifts.append(total)
            for regime, drift in train.param_drift.items():
                per_regime_drifts[regime].append(drift)

        drift_arr = np.array(total_drifts)

        # Trend of drift
        if len(drift_arr) >= 3:
            x = np.arange(len(drift_arr), dtype=np.float64)
            slope, _ = np.polyfit(x, drift_arr, 1)
            if slope < -0.01:
                trend = "converging"
            elif slope > 0.01:
                trend = "diverging"
            else:
                trend = "stable"
        else:
            slope = None
            trend = "insufficient_data"

        return {
            "status": "ok",
            "total_trains": len(trains_with_drift),
            "trend": trend,
            "slope": round(float(slope), 6) if slope is not None else None,
            "mean_drift": round(float(drift_arr.mean()), 6),
            "std_drift": round(float(drift_arr.std()), 6),
            "latest_drift": round(float(drift_arr[-1]), 6),
            "per_regime": {
                r: {
                    "mean": round(float(np.mean(drifts)), 6),
                    "latest": round(float(drifts[-1]), 6) if drifts else None,
                }
                for r, drifts in per_regime_drifts.items()
            },
        }

    def compute_exit_reason_analysis(self) -> Dict[str, Any]:
        """Analyze model accuracy by exit type (TP, SL, trailing stop, etc).

        A good model should be MORE accurate on TP exits (model was right)
        and LESS accurate on SL exits (model was wrong).
        """
        outcomes_with_exit = [o for o in self._outcomes if o.exit_reason]
        if len(outcomes_with_exit) < 10:
            return {"status": "insufficient_data", "count": len(outcomes_with_exit)}

        by_exit: Dict[str, List[OutcomeRecord]] = defaultdict(list)
        for o in outcomes_with_exit:
            by_exit[o.exit_reason].append(o)

        result = {"status": "ok", "exit_types": {}}
        for exit_type, records in by_exit.items():
            maml_errs = [abs(r.maml_confidence - r.realized_confidence) for r in records]
            ridge_errs = [abs(r.ridge_confidence - r.realized_confidence) for r in records]
            result["exit_types"][exit_type] = {
                "count": len(records),
                "maml_mae": round(float(np.mean(maml_errs)), 4),
                "ridge_mae": round(float(np.mean(ridge_errs)), 4),
                "win_rate": round(sum(1 for r in records if r.won) / len(records), 4),
                "avg_pnl": round(float(np.mean([r.pnl_pips for r in records])), 2),
            }

        return result

    def compute_required_sample_size(
        self,
        target_improvement: float = 2.0,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> Dict[str, Any]:
        """Calculate how many more trades are needed for statistical significance.

        Uses observed variance to estimate required sample size for paired t-test.
        """
        if len(self._outcomes) < 10:
            return {"status": "insufficient_data", "estimate": 100}

        maml_errors = np.array([abs(o.maml_confidence - o.realized_confidence) for o in self._outcomes])
        ridge_errors = np.array([abs(o.ridge_confidence - o.realized_confidence) for o in self._outcomes])
        diffs = maml_errors - ridge_errors
        observed_std = float(np.std(diffs, ddof=1))

        if observed_std <= 0:
            return {"status": "zero_variance", "estimate": 30}

        try:
            from scipy.stats import norm
            z_alpha = float(norm.ppf(1 - alpha / 2))
            z_beta = float(norm.ppf(power))
        except ImportError:
            z_alpha, z_beta = 1.96, 0.84

        # Required n for paired t-test
        effect_size = target_improvement / observed_std
        if effect_size <= 0:
            return {"status": "invalid_effect", "estimate": 500}

        required_n = int(np.ceil(((z_alpha + z_beta) / effect_size) ** 2))
        required_n = max(required_n, 30)

        remaining = max(0, required_n - len(self._outcomes))

        return {
            "status": "ok",
            "current_samples": len(self._outcomes),
            "required_samples": required_n,
            "remaining": remaining,
            "observed_std": round(observed_std, 4),
            "target_improvement": target_improvement,
            "alpha": alpha,
            "power": power,
            "progress_pct": round(min(100, len(self._outcomes) / required_n * 100), 1),
        }

    def _bootstrap_ci(
        self,
        data: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for the mean."""
        rng = np.random.RandomState(42)
        means = np.array([
            float(rng.choice(data, size=len(data), replace=True).mean())
            for _ in range(n_bootstrap)
        ])
        alpha = (1 - ci) / 2
        return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))

    # ── Report Generation ───────────────────────────────────────────────

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report.

        Returns structured report with all metrics, analysis, and a promotion verdict.
        """
        report: Dict[str, Any] = {
            "generated_at": time.time(),
            "data_summary": {
                "total_predictions": len(self._predictions),
                "total_outcomes": len(self._outcomes),
                "total_transitions": len(self._transitions),
                "total_meta_trains": len(self._meta_trains),
            },
        }

        # Not enough data for analysis
        if len(self._outcomes) < self.config.min_outcomes_for_report:
            report["status"] = "collecting_data"
            report["outcomes_needed"] = self.config.min_outcomes_for_report - len(self._outcomes)
            return report

        report["status"] = "ready"

        # 1. Overall MAE comparison
        maml_mae = self.compute_mae("maml")
        ridge_mae = self.compute_mae("ridge")
        report["mae_comparison"] = {
            "maml_mae": maml_mae,
            "ridge_mae": ridge_mae,
            "advantage": round(ridge_mae - maml_mae, 4) if maml_mae is not None and ridge_mae is not None else None,
            "maml_wins": maml_mae < ridge_mae if maml_mae is not None and ridge_mae is not None else None,
        }

        # 2. Per-regime MAE
        report["regime_mae"] = {
            "maml": self.compute_regime_mae("maml"),
            "ridge": self.compute_regime_mae("ridge"),
        }

        # 3. Adaptation speed
        report["adaptation_speed"] = self.compute_adaptation_speed()

        # 4. Regime divergence
        report["regime_divergence"] = self.compute_regime_divergence()

        # 5. Meta-train curve
        report["meta_train_curve"] = self.compute_meta_train_curve()

        # 6. Calibration
        report["calibration"] = self.compute_calibration()

        # 7. Win rate by model confidence
        report["win_rate_analysis"] = self.compute_win_rate_by_model()

        # 8. Rolling MAE (last 10, 25, 50 outcomes)
        report["rolling_mae"] = {
            "maml": {
                "last_10": self.compute_mae("maml", last_n=10),
                "last_25": self.compute_mae("maml", last_n=25),
                "last_50": self.compute_mae("maml", last_n=50),
                "all": maml_mae,
            },
            "ridge": {
                "last_10": self.compute_mae("ridge", last_n=10),
                "last_25": self.compute_mae("ridge", last_n=25),
                "last_50": self.compute_mae("ridge", last_n=50),
                "all": ridge_mae,
            },
        }

        # 9. Statistical significance
        report["significance_test"] = self.compute_significance_test()

        # 10. Calibration metrics (ECE, MCE, Brier)
        report["calibration_metrics"] = self.compute_calibration_metrics()

        # 11. Error time series analysis
        report["error_time_series"] = self.compute_error_time_series()

        # 12. Inner loop diagnostics
        report["inner_loop_diagnostics"] = self.compute_inner_loop_diagnostics()

        # 13. Parameter drift analysis
        report["param_drift"] = self.compute_param_drift_analysis()

        # 14. Exit reason analysis
        report["exit_reason_analysis"] = self.compute_exit_reason_analysis()

        # 15. Sample size requirement
        report["sample_size"] = self.compute_required_sample_size()

        # 16. Promotion verdict (uses all above)
        report["promotion_verdict"] = self._compute_promotion_verdict(report)

        return report

    def generate_summary(self) -> str:
        """Generate human-readable summary string."""
        report = self.generate_report()

        if report.get("status") == "collecting_data":
            needed = report.get("outcomes_needed", "?")
            total = len(self._outcomes)
            return (
                f"MAML Benchmark: Collecting data ({total} outcomes, need {needed} more)\n"
                f"Predictions recorded: {len(self._predictions)}"
            )

        lines = [
            "=" * 60,
            "MAML Ridge Benchmark Report",
            "=" * 60,
            "",
            "--- MAE Comparison (lower is better) ---",
        ]

        mae = report.get("mae_comparison", {})
        lines.append(f"  MAML MAE:  {mae.get('maml_mae', 'N/A')}")
        lines.append(f"  Ridge MAE: {mae.get('ridge_mae', 'N/A')}")
        advantage = mae.get("advantage")
        if advantage is not None:
            winner = "MAML" if advantage > 0 else "Ridge"
            lines.append(f"  Advantage: {winner} by {abs(advantage):.4f}")

        lines.append("")
        lines.append("--- Regime MAE ---")
        for regime in _REGIME_NAMES:
            m_mae = report.get("regime_mae", {}).get("maml", {}).get(regime)
            r_mae = report.get("regime_mae", {}).get("ridge", {}).get(regime)
            m_str = f"{m_mae:.4f}" if m_mae is not None else "N/A"
            r_str = f"{r_mae:.4f}" if r_mae is not None else "N/A"
            lines.append(f"  {regime:8s}  MAML={m_str}  Ridge={r_str}")

        lines.append("")
        lines.append("--- Adaptation Speed ---")
        adapt = report.get("adaptation_speed", {})
        lines.append(f"  Transitions: {adapt.get('transitions', 0)}")
        lines.append(f"  Converged: {adapt.get('converged', 0)}")
        avg_conv = adapt.get("avg_trades_to_converge")
        lines.append(f"  Avg trades to converge: {avg_conv if avg_conv else 'N/A'}")

        lines.append("")
        lines.append("--- Regime Divergence ---")
        div = report.get("regime_divergence", {})
        lines.append(f"  Divergence score: {div.get('divergence_score', 'N/A')}")
        lines.append(f"  Verdict: {div.get('verdict', 'N/A')}")

        lines.append("")
        lines.append("--- Meta-Train Curve ---")
        curve = report.get("meta_train_curve", {})
        lines.append(f"  Total trains: {curve.get('total_trains', 0)}")
        lines.append(f"  Trend: {curve.get('trend', 'N/A')}")
        improvement = curve.get("improvement_pct")
        lines.append(f"  Improvement: {improvement:.1f}%" if improvement is not None else "  Improvement: N/A")

        lines.append("")
        lines.append("--- Statistical Significance ---")
        sig = report.get("significance_test", {})
        if sig.get("status") == "ok":
            lines.append(f"  Paired t-test p-value: {sig.get('t_pvalue', 'N/A')}")
            lines.append(f"  Wilcoxon p-value:      {sig.get('wilcoxon_pvalue', 'N/A')}")
            lines.append(f"  Cohen's d:             {sig.get('cohens_d', 'N/A')} ({sig.get('effect_size', '')})")
            lines.append(f"  95% CI:                {sig.get('ci_95', 'N/A')}")
            lines.append(f"  Interpretation:        {sig.get('interpretation', 'N/A')}")
        else:
            lines.append(f"  Status: {sig.get('status', 'N/A')}")

        lines.append("")
        lines.append("--- Calibration (ECE/MCE/Brier) ---")
        cal = report.get("calibration_metrics", {})
        if cal.get("status") == "ok":
            mc = cal.get("maml", {})
            rc = cal.get("ridge", {})
            lines.append(f"  MAML:  Brier={mc.get('brier_score','N/A')}  ECE={mc.get('ece','N/A')}  MCE={mc.get('mce','N/A')}")
            lines.append(f"  Ridge: Brier={rc.get('brier_score','N/A')}  ECE={rc.get('ece','N/A')}  MCE={rc.get('mce','N/A')}")

        lines.append("")
        lines.append("--- Error Trend ---")
        ets = report.get("error_time_series", {})
        if ets.get("status") == "ok":
            mt = ets.get("maml", {})
            rt = ets.get("ridge", {})
            lines.append(f"  MAML:  trend={mt.get('trend','N/A')}  velocity={mt.get('velocity_pct','N/A')}%  autocorr={mt.get('autocorrelation_lag1','N/A')}")
            lines.append(f"  Ridge: trend={rt.get('trend','N/A')}  velocity={rt.get('velocity_pct','N/A')}%  autocorr={rt.get('autocorrelation_lag1','N/A')}")

        lines.append("")
        lines.append("--- Inner Loop Diagnostics ---")
        ild = report.get("inner_loop_diagnostics", {})
        if ild.get("status") == "ok":
            for regime, summary in ild.get("regime_summary", {}).items():
                healthy = "HEALTHY" if summary.get("healthy") else "UNSTABLE"
                lines.append(
                    f"  {regime:8s}  reduction={summary.get('avg_reduction_pct',0):.1f}%  "
                    f"monotonic={summary.get('monotonic_pct',0):.0f}%  [{healthy}]"
                )
        else:
            lines.append(f"  Status: {ild.get('status', 'N/A')}")

        lines.append("")
        lines.append("--- Parameter Drift ---")
        pd_data = report.get("param_drift", {})
        if pd_data.get("status") == "ok":
            lines.append(f"  Trend: {pd_data.get('trend', 'N/A')}")
            lines.append(f"  Mean drift: {pd_data.get('mean_drift', 'N/A')}")
            lines.append(f"  Latest drift: {pd_data.get('latest_drift', 'N/A')}")

        lines.append("")
        lines.append("--- Sample Size ---")
        ss = report.get("sample_size", {})
        if ss.get("status") == "ok":
            lines.append(f"  Progress: {ss.get('current_samples',0)}/{ss.get('required_samples',0)} ({ss.get('progress_pct',0):.1f}%)")
            lines.append(f"  Remaining: {ss.get('remaining',0)} trades needed")

        lines.append("")
        lines.append("--- Promotion Verdict ---")
        verdict = report.get("promotion_verdict", {})
        lines.append(f"  Decision: {verdict.get('decision', 'N/A')}")
        lines.append(f"  Confidence: {verdict.get('confidence', 'N/A')}")
        lines.append(f"  Score: {verdict.get('score', 'N/A')}")
        reasons = verdict.get("reasons", [])
        for r in reasons:
            lines.append(f"    - {r}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ── Internal ────────────────────────────────────────────────────────

    def _filter_outcomes(
        self,
        regime: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> List[OutcomeRecord]:
        """Filter outcomes by regime and/or recency."""
        outcomes = self._outcomes
        if regime:
            regime = regime.upper()
            outcomes = [o for o in outcomes if o.regime == regime]
        if last_n and last_n < len(outcomes):
            outcomes = outcomes[-last_n:]
        return outcomes

    def _update_convergence(self, outcome: OutcomeRecord) -> None:
        """Check if the most recent regime transition has converged."""
        if not self._transitions:
            return

        latest = self._transitions[-1]
        if latest.converged_after is not None:
            return  # Already converged

        latest.trades_since += 1

        # Need at least convergence_window trades
        if latest.trades_since < self.config.convergence_window:
            return

        # Get recent MAML errors in the new regime
        recent = [
            o for o in self._outcomes[-self.config.convergence_window:]
            if o.regime == latest.new_regime
        ]

        if len(recent) < self.config.convergence_window // 2:
            return  # Not enough regime-specific data

        mae = float(np.mean([abs(o.maml_confidence - o.realized_confidence) for o in recent]))

        if mae <= self.config.convergence_threshold:
            latest.converged_after = latest.trades_since
            logger.info(
                "MAML Benchmark: converged after %d trades (MAE=%.2f) for %s→%s transition",
                latest.trades_since, mae, latest.old_regime, latest.new_regime,
            )

    def _compute_promotion_verdict(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Determine whether MAML should be promoted from shadow to production.

        Criteria:
        1. MAML MAE must be lower than Ridge MAE by at least `promotion_mae_advantage`
        2. Meta-train curve must be improving or flat (not degrading)
        3. Regime divergence should be at least moderate
        4. Sufficient data (outcomes, transitions, meta-trains)
        """
        reasons: List[str] = []
        score = 0  # Points toward promotion

        # Check MAE advantage
        mae_comp = report.get("mae_comparison", {})
        advantage = mae_comp.get("advantage")
        if advantage is not None:
            if advantage >= self.config.promotion_mae_advantage:
                reasons.append(f"MAML MAE advantage: {advantage:.4f} (threshold: {self.config.promotion_mae_advantage})")
                score += 2
            elif advantage > 0:
                reasons.append(f"MAML MAE slightly better: {advantage:.4f} (need {self.config.promotion_mae_advantage})")
                score += 1
            else:
                reasons.append(f"Ridge MAE is better by {abs(advantage):.4f}")
                score -= 2

        # Check meta-train trend
        curve = report.get("meta_train_curve", {})
        trend = curve.get("trend")
        if trend == "improving":
            reasons.append("Meta-train loss improving")
            score += 1
        elif trend == "degrading":
            reasons.append("Meta-train loss degrading — learning is unstable")
            score -= 2
        elif trend == "flat":
            reasons.append("Meta-train loss flat — may have converged")
            score += 0

        # Check regime divergence
        div = report.get("regime_divergence", {})
        div_verdict = div.get("verdict")
        if div_verdict == "strong_divergence":
            reasons.append("Strong regime divergence — MAML is specializing")
            score += 2
        elif div_verdict == "moderate_divergence":
            reasons.append("Moderate regime divergence")
            score += 1
        elif div_verdict == "no_divergence":
            reasons.append("No regime divergence — MAML is not adapting to regimes")
            score -= 1

        # Check data sufficiency
        data = report.get("data_summary", {})
        if data.get("total_outcomes", 0) < 50:
            reasons.append(f"Only {data.get('total_outcomes', 0)} outcomes — need at least 50 for high confidence")
            score -= 1
        if data.get("total_meta_trains", 0) < 3:
            reasons.append(f"Only {data.get('total_meta_trains', 0)} meta-trains — need more training rounds")
            score -= 1

        # Adaptation speed
        adapt = report.get("adaptation_speed", {})
        if adapt.get("converged", 0) > 0:
            avg_conv = adapt.get("avg_trades_to_converge")
            if avg_conv is not None and avg_conv <= 15:
                reasons.append(f"Fast convergence: avg {avg_conv:.1f} trades after regime transition")
                score += 1

        # Statistical significance (HARD GATE)
        sig = report.get("significance_test", {})
        if sig.get("status") == "ok":
            if sig.get("significant_at_005") and sig.get("maml_better"):
                reasons.append(f"Statistically significant: p={sig['t_pvalue']:.4f}, d={sig['cohens_d']:.2f}")
                score += 2
            elif sig.get("significant_at_005") and not sig.get("maml_better"):
                reasons.append(f"Ridge is statistically significantly better: p={sig['t_pvalue']:.4f}")
                score -= 3  # Hard penalty
            elif not sig.get("significant_at_005"):
                reasons.append(f"Not statistically significant: p={sig['t_pvalue']:.4f} (need <0.05)")
                score -= 1

        # Calibration quality
        calib = report.get("calibration_metrics", {})
        if calib.get("status") == "ok":
            maml_ece = calib.get("maml", {}).get("ece", 1.0)
            ridge_ece = calib.get("ridge", {}).get("ece", 1.0)
            if maml_ece < ridge_ece:
                reasons.append(f"MAML better calibrated: ECE={maml_ece:.4f} vs Ridge ECE={ridge_ece:.4f}")
                score += 1
            elif maml_ece > 0.15:
                reasons.append(f"MAML poorly calibrated: ECE={maml_ece:.4f} (>0.15)")
                score -= 1

        # Error trend
        ets = report.get("error_time_series", {})
        maml_trend = ets.get("maml", {}).get("trend")
        if maml_trend == "improving":
            reasons.append("MAML error trend improving over time")
            score += 1
        elif maml_trend == "degrading":
            reasons.append("MAML error trend degrading — model may be overfitting")
            score -= 1

        # Decision
        if score >= 4:
            decision = "PROMOTE"
            confidence = "high"
        elif score >= 2:
            decision = "LIKELY_PROMOTE"
            confidence = "moderate"
        elif score >= 0:
            decision = "CONTINUE_SHADOW"
            confidence = "low"
        else:
            decision = "DO_NOT_PROMOTE"
            confidence = "high"

        return {
            "decision": decision,
            "confidence": confidence,
            "score": score,
            "reasons": reasons,
        }

    # ── Persistence ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist benchmark data to disk."""
        try:
            _BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

            state = {
                "version": 1,
                "saved_at": time.time(),
                "current_regime": self._current_regime,
                "trades_since_transition": self._trades_since_transition,
                "scan_cycle": self._scan_cycle,
                "predictions": [p.to_dict() for p in self._predictions[-self.config.max_predictions:]],
                "outcomes": [o.to_dict() for o in self._outcomes[-self.config.max_outcomes:]],
                "transitions": [t.to_dict() for t in self._transitions],
                "meta_trains": [m.to_dict() for m in self._meta_trains],
            }

            path = _BENCHMARK_DIR / "benchmark_state.json"
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
            os.replace(str(tmp_path), str(path))

            logger.debug(
                "MAML Benchmark: saved (outcomes=%d, predictions=%d)",
                len(self._outcomes), len(self._predictions),
            )
        except Exception as e:
            logger.warning("MAML Benchmark: save failed: %s", e)

    def _load_state(self) -> None:
        """Load persisted benchmark data."""
        path = _BENCHMARK_DIR / "benchmark_state.json"
        if not path.exists():
            return

        try:
            state = json.loads(path.read_text())

            if state.get("version", 0) != 1:
                logger.warning("MAML Benchmark: unknown state version %s — starting fresh", state.get("version"))
                return

            self._current_regime = state.get("current_regime", "NORMAL")
            self._trades_since_transition = state.get("trades_since_transition", 0)
            self._scan_cycle = state.get("scan_cycle", 0)

            self._predictions = [
                PredictionRecord.from_dict(d) for d in state.get("predictions", [])
            ]
            self._outcomes = [
                OutcomeRecord.from_dict(d) for d in state.get("outcomes", [])
            ]
            self._transitions = [
                RegimeTransition.from_dict(d) for d in state.get("transitions", [])
            ]
            self._meta_trains = [
                MetaTrainRecord.from_dict(d) for d in state.get("meta_trains", [])
            ]

            logger.info(
                "MAML Benchmark: loaded state (predictions=%d, outcomes=%d, transitions=%d, trains=%d)",
                len(self._predictions), len(self._outcomes),
                len(self._transitions), len(self._meta_trains),
            )
        except Exception as e:
            logger.warning("MAML Benchmark: load failed: %s — starting fresh", e)

    def force_save(self) -> None:
        """Force an immediate save (for shutdown hooks)."""
        self._save_state()
