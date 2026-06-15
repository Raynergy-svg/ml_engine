"""
Soak test orchestrator.

Runs the SOTA model in shadow mode alongside the legacy ensemble for N days
(or N backtest bars) and accumulates comparison statistics.

No trading decisions are made from SOTA signals during soak — it runs in
parallel, logging predictions and outcomes, until the comparison harness
has enough data to make a recommendation.

Typical soak: 500–2,000 bars (2–8 weeks of M15 data) for statistical significance.
"""

from __future__ import annotations

import json

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .model_comparison import BarPrediction, ModelComparisonResult, compare_models_on_history
from .error_correlation import ErrorCorrelation, analyze_error_correlation
from .meta_ensemble import MetaEnsemble, compute_meta_weights

logger = logging.getLogger(__name__)


@dataclass
class SoakResult:
    """Outcome of a soak test."""
    pair: str
    start_time: str
    end_time: str
    n_bars: int

    comparison: Optional[ModelComparisonResult] = None
    correlation: Optional[ErrorCorrelation] = None
    meta_ensemble: Optional[MetaEnsemble] = None

    recommendation: str = "UNDECIDED"  # STACK / REPLACE / DISCARD / UNDECIDED
    confidence: float = 0.0

    # Safety: minimum bars before making a recommendation
    MIN_BARS_FOR_DECISION: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "n_bars": self.n_bars,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 4),
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "correlation": self.correlation.to_dict() if self.correlation else None,
        }


def run_soak_comparison(
    pair: str,
    df: pd.DataFrame,
    legacy_signal_fn: Callable[[pd.DataFrame], BarPrediction],
    sota_signal_fn: Callable[[pd.DataFrame], BarPrediction],
    regime_labels: Optional[List[str]] = None,
    warmup_bars: int = 128,
) -> SoakResult:
    """Run a full soak: compare → correlate → meta-weight → recommend.

    This is the ONE function that answers "should we use this model?"
    """
    result = SoakResult(
        pair=pair,
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time="",
        n_bars=len(df),
    )

    if len(df) < result.MIN_BARS_FOR_DECISION:
        logger.warning(
            f"{pair}: soak has only {len(df)} bars; need {result.MIN_BARS_FOR_DECISION} "
            f"for statistical significance."
        )

    # Step 1: Parallel backtest
    comparison = compare_models_on_history(
        pair=pair,
        df=df,
        legacy_signal_fn=legacy_signal_fn,
        sota_signal_fn=sota_signal_fn,
        warmup_bars=warmup_bars,
    )
    result.comparison = comparison

    # Step 2: Error correlation analysis
    correlation = analyze_error_correlation(comparison, regime_labels=regime_labels)
    result.correlation = correlation
    result.recommendation = correlation.recommendation
    result.confidence = correlation.confidence

    # Step 3: Meta-ensemble weights (only if STACK recommendation)
    if result.recommendation == "STACK":
        leg_errors = np.array([
            1.0 if p.direction == "LONG" and a > 0 else
            (1.0 if p.direction == "SHORT" and a < 0 else 0.0)
            for p, a in zip(comparison.legacy_predictions, comparison.actual_returns)
        ])
        sota_errors = np.array([
            1.0 if p.direction == "LONG" and a > 0 else
            (1.0 if p.direction == "SHORT" and a < 0 else 0.0)
            for p, a in zip(comparison.sota_predictions, comparison.actual_returns)
        ])
        w_leg, w_sota = compute_meta_weights(leg_errors, sota_errors)
        meta = MetaEnsemble(pair=pair, legacy_weight=w_leg, sota_weight=w_sota)
        if regime_labels and correlation.regime_slices:
            for slice_ in correlation.regime_slices:
                # Per-regime weights: higher weight to model that wins in that regime
                if slice_.legacy_accuracy > slice_.sota_accuracy:
                    meta.regime_weights[slice_.regime] = (0.7, 0.3)
                elif slice_.sota_accuracy > slice_.legacy_accuracy:
                    meta.regime_weights[slice_.regime] = (0.3, 0.7)
                else:
                    meta.regime_weights[slice_.regime] = (0.5, 0.5)
        result.meta_ensemble = meta

    result.end_time = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"SOAK COMPLETE {pair}: recommendation={result.recommendation} "
        f"confidence={result.confidence:.2f} bars={result.n_bars}"
    )
    return result


def save_soak_report(result: SoakResult, path: str) -> None:
    """Persist soak report to JSON for operator review."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Soak report saved to {path}")


def load_soak_report(path: str) -> SoakResult:
    with open(path) as f:
        data = json.load(f)
    return SoakResult(
        pair=data["pair"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        n_bars=data["n_bars"],
        recommendation=data["recommendation"],
        confidence=data["confidence"],
    )
