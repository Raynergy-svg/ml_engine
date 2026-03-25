"""Feature Importance Analysis — Permutation-based feature ranking per pair.

Identifies which features matter most for each pair's prediction accuracy
and which are noise. Results guide feature pruning during retraining.

Key outputs:
  - Per-pair feature importance rankings
  - Top 5 / bottom 5 features per pair
  - Noise features (importance < 0.01) flagged for pruning
  - Critical features (importance > 0.1) flagged for preservation
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceResult:
    """Feature importance result for a single pair."""

    pair: str
    n_features: int
    n_samples: int
    top_features: List[Tuple[str, float]]     # [(name, importance), ...] sorted desc
    bottom_features: List[Tuple[str, float]]   # [(name, importance), ...] sorted asc
    noise_features: List[str]     # importance < 0.01
    critical_features: List[str]  # importance > 0.1
    timestamp: str = ""


class FeatureAnalyzer:
    """Analyzes feature importance using permutation importance.

    Permutation importance is model-agnostic: shuffles each feature column
    and measures accuracy drop. Large drop = important feature.

    Usage:
        analyzer = FeatureAnalyzer()
        result = analyzer.analyze(df, pair="EUR_USD", predict_fn=model.predict)
    """

    def __init__(
        self,
        results_path: str = "trained_data/models/feature_importance.json",
        n_repeats: int = 5,
        noise_threshold: float = 0.01,
        critical_threshold: float = 0.10,
    ):
        self.results_path = Path(results_path)
        self.n_repeats = n_repeats
        self.noise_threshold = noise_threshold
        self.critical_threshold = critical_threshold
        self._results: Dict[str, FeatureImportanceResult] = {}
        self._load_results()

    def analyze(
        self,
        df: pd.DataFrame,
        pair: str,
        predict_fn: Callable,
        y_true: Optional[np.ndarray] = None,
        lookahead: int = 5,
    ) -> FeatureImportanceResult:
        """Run permutation importance analysis for a pair.

        Args:
            df: Feature DataFrame (columns = feature names)
            pair: Currency pair
            predict_fn: Function(features_df) -> predictions array
            y_true: True labels. If None, computed from close price + lookahead.
            lookahead: Candles ahead for direction labeling

        Returns:
            FeatureImportanceResult with rankings
        """
        feature_names = list(df.columns)
        n_features = len(feature_names)

        # Generate true labels if not provided
        if y_true is None:
            if "close" in df.columns:
                close = df["close"].values
                y_true = np.array([
                    1 if close[min(i + lookahead, len(close) - 1)] > close[i] else 0
                    for i in range(len(close))
                ])
            else:
                logger.warning("No 'close' column and no y_true provided — cannot compute importance")
                return FeatureImportanceResult(
                    pair=pair,
                    n_features=n_features,
                    n_samples=len(df),
                    top_features=[],
                    bottom_features=[],
                    noise_features=[],
                    critical_features=[],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

        # Trim to valid range (exclude last lookahead rows)
        valid_len = len(df) - lookahead
        if valid_len < 50:
            logger.warning(f"Insufficient samples for feature analysis: {valid_len}")
            return FeatureImportanceResult(
                pair=pair, n_features=n_features, n_samples=valid_len,
                top_features=[], bottom_features=[], noise_features=[], critical_features=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        X = df.iloc[:valid_len].copy()
        y = y_true[:valid_len]

        # Baseline accuracy
        baseline_acc = self._compute_accuracy(X, y, predict_fn)
        if baseline_acc is None:
            return FeatureImportanceResult(
                pair=pair, n_features=n_features, n_samples=valid_len,
                top_features=[], bottom_features=[], noise_features=[], critical_features=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Permutation importance for each feature
        importances = {}
        for feat_idx, feat_name in enumerate(feature_names):
            drops = []
            for _ in range(self.n_repeats):
                X_permuted = X.copy()
                X_permuted.iloc[:, feat_idx] = np.random.permutation(
                    X_permuted.iloc[:, feat_idx].values
                )
                permuted_acc = self._compute_accuracy(X_permuted, y, predict_fn)
                if permuted_acc is not None:
                    drops.append(baseline_acc - permuted_acc)

            if drops:
                importances[feat_name] = float(np.mean(drops))

        # Sort by importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_features[:5]
        bottom_5 = sorted_features[-5:]

        noise = [name for name, imp in sorted_features if imp < self.noise_threshold]
        critical = [name for name, imp in sorted_features if imp > self.critical_threshold]

        result = FeatureImportanceResult(
            pair=pair,
            n_features=n_features,
            n_samples=valid_len,
            top_features=top_5,
            bottom_features=bottom_5,
            noise_features=noise,
            critical_features=critical,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._results[pair] = result
        self._save_results()

        logger.info(
            f"Feature analysis {pair}: {n_features} features, "
            f"{len(critical)} critical, {len(noise)} noise. "
            f"Top: {', '.join(n for n, _ in top_5[:3])}"
        )

        return result

    def _compute_accuracy(
        self, X: pd.DataFrame, y: np.ndarray, predict_fn: Callable
    ) -> Optional[float]:
        """Compute prediction accuracy."""
        try:
            predictions = predict_fn(X)
            if predictions is None:
                return None

            # Handle various prediction formats
            if isinstance(predictions, (list, np.ndarray)):
                pred_array = np.array(predictions).flatten()
            else:
                return None

            # Convert direction strings to binary if needed
            if len(pred_array) > 0 and isinstance(pred_array[0], str):
                pred_binary = np.array([1 if str(p).upper() == "LONG" else 0 for p in pred_array])
            elif len(pred_array) > 0 and pred_array.dtype in (np.float32, np.float64):
                pred_binary = (pred_array > 0.5).astype(int)
            else:
                pred_binary = pred_array.astype(int)

            min_len = min(len(pred_binary), len(y))
            if min_len == 0:
                return None

            correct = np.sum(pred_binary[:min_len] == y[:min_len])
            return float(correct / min_len)

        except Exception as e:
            logger.debug(f"Accuracy computation failed: {e}")
            return None

    def _load_results(self) -> None:
        """Load previous analysis results."""
        try:
            if self.results_path.exists():
                data = json.loads(self.results_path.read_text())
                for pair, rec in data.items():
                    # Convert tuples back from lists
                    rec["top_features"] = [tuple(x) for x in rec.get("top_features", [])]
                    rec["bottom_features"] = [tuple(x) for x in rec.get("bottom_features", [])]
                    self._results[pair] = FeatureImportanceResult(**rec)
        except Exception as e:
            logger.debug(f"Feature importance load failed: {e}")

    def _save_results(self) -> None:
        """Persist analysis results."""
        try:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for pair, result in self._results.items():
                d = asdict(result)
                # Convert tuples to lists for JSON serialization
                d["top_features"] = [list(x) for x in d["top_features"]]
                d["bottom_features"] = [list(x) for x in d["bottom_features"]]
                data[pair] = d
            self.results_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Feature importance save failed: {e}")

    def get_summary(self) -> str:
        """Get human-readable summary of feature importance analysis."""
        if not self._results:
            return "No feature importance analysis results yet."

        lines = ["Feature Importance Summary", "=" * 80]
        for pair, result in sorted(self._results.items()):
            lines.append(f"\n{pair} ({result.n_features} features, {result.n_samples} samples)")
            lines.append("-" * 40)
            if result.top_features:
                lines.append("  Top features:")
                for name, imp in result.top_features:
                    lines.append(f"    {name:30s} importance={imp:.4f}")
            if result.noise_features:
                lines.append(f"  Noise features ({len(result.noise_features)}): "
                             f"{', '.join(result.noise_features[:5])}")
            if result.critical_features:
                lines.append(f"  Critical features ({len(result.critical_features)}): "
                             f"{', '.join(result.critical_features)}")

        return "\n".join(lines)

    def get_prune_list(self, pair: str) -> List[str]:
        """Get list of features to prune for a pair (noise features).

        Args:
            pair: Currency pair

        Returns:
            List of feature names with importance < noise_threshold
        """
        if pair in self._results:
            return self._results[pair].noise_features
        return []
