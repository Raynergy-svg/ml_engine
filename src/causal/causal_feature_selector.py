"""Causal feature selection for factor models.

US-015: Selects features based on causal parents from discovered graph,
removing spuriously correlated factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CausalFeatureSelectorConfig:
    """Configuration for causal feature selection."""

    target_index: int = 0  # Index of target factor (returns)
    max_features: int = 10
    fallback_to_correlation: bool = True
    min_causal_confidence: float = 0.3


class CausalFeatureSelector:
    """Select features based on causal parents of target.

    Uses a discovered causal graph to identify genuine causal parents
    of the target variable. Falls back to correlation-based selection
    when causal graph is unavailable.
    """

    def __init__(self, config: Optional[CausalFeatureSelectorConfig] = None):
        self.cfg = config or CausalFeatureSelectorConfig()

    def select(
        self,
        X: np.ndarray,
        causal_graph: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Select feature indices.

        Args:
            X: Factor returns matrix (n_samples, n_factors)
            causal_graph: Optional directed adjacency matrix
            confidences: Optional edge confidence matrix

        Returns:
            List of selected feature indices (excluding target)
        """
        n_factors = X.shape[1]
        target = self.cfg.target_index

        if causal_graph is not None:
            # Select causal parents of target with sufficient confidence
            parents = []
            for i in range(n_factors):
                if i == target:
                    continue
                if causal_graph[i, target] == 1:
                    if confidences is not None and confidences[i, target] >= self.cfg.min_causal_confidence:
                        parents.append(i)
                    elif confidences is None:
                        parents.append(i)

            if len(parents) > 0:
                # If too many parents, select top by confidence
                if confidences is not None and len(parents) > self.cfg.max_features:
                    parents = sorted(parents, key=lambda i: confidences[i, target], reverse=True)
                    parents = parents[: self.cfg.max_features]
                logger.info("CausalFeatureSelector: selected %d causal parents", len(parents))
                return parents

        # Fallback: correlation-based selection
        if self.cfg.fallback_to_correlation:
            corrs = []
            for i in range(n_factors):
                if i == target:
                    continue
                c = np.abs(np.corrcoef(X[:, i], X[:, target])[0, 1])
                corrs.append((i, 0.0 if np.isnan(c) else c))
            corrs.sort(key=lambda x: x[1], reverse=True)
            selected = [i for i, _ in corrs[: self.cfg.max_features]]
            logger.info("CausalFeatureSelector: fell back to correlation, selected %d features", len(selected))
            return selected

        return []
