"""Causal discovery using PC-algorithm skeleton.

US-013: Identifies directed causal relationships between market factors.
"""

from __future__ import annotations

import itertools
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CausalDiscoveryConfig:
    """Configuration for causal discovery."""

    max_factors: int = 20
    alpha: float = 0.05  # significance level for CI tests
    max_cond_set_size: int = 3
    min_edge_confidence: float = 0.5


class CausalDiscovery:
    """PC-algorithm skeleton for causal graph discovery from observational data.

    Uses conditional independence testing (partial correlation) to identify
    the underlying causal structure among market factors.
    """

    def __init__(self, config: Optional[CausalDiscoveryConfig] = None):
        self.cfg = config or CausalDiscoveryConfig()
        self._adjacency: Optional[np.ndarray] = None
        self._confidences: Optional[np.ndarray] = None
        self._factor_names: List[str] = []

    def fit(self, X: np.ndarray, factor_names: Optional[List[str]] = None) -> "CausalDiscovery":
        """Run PC-algorithm skeleton on factor returns.

        Args:
            X: Array of shape (n_samples, n_factors)
            factor_names: Optional list of factor names.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.shape}")
        n_samples, n_factors = X.shape
        if n_factors > self.cfg.max_factors:
            raise ValueError(f"Too many factors: {n_factors} > {self.cfg.max_factors}")

        self._factor_names = factor_names or [f"F{i}" for i in range(n_factors)]

        # Start with complete graph
        adjacency = np.ones((n_factors, n_factors), dtype=int) - np.eye(n_factors, dtype=int)
        confidences = np.ones((n_factors, n_factors), dtype=float)

        # Phase 1: Remove edges based on conditional independence
        for size in range(self.cfg.max_cond_set_size + 1):
            for i, j in itertools.combinations(range(n_factors), 2):
                if adjacency[i, j] == 0:
                    continue
                neighbors = [k for k in range(n_factors) if adjacency[i, k] == 1 and k != j]
                if len(neighbors) < size:
                    continue
                for cond_set in itertools.combinations(neighbors, size):
                    pcorr = self._partial_correlation(X, i, j, list(cond_set))
                    stat = np.sqrt(n_samples - size - 3) * np.abs(pcorr)
                    pval = 2 * (1 - 0.5 * (1 + math.erf(stat / math.sqrt(2))))
                    if pval > self.cfg.alpha:
                        adjacency[i, j] = adjacency[j, i] = 0
                        confidences[i, j] = confidences[j, i] = 1 - pval
                        break

        # Phase 2: Orient v-structures (simplified)
        adjacency = self._orient_v_structures(X, adjacency)

        self._adjacency = adjacency
        self._confidences = confidences
        logger.info("CausalDiscovery: found %d edges", adjacency.sum() // 2)
        return self

    def _partial_correlation(self, X: np.ndarray, i: int, j: int, cond_set: List[int]) -> float:
        """Compute partial correlation of X[:,i] and X[:,j] given cond_set."""
        if not cond_set:
            corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)

        # Linear regression residuals
        from numpy.linalg import lstsq

        Z = X[:, cond_set]
        xi = X[:, i]
        xj = X[:, j]

        # Regress i on Z
        beta_i, *_ = lstsq(Z, xi, rcond=None)
        resid_i = xi - Z @ beta_i

        # Regress j on Z
        beta_j, *_ = lstsq(Z, xj, rcond=None)
        resid_j = xj - Z @ beta_j

        corr = np.corrcoef(resid_i, resid_j)[0, 1]
        return 0.0 if np.isnan(corr) else float(corr)


    def _orient_v_structures(self, X: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Simplified v-structure orientation."""
        n = adjacency.shape[0]
        oriented = adjacency.copy()
        for i, j, k in itertools.permutations(range(n), 3):
            if adjacency[i, j] and adjacency[k, j] and not adjacency[i, k]:
                # Potential v-structure i -> j <- k
                if oriented[j, i] == 1 and oriented[j, k] == 1:
                    oriented[i, j] = 1
                    oriented[j, i] = 0
                    oriented[k, j] = 1
                    oriented[j, k] = 0
        return oriented

    def get_adjacency(self) -> np.ndarray:
        if self._adjacency is None:
            raise RuntimeError("Model not fitted yet.")
        return self._adjacency.copy()

    def get_confidences(self) -> np.ndarray:
        if self._confidences is None:
            raise RuntimeError("Model not fitted yet.")
        return self._confidences.copy()

    def get_edges(self) -> List[Dict[str, Any]]:
        """Return list of edges with confidence scores."""
        if self._adjacency is None:
            raise RuntimeError("Model not fitted yet.")
        edges = []
        n = self._adjacency.shape[0]
        for i in range(n):
            for j in range(n):
                if self._adjacency[i, j] == 1 and self._confidences is not None:
                    edges.append({
                        "from": self._factor_names[i],
                        "to": self._factor_names[j],
                        "confidence": float(self._confidences[i, j]),
                    })
        return edges
