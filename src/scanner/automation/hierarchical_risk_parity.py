"""Hierarchical Risk Parity (HRP) portfolio optimizer.

US-073: Replace binary correlation blocking with continuous HRP-based position
weighting. Uses scipy hierarchical clustering on correlation matrix, recursive
bisection for weight assignment, and Ledoit-Wolf shrinkage for small samples.

References:
    - López de Prado, "Building Diversified Portfolios that Outperform
      Out-of-Sample", Journal of Portfolio Management, 2016
    - PyPortfolioOpt HRP implementation

Usage:
    optimizer = HRPOptimizer()
    weights = optimizer.get_weights(pairs, returns_data)
    # weights = {"EUR_USD": 0.15, "GBP_USD": 0.12, ...}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum observations for reliable correlation estimation
MIN_OBSERVATIONS = 20

# Weight bounds to prevent extreme concentration
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.99


@dataclass
class HRPResult:
    """Result of HRP weight computation."""

    weights: Dict[str, float] = field(default_factory=dict)
    n_assets: int = 0
    n_samples: int = 0
    linkage_method: str = "average"
    effective_n: float = 0.0  # Herfindahl diversification index
    shrinkage_intensity: float = 0.0
    ordered_pairs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "n_assets": self.n_assets,
            "n_samples": self.n_samples,
            "linkage_method": self.linkage_method,
            "effective_n": round(self.effective_n, 2),
            "shrinkage_intensity": round(self.shrinkage_intensity, 4),
            "ordered_pairs": self.ordered_pairs,
        }


class HRPOptimizer:
    """Hierarchical Risk Parity portfolio weight optimizer.

    Computes position weights via:
    1. Correlation estimation (with Ledoit-Wolf shrinkage for small samples)
    2. Quasi-diagonalization (hierarchical clustering + seriation)
    3. Recursive bisection (inverse-variance weight allocation)

    This replaces binary correlation blocking with continuous weights:
    correlated pairs get reduced weight instead of being removed entirely.
    """

    def __init__(
        self,
        linkage_method: str = "average",
        min_observations: int = MIN_OBSERVATIONS,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
    ):
        self.linkage_method = linkage_method
        self.min_observations = min_observations
        self.min_weight = min_weight
        self.max_weight = max_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(
        self,
        pairs: List[str],
        returns_data: Dict[str, Any],
    ) -> HRPResult:
        """Compute HRP portfolio weights from return series.

        Args:
            pairs: List of pair names to include
            returns_data: Dict mapping pair → pandas Series of returns
                (typically from df["close"].pct_change().dropna())

        Returns:
            HRPResult with normalized weights per pair.
        """
        result = HRPResult(linkage_method=self.linkage_method)

        # Filter to pairs that have sufficient data
        valid_pairs = []
        valid_returns = []
        for pair in pairs:
            ret = returns_data.get(pair)
            if ret is not None and len(ret) >= self.min_observations:
                valid_pairs.append(pair)
                valid_returns.append(np.asarray(ret, dtype=float))

        if len(valid_pairs) < 2:
            # Not enough pairs — equal weight fallback
            if valid_pairs:
                result.weights = {p: 1.0 / len(valid_pairs) for p in valid_pairs}
            else:
                result.weights = {p: 1.0 / len(pairs) for p in pairs}
            result.n_assets = len(result.weights)
            logger.debug("HRP: fallback to equal weights (%d pairs)", result.n_assets)
            return result

        # Align to common length (trim to shortest)
        min_len = min(len(r) for r in valid_returns)
        returns_matrix = np.column_stack([r[-min_len:] for r in valid_returns])
        result.n_samples = min_len
        result.n_assets = len(valid_pairs)

        # Step 1: Estimate correlation with shrinkage
        corr_matrix, shrinkage = self._estimate_correlation(returns_matrix)
        result.shrinkage_intensity = shrinkage

        # Step 2: Quasi-diagonalize
        ordered_indices, linkage_matrix = self._quasi_diagonalize(corr_matrix)
        result.ordered_pairs = [valid_pairs[i] for i in ordered_indices]

        # Step 3: Compute covariance for bisection (use correlation as proxy)
        # For HRP, we need variances — use return variances
        variances = np.var(returns_matrix, axis=0, ddof=1)
        variances = np.maximum(variances, 1e-10)  # floor
        cov_matrix = corr_matrix * np.outer(np.sqrt(variances), np.sqrt(variances))

        # Step 4: Recursive bisection
        raw_weights = self._recursive_bisection(cov_matrix, ordered_indices)

        # Step 5: Normalize and clip
        raw_weights = np.abs(raw_weights)
        total = raw_weights.sum()
        if total > 0:
            raw_weights = raw_weights / total
        else:
            raw_weights = np.ones(len(valid_pairs)) / len(valid_pairs)

        raw_weights = np.clip(raw_weights, self.min_weight, self.max_weight)
        raw_weights = raw_weights / raw_weights.sum()  # re-normalize after clipping

        # Build result
        result.weights = {valid_pairs[i]: float(raw_weights[i]) for i in range(len(valid_pairs))}
        result.effective_n = 1.0 / float(np.sum(raw_weights ** 2))  # Herfindahl

        logger.info(
            "HRP: %d pairs, %d samples, effective_n=%.1f, shrinkage=%.3f",
            result.n_assets, result.n_samples, result.effective_n, shrinkage,
        )

        return result

    # ------------------------------------------------------------------
    # Step 1: Correlation estimation with Ledoit-Wolf shrinkage
    # ------------------------------------------------------------------

    def _estimate_correlation(
        self, returns_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Estimate correlation matrix with shrinkage for small samples.

        Uses Ledoit-Wolf analytic shrinkage toward the identity matrix.
        Falls back to Pearson correlation for large samples.

        Returns:
            (correlation_matrix, shrinkage_intensity)
        """
        n_samples, n_assets = returns_matrix.shape
        shrinkage = 0.0

        if n_samples < 50:
            # Ledoit-Wolf shrinkage
            try:
                cov_sample = np.cov(returns_matrix, rowvar=False)
                # Analytic shrinkage toward scaled identity
                mu = np.trace(cov_sample) / n_assets
                delta = cov_sample - mu * np.eye(n_assets)
                # Shrinkage intensity (simplified Ledoit-Wolf formula)
                sum_sq = np.sum(delta ** 2) / n_assets
                # Asymptotic optimal shrinkage
                shrinkage = min(1.0, max(0.0, sum_sq / (n_samples * sum_sq + 1e-10)))
                cov_shrunk = (1 - shrinkage) * cov_sample + shrinkage * mu * np.eye(n_assets)
            except Exception:
                cov_shrunk = np.cov(returns_matrix, rowvar=False)
        else:
            cov_shrunk = np.cov(returns_matrix, rowvar=False)

        # Convert to correlation
        diag_sqrt = np.sqrt(np.maximum(np.diag(cov_shrunk), 1e-10))
        corr = cov_shrunk / np.outer(diag_sqrt, diag_sqrt)

        # Clip extreme correlations
        corr = np.clip(corr, -0.99, 0.99)
        np.fill_diagonal(corr, 1.0)

        # Ensure positive definite
        corr = self._ensure_positive_definite(corr)

        return corr, shrinkage

    @staticmethod
    def _ensure_positive_definite(matrix: np.ndarray, min_eigval: float = 1e-6) -> np.ndarray:
        """Adjust matrix if not positive definite."""
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals = np.maximum(eigvals, min_eigval)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T

    # ------------------------------------------------------------------
    # Step 2: Quasi-diagonalization via hierarchical clustering
    # ------------------------------------------------------------------

    def _quasi_diagonalize(
        self, corr_matrix: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """Reorder correlation matrix so correlated assets cluster on diagonal.

        Returns:
            (ordered_indices, linkage_matrix)
        """
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform

        # Convert correlation to distance: d = sqrt((1 - rho) / 2)
        distance_matrix = np.sqrt(np.maximum((1.0 - corr_matrix) / 2.0, 0.0))
        np.fill_diagonal(distance_matrix, 0.0)

        # Condensed distance for scipy
        condensed_dist = squareform(distance_matrix, checks=False)

        # Hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method=self.linkage_method)

        # Seriation: in-order tree traversal for quasi-diagonal ordering
        root = to_tree(linkage_matrix)
        ordered_indices = self._get_tree_ordering(root)

        return ordered_indices, linkage_matrix

    @staticmethod
    def _get_tree_ordering(node, order: Optional[List[int]] = None) -> List[int]:
        """In-order traversal of dendrogram to extract seriation order."""
        if order is None:
            order = []
        if node.is_leaf():
            order.append(node.id)
        else:
            if node.left is not None:
                HRPOptimizer._get_tree_ordering(node.left, order)
            if node.right is not None:
                HRPOptimizer._get_tree_ordering(node.right, order)
        return order

    # ------------------------------------------------------------------
    # Step 3: Recursive bisection with inverse-variance weighting
    # ------------------------------------------------------------------

    def _recursive_bisection(
        self, cov_matrix: np.ndarray, ordered_indices: List[int]
    ) -> np.ndarray:
        """Allocate weights via recursive bisection.

        Splits the quasi-diagonalized matrix recursively, assigning weight
        inversely proportional to cluster variance at each split.
        """
        n = cov_matrix.shape[0]
        weights = np.ones(n)

        # Work with ordered indices
        clusters = [list(range(len(ordered_indices)))]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Get cluster variances using ordered indices
                left_orig = [ordered_indices[i] for i in left]
                right_orig = [ordered_indices[i] for i in right]

                left_var = self._get_cluster_variance(cov_matrix, left_orig)
                right_var = self._get_cluster_variance(cov_matrix, right_orig)

                total_var = left_var + right_var
                if total_var < 1e-12:
                    left_weight = 0.5
                else:
                    left_weight = right_var / total_var  # inverse allocation

                right_weight = 1.0 - left_weight

                # Scale weights
                for i in left:
                    weights[ordered_indices[i]] *= left_weight
                for i in right:
                    weights[ordered_indices[i]] *= right_weight

                # Add sub-clusters for further splitting
                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights

    @staticmethod
    def _get_cluster_variance(cov_matrix: np.ndarray, indices: List[int]) -> float:
        """Compute cluster variance using inverse-variance weighting.

        Uses diagonal elements (asset variances) for inverse-variance weights,
        then computes the aggregate cluster variance.
        """
        sub_cov = cov_matrix[np.ix_(indices, indices)]
        diag = np.diag(sub_cov)
        volatilities = np.sqrt(np.maximum(diag, 1e-10))

        # Inverse-variance weights within cluster
        inv_var_w = 1.0 / volatilities
        inv_var_w = inv_var_w / inv_var_w.sum()

        # Cluster variance: w' * Cov * w
        cluster_var = float(inv_var_w @ sub_cov @ inv_var_w)
        return max(cluster_var, 1e-12)
