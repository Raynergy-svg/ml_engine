"""
Correlation Analysis Module.

Provides pair correlation analysis with Union-Find clustering
for diversification filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis.
    
    Attributes:
        pair: Instrument name
        correlated_pairs: List of highly correlated pairs
        correlation_values: Dict of pair -> correlation value
        correlation_group: Cluster identifier
        is_best_in_cluster: Whether this is the best pair in its cluster
        position_reduction: Factor to reduce position size (1.0 = no reduction)
        warning: Human-readable correlation warning
    """
    pair: str
    correlated_pairs: List[str] = field(default_factory=list)
    correlation_values: Dict[str, float] = field(default_factory=dict)
    correlation_group: Optional[str] = None
    is_best_in_cluster: bool = True
    position_reduction: float = 1.0
    warning: Optional[str] = None


class CorrelationAnalyzer:
    """Analyzes correlations between FX pairs.
    
    Features:
    - Correlation matrix calculation from returns
    - Union-Find clustering for correlated pairs
    - Diversification filtering (keep best from cluster)
    - Position size reduction for correlated pairs
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        lookback_periods: int = 100,
    ):
        """Initialize correlation analyzer.
        
        Args:
            threshold: Correlation threshold for grouping (default 0.7)
            lookback_periods: Number of periods for correlation calculation
        """
        self.threshold = threshold
        self.lookback_periods = lookback_periods
        
        # Union-Find data structures
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}
        
        # Cached correlation data
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._correlation_details: List[Dict[str, Any]] = []
    
    def _find(self, pair: str) -> str:
        """Find root of pair with path compression."""
        if pair not in self._parent:
            self._parent[pair] = pair
            self._rank[pair] = 0
        
        if self._parent[pair] != pair:
            self._parent[pair] = self._find(self._parent[pair])
        
        return self._parent[pair]
    
    def _union(self, p1: str, p2: str) -> None:
        """Union two pairs into same cluster with rank optimization."""
        r1, r2 = self._find(p1), self._find(p2)
        
        if r1 == r2:
            return
        
        # Union by rank
        if self._rank[r1] < self._rank[r2]:
            self._parent[r1] = r2
        elif self._rank[r1] > self._rank[r2]:
            self._parent[r2] = r1
        else:
            self._parent[r2] = r1
            self._rank[r1] += 1
    
    def analyze(
        self,
        returns_data: Dict[str, pd.Series],
        confidence_lookup: Optional[Dict[str, float]] = None,
    ) -> Dict[str, CorrelationResult]:
        """Analyze correlations between pairs.
        
        Args:
            returns_data: Dict mapping pair -> returns Series
            confidence_lookup: Optional dict mapping pair -> confidence score
            
        Returns:
            Dict mapping pair -> CorrelationResult
        """
        # Reset state
        self._parent = {}
        self._rank = {}
        self._correlation_details = []
        
        results: Dict[str, CorrelationResult] = {}
        
        if len(returns_data) < 2:
            return results
        
        try:
            # Build returns DataFrame
            pairs_with_data = list(returns_data.keys())
            
            returns_df = pd.DataFrame({
                pair: returns_data[pair].tail(self.lookback_periods)
                for pair in pairs_with_data
            })
            
            # Calculate correlation matrix
            self._correlation_matrix = returns_df.corr()
            
            confidence_lookup = confidence_lookup or {}
            
            # Find high correlations and build clusters
            for i, pair1 in enumerate(pairs_with_data):
                for pair2 in pairs_with_data[i+1:]:
                    if pair1 not in self._correlation_matrix.columns:
                        continue
                    if pair2 not in self._correlation_matrix.columns:
                        continue
                    
                    corr = self._correlation_matrix.loc[pair1, pair2]
                    
                    if abs(corr) > self.threshold:
                        # Union into same cluster
                        self._union(pair1, pair2)
                        
                        # Store correlation detail
                        corr_type = "positive" if corr > 0 else "negative"
                        conf1 = confidence_lookup.get(pair1, 0)
                        conf2 = confidence_lookup.get(pair2, 0)
                        better_pair = pair1 if conf1 >= conf2 else pair2
                        
                        self._correlation_details.append({
                            'pair1': pair1,
                            'pair2': pair2,
                            'corr': corr,
                            'type': corr_type,
                            'prefer': better_pair,
                            'conf1': conf1,
                            'conf2': conf2,
                        })
            
            # Build results for each pair
            for pair in pairs_with_data:
                root = self._find(pair)
                
                # Find all pairs in same cluster
                cluster_pairs = [
                    p for p in pairs_with_data
                    if self._find(p) == root and p != pair
                ]
                
                # Get correlation values with cluster pairs
                corr_values = {}
                for other in cluster_pairs:
                    if other in self._correlation_matrix.columns:
                        corr_values[other] = self._correlation_matrix.loc[pair, other]
                
                # Build warning string
                warning = None
                if cluster_pairs:
                    corr_strs = [
                        f"{p} ({corr_values.get(p, 0):+.2f})"
                        for p in cluster_pairs[:2]  # Show top 2
                    ]
                    warning = f"correlated with {', '.join(corr_strs)}"
                
                results[pair] = CorrelationResult(
                    pair=pair,
                    correlated_pairs=cluster_pairs,
                    correlation_values=corr_values,
                    correlation_group=root if cluster_pairs else None,
                    warning=warning,
                )
            
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
        
        return results
    
    def apply_diversification_filter(
        self,
        results: List[Any],  # List[PairAnalysis]
        correlation_results: Dict[str, CorrelationResult],
    ) -> List[Any]:
        """Filter to keep only the best pair from each correlation cluster.
        
        Args:
            results: List of PairAnalysis results
            correlation_results: Dict from analyze()
            
        Returns:
            Filtered list with only one pair per cluster
        """
        if not correlation_results:
            return results
        
        # Build clusters
        clusters: Dict[str, List[Any]] = {}
        unclustered: List[Any] = []
        
        for result in results:
            pair = result.pair
            corr_result = correlation_results.get(pair)
            
            if corr_result and corr_result.correlation_group:
                root = corr_result.correlation_group
                if root not in clusters:
                    clusters[root] = []
                clusters[root].append(result)
            else:
                unclustered.append(result)
        
        # Keep only best from each cluster
        filtered: List[Any] = []
        
        for root, cluster_results in clusters.items():
            # Sort by gates_passed (bool→True first) then confidence
            cluster_results.sort(
                key=lambda x: (x.gates_passed, x.confidence),
                reverse=True
            )
            best = cluster_results[0]
            
            # Update correlation result to mark as best
            if best.pair in correlation_results:
                correlation_results[best.pair].is_best_in_cluster = True
            
            # Mark filtered pairs
            for other in cluster_results[1:]:
                if other.pair in correlation_results:
                    correlation_results[other.pair].is_best_in_cluster = False
            
            filtered.append(best)
        
        # Add unclustered pairs
        filtered.extend(unclustered)
        
        # Re-sort by overall score
        filtered.sort(key=lambda x: x.overall_score, reverse=True)
        
        return filtered
    
    def calculate_position_reduction(
        self,
        results: List[Any],  # List[PairAnalysis]
        correlation_results: Dict[str, CorrelationResult],
    ) -> Dict[str, float]:
        """Calculate position size reduction factors for correlated pairs.
        
        Reduces position size based on number of correlated tradeable pairs:
        - 1 correlated pair = 50% each
        - 2 correlated pairs = 33% each
        - etc.
        
        Args:
            results: List of PairAnalysis results
            correlation_results: Dict from analyze()
            
        Returns:
            Dict mapping pair -> reduction factor (1.0 = no reduction)
        """
        reduction_factors: Dict[str, float] = {}
        
        # Get tradeable pairs
        tradeable_pairs: Set[str] = {
            r.pair for r in results if r.gates_passed
        }
        
        for result in results:
            pair = result.pair
            reduction_factors[pair] = 1.0
            
            if not result.gates_passed:
                continue
            
            corr_result = correlation_results.get(pair)
            if not corr_result or not corr_result.correlated_pairs:
                continue
            
            # Count correlated pairs that are also tradeable
            correlated_tradeable = [
                p for p in corr_result.correlated_pairs
                if p in tradeable_pairs
            ]
            
            if correlated_tradeable:
                # Reduce by 1/(1 + num_correlated)
                reduction = 1.0 / (1 + len(correlated_tradeable))
                reduction_factors[pair] = reduction
                corr_result.position_reduction = reduction
                
                # Update warning
                corr_result.warning = (
                    f"{corr_result.warning or ''} | "
                    f"position reduced to {reduction:.0%}"
                ).strip(" |")
        
        return reduction_factors
    
    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get the computed correlation matrix."""
        return self._correlation_matrix
    
    def get_correlation_details(self) -> List[Dict[str, Any]]:
        """Get detailed correlation information."""
        return self._correlation_details
