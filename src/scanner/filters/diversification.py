"""
Diversification Filter Module.

Filters correlated pairs to ensure portfolio diversification.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.scanner.analysis.correlation import CorrelationAnalyzer, CorrelationResult

logger = logging.getLogger(__name__)


class DiversificationFilter:
    """Filters trades for portfolio diversification.
    
    Combines correlation analysis with diversification rules:
    1. Identifies correlated pairs using CorrelationAnalyzer
    2. Filters to keep only best pair from each cluster
    3. Reduces position sizes for remaining correlated pairs
    
    This ensures the portfolio doesn't have excessive exposure
    to correlated moves.
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.7,
        max_correlation_group_size: int = 1,
        reduce_correlated_positions: bool = True,
    ):
        """Initialize diversification filter.
        
        Args:
            correlation_threshold: Threshold for considering pairs correlated
            max_correlation_group_size: Max pairs to keep per cluster
            reduce_correlated_positions: Whether to reduce position sizes
        """
        self.correlation_threshold = correlation_threshold
        self.max_correlation_group_size = max_correlation_group_size
        self.reduce_correlated_positions = reduce_correlated_positions
        
        # Correlation analyzer
        self._analyzer = CorrelationAnalyzer(threshold=correlation_threshold)
        
        # Cached results
        self._correlation_results: Dict[str, CorrelationResult] = {}
        self._filtered_pairs: Set[str] = set()
    
    def analyze_correlations(
        self,
        returns_data: Dict[str, pd.Series],
        confidence_lookup: Optional[Dict[str, float]] = None,
    ) -> Dict[str, CorrelationResult]:
        """Analyze pair correlations.
        
        Args:
            returns_data: Dict mapping pair -> returns Series
            confidence_lookup: Optional dict mapping pair -> confidence
            
        Returns:
            Dict mapping pair -> CorrelationResult
        """
        self._correlation_results = self._analyzer.analyze(
            returns_data, confidence_lookup
        )
        return self._correlation_results
    
    def filter(
        self,
        results: List[Any],  # List[PairAnalysis]
        returns_data: Optional[Dict[str, pd.Series]] = None,
        apply_position_reduction: bool = True,
    ) -> List[Any]:
        """Filter results for diversification.
        
        Args:
            results: List of PairAnalysis results
            returns_data: Optional returns data (uses cached if available)
            apply_position_reduction: Whether to reduce correlated positions
            
        Returns:
            Filtered list with diversification applied
        """
        if not results:
            return results
        
        # Analyze correlations if returns data provided
        if returns_data:
            confidence_lookup = {r.pair: r.confidence for r in results}
            self.analyze_correlations(returns_data, confidence_lookup)
        
        if not self._correlation_results:
            return results
        
        # Apply diversification filter (keep best from each cluster)
        filtered = self._analyzer.apply_diversification_filter(
            results, self._correlation_results
        )
        
        # Track which pairs were filtered out
        filtered_pairs = {r.pair for r in filtered}
        original_pairs = {r.pair for r in results}
        self._filtered_pairs = original_pairs - filtered_pairs
        
        # Apply position size reduction if enabled
        if apply_position_reduction and self.reduce_correlated_positions:
            reduction_factors = self._analyzer.calculate_position_reduction(
                filtered, self._correlation_results
            )
            
            for result in filtered:
                factor = reduction_factors.get(result.pair, 1.0)
                if factor < 1.0:
                    result.recommended_lots = round(
                        result.recommended_lots * factor, 2
                    )
                    
                    # Update correlation info on result
                    corr_result = self._correlation_results.get(result.pair)
                    if corr_result:
                        result.correlation_group = corr_result.correlation_group
        
        return filtered
    
    def get_correlation_warnings(self) -> Dict[str, str]:
        """Get correlation warnings for all pairs.
        
        Returns:
            Dict mapping pair -> warning string
        """
        warnings = {}
        
        for pair, result in self._correlation_results.items():
            if result.warning:
                warnings[pair] = result.warning
        
        return warnings
    
    def get_filtered_pairs(self) -> Set[str]:
        """Get set of pairs that were filtered out.
        
        Returns:
            Set of pair names that were removed by diversification filter
        """
        return self._filtered_pairs
    
    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get the correlation matrix from analyzer.
        
        Returns:
            Correlation matrix DataFrame or None
        """
        return self._analyzer.get_correlation_matrix()
    
    def get_cluster_summary(self) -> Dict[str, List[str]]:
        """Get summary of correlation clusters.
        
        Returns:
            Dict mapping cluster root -> list of member pairs
        """
        clusters: Dict[str, List[str]] = {}
        
        for pair, result in self._correlation_results.items():
            if result.correlation_group:
                root = result.correlation_group
                if root not in clusters:
                    clusters[root] = []
                clusters[root].append(pair)
        
        return clusters
    
    def should_skip_pair(
        self,
        pair: str,
        already_selected: Set[str],
    ) -> bool:
        """Check if a pair should be skipped due to correlation.
        
        Useful for real-time filtering during scan.
        
        Args:
            pair: Pair to check
            already_selected: Set of already selected pairs
            
        Returns:
            True if pair should be skipped
        """
        corr_result = self._correlation_results.get(pair)
        
        if not corr_result or not corr_result.correlated_pairs:
            return False
        
        # Check if any correlated pair already selected
        for correlated_pair in corr_result.correlated_pairs:
            if correlated_pair in already_selected:
                return True
        
        return False
    
    def get_best_from_cluster(
        self,
        cluster_pairs: List[str],
        results: List[Any],
    ) -> Optional[str]:
        """Get the best pair from a correlation cluster.
        
        Args:
            cluster_pairs: List of pairs in the cluster
            results: List of PairAnalysis results
            
        Returns:
            Best pair name or None
        """
        cluster_results = [
            r for r in results
            if r.pair in cluster_pairs
        ]
        
        if not cluster_results:
            return None
        
        # Sort by gates_passed then confidence
        cluster_results.sort(
            key=lambda x: (x.gates_passed, x.confidence),
            reverse=True
        )
        
        return cluster_results[0].pair
