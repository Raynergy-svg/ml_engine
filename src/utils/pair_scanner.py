"""
Multi-Pair Scanner for FX Trading.

Scans multiple currency pairs to find:
1. Which pairs the model predicts with highest confidence
2. Which pairs have the best historical accuracy
3. Which pairs have the best risk/reward setup right now
4. Correlation between pairs (for diversification)

Supported pairs (major + crosses):
- Majors: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
- Crosses: EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY, EUR/AUD, GBP/AUD
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Major FX pairs
MAJOR_PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
]

# Cross pairs
CROSS_PAIRS = [
    "EUR_GBP",
    "EUR_JPY",
    "GBP_JPY",
    "AUD_JPY",
    "EUR_AUD",
    "GBP_AUD",
    "EUR_CHF",
    "GBP_CHF",
    "AUD_NZD",
    "CAD_JPY",
]

# All tradeable pairs
ALL_PAIRS = MAJOR_PAIRS + CROSS_PAIRS

# Pip values (for position sizing)
PIP_VALUES = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "USD_CHF": 0.0001,
    "AUD_USD": 0.0001,
    "USD_CAD": 0.0001,
    "NZD_USD": 0.0001,
    "EUR_GBP": 0.0001,
    "EUR_JPY": 0.01,
    "GBP_JPY": 0.01,
    "AUD_JPY": 0.01,
    "EUR_AUD": 0.0001,
    "GBP_AUD": 0.0001,
    "EUR_CHF": 0.0001,
    "GBP_CHF": 0.0001,
    "AUD_NZD": 0.0001,
    "CAD_JPY": 0.01,
}


@dataclass
class PairAnalysis:
    """Analysis results for a single pair."""
    pair: str
    direction: str  # "LONG" or "SHORT"
    confidence: float
    volatility_percentile: float
    trend_strength: float
    optimal_entry_score: float
    historical_accuracy: Optional[float] = None
    expected_value: Optional[float] = None
    current_price: Optional[float] = None
    atr: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    @property
    def overall_score(self) -> float:
        """Combined score for ranking pairs."""
        # Weighted combination of factors
        score = (
            self.confidence * 0.3 +
            self.optimal_entry_score * 0.25 +
            self.trend_strength * 0.2 +
            (self.historical_accuracy or 0.5) * 0.15 +
            self.volatility_percentile * 0.1
        )
        return score


@dataclass
class ScanResult:
    """Results from scanning multiple pairs."""
    timestamp: datetime
    pair_analyses: List[PairAnalysis]
    best_long: Optional[PairAnalysis] = None
    best_short: Optional[PairAnalysis] = None
    best_overall: Optional[PairAnalysis] = None
    correlations: Optional[pd.DataFrame] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for display."""
        data = []
        for pa in self.pair_analyses:
            data.append({
                "pair": pa.pair,
                "direction": pa.direction,
                "confidence": f"{pa.confidence:.1%}",
                "trend": f"{pa.trend_strength:.2f}",
                "volatility": f"{pa.volatility_percentile:.1%}",
                "entry_score": f"{pa.optimal_entry_score:.2f}",
                "overall": f"{pa.overall_score:.2f}",
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("overall", ascending=False)
        return df


class PairScanner:
    """
    Scans multiple FX pairs to find the best trading opportunities.
    """
    
    def __init__(
        self,
        model: Any = None,
        feature_engineer: Any = None,
        oanda_client: Any = None,
        pairs: Optional[List[str]] = None,
        granularity: str = "H1",
    ):
        """
        Initialize the scanner.
        
        Args:
            model: Trained prediction model (ensemble or single)
            feature_engineer: FeatureEngineering instance
            oanda_client: OANDA API client for fetching data
            pairs: List of pairs to scan (default: all)
            granularity: Timeframe for analysis
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.oanda_client = oanda_client
        self.pairs = pairs or ALL_PAIRS
        self.granularity = granularity
        
        # Cache for historical accuracy
        self._historical_accuracy: Dict[str, float] = {}
    
    def scan_all(
        self,
        candles_per_pair: int = 500,
        min_confidence: float = 0.0,
        verbose: bool = True,
    ) -> ScanResult:
        """
        Scan all configured pairs.
        
        Args:
            candles_per_pair: Number of candles to fetch per pair
            min_confidence: Minimum confidence threshold
            verbose: Print progress
        
        Returns:
            ScanResult with all pair analyses
        """
        analyses = []
        
        for pair in self.pairs:
            if verbose:
                logger.info(f"Scanning {pair}...")
            
            try:
                analysis = self._analyze_pair(pair, candles_per_pair)
                if analysis.confidence >= min_confidence:
                    analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze {pair}: {e}")
                continue
        
        # Sort by overall score
        analyses.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Find best opportunities
        best_long = None
        best_short = None
        for a in analyses:
            if a.direction == "LONG" and (best_long is None or a.overall_score > best_long.overall_score):
                best_long = a
            if a.direction == "SHORT" and (best_short is None or a.overall_score > best_short.overall_score):
                best_short = a
        
        best_overall = analyses[0] if analyses else None
        
        # Calculate correlations
        correlations = self._calculate_correlations() if len(analyses) > 1 else None
        
        return ScanResult(
            timestamp=datetime.now(),
            pair_analyses=analyses,
            best_long=best_long,
            best_short=best_short,
            best_overall=best_overall,
            correlations=correlations,
        )
    
    def _analyze_pair(self, pair: str, n_candles: int) -> PairAnalysis:
        """Analyze a single pair."""
        # Fetch data
        df = self._fetch_pair_data(pair, n_candles)
        
        if df is None or len(df) < 100:
            raise ValueError(f"Insufficient data for {pair}")
        
        # Engineer features
        if self.feature_engineer is not None:
            df = self.feature_engineer.create_features(df, include_all=True)
        
        # Get model prediction
        direction, confidence = self._get_prediction(df, pair)
        
        # Calculate metrics
        volatility_pct = self._calculate_volatility_percentile(df)
        trend_strength = self._calculate_trend_strength(df)
        entry_score = self._calculate_entry_score(df)
        
        # Get current price and ATR
        current_price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1] if "atr" in df.columns else None
        
        # Historical accuracy (if available)
        hist_acc = self._historical_accuracy.get(pair)
        
        # Expected value
        ev = None
        if hist_acc is not None:
            # Simple EV: (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            # Assuming 2:1 R/R
            ev = (hist_acc * 2) - ((1 - hist_acc) * 1)
        
        return PairAnalysis(
            pair=pair,
            direction=direction,
            confidence=confidence,
            volatility_percentile=volatility_pct,
            trend_strength=trend_strength,
            optimal_entry_score=entry_score,
            historical_accuracy=hist_acc,
            expected_value=ev,
            current_price=current_price,
            atr=atr,
            timestamp=datetime.now(),
        )
    
    def _fetch_pair_data(self, pair: str, n_candles: int) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a pair."""
        if self.oanda_client is None:
            logger.warning(f"No OANDA client - cannot fetch {pair}")
            return None
        
        try:
            resp = self.oanda_client.get_candles(
                pair,
                granularity=self.granularity,
                count=n_candles,
                price="MBA",
            )
            
            candles = resp.get("candles", [])
            if not candles:
                return None
            
            # Parse candles
            data = []
            for c in candles:
                mid = c.get("mid", {})
                data.append({
                    "time": c.get("time"),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })
            
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {pair}: {e}")
            return None
    
    def _get_prediction(self, df: pd.DataFrame, pair: str) -> Tuple[str, float]:
        """Get model prediction for the pair."""
        if self.model is None:
            # No model - return neutral
            return "LONG", 0.5
        
        # Prepare features
        feature_cols = [c for c in df.columns if not c.startswith("target_") and c not in ["time", "open", "high", "low", "close", "volume"]]
        
        if not feature_cols:
            return "LONG", 0.5
        
        # Get last row (current state)
        X = df[feature_cols].iloc[-1:].values
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # For sequence models, we need to create a sequence
        if hasattr(self.model, "input_shape") and len(self.model.input_shape) == 3:
            seq_len = self.model.input_shape[1]
            if len(df) >= seq_len:
                X = df[feature_cols].iloc[-seq_len:].values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X = X[np.newaxis, :, :]  # Add batch dimension
        
        try:
            pred = self.model.predict(X)
            
            if isinstance(pred, dict):
                direction_prob = float(pred.get("direction", pred.get("output_0", [0.5]))[0])
                confidence = float(pred.get("confidence", [abs(direction_prob - 0.5) * 2])[0])
            else:
                direction_prob = float(np.asarray(pred).flatten()[0])
                confidence = abs(direction_prob - 0.5) * 2
            
            direction = "LONG" if direction_prob > 0.5 else "SHORT"
            
            return direction, confidence
            
        except Exception as e:
            logger.warning(f"Prediction failed for {pair}: {e}")
            return "LONG", 0.5
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """Calculate current volatility percentile."""
        if "atr" not in df.columns:
            # Calculate ATR
            high = df["high"]
            low = df["low"]
            close = df["close"].shift(1)
            tr = pd.concat([
                high - low,
                (high - close).abs(),
                (low - close).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
        else:
            atr = df["atr"]
        
        # Current ATR percentile
        current_atr = atr.iloc[-1]
        percentile = (atr < current_atr).mean()
        
        return float(percentile)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate current trend strength (0-1)."""
        if "adx" in df.columns:
            adx = df["adx"].iloc[-1]
            # Normalize to 0-1 (ADX typically 0-60)
            return min(adx / 60, 1.0)
        
        # Fallback: use price momentum
        returns = df["close"].pct_change(20)
        momentum = abs(returns.iloc[-1])
        return min(momentum * 10, 1.0)  # Scale and cap
    
    def _calculate_entry_score(self, df: pd.DataFrame) -> float:
        """Calculate optimal entry score (0-1)."""
        score = 0.5  # Neutral default
        
        # Factor 1: RSI not extreme
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if 30 < rsi < 70:
                score += 0.15
            elif 20 < rsi < 80:
                score += 0.05
        
        # Factor 2: Trend alignment
        if "sma_20" in df.columns and "sma_50" in df.columns:
            sma_20 = df["sma_20"].iloc[-1]
            sma_50 = df["sma_50"].iloc[-1]
            close = df["close"].iloc[-1]
            
            if (close > sma_20 > sma_50) or (close < sma_20 < sma_50):
                score += 0.15  # Aligned trend
        
        # Factor 3: Not overextended from MA
        if "sma_20" in df.columns:
            sma_20 = df["sma_20"].iloc[-1]
            close = df["close"].iloc[-1]
            distance = abs(close - sma_20) / sma_20
            
            if distance < 0.02:  # Within 2% of MA
                score += 0.1
        
        # Factor 4: Volatility not extreme
        vol_pct = self._calculate_volatility_percentile(df)
        if 0.3 < vol_pct < 0.8:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_correlations(self) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix between pairs."""
        # Would need historical data for all pairs
        # Placeholder - implement with cached data
        return None
    
    def set_historical_accuracy(self, pair: str, accuracy: float) -> None:
        """Set historical accuracy for a pair."""
        self._historical_accuracy[pair] = accuracy
    
    def get_diversified_portfolio(
        self,
        scan_result: ScanResult,
        max_pairs: int = 3,
        min_correlation_distance: float = 0.5,
    ) -> List[PairAnalysis]:
        """
        Select diversified pairs from scan results.
        
        Args:
            scan_result: Results from scan_all()
            max_pairs: Maximum pairs to select
            min_correlation_distance: Minimum "distance" between pairs
        
        Returns:
            List of selected pair analyses
        """
        analyses = scan_result.pair_analyses
        if len(analyses) <= max_pairs:
            return analyses
        
        # Simple diversification: avoid same base/quote currencies
        selected = []
        used_currencies = set()
        
        for analysis in analyses:
            base, quote = analysis.pair.split("_")
            
            # Check if currencies already used
            if base not in used_currencies and quote not in used_currencies:
                selected.append(analysis)
                used_currencies.add(base)
                used_currencies.add(quote)
                
                if len(selected) >= max_pairs:
                    break
        
        return selected


def scan_pairs_quick(
    oanda_client: Any,
    model: Any = None,
    feature_engineer: Any = None,
    pairs: Optional[List[str]] = None,
    granularity: str = "H1",
    verbose: bool = True,
) -> ScanResult:
    """
    Quick scan of FX pairs.
    
    Args:
        oanda_client: OANDA API client
        model: Optional prediction model
        feature_engineer: Optional feature engineer
        pairs: Pairs to scan (default: majors only)
        granularity: Timeframe
        verbose: Print progress
    
    Returns:
        ScanResult
    """
    scanner = PairScanner(
        model=model,
        feature_engineer=feature_engineer,
        oanda_client=oanda_client,
        pairs=pairs or MAJOR_PAIRS,
        granularity=granularity,
    )
    
    return scanner.scan_all(verbose=verbose)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Pair Scanner Test")
    print("-" * 50)
    
    # Test without OANDA client (mock data)
    scanner = PairScanner(pairs=["EUR_USD", "GBP_USD", "USD_JPY"])
    
    # Mock analysis
    analyses = [
        PairAnalysis(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.65,
            volatility_percentile=0.45,
            trend_strength=0.6,
            optimal_entry_score=0.7,
            historical_accuracy=0.52,
        ),
        PairAnalysis(
            pair="GBP_USD",
            direction="SHORT",
            confidence=0.72,
            volatility_percentile=0.55,
            trend_strength=0.7,
            optimal_entry_score=0.65,
            historical_accuracy=0.54,
        ),
        PairAnalysis(
            pair="USD_JPY",
            direction="LONG",
            confidence=0.58,
            volatility_percentile=0.35,
            trend_strength=0.4,
            optimal_entry_score=0.5,
            historical_accuracy=0.51,
        ),
    ]
    
    result = ScanResult(
        timestamp=datetime.now(),
        pair_analyses=analyses,
        best_long=analyses[0],
        best_short=analyses[1],
        best_overall=analyses[1],
    )
    
    print("\nScan Results:")
    print(result.to_dataframe().to_string(index=False))
    
    print(f"\nBest Long: {result.best_long.pair} ({result.best_long.confidence:.1%})")
    print(f"Best Short: {result.best_short.pair} ({result.best_short.confidence:.1%})")
    print(f"Best Overall: {result.best_overall.pair} (score: {result.best_overall.overall_score:.2f})")
    
    # Test diversification
    diversified = scanner.get_diversified_portfolio(result, max_pairs=2)
    print(f"\nDiversified Portfolio: {[p.pair for p in diversified]}")
    
    print("\nPair Scanner test complete!")

