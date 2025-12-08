"""
Enhanced reasoning module for stock market prediction with improved performance.
Includes advanced reasoning techniques, uncertainty quantification, and explainability.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm

# Configure logging
logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Enhanced reasoning engine for stock market prediction with improved explainability,
    uncertainty quantification, and decision support.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the reasoning engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.uncertainty_threshold = self.config.get("uncertainty_threshold", 0.3)
        self.output_dir = Path(self.config.get("output_dir", "./output"))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize state
        self.predictions = None
        self.uncertainties = None
        self.explanations = None
        self.decision_signals = None
        
        logger.info("ReasoningEngine initialized")
    
    def analyze_predictions(
        self,
        predictions: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        actual_values: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        ticker_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze predictions and generate insights.
        
        Args:
            predictions: Model predictions
            uncertainties: Prediction uncertainties (optional)
            actual_values: Actual values for comparison (optional)
            timestamps: Timestamps for predictions (optional)
            ticker_symbols: Ticker symbols for predictions (optional)
            
        Returns:
            Dictionary of analysis results
        """
        # Store predictions and uncertainties
        self.predictions = predictions
        self.uncertainties = uncertainties
        
        # Initialize results dictionary
        results = {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "metrics": {},
            "signals": {},
            "insights": []
        }
        
        # Calculate metrics if actual values are provided
        if actual_values is not None:
            metrics = self._calculate_metrics(predictions, actual_values)
            results["metrics"] = metrics
        
        # Generate trading signals
        signals = self._generate_signals(
            predictions, 
            uncertainties=uncertainties,
            timestamps=timestamps
        )
        results["signals"] = signals
        
        # Generate explanations
        explanations = self._generate_explanations(
            predictions, 
            uncertainties=uncertainties,
            actual_values=actual_values,
            timestamps=timestamps,
            ticker_symbols=ticker_symbols
        )
        results["explanations"] = explanations
        self.explanations = explanations
        
        # Generate insights
        insights = self._generate_insights(
            predictions, 
            uncertainties=uncertainties,
            actual_values=actual_values,
            signals=signals,
            timestamps=timestamps,
            ticker_symbols=ticker_symbols
        )
        results["insights"] = insights
        
        return results
    
    def _calculate_metrics(
        self, 
        predictions: np.ndarray, 
        actual_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            predictions: Model predictions
            actual_values: Actual values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays are flattened
        predictions = predictions.flatten()
        actual_values = actual_values.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)
        
        # Calculate directional accuracy
        direction_actual = np.sign(np.diff(actual_values, prepend=actual_values[0]))
        direction_pred = np.sign(np.diff(predictions, prepend=predictions[0]))
        directional_accuracy = np.mean(direction_actual == direction_pred)
        
        # Calculate profit factor (if predictions were used for trading)
        returns_actual = np.diff(actual_values, prepend=actual_values[0])
        returns_pred = np.diff(predictions, prepend=predictions[0])
        profitable_trades = returns_actual * np.sign(returns_pred) > 0
        profit_factor = np.sum(returns_actual[profitable_trades]) / (np.abs(np.sum(returns_actual[~profitable_trades])) + 1e-10)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
            "profit_factor": profit_factor
        }
    
    def _generate_signals(
        self,
        predictions: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate trading signals based on predictions and uncertainties.
        
        Args:
            predictions: Model predictions
            uncertainties: Prediction uncertainties (optional)
            timestamps: Timestamps for predictions (optional)
            
        Returns:
            Dictionary of trading signals
        """
        # Ensure predictions are flattened
        predictions = predictions.flatten()
        
        # Calculate price changes
        price_changes = np.diff(predictions, prepend=predictions[0])
        
        # Generate basic signals
        signals = {
            "price_change": price_changes,
            "direction": np.sign(price_changes)
        }
        
        # Generate signals with uncertainty if available
        if uncertainties is not None:
            uncertainties = uncertainties.flatten()
            
            # Calculate confidence scores
            confidence = 1.0 - uncertainties
            
            # Generate confidence-weighted signals
            confident_direction = np.sign(price_changes) * (confidence > self.confidence_threshold)
            
            signals.update({
                "uncertainty": uncertainties,
                "confidence": confidence,
                "confident_direction": confident_direction
            })
        
        # Generate trend signals
        if len(predictions) >= 3:
            # Simple moving average
            sma_3 = np.convolve(predictions, np.ones(3)/3, mode='valid')
            sma_3 = np.pad(sma_3, (2, 0), mode='edge')
            
            # Exponential moving average
            alpha = 0.2
            ema = np.zeros_like(predictions)
            ema[0] = predictions[0]
            for i in range(1, len(predictions)):
                ema[i] = alpha * predictions[i] + (1 - alpha) * ema[i-1]
            
            # Trend signals
            trend = np.sign(predictions - ema)
            
            signals.update({
                "sma_3": sma_3,
                "ema": ema,
                "trend": trend
            })
        
        # Store decision signals
        self.decision_signals = signals
        
        return signals
    
    def _generate_explanations(
        self,
        predictions: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        actual_values: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        ticker_symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for predictions.
        
        Args:
            predictions: Model predictions
            uncertainties: Prediction uncertainties (optional)
            actual_values: Actual values for comparison (optional)
            timestamps: Timestamps for predictions (optional)
            ticker_symbols: Ticker symbols for predictions (optional)
            
        Returns:
            List of explanation dictionaries
        """
        # Ensure predictions are flattened
        predictions = predictions.flatten()
        
        # Initialize explanations
        explanations = []
        
        # Generate explanations for each prediction
        for i in range(len(predictions)):
            explanation = {
                "index": i,
                "prediction": predictions[i],
                "timestamp": timestamps[i] if timestamps is not None else None,
                "ticker": ticker_symbols[i] if ticker_symbols is not None else None
            }
            
            # Add uncertainty if available
            if uncertainties is not None:
                uncertainty = uncertainties.flatten()[i]
                confidence = 1.0 - uncertainty
                
                explanation.update({
                    "uncertainty": uncertainty,
                    "confidence": confidence,
                    "confidence_level": "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
                })
            
            # Add actual value if available
            if actual_values is not None:
                actual = actual_values.flatten()[i]
                error = predictions[i] - actual
                
                explanation.update({
                    "actual": actual,
                    "error": error,
                    "error_percentage": error / actual * 100 if actual != 0 else float('inf')
                })
            
            # Add trend information if possible
            if i > 0:
                price_change = predictions[i] - predictions[i-1]
                direction = "up" if price_change > 0 else "down" if price_change < 0 else "flat"
                
                explanation.update({
                    "price_change": price_change,
                    "direction": direction
                })
            
            explanations.append(explanation)
        
        return explanations
    
    def _generate_insights(
        self,
        predictions: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        actual_values: Optional[np.ndarray] = None,
        signals: Dict[str, np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        ticker_symbols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate insights from predictions and signals.
        
        Args:
            predictions: Model predictions
            uncertainties: Prediction uncertainties (optional)
            actual_values: Actual values for comparison (optional)
            signals: Trading signals (optional)
            timestamps: Timestamps for predictions (optional)
            ticker_symbols: Ticker symbols for predictions (optional)
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Ensure predictions are flattened
        predictions = predictions.flatten()
        
        # Overall trend insight
        if len(predictions) > 1:
            overall_change = predictions[-1] - predictions[0]
            overall_percent = (overall_change / predictions[0]) * 100
            
            trend_message = f"Overall trend is {'upward' if overall_change > 0 else 'downward'} with a {abs(overall_percent):.2f}% change."
            insights.append(trend_message)
        
        # Uncertainty insights
        if uncertainties is not None:
            uncertainties = uncertainties.flatten()
            avg_uncertainty = np.mean(uncertainties)
            max_uncertainty = np.max(uncertainties)
            min_uncertainty = np.min(uncertainties)
            
            uncertainty_message = f"Average prediction uncertainty is {avg_uncertainty:.2f} (range: {min_uncertainty:.2f} to {max_uncertainty:.2f})."
            insights.append(uncertainty_message)
            
            # High uncertainty periods
            high_uncertainty_indices = np.where(uncertainties > self.uncertainty_threshold)[0]
            if len(high_uncertainty_indices) > 0:
                high_uncertainty_message = f"High uncertainty detected in {len(high_uncertainty_indices)} predictions."
                insights.append(high_uncertainty_message)
        
        # Performance insights if actual values are available
        if actual_values is not None:
            actual_values = actual_values.flatten()
            
            # Calculate errors
            errors = predictions - actual_values
            mse = np.mean(errors ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(errors))
            
            performance_message = f"Model performance: RMSE = {rmse:.4f}, MAE = {mae:.4f}."
            insights.append(performance_message)
            
            # Directional accuracy
            direction_actual = np.sign(np.diff(actual_values, prepend=actual_values[0]))
            direction_pred = np.sign(np.diff(predictions, prepend=predictions[0]))
            directional_accuracy = np.mean(direction_actual == direction_pred) * 100
            
            direction_message = f"Directional accuracy: {directional_accuracy:.2f}%."
            insights.append(direction_message)
        
        # Signal insights
        if signals is not None and "confident_direction" in signals:
            confident_direction = signals["confident_direction"]
            buy_signals = np.sum(confident_direction > 0)
            sell_signals = np.sum(confident_direction < 0)
            hold_signals = np.sum(confident_direction == 0)
            
            signal_message = f"Trading signals: {buy_signals} buy, {sell_signals} sell, {hold_signals} hold."
            insights.append(signal_message)
        
        return insights
    
    def visualize_predictions(
        self,
        predictions: Optional[np.ndarray] = None,
        uncertainties: Optional[np.ndarray] = None,
        actual_values: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        ticker_symbols: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> Optional[plt.Figure]:
        """
        Visualize predictions with uncertainties and actual values.
        
        Args:
            predictions: Model predictions (optional, uses stored predictions if None)
            uncertainties: Prediction uncertainties (optional, uses stored uncertainties if None)
            actual_values: Actual values for comparison (optional)
            timestamps: Timestamps for predictions (optional)
            ticker_symbols: Ticker symbols for predictions (optional)
            save_path: Path to save visualization (optional)
            show_plot: Whether to show the plot
            
        Returns:
            Matplotlib figure or None if no predictions available
        """
        # Use stored predictions if not provided
        predictions = predictions if predictions is not None else self.predictions
        uncertainties = uncertainties if uncertainties is not None else self.uncertainties
        
        if predictions is None:
            logger.warning("No predictions a<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>