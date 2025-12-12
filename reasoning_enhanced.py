"""
Enhanced reasoning module for stock market prediction with improved performance.
Includes advanced reasoning techniques, uncertainty quantification, and explainability.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
            uncertainties=uncertainties
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
            signals=signals
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
        uncertainties: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate trading signals based on predictions and uncertainties.
        
        Args:
            predictions: Model predictions
            uncertainties: Prediction uncertainties (optional)
            
        Returns:
            Dictionary of trading signals
        """
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
            uncertainties_arr = np.asarray(uncertainties, dtype=float)
            if uncertainties_arr.ndim == 0:
                uncertainties_arr = np.full_like(predictions, float(uncertainties_arr))
            else:
                uncertainties_arr = uncertainties_arr.flatten()
                if uncertainties_arr.size == 1 and predictions.size > 1:
                    uncertainties_arr = np.full_like(predictions, float(uncertainties_arr[0]))
                elif uncertainties_arr.size != predictions.size:
                    uncertainties_arr = np.full_like(predictions, float(np.mean(uncertainties_arr)))
            
            # Calculate confidence scores
            confidence = 1.0 - uncertainties_arr
            
            # Generate confidence-weighted signals
            confident_direction = np.sign(price_changes) * (confidence > self.confidence_threshold)
            
            signals.update({
                "uncertainty": uncertainties_arr,
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
    
    def _build_explanation_base(
        self,
        index: int,
        prediction: float,
        timestamps: Optional[np.ndarray],
        ticker_symbols: Optional[List[str]]
    ) -> Dict[str, Any]:
        return {
            "index": index,
            "prediction": prediction,
            "timestamp": timestamps[index] if timestamps is not None else None,
            "ticker": ticker_symbols[index] if ticker_symbols is not None else None
        }
    
    def _add_uncertainty_to_explanation(
        self,
        explanation: Dict[str, Any],
        uncertainty: float
    ) -> Dict[str, Any]:
        confidence = 1.0 - uncertainty

        if confidence > 0.8:
            confidence_level = "high"
        elif confidence > 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        explanation.update({
            "uncertainty": uncertainty,
            "confidence": confidence,
            "confidence_level": confidence_level
        })
        
        return explanation
    
    def _add_actual_to_explanation(
        self,
        explanation: Dict[str, Any],
        actual: float,
        prediction: float
    ) -> Dict[str, Any]:
        error = prediction - actual
        
        explanation.update({
            "actual": actual,
            "error": error,
            "error_percentage": error / actual * 100 if actual != 0 else float('inf')
        })
        
        return explanation
    
    def _add_trend_to_explanation(
        self,
        explanation: Dict[str, Any],
        current_prediction: float,
        previous_prediction: float
    ) -> Dict[str, Any]:
        price_change = current_prediction - previous_prediction

        if price_change > 0:
            direction = "up"
        elif price_change < 0:
            direction = "down"
        else:
            direction = "flat"
        
        explanation.update({
            "price_change": price_change,
            "direction": direction
        })
        
        return explanation

    def _normalize_uncertainties(
        self,
        predictions_flat: np.ndarray,
        uncertainties: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if uncertainties is None:
            return None

        uncertainties_arr = np.asarray(uncertainties, dtype=float)
        if uncertainties_arr.ndim == 0:
            return np.full_like(predictions_flat, float(uncertainties_arr))

        uncertainties_flat = uncertainties_arr.flatten()
        if uncertainties_flat.size == 1 and predictions_flat.size > 1:
            return np.full_like(predictions_flat, float(uncertainties_flat[0]))

        if uncertainties_flat.size != predictions_flat.size:
            return np.full_like(predictions_flat, float(np.mean(uncertainties_flat)))

        return uncertainties_flat

    def _apply_optional_uncertainty(
        self,
        explanation: Dict[str, Any],
        uncertainties_flat: Optional[np.ndarray],
        index: int
    ) -> None:
        if uncertainties_flat is None:
            return
        self._add_uncertainty_to_explanation(explanation, float(uncertainties_flat[index]))

    def _apply_optional_actual(
        self,
        explanation: Dict[str, Any],
        actual_values_flat: Optional[np.ndarray],
        index: int,
        prediction: float
    ) -> None:
        if actual_values_flat is None:
            return
        self._add_actual_to_explanation(explanation, float(actual_values_flat[index]), float(prediction))

    def _apply_optional_trend(
        self,
        explanation: Dict[str, Any],
        index: int,
        prediction: float,
        predictions_flat: np.ndarray
    ) -> None:
        if index <= 0:
            return
        self._add_trend_to_explanation(explanation, float(prediction), float(predictions_flat[index - 1]))
    
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
        predictions_flat = predictions.flatten()
        uncertainties_flat = self._normalize_uncertainties(predictions_flat, uncertainties)
        actual_values_flat = actual_values.flatten() if actual_values is not None else None
        
        explanations: List[Dict[str, Any]] = []
        
        for i, pred in enumerate(predictions_flat):
            explanation = self._build_explanation_base(
                index=i,
                prediction=pred,
                timestamps=timestamps,
                ticker_symbols=ticker_symbols
            )
            
            self._apply_optional_uncertainty(explanation, uncertainties_flat, i)
            self._apply_optional_actual(explanation, actual_values_flat, i, pred)
            self._apply_optional_trend(explanation, i, pred, predictions_flat)
            
            explanations.append(explanation)
        
        return explanations
    
    def _overall_trend_insight(self, predictions: np.ndarray) -> Optional[str]:
        if len(predictions) <= 1:
            return None

        overall_change = predictions[-1] - predictions[0]
        direction = "upward" if overall_change > 0 else "downward"

        if predictions[0] != 0:
            overall_percent = (overall_change / predictions[0]) * 100
            return f"Overall trend is {direction} with a {abs(overall_percent):.2f}% change."

        return f"Overall trend is {direction} with an absolute change of {overall_change:.4f}."

    def _uncertainty_insights(self, uncertainties: np.ndarray) -> List[str]:
        insights: List[str] = []

        avg_uncertainty = float(np.mean(uncertainties))
        max_uncertainty = float(np.max(uncertainties))
        min_uncertainty = float(np.min(uncertainties))
        insights.append(
            f"Average prediction uncertainty is {avg_uncertainty:.2f} (range: {min_uncertainty:.2f} to {max_uncertainty:.2f})."
        )

        high_uncertainty_count = int(np.sum(uncertainties > self.uncertainty_threshold))
        if high_uncertainty_count > 0:
            insights.append(f"High uncertainty detected in {high_uncertainty_count} predictions.")

        return insights

    def _performance_insights(self, predictions: np.ndarray, actual_values: np.ndarray) -> List[str]:
        insights: List[str] = []

        errors = predictions - actual_values
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        insights.append(f"Model performance: RMSE = {rmse:.4f}, MAE = {mae:.4f}.")

        direction_actual = np.sign(np.diff(actual_values, prepend=actual_values[0]))
        direction_pred = np.sign(np.diff(predictions, prepend=predictions[0]))
        directional_accuracy = np.mean(direction_actual == direction_pred) * 100
        insights.append(f"Directional accuracy: {directional_accuracy:.2f}%.")

        return insights

    def _signal_summary_insight(self, signals: Dict[str, np.ndarray]) -> Optional[str]:
        if "confident_direction" not in signals:
            return None

        confident_direction = signals["confident_direction"]
        buy_signals = int(np.sum(confident_direction > 0))
        sell_signals = int(np.sum(confident_direction < 0))
        hold_signals = int(np.sum(confident_direction == 0))

        return f"Trading signals: {buy_signals} buy, {sell_signals} sell, {hold_signals} hold."

    def _generate_insights(
        self,
        predictions: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        actual_values: Optional[np.ndarray] = None,
        signals: Optional[Dict[str, np.ndarray]] = None
    ) -> List[str]:
        """
        Generate insights from predictions and signals.
        
        Args:
            predictions: Model predictions
            uncertainties: Prediction uncertainties (optional)
            actual_values: Actual values for comparison (optional)
            signals: Trading signals (optional)
            
        Returns:
            List of insight strings
        """
        insights: List[str] = []
        # Ensure predictions are flattened
        predictions = predictions.flatten()

        trend_message = self._overall_trend_insight(predictions)
        if trend_message is not None:
            insights.append(trend_message)

        if uncertainties is not None:
            insights.extend(self._uncertainty_insights(uncertainties.flatten()))

        if actual_values is not None:
            insights.extend(self._performance_insights(predictions, actual_values.flatten()))

        if signals is not None:
            signal_message = self._signal_summary_insight(signals)
            if signal_message is not None:
                insights.append(signal_message)

        return insights
    
    def visualize_predictions(
        self,
        predictions: Optional[np.ndarray] = None,
        uncertainties: Optional[np.ndarray] = None,
        actual_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> Optional[plt.Figure]:
        """
        Visualize predictions with uncertainties and actual values.
        
        Args:
            predictions: Model predictions (optional, uses stored predictions if None)
            uncertainties: Prediction uncertainties (optional, uses stored uncertainties if None)
            actual_values: Actual values for comparison (optional)
            save_path: Path to save visualization (optional)
            show_plot: Whether to show the plot
            
        Returns:
            Matplotlib figure or None if no predictions available
        """
        # Use stored predictions if not provided
        predictions = predictions if predictions is not None else self.predictions
        uncertainties = uncertainties if uncertainties is not None else self.uncertainties
        
        if predictions is None:
            logger.warning("No predictions available for visualization")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Generate x-axis (indices)
        x = np.arange(len(predictions))
        ax.set_xlabel("Index")
        
        # Plot predictions
        ax.plot(x, predictions, label="Predictions", color="blue", linewidth=2)
        
        # Plot uncertainties as confidence bands
        if uncertainties is not None:
            lower_bound = predictions - uncertainties
            upper_bound = predictions + uncertainties
            ax.fill_between(x, lower_bound, upper_bound, alpha=0.3, color="blue", label="Uncertainty")
        
        # Plot actual values if provided
        if actual_values is not None:
            ax.plot(x, actual_values, label="Actual", color="green", linewidth=2, linestyle="--")
        
        # Formatting
        ax.set_ylabel("Value")
        ax.set_title("Predictions with Uncertainties")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
