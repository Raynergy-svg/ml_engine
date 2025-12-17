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


def fx_confidence_v1(
    *,
    signal: str,
    price: float,
    atr: float,
    spread_pips: float,
    max_spread_pips: float,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a conservative scalar confidence for FX Tier-1 gating.

    This is intentionally *fail-closed*: if any required inputs are missing or
    invalid, returns confidence=0.0.

    Until an FX-specific model head is wired in, this function combines:
    - actionable direction (buy/sell vs hold)
    - execution quality (spread vs configured max)
    - regime volatility proxy (ATR / price)

    `params` optionally tunes the mapping without code changes:
    - base_confidence: float
    - spread_soft_start_ratio: float in (0,1)
    - spread_penalty_max: float
    - atr_soft_start_pct: float (e.g., 0.001 = 0.10%)
    - atr_penalty_max: float

    Returns a dict: {confidence: float in [0,1], reasons: [str]}
    """
    reasons: list[str] = []
    try:
        params = params or {}
        sig = str(signal).strip().lower()
        if sig not in {"buy", "sell"}:
            return {"confidence": 0.0, "reasons": ["signal is not actionable"]}

        p = float(price)
        a = float(atr)
        sp = float(spread_pips)
        msp = float(max_spread_pips)
        if not np.isfinite(p) or p <= 0:
            return {"confidence": 0.0, "reasons": ["invalid price"]}
        if not np.isfinite(a) or a <= 0:
            return {"confidence": 0.0, "reasons": ["invalid ATR"]}
        if not np.isfinite(sp) or sp < 0:
            return {"confidence": 0.0, "reasons": ["invalid spread"]}
        if not np.isfinite(msp) or msp <= 0:
            return {"confidence": 0.0, "reasons": ["invalid max_spread_pips"]}

        base_conf = float(params.get("base_confidence", 0.85))
        spread_soft_start = float(params.get("spread_soft_start_ratio", 0.60))
        spread_penalty_max = float(params.get("spread_penalty_max", 0.20))
        atr_soft_start_pct = float(params.get("atr_soft_start_pct", 0.001))
        atr_penalty_max = float(params.get("atr_penalty_max", 0.15))

        # Base confidence for an actionable setup.
        conf = float(max(0.0, min(1.0, base_conf)))
        reasons.append(f"setup={sig}")

        # Spread penalty: only starts when spread approaches the cap.
        spread_ratio = float(max(0.0, sp / msp))
        if spread_ratio >= spread_soft_start:
            denom = max(1e-9, (1.0 - spread_soft_start))
            scaled = min(2.0, (spread_ratio - spread_soft_start) / denom)  # 0..2
            penalty = min(spread_penalty_max, spread_penalty_max * scaled)
            conf -= float(penalty)
            reasons.append(f"spread_ratio={spread_ratio:.2f}")
        else:
            reasons.append(f"spread_ok={spread_ratio:.2f}")

        # Volatility penalty: only kicks in above atr_soft_start_pct of price.
        vol_pct = float(a / p)
        if vol_pct > atr_soft_start_pct:
            scaled = min(2.0, (vol_pct - atr_soft_start_pct) / max(1e-9, atr_soft_start_pct))
            penalty = min(atr_penalty_max, 0.5 * atr_penalty_max * scaled)
            conf -= float(penalty)
            reasons.append(f"atr_pct={vol_pct:.3%}")
        else:
            reasons.append(f"atr_ok={vol_pct:.3%}")

        conf = float(max(0.0, min(1.0, conf)))
        return {"confidence": conf, "reasons": reasons}
    except Exception as e:
        return {"confidence": 0.0, "reasons": [f"confidence error: {e}"]}


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

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _extract_losses(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        losses: Dict[str, float] = {}
        for k in ("price_loss", "trend_loss", "risk_loss", "state_loss"):
            v = self._to_float(metrics.get(k))
            if v is not None:
                losses[k] = v
        return losses

    def _loss_balance_insights(
        self,
        losses: Dict[str, float],
        lag_threshold_ratio: float
    ) -> Dict[str, Any]:
        vals = np.array(list(losses.values()), dtype=float)
        baseline = float(np.median(vals)) if len(vals) else 0.0
        baseline = max(baseline, 1e-12)

        worst_name, worst_val = max(losses.items(), key=lambda kv: kv[1])

        lagging = (worst_val / baseline) >= float(lag_threshold_ratio)
        if lagging:
            message = f"{worst_name} is lagging (loss {worst_val:.4f} vs median {baseline:.4f})."
        else:
            message = "All heads look roughly balanced (losses within expected range)."

        return {
            "insight": message,
            "worst_name": worst_name,
            "lagging": lagging,
        }

    def _regression_insight(
        self,
        history: Optional[List[Dict[str, Any]]],
        worst_name: str,
        regression_ratio: float,
        min_history: int,
    ) -> Optional[str]:
        if not history or len(history) < int(min_history):
            return None

        prev = history[-2].get("val", {}) if isinstance(history[-2], dict) else {}
        cur = history[-1].get("val", {}) if isinstance(history[-1], dict) else {}

        prev_f = self._to_float(prev.get(worst_name))
        cur_f = self._to_float(cur.get(worst_name))
        if prev_f is None or cur_f is None or prev_f <= 0:
            return None

        if (cur_f / prev_f) >= float(regression_ratio):
            return f"{worst_name} regressed vs last epoch ({prev_f:.4f} -> {cur_f:.4f})."

        return None

    def _accuracy_insight(self, acc_f: Optional[float]) -> Optional[str]:
        if acc_f is None:
            return None
        if acc_f < 0.5:
            return f"state head accuracy is low ({acc_f:.2%}); check labels or class balance."
        return f"state head accuracy: {acc_f:.2%}."

    # Head descriptions for user-friendly diagnostics (only shown when needed)
    HEAD_TIPS: Dict[str, str] = {
        "price": "↑ `unified_head_loss_weights.price` or check feature scaling",
        "trend": "↑ `unified_head_loss_weights.trend` or increase `sequence_length`",
        "risk": "↑ `unified_head_loss_weights.risk` or adjust `risk_window`",
        "state": "↑ `unified_head_loss_weights.state` or check `state_classes`/class balance",
    }

    def summarize_head_health(
        self,
        head_metrics: Dict[str, Any],
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        lag_threshold_ratio: float = 1.25,
        regression_ratio: float = 1.10,
        min_history: int = 3,
        state_classes: int = 2,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Summarize multi-head training status - minimal output by default.

        Returns compact insights only when action is needed.
        Set verbose=True for detailed diagnostics.
        """
        metrics = dict(head_metrics or {})
        insights: List[str] = []

        losses = self._extract_losses(metrics)
        acc_f = self._to_float(metrics.get("state_acc"))

        lagging = False
        worst_name: Optional[str] = None
        regressed = False

        if losses:
            balance = self._loss_balance_insights(losses, lag_threshold_ratio)
            lagging = bool(balance["lagging"])
            worst_name = str(balance["worst_name"]) if balance.get("worst_name") else None

            # Check for regression (getting worse)
            if worst_name and history and len(history) >= 2:
                prev = history[-2].get("val", {}).get(worst_name)
                curr = losses.get(worst_name)
                if prev is not None and curr is not None:
                    if float(curr) > float(prev) * regression_ratio:
                        regressed = True

        # Determine if state head is healthy based on accuracy
        head = worst_name.replace("_loss", "") if worst_name else None
        accuracy_healthy = False
        if head == "state" and acc_f is not None:
            baseline = 1.0 / max(state_classes, 2)
            healthy_threshold = baseline + 0.20
            accuracy_healthy = acc_f >= healthy_threshold

        # Only output when something needs attention
        if lagging and not accuracy_healthy and head:
            if regressed:
                insights.append(f"⚠️  {head} regressing → {self.HEAD_TIPS.get(head, 'check config')}")
            elif verbose:
                insights.append(f"📉 {head} lagging → {self.HEAD_TIPS.get(head, 'check config')}")

        return {
            "head_metrics": metrics,
            "insights": insights,
            "lagging_head": head if (lagging and not accuracy_healthy) else None,
            "regressed": regressed,
            "state_acc_healthy": accuracy_healthy if head == "state" else None,
        }
    
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
