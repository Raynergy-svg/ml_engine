"""
ARIMA Hybrid Component for Ensemble Forecasting.

Integrates ARIMA (AutoRegressive Integrated Moving Average) as a stacking
component alongside Transformer and TCN for enhanced directional prediction.

Benefits:
- Captures long-term dependencies and autocorrelations
- Complements transformer's sequential pattern recognition
- Research shows LSTM+ARIMA ensembles enhance financial predictions
- Provides robustness across different market regimes

Usage:
    from src.models.arima_hybrid import ARIMAHybridComponent
    
    arima = ARIMAHybridComponent(order=(5, 1, 2))
    arima.fit(returns_series)
    arima_pred = arima.predict_direction(n_steps=1)
    
    # Hybrid ensemble prediction
    ensemble_pred = hybrid_ensemble_predict(
        transformer_pred, tcn_pred, arima_pred,
        weights=[0.5, 0.35, 0.15]
    )
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional statsmodels import with graceful fallback
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. ARIMA hybrid component disabled. Install with: pip install statsmodels")


class ARIMAHybridComponent:
    """
    ARIMA component for hybrid ensemble forecasting.
    
    Uses ARIMA to capture linear autocorrelations that neural networks
    may miss, providing complementary signal to Transformer/TCN.
    
    Args:
        order: ARIMA(p, d, q) order tuple (default: (5, 1, 2))
            - p: Autoregressive order (past values)
            - d: Differencing order (stationarity)
            - q: Moving average order (past errors)
        seasonal_order: Seasonal ARIMA order (P, D, Q, s) or None
        max_history: Maximum history length to fit (default: 500)
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 2),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        max_history: int = 500,
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMA. Install with: pip install statsmodels")
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.max_history = max_history
        self.model = None
        self.fitted_model = None
        self._last_train_data = None
    
    def fit(
        self,
        returns: np.ndarray,
        verbose: bool = False,
    ) -> 'ARIMAHybridComponent':
        """
        Fit ARIMA model on returns series.
        
        Args:
            returns: Array of log returns or simple returns
            verbose: Print fitting diagnostics
            
        Returns:
            self for method chaining
        """
        # Use recent history only (ARIMA doesn't need very long history)
        if len(returns) > self.max_history:
            returns = returns[-self.max_history:]
        
        # Store for refitting
        self._last_train_data = returns.copy()
        
        # Check stationarity
        if verbose:
            try:
                adf_result = adfuller(returns, autolag='AIC')
                logger.info(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
                if adf_result[1] > 0.05:
                    logger.warning("Series may be non-stationary (p > 0.05). Consider differencing.")
            except Exception as e:
                logger.warning(f"ADF test failed: {e}")
        
        try:
            # Fit ARIMA model
            self.model = ARIMA(
                returns,
                order=self.order,
                seasonal_order=self.seasonal_order,
            )
            self.fitted_model = self.model.fit()
            
            if verbose:
                logger.info(f"ARIMA{self.order} fitted. AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
                
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}. Using fallback.")
            self.fitted_model = None
        
        return self
    
    def predict_return(self, steps: int = 1) -> np.ndarray:
        """
        Predict future returns.
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Array of predicted returns
        """
        if self.fitted_model is None:
            # Fallback: return zero (no signal)
            return np.zeros(steps)
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return np.array(forecast)
        except Exception as e:
            logger.warning(f"ARIMA forecast failed: {e}")
            return np.zeros(steps)
    
    def predict_direction(self, steps: int = 1) -> np.ndarray:
        """
        Predict direction probabilities (UP=1, DOWN=0).
        
        Converts ARIMA return forecast to directional probability:
        - Positive forecast → probability > 0.5
        - Negative forecast → probability < 0.5
        - Magnitude affects confidence (larger = more confident)
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Array of direction probabilities [0, 1]
        """
        predicted_returns = self.predict_return(steps)
        
        # Convert to probability using sigmoid-like transformation
        # Scale factor makes typical FX returns map to reasonable probabilities
        scale = 100.0  # FX returns are typically 0.001-0.01
        probs = 1 / (1 + np.exp(-predicted_returns * scale))
        
        return np.clip(probs, 0.01, 0.99)
    
    def update(self, new_returns: np.ndarray) -> None:
        """
        Update model with new data (online learning).
        
        Refits ARIMA on expanded history for real-time adaptation.
        
        Args:
            new_returns: New return observations to incorporate
        """
        if self._last_train_data is None:
            self._last_train_data = new_returns
        else:
            self._last_train_data = np.concatenate([self._last_train_data, new_returns])
        
        # Refit with updated data
        self.fit(self._last_train_data)
    
    def save(self, path: Path) -> None:
        """Save fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'max_history': self.max_history,
            'last_train_data': self._last_train_data,
            'fitted_params': self.fitted_model.params if self.fitted_model else None,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"ARIMA model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ARIMAHybridComponent':
        """Load fitted model from disk."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        instance = cls(
            order=state['order'],
            seasonal_order=state['seasonal_order'],
            max_history=state['max_history'],
        )
        instance._last_train_data = state['last_train_data']
        
        # Refit to restore model
        if instance._last_train_data is not None:
            instance.fit(instance._last_train_data)
        
        logger.info(f"ARIMA model loaded from {path}")
        return instance


def hybrid_ensemble_predict(
    transformer_pred: np.ndarray,
    tcn_pred: np.ndarray,
    arima_pred: Optional[np.ndarray] = None,
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Combine predictions from multiple models into hybrid ensemble.
    
    Args:
        transformer_pred: Transformer direction probabilities [0, 1]
        tcn_pred: TCN direction probabilities [0, 1]
        arima_pred: ARIMA direction probabilities [0, 1] (optional)
        weights: Model weights [transformer, tcn, arima] (default: [0.5, 0.35, 0.15])
    
    Returns:
        Ensemble direction probabilities [0, 1]
    
    Example:
        >>> hybrid_pred = hybrid_ensemble_predict(
        ...     transformer_pred=np.array([0.65]),
        ...     tcn_pred=np.array([0.70]),
        ...     arima_pred=np.array([0.55]),
        ...     weights=[0.5, 0.35, 0.15]
        ... )
        >>> # = 0.5*0.65 + 0.35*0.70 + 0.15*0.55 = 0.6525
    """
    if arima_pred is not None and STATSMODELS_AVAILABLE:
        # 3-model ensemble
        if weights is None:
            weights = [0.50, 0.35, 0.15]  # Transformer, TCN, ARIMA
        
        # Normalize weights
        total = sum(weights[:3])
        w_trans, w_tcn, w_arima = [w / total for w in weights[:3]]
        
        ensemble = (
            w_trans * np.array(transformer_pred) +
            w_tcn * np.array(tcn_pred) +
            w_arima * np.array(arima_pred)
        )
    else:
        # 2-model ensemble (fallback when ARIMA unavailable)
        if weights is None:
            weights = [0.6, 0.4]  # Transformer, TCN
        
        # Normalize weights
        total = sum(weights[:2])
        w_trans, w_tcn = [w / total for w in weights[:2]]
        
        ensemble = (
            w_trans * np.array(transformer_pred) +
            w_tcn * np.array(tcn_pred)
        )
    
    return np.clip(ensemble, 0.0, 1.0)


def train_arima_for_pair(
    df: pd.DataFrame,
    instrument: str,
    order: Tuple[int, int, int] = (5, 1, 2),
    save_path: Optional[Path] = None,
) -> ARIMAHybridComponent:
    """
    Train ARIMA component for a specific currency pair.
    
    Args:
        df: DataFrame with OHLCV data
        instrument: Pair name (e.g., 'EUR_USD')
        order: ARIMA(p, d, q) order
        save_path: Path to save model (default: trained_data/models/{instrument}/arima.pkl)
    
    Returns:
        Fitted ARIMAHybridComponent
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available. Returning None for ARIMA.")
        return None
    
    logger.info(f"Training ARIMA{order} for {instrument}...")
    
    # Compute log returns
    close = df['close'].values
    returns = np.log(close[1:] / close[:-1])
    
    # Remove NaN/inf
    returns = returns[np.isfinite(returns)]
    
    # Fit ARIMA
    arima = ARIMAHybridComponent(order=order)
    arima.fit(returns, verbose=True)
    
    # Save if path provided
    if save_path is None:
        save_path = Path(f"trained_data/models/{instrument}/arima.pkl")
    
    arima.save(save_path)
    
    return arima


# Auto-selection of ARIMA order using AIC/BIC
def auto_select_arima_order(
    returns: np.ndarray,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    criterion: str = 'aic',
) -> Tuple[int, int, int]:
    """
    Automatically select optimal ARIMA order using information criteria.
    
    Args:
        returns: Return series
        max_p: Maximum AR order
        max_d: Maximum differencing order  
        max_q: Maximum MA order
        criterion: 'aic' or 'bic'
    
    Returns:
        Optimal (p, d, q) order tuple
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available. Returning default order (5, 1, 2).")
        return (5, 1, 2)
    
    best_score = np.inf
    best_order = (5, 1, 2)  # Default
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue  # Skip trivial model
                
                try:
                    model = ARIMA(returns, order=(p, d, q))
                    fitted = model.fit()
                    
                    score = fitted.aic if criterion == 'aic' else fitted.bic
                    
                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                        
                except Exception:
                    continue  # Skip invalid combinations
    
    logger.info(f"Auto-selected ARIMA{best_order} with {criterion.upper()}={best_score:.2f}")
    return best_order
