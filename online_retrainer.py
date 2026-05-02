"""
Online Retrainer - Connects drift detection to actual model retraining.

This module bridges the gap between:
1. DriftDetectionManager (detects when retraining is needed)
2. OnlineLearner (accumulates trade outcomes in replay buffer)
3. Gate model trainers (XGBoost, RF, Ridge)

Key features:
- Uses accumulated replay buffer data (not fresh OANDA data)
- Performs lightweight in-process retraining of sklearn gate models
- Has cooldown protection to prevent infinite retrain loops
- Supports incremental learning by mixing replay data with recent market data
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --- W&B Training Control Plane -------------------------------------------
# Imported lazily inside methods so that retrainer continues to work when
# wandb is uninstalled or networkless. The control plane exposes a single
# function ``pull_config`` per head + ``log_run`` that gracefully degrades
# (offline mode → ./wandb/, no API key → offline, no install → no-op).
# Mapping from the local model labels we retrain here to control-plane head
# names. Note: incremental online retraining trains the GATE models
# (xgboost/momentum, rf/risk fallback, ridge/confidence). Direction +
# regimes + meta-labeler are heavy retrains owned by the manual scripts.
_HEAD_BY_LOCAL_MODEL: Dict[str, str] = {
    "xgboost": "momentum",
    "rf": "risk",
    "ridge": "confidence",
}


def _safe_pull_config(head: str) -> Dict[str, Any]:
    """Pull the W&B control-plane config for ``head`` (offline-safe)."""
    try:
        from src.training.wandb_control_plane import pull_config
        cfg = pull_config(head)
        return dict(cfg or {})
    except Exception as exc:
        logger.warning(
            "Control-plane pull failed for head=%s: %s — auto-tweaking won't apply this cycle",
            head, exc,
        )
        return {}


def _safe_log_run(
    head: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: Optional[List[Path]] = None,
    run_name: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Log a retrain run via the control plane (offline-safe)."""
    try:
        from src.training.wandb_control_plane import log_run
        return log_run(
            head=head,
            config=config,
            metrics=metrics,
            artifacts=artifacts,
            run_name=run_name,
            extra_config=extra_config,
            tags=["auto_retrain"],
        )
    except Exception as exc:
        logger.debug("Control-plane log_run failed for %s: %s", head, exc)
        return None


def _hp_from_cfg(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    try:
        return (cfg.get("hyperparameters") or {}).get(key, default)
    except Exception:
        return default


@dataclass
class RetrainConfig:
    """Configuration for online retraining."""
    
    # Cooldown settings
    cooldown_minutes: int = 60  # Minimum time between retrains
    max_retrains_per_day: int = 3  # Daily limit
    
    # Data requirements
    min_samples_for_retrain: int = 50  # Minimum replay samples needed
    min_accuracy_drop: float = 0.05  # 5% accuracy drop triggers retrain
    
    # Training parameters
    epochs_per_retrain: int = 1  # Sklearn models don't have epochs, but used for TF if extended
    validation_split: float = 0.2  # Use 20% of replay data for validation
    
    # Model selection - which models to retrain
    retrain_xgboost: bool = True
    retrain_rf: bool = True
    retrain_ridge: bool = True
    retrain_transformer: bool = False  # Transformer is expensive, usually skip
    
    # Persistence
    state_file: str = "trained_data/online_retrain_state.json"


@dataclass
class RetrainState:
    """Tracks retraining state for cooldown management."""
    
    last_retrain_time: Optional[datetime] = None
    retrains_today: int = 0
    last_retrain_date: Optional[str] = None
    total_retrains: int = 0
    last_retrain_reason: str = ""
    last_retrain_metrics: Dict[str, Any] = field(default_factory=dict)
    retrain_history: List[Dict[str, Any]] = field(default_factory=list)


class OnlineRetrainer:
    """
    Handles incremental retraining of gate models using replay buffer data.
    
    Usage:
        # Initialize with data sources
        retrainer = OnlineRetrainer()
        
        # Connect to drift detection
        drift_manager.set_retrain_callback(retrainer.trigger_retrain)
        
        # Or check manually
        if retrainer.should_retrain():
            result = retrainer.trigger_retrain()
    """
    
    def __init__(
        self,
        config: Optional[RetrainConfig] = None,
        model_dir: str = "trained_data/models",
    ):
        self.config = config or RetrainConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self._state = RetrainState()
        self._load_state()
        
        # Thread safety
        self._retrain_lock = Lock()
        self._is_retraining = False
        
        logger.debug(f"OnlineRetrainer initialized (model_dir={self.model_dir})")
    
    def _load_state(self) -> None:
        """Load state from disk."""
        state_path = Path(self.config.state_file)
        if not state_path.exists():
            return
        
        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
            
            self._state.last_retrain_time = (
                datetime.fromisoformat(data['last_retrain_time'])
                if data.get('last_retrain_time') else None
            )
            self._state.retrains_today = data.get('retrains_today', 0)
            self._state.last_retrain_date = data.get('last_retrain_date')
            self._state.total_retrains = data.get('total_retrains', 0)
            self._state.last_retrain_reason = data.get('last_retrain_reason', '')
            self._state.last_retrain_metrics = data.get('last_retrain_metrics', {})
            self._state.retrain_history = data.get('retrain_history', [])[-100:]  # Keep last 100
            
            logger.info(f"Loaded retrain state: {self._state.total_retrains} total retrains")
            
        except Exception as e:
            logger.warning(f"Failed to load retrain state: {e}")
    
    def _save_state(self) -> None:
        """Save state to disk."""
        state_path = Path(self.config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'last_retrain_time': (
                self._state.last_retrain_time.isoformat()
                if self._state.last_retrain_time else None
            ),
            'retrains_today': self._state.retrains_today,
            'last_retrain_date': self._state.last_retrain_date,
            'total_retrains': self._state.total_retrains,
            'last_retrain_reason': self._state.last_retrain_reason,
            'last_retrain_metrics': self._state.last_retrain_metrics,
            'retrain_history': self._state.retrain_history[-100:],
            'saved_at': datetime.utcnow().isoformat(),
        }
        
        try:
            with open(state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save retrain state: {e}")
    
    def can_retrain(self) -> Tuple[bool, str]:
        """
        Check if retraining is allowed based on cooldown and daily limits.
        
        Returns:
            (can_retrain, reason)
        """
        now = datetime.utcnow()
        today = now.strftime('%Y-%m-%d')
        
        # Reset daily counter if new day
        if self._state.last_retrain_date != today:
            self._state.retrains_today = 0
            self._state.last_retrain_date = today
        
        # Check daily limit
        if self._state.retrains_today >= self.config.max_retrains_per_day:
            return False, f"Daily limit reached ({self.config.max_retrains_per_day})"
        
        # Check cooldown
        if self._state.last_retrain_time is not None:
            elapsed = (now - self._state.last_retrain_time).total_seconds() / 60
            remaining = self.config.cooldown_minutes - elapsed
            if remaining > 0:
                return False, f"Cooldown active ({remaining:.0f} min remaining)"
        
        # Check if already retraining
        if self._is_retraining:
            return False, "Retrain already in progress"
        
        return True, "Ready to retrain"
    
    def trigger_retrain(
        self,
        X_replay: Optional[np.ndarray] = None,
        y_replay: Optional[np.ndarray] = None,
        reason: str = "drift_detected",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Trigger incremental retraining of gate models.
        
        Args:
            X_replay: Feature matrix from replay buffer (optional, will load from files)
            y_replay: Labels from replay buffer (optional, will load from files)
            reason: Why retraining was triggered
            force: Bypass cooldown checks
            
        Returns:
            Dictionary with retrain results
        """
        result = {
            'triggered_at': datetime.utcnow().isoformat(),
            'reason': reason,
            'status': 'pending',
            'models_retrained': [],
            'metrics': {},
            'duration_seconds': 0,
        }
        
        # Check cooldown (unless forced)
        if not force:
            can_train, block_reason = self.can_retrain()
            if not can_train:
                result['status'] = 'blocked'
                result['blocked_reason'] = block_reason
                logger.info(f"🔄 Retrain blocked: {block_reason}")
                return result
        
        # Acquire lock
        if not self._retrain_lock.acquire(blocking=False):
            result['status'] = 'blocked'
            result['blocked_reason'] = "Another retrain is in progress"
            return result
        
        self._is_retraining = True
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Starting incremental retrain: {reason}")
            
            # Load replay data if not provided
            if X_replay is None or y_replay is None:
                X_replay, y_replay = self._load_replay_data()
            
            if X_replay is None or len(X_replay) < self.config.min_samples_for_retrain:
                result['status'] = 'skipped'
                result['skipped_reason'] = (
                    f"Insufficient data: {len(X_replay) if X_replay is not None else 0} samples "
                    f"(need {self.config.min_samples_for_retrain})"
                )
                logger.info(f"⚠️ Retrain skipped: {result['skipped_reason']}")
                return result
            
            logger.info(f"  Replay data: {len(X_replay)} samples, {X_replay.shape[-1]} features")
            
            # Split into train/val
            n = len(X_replay)
            n_val = max(1, int(n * self.config.validation_split))
            indices = np.random.permutation(n)
            
            X_train = X_replay[indices[n_val:]]
            y_train = y_replay[indices[n_val:]]
            X_val = X_replay[indices[:n_val]]
            y_val = y_replay[indices[:n_val]]
            
            logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Retrain each enabled model
            if self.config.retrain_xgboost:
                try:
                    xgb_metrics = self._retrain_xgboost(X_train, y_train, X_val, y_val)
                    result['models_retrained'].append('xgboost')
                    result['metrics']['xgboost'] = xgb_metrics
                    logger.info("  ✓ XGBoost retrained")
                except Exception as e:
                    logger.warning(f"  ✗ XGBoost retrain failed: {e}")
                    result['metrics']['xgboost'] = {'error': str(e)}
            
            if self.config.retrain_rf:
                try:
                    rf_metrics = self._retrain_rf(X_train, y_train, X_val, y_val)
                    result['models_retrained'].append('rf')
                    result['metrics']['rf'] = rf_metrics
                    logger.info("  ✓ RandomForest retrained")
                except Exception as e:
                    logger.warning(f"  ✗ RF retrain failed: {e}")
                    result['metrics']['rf'] = {'error': str(e)}
            
            if self.config.retrain_ridge:
                try:
                    ridge_metrics = self._retrain_ridge(X_train, y_train, X_val, y_val)
                    result['models_retrained'].append('ridge')
                    result['metrics']['ridge'] = ridge_metrics
                    logger.info("  ✓ Ridge retrained")
                except Exception as e:
                    logger.warning(f"  ✗ Ridge retrain failed: {e}")
                    result['metrics']['ridge'] = {'error': str(e)}
            
            # Update state
            duration = time.time() - start_time
            result['status'] = 'completed' if result['models_retrained'] else 'failed'
            result['duration_seconds'] = duration
            
            # Update state tracking
            self._state.last_retrain_time = datetime.utcnow()
            self._state.retrains_today += 1
            self._state.total_retrains += 1
            self._state.last_retrain_reason = reason
            self._state.last_retrain_metrics = result['metrics']
            
            # Add to history
            self._state.retrain_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'reason': reason,
                'status': result['status'],
                'models': result['models_retrained'],
                'duration': duration,
            })
            
            self._save_state()
            
            logger.info(
                f"✅ Retrain completed in {duration:.1f}s: "
                f"{len(result['models_retrained'])} models updated"
            )
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"❌ Retrain failed: {e}")
        
        finally:
            self._is_retraining = False
            self._retrain_lock.release()
        
        return result
    
    def _load_replay_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load replay buffer data from online learning storage.
        
        Returns:
            (X, y) arrays or (None, None) if no data
        """
        X_parts = []
        y_parts = []
        
        # Try loading from OnlineLearner buffer
        trade_buffer_path = Path("trained_data/online_learning/trade_buffer.json")
        if trade_buffer_path.exists():
            try:
                with open(trade_buffer_path, 'r') as f:
                    data = json.load(f)
                
                trades = data.get('trades', [])
                if trades:
                    # Find consistent feature size
                    feature_sizes = [len(t['features']) for t in trades if t.get('features')]
                    if feature_sizes:
                        from collections import Counter
                        target_size = Counter(feature_sizes).most_common(1)[0][0]
                        
                        valid_trades = [
                            t for t in trades 
                            if t.get('features') and len(t['features']) == target_size
                        ]
                        
                        if valid_trades:
                            X = np.array([t['features'] for t in valid_trades])
                            y = np.array([t['actual_outcome'] for t in valid_trades])
                            X_parts.append(X)
                            y_parts.append(y)
                            logger.info(f"  Loaded {len(X)} samples from trade buffer")
            except Exception as e:
                logger.warning(f"Failed to load trade buffer: {e}")
        
        # Try loading from drift manager replay buffer
        drift_state_path = Path("trained_data/drift_detection/drift_state.json")
        if drift_state_path.exists():
            try:
                with open(drift_state_path, 'r') as f:
                    data = json.load(f)
                
                # The drift state doesn't directly store X/y, but we can check if baseline exists
                if data.get('baseline_feature_stats'):
                    logger.info("  Drift baseline exists (replay data in memory)")
            except Exception as e:
                logger.warning(f"Failed to load drift state: {e}")
        
        if not X_parts:
            return None, None
        
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        
        return X, y
    
    def _retrain_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Retrain XGBoost momentum model incrementally."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed")
        
        from sklearn.preprocessing import StandardScaler
        
        # Load existing model if available
        model_path = self.model_dir / "xgb_momentum.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    existing_data = pickle.load(f)
                _ = existing_data.get('momentum_model')
            except Exception:
                pass
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # For binary classification from replay buffer
        # y is typically 0/1 (loss/profit)
        if y_train.ndim == 1:
            # Convert to momentum-style targets
            # momentum_score = probability of profit
            y_train_momentum = y_train.astype(float)
            y_val_momentum = y_val.astype(float)
        else:
            y_train_momentum = y_train[:, 0]
            y_val_momentum = y_val[:, 0]
        
        # Pull W&B control-plane config for the momentum head.
        cp_cfg = _safe_pull_config("momentum")
        n_estimators = int(_hp_from_cfg(cp_cfg, "n_estimators", 50))
        max_depth = int(_hp_from_cfg(cp_cfg, "max_depth", 4))
        learning_rate = float(_hp_from_cfg(cp_cfg, "learning_rate", 0.1))

        # Train new model — HPs sourced from control plane (with sane defaults).
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(
            X_train_scaled, y_train_momentum,
            eval_set=[(X_val_scaled, y_val_momentum)],
            verbose=False,
        )

        # Evaluate
        pred = model.predict(X_val_scaled)
        mae = float(np.mean(np.abs(pred - y_val_momentum)))

        # Save
        save_data = {
            'momentum_model': model,
            'scaler': scaler,
            'n_features': X_train.shape[-1],
            'retrained_at': datetime.utcnow().isoformat(),
            'method': 'incremental',
        }

        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)

        # Log to W&B (auto-retrain source).
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _safe_log_run(
            head="momentum",
            config=cp_cfg,
            metrics={
                "val_mae": mae,
                "n_train": int(len(y_train_momentum)),
                "n_val": int(len(y_val_momentum)),
            },
            artifacts=[model_path] if model_path.exists() else None,
            run_name=f"auto_momentum_{ts}",
            extra_config={"source": "auto_retrain", "method": "incremental"},
        )

        return {'momentum_mae': mae}
    
    def _retrain_rf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Retrain RandomForest risk model incrementally."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # For binary outcomes, we predict "risk" as inverse of profit probability
        if y_train.ndim == 1:
            # Risk = 1 - profit_prob
            y_train_risk = 1.0 - y_train.astype(float)
            y_val_risk = 1.0 - y_val.astype(float)
        else:
            y_train_risk = y_train[:, 0]
            y_val_risk = y_val[:, 0]
        
        # Pull W&B control-plane config for the risk head.
        cp_cfg = _safe_pull_config("risk")
        n_estimators = int(_hp_from_cfg(cp_cfg, "n_estimators", 50))
        max_depth = int(_hp_from_cfg(cp_cfg, "max_depth", 6))

        # Train new model — HPs sourced from control plane.
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train_risk)

        # Evaluate
        pred = model.predict(X_val_scaled)
        mae = float(np.mean(np.abs(pred - y_val_risk)))

        # Save
        model_path = self.model_dir / "rf_risk.pkl"
        save_data = {
            'model': model,
            'scaler': scaler,
            'n_features': X_train.shape[-1],
            'retrained_at': datetime.utcnow().isoformat(),
            'method': 'incremental',
        }

        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)

        # Log to W&B.
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _safe_log_run(
            head="risk",
            config=cp_cfg,
            metrics={
                "val_mae": mae,
                "n_train": int(len(y_train_risk)),
                "n_val": int(len(y_val_risk)),
            },
            artifacts=[model_path],
            run_name=f"auto_risk_{ts}",
            extra_config={"source": "auto_retrain", "method": "incremental"},
        )

        return {'risk_mae': mae}
    
    def _retrain_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Retrain Ridge confidence model incrementally."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Confidence = scaled profit probability (0-100)
        if y_train.ndim == 1:
            y_train_conf = y_train.astype(float) * 100
            y_val_conf = y_val.astype(float) * 100
        else:
            y_train_conf = y_train[:, 0] * 100
            y_val_conf = y_val[:, 0] * 100
        
        # Pull W&B control-plane config for the confidence head.
        # Ridge is the lightweight fallback for incremental retrain; the heavy
        # LightGBM confidence retrain is handled by manual scripts.
        cp_cfg = _safe_pull_config("confidence")

        # Ridge alpha isn't on the strategic exposed list — keep at default.
        # Logging the run still gives operators visibility into auto retrain
        # cadence + drift outcomes.
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train_conf)

        # Evaluate
        pred = model.predict(X_val_scaled)
        mae = float(np.mean(np.abs(pred - y_val_conf)))

        # Save
        model_path = self.model_dir / "ridge_confidence.pkl"
        save_data = {
            'model': model,
            'scaler': scaler,
            'n_features': X_train.shape[-1],
            'retrained_at': datetime.utcnow().isoformat(),
            'method': 'incremental',
        }

        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)

        # Log to W&B.
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _safe_log_run(
            head="confidence",
            config=cp_cfg,
            metrics={
                "val_mae": mae,
                "n_train": int(len(y_train_conf)),
                "n_val": int(len(y_val_conf)),
            },
            artifacts=[model_path],
            run_name=f"auto_confidence_{ts}",
            extra_config={"source": "auto_retrain", "method": "incremental_ridge_fallback"},
        )

        return {'confidence_mae': mae}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current retrainer status."""
        can_train, reason = self.can_retrain()
        
        return {
            'can_retrain': can_train,
            'reason': reason,
            'is_retraining': self._is_retraining,
            'total_retrains': self._state.total_retrains,
            'retrains_today': self._state.retrains_today,
            'max_per_day': self.config.max_retrains_per_day,
            'cooldown_minutes': self.config.cooldown_minutes,
            'last_retrain': (
                self._state.last_retrain_time.isoformat()
                if self._state.last_retrain_time else None
            ),
            'last_reason': self._state.last_retrain_reason,
            'last_metrics': self._state.last_retrain_metrics,
        }


# =============================================================================
# INTEGRATION HELPER - Creates retrain callback for DriftDetectionManager
# =============================================================================

def create_retrain_callback(
    retrainer: Optional[OnlineRetrainer] = None,
    market_intel: Optional[Any] = None,  # MarketIntelligence instance
) -> Callable[[], Dict[str, Any]]:
    """
    Create a callback function for DriftDetectionManager.
    
    This callback:
    1. Gets replay data from MarketIntelligence
    2. Triggers OnlineRetrainer with that data
    3. Marks model as updated in online learner
    
    Args:
        retrainer: OnlineRetrainer instance (created if None)
        market_intel: MarketIntelligence instance for replay data
        
    Returns:
        Callback function suitable for drift_manager.set_retrain_callback()
    """
    if retrainer is None:
        retrainer = OnlineRetrainer()
    
    def callback() -> Dict[str, Any]:
        """Drift-triggered retrain callback."""
        
        # Get replay data from market intelligence
        X, y = None, None
        if market_intel is not None:
            try:
                X, y = market_intel.get_replay_data()
                logger.info(f"Got {len(X) if X is not None else 0} samples from replay buffer")
            except Exception as e:
                logger.warning(f"Failed to get replay data: {e}")
        
        # Trigger retrain
        result = retrainer.trigger_retrain(
            X_replay=X,
            y_replay=y,
            reason="drift_detected",
        )
        
        # Mark model as updated if successful
        if result['status'] == 'completed' and market_intel is not None:
            try:
                market_intel.mark_model_updated()
            except Exception as e:
                logger.warning(f"Failed to mark model updated: {e}")
        
        return result
    
    return callback


# Example usage / testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create retrainer
    retrainer = OnlineRetrainer()
    
    # Check status
    status = retrainer.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    # Create fake replay data for testing
    np.random.seed(42)
    X_test = np.random.randn(100, 50)
    y_test = np.random.randint(0, 2, 100)
    
    # Trigger retrain (forced to bypass cooldown)
    result = retrainer.trigger_retrain(
        X_replay=X_test,
        y_replay=y_test,
        reason="test",
        force=True,
    )
    
    print(f"\nRetrain result: {json.dumps(result, indent=2, default=str)}")
