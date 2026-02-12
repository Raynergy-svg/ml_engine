"""
Optuna Hyperparameter Optimization for ML Engine.

This module provides automated hyperparameter tuning using Optuna's
advanced optimization algorithms with support for:
- Multi-objective optimization
- Pruning strategies (MedianPruner, SuccessiveHalving)
- Persistent storage (SQLite, MySQL, PostgreSQL)
- Distributed optimization
- Visualization and analysis tools

Example:
    >>> from src.training.optuna_tuner import OptunaConfig, OptunaTuner
    >>> config = OptunaConfig(n_trials=100, study_name="fx_trading_bot")
    >>> tuner = OptunaTuner(config)
    >>> best_params = tuner.optimize(train_fn, X_train, y_train)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check if Optuna is available
try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.study import Study
    from optuna.trial import Trial, FrozenTrial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


@dataclass
class OptunaConfig:
    """
    Configuration for Optuna hyperparameter optimization.
    
    Attributes:
        study_name: Name of the optimization study
        n_trials: Number of trials to run
        timeout: Maximum time in seconds (None for no limit)
        n_jobs: Number of parallel jobs (-1 for all cores)
        sampler: Sampling algorithm ('tpe', 'cmaes', 'random')
        pruner: Pruning strategy ('median', 'successive_halving', 'hyperband', None)
        storage: Database URL for persistent storage (None for in-memory)
        load_if_exists: Whether to load existing study or create new
        direction: Optimization direction ('minimize' or 'maximize')
        directions: For multi-objective optimization (overrides direction)
        metric_names: Names of metrics for multi-objective optimization
    """
    study_name: str = "ml_engine_optimization"
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    sampler: str = "tpe"  # 'tpe', 'cmaes', 'random'
    pruner: Optional[str] = "median"  # 'median', 'successive_halving', 'hyperband', None
    storage: Optional[str] = None  # e.g., "sqlite:///trained_data/optuna.db"
    load_if_exists: bool = True
    direction: str = "minimize"
    directions: Optional[List[str]] = None  # For multi-objective
    metric_names: Optional[List[str]] = None
    
    # Pruner-specific settings
    pruner_n_startup_trials: int = 5
    pruner_n_warmup_steps: int = 10
    pruner_interval_steps: int = 1
    
    # Sampler-specific settings
    sampler_n_startup_trials: int = 10
    sampler_seed: Optional[int] = None
    
    # Logging and visualization
    show_progress_bar: bool = True
    verbosity: int = 1  # 0=silent, 1=info, 2=debug


class OptunaTuner:
    """
    Hyperparameter optimizer using Optuna.
    
    This class wraps Optuna's study creation and optimization with sensible
    defaults for the ML Engine trading bot. It supports both single-objective
    and multi-objective optimization.
    
    Example:
        >>> def objective(trial):
        ...     lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        ...     batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        ...     # Train model and return validation loss
        ...     return val_loss
        >>> 
        >>> config = OptunaConfig(n_trials=50, study_name="model_tuning")
        >>> tuner = OptunaTuner(config)
        >>> best_params = tuner.optimize(objective)
    """
    
    def __init__(self, config: OptunaConfig):
        """
        Initialize Optuna tuner.
        
        Args:
            config: Optuna configuration
            
        Raises:
            ImportError: If Optuna is not installed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Install with: pip install optuna"
            )
        
        self.config = config
        self.study: Optional[Study] = None
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure Optuna logging based on verbosity setting."""
        if self.config.verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif self.config.verbosity == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """
        Create Optuna sampler based on configuration.
        
        Returns:
            Configured sampler instance
        """
        sampler_name = self.config.sampler.lower()
        seed = self.config.sampler_seed
        
        if sampler_name == "tpe":
            return TPESampler(
                n_startup_trials=self.config.sampler_n_startup_trials,
                seed=seed,
            )
        elif sampler_name == "cmaes":
            return CmaEsSampler(
                n_startup_trials=self.config.sampler_n_startup_trials,
                seed=seed,
            )
        elif sampler_name == "random":
            return RandomSampler(seed=seed)
        else:
            logger.warning(f"Unknown sampler '{sampler_name}', using TPE")
            return TPESampler(seed=seed)
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """
        Create Optuna pruner based on configuration.
        
        Returns:
            Configured pruner instance or None
        """
        if self.config.pruner is None:
            return None
        
        pruner_name = self.config.pruner.lower()
        
        if pruner_name == "median":
            return MedianPruner(
                n_startup_trials=self.config.pruner_n_startup_trials,
                n_warmup_steps=self.config.pruner_n_warmup_steps,
                interval_steps=self.config.pruner_interval_steps,
            )
        elif pruner_name == "successive_halving":
            return SuccessiveHalvingPruner()
        elif pruner_name == "hyperband":
            return HyperbandPruner()
        else:
            logger.warning(f"Unknown pruner '{pruner_name}', using Median")
            return MedianPruner()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists for database files."""
        if self.config.storage and self.config.storage.startswith("sqlite:///"):
            db_path = Path(self.config.storage.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create_study(self) -> Study:
        """
        Create or load an Optuna study.
        
        Returns:
            Optuna study instance
        """
        self._ensure_storage_dir()
        
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        # Determine optimization direction(s)
        if self.config.directions is not None:
            # Multi-objective optimization
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                load_if_exists=self.config.load_if_exists,
                sampler=sampler,
                pruner=pruner,
                directions=self.config.directions,
            )
        else:
            # Single-objective optimization
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                load_if_exists=self.config.load_if_exists,
                sampler=sampler,
                pruner=pruner,
                direction=self.config.direction,
            )
        
        self.study = study
        logger.info(f"Created study '{self.config.study_name}' with {len(study.trials)} existing trials")
        return study
    
    def optimize(
        self,
        objective: Callable[[Trial], float | List[float]],
        callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective: Objective function that takes a trial and returns metric(s)
            callbacks: Optional callbacks to call after each trial
            
        Returns:
            Dictionary containing:
                - best_params: Best hyperparameters found
                - best_value(s): Best metric value(s)
                - n_trials: Number of trials completed
                - study: The Optuna study object
                
        Example:
            >>> def objective(trial):
            ...     lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            ...     # Train and validate
            ...     return validation_loss
            >>> 
            >>> tuner = OptunaTuner(config)
            >>> results = tuner.optimize(objective)
            >>> print(f"Best params: {results['best_params']}")
        """
        if self.study is None:
            self.create_study()
        
        logger.info(f"Starting optimization with {self.config.n_trials} trials")
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.config.show_progress_bar,
            callbacks=callbacks,
        )
        
        # Extract results
        if self.config.directions is None:
            # Single-objective
            best_trial = self.study.best_trial
            results = {
                'best_params': best_trial.params,
                'best_value': best_trial.value,
                'n_trials': len(self.study.trials),
                'study': self.study,
                'best_trial': best_trial,
            }
            logger.info(f"Optimization complete. Best value: {best_trial.value:.6f}")
            logger.info(f"Best params: {best_trial.params}")
        else:
            # Multi-objective
            best_trials = self.study.best_trials
            results = {
                'best_trials': best_trials,
                'n_trials': len(self.study.trials),
                'study': self.study,
                'pareto_front': [(t.params, t.values) for t in best_trials],
            }
            logger.info(f"Optimization complete. Found {len(best_trials)} Pareto-optimal solutions")
        
        return results
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best hyperparameters from completed study.
        
        Returns:
            Dictionary of best hyperparameters
            
        Raises:
            ValueError: If study hasn't been created or optimized
        """
        if self.study is None:
            raise ValueError("No study created. Call optimize() first")
        
        if len(self.study.trials) == 0:
            raise ValueError("No trials completed")
        
        return self.study.best_trial.params
    
    def get_study_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the optimization study.
        
        Returns:
            Dictionary with study statistics
        """
        if self.study is None:
            return {}
        
        trials = self.study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
        
        stats = {
            'n_trials': len(trials),
            'n_completed': len(completed_trials),
            'n_pruned': len(pruned_trials),
            'n_failed': len(failed_trials),
        }
        
        if completed_trials and self.config.directions is None:
            values = [t.value for t in completed_trials]
            stats.update({
                'best_value': self.study.best_value,
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'min_value': np.min(values),
                'max_value': np.max(values),
            })
        
        return stats
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save plot (if None, displays interactively)
        """
        if self.study is None:
            raise ValueError("No study created")
        
        try:
            import optuna.visualization as vis
            fig = vis.plot_optimization_history(self.study)
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Saved optimization history to {save_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not available for visualization")
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        Plot parameter importances.
        
        Args:
            save_path: Path to save plot (if None, displays interactively)
        """
        if self.study is None:
            raise ValueError("No study created")
        
        try:
            import optuna.visualization as vis
            fig = vis.plot_param_importances(self.study)
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Saved parameter importances to {save_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not available for visualization")


def create_transformer_objective(
    train_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Callable[[Trial], float]:
    """
    Create objective function for Transformer model hyperparameter optimization.
    
    Args:
        train_fn: Training function that accepts params dict and returns val_loss
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        fixed_params: Parameters to keep fixed during optimization
        
    Returns:
        Objective function for Optuna
        
    Example:
        >>> def train_model(params):
        ...     # Build and train model with params
        ...     return val_loss
        >>> 
        >>> objective = create_transformer_objective(
        ...     train_model, X_train, y_train, X_val, y_val
        ... )
        >>> tuner = OptunaTuner(config)
        >>> results = tuner.optimize(objective)
    """
    fixed_params = fixed_params or {}
    
    def objective(trial: Trial) -> float:
        # Transformer architecture parameters
        params = {
            'd_model': trial.suggest_categorical('d_model', [16, 32, 64, 128]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        }
        
        # Merge with fixed parameters
        params.update(fixed_params)
        
        # Train model
        val_loss = train_fn(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            trial=trial,  # For pruning support
        )
        
        return val_loss
    
    return objective


if __name__ == "__main__":
    # Example usage
    if OPTUNA_AVAILABLE:
        # Simple example
        def dummy_objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return (x - 2) ** 2 + (y + 3) ** 2
        
        config = OptunaConfig(
            study_name="test_optimization",
            n_trials=20,
            direction="minimize",
        )
        
        tuner = OptunaTuner(config)
        results = tuner.optimize(dummy_objective)
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Best value: {results['best_value']:.6f}")
        print(f"Study statistics: {tuner.get_study_statistics()}")
