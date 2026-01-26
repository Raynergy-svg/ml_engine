"""
Enterprise Training Integration Module

Integrates enterprise_training.py with modular_trainers.py for
professional-grade ML training with:

- Full experiment tracking (MLflow)
- Walk-forward cross-validation
- Statistical validation with bootstrap CI
- Resource monitoring
- Automatic report generation
- Model comparison dashboards

Usage:
    from enterprise_integration import run_enterprise_ensemble_training
    
    results = run_enterprise_ensemble_training(
        data=modular_data,
        config=config,
        instrument="EUR_USD",
        granularity="H1",
        enable_cross_validation=True,
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from enterprise_training import (
    ExperimentConfig,
    ExperimentTracker,
    TrainingExperiment,
    WalkForwardConfig,
    WalkForwardValidator,
    StatisticalValidator,
    ResourceMonitor,
    ReproducibilityManager,
    DataValidator,
    DataValidationResult,
    setup_enterprise_logging,
    MLFLOW_AVAILABLE,
    SCIPY_AVAILABLE,
)

# Import modular trainers
try:
    from modular_trainers import TrainerConfig
    from modular_trainers import TransformerDirectionTrainer
    from modular_trainers import XGBoostTrainer
    from modular_trainers import RandomForestTrainer
    from modular_trainers import RidgeTrainer
    from modular_trainers import HistGradientBoostingDirectionTrainer
    from modular_trainers import train_all_modular
    MODULAR_TRAINERS_AVAILABLE = True
except ImportError:
    MODULAR_TRAINERS_AVAILABLE = False
    TrainerConfig = None
    TransformerDirectionTrainer = None
    logging.warning("modular_trainers not available")


logger = logging.getLogger("enterprise_integration")


# =============================================================================
# ENTERPRISE TRAINING CONFIGURATION
# =============================================================================

@dataclass
class EnterpriseConfig:
    """Extended configuration for enterprise training."""
    
    # Experiment metadata
    experiment_name: str = "ensemble_training"
    model_type: str = "ensemble"
    instrument: str = "EUR_USD"
    granularity: str = "H1"
    description: str = ""
    
    # Base trainer config
    trainer_config: Optional[TrainerConfig] = None
    
    # Enterprise features
    enable_mlflow: bool = True
    enable_cross_validation: bool = True
    enable_statistical_validation: bool = True
    enable_resource_monitoring: bool = True
    enable_data_validation: bool = True
    
    # Walk-forward CV settings
    cv_n_splits: int = 5
    cv_train_period: int = 2000
    cv_test_period: int = 500
    cv_gap: int = 24  # Gap to prevent look-ahead bias
    
    # Bootstrap CI settings
    bootstrap_n_samples: int = 1000
    bootstrap_confidence: float = 0.95
    
    # Paths
    save_dir: str = "trained_data/models"
    log_dir: str = "trained_data/logs"
    experiment_dir: str = "trained_data/experiments"
    mlflow_uri: str = "trained_data/mlruns"
    
    # Training settings
    warm_start: bool = False
    train_histgb: bool = True  # For hybrid voting


# =============================================================================
# TRAINING RESULTS
# =============================================================================

@dataclass  
class ModelMetrics:
    """Metrics for a single model."""
    model_name: str
    val_accuracy: float = 0.0
    val_loss: float = 0.0
    balanced_accuracy: float = 0.0
    
    # Additional metrics (model-specific)
    mae: float = 0.0
    r2_score: float = 0.0
    auc: float = 0.0
    
    # Cross-validation results
    cv_mean: float = 0.0
    cv_std: float = 0.0
    cv_ci_lower: float = 0.0
    cv_ci_upper: float = 0.0
    
    # Bootstrap CI
    bootstrap_mean: float = 0.0
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0
    
    # Training info
    n_params: int = 0
    training_time_seconds: float = 0.0
    epochs_trained: int = 0


@dataclass
class EnsembleResults:
    """Complete results from enterprise ensemble training."""
    experiment_id: str
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Model metrics
    transformer_metrics: Optional[ModelMetrics] = None
    xgboost_metrics: Optional[ModelMetrics] = None
    rf_metrics: Optional[ModelMetrics] = None
    ridge_metrics: Optional[ModelMetrics] = None
    histgb_metrics: Optional[ModelMetrics] = None
    
    # Ensemble-level metrics
    ensemble_agreement_rate: float = 0.0
    hybrid_voting_accuracy: float = 0.0
    
    # Cross-validation summary
    cv_results: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical significance
    statistical_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Data validation
    data_validation: Optional[DataValidationResult] = None
    
    # Resource usage
    resource_summary: Dict[str, float] = field(default_factory=dict)
    
    # Config used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Environment info
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Artifacts
    model_paths: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, path: str):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        data = self.to_dict()
        
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return obj
        
        # Recursively convert
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert(obj)
        
        data = deep_convert(data)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📊 Results saved: {path}")


# =============================================================================
# ENTERPRISE ENSEMBLE TRAINER
# =============================================================================

class EnterpriseEnsembleTrainer:
    """
    Enterprise-grade ensemble trainer with MLOps features.
    
    Wraps modular_trainers with:
    - Full experiment lifecycle management
    - Cross-validation with proper time-series splits
    - Statistical significance testing
    - Comprehensive result reporting
    """
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        
        # Setup components
        self.reproducibility = ReproducibilityManager()
        self.resource_monitor = ResourceMonitor() if config.enable_resource_monitoring else None
        self.data_validator = DataValidator() if config.enable_data_validation else None
        self.stat_validator = StatisticalValidator()
        
        # Setup experiment tracking
        self.experiment: Optional[TrainingExperiment] = None
        
        # Setup walk-forward validator
        if config.enable_cross_validation:
            cv_config = WalkForwardConfig(
                n_splits=config.cv_n_splits,
                train_period=config.cv_train_period,
                test_period=config.cv_test_period,
                gap=config.cv_gap,
                expanding=True,
            )
            self.walk_forward = WalkForwardValidator(cv_config)
        else:
            self.walk_forward = None
        
        # Results
        self.results: Optional[EnsembleResults] = None
        self._trainers: Dict[str, Any] = {}
        
        # Setup directories
        for dir_path in [config.save_dir, config.log_dir, config.experiment_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        data: Dict[str, Dict[str, np.ndarray]],
        trainer_config: Optional[TrainerConfig] = None,
    ) -> EnsembleResults:
        """
        Train full ensemble with enterprise features.
        
        Args:
            data: Dict from load_all_modular_data() with:
                  - 'direction'/'tcn': direction prediction data
                  - 'xgboost': momentum data
                  - 'rf': risk data
                  - 'ridge': confidence data
            trainer_config: Optional TrainerConfig for model hyperparameters
        
        Returns:
            EnsembleResults with all metrics and validation results
        """
        # Initialize results
        experiment_id = f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = EnsembleResults(
            experiment_id=experiment_id,
            experiment_name=self.config.experiment_name,
            config=asdict(self.config),
        )
        
        # Set seeds for reproducibility
        seed_info = self.reproducibility.set_seeds()
        self.results.environment = self.reproducibility.get_environment_fingerprint()
        self.results.environment['seed'] = str(seed_info['seed'])
        
        # Create experiment config
        exp_config = ExperimentConfig(
            name=self.config.experiment_name,
            model_type=self.config.model_type,
            instrument=self.config.instrument,
            granularity=self.config.granularity,
            log_dir=self.config.log_dir,
        )
        
        # Start experiment context
        self.experiment = TrainingExperiment(
            exp_config,
            enable_mlflow=self.config.enable_mlflow,
            enable_monitoring=self.config.enable_resource_monitoring,
            log_dir=self.config.log_dir,
        )
        
        with self.experiment:
            start_time = time.time()
            
            # 1. Data validation
            if self.config.enable_data_validation:
                self._validate_data(data)
            
            # 2. Cross-validation (if enabled)
            if self.config.enable_cross_validation:
                self._run_cross_validation(data, trainer_config)
            
            # 3. Train full ensemble
            self._train_ensemble(data, trainer_config)
            
            # 4. Statistical validation
            if self.config.enable_statistical_validation:
                self._run_statistical_validation(data)
            
            # 5. Log ensemble metrics
            self._log_ensemble_metrics()
            
            # Record training time
            self.results.resource_summary['total_time_seconds'] = time.time() - start_time
        
        # Get resource summary
        if self.experiment.resource_monitor:
            self.results.resource_summary.update(
                self.experiment.resource_monitor._compute_summary()
            )
        
        # Save results
        results_path = Path(self.config.experiment_dir) / f"{experiment_id}_results.json"
        self.results.save(str(results_path))
        
        return self.results
    
    def _validate_data(self, data: Dict[str, Dict[str, np.ndarray]]):
        """Validate all training data."""
        logger.info("=" * 60)
        logger.info("DATA VALIDATION")
        logger.info("=" * 60)
        
        all_warnings = []
        all_errors = []
        
        for name, dataset in data.items():
            if 'X_train' in dataset:
                X = dataset['X_train']
                y = dataset['y_train']
                
                result = self.data_validator.validate(X)
                y_result = self.data_validator.validate(y)
                
                if result.warnings or y_result.warnings:
                    all_warnings.extend([f"{name}: {w}" for w in result.warnings])
                    all_warnings.extend([f"{name}_y: {w}" for w in y_result.warnings])
                
                if result.errors or y_result.errors:
                    all_errors.extend([f"{name}: {e}" for e in result.errors])
                    all_errors.extend([f"{name}_y: {e}" for e in y_result.errors])
                
                logger.info(f"  {name}: X_train shape={X.shape}, valid={result.passed}")
        
        self.results.data_validation = DataValidationResult(
            passed=len(all_errors) == 0,
            warnings=all_warnings,
            errors=all_errors,
        )
        
        if all_errors:
            logger.error(f"❌ Data validation failed: {all_errors}")
        elif all_warnings:
            logger.warning(f"⚠️ Data validation warnings: {all_warnings}")
        else:
            logger.info("✅ All data validation checks passed")
    
    def _run_cross_validation(
        self,
        data: Dict[str, Dict[str, np.ndarray]],
        trainer_config: Optional[TrainerConfig],
    ):
        """Run walk-forward cross-validation on direction model."""
        logger.info("=" * 60)
        logger.info("WALK-FORWARD CROSS-VALIDATION")
        logger.info("=" * 60)
        
        # Get direction data
        dir_data = data.get('direction', data.get('tcn'))
        if dir_data is None:
            logger.warning("No direction data for cross-validation")
            return
        
        X = dir_data['X_train']
        y = dir_data['y_train']
        
        # For sequences, need to handle differently
        if len(X.shape) == 3:
            # Already sequences, flatten for CV indices
            X_flat = X.reshape(len(X), -1)
        else:
            X_flat = X
        
        logger.info(f"Running {self.config.cv_n_splits}-fold walk-forward CV on {len(X)} samples")
        
        fold_accuracies = []
        fold_balanced_accs = []
        
        for fold, (train_idx, test_idx) in enumerate(self.walk_forward.split(X_flat)):
            # Train a fresh transformer on this fold
            config = trainer_config or TrainerConfig(epochs=50, patience=10)  # Reduced for CV
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]
            
            # Quick training for CV (reduced epochs)
            try:
                trainer = TransformerDirectionTrainer(config)
                
                # Simplified training for CV folds
                metrics = trainer.train(
                    X_train_fold, y_train_fold,
                    X_test_fold, y_test_fold,
                    feature_names=dir_data.get('feature_names'),
                    instrument=f"{self.config.instrument}_cv{fold}",
                )
                
                # Get predictions
                y_pred = trainer.predict(X_test_fold)
                y_pred_binary = (y_pred > 0.5).astype(int)
                
                # Calculate metrics
                accuracy = np.mean(y_pred_binary == y_test_fold)
                
                # Balanced accuracy
                pos_idx = y_test_fold == 1
                neg_idx = y_test_fold == 0
                pos_acc = np.mean(y_pred_binary[pos_idx] == 1) if pos_idx.any() else 0
                neg_acc = np.mean(y_pred_binary[neg_idx] == 0) if neg_idx.any() else 0
                balanced_acc = (pos_acc + neg_acc) / 2
                
                fold_accuracies.append(accuracy)
                fold_balanced_accs.append(balanced_acc)
                
                logger.info(f"  Fold {fold + 1}: accuracy={accuracy:.4f}, balanced={balanced_acc:.4f}")
                
            except Exception as e:
                logger.warning(f"Fold {fold + 1} failed: {e}")
                continue
        
        if fold_accuracies:
            # Aggregate CV results
            cv_results = {
                'accuracy_mean': np.mean(fold_accuracies),
                'accuracy_std': np.std(fold_accuracies),
                'accuracy_ci95': 1.96 * np.std(fold_accuracies) / np.sqrt(len(fold_accuracies)),
                'balanced_acc_mean': np.mean(fold_balanced_accs),
                'balanced_acc_std': np.std(fold_balanced_accs),
                'n_folds': len(fold_accuracies),
            }
            
            self.results.cv_results = cv_results
            
            logger.info(f"\n📊 Cross-Validation Summary:")
            logger.info(f"   Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            logger.info(f"   Balanced: {cv_results['balanced_acc_mean']:.4f} ± {cv_results['balanced_acc_std']:.4f}")
            logger.info(f"   95% CI: [{cv_results['accuracy_mean'] - cv_results['accuracy_ci95']:.4f}, "
                       f"{cv_results['accuracy_mean'] + cv_results['accuracy_ci95']:.4f}]")
    
    def _train_ensemble(
        self,
        data: Dict[str, Dict[str, np.ndarray]],
        trainer_config: Optional[TrainerConfig],
    ):
        """Train full ensemble using modular_trainers."""
        logger.info("=" * 60)
        logger.info("FULL ENSEMBLE TRAINING")
        logger.info("=" * 60)
        
        config = trainer_config or TrainerConfig()
        
        # Train all models
        start_time = time.time()
        
        self._trainers = train_all_modular(
            data=data,
            config=config,
            save_dir=self.config.save_dir,
            use_transformer=True,
            warm_start=self.config.warm_start,
            train_histgb=self.config.train_histgb,
            instrument=self.config.instrument,
            data_range=self._get_data_range(data),
        )
        
        training_time = time.time() - start_time
        
        # Extract metrics from each trainer
        self._extract_trainer_metrics(training_time)
        
        # Save model paths
        self.results.model_paths = {
            'transformer': str(Path(self.config.save_dir) / "transformer_direction.keras"),
            'xgboost': str(Path(self.config.save_dir) / "xgb_momentum.pkl"),
            'rf': str(Path(self.config.save_dir) / "rf_risk.pkl"),
            'ridge': str(Path(self.config.save_dir) / "ridge_confidence.pkl"),
        }
        
        if 'histgb' in self._trainers:
            self.results.model_paths['histgb'] = str(Path(self.config.save_dir) / "histgb_direction.pkl")
    
    def _extract_trainer_metrics(self, training_time: float):
        """Extract metrics from trained models."""
        # Transformer/Direction
        if 'transformer' in self._trainers or 'direction' in self._trainers:
            trainer = self._trainers.get('transformer') or self._trainers.get('direction')
            metrics = getattr(trainer, 'metrics', {})
            
            self.results.transformer_metrics = ModelMetrics(
                model_name="Transformer",
                val_accuracy=metrics.get('val_accuracy', 0),
                val_loss=metrics.get('val_loss', 0),
                balanced_accuracy=metrics.get('balanced_accuracy', 0),
                training_time_seconds=training_time / 4,  # Approximate
            )
        
        # XGBoost
        if 'xgboost' in self._trainers:
            trainer = self._trainers['xgboost']
            metrics = getattr(trainer, 'metrics', {})
            
            self.results.xgboost_metrics = ModelMetrics(
                model_name="XGBoost",
                val_accuracy=metrics.get('accel_accuracy', 0),
                mae=metrics.get('momentum_mae', 0),
            )
        
        # Random Forest
        if 'rf' in self._trainers:
            trainer = self._trainers['rf']
            metrics = getattr(trainer, 'metrics', {})
            
            self.results.rf_metrics = ModelMetrics(
                model_name="RandomForest",
                mae=metrics.get('drawdown_mae', 0),
            )
        
        # Ridge
        if 'ridge' in self._trainers:
            trainer = self._trainers['ridge']
            metrics = getattr(trainer, 'metrics', {})
            
            self.results.ridge_metrics = ModelMetrics(
                model_name="Ridge",
                mae=metrics.get('confidence_mae', 0),
                r2_score=metrics.get('r2', 0),
            )
        
        # HistGB
        if 'histgb' in self._trainers:
            trainer = self._trainers['histgb']
            metrics = getattr(trainer, 'metrics', {})
            
            self.results.histgb_metrics = ModelMetrics(
                model_name="HistGradientBoosting",
                val_accuracy=metrics.get('val_accuracy', 0),
                balanced_accuracy=metrics.get('balanced_accuracy', 0),
            )
    
    def _run_statistical_validation(self, data: Dict[str, Dict[str, np.ndarray]]):
        """Run statistical validation with bootstrap CI."""
        logger.info("=" * 60)
        logger.info("STATISTICAL VALIDATION")
        logger.info("=" * 60)
        
        # Get validation predictions from transformer
        dir_data = data.get('direction', data.get('tcn'))
        if dir_data is None or 'transformer' not in self._trainers:
            logger.warning("Cannot run statistical validation without direction data and transformer")
            return
        
        trainer = self._trainers['transformer']
        X_val = dir_data['X_val']
        y_val = dir_data['y_val']
        
        try:
            y_pred = trainer.predict(X_val)
            
            if y_pred is not None:
                # Accuracy CI
                def accuracy_fn(y_t, y_p):
                    return np.mean((y_p > 0.5).astype(int) == y_t)
                
                accuracy_ci = self.stat_validator.bootstrap_confidence_interval(
                    accuracy_fn,
                    y_val.flatten(),
                    y_pred.flatten(),
                    n_bootstrap=self.config.bootstrap_n_samples,
                    confidence=self.config.bootstrap_confidence,
                )
                
                self.results.statistical_validation['accuracy_bootstrap'] = accuracy_ci
                
                # Update transformer metrics with bootstrap CI
                if self.results.transformer_metrics:
                    self.results.transformer_metrics.bootstrap_mean = accuracy_ci['mean']
                    self.results.transformer_metrics.bootstrap_ci_lower = accuracy_ci['ci_lower']
                    self.results.transformer_metrics.bootstrap_ci_upper = accuracy_ci['ci_upper']
                
                logger.info(f"📊 Bootstrap CI (accuracy): {accuracy_ci['mean']:.4f} "
                           f"[{accuracy_ci['ci_lower']:.4f}, {accuracy_ci['ci_upper']:.4f}]")
                
                # Test if better than random (50%)
                is_significant = accuracy_ci['ci_lower'] > 0.5
                logger.info(f"   Significantly better than random: {'✅ Yes' if is_significant else '❌ No'}")
                
                self.results.statistical_validation['better_than_random'] = is_significant
                
        except Exception as e:
            logger.warning(f"Statistical validation failed: {e}")
    
    def _log_ensemble_metrics(self):
        """Log ensemble-level metrics."""
        logger.info("=" * 60)
        logger.info("ENSEMBLE SUMMARY")
        logger.info("=" * 60)
        
        # Log to experiment
        if self.experiment:
            metrics = {}
            
            if self.results.transformer_metrics:
                metrics['transformer/val_accuracy'] = self.results.transformer_metrics.val_accuracy
                metrics['transformer/balanced_accuracy'] = self.results.transformer_metrics.balanced_accuracy
            
            if self.results.xgboost_metrics:
                metrics['xgboost/accel_accuracy'] = self.results.xgboost_metrics.val_accuracy
                metrics['xgboost/momentum_mae'] = self.results.xgboost_metrics.mae
            
            if self.results.rf_metrics:
                metrics['rf/drawdown_mae'] = self.results.rf_metrics.mae
            
            if self.results.ridge_metrics:
                metrics['ridge/r2_score'] = self.results.ridge_metrics.r2_score
            
            if metrics:
                self.experiment.log_metrics(metrics)
        
        # Print summary table
        logger.info("\n┌────────────────────┬─────────────┬──────────────┐")
        logger.info("│ Model              │ Primary     │ Secondary    │")
        logger.info("├────────────────────┼─────────────┼──────────────┤")
        
        if self.results.transformer_metrics:
            m = self.results.transformer_metrics
            logger.info(f"│ Transformer        │ acc={m.val_accuracy:.3f}   │ bal={m.balanced_accuracy:.3f}    │")
        
        if self.results.xgboost_metrics:
            m = self.results.xgboost_metrics
            logger.info(f"│ XGBoost            │ acc={m.val_accuracy:.3f}   │ mae={m.mae:.4f}   │")
        
        if self.results.rf_metrics:
            m = self.results.rf_metrics
            logger.info(f"│ Random Forest      │ mae={m.mae:.4f}  │              │")
        
        if self.results.ridge_metrics:
            m = self.results.ridge_metrics
            logger.info(f"│ Ridge              │ r2={m.r2_score:.4f}   │ mae={m.mae:.4f}   │")
        
        logger.info("└────────────────────┴─────────────┴──────────────┘")
        
        if self.results.cv_results:
            cv = self.results.cv_results
            logger.info(f"\n📊 Cross-Validation: {cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}")
    
    def _get_data_range(self, data: Dict[str, Dict[str, np.ndarray]]) -> str:
        """Get date range string from data."""
        # Try to extract from data if available
        return f"{self.config.instrument}_{datetime.now().strftime('%Y%m%d')}"


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_enterprise_ensemble_training(
    data: Dict[str, Dict[str, np.ndarray]],
    experiment_name: str = "ensemble_training",
    instrument: str = "EUR_USD",
    granularity: str = "H1",
    trainer_config: Optional[TrainerConfig] = None,
    enable_cross_validation: bool = True,
    enable_statistical_validation: bool = True,
    save_dir: str = "trained_data/models",
    **kwargs,
) -> EnsembleResults:
    """
    Convenience function for enterprise ensemble training.
    
    This is the recommended entry point for professional training.
    
    Args:
        data: Dict from load_all_modular_data()
        experiment_name: Name for this training run
        instrument: Trading instrument (e.g., "EUR_USD")
        granularity: Timeframe (e.g., "H1")
        trainer_config: Optional TrainerConfig for model hyperparameters
        enable_cross_validation: Run walk-forward CV
        enable_statistical_validation: Compute bootstrap CIs
        save_dir: Where to save models
        **kwargs: Additional EnterpriseConfig parameters
    
    Returns:
        EnsembleResults with complete training results
    
    Example:
        from modular_data_loaders import load_all_modular_data
        from enterprise_integration import run_enterprise_ensemble_training
        
        # Load data
        data = load_all_modular_data(df, config)
        
        # Train with enterprise features
        results = run_enterprise_ensemble_training(
            data=data,
            experiment_name="EUR_USD_H1_experiment",
            instrument="EUR_USD",
            granularity="H1",
        )
        
        # Check results
        print(f"Transformer accuracy: {results.transformer_metrics.val_accuracy:.2%}")
        print(f"Cross-validation: {results.cv_results['accuracy_mean']:.2%} ± {results.cv_results['accuracy_std']:.2%}")
    """
    # Create config
    config = EnterpriseConfig(
        experiment_name=experiment_name,
        instrument=instrument,
        granularity=granularity,
        trainer_config=trainer_config,
        enable_cross_validation=enable_cross_validation,
        enable_statistical_validation=enable_statistical_validation,
        save_dir=save_dir,
        **kwargs,
    )
    
    # Create trainer and run
    trainer = EnterpriseEnsembleTrainer(config)
    results = trainer.train(data, trainer_config)
    
    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_training_report(results: EnsembleResults, output_path: str = None) -> str:
    """
    Generate a markdown training report.
    
    Args:
        results: EnsembleResults from training
        output_path: Optional path to save report
    
    Returns:
        Markdown string
    """
    lines = [
        "# ML Ensemble Training Report",
        "",
        f"**Experiment:** {results.experiment_name}",
        f"**Timestamp:** {results.timestamp}",
        f"**Experiment ID:** {results.experiment_id}",
        "",
        "## Model Performance",
        "",
        "| Model | Primary Metric | Secondary | Bootstrap CI |",
        "|-------|---------------|-----------|--------------|",
    ]
    
    if results.transformer_metrics:
        m = results.transformer_metrics
        ci = f"[{m.bootstrap_ci_lower:.3f}, {m.bootstrap_ci_upper:.3f}]" if m.bootstrap_ci_lower else "N/A"
        lines.append(f"| Transformer | acc={m.val_accuracy:.3f} | balanced={m.balanced_accuracy:.3f} | {ci} |")
    
    if results.xgboost_metrics:
        m = results.xgboost_metrics
        lines.append(f"| XGBoost | accel_acc={m.val_accuracy:.3f} | mae={m.mae:.4f} | N/A |")
    
    if results.rf_metrics:
        m = results.rf_metrics
        lines.append(f"| Random Forest | mae={m.mae:.4f} | - | N/A |")
    
    if results.ridge_metrics:
        m = results.ridge_metrics
        lines.append(f"| Ridge | r2={m.r2_score:.3f} | mae={m.mae:.4f} | N/A |")
    
    lines.extend([
        "",
        "## Cross-Validation Results",
        "",
    ])
    
    if results.cv_results:
        cv = results.cv_results
        lines.extend([
            f"- **Folds:** {cv.get('n_folds', 'N/A')}",
            f"- **Accuracy:** {cv.get('accuracy_mean', 0):.4f} ± {cv.get('accuracy_std', 0):.4f}",
            f"- **95% CI:** [{cv.get('accuracy_mean', 0) - cv.get('accuracy_ci95', 0):.4f}, "
            f"{cv.get('accuracy_mean', 0) + cv.get('accuracy_ci95', 0):.4f}]",
            f"- **Balanced Accuracy:** {cv.get('balanced_acc_mean', 0):.4f} ± {cv.get('balanced_acc_std', 0):.4f}",
        ])
    else:
        lines.append("Cross-validation not run.")
    
    lines.extend([
        "",
        "## Statistical Validation",
        "",
    ])
    
    if results.statistical_validation:
        stat = results.statistical_validation
        lines.extend([
            f"- **Significantly better than random:** {'✅ Yes' if stat.get('better_than_random') else '❌ No'}",
        ])
        
        if 'accuracy_bootstrap' in stat:
            bs = stat['accuracy_bootstrap']
            lines.append(f"- **Bootstrap CI:** [{bs.get('ci_lower', 0):.4f}, {bs.get('ci_upper', 0):.4f}]")
    
    lines.extend([
        "",
        "## Resource Usage",
        "",
    ])
    
    if results.resource_summary:
        rs = results.resource_summary
        lines.extend([
            f"- **Total training time:** {rs.get('total_time_seconds', 0):.1f}s",
            f"- **Peak memory:** {rs.get('memory_max_gb', 0):.2f} GB",
            f"- **Average CPU:** {rs.get('cpu_mean', 0):.1f}%",
        ])
    
    lines.extend([
        "",
        "## Configuration",
        "",
        f"- **Instrument:** {results.config.get('instrument', 'N/A')}",
        f"- **Granularity:** {results.config.get('granularity', 'N/A')}",
        f"- **Warm start:** {results.config.get('warm_start', False)}",
        "",
        "---",
        f"*Generated: {datetime.now().isoformat()}*",
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).write_text(report)
        logger.info(f"📄 Report saved: {output_path}")
    
    return report


if __name__ == "__main__":
    print("Enterprise Integration Module")
    print("=" * 50)
    
    # Create sample config
    config = EnterpriseConfig(
        experiment_name="test_experiment",
        instrument="EUR_USD",
        granularity="H1",
    )
    
    print(f"Config: {config}")
    print("\n✅ Enterprise integration ready!")
    print("\nUsage:")
    print("  from enterprise_integration import run_enterprise_ensemble_training")
    print("  results = run_enterprise_ensemble_training(data, instrument='EUR_USD')")
