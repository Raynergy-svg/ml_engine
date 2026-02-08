"""
Enterprise-Grade ML Training Infrastructure

Professional training pipeline with industry-standard MLOps features:
- MLflow experiment tracking & model registry
- Reproducibility guarantees (seed management, data versioning)
- Walk-forward cross-validation for time series
- Statistical significance testing (bootstrap CI, t-tests)
- Resource monitoring (GPU/memory)
- Structured JSON logging
- Data validation & drift detection
- Checkpoint management with lineage tracking

Usage:
    from enterprise_training import EnterpriseTrainer, TrainingExperiment

    experiment = TrainingExperiment(
        name="EUR_USD_H1_ensemble",
        model_type="ensemble",
        instrument="EUR_USD",
    )

    with experiment:
        trainer = EnterpriseTrainer(experiment)
        trainer.train(X, y)
        trainer.validate()
        trainer.log_results()

References:
- MLflow: https://mlflow.org/
- Walk-forward validation: Lopez de Prado (2018)
- EWC: Kirkpatrick et al. (2017)
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import random
import signal
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, List, Optional,
    Tuple, Union
)

import numpy as np

try:
    import mlflow
    from mlflow.tracking import MlflowClient  # noqa: F401
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_enterprise_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    json_format: bool = True,
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure enterprise-grade structured logging.

    Features:
    - JSON format for log aggregation (ELK, Splunk compatible)
    - File and console output
    - Correlation IDs for distributed tracing
    - Contextual fields (experiment, model, epoch)
    """
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("enterprise_training")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    if STRUCTLOG_AVAILABLE and json_format:
        # Structured JSON logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        formatter = logging.Formatter("%(message)s")
    else:
        # Fallback to standard formatting
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler for persistent logs
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_suffix = f"_{experiment_name}" if experiment_name else ""
        log_file = Path(log_dir) / f"training{exp_suffix}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Also create JSON log for structured queries
        json_log = Path(log_dir) / f"training{exp_suffix}_{timestamp}.jsonl"
        json_handler = logging.FileHandler(json_log)
        json_handler.setLevel(logging.DEBUG)
        json_formatter = logging.Formatter("%(message)s")
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)

    return logger


logger = logging.getLogger("enterprise_training")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

@dataclass
class ReproducibilityConfig:
    """Configuration for reproducible training."""
    seed: int = 42
    deterministic: bool = True
    hash_data: bool = True
    log_environment: bool = True


class ReproducibilityManager:
    """
    Ensure reproducible training across runs.

    Sets seeds for:
    - Python random
    - NumPy
    - TensorFlow
    - Environment variables

    Logs environment fingerprint for debugging.
    """

    def __init__(self, config: ReproducibilityConfig = None):
        self.config = config or ReproducibilityConfig()
        self._env_fingerprint: Dict[str, str] = {}

    def set_seeds(self) -> Dict[str, Any]:
        """Set global seeds for reproducibility."""
        seed = self.config.seed

        # Python
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # TensorFlow (if available)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)

            if self.config.deterministic:
                # Enable deterministic ops (may slow down training)
                os.environ['TF_DETERMINISTIC_OPS'] = '1'
                os.environ['PYTHONHASHSEED'] = str(seed)
        except ImportError:
            pass

        # PyTorch (if available)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        seed_info = {
            'seed': seed,
            'deterministic': self.config.deterministic,
            'timestamp': datetime.now().isoformat(),
        }

        logger.info(f"🎲 Seeds set: {seed} (deterministic={self.config.deterministic})")
        return seed_info

    def get_environment_fingerprint(self) -> Dict[str, str]:
        """Capture environment for reproducibility tracking."""
        if self._env_fingerprint:
            return self._env_fingerprint

        fingerprint = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
        }

        # Package versions
        packages = ['numpy', 'tensorflow', 'pandas', 'scikit-learn', 'xgboost']
        for pkg in packages:
            try:
                module = __import__(pkg)
                fingerprint[f'{pkg}_version'] = getattr(module, '__version__', 'unknown')
            except ImportError:
                fingerprint[f'{pkg}_version'] = 'not_installed'

        # Git info (if available)
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            fingerprint['git_commit'] = git_hash[:12]
        except (subprocess.CalledProcessError, FileNotFoundError):
            fingerprint['git_commit'] = 'unknown'

        self._env_fingerprint = fingerprint
        return fingerprint

    @staticmethod
    def compute_data_hash(
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 1000
    ) -> str:
        """
        Compute deterministic hash of training data.

        Uses sampling for efficiency on large datasets.
        """
        n = min(n_samples, len(X))
        indices = np.linspace(0, len(X) - 1, n, dtype=int)

        X_sample = X[indices]
        y_sample = y[indices] if hasattr(y, '__getitem__') else y

        data_bytes = X_sample.tobytes()
        if isinstance(y_sample, np.ndarray):
            data_bytes += y_sample.tobytes()

        return hashlib.sha256(data_bytes).hexdigest()[:16]


# =============================================================================
# RESOURCE MONITORING
# =============================================================================

@dataclass
class ResourceStats:
    """Resource utilization snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_gb: float
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    gpu_utilization: float = 0.0


class ResourceMonitor:
    """
    Monitor GPU and system resources during training.

    Runs in background thread, collects stats at intervals.
    Useful for:
    - Detecting memory leaks
    - Capacity planning
    - Cost optimization
    """

    def __init__(
        self,
        interval_seconds: float = 5.0,
        log_to_mlflow: bool = True,
        alert_memory_threshold: float = 90.0,
    ):
        self.interval = interval_seconds
        self.log_to_mlflow = log_to_mlflow
        self.alert_threshold = alert_memory_threshold

        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self.history: List[ResourceStats] = []
        self._step = 0

    def start(self):
        """Start background resource monitoring."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, resource monitoring disabled")
            return

        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("📊 Resource monitoring started")

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return summary statistics."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=10)

        summary = self._compute_summary()
        logger.info(f"📊 Resource monitoring stopped: {summary}")
        return summary

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                stats = self._collect_stats()
                self.history.append(stats)

                # Log to MLflow if available
                if self.log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
                    mlflow.log_metrics({
                        'system/cpu_percent': stats.cpu_percent,
                        'system/memory_percent': stats.memory_percent,
                        'system/memory_gb': stats.memory_gb,
                    }, step=self._step)
                    self._step += 1

                # Alert on high memory usage
                if stats.memory_percent > self.alert_threshold:
                    logger.warning(
                        f"⚠️ High memory usage: {stats.memory_percent:.1f}%"
                    )

            except Exception as e:
                logger.debug(f"Resource monitoring error: {e}")

            time.sleep(self.interval)

    def _collect_stats(self) -> ResourceStats:
        """Collect current resource statistics."""
        stats = ResourceStats(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent() if PSUTIL_AVAILABLE else 0,
            memory_percent=psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0,
            memory_gb=psutil.virtual_memory().used / (1024**3) if PSUTIL_AVAILABLE else 0,
        )

        # Check for GPU (TensorFlow Metal on Mac)
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            stats.gpu_available = len(gpus) > 0
        except Exception:
            pass

        return stats

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute resource usage summary."""
        if not self.history:
            return {}

        return {
            'cpu_mean': np.mean([h.cpu_percent for h in self.history]),
            'cpu_max': np.max([h.cpu_percent for h in self.history]),
            'memory_mean_gb': np.mean([h.memory_gb for h in self.history]),
            'memory_max_gb': np.max([h.memory_gb for h in self.history]),
            'duration_seconds': self.history[-1].timestamp - self.history[0].timestamp,
            'n_samples': len(self.history),
        }


# =============================================================================
# EXPERIMENT TRACKING
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for an ML experiment."""
    name: str
    model_type: str
    instrument: str = ""
    granularity: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    # Training params
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 20

    # Data params
    sequence_length: int = 60
    direction_threshold: float = 0.0015
    direction_lookahead: int = 24

    # Model params
    hidden_size: int = 64
    num_layers: int = 3
    dropout: float = 0.3

    # Continual learning
    use_ema: bool = True
    use_ewc: bool = True
    use_replay: bool = True

    # Paths
    model_dir: str = "trained_data/models"
    log_dir: str = "trained_data/logs"
    checkpoint_dir: str = "trained_data/checkpoints"


class ExperimentTracker:
    """
    Track experiments using MLflow and WandB.

    Features:
    - Automatic parameter logging to MLflow AND WandB
    - Metric tracking with history
    - Model artifact storage
    - Experiment comparison
    - WandB real-time visualization
    """

    def __init__(
        self,
        tracking_uri: str = "trained_data/mlruns",
        experiment_name: Optional[str] = None,
        wandb_project: str = "ml_engine",
        wandb_tags: Optional[List[str]] = None,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name or "ml_engine_training"
        self.wandb_project = wandb_project
        self.wandb_tags = wandb_tags or []
        self._run_id: Optional[str] = None
        self._wandb_run = None
        self._start_time: Optional[float] = None

        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, using file-based tracking")
            return

        # Create tracking directory
        Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)

        # Set tracking URI - use SQLite backend (file backend deprecated Feb 2026)
        db_path = Path(self.tracking_uri) / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.absolute()}")

        # Create or get experiment
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"📊 MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup error: {e}")

    def _setup_wandb(self, run_name: str, config: Optional[ExperimentConfig] = None):
        """Initialize WandB tracking."""
        if not WANDB_AVAILABLE:
            logger.debug("WandB not available, skipping")
            return

        try:
            # Convert config to dict for WandB
            config_dict = asdict(config) if config else {}

            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=run_name,
                tags=self.wandb_tags,
                config=config_dict,
                reinit=True,  # Allow multiple runs in same process
            )
            logger.info(f"📈 WandB run: {run_name} (project: {self.wandb_project})")
        except Exception as e:
            logger.warning(f"WandB setup error: {e}")
            self._wandb_run = None

    def _finish_wandb(self):
        """Finish WandB run."""
        if WANDB_AVAILABLE and self._wandb_run is not None:
            try:
                wandb.finish()
                logger.debug("WandB run finished")
            except Exception as e:
                logger.warning(f"WandB finish error: {e}")
            finally:
                self._wandb_run = None

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
        wandb_tags: Optional[List[str]] = None,
    ):
        """
        Context manager for MLflow AND WandB run.

        Usage:
            with tracker.start_run(config=config, wandb_tags=["GBP_JPY", "H1"]):
                # Training code
                tracker.log_metric("loss", 0.5)
        """
        self._start_time = time.time()
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Merge tags
        all_tags = list(set(self.wandb_tags + (wandb_tags or [])))
        self.wandb_tags = all_tags

        # Start WandB
        self._setup_wandb(run_name, config)

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name) as run:
                self._run_id = run.info.run_id

                # Log config as parameters
                if config:
                    self._log_config(config)

                try:
                    yield self
                finally:
                    duration = time.time() - self._start_time
                    mlflow.log_metric("training_duration_seconds", duration)
                    # Log duration to WandB too
                    if WANDB_AVAILABLE and self._wandb_run:
                        wandb.log({"training_duration_seconds": duration})
                    self._finish_wandb()
        else:
            # Fallback to file-based tracking
            try:
                yield self
            finally:
                self._finish_wandb()

    def _log_config(self, config: ExperimentConfig):
        """Log experiment configuration as MLflow params."""
        params = asdict(config)

        # Flatten nested dicts and convert to strings
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    flat_params[f"{key}.{k}"] = str(v)
            else:
                flat_params[key] = str(value)

        # MLflow has 500 param limit per run
        for key, value in list(flat_params.items())[:500]:
            try:
                mlflow.log_param(key, value)
            except Exception:
                pass  # Ignore param logging errors

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric to MLflow AND WandB."""
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metric(name, value, step=step)

        # Log to WandB
        if WANDB_AVAILABLE and self._wandb_run:
            log_data = {name: value}
            if step is not None:
                wandb.log(log_data, step=step)
            else:
                wandb.log(log_data)

        logger.debug(f"Metric: {name}={value}" + (f" (step={step})" if step else ""))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics to MLflow AND WandB."""
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

        # Log to WandB
        if WANDB_AVAILABLE and self._wandb_run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        logger.debug(f"Metrics (step={step}): {metrics}")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        """Log a file as artifact to MLflow AND WandB."""
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_artifact(path, artifact_path)

        # Log to WandB
        if WANDB_AVAILABLE and self._wandb_run:
            try:
                wandb.save(path)
            except Exception as e:
                logger.debug(f"WandB artifact save failed: {e}")

        logger.info(f"📦 Artifact logged: {path}")

    def log_model_summary(self, model, model_name: str = "model"):
        """Log model architecture summary."""
        summary_lines = []

        if hasattr(model, 'summary'):
            # Keras model
            model.summary(print_fn=lambda x: summary_lines.append(x))

        summary_text = "\n".join(summary_lines)

        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_text(summary_text, f"{model_name}_summary.txt")

        # Log to WandB
        if WANDB_AVAILABLE and self._wandb_run:
            wandb.log({f"{model_name}_summary": wandb.Html(f"<pre>{summary_text}</pre>")})

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.set_tag(key, value)

        # WandB tags are set at init, but we can add to config
        if WANDB_AVAILABLE and self._wandb_run:
            wandb.config.update({key: value}, allow_val_change=True)

    def get_run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self._run_id

    def get_wandb_url(self) -> Optional[str]:
        """Get WandB run URL for easy access."""
        if WANDB_AVAILABLE and self._wandb_run:
            return self._wandb_run.get_url()
        return None


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    For 12,000 H1 candles (~500 trading days):
    - min_train_samples: 4000 (~167 days) - minimum for reliable Transformer training
    - test_period: 1000 (~42 days) - enough for statistically meaningful validation
    - n_splits: 3 - balances regime coverage vs training data adequacy
    - gap: 24 (1 day) - prevents label leakage from overlapping windows
    """
    n_splits: int = 3  # Reduced from 5 - ensures adequate training data per fold
    min_train_samples: int = 4000  # Minimum training samples (prevents underfitting in early folds)
    train_period: int = 6000  # Target training window for sliding mode
    test_period: int = 1000  # ~42 days H1 - more stable validation estimates
    gap: int = 24  # Gap between train and test to prevent leakage
    expanding: bool = True  # Expanding vs sliding window
    purge_overlap: int = 0  # Additional purge for overlapping labels


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series.

    Critical for financial ML to:
    - Prevent look-ahead bias
    - Simulate realistic deployment scenario
    - Test model stability across market regimes

    References:
    - Lopez de Prado, "Advances in Financial Machine Learning" (2018)
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for walk-forward validation.

        Ensures minimum training samples per fold to prevent underfitting,
        especially important for deep learning models like Transformers.

        Yields:
            (train_indices, test_indices) for each fold
        """
        n_samples = len(X)
        cfg = self.config

        # Use min_train_samples to ensure adequate training data
        min_train = getattr(cfg, 'min_train_samples', cfg.train_period) + cfg.gap
        available_space = n_samples - cfg.test_period - min_train

        if available_space < cfg.n_splits:
            logger.warning(
                f"Insufficient data for {cfg.n_splits} folds with min_train={min_train}, "
                f"reducing to {max(1, available_space)} folds"
            )
            n_splits = max(1, available_space)
        else:
            n_splits = cfg.n_splits

        # Position test windows to ensure minimum training samples
        # First test window starts after min_train samples
        first_test_start = min_train
        last_test_start = n_samples - cfg.test_period

        test_starts = np.linspace(
            first_test_start,
            last_test_start,
            n_splits,
            dtype=int
        )

        logger.info(
            f"Walk-forward: {n_splits} folds, min_train={min_train}, "
            f"test_period={cfg.test_period}, total_samples={n_samples}"
        )

        for fold, test_start in enumerate(test_starts):
            # Training indices
            if cfg.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - cfg.gap - cfg.train_period)

            train_end = test_start - cfg.gap - cfg.purge_overlap

            # Ensure minimum training samples even in expanding mode
            if train_end - train_start < getattr(cfg, 'min_train_samples', 2000):
                logger.warning(
                    f"Fold {fold + 1}: Training samples ({train_end - train_start}) "
                    f"below minimum ({getattr(cfg, 'min_train_samples', 2000)}), skipping"
                )
                continue

            # Test indices
            test_end = min(test_start + cfg.test_period, n_samples)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            logger.info(
                f"Fold {fold + 1}/{n_splits}: "
                f"train=[{train_start}:{train_end}] ({len(train_idx)}), "
                f"test=[{test_start}:{test_end}] ({len(test_idx)})"
            )

            yield train_idx, test_idx

    def cross_validate(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        fit_kwargs: Dict[str, Any] = None,
        warm_start_path: str = None,
    ) -> Dict[str, Any]:
        """
        Run full walk-forward cross-validation.

        Args:
            model_fn: Callable that returns a fresh model instance
            X: Features
            y: Labels
            fit_kwargs: Additional kwargs for model.fit()
            warm_start_path: Path to pre-trained weights for continual learning

        Returns:
            Dictionary with per-fold and aggregate metrics
        """
        fit_kwargs = fit_kwargs or {}

        results = {
            'fold_metrics': [],
            'fold_info': [],
        }

        for fold, (train_idx, test_idx) in enumerate(self.split(X, y)):
            # Create fresh model
            model = model_fn()

            # Load pre-trained weights for continual learning (EWC)
            if warm_start_path:
                try:
                    model.load_weights(warm_start_path)
                    logger.info(f"Fold {fold + 1}: Loaded warm-start weights from {warm_start_path}")
                except Exception as e:
                    logger.warning(f"Fold {fold + 1}: Could not load warm-start weights: {e}")

            # Train
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Handle sequences (3D data)
            if len(X.shape) == 3:
                # Already sequences, use directly
                pass

            # Fit model
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                verbose=0,
                **fit_kwargs
            )

            # Evaluate
            y_pred = model.predict(X_test, verbose=0)

            # Calculate metrics
            fold_metrics = self._calculate_metrics(y_test, y_pred)
            fold_metrics['fold'] = fold

            results['fold_metrics'].append(fold_metrics)
            results['fold_info'].append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_range': (int(train_idx[0]), int(train_idx[-1])),
                'test_range': (int(test_idx[0]), int(test_idx[-1])),
            })

            logger.info(
                f"Fold {fold + 1}: accuracy={fold_metrics.get('accuracy', 0):.4f}, "
                f"auc={fold_metrics.get('auc', 0):.4f}"
            )

        # Aggregate statistics
        metric_names = [k for k in results['fold_metrics'][0] if k != 'fold']

        for metric in metric_names:
            scores = [f[metric] for f in results['fold_metrics']]
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
            results[f'{metric}_ci95'] = 1.96 * np.std(scores) / np.sqrt(len(scores))

        return results

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}

        # Handle different prediction formats
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

        # Binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Accuracy
        metrics['accuracy'] = np.mean(y_pred_binary == y_true)

        # AUC (if scipy available)
        if SCIPY_AVAILABLE:
            try:
                from sklearn.metrics import roc_auc_score
                metrics['auc'] = roc_auc_score(y_true, y_pred)
            except Exception:
                metrics['auc'] = 0.5

        # Balanced accuracy
        pos_idx = y_true == 1
        neg_idx = y_true == 0
        pos_acc = np.mean(y_pred_binary[pos_idx] == y_true[pos_idx]) if pos_idx.any() else 0
        neg_acc = np.mean(y_pred_binary[neg_idx] == y_true[neg_idx]) if neg_idx.any() else 0
        metrics['balanced_accuracy'] = (pos_acc + neg_acc) / 2

        return metrics


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================

class StatisticalValidator:
    """
    Statistical significance testing for ML models.

    Methods:
    - Bootstrap confidence intervals
    - Paired t-test for model comparison
    - Sharpe ratio significance test
    """

    @staticmethod
    def bootstrap_confidence_interval(
        metric_fn: Callable,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Bootstrap confidence interval for any metric.

        Args:
            metric_fn: Function(y_true, y_pred) -> float
            y_true: True labels
            y_pred: Predictions
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95)
            seed: Random seed

        Returns:
            Dictionary with mean, std, and CI bounds
        """
        np.random.seed(seed)

        n = len(y_true)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            try:
                score = metric_fn(y_true[idx], y_pred[idx])
                bootstrap_scores.append(score)
            except Exception:
                continue

        if not bootstrap_scores:
            return {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0}

        bootstrap_scores = np.array(bootstrap_scores)

        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_scores, alpha * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

        return {
            'mean': float(np.mean(bootstrap_scores)),
            'std': float(np.std(bootstrap_scores)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_bootstrap': n_bootstrap,
            'confidence': confidence,
        }

    @staticmethod
    def paired_ttest(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Paired t-test for comparing two models on same folds.

        Use this when you have cross-validation scores from two models.
        """
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        mean_diff = np.mean(scores_a - scores_b)
        std_diff = np.std(scores_a - scores_b)

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'better_model': 'A' if mean_diff > 0 else 'B',
            'alpha': alpha,
        }

    @staticmethod
    def sharpe_ratio_significance(
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """
        Test if Sharpe ratio is significantly different from benchmark.

        Uses asymptotic distribution of Sharpe ratio.
        """
        n = len(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return {'sharpe_ratio': 0, 'p_value': 1.0, 'significant': False}

        # Annualized Sharpe
        sharpe = (mean_return * np.sqrt(periods_per_year)) / std_return

        # Standard error of Sharpe (Lo, 2002)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)

        # Z-test
        z_stat = (sharpe - benchmark_sharpe) / se_sharpe

        if SCIPY_AVAILABLE:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            p_value = 1.0  # Conservative fallback

        return {
            'sharpe_ratio': float(sharpe),
            'std_error': float(se_sharpe),
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01,
        }


# =============================================================================
# DATA VALIDATION
# =============================================================================

@dataclass
class DataValidationResult:
    """Result of data validation checks."""
    passed: bool
    checks: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataValidator:
    """
    Validate data quality and detect drift.

    Checks:
    - Missing values
    - Duplicate timestamps
    - Value ranges
    - Distribution drift (KS test)
    """

    def __init__(
        self,
        reference_stats: Optional[Dict[str, Any]] = None,
        drift_threshold: float = 0.05,
    ):
        self.reference_stats = reference_stats or {}
        self.drift_threshold = drift_threshold

    def validate(self, df) -> DataValidationResult:
        """Run comprehensive data validation."""
        result = DataValidationResult(passed=True)

        # Check 1: Missing values
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                missing = df.isnull().sum()
                if missing.any():
                    result.checks['missing_values'] = missing[missing > 0].to_dict()
                    result.warnings.append(
                        f"Missing values in: {list(missing[missing > 0].index)}"
                    )
        except ImportError:
            pass

        # Check 2: Array checks for numpy
        if isinstance(df, np.ndarray):
            # Check for NaN
            nan_count = np.isnan(df).sum()
            if nan_count > 0:
                result.checks['nan_count'] = int(nan_count)
                result.warnings.append(f"Found {nan_count} NaN values")

            # Check for infinite values
            inf_count = np.isinf(df).sum()
            if inf_count > 0:
                result.checks['inf_count'] = int(inf_count)
                result.errors.append(f"Found {inf_count} infinite values")
                result.passed = False

            # Check for extreme values
            if df.size > 0:
                max_val = np.abs(df[~np.isnan(df)]).max() if not np.all(np.isnan(df)) else 0
                if max_val > 1e10:
                    result.warnings.append(f"Extreme values detected: max={max_val:.2e}")

        # Check 3: Drift detection
        if self.reference_stats:
            drift_result = self._detect_drift(df)
            result.checks['drift'] = drift_result

            drifted_features = [k for k, v in drift_result.items() if v.get('drifted', False)]
            if drifted_features:
                result.warnings.append(f"Drift detected in: {drifted_features}")

        return result

    def _detect_drift(self, df) -> Dict[str, Dict]:
        """Detect distribution drift using KS test."""
        if not SCIPY_AVAILABLE:
            return {}

        drift_results = {}

        # For numpy arrays, check overall distribution
        if isinstance(df, np.ndarray) and 'overall' in self.reference_stats:
            ref_dist = self.reference_stats['overall']

            # Flatten if needed
            current = df.flatten() if len(df.shape) > 1 else df
            current = current[~np.isnan(current)]

            if len(current) > 0 and len(ref_dist) > 0:
                ks_stat, p_value = stats.ks_2samp(ref_dist, current)
                drift_results['overall'] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drifted': p_value < self.drift_threshold
                }

        return drift_results

    def compute_reference_stats(self, df) -> Dict[str, Any]:
        """
        Compute reference statistics from training data.

        Call this on training data, then use result for drift detection.
        """
        stats_dict = {}

        if isinstance(df, np.ndarray):
            flat = df.flatten()
            flat = flat[~np.isnan(flat)]

            stats_dict['overall'] = flat.tolist()[:10000]  # Sample for efficiency
            stats_dict['mean'] = float(np.mean(flat))
            stats_dict['std'] = float(np.std(flat))
            stats_dict['min'] = float(np.min(flat))
            stats_dict['max'] = float(np.max(flat))

        return stats_dict


# =============================================================================
# ERROR HANDLING & RECOVERY
# =============================================================================

class TrainingErrorHandler:
    """
    Handle training errors with automatic recovery.

    Features:
    - Signal handling (SIGINT, SIGTERM)
    - Emergency checkpoint saving
    - OOM recovery
    """

    def __init__(
        self,
        checkpoint_dir: str = "trained_data/checkpoints",
        max_retries: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.current_epoch = 0
        self._interrupted = False

    @contextmanager
    def training_context(self, model, experiment_id: str = ""):
        """
        Context manager for safe training with error recovery.

        Usage:
            with handler.training_context(model, "exp_001"):
                model.fit(...)
        """
        # Setup signal handlers
        original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        original_sigterm = signal.signal(signal.SIGTERM, self._handle_interrupt)

        try:
            yield self
        except KeyboardInterrupt:
            logger.warning(f"Training interrupted at epoch {self.current_epoch}")
            self._save_emergency_checkpoint(model, experiment_id, "interrupted")
            raise
        except MemoryError as e:
            logger.error(f"Memory error at epoch {self.current_epoch}: {e}")
            self._save_emergency_checkpoint(model, experiment_id, "oom")
            self._try_recover_memory()
            raise
        except Exception as e:
            logger.error(f"Training error: {e}\n{traceback.format_exc()}")
            self._save_emergency_checkpoint(model, experiment_id, "error")
            raise
        finally:
            # Restore signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self._interrupted = True
        logger.warning(f"Received signal {signum}, finishing current epoch...")

    def _save_emergency_checkpoint(
        self,
        model,
        experiment_id: str,
        reason: str,
    ):
        """Save emergency checkpoint on error/interrupt."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"emergency_{experiment_id}_{reason}_{timestamp}"

            if hasattr(model, 'save_weights'):
                model.save_weights(str(checkpoint_path))
            elif hasattr(model, 'save'):
                model.save(str(checkpoint_path))

            # Save metadata
            meta = {
                'reason': reason,
                'epoch': self.current_epoch,
                'timestamp': datetime.now().isoformat(),
                'experiment_id': experiment_id,
            }
            with open(f"{checkpoint_path}_meta.json", 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"💾 Emergency checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")

    def _try_recover_memory(self):
        """Attempt to recover from OOM."""
        logger.info("Attempting memory recovery...")
        gc.collect()

        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass


# =============================================================================
# TRAINING EXPERIMENT (Main Entry Point)
# =============================================================================

class TrainingExperiment:
    """
    Enterprise training experiment orchestrator.

    Combines all enterprise features:
    - MLflow tracking
    - Reproducibility
    - Resource monitoring
    - Error handling
    - Statistical validation

    Usage:
        config = ExperimentConfig(
            name="EUR_USD_transformer",
            model_type="transformer",
            instrument="EUR_USD",
        )

        experiment = TrainingExperiment(config)

        with experiment:
            # Your training code
            experiment.log_metric("val_accuracy", 0.65)

        results = experiment.get_results()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        enable_mlflow: bool = True,
        enable_monitoring: bool = True,
        log_dir: Optional[str] = None,
    ):
        self.config = config
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.enable_monitoring = enable_monitoring and PSUTIL_AVAILABLE

        log_dir = log_dir or config.log_dir

        # Initialize components
        self.reproducibility = ReproducibilityManager()
        self.tracker = ExperimentTracker(
            experiment_name=config.name
        ) if self.enable_mlflow else None
        self.resource_monitor = ResourceMonitor() if self.enable_monitoring else None
        self.error_handler = TrainingErrorHandler(config.checkpoint_dir)
        self.data_validator = DataValidator()
        self.statistical_validator = StatisticalValidator()

        # Setup logging
        self.logger = setup_enterprise_logging(
            log_dir=log_dir,
            experiment_name=config.name,
        )

        # Results storage
        self._results: Dict[str, Any] = {
            'config': asdict(config),
            'metrics': {},
            'artifacts': [],
            'validation': {},
        }
        self._start_time: Optional[float] = None

    def __enter__(self):
        """Start experiment context."""
        self._start_time = time.time()

        # Set seeds
        seed_info = self.reproducibility.set_seeds()
        self._results['seed_info'] = seed_info

        # Log environment
        env_info = self.reproducibility.get_environment_fingerprint()
        self._results['environment'] = env_info

        # Start resource monitoring
        if self.resource_monitor:
            self.resource_monitor.start()

        # Start MLflow run
        if self.tracker:
            self._mlflow_context = self.tracker.start_run(
                run_name=f"{self.config.model_type}_{datetime.now().strftime('%H%M%S')}",
                config=self.config,
            )
            self._mlflow_context.__enter__()

        logger.info(f"🚀 Experiment started: {self.config.name}")
        logger.info(f"   Model: {self.config.model_type}")
        logger.info(f"   Instrument: {self.config.instrument}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End experiment context."""
        duration = time.time() - self._start_time if self._start_time else 0
        self._results['duration_seconds'] = duration

        # Stop resource monitoring
        if self.resource_monitor:
            resource_summary = self.resource_monitor.stop()
            self._results['resource_usage'] = resource_summary

        # End MLflow run
        if self.tracker and hasattr(self, '_mlflow_context'):
            self._mlflow_context.__exit__(exc_type, exc_val, exc_tb)

        # Save results
        self._save_results()

        status = "completed" if exc_type is None else f"failed ({exc_type.__name__})"
        logger.info(f"🏁 Experiment {status} in {duration:.1f}s")

        return False  # Don't suppress exceptions

    def validate_data(self, X: np.ndarray, y: np.ndarray) -> DataValidationResult:
        """Validate training data."""
        result = self.data_validator.validate(X)

        # Also validate labels
        y_result = self.data_validator.validate(y)
        result.checks['labels'] = y_result.checks
        result.warnings.extend(y_result.warnings)
        result.errors.extend(y_result.errors)

        self._results['validation']['data'] = asdict(result)

        if result.passed:
            logger.info("✅ Data validation passed")
        else:
            logger.error(f"❌ Data validation failed: {result.errors}")

        return result

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if name not in self._results['metrics']:
            self._results['metrics'][name] = []
        self._results['metrics'][name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat(),
        })

        if self.tracker:
            self.tracker.log_metric(name, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log epoch-level metrics."""
        all_metrics = {}

        for k, v in train_metrics.items():
            all_metrics[f'train_{k}'] = v

        for k, v in val_metrics.items():
            all_metrics[f'val_{k}'] = v

        self.log_metrics(all_metrics, step=epoch)

        # Update error handler epoch
        self.error_handler.current_epoch = epoch

    def compute_bootstrap_ci(
        self,
        metric_fn: Callable,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str = "accuracy",
    ) -> Dict[str, float]:
        """Compute bootstrap confidence interval for a metric."""
        ci = self.statistical_validator.bootstrap_confidence_interval(
            metric_fn, y_true, y_pred
        )

        self._results['validation'][f'{metric_name}_bootstrap'] = ci

        logger.info(
            f"📊 {metric_name}: {ci['mean']:.4f} "
            f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] (95% CI)"
        )

        return ci

    def get_results(self) -> Dict[str, Any]:
        """Get all experiment results."""
        return self._results.copy()

    def _save_results(self):
        """Save experiment results to file."""
        results_dir = Path(self.config.log_dir) / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{self.config.name}_{timestamp}.json"

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        results_json = json.loads(
            json.dumps(self._results, default=convert)
        )

        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"📄 Results saved: {results_file}")

        # Log as MLflow artifact
        if self.tracker and MLFLOW_AVAILABLE and mlflow.active_run():
            self.tracker.log_artifact(str(results_file))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_enterprise_training(
    model_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ExperimentConfig,
    fit_kwargs: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Run training with full enterprise features.

    This is a convenience function that handles:
    - Experiment tracking
    - Data validation
    - Training execution
    - Statistical validation

    Args:
        model_fn: Callable that returns a compiled model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Experiment configuration
        fit_kwargs: Additional kwargs for model.fit()

    Returns:
        Dictionary with all results and metrics
    """
    fit_kwargs = fit_kwargs or {}

    experiment = TrainingExperiment(config)

    with experiment:
        # Validate data
        experiment.validate_data(X_train, y_train)

        # Create model
        model = model_fn()

        # Log model summary
        if experiment.tracker:
            experiment.tracker.log_model_summary(model)

        # Train with error handling
        with experiment.error_handler.training_context(model, config.name):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.epochs,
                batch_size=config.batch_size,
                verbose=1,
                **fit_kwargs
            )

        # Log final metrics
        if hasattr(history, 'history'):
            final_metrics = {
                k: v[-1] if isinstance(v, list) else v
                for k, v in history.history.items()
            }
            experiment.log_metrics(final_metrics)

        # Compute bootstrap CI for accuracy
        y_pred = model.predict(X_val, verbose=0)

        def accuracy_fn(y_t, y_p):
            return np.mean((y_p > 0.5).astype(int) == y_t)

        experiment.compute_bootstrap_ci(
            accuracy_fn, y_val.flatten(), y_pred.flatten(), "accuracy"
        )

    return experiment.get_results()


# =============================================================================
# TRAINING REPORT GENERATOR
# =============================================================================

def generate_training_report(
    output_path: Union[str, Path],
    experiment_config: Optional[ExperimentConfig] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_paths: Optional[Dict[str, str]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
    include_plots: bool = False,
) -> Path:
    """
    Generate a comprehensive Markdown training report.

    Args:
        output_path: Path to save the report
        experiment_config: ExperimentConfig with training parameters
        metrics: Dictionary of model metrics (per-model)
        model_paths: Dictionary of model save paths
        validation_results: Results from statistical validation
        include_plots: Whether to embed plot references

    Returns:
        Path to the generated report
    """
    output_path = Path(output_path)

    timestamp = datetime.now()

    # Build report sections
    sections = []

    # Header
    sections.append(f"""# Enterprise Training Report

**Generated**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Report ID**: {hashlib.md5(str(timestamp).encode(), usedforsecurity=False).hexdigest()[:8]}
""")

    # Experiment Configuration
    if experiment_config:
        sections.append(f"""## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Name | {experiment_config.name} |
| Model Type | {experiment_config.model_type} |
| Instrument | {experiment_config.instrument} |
| Granularity | {experiment_config.granularity} |
| Epochs | {experiment_config.epochs} |
| Batch Size | {experiment_config.batch_size} |
| Learning Rate | {experiment_config.learning_rate} |
""")

    # Model Performance
    if metrics:
        sections.append("## Model Performance\n")

        for model_name, model_metrics in metrics.items():
            sections.append(f"### {model_name}\n")
            sections.append("| Metric | Value |")
            sections.append("|--------|-------|")

            for metric_name, metric_value in model_metrics.items():
                if isinstance(metric_value, float):
                    if 'accuracy' in metric_name.lower() or 'auc' in metric_name.lower():
                        formatted = f"{metric_value:.2%}"
                    elif metric_value < 0.01:
                        formatted = f"{metric_value:.2e}"
                    else:
                        formatted = f"{metric_value:.4f}"
                else:
                    formatted = str(metric_value)
                sections.append(f"| {metric_name} | {formatted} |")
            sections.append("")

    # Validation Results
    if validation_results:
        sections.append("## Statistical Validation\n")

        if 'bootstrap_ci' in validation_results:
            ci = validation_results['bootstrap_ci']
            sections.append(f"- **Bootstrap 95% CI**: [{ci['lower']:.2%}, {ci['upper']:.2%}]")
            sections.append(f"- **Bootstrap Mean**: {ci['mean']:.2%}")

        if 'cross_validation' in validation_results:
            cv = validation_results['cross_validation']
            sections.append(f"- **CV Mean**: {cv['mean']:.2%} ± {cv['std']:.2%}")
            sections.append(f"- **CV Folds**: {cv.get('n_folds', 'N/A')}")

        sections.append("")

    # Model Paths
    if model_paths:
        sections.append("## Saved Models\n")
        sections.append("| Model | Path |")
        sections.append("|-------|------|")
        for model_name, path in model_paths.items():
            sections.append(f"| {model_name} | `{path}` |")
        sections.append("")

    # Footer
    sections.append(f"""---
*Report generated by Enterprise Training Infrastructure*
*Python {platform.python_version()} on {platform.system()} {platform.machine()}*
""")

    # Write report
    report_content = "\n".join(sections)
    output_path.write_text(report_content)

    return output_path


# =============================================================================
# KERAS CALLBACK FOR ENTERPRISE LOGGING
# =============================================================================

class EnterpriseTrainingCallback:
    """
    Keras callback for enterprise logging during training.

    Integrates with TrainingExperiment for:
    - Per-epoch metric logging
    - Resource monitoring updates
    - Error handler epoch tracking
    """

    def __init__(self, experiment: TrainingExperiment):
        self.experiment = experiment

    def get_callback(self):
        """Get Keras callback instance."""
        import tensorflow as tf

        experiment = self.experiment

        class _Callback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}

                # Log to experiment
                train_metrics = {k: v for k, v in logs.items() if not k.startswith('val_')}
                val_metrics = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}

                experiment.log_epoch(epoch + 1, train_metrics, val_metrics)

        return _Callback()


if __name__ == "__main__":
    # Example usage
    print("Enterprise Training Infrastructure")
    print("=" * 50)

    # Create experiment config
    config = ExperimentConfig(
        name="test_experiment",
        model_type="transformer",
        instrument="EUR_USD",
        granularity="H1",
        epochs=10,
    )

    print(f"Config: {config}")

    # Test reproducibility
    repro = ReproducibilityManager()
    repro.set_seeds()
    print(f"\nEnvironment: {repro.get_environment_fingerprint()}")

    # Test data validation
    validator = DataValidator()
    test_data = np.random.randn(100, 10)
    result = validator.validate(test_data)
    print(f"\nData validation: {result}")

    print("\n✅ Enterprise training infrastructure ready!")
