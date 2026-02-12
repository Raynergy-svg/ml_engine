"""
Model Registry and Versioning System for ML Engine.

This module provides centralized model artifact management with:
- Version tracking with semantic versioning
- Model metadata (hyperparameters, metrics, datasets)
- Git commit hash tracking for reproducibility
- Model promotion workflow (dev → staging → production)
- MLflow integration for experiment tracking
- Comparison utilities for model versions

Example:
    >>> from src.utils.model_registry import ModelRegistry, ModelMetadata
    >>> registry = ModelRegistry()
    >>> metadata = ModelMetadata(
    ...     model_name="transformer_direction",
    ...     version="1.0.0",
    ...     hyperparameters={"d_model": 32, "num_heads": 4},
    ...     metrics={"val_loss": 0.45, "accuracy": 0.78},
    ... )
    >>> registry.register_model(metadata, model_path="trained_data/models/model.keras")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.debug("MLflow not available. Model registry will work without experiment tracking.")


@dataclass
class ModelMetadata:
    """
    Metadata for a trained model.
    
    Attributes:
        model_name: Name of the model (e.g., "transformer_direction")
        version: Semantic version string (e.g., "1.0.0")
        timestamp: ISO 8601 timestamp of model creation
        hyperparameters: Dictionary of hyperparameters used
        metrics: Dictionary of evaluation metrics
        dataset_hash: Hash of training dataset for reproducibility
        git_commit: Git commit hash when model was trained
        environment: Python environment information
        description: Human-readable model description
        tags: Optional tags for categorization
        parent_version: Optional parent version for incremental training
    """
    model_name: str
    version: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    dataset_hash: Optional[str] = None
    git_commit: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        """Create metadata from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> ModelMetadata:
        """Create metadata from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ModelRegistry:
    """
    Centralized registry for model artifacts.
    
    Provides version control, metadata tracking, and model promotion
    workflow for ML models.
    
    Example:
        >>> registry = ModelRegistry(registry_path="trained_data/models")
        >>> # Register a new model
        >>> metadata = ModelMetadata(
        ...     model_name="transformer",
        ...     version="1.0.0",
        ...     metrics={"val_loss": 0.45}
        ... )
        >>> registry.register_model(metadata, "model.keras")
        >>> 
        >>> # List all versions
        >>> versions = registry.list_versions("transformer")
        >>> 
        >>> # Get best model by metric
        >>> best_model = registry.get_best_model("transformer", metric="val_loss")
    """
    
    def __init__(
        self,
        registry_path: str = "trained_data/models",
        enable_mlflow: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Initialize model registry.
        
        Args:
            registry_path: Base path for model storage
            enable_mlflow: Whether to enable MLflow integration
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_dir = self.registry_path / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        if self.enable_mlflow:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.mlflow_client = MlflowClient()
        else:
            self.mlflow_client = None
        
        if enable_mlflow and not MLFLOW_AVAILABLE:
            logger.warning("MLflow requested but not available. Install with: pip install mlflow")
    
    def _get_metadata_path(self, model_name: str, version: str) -> Path:
        """Get path to metadata file for a model version."""
        return self.metadata_dir / f"{model_name}_v{version}.json"
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.registry_path,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Could not get git commit hash")
            return None
    
    def _compute_dataset_hash(self, dataset_path: Optional[str]) -> Optional[str]:
        """Compute hash of training dataset."""
        if dataset_path is None:
            return None
        
        path = Path(dataset_path)
        if not path.exists():
            logger.warning(f"Dataset file not found: {dataset_path}")
            return None
        
        try:
            hasher = hashlib.sha256()
            with open(path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute dataset hash: {e}")
            return None
    
    def register_model(
        self,
        metadata: ModelMetadata,
        model_path: str,
        dataset_path: Optional[str] = None,
        stage: str = "dev",
    ) -> None:
        """
        Register a new model version in the registry.
        
        Args:
            metadata: Model metadata
            model_path: Path to the model file
            dataset_path: Optional path to training dataset for hashing
            stage: Model stage ('dev', 'staging', 'production')
            
        Example:
            >>> metadata = ModelMetadata(
            ...     model_name="transformer",
            ...     version="1.0.0",
            ...     hyperparameters={"d_model": 32},
            ...     metrics={"val_loss": 0.45},
            ... )
            >>> registry.register_model(metadata, "model.keras")
        """
        # Enrich metadata with auto-tracked information
        if metadata.git_commit is None:
            metadata.git_commit = self._get_git_commit()
        
        if metadata.dataset_hash is None and dataset_path:
            metadata.dataset_hash = self._compute_dataset_hash(dataset_path)
        
        # Add environment information
        if not metadata.environment:
            import platform
            import sys
            metadata.environment = {
                "python_version": sys.version,
                "platform": platform.platform(),
            }
            
            # Add TensorFlow version if available
            try:
                import tensorflow as tf
                metadata.environment["tensorflow_version"] = tf.__version__
            except ImportError:
                pass
        
        # Save metadata to file
        metadata_path = self._get_metadata_path(metadata.model_name, metadata.version)
        with open(metadata_path, 'w') as f:
            f.write(metadata.to_json())
        
        logger.info(f"✅ Registered model {metadata.model_name} v{metadata.version} at {stage} stage")
        
        # Log to MLflow if enabled
        if self.enable_mlflow:
            self._log_to_mlflow(metadata, model_path, stage)
    
    def _log_to_mlflow(
        self,
        metadata: ModelMetadata,
        model_path: str,
        stage: str,
    ) -> None:
        """Log model to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{metadata.model_name}_v{metadata.version}"):
                # Log parameters
                for key, value in metadata.hyperparameters.items():
                    mlflow.log_param(key, value)
                
                # Log metrics
                for key, value in metadata.metrics.items():
                    mlflow.log_metric(key, value)
                
                # Log tags
                mlflow.set_tags({
                    "model_name": metadata.model_name,
                    "version": metadata.version,
                    "stage": stage,
                    **metadata.tags,
                })
                
                if metadata.git_commit:
                    mlflow.set_tag("git_commit", metadata.git_commit)
                
                # Log model artifact
                if Path(model_path).exists():
                    mlflow.log_artifact(model_path)
                
                logger.info(f"Logged to MLflow: {metadata.model_name} v{metadata.version}")
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")
    
    def get_metadata(self, model_name: str, version: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Model metadata or None if not found
        """
        metadata_path = self._get_metadata_path(model_name, version)
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return ModelMetadata.from_json(f.read())
    
    def list_versions(self, model_name: str) -> List[ModelMetadata]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of metadata for all versions, sorted by timestamp
        """
        versions = []
        pattern = f"{model_name}_v*.json"
        
        for metadata_file in self.metadata_dir.glob(pattern):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = ModelMetadata.from_json(f.read())
                    versions.append(metadata)
            except Exception as e:
                logger.warning(f"Could not load metadata from {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda m: m.timestamp, reverse=True)
        return versions
    
    def get_best_model(
        self,
        model_name: str,
        metric: str,
        higher_is_better: bool = False,
    ) -> Optional[ModelMetadata]:
        """
        Get the best model version based on a metric.
        
        Args:
            model_name: Name of the model
            metric: Metric name to optimize
            higher_is_better: Whether higher values are better
            
        Returns:
            Metadata of best model or None if no models found
            
        Example:
            >>> # Get model with lowest validation loss
            >>> best = registry.get_best_model("transformer", "val_loss")
            >>> 
            >>> # Get model with highest accuracy
            >>> best = registry.get_best_model("transformer", "accuracy", higher_is_better=True)
        """
        versions = self.list_versions(model_name)
        
        if not versions:
            return None
        
        # Filter versions that have the requested metric
        valid_versions = [v for v in versions if metric in v.metrics]
        
        if not valid_versions:
            logger.warning(f"No versions of {model_name} have metric '{metric}'")
            return None
        
        # Find best version
        if higher_is_better:
            best = max(valid_versions, key=lambda v: v.metrics[metric])
        else:
            best = min(valid_versions, key=lambda v: v.metrics[metric])
        
        return best
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Dictionary with comparison results
        """
        meta1 = self.get_metadata(model_name, version1)
        meta2 = self.get_metadata(model_name, version2)
        
        if meta1 is None or meta2 is None:
            raise ValueError(f"One or both versions not found: {version1}, {version2}")
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "hyperparameter_changes": {},
            "metric_changes": {},
        }
        
        # Compare hyperparameters
        all_params = set(meta1.hyperparameters.keys()) | set(meta2.hyperparameters.keys())
        for param in all_params:
            val1 = meta1.hyperparameters.get(param)
            val2 = meta2.hyperparameters.get(param)
            if val1 != val2:
                comparison["hyperparameter_changes"][param] = {
                    "v1": val1,
                    "v2": val2,
                }
        
        # Compare metrics
        all_metrics = set(meta1.metrics.keys()) | set(meta2.metrics.keys())
        for metric in all_metrics:
            val1 = meta1.metrics.get(metric)
            val2 = meta2.metrics.get(metric)
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else 0
                comparison["metric_changes"][metric] = {
                    "v1": val1,
                    "v2": val2,
                    "diff": diff,
                    "pct_change": pct_change,
                }
        
        return comparison
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        from_stage: str,
        to_stage: str,
    ) -> None:
        """
        Promote a model from one stage to another.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            from_stage: Current stage
            to_stage: Target stage
            
        Example:
            >>> # Promote from dev to staging
            >>> registry.promote_model("transformer", "1.0.0", "dev", "staging")
            >>> 
            >>> # Promote to production
            >>> registry.promote_model("transformer", "1.0.0", "staging", "production")
        """
        metadata = self.get_metadata(model_name, version)
        if metadata is None:
            raise ValueError(f"Model {model_name} v{version} not found")
        
        # Update tags with new stage
        metadata.tags["stage"] = to_stage
        metadata.tags["promoted_from"] = from_stage
        metadata.tags["promoted_at"] = datetime.now(timezone.utc).isoformat()
        
        # Save updated metadata
        metadata_path = self._get_metadata_path(model_name, version)
        with open(metadata_path, 'w') as f:
            f.write(metadata.to_json())
        
        logger.info(f"✅ Promoted {model_name} v{version} from {from_stage} to {to_stage}")
        
        # Update MLflow if enabled
        if self.enable_mlflow:
            try:
                # Find the model in MLflow and update stage
                # Note: This requires MLflow model registry
                pass  # Implementation depends on MLflow setup
            except Exception as e:
                logger.warning(f"Could not update MLflow stage: {e}")


def create_model_metadata_from_training(
    model_name: str,
    version: str,
    trainer_config: Any,
    metrics: Dict[str, float],
    description: str = "",
) -> ModelMetadata:
    """
    Create model metadata from training configuration and results.
    
    Args:
        model_name: Name of the model
        version: Version string
        trainer_config: Training configuration object
        metrics: Dictionary of evaluation metrics
        description: Model description
        
    Returns:
        ModelMetadata instance
        
    Example:
        >>> from src.training.modular_trainers import TrainerConfig
        >>> config = TrainerConfig(d_model=32, num_heads=4)
        >>> metrics = {"val_loss": 0.45, "accuracy": 0.78}
        >>> metadata = create_model_metadata_from_training(
        ...     "transformer", "1.0.0", config, metrics
        ... )
    """
    # Extract hyperparameters from config
    hyperparameters = {}
    if hasattr(trainer_config, '__dict__'):
        hyperparameters = {
            k: v for k, v in trainer_config.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
    
    return ModelMetadata(
        model_name=model_name,
        version=version,
        hyperparameters=hyperparameters,
        metrics=metrics,
        description=description,
    )


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # Create and register a model
    metadata = ModelMetadata(
        model_name="transformer_direction",
        version="1.0.0",
        hyperparameters={
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.2,
        },
        metrics={
            "val_loss": 0.45,
            "val_accuracy": 0.78,
        },
        description="Initial production model",
    )
    
    print(f"Created metadata: {metadata.model_name} v{metadata.version}")
    print(f"Metrics: {metadata.metrics}")
