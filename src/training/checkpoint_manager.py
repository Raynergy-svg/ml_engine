"""Checkpoint management for Buddy model training.

This module provides the CheckpointManager class for loading, saving, and
validating model checkpoints with proper metadata handling.

Handles:
- Loading Keras models with custom objects
- Keras 2 to Keras 3 migration (automatic)
- Checkpoint metadata management (.meta.json files)
- Input shape validation

Example usage:
    from src.training.checkpoint_manager import CheckpointManager, CheckpointMetadata
    
    manager = CheckpointManager(model_dir="trained_data/models")
    
    # Load a checkpoint
    model = manager.load_checkpoint(
        model_path="trained_data/models/EUR_USD/transformer_direction.keras",
        seq_len=60,
        feature_dim=42,
    )
    
    # Get metadata path
    meta_path = manager.get_meta_path("trained_data/models/buddy_tf.keras")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default metadata filename for Buddy checkpoints
BUDDY_META_FILENAME = "buddy_tf.meta.json"

# Default model directory
DEFAULT_MODEL_DIR = Path("trained_data/models")


@dataclass
class CheckpointMetadata:
    """Metadata associated with a model checkpoint.
    
    Attributes:
        created_at: Timestamp when checkpoint was created.
        instrument: Trading instrument the model was trained on.
        model_type: Model architecture (tcn, lstm, transformer).
        seq_len: Sequence length used during training.
        feature_dim: Number of input features.
        feature_names: List of feature names in order.
        epochs_trained: Number of epochs the model was trained.
        validation_accuracy: Best validation accuracy achieved.
        tier2_calibration: Tier-2 calibration data (optional).
        training_config: Full training configuration (optional).
        scaler_params: Scaler parameters for feature normalization.
        extra: Additional metadata fields.
    """
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    instrument: str = "GENERIC"
    model_type: str = "tcn"
    seq_len: int = 60
    feature_dim: int = 0
    feature_names: list[str] = field(default_factory=list)
    epochs_trained: int = 0
    validation_accuracy: float = 0.0
    tier2_calibration: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    scaler_params: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "created_at": self.created_at,
            "instrument": self.instrument,
            "model_type": self.model_type,
            "seq_len": self.seq_len,
            "feature_dim": self.feature_dim,
            "feature_names": self.feature_names,
            "epochs_trained": self.epochs_trained,
            "validation_accuracy": self.validation_accuracy,
            "tier2_calibration": self.tier2_calibration,
            "training_config": self.training_config,
            "scaler_params": self.scaler_params,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary (e.g., loaded from JSON)."""
        known_keys = {
            "created_at", "instrument", "model_type", "seq_len", "feature_dim",
            "feature_names", "epochs_trained", "validation_accuracy",
            "tier2_calibration", "training_config", "scaler_params",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        
        return cls(
            created_at=data.get("created_at", datetime.now().isoformat()),
            instrument=data.get("instrument", "GENERIC"),
            model_type=data.get("model_type", "tcn"),
            seq_len=data.get("seq_len", 60),
            feature_dim=data.get("feature_dim", 0),
            feature_names=data.get("feature_names", []),
            epochs_trained=data.get("epochs_trained", 0),
            validation_accuracy=data.get("validation_accuracy", 0.0),
            tier2_calibration=data.get("tier2_calibration"),
            training_config=data.get("training_config"),
            scaler_params=data.get("scaler_params"),
            extra=extra,
        )

    @classmethod
    def from_json_file(cls, path: Path | str) -> CheckpointMetadata:
        """Load metadata from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path | str) -> None:
        """Save metadata to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved checkpoint metadata to {path}")


class CheckpointManager:
    """Manages model checkpoint loading, saving, and metadata.
    
    Provides centralized handling for:
    - Loading Keras models with custom objects
    - Automatic Keras 2 to Keras 3 migration
    - Metadata file management
    - Input shape validation
    
    Attributes:
        model_dir: Base directory for model storage.
        custom_objects: Custom Keras objects for model loading.
    """
    
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        custom_objects: dict[str, Any] | None = None,
    ) -> None:
        """Initialize CheckpointManager.
        
        Args:
            model_dir: Base directory for model storage.
            custom_objects: Custom Keras objects for model loading.
                           If None, will attempt to load standard custom heads.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._custom_objects = custom_objects

    @property
    def custom_objects(self) -> dict[str, Any]:
        """Get custom objects for Keras model loading.
        
        Lazily loads custom head classes if not already provided.
        """
        if self._custom_objects is not None:
            return self._custom_objects
        
        # Try to load standard custom objects
        try:
            from ml_head_engine import MLEngineHead
            from mr_engine import MREngineHead
            from ms_head_engine import MSEngineHead
            from mt_engine import MTEngineHead
            from mx_head_engine import MXEngineHead
            
            self._custom_objects = {
                "MLEngineHead": MLEngineHead,
                "MREngineHead": MREngineHead,
                "MSEngineHead": MSEngineHead,
                "MTEngineHead": MTEngineHead,
                "MXEngineHead": MXEngineHead,
            }
        except ImportError as e:
            logger.warning(f"Could not load custom head classes: {e}")
            self._custom_objects = {}
        
        return self._custom_objects

    def get_meta_path(self, model_path: str | Path) -> Path | None:
        """Get the metadata file path for a checkpoint.
        
        Searches for metadata in the following order:
        1. Same path with .meta.json suffix
        2. Default Buddy metadata location (for buddy_tf.keras)
        
        Args:
            model_path: Path to the model checkpoint file.
            
        Returns:
            Path to metadata file if found, None otherwise.
        """
        p = Path(model_path)
        
        # Try standard .meta.json suffix
        try:
            if p.suffix == ".keras":
                candidate = p.with_suffix(".meta.json")
                if candidate.exists():
                    return candidate
        except Exception:
            pass

        # Special-case: default Buddy checkpoint keeps metadata in buddy_tf.meta.json
        try:
            default_model = self.model_dir / "buddy_tf.keras"
            if p.resolve() == default_model.resolve():
                candidate = self.model_dir / BUDDY_META_FILENAME
                if candidate.exists():
                    return candidate
        except Exception:
            pass
        
        return None

    def load_metadata(self, model_path: str | Path) -> CheckpointMetadata | None:
        """Load metadata for a checkpoint if available.
        
        Args:
            model_path: Path to the model checkpoint file.
            
        Returns:
            CheckpointMetadata if found, None otherwise.
        """
        meta_path = self.get_meta_path(model_path)
        if meta_path is None:
            return None
        
        try:
            return CheckpointMetadata.from_json_file(meta_path)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {meta_path}: {e}")
            return None

    def load_checkpoint(
        self,
        model_path: str | Path,
        seq_len: int,
        feature_dim: int,
        validate_shape: bool = True,
    ) -> Any:
        """Load a model checkpoint with validation.
        
        Args:
            model_path: Path to the model checkpoint file.
            seq_len: Expected sequence length.
            feature_dim: Expected feature dimension.
            validate_shape: Whether to validate input shape matches expectations.
            
        Returns:
            Loaded Keras model.
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ValueError: If input shape validation fails.
        """
        import tensorflow as tf
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        logger.info(f"Loading checkpoint: {model_path}")
        
        # Try standard load first
        try:
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
                custom_objects=self.custom_objects,
            )
        except Exception as e:
            err_str = str(e)
            # Handle Keras 2 → Keras 3 incompatibility
            if "keras.src.engine.functional" in err_str or "cannot be imported" in err_str:
                # Attempt weights-only migration
                logger.warning(f"Keras 2 model detected, attempting migration: {model_path}")
                model = self._migrate_keras2_to_keras3(
                    str(model_path), seq_len, feature_dim, self.custom_objects
                )
            else:
                raise

        # Validate input shape if requested
        if validate_shape:
            in_shape = tuple(getattr(model, "input_shape", ()) or ())
            if len(in_shape) >= 3:
                if int(in_shape[1]) != int(seq_len) or int(in_shape[2]) != int(feature_dim):
                    raise ValueError(
                        f"Checkpoint input_shape {in_shape} incompatible with "
                        f"seq_len={seq_len}, feature_dim={feature_dim}"
                    )
        
        logger.info(f"Successfully loaded checkpoint: {model_path}")
        return model

    def _migrate_keras2_to_keras3(
        self,
        model_path: str,
        seq_len: int,
        feature_dim: int,
        custom_objects: dict[str, Any],
    ) -> Any:
        """Migrate a Keras 2 model to Keras 3 format.
        
        This handles the transition from Keras 2 (TF 2.x) to Keras 3 (TF 2.16+)
        by extracting weights and rebuilding the model architecture.
        
        Args:
            model_path: Path to the Keras 2 model file.
            seq_len: Sequence length for the model.
            feature_dim: Feature dimension for the model.
            custom_objects: Custom Keras objects.
            
        Returns:
            Migrated Keras 3 model.
            
        Raises:
            ValueError: If migration fails.
        """
        import shutil
        import tempfile
        import zipfile
        
        import tensorflow as tf
        from tensorflow.keras import layers
        
        from rich.console import Console
        console = Console()
        
        try:
            console.print(f"[yellow]Attempting Keras 2→3 migration for: {model_path}[/yellow]")
            
            # Extract weights from old archive
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(model_path, 'r') as zf:
                    zf.extractall(tmpdir)
                
                weights_path = Path(tmpdir) / "model.weights.h5"
                if not weights_path.exists():
                    # Try alternate path
                    weights_path = Path(tmpdir) / "variables" / "variables.data-00000-of-00001"
                    if not weights_path.exists():
                        raise ValueError("Cannot find weights in old model archive")
            
            # Build fresh model with same architecture
            inp = layers.Input(shape=(seq_len, feature_dim), name='input_features')
            
            # TCN-style encoder
            x = layers.Conv1D(64, 3, padding='causal', activation='relu')(inp)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=4)(x)
            x = layers.GlobalAveragePooling1D()(x)
            
            # Multi-head outputs
            dir_branch = layers.Dense(32, activation='relu')(x)
            price_branch = layers.Dense(32, activation='relu')(x)
            risk_branch = layers.Dense(32, activation='relu')(x)
            state_branch = layers.Dense(32, activation='relu')(x)
            trend_branch = layers.Dense(32, activation='relu')(x)
            
            direction = layers.Dense(1, activation='sigmoid', name='direction', dtype='float32')(dir_branch)
            price = layers.Dense(1, activation='linear', name='price', dtype='float32')(price_branch)
            risk = layers.Dense(1, activation='sigmoid', name='risk', dtype='float32')(risk_branch)
            state_logits = layers.Dense(2, activation='softmax', name='state_logits', dtype='float32')(state_branch)
            trend = layers.Dense(1, activation='linear', name='trend', dtype='float32')(trend_branch)
            
            fresh_model = tf.keras.Model(
                inputs=inp,
                outputs={
                    'direction': direction,
                    'price': price,
                    'risk': risk,
                    'state_logits': state_logits,
                    'trend': trend,
                },
                name='MetalOptimizedTCN',
            )

            # Try to load weights
            try:
                fresh_model.load_weights(str(weights_path))
                console.print("[green]✓ Weights migrated successfully[/green]")

                # Save as new Keras 3 model (backup old first)
                backup_path = model_path + ".keras2_backup"
                if not Path(backup_path).exists():
                    shutil.copy(model_path, backup_path)
                    console.print(f"[dim]Backup saved: {backup_path}[/dim]")
                
                fresh_model.save(model_path)
                console.print(f"[green]✓ Model migrated to Keras 3: {model_path}[/green]")
                return fresh_model

            except Exception as weight_err:
                console.print(f"[yellow]Weight loading failed (architecture mismatch): {weight_err}[/yellow]")
                raise ValueError(f"Cannot migrate: architecture incompatible. Delete old model and retrain.")

        except zipfile.BadZipFile:
            raise ValueError(f"Model file {model_path} is not a valid .keras archive")

    def save_checkpoint(
        self,
        model: Any,
        checkpoint_path: str | Path,
        metadata: CheckpointMetadata | None = None,
    ) -> Path:
        """Save a model checkpoint with optional metadata.
        
        Args:
            model: Keras model to save.
            checkpoint_path: Path to save the checkpoint.
            metadata: Optional metadata to save alongside.
            
        Returns:
            Path to the saved checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model.save(str(checkpoint_path))
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save metadata if provided
        if metadata is not None:
            meta_path = checkpoint_path.with_suffix(".meta.json")
            metadata.save(meta_path)
        
        return checkpoint_path

    def list_checkpoints(self, pattern: str = "*.keras") -> list[Path]:
        """List all checkpoints matching a pattern.
        
        Args:
            pattern: Glob pattern for checkpoint files.
            
        Returns:
            List of checkpoint paths.
        """
        return sorted(self.model_dir.rglob(pattern))

    def get_pair_checkpoint_path(
        self,
        instrument: str,
        model_type: str = "transformer",
        filename: str = "direction",
    ) -> Path:
        """Get the checkpoint path for a specific trading pair.
        
        Args:
            instrument: Trading instrument (e.g., "EUR_USD").
            model_type: Model architecture type.
            filename: Base filename for the checkpoint.
            
        Returns:
            Path to the checkpoint file.
        """
        if instrument and instrument != "GENERIC":
            pair_dir = self.model_dir / instrument
        else:
            pair_dir = self.model_dir
        
        pair_dir.mkdir(parents=True, exist_ok=True)
        
        return pair_dir / f"{model_type}_{filename}.keras"


def load_buddy_checkpoint(
    *,
    model_path: str,
    seq_len: int,
    feature_dim: int,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> Any:
    """Convenience function to load a Buddy checkpoint.
    
    This is a backward-compatible function that wraps CheckpointManager.
    
    Args:
        model_path: Path to the model checkpoint.
        seq_len: Expected sequence length.
        feature_dim: Expected feature dimension.
        model_dir: Base model directory.
        
    Returns:
        Loaded Keras model.
    """
    manager = CheckpointManager(model_dir=model_dir)
    return manager.load_checkpoint(model_path, seq_len, feature_dim)


def meta_path_for_checkpoint(model_path: str) -> Path | None:
    """Get metadata path for a checkpoint (backward compatibility).
    
    Args:
        model_path: Path to the checkpoint.
        
    Returns:
        Path to metadata file if found, None otherwise.
    """
    manager = CheckpointManager()
    return manager.get_meta_path(model_path)


__all__ = [
    "BUDDY_META_FILENAME",
    "DEFAULT_MODEL_DIR",
    "CheckpointMetadata",
    "CheckpointManager",
    "load_buddy_checkpoint",
    "meta_path_for_checkpoint",
]
