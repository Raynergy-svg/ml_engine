"""
Test the exact meta-labeler training flow with improved structure and best practices.

This module provides a comprehensive test suite for meta-labeler training,
including proper error handling, logging, and validation.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom Exceptions
class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class DataGenerationError(Exception):
    """Raised when test data generation fails."""
    pass


class PredictionError(Exception):
    """Raised when prediction computation fails."""
    pass


class MetaLabelerTrainingError(Exception):
    """Raised when meta-labeler training fails."""
    pass


@dataclass
class TestConfig:
    """Configuration for meta-labeler testing."""
    
    model_path: str
    n_train: int = 500
    n_val: int = 100
    batch_size: int = 32
    min_confidence_threshold: float = 0.55
    n_estimators: int = 100
    max_depth: int = 2
    learning_rate: float = 0.05
    use_reduced_features: bool = True
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_train <= 0 or self.n_val <= 0:
            raise ValueError("n_train and n_val must be positive")
        if not 0 < self.min_confidence_threshold < 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def setup_environment() -> None:
    """Setup the Python environment and import dependencies."""
    try:
        # Add project path to sys.path
        project_path = Path('/Users/davidcertan/Desktop/ml_engine')
        if str(project_path) not in sys.path:
            sys.path.insert(0, str(project_path))
        
        # Import TensorFlow and Keras
        import tensorflow as tf
        from tensorflow import keras
        
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"Keras version: {keras.__version__}")
        
    except ImportError as e:
        logger.error(f"Failed to import required dependencies: {e}")
        raise


def validate_model_path(model_path: Path) -> None:
    """
    Validate that the model file exists and is readable.
    
    Args:
        model_path: Path to the model file
        
    Raises:
        ModelLoadError: If model file doesn't exist or is not readable
    """
    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")
    if not model_path.is_file():
        raise ModelLoadError(f"Model path is not a file: {model_path}")
    if not model_path.suffix == '.keras':
        logger.warning(f"Model file doesn't have .keras extension: {model_path}")


def load_model(model_path: Path) -> Any:
    """
    Load a Keras model from the specified path with validation.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded Keras model
        
    Raises:
        ModelLoadError: If model loading fails
    """
    validate_model_path(model_path)
    
    try:
        from tensorflow import keras
        model = keras.models.load_model(str(model_path))
        
        logger.info("Model loaded successfully")
        logger.info(f"  Input shape: {model.input_shape}")
        logger.info(f"  Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise ModelLoadError(f"Model loading failed: {e}") from e


def generate_test_data(
    model: Any,
    config: TestConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test data matching the model's expected input shape.
    
    Args:
        model: Keras model to generate data for
        config: Test configuration
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
        
    Raises:
        DataGenerationError: If data generation fails
    """
    try:
        # Extract model dimensions
        n_features = model.input_shape[-1]
        seq_len = model.input_shape[1]
        
        logger.info("Generating test data:")
        logger.info(f"  Sequence length: {seq_len}")
        logger.info(f"  Number of features: {n_features}")
        logger.info(f"  Training samples: {config.n_train}")
        logger.info(f"  Validation samples: {config.n_val}")
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            logger.info(f"Random seed set to: {config.random_seed}")
        
        # Generate training data
        X_train = np.random.randn(
            config.n_train, seq_len, n_features
        ).astype(np.float32)
        y_train = np.random.randint(
            0, 2, (config.n_train,)
        ).astype(np.float32)
        
        # Generate validation data
        X_val = np.random.randn(
            config.n_val, seq_len, n_features
        ).astype(np.float32)
        y_val = np.random.randint(
            0, 2, (config.n_val,)
        ).astype(np.float32)
        
        logger.info("Test data generated successfully")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  y_train shape: {y_train.shape}")
        logger.info(f"  X_val shape: {X_val.shape}")
        logger.info(f"  y_val shape: {y_val.shape}")
        
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        logger.error(f"Failed to generate test data: {e}")
        raise DataGenerationError(f"Data generation failed: {e}") from e


def process_predictions(raw_probs: np.ndarray) -> np.ndarray:
    """
    Process raw model predictions into probability values.
    
    Handles various output shapes from the model:
    - 3D shape (batch, seq_len, 1): Take last timestep
    - 2D shape (batch, seq_len): Flatten if seq_len > 1
    - 1D shape: Return as-is
    
    Args:
        raw_probs: Raw predictions from model.predict()
        
    Returns:
        Processed 1D array of probabilities
        
    Raises:
        PredictionError: If processing fails
    """
    try:
        logger.debug(f"Processing predictions with shape: {raw_probs.shape}")
        
        if raw_probs.ndim == 3:
            # 3D output: (batch, seq_len, features)
            # Take the last timestep
            probs = raw_probs[:, -1]
            if probs.ndim > 1:
                probs = probs.flatten()
        elif raw_probs.ndim == 2:
            # 2D output: (batch, seq_len) or (batch, 1)
            if raw_probs.shape[1] > 1:
                # Take last timestep
                probs = raw_probs[:, -1]
            else:
                # Flatten single feature
                probs = raw_probs.flatten()
        else:
            # 1D output
            probs = raw_probs.flatten()
        
        # Validate probability values
        if np.any(np.isnan(probs)):
            logger.warning("NaN values detected in predictions")
            probs = np.nan_to_num(probs, nan=0.5)
        
        if np.any(probs < 0) or np.any(probs > 1):
            logger.warning("Predictions outside [0, 1] range detected")
            probs = np.clip(probs, 0, 1)
        
        logger.debug(f"Processed predictions shape: {probs.shape}")
        return probs
        
    except Exception as e:
        logger.error(f"Failed to process predictions: {e}")
        raise PredictionError(f"Prediction processing failed: {e}") from e


def compute_probabilities(
    model: Any,
    X: np.ndarray,
    batch_size: int = 32,
    data_name: str = "data"
) -> np.ndarray:
    """
    Compute probabilities from model predictions.
    
    Args:
        model: Keras model
        X: Input data
        batch_size: Batch size for prediction
        data_name: Name for logging purposes
        
    Returns:
        1D array of probabilities
        
    Raises:
        PredictionError: If computation fails
    """
    try:
        logger.info(f"Computing probabilities for {data_name}")
        
        raw_probs = model.predict(X, batch_size=batch_size, verbose=0)
        probs = process_predictions(raw_probs)
        
        logger.info(f"  {data_name} probabilities shape: {probs.shape}")
        logger.info(f"  {data_name} probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
        logger.info(f"  {data_name} probabilities mean: {probs.mean():.4f}")
        
        return probs
        
    except Exception as e:
        logger.error(f"Failed to compute probabilities for {data_name}: {e}")
        raise PredictionError(f"Probability computation failed: {e}") from e


def train_meta_labeler_wrapper(
    X_train: np.ndarray,
    y_train: np.ndarray,
    primary_probs_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    primary_probs_val: np.ndarray,
    config: TestConfig
) -> Tuple[Any, Dict[str, Any]]:
    """
    Wrapper function for training the meta-labeler with proper error handling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        primary_probs_train: Primary model probabilities for training
        X_val: Validation features
        y_val: Validation labels
        primary_probs_val: Primary model probabilities for validation
        config: Test configuration
        
    Returns:
        Tuple of (meta_labeler, metrics)
        
    Raises:
        MetaLabelerTrainingError: If training fails
    """
    try:
        logger.info("Training meta-labeler")
        logger.info(
            f"  Configuration: n_estimators={config.n_estimators}, "
            f"max_depth={config.max_depth}, "
            f"learning_rate={config.learning_rate}"
        )
        
        # Import here to avoid unnecessary imports if not needed
        from src.training.meta_labeling import train_meta_labeler, MetaLabelingConfig
        
        meta_config = MetaLabelingConfig(
            use_xgboost=True,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_confidence_threshold=config.min_confidence_threshold,
            use_reduced_features=config.use_reduced_features,
        )
        
        meta_labeler, meta_labeler_metrics = train_meta_labeler(
            X_train=X_train,
            y_train=y_train,
            primary_probs_train=primary_probs_train,
            X_val=X_val,
            y_val=y_val,
            primary_probs_val=primary_probs_val,
            config=meta_config,
        )
        
        logger.info("Meta-labeler trained successfully")
        logger.info(f"  Metrics: {meta_labeler_metrics}")
        
        return meta_labeler, meta_labeler_metrics
        
    except Exception as e:
        logger.error(f"Meta-labeler training failed: {e}")
        raise MetaLabelerTrainingError(f"Training failed: {e}") from e


def run_meta_labeler_test(config: TestConfig) -> Dict[str, Any]:
    """
    Run the complete meta-labeler test workflow.
    
    Args:
        config: Test configuration
        
    Returns:
        Dictionary containing test results and metrics
        
    Raises:
        Various exceptions if any step fails
    """
    results: Dict[str, Any] = {
        'success': False,
        'errors': [],
        'metrics': {}
    }
    
    try:
        # Step 0: Setup environment
        logger.info("=" * 60)
        logger.info("Starting meta-labeler test workflow")
        logger.info("=" * 60)
        
        setup_environment()
        
        # Step 1: Load model
        logger.info("\n--- Step 1: Loading Model ---")
        model_path = Path(config.model_path)
        model = load_model(model_path)
        results['model'] = {
            'path': str(model_path),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
        
        # Step 2: Generate test data
        logger.info("\n--- Step 2: Generating Test Data ---")
        X_train, y_train, X_val, y_val = generate_test_data(model, config)
        results['data'] = {
            'n_train': config.n_train,
            'n_val': config.n_val,
            'X_train_shape': X_train.shape,
            'X_val_shape': X_val.shape
        }
        
        # Step 3: Compute validation probabilities
        logger.info("\n--- Step 3: Computing Validation Probabilities ---")
        val_probs = compute_probabilities(
            model, X_val, config.batch_size, "validation"
        )
        results['val_probs'] = {
            'shape': val_probs.shape,
            'mean': float(val_probs.mean()),
            'std': float(val_probs.std()),
            'min': float(val_probs.min()),
            'max': float(val_probs.max())
        }
        
        # Step 4: Compute training probabilities
        logger.info("\n--- Step 4: Computing Training Probabilities ---")
        train_probs = compute_probabilities(
            model, X_train, config.batch_size, "training"
        )
        results['train_probs'] = {
            'shape': train_probs.shape,
            'mean': float(train_probs.mean()),
            'std': float(train_probs.std()),
            'min': float(train_probs.min()),
            'max': float(train_probs.max())
        }
        
        # Step 5: Train meta-labeler
        logger.info("\n--- Step 5: Training Meta-Labeler ---")
        meta_labeler, meta_labeler_metrics = train_meta_labeler_wrapper(
            X_train=X_train,
            y_train=y_train,
            primary_probs_train=train_probs,
            X_val=X_val,
            y_val=y_val,
            primary_probs_val=val_probs,
            config=config
        )
        results['meta_labeler'] = meta_labeler_metrics
        results['success'] = True
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Test workflow completed successfully!")
        logger.info("=" * 60)
        logger.info("Summary:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Training samples: {config.n_train}")
        logger.info(f"  Validation samples: {config.n_val}")
        logger.info(f"  Meta-labeler metrics: {meta_labeler_metrics}")
        
        return results
        
    except Exception as e:
        error_msg = f"Test workflow failed: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
        results['success'] = False
        raise


def main() -> None:
    """Main entry point for the test script."""
    try:
        # Create configuration
        config = TestConfig(
            model_path='/Users/davidcertan/Desktop/ml_engine/trained_data/models/GBP_USD/transformer_direction.keras',
            n_train=500,
            n_val=100,
            batch_size=32,
            min_confidence_threshold=0.55,
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            use_reduced_features=True,
            random_seed=42  # For reproducibility
        )
        
        # Run test workflow
        results = run_meta_labeler_test(config)
        
        # Exit with appropriate code
        sys.exit(0 if results['success'] else 1)
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
