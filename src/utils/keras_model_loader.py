"""
Keras Model Loader - Cross-Version Compatibility Solution

This module provides comprehensive solutions for loading Keras 2.15.0 models in a Keras 3.x environment.
It implements multiple approaches to handle the module path incompatibility issues that arise when
loading models saved with older Keras versions in newer environments.

Author: ML Engine Team
Date: 2025-01-29
"""

import json
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import zipfile

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _compact_error(exc: Exception, max_len: int = 320) -> str:
    """Return a compact single-line error message for logging."""
    msg = str(exc).replace("\n", " ").strip()
    if len(msg) > max_len:
        return msg[: max_len - 3] + "..."
    return msg


_EXPECTED_COMPAT_ERROR_FRAGMENTS = (
    "could not deserialize class 'functional' because its parent module "
    "keras.src.engine.functional cannot be imported",
    "could not deserialize class 'functional' because its parent module "
    "keras.src.models.functional cannot be imported",
    "keras.src.engine.functional",
    "keras.src.models.functional",
)


def _is_expected_compat_load_error(err: Union[str, Exception]) -> bool:
    """True for known cross-version Keras deserialization mismatches."""
    msg = str(err).lower()
    return any(fragment in msg for fragment in _EXPECTED_COMPAT_ERROR_FRAGMENTS)


def _model_prefers_tf_keras(model_metadata: Dict[str, Any]) -> bool:
    """Infer whether model was likely saved with Keras 2.x/tf.keras serialization."""
    saved_version = str(model_metadata.get('keras_version') or "").strip()
    if saved_version.startswith("2."):
        return True

    cfg = model_metadata.get('config')
    cfg_blob = json.dumps(cfg, ensure_ascii=True).lower() if cfg is not None else ""
    return "keras.src.engine." in cfg_blob or "tensorflow.python.keras" in cfg_blob


def _model_prefers_keras_native(model_metadata: Dict[str, Any]) -> bool:
    """Infer whether model was likely saved with native Keras 3 serialization."""
    saved_version = str(model_metadata.get('keras_version') or "").strip()
    if saved_version.startswith("3."):
        return True

    cfg = model_metadata.get('config')
    cfg_blob = json.dumps(cfg, ensure_ascii=True).lower() if cfg is not None else ""
    return "keras.src.models." in cfg_blob


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_keras_version() -> str:
    """
    Detect the current Keras version string.

    This is a convenience function that returns only the version string.
    For detailed version information, use detect_keras_version_info().

    Returns:
        String containing the Keras version (e.g., '3.0.5', '2.15.0')

    Raises:
        ImportError: If Keras is not installed

    Example:
        >>> version = detect_keras_version()
        >>> print(f"Using Keras {version}")
    """
    try:
        import keras
        return keras.__version__
    except ImportError:
        try:
            import tensorflow as tf
            return tf.__version__
        except ImportError:
            raise ImportError("Neither Keras nor TensorFlow is installed")


def detect_keras_version_info() -> Dict[str, Any]:
    """
    Detect which Keras version and implementation is currently installed.

    Returns:
        Dict containing version information:
        - 'keras_version': Version string of keras package
        - 'tf_keras_available': Whether tf_keras is available
        - 'tf_keras_version': Version of tf_keras if available
        - 'tensorflow_available': Whether tensorflow is available
        - 'tensorflow_version': Version of tensorflow if available
        - 'is_keras3': Whether this is Keras 3.x
    """
    info = {
        'keras_version': None,
        'tf_keras_available': False,
        'tf_keras_version': None,
        'tensorflow_available': False,
        'tensorflow_version': None,
        'is_keras3': False
    }

    # Try to detect keras
    try:
        import keras
        info['keras_version'] = keras.__version__
        info['is_keras3'] = info['keras_version'].startswith('3.')
        logger.info(f"Detected keras version: {info['keras_version']}")
    except ImportError:
        logger.warning("Keras package not found")

    # Try to detect tf_keras
    try:
        import tf_keras
        info['tf_keras_available'] = True
        info['tf_keras_version'] = tf_keras.__version__
        logger.info(f"Detected tf_keras version: {info['tf_keras_version']}")
    except ImportError:
        logger.info("tf_keras package not available")

    # Try to detect tensorflow
    try:
        import tensorflow as tf
        info['tensorflow_available'] = True
        info['tensorflow_version'] = tf.__version__
        logger.info(f"Detected tensorflow version: {info['tensorflow_version']}")
    except ImportError:
        logger.info("TensorFlow package not available")

    return info


def detect_model_format(model_path: str) -> str:
    """
    Detect the format of a saved Keras model.

    This is a convenience function that wraps check_model_format for
    compatibility with the required signature.

    Args:
        model_path: Path to the model file or directory

    Returns:
        String indicating format: 'keras', 'h5', 'saved_model', or 'unknown'

    Raises:
        FileNotFoundError: If the model path doesn't exist

    Example:
        >>> format = detect_model_format('path/to/model.keras')
        >>> print(f"Model format: {format}")
    """
    return check_model_format(model_path)


def check_model_format(model_path: Union[str, Path]) -> str:
    """
    Detect the format of a saved Keras model.

    Args:
        model_path: Path to the model file or directory

    Returns:
        String indicating format: 'keras', 'h5', 'saved_model', or 'unknown'

    Raises:
        FileNotFoundError: If the model path doesn't exist
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Check for .keras format (directory with config.json and weights)
    if model_path.is_dir():
        config_path = model_path / "config.json"
        if config_path.exists():
            logger.info(f"Detected .keras format at: {model_path}")
            return 'keras'

        # Check for SavedModel format
        saved_model_pb = model_path / "saved_model.pb"
        if saved_model_pb.exists():
            logger.info(f"Detected SavedModel format at: {model_path}")
            return 'saved_model'

        # Check for HDF5 directory format
        assets_path = model_path / "assets"
        variables_path = model_path / "variables"
        if assets_path.exists() and variables_path.exists():
            logger.info(f"Detected HDF5 directory format at: {model_path}")
            return 'h5'

    # Check for .h5 file format
    elif model_path.is_file():
        if model_path.suffix == '.h5' or model_path.suffix == '.hdf5':
            logger.info(f"Detected .h5 file format at: {model_path}")
            return 'h5'

        # Check if it's a zip file (newer .keras format)
        if model_path.suffix == '.keras':
            try:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    if 'config.json' in zip_ref.namelist():
                        logger.info(f"Detected .keras zip format at: {model_path}")
                        return 'keras'
            except Exception as e:
                logger.warning(f"Failed to check if {model_path} is a .keras zip: {e}")

    logger.warning(f"Unknown model format at: {model_path}")
    return 'unknown'


def validate_model_loading(model: Any, expected_input_shape: Optional[Tuple] = None) -> Dict[str, Any]:
    """
    Validate that a model was loaded correctly.

    Args:
        model: The loaded Keras model
        expected_input_shape: Optional expected input shape (excluding batch dimension)

    Returns:
        Dict containing validation results:
        - 'valid': Boolean indicating if model is valid
        - 'input_shape': Actual input shape
        - 'output_shape': Output shape
        - 'num_layers': Number of layers
        - 'trainable_params': Number of trainable parameters
        - 'non_trainable_params': Number of non-trainable parameters
        - 'optimizer': Optimizer instance or None
        - 'warnings': List of validation warnings
    """
    validation = {
        'valid': False,
        'input_shape': None,
        'output_shape': None,
        'num_layers': 0,
        'trainable_params': 0,
        'non_trainable_params': 0,
        'optimizer': None,
        'warnings': []
    }

    try:
        # Check if model has expected attributes
        if not hasattr(model, 'input_shape') or not hasattr(model, 'output_shape'):
            validation['warnings'].append("Model missing input_shape or output_shape attributes")
            return validation

        validation['input_shape'] = model.input_shape
        validation['output_shape'] = model.output_shape
        validation['num_layers'] = len(model.layers)
        validation['trainable_params'] = sum([np.prod(w.shape) for w in model.trainable_weights])
        validation['non_trainable_params'] = sum([np.prod(w.shape) for w in model.non_trainable_weights])

        # Check optimizer
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            validation['optimizer'] = model.optimizer
        else:
            validation['warnings'].append("Model has no optimizer (weights-only load)")

        # Validate against expected input shape
        if expected_input_shape is not None:
            actual_shape = model.input_shape[1:]  # Exclude batch dimension
            if actual_shape != expected_input_shape:
                validation['warnings'].append(
                    f"Input shape mismatch: expected {expected_input_shape}, got {actual_shape}"
                )

        validation['valid'] = True
        logger.info(
            f"Model validation passed: {validation['num_layers']} layers, "
            f"{validation['trainable_params']:,} trainable params"
        )

    except Exception as e:
        validation['warnings'].append(f"Validation error: {str(e)}")
        logger.error(f"Model validation failed: {e}")

    return validation


def extract_model_metadata(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from a saved Keras model without loading it.

    Args:
        model_path: Path to the model file or directory

    Returns:
        Dict containing model metadata:
        - 'format': Model format
        - 'keras_version': Keras version used to save
        - 'backend': Backend used
        - 'model_name': Name of the model
        - 'config': Full model configuration (if available)
    """
    metadata = {
        'format': check_model_format(model_path),
        'keras_version': None,
        'backend': None,
        'model_name': None,
        'config': None
    }

    try:
        model_path = Path(model_path)

        if metadata['format'] == 'keras':
            archive_metadata = None
            config = None

            # Try to read config.json / metadata.json
            if model_path.is_dir():
                config_path = model_path / "config.json"
                archive_metadata_path = model_path / "metadata.json"
            else:
                # It's a .keras zip file
                config_path = model_path  # Will be handled below

            if model_path.is_file() and model_path.suffix == '.keras':
                # Extract from zip
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    if 'config.json' in zip_ref.namelist():
                        with zip_ref.open('config.json') as f:
                            config = json.load(f)
                    if 'metadata.json' in zip_ref.namelist():
                        with zip_ref.open('metadata.json') as f:
                            archive_metadata = json.load(f)
            elif config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                if archive_metadata_path.exists():
                    with open(archive_metadata_path) as f:
                        archive_metadata = json.load(f)

            if config is not None:
                metadata['config'] = config
                metadata['keras_version'] = config.get('keras_version')
                metadata['backend'] = config.get('backend')
                metadata['model_name'] = config.get('config', {}).get('name')

            if archive_metadata:
                metadata['keras_version'] = metadata['keras_version'] or archive_metadata.get('keras_version')
                metadata['backend'] = metadata['backend'] or archive_metadata.get('backend')
                metadata['model_name'] = metadata['model_name'] or archive_metadata.get('model_name')

        logger.info(
            f"Extracted metadata: Keras {metadata['keras_version']}, "
            f"Backend: {metadata['backend']}, Name: {metadata['model_name']}"
        )

    except Exception as e:
        logger.warning(f"Failed to extract metadata from {model_path}: {e}")

    return metadata


# =============================================================================
# APPROACH 0: Native Keras 3.x (For models saved with Keras 3.x)
# =============================================================================

def load_with_keras_native(
    model_path: Union[str, Path],
    custom_objects: Optional[Dict] = None,
    compile: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a Keras model using native Keras 3.x (standalone keras package).

    This approach uses the native `keras` package (Keras 3.x) directly, which
    is required for models that were saved with Keras 3.x. These models have
    module paths like 'keras.src.models.functional' which are incompatible with
    tf_keras.

    Args:
        model_path: Path to the model file or directory
        custom_objects: Optional dictionary mapping names to custom layer classes
        compile: Whether to compile the model with the saved optimizer (default False
                to avoid issues with custom LR schedules)

    Returns:
        Tuple of (model, metadata) where metadata contains:
        - 'approach': 'keras_native'
        - 'success': Boolean indicating success
        - 'optimizer_preserved': Whether optimizer state was preserved
        - 'warnings': List of warnings

    Raises:
        ImportError: If keras is not available
        Exception: If model loading fails
    """
    metadata = {
        'approach': 'keras_native',
        'success': False,
        'optimizer_preserved': False,
        'warnings': []
    }

    logger.info(f"Attempting to load model using native Keras 3.x: {model_path}")

    try:
        import keras
        logger.info(f"Using native keras package version {keras.__version__}")
    except ImportError:
        error_msg = "Native keras package is not available"
        metadata['warnings'].append(error_msg)
        logger.error(error_msg)
        raise ImportError(error_msg)

    # Ensure custom LR schedules are registered before loading
    try:
        from src.training.m1_metal_optimizer import WarmupCosineDecaySchedule, CosineDecayRestarts  # noqa: F401
        logger.debug("Custom LR schedules imported for registration")
    except ImportError:
        logger.debug("Custom LR schedules not available (this is OK for inference)")

    try:
        # Build custom objects dict with standard replacements
        all_custom_objects = custom_objects.copy() if custom_objects else {}

        # Load the model - use compile=False by default to avoid optimizer issues
        model = keras.models.load_model(
            str(model_path),
            custom_objects=all_custom_objects,
            compile=compile
        )

        metadata['success'] = True
        metadata['optimizer_preserved'] = (compile and model.optimizer is not None)

        if metadata['optimizer_preserved']:
            logger.info("Model loaded successfully with optimizer state preserved")
        else:
            logger.info("Model loaded successfully (weights only, no optimizer)")
            metadata['warnings'].append("Optimizer state not preserved (compile=False)")

        return model, metadata

    except Exception as e:
        error_str = str(e).lower()

        # If it's an optimizer issue, retry with compile=False
        if compile and ('optimizer' in error_str or 'warmup' in error_str.lower() or
                        'schedule' in error_str.lower() or 'deserialize' in error_str):
            err = _compact_error(e)
            logger.warning(f"Optimizer/schedule issue detected, retrying with compile=False: {err}")
            metadata['warnings'].append(f"Optimizer issue: {err}")

            try:
                model = keras.models.load_model(
                    str(model_path),
                    custom_objects=all_custom_objects if 'all_custom_objects' in dir() else custom_objects,
                    compile=False
                )

                metadata['success'] = True
                metadata['optimizer_preserved'] = False
                metadata['warnings'].append(
                    "Optimizer state skipped due to deserialization issue - "
                    "model will need recompilation before training"
                )

                logger.info("Model loaded successfully with compile=False")
                return model, metadata

            except Exception as retry_e:
                error_msg = f"Failed to load model even with compile=False: {_compact_error(retry_e)}"
                metadata['warnings'].append(error_msg)
                logger.error(error_msg)
                raise Exception(error_msg) from retry_e

        compact = _compact_error(e)
        error_msg = f"Failed to load model with native keras: {compact}"
        metadata['warnings'].append(error_msg)
        if _is_expected_compat_load_error(compact):
            logger.debug(
                "Native keras compatibility miss for %s (will fallback): %s",
                model_path,
                compact,
            )
        else:
            logger.error(error_msg)
        raise Exception(error_msg) from e


# =============================================================================
# APPROACH 1: tf.keras Backward Compatibility (Recommended)
# =============================================================================

def load_with_tf_keras(
    model_path: Union[str, Path],
    custom_objects: Optional[Dict] = None,
    compile: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a Keras model using tf.keras for backward compatibility.

    This approach uses tf.keras (or tf_keras package) which maintains compatibility
    with models saved using older Keras versions. This is the recommended approach
    for immediate fixes when loading Keras 2.15.0 models.

    Args:
        model_path: Path to the model file or directory
        custom_objects: Optional dictionary mapping names to custom layer classes
        compile: Whether to compile the model with the saved optimizer

    Returns:
        Tuple of (model, metadata) where metadata contains:
        - 'approach': 'tf_keras'
        - 'success': Boolean indicating success
        - 'optimizer_preserved': Whether optimizer state was preserved
        - 'warnings': List of warnings

    Raises:
        ImportError: If tf.keras or tf_keras is not available
        Exception: If model loading fails
    """
    metadata = {
        'approach': 'tf_keras',
        'success': False,
        'optimizer_preserved': False,
        'warnings': []
    }

    logger.info(f"Attempting to load model using tf.keras: {model_path}")

    # Try tf_keras first (newer, standalone package)
    try:
        import tf_keras as keras
        logger.info("Using tf_keras package")
        keras_source = 'tf_keras'
    except ImportError:
        # Fall back to tf.keras from TensorFlow
        try:
            import tensorflow as tf
            keras = tf.keras
            logger.info("Using tf.keras from TensorFlow")
            keras_source = 'tf.keras'
        except ImportError:
            error_msg = "Neither tf_keras nor TensorFlow (tf.keras) is available"
            metadata['warnings'].append(error_msg)
            logger.error(error_msg)
            raise ImportError(error_msg)

    try:
        # Load the model
        model = keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=compile
        )

        metadata['success'] = True
        metadata['optimizer_preserved'] = (compile and model.optimizer is not None)

        if metadata['optimizer_preserved']:
            logger.info("Model loaded successfully with optimizer state preserved")
        else:
            logger.info("Model loaded successfully (weights only, optimizer not compiled)")
            metadata['warnings'].append("Optimizer state not preserved (weights-only load)")

        return model, metadata

    except Exception as e:
        error_str = str(e).lower()

        # Handle optimizer variable mismatch - retry without optimizer
        if compile and ('optimizer' in error_str or 'variable' in error_str or 'mismatch' in error_str):
            err = _compact_error(e)
            logger.warning(f"Optimizer state mismatch detected, retrying with compile=False: {err}")
            metadata['warnings'].append(f"Optimizer state mismatch: {err}")

            try:
                model = keras.models.load_model(
                    str(model_path),
                    custom_objects=custom_objects,
                    compile=False
                )

                metadata['success'] = True
                metadata['optimizer_preserved'] = False
                metadata['warnings'].append(
                    "Optimizer state skipped due to variable mismatch - "
                    "model will need recompilation before training"
                )

                logger.info("Model loaded successfully (weights only, optimizer skipped due to mismatch)")
                return model, metadata

            except Exception as retry_e:
                error_msg = f"Failed to load model even without optimizer: {_compact_error(retry_e)}"
                metadata['warnings'].append(error_msg)
                logger.error(error_msg)
                raise Exception(error_msg) from retry_e

        compact = _compact_error(e)
        error_msg = f"Failed to load model with {keras_source}: {compact}"
        metadata['warnings'].append(error_msg)
        if _is_expected_compat_load_error(compact):
            logger.debug(
                "%s compatibility miss for %s (will fallback): %s",
                keras_source,
                model_path,
                compact,
            )
        else:
            logger.error(error_msg)
        raise Exception(error_msg) from e


# =============================================================================
# APPROACH 2: Model Rebuilding + Weight Loading (For Keras 3.x Migration)
# =============================================================================

def parse_model_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse the model configuration from a saved Keras model file.

    Args:
        config_path: Path to config.json or .keras file

    Returns:
        Dict containing the parsed configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)

    # Handle .keras zip files
    if config_path.is_file() and config_path.suffix == '.keras':
        with zipfile.ZipFile(config_path, 'r') as zip_ref:
            if 'config.json' in zip_ref.namelist():
                with zip_ref.open('config.json') as f:
                    config = json.load(f)
                    logger.info("Parsed config from .keras zip file")
                    return config
            else:
                raise FileNotFoundError("config.json not found in .keras file")

    # Handle directory format
    elif config_path.is_dir():
        config_file = config_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in directory: {config_path}")
        with open(config_file) as f:
            config = json.load(f)
            logger.info("Parsed config from directory")
            return config

    # Handle direct JSON config files (config.json / *.arch.json)
    elif config_path.is_file() and config_path.suffix == ".json":
        with open(config_path) as f:
            config = json.load(f)
            logger.info("Parsed config from JSON file")
            return config

    else:
        raise FileNotFoundError(f"Invalid config path: {config_path}")


def rebuild_tcn_architecture(
    config: Dict[str, Any],
    optimizer_config: Optional[Dict] = None
) -> Any:
    """
    Rebuild a TCN architecture from its configuration using Keras 3.x APIs.

    This is a convenience function that wraps rebuild_tcn_model_from_config
    for compatibility with the required signature.

    Args:
        config: Model configuration dictionary
        optimizer_config: Optional optimizer configuration

    Returns:
        Rebuilt Keras model

    Raises:
        ValueError: If configuration is invalid or unsupported

    Example:
        >>> config = {'config': {'layers': [...]}}
        >>> model = rebuild_tcn_architecture(config)
    """
    return rebuild_tcn_model_from_config(config, optimizer_config)


def rebuild_tcn_model_from_config(
    config: Dict[str, Any],
    optimizer_config: Optional[Dict] = None
) -> Any:
    """
    Rebuild a TCN model from its configuration using Keras 3.x APIs.

    This function reconstructs the model architecture from the saved configuration
    using current Keras 3.x APIs, avoiding the deserialization issues with old
    module paths.

    Args:
        config: Model configuration dictionary
        optimizer_config: Optional optimizer configuration

    Returns:
        Rebuilt Keras model

    Raises:
        ValueError: If configuration is invalid or unsupported
    """
    try:
        import keras
    except ImportError:
        raise ImportError("Keras is required for model rebuilding")

    logger.info("Rebuilding TCN model from configuration")

    # Extract layer configurations
    layers_config = config.get('config', {}).get('layers', [])
    if not layers_config:
        raise ValueError("No layers found in configuration")

    # Build the model layer by layer
    layer_map = {}  # Map layer names to layer instances
    unsupported_layers = []

    for layer_config in layers_config:
        class_name = layer_config.get('class_name')
        name = layer_config.get('config', {}).get('name')
        layer_config_dict = _normalize_layer_config_for_compat(
            class_name,
            layer_config.get('config', {}),
        )

        logger.info(f"Building layer: {class_name} ({name})")

        # Handle input layer
        if class_name == 'InputLayer':
            input_shape = layer_config_dict.get('input_shape') or layer_config_dict.get('shape')
            batch_size = layer_config_dict.get('batch_size')
            if input_shape:
                keras_version = str(getattr(keras, "__version__", "") or "")
                try:
                    keras_major = int(keras_version.split(".", 1)[0])
                except (TypeError, ValueError):
                    keras_major = 0
                input_kwargs = {
                    "batch_size": batch_size,
                    "dtype": layer_config_dict.get('dtype', 'float32'),
                    "sparse": layer_config_dict.get('sparse', False),
                    "ragged": layer_config_dict.get('ragged', False),
                    "name": name,
                }
                if keras_major >= 3:
                    input_kwargs["shape"] = tuple(input_shape)
                else:
                    input_kwargs["input_shape"] = tuple(input_shape)
                layer_map[name] = keras.layers.InputLayer(**input_kwargs)
                continue
            raise ValueError(f"Input layer '{name}' is missing a supported input shape")

        # Build other layers
        layer = None

        if class_name == 'GaussianNoise':
            layer = keras.layers.GaussianNoise(
                stddev=layer_config_dict.get('stddev', 0.1),
                name=name
            )

        elif class_name == 'SpatialDropout1D':
            layer = keras.layers.SpatialDropout1D(
                rate=layer_config_dict.get('rate', 0.1),
                name=name
            )

        elif class_name == 'Conv1D':
            layer = keras.layers.Conv1D(
                filters=layer_config_dict.get('filters', 64),
                kernel_size=layer_config_dict.get('kernel_size', 3),
                strides=layer_config_dict.get('strides', 1),
                padding=layer_config_dict.get('padding', 'valid'),
                activation=layer_config_dict.get('activation', 'linear'),
                use_bias=layer_config_dict.get('use_bias', True),
                kernel_initializer=layer_config_dict.get('kernel_initializer', 'glorot_uniform'),
                name=name
            )

        elif class_name == 'BatchNormalization':
            layer = keras.layers.BatchNormalization(
                axis=layer_config_dict.get('axis', -1),
                momentum=layer_config_dict.get('momentum', 0.99),
                epsilon=layer_config_dict.get('epsilon', 0.001),
                name=name
            )

        elif class_name == 'Dropout':
            layer = keras.layers.Dropout(
                rate=layer_config_dict.get('rate', 0.5),
                name=name
            )

        elif class_name == 'GlobalAveragePooling1D':
            layer = keras.layers.GlobalAveragePooling1D(name=name)

        elif class_name == 'Dense':
            layer = keras.layers.Dense(
                units=layer_config_dict.get('units', 32),
                activation=layer_config_dict.get('activation', 'linear'),
                use_bias=layer_config_dict.get('use_bias', True),
                kernel_initializer=layer_config_dict.get('kernel_initializer', 'glorot_uniform'),
                name=name
            )

        elif class_name == 'Softmax':
            layer = keras.layers.Softmax(name=name)

        else:
            unsupported_layers.append(class_name)
            logger.warning(f"Unsupported layer class for rebuild: {class_name}")
            continue

        if layer is not None:
            layer_map[name] = layer

    # Reconstruct the model using Functional API
    if not layer_map:
        raise ValueError("No valid layers were reconstructed")

    # Find input and output layers
    input_layers_config = config.get('config', {}).get('input_layers', [])
    output_layers_config = config.get('config', {}).get('output_layers', [])

    if not input_layers_config or not output_layers_config:
        raise ValueError("Input or output layers not specified in configuration")

    # Get the first input layer
    input_name = _extract_layer_ref_name(input_layers_config)
    if not input_name:
        raise ValueError(f"Could not determine input layer from configuration: {input_layers_config}")
    if input_name not in layer_map:
        raise ValueError(f"Input layer '{input_name}' not found in reconstructed layers")

    if unsupported_layers:
        unsupported_summary = ", ".join(sorted(set(unsupported_layers)))
        raise ValueError(
            "Rebuild only supports linear TCN-style stacks; "
            f"unsupported layers present: {unsupported_summary}"
        )

    logger.warning("Using simplified sequential reconstruction - may not match exact architecture")

    # Create a list of layers in order
    layer_list = list(layer_map.values())

    # Build the model
    model = keras.Sequential(layer_list, name=config.get('config', {}).get('name', 'rebuilt_model'))

    # Compile with optimizer if provided
    if optimizer_config:
        try:
            # Parse optimizer configuration
            optimizer_class_name = optimizer_config.get('class_name', 'Adam')
            optimizer_config_dict = optimizer_config.get('config', {})

            # Create optimizer
            if optimizer_class_name == 'Adam':
                optimizer = keras.optimizers.Adam(
                    learning_rate=optimizer_config_dict.get('learning_rate', 0.001),
                    beta_1=optimizer_config_dict.get('beta_1', 0.9),
                    beta_2=optimizer_config_dict.get('beta_2', 0.999)
                )
            elif optimizer_class_name == 'SGD':
                optimizer = keras.optimizers.SGD(
                    learning_rate=optimizer_config_dict.get('learning_rate', 0.01),
                    momentum=optimizer_config_dict.get('momentum', 0.0)
                )
            else:
                logger.warning(f"Unsupported optimizer class: {optimizer_class_name}, using Adam")
                optimizer = keras.optimizers.Adam()

            # Compile model
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            logger.info(f"Model compiled with {optimizer_class_name} optimizer")

        except Exception as e:
            logger.warning(f"Failed to compile model with optimizer: {e}")

    logger.info("TCN model rebuilt successfully")
    return model


def _extract_layer_ref_name(layer_refs: Any) -> Optional[str]:
    """Extract the first layer name from flat or nested Keras layer refs."""
    if isinstance(layer_refs, list):
        if layer_refs and isinstance(layer_refs[0], str):
            return layer_refs[0]
        for value in layer_refs:
            resolved = _extract_layer_ref_name(value)
            if resolved:
                return resolved
    return None


def load_by_rebuilding(
    model_path: Union[str, Path],
    custom_objects: Optional[Dict] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a Keras model by rebuilding it from configuration and loading weights.

    This approach is useful for migrating models to Keras 3.x when backward
    compatibility is not available. It rebuilds the architecture using current
    APIs and loads the weights separately.

    Args:
        model_path: Path to the model file or directory
        custom_objects: Optional dictionary mapping names to custom layer classes

    Returns:
        Tuple of (model, metadata) where metadata contains:
        - 'approach': 'rebuild'
        - 'success': Boolean indicating success
        - 'optimizer_preserved': False (always false for this approach)
        - 'warnings': List of warnings

    Raises:
        Exception: If model rebuilding fails
    """
    metadata = {
        'approach': 'rebuild',
        'success': False,
        'optimizer_preserved': False,
        'warnings': []
    }

    logger.info(f"Attempting to load model by rebuilding: {model_path}")

    try:
        # Parse configuration
        config = parse_model_config(model_path)

        # Extract optimizer config if available
        optimizer_config = None
        if 'optimizer' in config:
            optimizer_config = config['optimizer']

        # Rebuild the model
        model = rebuild_tcn_model_from_config(config, optimizer_config)

        # Try to load weights
        model_path = Path(model_path)
        weights_loaded = False

        if model_path.is_dir():
            # Check for model.weights.h5 or weights.h5
            weights_files = list(model_path.glob("*.weights.h5")) + list(model_path.glob("weights.h5"))
            if weights_files:
                weights_path = weights_files[0]
                try:
                    model.load_weights(str(weights_path), by_name=True)
                    weights_loaded = True
                    logger.info(f"Loaded weights from: {weights_path}")
                except Exception as e:
                    metadata['warnings'].append(f"Failed to load weights: {str(e)}")
                    logger.warning(f"Failed to load weights: {e}")
        elif model_path.is_file() and model_path.suffix == '.keras':
            # Extract weights from .keras zip
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                weights_files = [f for f in zip_ref.namelist() if f.endswith('.weights.h5') or f == 'model.weights.h5']
                if weights_files:
                    try:
                        # Extract weights to temp file
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                            with zip_ref.open(weights_files[0]) as wf:
                                tmp.write(wf.read())
                            tmp_path = tmp.name

                        model.load_weights(tmp_path, by_name=True)
                        weights_loaded = True
                        logger.info("Loaded weights from .keras file")

                        # Clean up temp file
                        os.unlink(tmp_path)
                    except Exception as e:
                        metadata['warnings'].append(f"Failed to load weights: {str(e)}")
                        logger.warning(f"Failed to load weights: {e}")

        if not weights_loaded:
            metadata['warnings'].append("Weights could not be loaded - model initialized randomly")
            logger.warning("Model initialized with random weights")

        metadata['success'] = True
        metadata['warnings'].append("Optimizer state not preserved (rebuild approach)")

        logger.info("Model rebuilt successfully")
        return model, metadata

    except Exception as e:
        error_msg = f"Failed to rebuild model: {str(e)}"
        metadata['warnings'].append(error_msg)
        logger.error(error_msg)
        raise Exception(error_msg) from e


# =============================================================================
# APPROACH 3: Custom Deserialization (Advanced)
# =============================================================================

def register_custom_deserializers() -> Dict[str, Any]:
    """
    Register custom deserializers to handle old Keras module paths.

    This function creates a mapping from old module paths to new Keras 3.x classes,
    allowing models saved with Keras 2.15.0 to be deserialized correctly.

    Handles the common issue where models saved with Keras 2.x reference internal
    module paths like 'keras.src.engine.functional' which don't exist in Keras 3.x.

    Returns:
        Dictionary mapping old class paths to new classes
    """
    try:
        import keras
    except ImportError:
        logger.warning("Keras not available for custom deserializers")
        return {}

    class LegacyTFOpLambda(keras.layers.Layer):
        """Compatibility shim for serialized TFOpLambda add nodes."""

        def __init__(self, function: str = "__operators__.add", **kwargs):
            super().__init__(**kwargs)
            self.function = function

        def call(self, inputs, y=None, **kwargs):
            if y is None and isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                inputs, y = inputs
            if self.function in {"__operators__.add", "math.add", "add"}:
                return inputs if y is None else inputs + y
            raise ValueError(f"Unsupported LegacyTFOpLambda function: {self.function}")

        def get_config(self):
            config = super().get_config()
            config["function"] = self.function
            return config

    class LegacyAddOperation(keras.layers.Layer):
        """Accept Keras 3 op-style add calls that pass two positional tensors."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True

        def call(self, inputs, y=None, **kwargs):
            del kwargs
            if y is not None:
                return inputs + y
            if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                return inputs[0] + inputs[1]
            raise ValueError(
                "LegacyAddOperation expects either two positional tensors or a 2-item input list"
            )

    class LegacyMultiHeadAttention(keras.layers.MultiHeadAttention):
        """Accept legacy list-style positional inputs emitted by old configs."""

        def build(self, query_shape, value_shape=None, key_shape=None):
            if value_shape is None and isinstance(query_shape, (list, tuple)):
                if len(query_shape) == 2:
                    query_shape, value_shape = query_shape
                elif len(query_shape) == 3:
                    query_shape, value_shape, key_shape = query_shape
            if value_shape is None:
                value_shape = query_shape
            if key_shape is None:
                key_shape = value_shape

            build_from_signature = getattr(self, "_build_from_signature", None)
            if callable(build_from_signature):
                build_from_signature(query_shape, value_shape, key_shape)
                self.built = True
                return

            return super().build(query_shape, value_shape, key_shape)

        def compute_output_spec(self, query, value=None, key=None, **kwargs):
            if value is None and isinstance(query, (list, tuple)) and len(query) == 2:
                query, value = query
            return super().compute_output_spec(query, value=value, key=key, **kwargs)

        def call(self, query, value=None, key=None, **kwargs):
            if value is None and isinstance(query, (list, tuple)) and len(query) == 2:
                query, value = query
            return super().call(query, value=value, key=key, **kwargs)

    # Map old Keras 2.x module paths to new Keras 3.x classes
    custom_objects = {
        # Functional model paths (critical for Keras 2.15.0 -> 3.x migration)
        'keras.src.engine.functional.Functional': keras.Model,
        'keras.engine.functional.Functional': keras.Model,
        'keras.src.models.functional.Functional': keras.Model,  # Keras 3.x path
        'Functional': keras.Model,

        # Sequential model paths
        'keras.src.engine.sequential.Sequential': keras.Sequential,
        'keras.engine.sequential.Sequential': keras.Sequential,

        # Training model paths
        'keras.src.engine.training.Model': keras.Model,
        'keras.engine.training.Model': keras.Model,

        # Layer mappings (common layers)
        'keras.layers.InputLayer': keras.layers.InputLayer,
        'keras.src.layers.core.input_layer.InputLayer': keras.layers.InputLayer,
        'keras.layers.GaussianNoise': keras.layers.GaussianNoise,
        'keras.src.layers.regularization.gaussian_noise.GaussianNoise': keras.layers.GaussianNoise,
        'keras.layers.SpatialDropout1D': keras.layers.SpatialDropout1D,
        'keras.src.layers.regularization.spatial_dropout1d.SpatialDropout1D': keras.layers.SpatialDropout1D,
        'keras.layers.Conv1D': keras.layers.Conv1D,
        'keras.src.layers.convolutional.conv1d.Conv1D': keras.layers.Conv1D,
        'keras.layers.BatchNormalization': keras.layers.BatchNormalization,
        'keras.src.layers.normalization.batch_normalization.BatchNormalization': keras.layers.BatchNormalization,
        'keras.layers.LayerNormalization': keras.layers.LayerNormalization,
        'keras.src.layers.normalization.layer_normalization.LayerNormalization': keras.layers.LayerNormalization,
        'keras.layers.Dropout': keras.layers.Dropout,
        'keras.src.layers.regularization.dropout.Dropout': keras.layers.Dropout,
        'keras.layers.GlobalAveragePooling1D': keras.layers.GlobalAveragePooling1D,
        'keras.src.layers.pooling.global_average_pooling1d.GlobalAveragePooling1D': keras.layers.GlobalAveragePooling1D,
        'keras.layers.Dense': keras.layers.Dense,
        'keras.src.layers.core.dense.Dense': keras.layers.Dense,
        'keras.layers.Softmax': keras.layers.Softmax,
        'keras.src.layers.activations.softmax.Softmax': keras.layers.Softmax,
        'keras.layers.MultiHeadAttention': LegacyMultiHeadAttention,
        'keras.src.layers.attention.multi_head_attention.MultiHeadAttention': LegacyMultiHeadAttention,
        'keras.layers.Add': keras.layers.Add,
        'keras.src.layers.merging.add.Add': keras.layers.Add,
        'keras.src.ops.numpy.Add': LegacyAddOperation,  # Keras 3.x op path uses 2 positional inputs
        'Add': keras.layers.Add,  # Short name fallback
        'keras.layers.Concatenate': keras.layers.Concatenate,
        'keras.src.layers.merging.concatenate.Concatenate': keras.layers.Concatenate,
        'keras.layers.Flatten': keras.layers.Flatten,
        'keras.src.layers.reshaping.flatten.Flatten': keras.layers.Flatten,
        'keras.layers.Reshape': keras.layers.Reshape,
        'keras.src.layers.reshaping.reshape.Reshape': keras.layers.Reshape,
        'LegacyAddOperation': LegacyAddOperation,
        'LegacyMultiHeadAttention': LegacyMultiHeadAttention,
        'LegacyTFOpLambda': LegacyTFOpLambda,
        'TFOpLambda': LegacyTFOpLambda,
        'keras.src.layers.core.tf_op_layer.TFOpLambda': LegacyTFOpLambda,
        'keras.layers.core.tf_op_layer.TFOpLambda': LegacyTFOpLambda,
    }

    # Try to add LSTM/GRU if available
    try:
        custom_objects['keras.layers.LSTM'] = keras.layers.LSTM
        custom_objects['keras.src.layers.rnn.lstm.LSTM'] = keras.layers.LSTM
        custom_objects['keras.layers.GRU'] = keras.layers.GRU
        custom_objects['keras.src.layers.rnn.gru.GRU'] = keras.layers.GRU
    except AttributeError:
        pass  # Not available in all Keras builds

    # Add custom learning rate schedules
    try:
        from src.training.m1_metal_optimizer import WarmupCosineDecaySchedule, CosineDecayRestarts
        custom_objects['WarmupCosineDecaySchedule'] = WarmupCosineDecaySchedule
        custom_objects['src.training.m1_metal_optimizer>WarmupCosineDecaySchedule'] = WarmupCosineDecaySchedule
        custom_objects['ml_engine>WarmupCosineDecaySchedule'] = WarmupCosineDecaySchedule
        custom_objects['CosineDecayRestarts'] = CosineDecayRestarts
        custom_objects['src.training.m1_metal_optimizer>CosineDecayRestarts'] = CosineDecayRestarts
        custom_objects['ml_engine>CosineDecayRestarts'] = CosineDecayRestarts
        logger.debug("Registered WarmupCosineDecaySchedule and CosineDecayRestarts")
    except ImportError as e:
        logger.warning(f"Could not import custom LR schedules: {e}")

    logger.info(f"Registered {len(custom_objects)} custom deserializers")
    return custom_objects


def update_module_paths_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update old module paths in model configuration to new Keras 3.x paths.

    Args:
        config: Original model configuration

    Returns:
        Updated configuration with corrected module paths
    """
    import copy

    # Deep copy to avoid modifying original
    updated_config = copy.deepcopy(config)

    # Map old module paths to new ones
    path_mappings = {
        'keras.src.engine.functional': 'keras',
        'keras.src.layers': 'keras.layers',
        'keras.src.optimizers': 'keras.optimizers',
    }

    # Update module field if present
    if 'module' in updated_config:
        old_module = updated_config['module']
        for old_path, new_path in path_mappings.items():
            if old_module.startswith(old_path):
                updated_config['module'] = new_path + old_module[len(old_path):]
                logger.info(f"Updated module path: {old_module} -> {updated_config['module']}")
                break

    # Update layer configurations
    if 'config' in updated_config and 'layers' in updated_config['config']:
        for layer_config in updated_config['config']['layers']:
            if 'module' in layer_config:
                old_module = layer_config['module']
                for old_path, new_path in path_mappings.items():
                    if old_module.startswith(old_path):
                        layer_config['module'] = new_path + old_module[len(old_path):]
                        break

    return updated_config


def _build_model_from_serialized_config(
    keras_module: Any,
    config: Dict[str, Any],
    custom_objects: Dict[str, Any],
) -> Any:
    """Build a model from a full serialized Keras config or raw model config."""
    sanitized_config = _sanitize_serialized_keras_config(
        config=config,
        keras_module=keras_module,
        custom_objects=custom_objects,
    )

    class_name = sanitized_config.get("class_name")
    model_config = sanitized_config.get("config", sanitized_config)

    if class_name == "Sequential":
        return keras_module.Sequential.from_config(
            model_config,
            custom_objects=custom_objects,
        )

    if class_name in {"Functional", "Model"} or (
        isinstance(model_config, dict)
        and "layers" in model_config
        and "input_layers" in model_config
    ):
        return keras_module.Model.from_config(
            model_config,
            custom_objects=custom_objects,
        )

    deserialize = getattr(getattr(keras_module, "saving", None), "deserialize_keras_object", None)
    if deserialize is not None:
        return deserialize(sanitized_config, custom_objects=custom_objects)

    return keras_module.models.model_from_config(
        sanitized_config,
        custom_objects=custom_objects,
    )


def _resolve_serialized_class(
    keras_module: Any,
    config: Dict[str, Any],
    custom_objects: Dict[str, Any],
) -> Any:
    """Resolve a serialized Keras class to a runtime class when possible."""
    class_name = config.get("class_name")
    module = config.get("module")
    if not class_name:
        return None

    fq_name = f"{module}.{class_name}" if module else class_name
    if fq_name in custom_objects:
        return custom_objects[fq_name]
    if class_name in custom_objects:
        return custom_objects[class_name]

    layers_mod = getattr(keras_module, "layers", None)
    if layers_mod is not None and hasattr(layers_mod, class_name):
        return getattr(layers_mod, class_name)
    if hasattr(keras_module, class_name):
        return getattr(keras_module, class_name)
    return None


def _filter_constructor_kwargs(cls: Any, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop serialized kwargs that the current constructor does not accept."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return config_dict

    allowed = {
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    # Most Keras layers expose ``**kwargs`` only to forward a narrow set of
    # base layer args such as ``name``/``dtype``. Legacy archives also include
    # shape hints (for example MultiHeadAttention query/key/value shapes) that
    # must not be passed back into modern constructors.
    allowed.update(
        {
            "name",
            "dtype",
            "input_shape",
            "batch_input_shape",
            "sparse",
            "ragged",
            "input_tensor",
        }
    )
    return {key: value for key, value in config_dict.items() if key in allowed}


def _normalize_layer_config_for_compat(
    class_name: Optional[str],
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Rewrite known cross-version config fields into broadly supported forms."""
    normalized = dict(config_dict)

    # Keras 2 → 3: 'seed' was accepted by Dense, Conv1D, etc. in Keras 2.x
    # but removed in Keras 3.x.  Strip it from layers that no longer accept it.
    _LAYERS_SEED_REMOVED_IN_K3 = {
        "Dense", "Conv1D", "Conv2D", "Conv3D",
        "Conv1DTranspose", "Conv2DTranspose", "Conv3DTranspose",
        "DepthwiseConv1D", "DepthwiseConv2D",
        "SeparableConv1D", "SeparableConv2D",
        "EinsumDense",
        "Embedding",
        "MultiHeadAttention",
        "Add", "Multiply", "Concatenate", "Average",
        "LayerNormalization",
    }
    if class_name in _LAYERS_SEED_REMOVED_IN_K3:
        normalized.pop("seed", None)

    if class_name == "InputLayer":
        batch_shape = normalized.pop("batch_shape", None)
        batch_input_shape = normalized.pop("batch_input_shape", None)
        normalized.pop("optional", None)

        input_shape = batch_shape if batch_shape is not None else batch_input_shape
        if input_shape is not None:
            if isinstance(input_shape, tuple):
                input_shape = list(input_shape)

            if isinstance(input_shape, list) and input_shape:
                batch_size = input_shape[0]
                feature_shape = input_shape[1:]
                if "input_shape" not in normalized:
                    normalized["input_shape"] = feature_shape
                if batch_size is not None and "batch_size" not in normalized:
                    normalized["batch_size"] = batch_size
            elif "input_shape" not in normalized:
                normalized["input_shape"] = input_shape

    if class_name == "BatchNormalization":
        axis = normalized.get("axis")
        if isinstance(axis, list) and len(axis) == 1:
            normalized["axis"] = axis[0]

    return normalized


def _normalize_build_config_for_compat(
    class_name: Optional[str],
    build_config: Any,
    keras_module: Any,
    custom_objects: Dict[str, Any],
) -> Optional[Any]:
    """Preserve only the build_config forms that modern Keras still needs."""
    if not isinstance(build_config, dict):
        return None

    sanitized = _sanitize_serialized_keras_config(build_config, keras_module, custom_objects)

    if class_name == "MultiHeadAttention":
        shapes_dict = sanitized.get("shapes_dict")
        if isinstance(shapes_dict, dict) and shapes_dict.get("query_shape") is not None:
            compact_shapes = {
                key: value
                for key, value in shapes_dict.items()
                if value is not None
            }
            if compact_shapes.get("key_shape") is None:
                compact_shapes["key_shape"] = (
                    compact_shapes.get("value_shape")
                    or compact_shapes.get("query_shape")
                )
            keras_version = str(getattr(keras_module, "__version__", "") or "")
            try:
                keras_major = int(keras_version.split(".", 1)[0])
            except (TypeError, ValueError):
                keras_major = 0
            if keras_major and keras_major < 3:
                return compact_shapes
            return {"shapes_dict": compact_shapes}
        if sanitized.get("query_shape") is not None:
            compact_shapes = {
                key: sanitized[key]
                for key in ("query_shape", "value_shape", "key_shape")
                if sanitized.get(key) is not None
            }
            if compact_shapes.get("key_shape") is None:
                compact_shapes["key_shape"] = (
                    compact_shapes.get("value_shape")
                    or compact_shapes.get("query_shape")
                )
            return compact_shapes

    return None


def _keras_major_version(keras_module: Any) -> int:
    """Best-effort major-version detection for standalone or tf.keras packages."""
    version = str(getattr(keras_module, "__version__", "") or "")
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def _collapse_serialized_dtype_policy(config: Any) -> Any:
    """Collapse serialized Keras dtype policy objects to plain dtype names."""
    if not isinstance(config, dict):
        return None

    if config.get("class_name") != "DTypePolicy":
        return None

    inner = config.get("config")
    if isinstance(inner, dict) and isinstance(inner.get("name"), str):
        return inner["name"]
    return None


def _normalize_legacy_inbound_nodes(nodes: Any) -> Any:
    """Wrap flat legacy node refs so modern Keras sees a list of input refs per node."""
    if not isinstance(nodes, list):
        return nodes

    normalized = []
    for node in nodes:
        if (
            isinstance(node, list)
            and len(node) in {3, 4}
            and isinstance(node[0], str)
            and isinstance(node[1], int)
            and isinstance(node[2], int)
        ):
            normalized.append([node])
        else:
            normalized.append(node)
    return normalized


def _normalize_inbound_nodes_for_keras2(nodes: Any) -> Any:
    """Convert Keras 3 node dicts and wrap flat refs for Keras 2 runtimes."""
    if not isinstance(nodes, list):
        return nodes

    normalized = []
    for node in nodes:
        if isinstance(node, dict) and "args" in node:
            normalized.append(_convert_keras3_node_data(node))
            continue
        if (
            isinstance(node, list)
            and len(node) in {3, 4}
            and isinstance(node[0], str)
            and isinstance(node[1], int)
            and isinstance(node[2], int)
        ):
            normalized.append([node])
            continue
        normalized.append(node)
    return normalized


def _convert_keras3_tensor_ref(value: Any) -> Any:
    """Convert a Keras 3 serialized tensor payload into a legacy tensor ref."""
    if isinstance(value, dict):
        class_name = value.get("class_name")
        config = value.get("config")
        if class_name == "__keras_tensor__" and isinstance(config, dict):
            keras_history = config.get("keras_history")
            if (
                isinstance(keras_history, list)
                and len(keras_history) >= 3
                and isinstance(keras_history[0], str)
            ):
                return keras_history[:3]
        if class_name == "__tensor__" and isinstance(config, dict):
            return ["_CONSTANT_VALUE", -1, config.get("value")]
        return value
    if isinstance(value, tuple):
        return [_convert_keras3_tensor_ref(item) for item in value]
    if isinstance(value, list):
        return [_convert_keras3_tensor_ref(item) for item in value]
    return ["_CONSTANT_VALUE", -1, value]


def _attach_kwargs_to_first_tensor_ref(node_data: Any, kwargs: Dict[str, Any]) -> tuple[Any, bool]:
    """Attach call kwargs to the first legacy tensor ref in a node structure."""
    if isinstance(node_data, list):
        if len(node_data) in {3, 4} and isinstance(node_data[0], str):
            merged_kwargs = {}
            if len(node_data) == 4 and isinstance(node_data[3], dict):
                merged_kwargs.update(node_data[3])
            merged_kwargs.update(kwargs)
            return node_data[:3] + [merged_kwargs], True

        updated = []
        attached = False
        for item in node_data:
            if attached:
                updated.append(item)
                continue
            new_item, attached = _attach_kwargs_to_first_tensor_ref(item, kwargs)
            updated.append(new_item)
        return updated, attached

    return node_data, False


def _convert_keras3_node_data(node: Dict[str, Any]) -> Any:
    """Translate Keras 3 ``{'args': ..., 'kwargs': ...}`` nodes to legacy refs."""
    raw_args = node.get("args")
    raw_kwargs = node.get("kwargs")

    args = raw_args if isinstance(raw_args, list) else ([] if raw_args is None else [raw_args])
    converted_args = [_convert_keras3_tensor_ref(arg) for arg in args]
    if not converted_args:
        return []

    if len(converted_args) == 1:
        node_data = converted_args[0]
        if not (
            isinstance(node_data, list)
            and not (len(node_data) in {3, 4} and isinstance(node_data[0], str))
        ):
            node_data = [node_data]
    else:
        node_data = converted_args

    if isinstance(raw_kwargs, dict) and raw_kwargs:
        node_data, _ = _attach_kwargs_to_first_tensor_ref(node_data, raw_kwargs)

    return node_data


def _is_legacy_tensor_ref(value: Any) -> bool:
    """True for old-style layer refs like ['layer_name', 0, 0]."""
    return (
        isinstance(value, list)
        and len(value) in {3, 4}
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and isinstance(value[2], int)
    )


def _legacy_tensor_ref_to_input_data(ref: Any) -> Any:
    """Convert a legacy tensor ref into legacy inbound-node input data."""
    if not _is_legacy_tensor_ref(ref):
        return ref
    if len(ref) == 4 and isinstance(ref[3], dict) and ref[3]:
        return ref
    return ref[:3]


def _rewrite_legacy_call_kwargs(class_name: Optional[str], nodes: Any) -> Any:
    """Convert old kwargs-based graph wiring into positional input refs where needed."""
    if not isinstance(nodes, list):
        return nodes

    rewritten = []
    for node in nodes:
        if not (isinstance(node, list) and node and isinstance(node[0], list)):
            rewritten.append(node)
            continue

        if len(node) != 1 or len(node[0]) not in {3, 4}:
            rewritten.append(node)
            continue

        input_data = node[0]
        kwargs = input_data[3] if len(input_data) == 4 and isinstance(input_data[3], dict) else {}
        if not kwargs:
            rewritten.append(node)
            continue

        base_kwargs = {k: v for k, v in kwargs.items() if k != "name" or v is not None}
        base_ref = input_data[:3] if not base_kwargs else [input_data[0], input_data[1], input_data[2], base_kwargs]

        if class_name == "MultiHeadAttention" and _is_legacy_tensor_ref(base_kwargs.get("value")):
            extra_ref = _legacy_tensor_ref_to_input_data(base_kwargs.pop("value"))
            base_ref = input_data[:3] if not base_kwargs else [input_data[0], input_data[1], input_data[2], base_kwargs]
            rewritten.append([base_ref, extra_ref])
            continue

        if class_name == "TFOpLambda" and _is_legacy_tensor_ref(base_kwargs.get("y")):
            extra_ref = _legacy_tensor_ref_to_input_data(base_kwargs.pop("y"))
            base_ref = input_data[:3] if not base_kwargs else [input_data[0], input_data[1], input_data[2], base_kwargs]
            rewritten.append([base_ref, extra_ref])
            continue

        rewritten.append([base_ref])

    return rewritten


def _sanitize_serialized_keras_config(
    config: Any,
    keras_module: Any,
    custom_objects: Dict[str, Any],
) -> Any:
    """Recursively sanitize serialized Keras configs for cross-version loading."""
    collapsed_dtype = _collapse_serialized_dtype_policy(config)
    if collapsed_dtype is not None:
        return collapsed_dtype

    if isinstance(config, list):
        return [
            _sanitize_serialized_keras_config(item, keras_module, custom_objects)
            for item in config
        ]

    if not isinstance(config, dict):
        return config

    sanitized = {}
    for key, value in config.items():
        if key == "build_config":
            normalized_build_config = _normalize_build_config_for_compat(
                config.get("class_name"),
                value,
                keras_module,
                custom_objects,
            )
            if normalized_build_config is not None:
                if config.get("class_name") == "MultiHeadAttention" and _keras_major_version(keras_module) < 3:
                    continue
                sanitized[key] = normalized_build_config
            continue
        if key == "inbound_nodes":
            sanitized_nodes = _sanitize_serialized_keras_config(
                value,
                keras_module,
                custom_objects,
            )
            if _keras_major_version(keras_module) < 3:
                sanitized_nodes = _normalize_inbound_nodes_for_keras2(sanitized_nodes)
            else:
                sanitized_nodes = _normalize_legacy_inbound_nodes(sanitized_nodes)
            sanitized[key] = _rewrite_legacy_call_kwargs(
                config.get("class_name"),
                sanitized_nodes,
            )
            continue
        if key == "config" and "class_name" in config and isinstance(value, dict):
            cls = _resolve_serialized_class(keras_module, config, custom_objects)
            nested = _sanitize_serialized_keras_config(value, keras_module, custom_objects)
            nested = _normalize_layer_config_for_compat(config.get("class_name"), nested)
            keras_major = _keras_major_version(keras_module)
            if config.get("class_name") == "MultiHeadAttention" and keras_major < 3:
                build_shapes = _normalize_build_config_for_compat(
                    "MultiHeadAttention",
                    config.get("build_config"),
                    keras_module,
                    custom_objects,
                )
                if isinstance(build_shapes, dict):
                    for shape_key in ("query_shape", "value_shape", "key_shape"):
                        shape_value = build_shapes.get(shape_key)
                        if shape_value is not None and shape_key not in nested:
                            nested[shape_key] = shape_value
            if (
                cls is not None
                and config.get("class_name") not in {"Functional", "Sequential", "Model"}
                and not (config.get("class_name") == "MultiHeadAttention" and keras_major < 3)
            ):
                nested = _filter_constructor_kwargs(cls, nested)
            sanitized[key] = nested
        else:
            sanitized[key] = _sanitize_serialized_keras_config(value, keras_module, custom_objects)

    if config.get("class_name") == "MultiHeadAttention":
        sanitized["registered_name"] = "LegacyMultiHeadAttention"
    elif config.get("class_name") == "Add" and config.get("module") == "keras.src.ops.numpy":
        sanitized["registered_name"] = "LegacyAddOperation"
    elif config.get("class_name") == "TFOpLambda":
        sanitized["registered_name"] = "LegacyTFOpLambda"
    return sanitized


def load_with_custom_deserialization(
    model_path: Union[str, Path],
    custom_objects: Optional[Dict] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a Keras model using custom deserialization with updated module paths.

    This advanced approach modifies the model configuration to update old
    module paths to new Keras 3.x paths, then uses custom deserializers
    to load the model.

    Args:
        model_path: Path to the model file or directory
        custom_objects: Optional dictionary mapping names to custom layer classes

    Returns:
        Tuple of (model, metadata) where metadata contains:
        - 'approach': 'custom_deserialization'
        - 'success': Boolean indicating success
        - 'optimizer_preserved': Boolean indicating if optimizer was preserved
        - 'warnings': List of warnings

    Raises:
        Exception: If custom deserialization fails
    """
    metadata = {
        'approach': 'custom_deserialization',
        'success': False,
        'optimizer_preserved': False,
        'warnings': []
    }

    logger.info(f"Attempting to load model with custom deserialization: {model_path}")

    try:
        import keras
    except ImportError:
        error_msg = "Keras is required for custom deserialization"
        metadata['warnings'].append(error_msg)
        logger.error(error_msg)
        raise ImportError(error_msg)

    try:
        # Register custom deserializers
        registered_objects = register_custom_deserializers()

        # Merge with provided custom objects
        if custom_objects:
            registered_objects.update(custom_objects)

        # Parse and update configuration
        config = parse_model_config(model_path)
        updated_config = update_module_paths_in_config(config)

        # Build the model from the serialized object config. Functional models need
        # Model.from_config(inner_config), not keras.models.model_from_config(...).
        model = _build_model_from_serialized_config(
            keras_module=keras,
            config=updated_config,
            custom_objects=registered_objects,
        )

        # Try to load weights
        model_path = Path(model_path)
        weights_loaded = False

        if model_path.is_dir():
            weights_files = list(model_path.glob("*.weights.h5")) + list(model_path.glob("weights.h5"))
            if weights_files:
                try:
                    model.load_weights(str(weights_files[0]), by_name=True)
                    weights_loaded = True
                    logger.info(f"Loaded weights from: {weights_files[0]}")
                except Exception as e:
                    metadata['warnings'].append(f"Failed to load weights: {str(e)}")
                    logger.warning(f"Failed to load weights: {e}")
        elif model_path.is_file() and model_path.suffix == '.keras':
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                weights_files = [f for f in zip_ref.namelist() if f.endswith('.weights.h5') or f == 'model.weights.h5']
                if weights_files:
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                            with zip_ref.open(weights_files[0]) as wf:
                                tmp.write(wf.read())
                            tmp_path = tmp.name

                        model.load_weights(tmp_path, by_name=True)
                        weights_loaded = True
                        logger.info("Loaded weights from .keras file")

                        os.unlink(tmp_path)
                    except Exception as e:
                        metadata['warnings'].append(f"Failed to load weights: {str(e)}")
                        logger.warning(f"Failed to load weights: {e}")

        # Try to restore optimizer state
        optimizer_preserved = False
        if 'optimizer' in updated_config:
            try:
                optimizer_config = updated_config['optimizer']
                optimizer_class_name = optimizer_config.get('class_name', 'Adam')
                optimizer_config_dict = optimizer_config.get('config', {})

                if optimizer_class_name == 'Adam':
                    optimizer = keras.optimizers.Adam(
                        learning_rate=optimizer_config_dict.get('learning_rate', 0.001),
                        beta_1=optimizer_config_dict.get('beta_1', 0.9),
                        beta_2=optimizer_config_dict.get('beta_2', 0.999)
                    )
                elif optimizer_class_name == 'SGD':
                    optimizer = keras.optimizers.SGD(
                        learning_rate=optimizer_config_dict.get('learning_rate', 0.01),
                        momentum=optimizer_config_dict.get('momentum', 0.0)
                    )
                else:
                    optimizer = keras.optimizers.Adam()

                model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                optimizer_preserved = True
                logger.info(f"Restored optimizer: {optimizer_class_name}")

            except Exception as e:
                metadata['warnings'].append(f"Failed to restore optimizer: {str(e)}")
                logger.warning(f"Failed to restore optimizer: {e}")

        if not weights_loaded:
            metadata['warnings'].append("Weights could not be loaded - model initialized randomly")

        metadata['success'] = True
        metadata['optimizer_preserved'] = optimizer_preserved

        if not optimizer_preserved:
            metadata['warnings'].append("Optimizer state not fully preserved (configuration only)")

        logger.info("Model loaded with custom deserialization")
        return model, metadata

    except Exception as e:
        error_msg = f"Failed to load model with custom deserialization: {str(e)}"
        metadata['warnings'].append(error_msg)
        logger.error(error_msg)
        raise Exception(error_msg) from e


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def load_weights_by_name(model: Any, weights_path: Union[str, Path]) -> None:
    """
    Load weights into a Keras model using by_name=True parameter.

    This function loads weights from a weights file, matching layers by name
    rather than by topology. This is useful when loading weights into models
    with slightly different architectures.

    Args:
        model: The Keras model to load weights into
        weights_path: Path to the weights file (.h5 or .weights.h5)

    Raises:
        FileNotFoundError: If the weights file doesn't exist
        ValueError: If the weights file is invalid
        Exception: If weight loading fails

    Example:
        >>> import keras
        >>> model = keras.Sequential([...])
        >>> load_weights_by_name(model, 'path/to/model.weights.h5')
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    logger.info(f"Loading weights from: {weights_path}")

    try:
        # Load weights with by_name=True
        model.load_weights(str(weights_path), by_name=True)
        logger.info("Weights loaded successfully with by_name=True")

    except Exception as e:
        error_msg = f"Failed to load weights from {weights_path}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def reinitialize_optimizer(
    model: Any,
    optimizer_config: Dict[str, Any],
    loss: Union[str, Any] = 'categorical_crossentropy',
    metrics: Optional[list] = None
) -> None:
    """
    Reinitialize a model's optimizer with matching hyperparameters from saved configuration.

    This function creates a new optimizer instance based on the saved configuration
    and compiles the model with it. This avoids variable mismatch warnings that occur
    when loading models with optimizer states from different Keras versions.

    Args:
        model: The Keras model to recompile
        optimizer_config: Optimizer configuration dictionary (from saved model)
        loss: Loss function to use for compilation
        metrics: Optional list of metrics to use for compilation

    Raises:
        ValueError: If optimizer configuration is invalid
        ImportError: If required optimizer class is not available
        Exception: If optimizer reinitialization fails

    Example:
        >>> import keras
        >>> model = keras.Sequential([...])
        >>> optimizer_config = {
        ...     'class_name': 'Adam',
        ...     'config': {'learning_rate': 0.001, 'beta_1': 0.9, 'beta_2': 0.999}
        ... }
        >>> reinitialize_optimizer(model, optimizer_config)
    """
    try:
        import keras
    except ImportError:
        raise ImportError("Keras is required for optimizer reinitialization")

    if metrics is None:
        metrics = ['accuracy']

    logger.info("Reinitializing optimizer from configuration")

    try:
        # Extract optimizer class name and configuration
        optimizer_class_name = optimizer_config.get('class_name', 'Adam')
        optimizer_config_dict = optimizer_config.get('config', {})

        logger.info(f"Optimizer class: {optimizer_class_name}")
        logger.info(f"Optimizer config: {optimizer_config_dict}")

        # Create optimizer based on class name
        optimizer = None

        if optimizer_class_name == 'Adam':
            optimizer = keras.optimizers.Adam(
                learning_rate=optimizer_config_dict.get('learning_rate', 0.001),
                beta_1=optimizer_config_dict.get('beta_1', 0.9),
                beta_2=optimizer_config_dict.get('beta_2', 0.999),
                epsilon=optimizer_config_dict.get('epsilon', 1e-7),
                amsgrad=optimizer_config_dict.get('amsgrad', False)
            )

        elif optimizer_class_name == 'SGD':
            optimizer = keras.optimizers.SGD(
                learning_rate=optimizer_config_dict.get('learning_rate', 0.01),
                momentum=optimizer_config_dict.get('momentum', 0.0),
                nesterov=optimizer_config_dict.get('nesterov', False)
            )

        elif optimizer_class_name == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(
                learning_rate=optimizer_config_dict.get('learning_rate', 0.001),
                rho=optimizer_config_dict.get('rho', 0.9),
                momentum=optimizer_config_dict.get('momentum', 0.0),
                epsilon=optimizer_config_dict.get('epsilon', 1e-7)
            )

        elif optimizer_class_name == 'Adagrad':
            optimizer = keras.optimizers.Adagrad(
                learning_rate=optimizer_config_dict.get('learning_rate', 0.01),
                initial_accumulator_value=optimizer_config_dict.get('initial_accumulator_value', 0.1),
                epsilon=optimizer_config_dict.get('epsilon', 1e-7)
            )

        elif optimizer_class_name == 'Adadelta':
            optimizer = keras.optimizers.Adadelta(
                learning_rate=optimizer_config_dict.get('learning_rate', 1.0),
                rho=optimizer_config_dict.get('rho', 0.95),
                epsilon=optimizer_config_dict.get('epsilon', 1e-7)
            )

        elif optimizer_class_name == 'Adamax':
            optimizer = keras.optimizers.Adamax(
                learning_rate=optimizer_config_dict.get('learning_rate', 0.002),
                beta_1=optimizer_config_dict.get('beta_1', 0.9),
                beta_2=optimizer_config_dict.get('beta_2', 0.999),
                epsilon=optimizer_config_dict.get('epsilon', 1e-7)
            )

        elif optimizer_class_name == 'Nadam':
            optimizer = keras.optimizers.Nadam(
                learning_rate=optimizer_config_dict.get('learning_rate', 0.002),
                beta_1=optimizer_config_dict.get('beta_1', 0.9),
                beta_2=optimizer_config_dict.get('beta_2', 0.999),
                epsilon=optimizer_config_dict.get('epsilon', 1e-7)
            )

        elif optimizer_class_name == 'Ftrl':
            optimizer = keras.optimizers.Ftrl(
                learning_rate=optimizer_config_dict.get('learning_rate', 0.001),
                learning_rate_power=optimizer_config_dict.get('learning_rate_power', -0.5),
                initial_accumulator_value=optimizer_config_dict.get('initial_accumulator_value', 0.1),
                l1_regularization_strength=optimizer_config_dict.get('l1_regularization_strength', 0.0),
                l2_regularization_strength=optimizer_config_dict.get('l2_regularization_strength', 0.0)
            )

        else:
            # Try to create optimizer dynamically from class name
            try:
                optimizer_class = getattr(keras.optimizers, optimizer_class_name)
                optimizer = optimizer_class(**optimizer_config_dict)
                logger.info(f"Created optimizer dynamically: {optimizer_class_name}")
            except AttributeError:
                logger.warning(
                    f"Unsupported optimizer class: {optimizer_class_name}, "
                    f"falling back to Adam"
                )
                optimizer = keras.optimizers.Adam()

        if optimizer is None:
            raise ValueError(f"Failed to create optimizer: {optimizer_class_name}")

        # Compile the model with the new optimizer
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        logger.info(
            f"Model recompiled successfully with {optimizer_class_name} optimizer "
            f"(learning_rate={optimizer_config_dict.get('learning_rate', 'default')})"
        )

    except Exception as e:
        error_msg = f"Failed to reinitialize optimizer: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def validate_model_structure(
    model: Any,
    expected_config: Dict[str, Any]
) -> bool:
    """
    Validate that a loaded model matches the expected structure.

    This function compares the loaded model's configuration against an expected
    configuration to ensure the model structure is correct. This is useful for
    verifying that a model was loaded correctly, especially when using the
    rebuilding approach.

    Args:
        model: The loaded Keras model
        expected_config: Expected model configuration dictionary

    Returns:
        Boolean indicating whether the model structure matches

    Raises:
        ValueError: If model or config is invalid
    """
    logger.info("Validating model structure against expected configuration")

    try:
        # Check basic attributes
        if not hasattr(model, 'layers'):
            logger.error("Model has no layers attribute")
            return False

        # Compare number of layers
        expected_layers = expected_config.get('config', {}).get('layers', [])
        actual_num_layers = len(model.layers)
        expected_num_layers = len(expected_layers)

        if actual_num_layers != expected_num_layers:
            logger.warning(
                f"Layer count mismatch: expected {expected_num_layers}, "
                f"got {actual_num_layers}"
            )
            return False

        # Compare layer types
        for i, (layer, expected_layer) in enumerate(zip(model.layers, expected_layers)):
            expected_class_name = expected_layer.get('class_name')
            actual_class_name = layer.__class__.__name__

            if expected_class_name != actual_class_name:
                logger.warning(
                    f"Layer {i} type mismatch: expected {expected_class_name}, "
                    f"got {actual_class_name}"
                )
                return False

        # Compare input/output shapes
        expected_input_layers = expected_config.get('config', {}).get('input_layers', [])
        if expected_input_layers:
            expected_input_name = expected_input_layers[0][0]
            expected_input_layer = next(
                (layer for layer in expected_layers if layer.get('config', {}).get('name') == expected_input_name),
                None
            )

            if expected_input_layer:
                expected_shape = expected_input_layer.get('config', {}).get('batch_input_shape')
                if expected_shape and model.input_shape != expected_shape:
                    logger.warning(
                        f"Input shape mismatch: expected {expected_shape}, "
                        f"got {model.input_shape}"
                    )
                    return False

        logger.info("Model structure validation passed")
        return True

    except Exception as e:
        logger.error(f"Model structure validation failed: {e}")
        return False


# =============================================================================
# KERAS MODEL LOADER CLASS
# =============================================================================

class KerasModelLoader:
    """
    Main class for loading Keras models with cross-version compatibility.

    This class provides a high-level interface for loading Keras models,
    particularly for handling Keras 2.15.0 models in Keras 3.x environments.
    It automatically detects the best loading approach and provides comprehensive
    metadata about the loaded model.

    Attributes:
        model_path: Path to the model file or directory
        model: The loaded Keras model (None until load_model() is called)
        metadata: Dictionary containing model metadata and loading information

    Example:
        >>> loader = KerasModelLoader('path/to/model.keras')
        >>> model = loader.load_model()
        >>> info = loader.get_model_info()
        >>> print(f"Model has {info['validation']['num_layers']} layers")
    """

    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the KerasModelLoader with a model path.

        Args:
            model_path: Path to the model file or directory

        Raises:
            FileNotFoundError: If the model path doesn't exist
            ValueError: If the model format is not recognized
        """
        self.model_path = Path(model_path)
        self.model: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}

        # Validate model path exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        # Detect model format
        self.model_format = check_model_format(self.model_path)
        if self.model_format == 'unknown':
            raise ValueError(f"Unknown model format: {self.model_path}")

        # Extract basic metadata
        self._extract_initial_metadata()

        logger.info(
            f"KerasModelLoader initialized for {self.model_path} "
            f"(format: {self.model_format})"
        )

    def _extract_initial_metadata(self) -> None:
        """
        Extract initial metadata from the model file.

        This method is called during initialization to gather basic information
        about the model without loading it.
        """
        try:
            # Extract model metadata
            model_metadata = extract_model_metadata(self.model_path)

            # Detect Keras version
            keras_version_info = detect_keras_version_info()

            # Store initial metadata
            self.metadata = {
                'model_path': str(self.model_path),
                'model_format': self.model_format,
                'model_metadata': model_metadata,
                'keras_version_info': keras_version_info,
                'loaded': False,
                'load_time': None,
                'approach_used': None
            }

        except Exception as e:
            logger.warning(f"Failed to extract initial metadata: {e}")
            self.metadata = {
                'model_path': str(self.model_path),
                'model_format': self.model_format,
                'error': str(e)
            }

    def load_model(
        self,
        custom_objects: Optional[Dict] = None,
        preferred_approach: Optional[str] = None,
        compile: bool = True,
        expected_input_shape: Optional[Tuple] = None
    ) -> Any:
        """
        Load the Keras model using the best available approach.

        This method attempts to load the model using multiple approaches with
        automatic fallback. It uses the intelligent loader function to determine
        the best approach based on the environment and model format.

        Args:
            custom_objects: Optional dictionary mapping names to custom layer classes
            preferred_approach: Force a specific approach ('tf_keras', 'rebuild', 'custom_deserialization')
            compile: Whether to compile the model with the saved optimizer
            expected_input_shape: Optional expected input shape for validation

        Returns:
            The loaded Keras model

        Raises:
            Exception: If model loading fails

        Example:
            >>> loader = KerasModelLoader('path/to/model.keras')
            >>> model = loader.load_model(compile=True)
            >>> predictions = model.predict(X_test)
        """
        import time

        logger.info(f"Loading model from: {self.model_path}")

        start_time = time.time()

        try:
            # Use the intelligent loader
            self.model, load_metadata = load_keras_model(
                self.model_path,
                custom_objects=custom_objects,
                preferred_approach=preferred_approach,
                compile=compile,
                expected_input_shape=expected_input_shape
            )

            # Update metadata
            self.metadata.update(load_metadata)
            self.metadata['loaded'] = True
            self.metadata['load_time'] = time.time() - start_time

            logger.info(
                f"Model loaded successfully in {self.metadata['load_time']:.2f}s "
                f"using approach: {load_metadata['approach_used']}"
            )

            return self.model

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            self.metadata['error'] = error_msg
            raise Exception(error_msg) from e

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return comprehensive information about the loaded model.

        This method returns a dictionary containing metadata about the model,
        including loading information, validation results, and model structure.

        Returns:
            Dictionary containing model information:
            - 'model_path': Path to the model file
            - 'model_format': Format of the model file
            - 'loaded': Whether the model has been loaded
            - 'load_time': Time taken to load the model (seconds)
            - 'approach_used': Which loading approach was used
            - 'optimizer_preserved': Whether optimizer state was preserved
            - 'validation': Model validation results
            - 'keras_version_info': Keras version information
            - 'model_metadata': Extracted model metadata
            - 'warnings': List of warnings

        Raises:
            ValueError: If model has not been loaded yet

        Example:
            >>> loader = KerasModelLoader('path/to/model.keras')
            >>> model = loader.load_model()
            >>> info = loader.get_model_info()
            >>> print(f"Model: {info['model_metadata']['model_name']}")
            >>> print(f"Layers: {info['validation']['num_layers']}")
            >>> print(f"Parameters: {info['validation']['trainable_params']:,}")
        """
        if not self.metadata.get('loaded', False):
            raise ValueError("Model has not been loaded yet. Call load_model() first.")

        return self.metadata

    def get_model_summary(self) -> str:
        """
        Get a formatted summary of the loaded model.

        This method returns a string containing a human-readable summary of the
        model, including architecture, parameters, and loading information.

        Returns:
            Formatted string containing model summary

        Raises:
            ValueError: If model has not been loaded yet

        Example:
            >>> loader = KerasModelLoader('path/to/model.keras')
            >>> model = loader.load_model()
            >>> print(loader.get_model_summary())
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet. Call load_model() first.")

        import io
        import sys

        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            self.model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout

        return summary

    def validate(self, expected_input_shape: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Validate the loaded model structure and weights.

        This method performs comprehensive validation of the loaded model,
        checking structure, weights, and optional input shape matching.

        Args:
            expected_input_shape: Optional expected input shape for validation

        Returns:
            Dictionary containing validation results

        Raises:
            ValueError: If model has not been loaded yet

        Example:
            >>> loader = KerasModelLoader('path/to/model.keras')
            >>> model = loader.load_model()
            >>> validation = loader.validate(expected_input_shape=(60, 14))
            >>> if validation['valid']:
            ...     print("Model is valid!")
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet. Call load_model() first.")

        return validate_model_loading(self.model, expected_input_shape)


# =============================================================================
# MAIN INTELLIGENT LOADER FUNCTION
# =============================================================================

def load_keras_model(
    model_path: Union[str, Path],
    custom_objects: Optional[Dict] = None,
    preferred_approach: Optional[str] = None,
    compile: bool = True,
    expected_input_shape: Optional[Tuple] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Intelligently load a Keras model using multiple approaches with automatic fallback.

    This function attempts to load a Keras model using different approaches in order:
    1. tf.keras backward compatibility (if available)
    2. Model rebuilding + weight loading (for Keras 3.x migration)
    3. Custom deserialization (advanced)

    It automatically detects the best approach based on the environment and model format,
    and provides detailed metadata about the loading process.

    Args:
        model_path: Path to the model file or directory
        custom_objects: Optional dictionary mapping names to custom layer classes
        preferred_approach: Force a specific approach ('tf_keras', 'rebuild', 'custom_deserialization')
        compile: Whether to compile the model with the saved optimizer (for tf_keras approach)
        expected_input_shape: Optional expected input shape for validation

    Returns:
        Tuple of (model, metadata) where metadata contains:
        - 'approach_used': Which approach succeeded
        - 'success': Boolean indicating success
        - 'optimizer_preserved': Whether optimizer state was preserved
        - 'validation': Model validation results
        - 'warnings': List of warnings
        - 'keras_version_info': Keras version information

    Raises:
        FileNotFoundError: If model path doesn't exist
        Exception: If all loading approaches fail
    """
    logger.info(f"Loading Keras model: {model_path}")

    # Detect Keras version info
    keras_version_info = detect_keras_version_info()

    # Check model format
    model_format = check_model_format(model_path)

    # Extract metadata
    model_metadata = extract_model_metadata(model_path)

    # Initialize result metadata
    result_metadata = {
        'approach_used': None,
        'success': False,
        'optimizer_preserved': False,
        'validation': None,
        'warnings': [],
        'keras_version_info': keras_version_info,
        'model_format': model_format,
        'model_metadata': model_metadata
    }

    # Add warnings about version mismatch
    if model_metadata['keras_version']:
        saved_version = model_metadata['keras_version']
        current_version = keras_version_info.get('keras_version')
        if saved_version and current_version:
            if saved_version.startswith('2.') and current_version.startswith('3.'):
                result_metadata['warnings'].append(
                    f"Version mismatch: model saved with Keras {saved_version}, "
                    f"loading with Keras {current_version}"
                )
                logger.warning(f"Version mismatch: {saved_version} -> {current_version}")

    # Determine which approaches to try
    approaches_to_try = []

    if preferred_approach:
        approaches_to_try.append(preferred_approach)
    else:
        # Prefer the approach inferred from model metadata first, then fallback.
        tf_available = bool(keras_version_info['tf_keras_available'] or keras_version_info['tensorflow_available'])
        native_available = bool(keras_version_info.get('keras_version'))

        prefers_tf = _model_prefers_tf_keras(model_metadata)
        prefers_native = _model_prefers_keras_native(model_metadata)

        if prefers_tf:
            if tf_available:
                approaches_to_try.append('tf_keras')
            if native_available:
                approaches_to_try.append('keras_native')
        elif prefers_native:
            if native_available:
                approaches_to_try.append('keras_native')
            if tf_available:
                approaches_to_try.append('tf_keras')
        else:
            # Fallback heuristic from runtime environment when metadata is unclear.
            keras_ver = str(keras_version_info.get('keras_version') or '')
            if native_available and keras_ver.startswith('3.'):
                approaches_to_try.append('keras_native')
            if tf_available:
                approaches_to_try.append('tf_keras')

        approaches_to_try.append('custom_deserialization')
        approaches_to_try.append('rebuild')

        # Preserve order while removing accidental duplicates.
        approaches_to_try = list(dict.fromkeys(approaches_to_try))

    # Try each approach
    model = None
    last_error = None

    for approach in approaches_to_try:
        logger.info(f"Trying approach: {approach}")

        try:
            if approach == 'keras_native':
                model, approach_metadata = load_with_keras_native(
                    model_path, custom_objects, compile=False  # Always compile=False to avoid LR schedule issues
                )
            elif approach == 'tf_keras':
                model, approach_metadata = load_with_tf_keras(
                    model_path, custom_objects, compile=compile
                )
            elif approach == 'rebuild':
                model, approach_metadata = load_by_rebuilding(
                    model_path, custom_objects
                )
            elif approach == 'custom_deserialization':
                model, approach_metadata = load_with_custom_deserialization(
                    model_path, custom_objects
                )
            else:
                logger.warning(f"Unknown approach: {approach}")
                continue

            # Success!
            result_metadata['approach_used'] = approach
            result_metadata['success'] = True
            result_metadata['optimizer_preserved'] = approach_metadata['optimizer_preserved']
            result_metadata['warnings'].extend(approach_metadata['warnings'])

            # Validate the model
            validation = validate_model_loading(model, expected_input_shape)
            result_metadata['validation'] = validation
            result_metadata['warnings'].extend(validation['warnings'])

            logger.info(f"Model loaded successfully using approach: {approach}")
            break

        except Exception as e:
            last_error = e
            err = _compact_error(e)
            if _is_expected_compat_load_error(err):
                logger.debug(f"Approach '{approach}' incompatibility (expected fallback): {err}")
            else:
                logger.warning(f"Approach '{approach}' failed: {err}")
            result_metadata['warnings'].append(f"Approach '{approach}' failed: {err}")

    # Check if any approach succeeded
    if model is None:
        error_msg = f"All loading approaches failed. Last error: {_compact_error(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    return model, result_metadata


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the Keras model loader.

    This demonstrates how to use the KerasModelLoader class and utility functions
    to load a Keras 2.15.0 model in a Keras 3.x environment.
    """

    print("=" * 80)
    print("Keras Model Loader - Cross-Version Compatibility Solution")
    print("=" * 80)
    print()

    # Example 1: Detect Keras version
    print("Example 1: Detect Keras Version")
    print("-" * 40)
    try:
        version = detect_keras_version()
        print(f"  Keras version: {version}")
    except ImportError as e:
        print(f"  Error: {e}")
    print()

    # Example 2: Detect model format (replace with actual path)
    print("Example 2: Detect Model Format")
    print("-" * 40)
    example_model_path = "path/to/your/model.keras"  # Replace with actual path
    print(f"  Checking format for: {example_model_path}")
    try:
        model_format = detect_model_format(example_model_path)
        print(f"  Format: {model_format}")
    except FileNotFoundError:
        print("  Model not found (this is expected in the example)")
    print()

    # Example 3: Extract model metadata
    print("Example 3: Extract Model Metadata")
    print("-" * 40)
    print(f"  Extracting metadata from: {example_model_path}")
    try:
        metadata = extract_model_metadata(example_model_path)
        print(f"  Format: {metadata['format']}")
        print(f"  Keras Version: {metadata['keras_version']}")
        print(f"  Backend: {metadata['backend']}")
        print(f"  Model Name: {metadata['model_name']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Example 4: Using KerasModelLoader class (recommended)
    print("Example 4: Using KerasModelLoader Class (Recommended)")
    print("-" * 40)
    print(f"  Loading model from: {example_model_path}")
    try:
        # Initialize loader
        loader = KerasModelLoader(example_model_path)

        # Load the model
        model = loader.load_model(
            compile=True,
            expected_input_shape=(60, 14)  # Adjust based on your model
        )

        # Get model information
        info = loader.get_model_info()

        print("  ✓ Model loaded successfully!")
        print(f"  Approach used: {info['approach_used']}")
        print(f"  Optimizer preserved: {info['optimizer_preserved']}")
        print(f"  Validation passed: {info['validation']['valid']}")
        print(f"  Number of layers: {info['validation']['num_layers']}")
        print(f"  Trainable parameters: {info['validation']['trainable_params']:,}")
        print(f"  Load time: {info['load_time']:.2f}s")

        if info['warnings']:
            print("\n  Warnings:")
            for warning in info['warnings']:
                print(f"    - {warning}")

        # Get model summary
        print("\n  Model Summary:")
        print(loader.get_model_summary())

    except FileNotFoundError:
        print("  Model not found (this is expected in the example)")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Example 5: Load model with intelligent loader (functional approach)
    print("Example 5: Load Model with Intelligent Loader (Functional)")
    print("-" * 40)
    print(f"  Loading model from: {example_model_path}")
    try:
        model, load_metadata = load_keras_model(
            example_model_path,
            compile=True,
            expected_input_shape=(60, 14)  # Adjust based on your model
        )

        print("  ✓ Model loaded successfully!")
        print(f"  Approach used: {load_metadata['approach_used']}")
        print(f"  Optimizer preserved: {load_metadata['optimizer_preserved']}")
        print(f"  Validation passed: {load_metadata['validation']['valid']}")

    except FileNotFoundError:
        print("  Model not found (this is expected in the example)")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Example 6: Load model with specific approach
    print("Example 6: Load Model with Specific Approach")
    print("-" * 40)
    print("  Available approaches:")
    print("    - tf_keras: Use tf.keras for backward compatibility (recommended)")
    print("    - rebuild: Rebuild architecture and load weights")
    print("    - custom_deserialization: Use custom deserializers")
    print()
    print(f"  Loading with tf_keras approach: {example_model_path}")
    try:
        model, metadata = load_keras_model(
            example_model_path,
            preferred_approach='tf_keras',
            compile=True
        )
        print("  ✓ Loaded with tf_keras approach")
    except FileNotFoundError:
        print("  Model not found (this is expected in the example)")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Example 7: Load weights only (no optimizer)
    print("Example 7: Load Weights Only")
    print("-" * 40)
    print("  Loading weights without optimizer state")
    try:
        model, metadata = load_keras_model(
            example_model_path,
            compile=False  # Don't compile with saved optimizer
        )
        print("  ✓ Weights loaded (optimizer not compiled)")
    except FileNotFoundError:
        print("  Model not found (this is expected in the example)")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Example 8: Reinitialize optimizer
    print("Example 8: Reinitialize Optimizer")
    print("-" * 40)
    print("  Reinitializing optimizer with saved configuration")
    try:
        import keras
        # Create a simple model for demonstration
        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(5,))
        ])

        optimizer_config = {
            'class_name': 'Adam',
            'config': {
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999
            }
        }

        reinitialize_optimizer(model, optimizer_config)
        print("  ✓ Optimizer reinitialized successfully")

    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Example 9: Load weights by name
    print("Example 9: Load Weights by Name")
    print("-" * 40)
    print("  Loading weights using by_name=True parameter")
    try:
        import keras
        # Create a simple model for demonstration
        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(5,), name='dense_1')
        ])

        # Note: This would normally load from an actual weights file
        # weights_path = 'path/to/model.weights.h5'
        # load_weights_by_name(model, weights_path)
        print("  ✓ Function ready to load weights by name")
        print("  (Replace weights_path with actual file path)")

    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Example 10: Validate model structure
    print("Example 10: Validate Model Structure")
    print("-" * 40)
    print("  Validating model structure against expected configuration")
    try:
        import keras
        # Create a simple model for demonstration
        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(5,), name='dense_1'),
            keras.layers.Dense(1, name='dense_2')
        ])

        expected_config = {
            'config': {
                'layers': [
                    {'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, 5)}},
                    {'class_name': 'Dense', 'config': {'name': 'dense_1', 'units': 10}},
                    {'class_name': 'Dense', 'config': {'name': 'dense_2', 'units': 1}}
                ]
            }
        }

        is_valid = validate_model_structure(model, expected_config)
        print(f"  Model structure valid: {is_valid}")

    except Exception as e:
        print(f"  Error: {e}")
    print()

    print("=" * 80)
    print("For production use:")
    print("  1. Replace 'path/to/your/model.keras' with your actual model path")
    print("  2. Adjust expected_input_shape based on your model")
    print("  3. Use KerasModelLoader class for the best experience")
    print("  4. Use the returned model for inference or further training")
    print("  5. Check warnings for any issues")
    print("=" * 80)
