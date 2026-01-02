"""Configuration validation module for ML Engine."""

import logging
from typing import Any, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration dictionaries for ML Engine."""
    
    # Required top-level keys
    REQUIRED_KEYS = ["device"]
    
    # Valid architecture types
    VALID_ARCHITECTURES = ["lstm", "attention_lstm", "gru", "transformer", "tcn", "ensemble"]
    
    # Valid optimizer types
    VALID_OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop"]
    
    # Valid scheduler types
    VALID_SCHEDULERS = ["plateau", "cosine", "step", "exponential"]
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required keys
        for key in ConfigValidator.REQUIRED_KEYS:
            if key not in config:
                errors.append(f"Missing required configuration key: {key}")
        
        # Validate device
        if "device" in config:
            device = config["device"]
            if device not in ["cpu", "cuda", "mps"]:
                errors.append(f"Invalid device: {device}. Must be 'cpu', 'cuda', or 'mps'")
        
        # Validate model configuration
        if "model" in config:
            model_errors = ConfigValidator._validate_model(config["model"])
            errors.extend(model_errors)
        
        # Validate optimizer configuration
        if "optimizer" in config:
            optimizer_errors = ConfigValidator._validate_optimizer(config["optimizer"])
            errors.extend(optimizer_errors)
        
        # Validate scheduler configuration
        if "scheduler" in config:
            scheduler_errors = ConfigValidator._validate_scheduler(config["scheduler"])
            errors.extend(scheduler_errors)
        
        # Validate training configuration
        if "training" in config:
            training_errors = ConfigValidator._validate_training(config["training"])
            errors.extend(training_errors)
        
        # Validate paths
        if "paths" in config:
            path_errors = ConfigValidator._validate_paths(config["paths"])
            errors.extend(path_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_model(model_config: Dict[str, Any]) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        # Check architecture
        if "architecture" in model_config:
            arch = model_config["architecture"]
            if arch not in ConfigValidator.VALID_ARCHITECTURES:
                errors.append(
                    f"Invalid architecture: {arch}. Must be one of {ConfigValidator.VALID_ARCHITECTURES}"
                )
        
        # Validate numeric parameters
        if "hidden_size" in model_config:
            if not isinstance(model_config["hidden_size"], int) or model_config["hidden_size"] <= 0:
                errors.append("hidden_size must be a positive integer")
        
        if "num_layers" in model_config:
            if not isinstance(model_config["num_layers"], int) or model_config["num_layers"] <= 0:
                errors.append("num_layers must be a positive integer")
        
        if "dropout" in model_config:
            dropout = model_config["dropout"]
            if not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1:
                errors.append("dropout must be a number between 0 and 1")
        
        return errors
    
    @staticmethod
    def _validate_optimizer(optimizer_config: Dict[str, Any]) -> List[str]:
        """Validate optimizer configuration."""
        errors = []
        
        # Check optimizer type
        if "type" in optimizer_config:
            opt_type = optimizer_config["type"].lower()
            if opt_type not in ConfigValidator.VALID_OPTIMIZERS:
                errors.append(
                    f"Invalid optimizer type: {opt_type}. Must be one of {ConfigValidator.VALID_OPTIMIZERS}"
                )
        
        # Validate learning rate
        if "learning_rate" in optimizer_config:
            lr = optimizer_config["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append("learning_rate must be a positive number")
        
        # Validate weight decay
        if "weight_decay" in optimizer_config:
            wd = optimizer_config["weight_decay"]
            if not isinstance(wd, (int, float)) or wd < 0:
                errors.append("weight_decay must be a non-negative number")
        
        return errors
    
    @staticmethod
    def _validate_scheduler(scheduler_config: Dict[str, Any]) -> List[str]:
        """Validate scheduler configuration."""
        errors = []
        
        # Check scheduler type
        if "type" in scheduler_config:
            sched_type = scheduler_config["type"].lower()
            if sched_type not in ConfigValidator.VALID_SCHEDULERS:
                errors.append(
                    f"Invalid scheduler type: {sched_type}. Must be one of {ConfigValidator.VALID_SCHEDULERS}"
                )
        
        return errors
    
    @staticmethod
    def _validate_training(training_config: Dict[str, Any]) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        # Validate epochs
        if "epochs" in training_config:
            epochs = training_config["epochs"]
            if not isinstance(epochs, int) or epochs <= 0:
                errors.append("epochs must be a positive integer")
        
        # Validate early_stopping_patience
        if "early_stopping_patience" in training_config:
            patience = training_config["early_stopping_patience"]
            if not isinstance(patience, int) or patience <= 0:
                errors.append("early_stopping_patience must be a positive integer")
        
        return errors
    
    @staticmethod
    def _validate_paths(paths_config: Dict[str, Any]) -> List[str]:
        """Validate paths configuration."""
        errors = []
        
        # Check if paths exist or can be created
        path_keys = ["model_dir", "data_dir", "log_dir", "checkpoint_dir"]
        for key in path_keys:
            if key in paths_config:
                path = Path(paths_config[key])
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory for {key}: {e}")
        
        return errors


def validate_config(config: Dict[str, Any], raise_on_error: bool = False) -> bool:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        raise_on_error: If True, raise ValueError on validation errors
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If raise_on_error is True and validation fails
    """
    is_valid, errors = ConfigValidator.validate(config)
    
    if not is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        logger.error(error_msg)
        
        if raise_on_error:
            raise ValueError(error_msg)
    
    return is_valid


if __name__ == "__main__":
    # Test the validator
    test_config = {
        "device": "cpu",
        "model": {
            "architecture": "lstm",
            "hidden_size": 128,
            "num_layers": 3,
            "dropout": 0.2
        },
        "optimizer": {
            "type": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        },
        "training": {
            "epochs": 100,
            "early_stopping_patience": 10
        }
    }
    
    is_valid, errors = ConfigValidator.validate(test_config)
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
