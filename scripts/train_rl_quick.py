#!/usr/bin/env python
"""
Quick RL Training Script for Gate Thresholds and Exit Timing.

Generates synthetic data and trains RL models with minimal timesteps
to verify the pipeline works.

Improvements:
- Enhanced code readability with type hints and dataclasses
- Performance optimizations through vectorization
- Best practices with logging and proper error handling
- Comprehensive edge case handling and validation
"""
from __future__ import annotations

import os
import sys
import logging
import argparse
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np

# Suppress TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Configuration Constants
# ============================================================================

class TrainingMode(Enum):
    """Training mode options."""
    GATES_ONLY = "gates_only"
    EXITS_ONLY = "exits_only"
    BOTH = "both"


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 5000
    base_price: float = 1.0
    return_std: float = 0.001
    momentum_factor: float = 0.3
    noise_std: float = 0.2
    n_features: int = 10
    price_path_length: int = 20
    n_scenarios: int = 500
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    timesteps: int = 10000
    buffer_size: int = 10000
    batch_size: int = 64
    eval_freq: int = 1000
    verbose: int = 1
    model_dir: str = "trained_data/models"
    log_dir: str = "trained_data/logs"


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(verbose: bool = True) -> logging.Logger:
    """
    Configure logging for the training script.
    
    Args:
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


# Global logger instance
logger = setup_logging()


# ============================================================================
# Data Generation Functions
# ============================================================================

def generate_returns_with_momentum(
    n_samples: int,
    std: float,
    momentum_factor: float,
    random_seed: int
) -> np.ndarray:
    """
    Generate returns with momentum component using vectorized operations.
    
    Args:
        n_samples: Number of samples to generate
        std: Standard deviation of returns
        momentum_factor: Strength of momentum (0-1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of returns with momentum
    """
    rng = np.random.default_rng(random_seed)
    
    # Generate base returns
    base_returns = rng.normal(0, std, n_samples)
    
    # Apply momentum using cumulative sum with decay
    if momentum_factor > 0:
        # Vectorized momentum calculation
        momentum = np.zeros(n_samples)
        momentum[0] = base_returns[0]
        for i in range(1, n_samples):
            momentum[i] = momentum_factor * momentum[i-1] + (1 - momentum_factor) * base_returns[i]
        returns = momentum
    else:
        returns = base_returns
    
    return returns


def generate_features(
    returns: np.ndarray,
    n_features: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate feature matrix from returns.
    
    Args:
        returns: Array of returns
        n_features: Number of features to generate
        rng: Random number generator
        
    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    n_samples = len(returns)
    features = np.zeros((n_samples, n_features))
    
    # Feature 0: Volatility proxy (absolute returns)
    features[:, 0] = np.abs(returns) * 100
    
    # Feature 1: Momentum (5-period moving average)
    if n_samples >= 5:
        kernel = np.ones(5) / 5
        features[:, 1] = np.convolve(returns, kernel, mode='same') * 100
    else:
        features[:, 1] = returns * 100
    
    # Feature 2: ADX-like indicator
    features[:, 2] = rng.uniform(20, 80, n_samples)
    
    # Feature 3: RSI-like indicator
    features[:, 3] = rng.uniform(0, 100, n_samples)
    
    # Feature 4: Z-score
    features[:, 4] = rng.uniform(-1, 1, n_samples)
    
    # Features 5-9: Additional random features
    for i in range(5, n_features):
        features[:, i] = rng.normal(0, 1, n_samples)
    
    return features


def generate_ensemble_predictions(
    returns: np.ndarray,
    noise_std: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate ensemble predictions with noise.
    
    Args:
        returns: Array of returns
        noise_std: Standard deviation of noise
        rng: Random number generator
        
    Returns:
        Array of predictions clipped to [0.1, 0.9]
    """
    true_direction = (returns > 0).astype(float)
    noise = rng.normal(0, noise_std, len(returns))
    predictions = true_direction * 0.6 + 0.2 + noise
    
    return np.clip(predictions, 0.1, 0.9)


def generate_synthetic_data(config: SyntheticDataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic market data for RL training.
    
    Args:
        config: Configuration for data generation
        
    Returns:
        Tuple of (prices, features, ensemble_predictions)
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Validate configuration
    if config.n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if config.n_features < 5:
        raise ValueError("n_features must be at least 5")
    
    logger.info(f"Generating synthetic data with {config.n_samples} samples")
    
    # Initialize random generator
    rng = np.random.default_rng(config.random_seed)
    
    # Generate returns with momentum
    returns = generate_returns_with_momentum(
        config.n_samples,
        config.return_std,
        config.momentum_factor,
        config.random_seed
    )
    
    # Generate prices from returns
    prices = config.base_price * np.exp(np.cumsum(returns))
    
    # Generate features
    features = generate_features(returns, config.n_features, rng)
    
    # Generate ensemble predictions
    ensemble_preds = generate_ensemble_predictions(returns, config.noise_std, rng)
    
    logger.info(f"  Prices shape: {prices.shape}")
    logger.info(f"  Features shape: {features.shape}")
    logger.info(f"  Predictions shape: {ensemble_preds.shape}")
    
    return prices, features, ensemble_preds


def generate_trade_scenarios(config: SyntheticDataConfig) -> List[Any]:
    """
    Generate trade scenarios for exit timing training.
    
    Args:
        config: Configuration for scenario generation
        
    Returns:
        List of TradeScenario objects
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Validate configuration
    if config.n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    if config.price_path_length <= 0:
        raise ValueError("price_path_length must be positive")
    
    logger.info(f"Generating {config.n_scenarios} trade scenarios")
    
    # Import TradeScenario here to avoid circular imports
    from src.rl.optimal_exit_env import TradeScenario
    
    # Initialize random generator with different seed
    rng = np.random.default_rng(config.random_seed + 1000)
    
    scenarios = []
    
    for i in range(config.n_scenarios):
        try:
            # Random entry price
            entry_price = 1.0 + rng.normal(0, 0.01)
            
            # Random price path
            returns = rng.normal(0, 0.002, config.price_path_length)
            price_path = entry_price * np.exp(np.cumsum(returns))
            
            # Random direction
            direction = rng.choice([1, -1])
            
            # Create scenario
            scenario = TradeScenario(
                entry_price=entry_price,
                direction=direction,
                prices=price_path,
                momentum=rng.normal(0, 0.01, config.price_path_length),
                atr=np.abs(rng.normal(0, 0.002, config.price_path_length)) + 0.001,
            )
            scenarios.append(scenario)
            
        except Exception as e:
            logger.warning(f"Failed to generate scenario {i}: {e}")
            continue
    
    logger.info(f"  Successfully generated {len(scenarios)} scenarios")
    
    if len(scenarios) == 0:
        raise RuntimeError("Failed to generate any trade scenarios")
    
    return scenarios


# ============================================================================
# Directory Management
# ============================================================================

def ensure_directories_exist(config: TrainingConfig) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Training configuration
        
    Raises:
        OSError: If directory creation fails
    """
    directories = [
        Path(config.model_dir),
        Path(config.log_dir) / "rl_gates",
        Path(config.log_dir) / "rl_exits",
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


# ============================================================================
# Training Functions
# ============================================================================

def train_gate_thresholds(
    config: TrainingConfig,
    data_config: SyntheticDataConfig
) -> Optional[Dict[str, Any]]:
    """
    Train Gate Threshold RL model using SAC.
    
    Args:
        config: Training configuration
        data_config: Data generation configuration
        
    Returns:
        Training statistics dictionary, or None if training failed
        
    Raises:
        ImportError: If required modules cannot be imported
        RuntimeError: If training fails
    """
    logger.info("=" * 60)
    logger.info("TRAINING RL GATE THRESHOLDS (SAC)")
    logger.info("=" * 60)
    
    try:
        from src.rl.gate_threshold_env import GateThresholdRL, GateRLConfig
    except ImportError as e:
        logger.error(f"Failed to import gate threshold modules: {e}")
        raise
    
    try:
        # Generate training data
        logger.info("Generating synthetic training data...")
        prices, features, ensemble_preds = generate_synthetic_data(data_config)
        
        # Validate data shapes
        if len(prices) != len(features) or len(prices) != len(ensemble_preds):
            raise ValueError(
                f"Data shape mismatch: prices={len(prices)}, "
                f"features={len(features)}, preds={len(ensemble_preds)}"
            )
        
        # Create trainer configuration
        rl_config = GateRLConfig(
            total_timesteps=config.timesteps,
            buffer_size=min(config.buffer_size, config.timesteps),
            batch_size=config.batch_size,
        )
        
        # Create trainer
        trainer = GateThresholdRL(config=rl_config)
        
        # Train
        logger.info(f"Starting SAC training for {config.timesteps:,} timesteps...")
        stats = trainer.train(
            prices=prices,
            features=features,
            ensemble_preds=ensemble_preds,
            timesteps=config.timesteps,
            eval_freq=max(config.eval_freq, config.timesteps // 5),
            verbose=config.verbose,
        )
        
        # Save model
        model_path = Path(config.model_dir) / "sac_gate_thresholds.zip"
        trainer.save()
        logger.info(f"✓ Gate threshold model saved to: {model_path}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Gate threshold training failed: {e}")
        logger.debug(traceback.format_exc())
        return None


def train_exit_timing(
    config: TrainingConfig,
    data_config: SyntheticDataConfig
) -> Optional[Dict[str, Any]]:
    """
    Train Exit Timing RL model using PPO.
    
    Args:
        config: Training configuration
        data_config: Data generation configuration
        
    Returns:
        Training statistics dictionary, or None if training failed
        
    Raises:
        ImportError: If required modules cannot be imported
        RuntimeError: If training fails
    """
    logger.info("=" * 60)
    logger.info("TRAINING RL EXIT TIMING (PPO)")
    logger.info("=" * 60)
    
    try:
        from src.rl.optimal_exit_env import OptimalExitRL, ExitRLConfig
    except ImportError as e:
        logger.error(f"Failed to import exit timing modules: {e}")
        raise
    
    try:
        # Generate trade scenarios
        logger.info("Generating trade scenarios...")
        scenarios = generate_trade_scenarios(data_config)
        
        # Validate scenarios
        if not scenarios:
            raise ValueError("No scenarios generated for training")
        
        # Create trainer configuration
        rl_config = ExitRLConfig(
            total_timesteps=config.timesteps,
        )
        
        # Create trainer
        trainer = OptimalExitRL(config=rl_config)
        
        # Train
        logger.info(f"Starting PPO training for {config.timesteps:,} timesteps...")
        stats = trainer.train(
            scenarios=scenarios,
            timesteps=config.timesteps,
            verbose=config.verbose,
        )
        
        # Save model
        model_path = Path(config.model_dir) / "ppo_optimal_exit.zip"
        trainer.save()
        logger.info(f"✓ Exit timing model saved to: {model_path}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Exit timing training failed: {e}")
        logger.debug(traceback.format_exc())
        return None


# ============================================================================
# Result Reporting
# ============================================================================

def report_saved_models(config: TrainingConfig) -> None:
    """
    Report all saved RL models.
    
    Args:
        config: Training configuration
    """
    models_dir = Path(config.model_dir)
    
    if not models_dir.exists():
        logger.warning(f"Models directory does not exist: {models_dir}")
        return
    
    # Find all RL models
    rl_models = (
        list(models_dir.glob("*sac*")) +
        list(models_dir.glob("*ppo*")) +
        list(models_dir.glob("*rl*"))
    )
    
    if rl_models:
        logger.info("\nSaved RL models:")
        for model_path in sorted(rl_models):
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"  • {model_path.name} ({size_mb:.2f} MB)")
    else:
        logger.info("\nNo RL models found")


def print_usage_instructions() -> None:
    """Print usage instructions for trained models."""
    logger.info("\nUsage:")
    logger.info("  python main.py buddy --use-rl-gates --use-rl-exits -I EUR_USD")


# ============================================================================
# Main Function
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Quick RL training script for gate thresholds and exit timing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Training timesteps per model"
    )
    
    parser.add_argument(
        "--gates-only",
        action="store_true",
        help="Train only gate thresholds"
    )
    
    parser.add_argument(
        "--exits-only",
        action="store_true",
        help="Train only exit timing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of synthetic samples to generate"
    )
    
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=500,
        help="Number of trade scenarios to generate"
    )
    
    return parser.parse_args()


def determine_training_mode(args: argparse.Namespace) -> TrainingMode:
    """
    Determine training mode from arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Training mode enum
    """
    if args.gates_only:
        return TrainingMode.GATES_ONLY
    elif args.exits_only:
        return TrainingMode.EXITS_ONLY
    else:
        return TrainingMode.BOTH


def main() -> int:
    """
    Main entry point for RL training script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    global logger
    logger = setup_logging(args.verbose)
    
    logger.info("=" * 60)
    logger.info("RL TRAINING SCRIPT STARTED")
    logger.info("=" * 60)
    
    # Determine training mode
    training_mode = determine_training_mode(args)
    logger.info(f"Training mode: {training_mode.value}")
    
    # Create configurations
    training_config = TrainingConfig(
        timesteps=args.timesteps,
        verbose=args.verbose,
    )
    
    data_config = SyntheticDataConfig(
        n_samples=args.n_samples,
        n_scenarios=args.n_scenarios,
    )
    
    # Ensure directories exist
    try:
        ensure_directories_exist(training_config)
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        return 1
    
    # Track training success
    training_results = {
        "gates": False,
        "exits": False,
    }
    
    # Train gate thresholds
    if training_mode != TrainingMode.EXITS_ONLY:
        logger.info("\nStarting gate threshold training...")
        try:
            result = train_gate_thresholds(training_config, data_config)
            training_results["gates"] = result is not None
        except Exception as e:
            logger.error(f"Gate threshold training encountered an error: {e}")
            logger.debug(traceback.format_exc())
            training_results["gates"] = False
    
    # Train exit timing
    if training_mode != TrainingMode.GATES_ONLY:
        logger.info("\nStarting exit timing training...")
        try:
            result = train_exit_timing(training_config, data_config)
            training_results["exits"] = result is not None
        except Exception as e:
            logger.error(f"Exit timing training encountered an error: {e}")
            logger.debug(traceback.format_exc())
            training_results["exits"] = False
    
    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("RL TRAINING COMPLETE")
    logger.info("=" * 60)
    
    # Report training status
    logger.info("\nTraining Results:")
    logger.info(f"  Gate Thresholds: {'✓ SUCCESS' if training_results['gates'] else '✗ FAILED'}")
    logger.info(f"  Exit Timing: {'✓ SUCCESS' if training_results['exits'] else '✗ FAILED'}")
    
    # Report saved models
    report_saved_models(training_config)
    
    # Print usage instructions
    if any(training_results.values()):
        print_usage_instructions()
    
    # Return exit code
    return 0 if any(training_results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
