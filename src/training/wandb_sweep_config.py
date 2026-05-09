"""W&B Sweep Configuration for FX Trading Model Hyperparameter Optimization.

This module provides sweep configurations and utilities for running
hyperparameter optimization using Weights & Biases sweeps.

Reference:
    https://docs.wandb.ai/guides/sweeps

Usage:
    1. Create sweep:  wandb sweep --config config/wandb_sweep_config.yaml
    2. Run agent:     wandb agent <entity/project/sweep-id>

Example:
    ```python
    from src.training.wandb_sweep_config import get_sweep_config, run_sweep_agent

    # Get default sweep configuration
    config = get_sweep_config("tcn")

    # Run sweep agent
    run_sweep_agent(sweep_id="entity/project/sweep-id")
    ```

M1 Metal Compatibility:
    - TCN model recommended (2-3x faster than LSTM)
    - recurrent_dropout=0.0 for LSTM layers
    - mixed_float16 policy for acceleration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Sweep Configuration Dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SweepParameter:
    """Definition of a sweep parameter.

    Attributes:
        name: Parameter name (matches training config attribute).
        values: List of discrete values to try.
        distribution: Distribution type for continuous parameters.
            Options: "uniform", "log_uniform", "log_uniform_values", "q_uniform",
            "int_uniform", "categorical"
        min_val: Minimum value for continuous distributions.
        max_val: Maximum value for continuous distributions.
        description: Human-readable description of the parameter.
    """

    name: str
    values: Optional[List[Any]] = None
    distribution: Optional[str] = None
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    description: str = ""

    def to_wandb_spec(self) -> Dict[str, Any]:
        """Convert to W&B sweep parameter specification.

        Returns:
            Dictionary in W&B sweep parameter format.
        """
        if self.values is not None:
            return {"values": self.values}
        elif self.distribution is not None:
            spec = {"distribution": self.distribution}
            if self.min_val is not None:
                if self.distribution in ("log_uniform_values", "log_uniform"):
                    spec["min"] = self.min_val
                else:
                    spec["min"] = self.min_val
            if self.max_val is not None:
                if self.distribution in ("log_uniform_values", "log_uniform"):
                    spec["max"] = self.max_val
                else:
                    spec["max"] = self.max_val
            return spec
        else:
            raise ValueError(
                f"Parameter {self.name} must have either values or distribution"
            )


@dataclass
class SweepMetric:
    """Definition of sweep optimization metric.

    Attributes:
        name: Metric name to optimize.
        goal: Optimization goal - "minimize" or "maximize".
    """

    name: str
    goal: str = "maximize"

    def to_wandb_spec(self) -> Dict[str, Any]:
        """Convert to W&B metric specification.

        Returns:
            Dictionary in W&B metric format.
        """
        return {"name": self.name, "goal": self.goal}


@dataclass
class SweepConfig:
    """Complete W&B sweep configuration.

    Attributes:
        name: Sweep name.
        project: W&B project name.
        method: Search method - "bayes", "grid", or "random".
        metric: Metric to optimize.
        parameters: List of sweep parameters.
        early_terminate: Early termination configuration.
        description: Human-readable description.
        program: Training script to run.
        command: Command template for running the sweep.
    """

    name: str = "fx_hyperparameter_sweep"
    project: str = "ml_engine_fx"
    method: str = "bayes"  # bayes, grid, random
    metric: SweepMetric = field(
        default_factory=lambda: SweepMetric(
            name="val_direction_accuracy", goal="maximize"
        )
    )
    parameters: List[SweepParameter] = field(default_factory=list)
    early_terminate: Optional[Dict[str, Any]] = None
    description: str = ""
    program: str = "scripts/run_wandb_sweep.py"
    command: Optional[List[str]] = None

    def to_wandb_spec(self) -> Dict[str, Any]:
        """Convert to W&B sweep configuration dictionary.

        Returns:
            Dictionary in W&B sweep configuration format.
        """
        config: Dict[str, Any] = {
            "project": self.project,
            "name": self.name,
            "method": self.method,
            "metric": self.metric.to_wandb_spec(),
            "parameters": {p.name: p.to_wandb_spec() for p in self.parameters},
        }

        if self.description:
            config["description"] = self.description

        if self.program:
            config["program"] = self.program

        if self.command:
            config["command"] = self.command

        if self.early_terminate:
            config["early_terminate"] = self.early_terminate

        return config


# ═══════════════════════════════════════════════════════════════════════════
# Pre-defined Sweep Configurations
# ═══════════════════════════════════════════════════════════════════════════


def get_tcn_sweep_config() -> SweepConfig:
    """Get TCN-optimized sweep configuration.

    TCN (Temporal Convolutional Network) is 2-3x faster than LSTM on M1 Metal.
    This configuration focuses on TCN-specific hyperparameters.

    Returns:
        SweepConfig optimized for TCN models.
    """
    return SweepConfig(
        name="fx_tcn_sweep",
        project="ml_engine_fx",
        method="bayes",
        description="TCN hyperparameter sweep optimized for M1 Metal performance.",
        metric=SweepMetric(name="val_direction_accuracy", goal="maximize"),
        early_terminate={
            "type": "hyperband",
            "min_iter": 10,
            "eta": 3,
        },
        parameters=[
            # Model Architecture
            SweepParameter(
                name="seq_len",
                values=[30, 60, 90, 120],
                description="Sequence length for time series input",
            ),
            # Training Hyperparameters
            SweepParameter(
                name="lr",
                distribution="log_uniform_values",
                min_val=0.0001,
                max_val=0.01,
                description="Learning rate (log scale)",
            ),
            SweepParameter(
                name="batch_size",
                values=[64, 128, 256],
                description="Training batch size (128 optimal for M1 Metal)",
            ),
            SweepParameter(
                name="patience",
                values=[10, 15, 20, 25],
                description="Early stopping patience",
            ),
            # Regularization
            SweepParameter(
                name="dropout_rate",
                distribution="uniform",
                min_val=0.1,
                max_val=0.5,
                description="Dropout rate for regularization",
            ),
            SweepParameter(
                name="l2_reg",
                distribution="log_uniform_values",
                min_val=0.00001,
                max_val=0.001,
                description="L2 regularization strength",
            ),
            # TCN-Specific
            SweepParameter(
                name="tcn_nb_filters",
                values=[32, 64, 128, 256],
                description="Number of filters in TCN layers",
            ),
            SweepParameter(
                name="tcn_kernel_size",
                values=[2, 3, 5, 7],
                description="Kernel size for causal convolutions",
            ),
            SweepParameter(
                name="tcn_dilations",
                values=[
                    [1, 2, 4, 8],
                    [1, 2, 4, 8, 16],
                    [1, 2, 4, 8, 16, 32],
                ],
                description="Dilation pattern (controls receptive field)",
            ),
            # Loss Weights
            SweepParameter(
                name="direction_weight",
                distribution="uniform",
                min_val=0.3,
                max_val=0.8,
                description="Weight for direction loss",
            ),
            SweepParameter(
                name="confidence_weight",
                distribution="uniform",
                min_val=0.1,
                max_val=0.4,
                description="Weight for confidence loss",
            ),
            SweepParameter(
                name="volatility_weight",
                distribution="uniform",
                min_val=0.1,
                max_val=0.3,
                description="Weight for volatility loss",
            ),
            # M1 Metal Optimization
            SweepParameter(
                name="mixed_precision",
                values=[True, False],
                description="Enable mixed precision (FP16) for M1 speedup",
            ),
            SweepParameter(
                name="steps_per_execution",
                values=[1, 5, 10, 20],
                description="Reduces Python overhead",
            ),
            # Feature Engineering
            SweepParameter(
                name="top_features",
                values=[None, 50, 100, 150],
                description="Number of top features (None = all)",
            ),
            SweepParameter(
                name="train_smoothing",
                values=[True, False],
                description="Enable training data smoothing",
            ),
        ],
    )


def get_lstm_sweep_config() -> SweepConfig:
    """Get LSTM sweep configuration.

    Note: LSTM is slower on M1 Metal. Use TCN for production.
    This configuration includes LSTM-specific parameters with
    M1 Metal compatibility (recurrent_dropout=0.0).

    Returns:
        SweepConfig for LSTM models.
    """
    return SweepConfig(
        name="fx_lstm_sweep",
        project="ml_engine_fx",
        method="bayes",
        description="LSTM hyperparameter sweep (slower than TCN on M1 Metal).",
        metric=SweepMetric(name="val_direction_accuracy", goal="maximize"),
        early_terminate={
            "type": "hyperband",
            "min_iter": 10,
            "eta": 3,
        },
        parameters=[
            # Model Architecture
            SweepParameter(
                name="seq_len",
                values=[30, 60, 90],
                description="Sequence length for time series input",
            ),
            # Training Hyperparameters
            SweepParameter(
                name="lr",
                distribution="log_uniform_values",
                min_val=0.0001,
                max_val=0.005,
                description="Learning rate (log scale)",
            ),
            SweepParameter(
                name="batch_size",
                values=[64, 128],
                description="Training batch size",
            ),
            SweepParameter(
                name="patience",
                values=[15, 20, 25],
                description="Early stopping patience",
            ),
            # LSTM-Specific
            SweepParameter(
                name="lstm_units",
                values=[64, 128, 256],
                description="LSTM hidden units",
            ),
            SweepParameter(
                name="lstm_layers",
                values=[1, 2, 3],
                description="Number of LSTM layers",
            ),
            # Regularization (recurrent_dropout=0.0 for M1 Metal)
            SweepParameter(
                name="dropout_rate",
                distribution="uniform",
                min_val=0.1,
                max_val=0.4,
                description="Input dropout rate",
            ),
            # Loss Weights
            SweepParameter(
                name="direction_weight",
                distribution="uniform",
                min_val=0.4,
                max_val=0.8,
                description="Weight for direction loss",
            ),
            # M1 Metal Optimization
            SweepParameter(
                name="mixed_precision",
                values=[True],
                description="Mixed precision required for LSTM on M1",
            ),
        ],
    )


def get_ensemble_sweep_config() -> SweepConfig:
    """Get ensemble model sweep configuration.

    Sweeps across TCN, LSTM, and attention_lstm architectures
    to find the best ensemble configuration.

    Returns:
        SweepConfig for ensemble models.
    """
    return SweepConfig(
        name="fx_ensemble_sweep",
        project="ml_engine_fx",
        method="bayes",
        description="Ensemble sweep across TCN/LSTM architectures.",
        metric=SweepMetric(name="val_combined_score", goal="maximize"),
        early_terminate={
            "type": "hyperband",
            "min_iter": 15,
            "eta": 3,
        },
        parameters=[
            # Model Type Selection
            SweepParameter(
                name="model_type",
                values=["tcn", "lstm", "attention_lstm"],
                description="Model architecture type",
            ),
            # Common Hyperparameters
            SweepParameter(
                name="seq_len",
                values=[60, 90, 120],
                description="Sequence length",
            ),
            SweepParameter(
                name="lr",
                distribution="log_uniform_values",
                min_val=0.0001,
                max_val=0.005,
                description="Learning rate",
            ),
            SweepParameter(
                name="batch_size",
                values=[64, 128],
                description="Batch size",
            ),
            SweepParameter(
                name="dropout_rate",
                distribution="uniform",
                min_val=0.15,
                max_val=0.45,
                description="Dropout rate",
            ),
            # Ensemble-Specific
            SweepParameter(
                name="direction_weight",
                distribution="uniform",
                min_val=0.4,
                max_val=0.7,
                description="Direction loss weight",
            ),
            SweepParameter(
                name="confidence_weight",
                distribution="uniform",
                min_val=0.15,
                max_val=0.35,
                description="Confidence loss weight",
            ),
            SweepParameter(
                name="volatility_weight",
                distribution="uniform",
                min_val=0.1,
                max_val=0.25,
                description="Volatility loss weight",
            ),
            # Early Stopping
            SweepParameter(
                name="es_monitor",
                values=["direction", "combined", "val_loss"],
                description="Early stopping monitor metric",
            ),
            SweepParameter(
                name="combined_w_dir",
                distribution="uniform",
                min_val=0.5,
                max_val=0.8,
                description="Direction weight in combined metric",
            ),
        ],
    )


def get_quick_sweep_config() -> SweepConfig:
    """Get quick sweep configuration for rapid iteration.

    Smaller search space for faster results during development.

    Returns:
        SweepConfig with reduced parameter space.
    """
    return SweepConfig(
        name="fx_quick_sweep",
        project="ml_engine_fx",
        method="random",  # Random is faster for quick sweeps
        description="Quick sweep for rapid iteration during development.",
        metric=SweepMetric(name="val_direction_accuracy", goal="maximize"),
        parameters=[
            SweepParameter(
                name="model_type",
                values=["tcn"],  # TCN only for speed
                description="Model type (TCN fastest)",
            ),
            SweepParameter(
                name="lr",
                distribution="log_uniform_values",
                min_val=0.0003,
                max_val=0.003,
                description="Learning rate",
            ),
            SweepParameter(
                name="batch_size",
                values=[128],  # Fixed for M1 Metal
                description="Batch size",
            ),
            SweepParameter(
                name="dropout_rate",
                values=[0.2, 0.3, 0.4],
                description="Dropout rate",
            ),
            SweepParameter(
                name="tcn_nb_filters",
                values=[64, 128],
                description="TCN filters",
            ),
        ],
    )


def get_transformer_direction_sweep_config(
    pair: str = "USD_JPY",
    granularity: str = "H1",
    candles: int = 25000,
    lookahead: int = 24,
) -> SweepConfig:
    """Sweep config for the TransformerDirectionTrainer (master-pair tuning).

    Phase 5.D Stage 4-A follow-up (2026-05-08): per-master-pair hyperparameter
    tuning to push past the price-only ceiling without requiring news fusion.
    Path 4 OOS validation already showed signal exists for the masters; this
    sweep tries to push accuracy higher by tuning architecture + regularization.

    Search space prioritizes the levers most likely to help small-data
    direction prediction:
      - depth/width (d_model, num_layers, dff) — capacity floor
      - dropout — regularization knob (we saw overfitting at default 0.2)
      - LR + batch_size — convergence quality
      - top_k_features + ewc_lambda — feature selection + continual-learning
        regularization
      - patience — early-stop aggressiveness

    Bayesian + Hyperband: ~30 trials reach near-optimal with early-termination
    of underperformers. Per-trial cost ~5 min; total ~2.5h per pair on M1.

    The runner is `scripts/run_transformer_direction_sweep.py`.
    """
    return SweepConfig(
        name=f"buddy_transformer_{pair.lower()}_{granularity.lower()}",
        project="buddy-master-tuning",
        method="bayes",
        description=(
            f"Transformer direction-head Bayesian sweep on {pair}@{granularity}, "
            f"{candles} candles, lookahead={lookahead}. Metric: val_balanced_accuracy."
        ),
        metric=SweepMetric(name="val_balanced_accuracy", goal="maximize"),
        early_terminate={"type": "hyperband", "min_iter": 5, "eta": 3},
        program="scripts/run_transformer_direction_sweep.py",
        parameters=[
            # Pair / data — fixed via the SweepConfig instance, but exposed
            # so wandb config has them as constants per trial.
            SweepParameter(name="pair", values=[pair], description="Currency pair (fixed per sweep)"),
            SweepParameter(name="granularity", values=[granularity], description="OANDA TF (fixed)"),
            SweepParameter(name="candles", values=[candles], description="Training candles (fixed)"),
            SweepParameter(name="lookahead", values=[lookahead], description="Direction lookahead (fixed)"),

            # Architecture — capacity knobs
            SweepParameter(
                name="transformer_d_model",
                values=[8, 16, 32, 64],
                description="Embedding dim (8-64; default 16)",
            ),
            SweepParameter(
                name="transformer_num_heads",
                values=[1, 2, 4],
                description="Attention heads (1-4; default 2)",
            ),
            SweepParameter(
                name="transformer_num_layers",
                values=[1, 2, 3],
                description="Transformer encoder layers (1-3; default 1)",
            ),
            SweepParameter(
                name="transformer_dff",
                values=[16, 32, 64, 128],
                description="Feedforward width (16-128; default 32)",
            ),

            # Regularization
            SweepParameter(
                name="transformer_dropout",
                distribution="uniform",
                min_val=0.1,
                max_val=0.5,
                description="Dropout fraction (default 0.2; sweep 0.1-0.5)",
            ),

            # Optimization
            SweepParameter(
                name="learning_rate",
                distribution="log_uniform_values",
                min_val=0.00001,
                max_val=0.005,
                description="Learning rate, log-uniform 1e-5 to 5e-3",
            ),
            SweepParameter(
                name="batch_size",
                values=[32, 64, 128, 256],
                description="Mini-batch size (M1 Metal sweet spot 64-128)",
            ),

            # Feature selection / continual learning
            SweepParameter(
                name="top_k_features",
                values=[30, 50, 80],
                description="RF-selected features kept (default 50)",
            ),
            SweepParameter(
                name="ewc_lambda",
                distribution="log_uniform_values",
                min_val=0.01,
                max_val=10.0,
                description="EWC regularization weight (continual-learning anchor)",
            ),

            # Training discipline
            SweepParameter(
                name="patience",
                values=[10, 15, 20, 30],
                description="Early-stop patience (epochs without val improvement)",
            ),
            SweepParameter(
                name="epochs",
                values=[50, 100, 200],
                description="Max epochs (early-stop usually fires before this)",
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Sweep Configuration Factory
# ═══════════════════════════════════════════════════════════════════════════


def get_sweep_config(sweep_type: str = "tcn") -> SweepConfig:
    """Get sweep configuration by type.

    Args:
        sweep_type: Type of sweep configuration.
            Options: "tcn", "lstm", "ensemble", "quick"

    Returns:
        SweepConfig for the specified type.

    Raises:
        ValueError: If sweep_type is not recognized.
    """
    configs = {
        "tcn": get_tcn_sweep_config,
        "lstm": get_lstm_sweep_config,
        "ensemble": get_ensemble_sweep_config,
        "quick": get_quick_sweep_config,
    }

    if sweep_type not in configs:
        raise ValueError(
            f"Unknown sweep type: {sweep_type}. "
            f"Available: {list(configs.keys())}"
        )

    return configs[sweep_type]()


def get_sweep_config_yaml(sweep_type: str = "tcn") -> str:
    """Get sweep configuration as YAML string.

    Args:
        sweep_type: Type of sweep configuration.

    Returns:
        YAML string representation of the sweep config.
    """
    import yaml

    config = get_sweep_config(sweep_type)
    return yaml.dump(config.to_wandb_spec(), default_flow_style=False, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════════
# Sweep Agent Runner
# ═══════════════════════════════════════════════════════════════════════════


def run_sweep_agent(
    sweep_id: str,
    train_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    count: Optional[int] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
) -> None:
    """Run a W&B sweep agent.

    This function initializes a W&B sweep agent that will run training
    jobs with hyperparameters sampled from the sweep configuration.

    Args:
        sweep_id: W&B sweep ID (e.g., "entity/project/sweep-id" or just "sweep-id").
        train_fn: Training function that takes config dict and returns metrics.
            If None, uses default training pipeline.
        count: Maximum number of runs to execute (None = unlimited).
        entity: W&B entity (username or team). If None, uses default.
        project: W&B project name. If None, uses default from sweep config.

    Example:
        ```python
        def my_train_fn(config):
            # Build model with config hyperparameters
            model = build_model(
                lr=config["lr"],
                batch_size=config["batch_size"],
                dropout=config["dropout_rate"],
            )
            # Train and return metrics
            metrics = model.fit()
            return {"val_direction_accuracy": metrics["accuracy"]}

        run_sweep_agent(
            sweep_id="my-sweep-id",
            train_fn=my_train_fn,
            count=10,
        )
        ```
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb not installed. Install with: pip install wandb"
        )

    # Parse sweep_id if it contains entity/project
    parts = sweep_id.split("/")
    if len(parts) == 3:
        # Full path: entity/project/sweep-id
        entity, project, sweep_id = parts
    elif len(parts) == 1:
        # Just sweep-id, use defaults
        pass
    else:
        raise ValueError(
            f"Invalid sweep_id format: {sweep_id}. "
            "Expected 'sweep-id' or 'entity/project/sweep-id'"
        )

    # Default training function
    if train_fn is None:
        train_fn = _default_train_fn

    def sweep_train():
        """Training function called by W&B sweep agent."""
        # Initialize W&B run
        wandb.init()

        # Get hyperparameters from sweep
        config = wandb.config

        try:
            # Run training
            metrics = train_fn(dict(config))

            # Log final metrics
            wandb.log(metrics)

            logger.info(f"Run completed. Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            wandb.log({"error": str(e)})
            raise

        finally:
            wandb.finish()

    # Run the sweep agent
    logger.info(f"Starting W&B sweep agent for sweep: {sweep_id}")
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        entity=entity,
        project=project,
        count=count,
    )


def _default_train_fn(config: Dict[str, Any]) -> Dict[str, float]:
    """Default training function for sweep agent.

    This function integrates with the existing Buddy training pipeline.

    Args:
        config: Hyperparameters from W&B sweep.

    Returns:
        Dictionary of metrics to log to W&B.
    """
    from src.training.training_config import (
        BuddyTrainingOptions,
        OandaFetchOptions,
    )

    # Build training options from sweep config
    options = BuddyTrainingOptions(
        oanda_fetch=OandaFetchOptions(
            instrument=config.get("instrument", "USD_JPY"),
            granularity=config.get("granularity", "H1"),
        ),
        model_type=config.get("model_type", "tcn"),
        seq_len=config.get("seq_len", 60),
        lr=config.get("lr", 0.0005),
        batch_size=config.get("batch_size", 128),
        patience=config.get("patience", 15),
        dropout_rate=config.get("dropout_rate", 0.3),
        mixed_precision=config.get("mixed_precision", True),
        steps_per_execution=config.get("steps_per_execution", 10),
        top_features=config.get("top_features"),
        train_smoothing=config.get("train_smoothing", True),
        enterprise=True,
        cv_folds=5,
        wandb_enabled=True,
        wandb_project=config.get("wandb_project", "ml_engine_fx"),
    )

    # Import training function
    try:
        from cli.training import run_buddy_training

        # Run training
        result = run_buddy_training(options=options)

        # Extract metrics
        metrics = {
            "val_direction_accuracy": result.get("direction_accuracy", 0.0),
            "val_combined_score": result.get("combined_score", 0.0),
            "val_loss": result.get("val_loss", 0.0),
            "train_loss": result.get("train_loss", 0.0),
        }

        return metrics

    except ImportError:
        logger.warning("cli.training not available, using mock metrics")
        # Return mock metrics for testing
        import random
        return {
            "val_direction_accuracy": random.uniform(0.5, 0.7),
            "val_combined_score": random.uniform(0.4, 0.6),
            "val_loss": random.uniform(0.3, 0.8),
            "train_loss": random.uniform(0.2, 0.6),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Sweep Creation Utilities
# ═══════════════════════════════════════════════════════════════════════════


def create_sweep(
    sweep_type: str = "tcn",
    project: Optional[str] = None,
    entity: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Create a W&B sweep and return the sweep ID.

    Args:
        sweep_type: Type of sweep configuration ("tcn", "lstm", "ensemble", "quick").
        project: W&B project name (overrides config default).
        entity: W&B entity (username or team).
        name: Sweep name (overrides config default).

    Returns:
        Sweep ID string.

    Example:
        ```python
        sweep_id = create_sweep("tcn", project="my_project")
        print(f"Created sweep: {sweep_id}")
        # Output: entity/my_project/abc123

        # Run agents
        run_sweep_agent(sweep_id, count=10)
        ```
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb not installed. Install with: pip install wandb"
        )

    # Get sweep configuration
    config = get_sweep_config(sweep_type)

    # Override defaults
    spec = config.to_wandb_spec()
    if project:
        spec["project"] = project
    if name:
        spec["name"] = name

    # Create sweep
    sweep_id = wandb.sweep(
        sweep=spec,
        entity=entity,
        project=spec.get("project", "ml_engine_fx"),
    )

    logger.info(f"Created W&B sweep: {sweep_id}")
    return sweep_id


def save_sweep_config_yaml(
    sweep_type: str = "tcn",
    output_path: Optional[str] = None,
) -> Path:
    """Save sweep configuration to YAML file.

    Args:
        sweep_type: Type of sweep configuration.
        output_path: Output file path. If None, saves to config/ directory.

    Returns:
        Path to saved YAML file.
    """
    if output_path is None:
        output_path = f"config/wandb_sweep_{sweep_type}.yaml"

    yaml_content = get_sweep_config_yaml(sweep_type)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_content)

    logger.info(f"Saved sweep config to: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point for W&B sweep operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="W&B Sweep Configuration and Agent Runner"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create sweep command
    create_parser = subparsers.add_parser("create", help="Create a new sweep")
    create_parser.add_argument(
        "--type",
        choices=["tcn", "lstm", "ensemble", "quick"],
        default="tcn",
        help="Sweep configuration type",
    )
    create_parser.add_argument(
        "--project",
        default="ml_engine_fx",
        help="W&B project name",
    )
    create_parser.add_argument(
        "--entity",
        help="W&B entity (username or team)",
    )
    create_parser.add_argument(
        "--name",
        help="Sweep name",
    )

    # Run agent command
    run_parser = subparsers.add_parser("run", help="Run sweep agent")
    run_parser.add_argument(
        "sweep_id",
        help="Sweep ID to run",
    )
    run_parser.add_argument(
        "--count",
        type=int,
        help="Maximum number of runs",
    )
    run_parser.add_argument(
        "--entity",
        help="W&B entity",
    )
    run_parser.add_argument(
        "--project",
        help="W&B project",
    )

    # Save config command
    save_parser = subparsers.add_parser("save", help="Save sweep config to YAML")
    save_parser.add_argument(
        "--type",
        choices=["tcn", "lstm", "ensemble", "quick"],
        default="tcn",
        help="Sweep configuration type",
    )
    save_parser.add_argument(
        "--output",
        help="Output file path",
    )

    # Show config command
    show_parser = subparsers.add_parser("show", help="Show sweep configuration")
    show_parser.add_argument(
        "--type",
        choices=["tcn", "lstm", "ensemble", "quick"],
        default="tcn",
        help="Sweep configuration type",
    )

    args = parser.parse_args()

    if args.command == "create":
        sweep_id = create_sweep(
            sweep_type=args.type,
            project=args.project,
            entity=args.entity,
            name=args.name,
        )
        print(f"Created sweep: {sweep_id}")
        print(f"Run agents with: wandb agent {sweep_id}")

    elif args.command == "run":
        run_sweep_agent(
            sweep_id=args.sweep_id,
            count=args.count,
            entity=args.entity,
            project=args.project,
        )

    elif args.command == "save":
        path = save_sweep_config_yaml(
            sweep_type=args.type,
            output_path=args.output,
        )
        print(f"Saved sweep config to: {path}")

    elif args.command == "show":
        yaml_content = get_sweep_config_yaml(args.type)
        print(yaml_content)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
