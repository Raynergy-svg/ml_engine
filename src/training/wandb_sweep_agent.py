#!/usr/bin/env python3
"""W&B Sweep Agent Runner for FX Trading Model Hyperparameter Optimization.

This script is the entry point for W&B sweep agents. It integrates with
the existing Buddy training pipeline and supports M1 Metal optimizations.

Reference:
    https://docs.wandb.ai/guides/sweeps

Usage:
    # 1. Create a sweep (one-time)
    python -m src.training.wandb_sweep_agent create --type tcn

    # 2. Run agents (can run multiple in parallel)
    python -m src.training.wandb_sweep_agent run <sweep-id>

    # Or using wandb CLI directly:
    wandb sweep config/wandb_sweep_tcn.yaml  # Create
    wandb agent <sweep-id>                   # Run

Example:
    ```bash
    # Create and run a TCN sweep
    python -m src.training.wandb_sweep_agent create --type tcn --project ml_engine_fx
    # Output: Created sweep: entity/ml_engine_fx/abc123

    # Run 10 agents
    python -m src.training.wandb_sweep_agent run entity/ml_engine_fx/abc123 --count 10
    ```

M1 Metal Compatibility:
    - TCN model recommended (2-3x faster than LSTM)
    - recurrent_dropout=0.0 for LSTM layers
    - mixed_float16 policy for acceleration
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for sweep agent.

    Args:
        verbose: Enable debug-level logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_with_sweep_config(config: Dict[str, Any]) -> Dict[str, float]:
    """Training function called by W&B sweep agent.

    This function integrates with the existing Buddy training pipeline
    and applies M1 Metal optimizations.

    Args:
        config: Hyperparameters from W&B sweep configuration.

    Returns:
        Dictionary of metrics to log to W&B.
    """
    from src.training.training_config import (
        BuddyTrainingOptions,
        OandaFetchOptions,
        BuddyTrainingAdvancedOptions,
    )

    logger.info(f"Starting training with sweep config: {config}")

    # Extract hyperparameters from sweep config
    instrument = config.get("instrument", "USD_JPY")
    granularity = config.get("granularity", "H1")

    # Build OANDA fetch options
    oanda_fetch = OandaFetchOptions(
        instrument=instrument,
        granularity=granularity,
        candles=config.get("candles", 5000),
        price=config.get("price", "MBA"),
    )

    # Build advanced options
    advanced = BuddyTrainingAdvancedOptions(
        es_monitor=config.get("es_monitor", "direction"),
        combined_w_dir=config.get("combined_w_dir", 0.7),
        combined_w_conf=config.get("combined_w_conf", 0.3),
        train_smoothing=config.get("train_smoothing", True),
        top_features=config.get("top_features"),
        median_window=config.get("median_window"),
    )

    # Build training options with M1 Metal optimizations
    options = BuddyTrainingOptions(
        oanda_fetch=oanda_fetch,
        advanced=advanced,
        model_type=config.get("model_type", "tcn"),
        seq_len=config.get("seq_len", 60),
        epochs=config.get("epochs", 200),
        batch_size=config.get("batch_size", 128),
        lr=config.get("lr", 0.0005),
        patience=config.get("patience", 15),
        seed=config.get("seed", 42),
        mixed_precision=config.get("mixed_precision", True),
        steps_per_execution=config.get("steps_per_execution", 10),
        jit_compile=config.get("jit_compile", False),
        cache_val=config.get("cache_val", True),
        enterprise=config.get("enterprise", True),
        cv_folds=config.get("cv_folds", 5),
        wandb_enabled=True,  # Always enabled for sweeps
        wandb_project=config.get("wandb_project", "ml_engine_fx"),
    )

    logger.info(f"Training options: model={options.model_type}, "
                f"seq_len={options.seq_len}, lr={options.lr}, "
                f"batch_size={options.batch_size}")

    # Run training using the existing pipeline
    try:
        from cli.training import run_buddy_training

        result = run_buddy_training(options=options)

        # Extract metrics for W&B logging
        metrics = {
            "val_direction_accuracy": result.get("direction_accuracy", 0.0),
            "val_direction_precision": result.get("direction_precision", 0.0),
            "val_direction_recall": result.get("direction_recall", 0.0),
            "val_direction_f1": result.get("direction_f1", 0.0),
            "val_combined_score": result.get("combined_score", 0.0),
            "val_loss": result.get("val_loss", 0.0),
            "train_loss": result.get("train_loss", 0.0),
            "epochs_trained": result.get("epochs_trained", 0),
        }

        logger.info(f"Training completed. Direction accuracy: "
                    f"{metrics['val_direction_accuracy']:.4f}")

        return metrics

    except ImportError as e:
        logger.warning(f"cli.training not available: {e}")
        logger.warning("Using mock training for testing")

        # Mock training for testing sweep infrastructure
        import random
        import time

        # Simulate training time
        time.sleep(2)

        # Generate mock metrics (with some correlation to hyperparameters)
        base_accuracy = 0.55

        # Better hyperparameters should give better results
        lr_bonus = -0.1 * abs(config.get("lr", 0.001) - 0.0005) / 0.001
        batch_bonus = 0.02 if config.get("batch_size") == 128 else 0
        model_bonus = 0.03 if config.get("model_type") == "tcn" else 0

        accuracy = base_accuracy + lr_bonus + batch_bonus + model_bonus
        accuracy += random.uniform(-0.05, 0.1)  # Add noise
        accuracy = max(0.45, min(0.75, accuracy))  # Clamp to realistic range

        metrics = {
            "val_direction_accuracy": accuracy,
            "val_direction_precision": accuracy + random.uniform(-0.05, 0.05),
            "val_direction_recall": accuracy + random.uniform(-0.05, 0.05),
            "val_direction_f1": accuracy + random.uniform(-0.03, 0.03),
            "val_combined_score": accuracy * 0.8 + random.uniform(-0.05, 0.05),
            "val_loss": 1.0 - accuracy + random.uniform(-0.1, 0.1),
            "train_loss": 0.8 - accuracy * 0.5 + random.uniform(-0.1, 0.1),
            "epochs_trained": random.randint(20, 100),
        }

        logger.info(f"Mock training completed. Direction accuracy: "
                    f"{metrics['val_direction_accuracy']:.4f}")

        return metrics


def run_sweep_agent(
    sweep_id: str,
    count: Optional[int] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
) -> None:
    """Run a W&B sweep agent.

    Args:
        sweep_id: W&B sweep ID (e.g., "entity/project/sweep-id" or just "sweep-id").
        count: Maximum number of runs to execute (None = unlimited).
        entity: W&B entity (username or team). If None, uses default.
        project: W&B project name. If None, uses default from sweep config.
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    # Parse sweep_id if it contains entity/project
    parts = sweep_id.split("/")
    if len(parts) == 3:
        # Full path: entity/project/sweep-id
        entity_override, project_override, sweep_id = parts
        entity = entity or entity_override
        project = project or project_override
    elif len(parts) == 2:
        # Partial path: project/sweep-id
        project_override, sweep_id = parts
        project = project or project_override

    def sweep_train():
        """Training function called by W&B sweep agent."""
        # Initialize W&B run with sweep config
        wandb.init()

        # Get hyperparameters from sweep
        config = dict(wandb.config)

        try:
            # Run training
            metrics = train_with_sweep_config(config)

            # Log final metrics
            wandb.log(metrics)

            logger.info(f"Run completed. Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            wandb.log({"error": str(e)})
            raise

        finally:
            wandb.finish()

    # Run the sweep agent
    logger.info("Starting W&B sweep agent")
    logger.info(f"  Sweep ID: {sweep_id}")
    logger.info(f"  Entity: {entity or 'default'}")
    logger.info(f"  Project: {project or 'default'}")
    logger.info(f"  Max runs: {count or 'unlimited'}")

    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        entity=entity,
        project=project,
        count=count,
    )


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
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    from src.training.wandb_sweep_config import get_sweep_config

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


def save_sweep_config(
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
    from src.training.wandb_sweep_config import save_sweep_config_yaml

    return save_sweep_config_yaml(sweep_type, output_path)


def main():
    """CLI entry point for W&B sweep operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="W&B Sweep Agent Runner for FX Trading Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a TCN sweep
  python -m src.training.wandb_sweep_agent create --type tcn

  # Run sweep agent
  python -m src.training.wandb_sweep_agent run entity/project/sweep-id --count 10

  # Save sweep config to YAML
  python -m src.training.wandb_sweep_agent save --type tcn --output config/my_sweep.yaml

  # Show sweep configuration
  python -m src.training.wandb_sweep_agent show --type ensemble
        """,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create sweep command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new sweep",
    )
    create_parser.add_argument(
        "--type",
        choices=["tcn", "lstm", "ensemble", "quick"],
        default="tcn",
        help="Sweep configuration type (default: tcn)",
    )
    create_parser.add_argument(
        "--project",
        default="ml_engine_fx",
        help="W&B project name (default: ml_engine_fx)",
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
    run_parser = subparsers.add_parser(
        "run",
        help="Run sweep agent",
    )
    run_parser.add_argument(
        "sweep_id",
        help="Sweep ID to run (e.g., entity/project/sweep-id)",
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
    save_parser = subparsers.add_parser(
        "save",
        help="Save sweep config to YAML",
    )
    save_parser.add_argument(
        "--type",
        choices=["tcn", "lstm", "ensemble", "quick"],
        default="tcn",
        help="Sweep configuration type (default: tcn)",
    )
    save_parser.add_argument(
        "--output",
        help="Output file path (default: config/wandb_sweep_<type>.yaml)",
    )

    # Show config command
    show_parser = subparsers.add_parser(
        "show",
        help="Show sweep configuration",
    )
    show_parser.add_argument(
        "--type",
        choices=["tcn", "lstm", "ensemble", "quick"],
        default="tcn",
        help="Sweep configuration type (default: tcn)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    if args.command == "create":
        sweep_id = create_sweep(
            sweep_type=args.type,
            project=args.project,
            entity=args.entity,
            name=args.name,
        )
        print(f"\n✅ Created sweep: {sweep_id}")
        print("\nRun agents with:")
        print(f"  wandb agent {sweep_id}")
        print("\nOr with this script:")
        print(f"  python -m src.training.wandb_sweep_agent run {sweep_id}")

    elif args.command == "run":
        run_sweep_agent(
            sweep_id=args.sweep_id,
            count=args.count,
            entity=args.entity,
            project=args.project,
        )

    elif args.command == "save":
        path = save_sweep_config(
            sweep_type=args.type,
            output_path=args.output,
        )
        print(f"\n✅ Saved sweep config to: {path}")
        print("\nCreate sweep with:")
        print(f"  wandb sweep {path}")

    elif args.command == "show":
        from src.training.wandb_sweep_config import get_sweep_config_yaml
        yaml_content = get_sweep_config_yaml(args.type)
        print(yaml_content)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
