#!/usr/bin/env python3
"""Legacy/deprecated CLI command implementations for ML Engine Trading Bot.

These functions are deprecated and maintained for backwards compatibility only.
They will be removed in a future version.
"""
from __future__ import annotations

import logging
import time
import warnings
from functools import wraps
from typing import Any

from rich.console import Console
from rich.live import Live

from src.utils import load_config
from cli.fx_trading import generate_dashboard


console = Console()


def _deprecated(func):
    """Decorator to mark functions as deprecated and emit warnings."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future version. "
            "Use the 'buddy' or 'train-buddy' commands instead.",
            DeprecationWarning,
            stacklevel=2
        )
        console.print(f"[yellow]⚠ DEPRECATED: {func.__name__} is deprecated and will be removed in a future version.[/yellow]")
        return func(*args, **kwargs)
    return wrapper


@_deprecated
def train_model(config_path: str) -> None:
    """Run model training with live progress dashboard.
    
    DEPRECATED: Use train_buddy instead.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        
        # Validate training configuration
        if "training" not in config:
            raise ValueError("Configuration missing 'training' section")
        if "epochs" not in config["training"]:
            raise ValueError("Configuration missing 'training.epochs'")
        
        console.print("[bold blue]Initializing ML Engine...[/bold blue]")
        
        # Load training data
        from data_loader import MarketDataLoader
        data_loader = MarketDataLoader(config)
        
        console.print("[bold blue]Loading market data...[/bold blue]")
        
        # Get data directory from config
        data_dir = config.get("data", {}).get("data_dir", "market_data/")
        
        # Load multiple tickers from the data directory
        data_dict = data_loader.load_multiple_tickers(data_dir)
        
        if not data_dict:
            raise ValueError(f"No data files found in {data_dir}")
        
        # Combine all ticker data
        df = data_loader.combine_ticker_data(data_dict, method="concat")
        console.print(f"[cyan]Loaded {len(data_dict)} tickers, {len(df)} total rows[/cyan]")
        
        # Preprocess the data
        console.print("[bold blue]Preprocessing data...[/bold blue]")
        x_train, y_train, x_val, y_val, _, _ = data_loader.preprocess(
            df,
            add_features=True,
            scaler_type="standard",
            sequence_length=config.get("data", {}).get("sequence_length", 60),
            test_size=0.2,
        )
        
        console.print(f"[bold green]Data prepared: {len(x_train)} training samples, {len(x_val)} validation samples[/bold green]")
        
        # Update config with correct input size based on actual features
        input_size = x_train.shape[-1]  # Get feature dimension
        if "model" not in config:
            config["model"] = {}
        config["model"]["input_size"] = input_size
        console.print(f"[cyan]Model input size set to: {input_size}[/cyan]")

        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)
        
        # Get epochs from config (not nested in 'training')
        epochs = config.get("epochs", 100)
        console.print(f"[bold green]Training for {epochs} epochs[/bold green]")
        
        # Train the model and get results
        console.print("[bold green]Starting training...[/bold green]")
        result = engine.train(x_train, y_train, x_val, y_val, epochs=epochs)
        
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print(f"[cyan]Total epochs trained: {result.get('total_epochs', epochs)}[/cyan]")
        console.print(f"[cyan]Best validation loss: {result['best_val_loss']:.6f}[/cyan]")
        console.print(f"[cyan]Resumed from checkpoint: {result.get('resumed', False)}[/cyan]")
                
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logging.error(f"Unexpected error in train_model: {e}", exc_info=True)
        raise


@_deprecated
def evaluate_model(config_path: str) -> None:
    """Run model evaluation with results dashboard.
    
    DEPRECATED: Use buddy inference commands instead.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        RuntimeError: If evaluation fails
    """
    try:
        console.print("[bold blue]Loading configuration and model...[/bold blue]")
        config = load_config(config_path)
        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)

        api_logs = [f"[blue]{time.strftime('%H:%M:%S')}[/blue] Starting evaluation"]
        
        with Live(generate_dashboard(0, 10, 200, 0.0, 0.0, 0.0, api_logs, config)) as live:
            try:
                console.print("[bold green]Evaluating model...[/bold green]")
                metrics = engine.evaluate_model()
                
                # Display evaluation results
                api_logs.append("[green]Evaluation completed[/green]")
                if metrics:
                    for key, value in metrics.items():
                        api_logs.append(f"[cyan]{key}: {value:.4f}[/cyan]")
                
                live.update(generate_dashboard(10, 10, 0, 0.0, 0.0, 0.0, api_logs, config))
                console.print("[bold green]Evaluation completed successfully![/bold green]")
                
            except Exception as e:
                console.print(f"[red]Evaluation error: {e}[/red]")
                logging.error(f"Evaluation failed: {e}", exc_info=True)
                raise RuntimeError(f"Evaluation failed: {e}")
                
    except Exception as e:
        console.print(f"[red]Failed to evaluate model: {e}[/red]")
        logging.error(f"Model evaluation error: {e}", exc_info=True)
        raise


@_deprecated
def visualize_dashboard(config_path: str) -> None:
    """Display dashboard visualizations using the visualizer module.
    
    DEPRECATED: Use buddy commands instead.
    """
    import visualizer

    config = load_config(config_path)
    console.print("[bold blue]Launching visualizations...[/bold blue]")
    visualizer.display_dashboard(
        config
    )  # Assumes implementation exists in visualizer.py


@_deprecated
def openai_tune(config_path: str) -> None:
    """Execute OpenAI-based auto-tuning using the openai_integration module.
    
    DEPRECATED: This functionality is no longer actively maintained.
    """
    try:
        import openai_integration
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "openai_integration requires the `openai` package; install it to use openai-tune."
        ) from e

    current_config = load_config(config_path)

    # Only set credentials when we actually use OpenAI integration.
    openai_integration.set_openai_credentials(current_config)

    # Gather current metrics from the engine
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(current_config)
    train_loss = (
        engine.train_losses[-1]
        if hasattr(engine, "train_losses") and engine.train_losses
        else None
    )
    val_loss = (
        engine.val_losses[-1]
        if hasattr(engine, "val_losses") and engine.val_losses
        else None
    )
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_config.get("model", {}).get("learning_rate", 0.001),
        "epochs_completed": len(engine.train_losses)
        if hasattr(engine, "train_losses")
        else 0,
        "device": current_config.get("hardware", {}).get("device", "cpu"),
        "batch_size": current_config.get("model", {}).get("batch_size", 32),
    }

    # Call the integration function with metrics and config in proper order
    result = openai_integration.query_for_auto_configuration(metrics, current_config)

    if result:
        _ = openai_integration.report_config_changes(current_config, result, config_path=config_path)
        console.print(
            f"[bold green]Configuration updated successfully and written to {config_path}[/bold green]"
        )
        console.print(f"[cyan]Updates Applied:[/cyan] {result}")
    else:
        console.print(
            "[bold red]Failed to get configuration updates from OpenAI[/bold red]"
        )


@_deprecated
def predict_price(config_path: str) -> None:
    """Use ML engine to predict prices given a dataset.
    
    DEPRECATED: Use buddy inference commands instead.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        console.print("[bold blue]Loading model for prediction...[/bold blue]")
        config = load_config(config_path)
        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)
        
        console.print("[bold green]Generating prediction...[/bold green]")
        prediction = engine.predict_price()
        
        console.print(f"[bold yellow]Predicted Price: {prediction}[/bold yellow]")
        logging.info(f"Price prediction completed: {prediction}")
        
    except AttributeError as e:
        console.print(f"[red]Method not available: {e}[/red]")
        logging.error(f"predict_price method error: {e}")
        raise RuntimeError(f"Prediction method not available: {e}")
    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        logging.error(f"Price prediction error: {e}", exc_info=True)
        raise


@_deprecated
def realtime_loop(config_path: str) -> None:
    """Run the ML engine in a real-time loop for continuous inference.
    
    DEPRECATED: Use buddy scan or fx-paper-trade instead.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.run_realtime_loop()
    console.print("[bold yellow]Realtime Loop command executed[/bold yellow]")


@_deprecated
def tune_model(config_path: str) -> None:
    """Perform advanced hyperparameter tuning using ML engine methods.
    
    DEPRECATED: Manual hyperparameter tuning is recommended.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.tune_hyperparameters()
    console.print("[bold yellow]Tune Model command executed[/bold yellow]")


@_deprecated
def profile_pipeline(config_path: str) -> None:
    """Profile the ML pipeline for bottlenecks.
    
    DEPRECATED: Use Python profiling tools directly.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.profile_pipeline()
    console.print("[bold yellow]Profile Pipeline command executed[/bold yellow]")


@_deprecated
def run_ai_assistant(config_path: str) -> None:
    """Legacy AI assistant entrypoint.
    
    DEPRECATED: AI assistant entrypoint removed. Use `python main.py buddy` / `python main.py train-buddy`.
    """
    _ = config_path
    raise RuntimeError("Retired: AI assistant entrypoint removed. Use `python main.py buddy` / `python main.py train-buddy`.")


@_deprecated
def train_unified(config_path: str, csv_path: str | None = None, *, checkpoint_path: str | None = None) -> None:
    """Legacy alias: train Buddy TF model from main.py only.
    
    DEPRECATED: Use train_buddy instead.
    """
    _ = checkpoint_path
    # Import here to avoid circular imports
    from cli.commands import train_buddy
    train_buddy(config_path, csv_path)


@_deprecated
def train_oanda_unified(
    config_path: str,
    *,
    instruments: str,
    granularity: str,
    candles: int,
    checkpoint_path: str | None = None,
    all_features: bool = False,
) -> None:
    """Legacy alias: uses repo-local USDJPY CSV unless --csv is provided.
    
    DEPRECATED: Use train_buddy with --oanda-live instead.
    """
    _ = (instruments, granularity, candles, checkpoint_path)
    # Import here to avoid circular imports
    from cli.commands import train_buddy
    train_buddy(config_path, None, all_features=all_features)


@_deprecated
def chat_unified(config_path: str, metrics_path: str | None = None) -> None:
    """Interactive chat over the latest unified head metrics.
    
    DEPRECATED: Use buddy commands instead.
    """
    from unified_chat import run_unified_chat

    run_unified_chat(config_path, metrics_path=metrics_path)


@_deprecated
def talk_unified(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
    ticker: str | None = None,
    period: str = "5d",
    interval: str = "1h",
    verbose: bool = False,
) -> None:
    """Legacy unified talk command.
    
    DEPRECATED: Unified talk has been replaced by TF-only Buddy.
    """
    _ = (checkpoint_path, csv_path, ticker, period, interval, verbose)
    raise RuntimeError("Retired: unified talk has been replaced by TF-only Buddy.")
