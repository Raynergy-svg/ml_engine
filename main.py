#!/usr/bin/env python3
"""
Enhanced ML Engine Trading Bot CLI

A command-line interface for the ML trading engine that handles:
- Command processing and orchestration
- Interactive dashboard visualization
- Configuration management
- Training/evaluation progress display
- Real-time monitoring
"""

import os
import sys
import time
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import torch
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from ml_engine import ai_assistant, EnhancedMLEngine
import ml_engine
from utils import setup_logging, load_config

# New orchestrating imports:
import visualizer
import openai_integration

# Initialize console and logging
console = Console()
logger = setup_logging(log_file="cli.log")

# Load configuration from config.yaml
config = load_config("config.yaml")

# Set OpenAI credentials using the configuration read from YAML.
openai_integration.set_openai_credentials(config)


def generate_dashboard(
    epoch: int,
    total_epochs: int,
    latency: int,
    train_loss: float,
    val_loss: float,
    lr_delta: float,
    api_logs: list,
    config: Dict[str, Any],
) -> Layout:
    """Generate the rich dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", size=8),
        Layout(name="footer", size=8),
    )
    metrics_table = Table(show_header=True, header_style="bold magenta", expand=True)
    metrics_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    metrics_table.add_column("Value", style="green")
    metrics_table.add_row("Epoch", f"{epoch}/{total_epochs}")
    metrics_table.add_row("Training Loss", f"{train_loss:.6f}")
    metrics_table.add_row("Validation Loss", f"{val_loss:.6f}")
    metrics_table.add_row("API Latency", f"{latency}ms")
    lr_color = "green" if lr_delta >= 0 else "red"
    metrics_table.add_row("LR Change", f"[{lr_color}]{lr_delta:+.2f}%[/{lr_color}]")
    metrics_table.add_row(
        "Device", f"{config.get('hardware', {}).get('device', 'cpu')}"
    )
    status = (
        "[bold green]Active[/bold green]"
        if epoch < total_epochs
        else "[bold yellow]Complete[/bold yellow]"
    )
    metrics_table.add_row("Status", status)
    layout["header"].update(
        Panel(metrics_table, title="[bold]Training Metrics[/bold]", border_style="blue")
    )
    body_layout = Layout()
    body_layout.split_row(
        Layout(name="progress", ratio=1), Layout(name="logs", ratio=2)
    )
    progress_pct = (epoch / total_epochs * 100) if total_epochs > 0 else 0
    bar = "█" * int(progress_pct / 2) + "░" * (50 - int(progress_pct / 2))
    eta = (total_epochs - epoch) * latency / 1000
    progress_text = f"{progress_pct:.1f}%\n{bar}\nETA: {eta:.1f}s"
    body_layout["progress"].update(
        Panel(progress_text, title="[bold]Progress[/bold]", border_style="green")
    )
    current_time = time.strftime("%H:%M:%S")
    if not api_logs:
        api_logs.append(f"[grey]{current_time}[/grey] Starting training...")
    formatted_logs = [
        log if log.startswith("[") else f"[grey]{current_time}[/grey] {log}"
        for log in api_logs[-5:]
    ]
    logs_content = "\n".join(formatted_logs)
    body_layout["logs"].update(
        Panel(logs_content, title="[bold]Recent Updates[/bold]", border_style="yellow")
    )
    layout["body"].update(body_layout)
    config_table = Table(show_header=True, header_style="bold magenta", expand=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    model_config = config.get("model", {})
    config_table.add_row("Learning Rate", f"{model_config.get('learning_rate', 'N/A')}")
    config_table.add_row("Batch Size", f"{model_config.get('batch_size', 'N/A')}")
    config_table.add_row("Architecture", f"{model_config.get('architecture', 'N/A')}")
    config_table.add_row("Hidden Size", f"{model_config.get('hidden_size', 'N/A')}")
    config_table.add_row("Num Layers", f"{model_config.get('num_layers', 'N/A')}")
    config_table.add_row("Dropout", f"{model_config.get('dropout', 'N/A')}")
    optimizer_config = model_config.get("optimizer", {})
    config_table.add_row("Optimizer", f"{optimizer_config.get('type', 'Adam')}")
    logs_table = Table(show_header=True, header_style="bold magenta", expand=True)
    logs_table.add_column("Latest Log", style="green")
    latest_log = api_logs[-1] if api_logs else "No logs available"
    logs_table.add_row(latest_log)
    footer_layout = Layout()
    footer_layout.split_row(
        Panel(config_table, title="[bold]Configuration[/bold]", border_style="blue"),
        Panel(logs_table, title="[bold]Log Info[/bold]", border_style="blue"),
    )
    layout["footer"].update(footer_layout)
    return layout


# CLI Command Implementations
def train_model(config_path: str, choose_csv: bool = False) -> None:
    """Run model training with live progress dashboard."""
    config = load_config(Path(config_path))
    engine = EnhancedMLEngine(config)

    # changed: create dummy training data as non-empty tensors
    import torch

    X_train_data = [torch.tensor([0.0])]  # dummy training data with one sample
    y_train_data = [torch.tensor([0.0])]  # dummy target data with one sample

    api_logs = [f"[blue]{time.strftime('%H:%M:%S')}[/blue] Starting training"]
    with Live(
        generate_dashboard(
            0, config["training"]["epochs"], 200, 0, 0, 0, api_logs, config
        ),
        refresh_per_second=10,
    ) as live:
        try:
            for epoch, metrics in engine.train(
                X_train_data, y_train_data
            ):  # changed: using dummy data lists
                # Update dashboard with training progress
                api_logs.append(
                    f"[grey]{time.strftime('%H:%M:%S')}[/grey] {metrics['message']}"
                )
                if len(api_logs) > 10:
                    api_logs = api_logs[-10:]
                live.update(
                    generate_dashboard(
                        epoch,
                        config["training"]["epochs"],
                        metrics["latency"],
                        metrics["train_loss"],
                        metrics["val_loss"],
                        metrics["lr_delta"],
                        api_logs,
                        config,
                    )
                )
        except KeyboardInterrupt:
            api_logs.append("[red]Training interrupted by user[/red]")
            live.update(
                generate_dashboard(
                    epoch,
                    config["training"]["epochs"],
                    metrics["latency"],
                    metrics["train_loss"],
                    metrics["val_loss"],
                    metrics["lr_delta"],
                    api_logs,
                    config,
                )
            )
            engine.save_model("interrupted_model.pth")
            return


def evaluate_model(config_path: str) -> None:
    """Run model evaluation with results dashboard."""
    config = load_config(Path(config_path))
    engine = EnhancedMLEngine(config)

    api_logs = [f"[blue]{time.strftime('%H:%M:%S')}[/blue] Starting evaluation"]
    # Provide placeholder values for generate_dashboard arguments
    with Live(generate_dashboard(0, 10, 200, 0.0, 0.0, 0.0, api_logs, config)) as live:
        metrics = engine.evaluate_model()
        # Update dashboard with evaluation results
        # ... dashboard updates ...


# New CLI command implementations
def visualize_dashboard(config_path: str) -> None:
    """Display dashboard visualizations using the visualizer module."""
    config = load_config(Path(config_path))
    console.print("[bold blue]Launching visualizations...[/bold blue]")
    visualizer.display_dashboard(
        config
    )  # Assumes implementation exists in visualizer.py


def openai_tune(config_path: str) -> None:
    """Execute OpenAI-based auto-tuning using the openai_integration module."""
    current_config = load_config(Path(config_path))  # Convert config_path to Path

    # Gather current metrics from the engine
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
        updated_config = openai_integration.report_config_changes(
            current_config, result
        )
        console.print(
            "[bold green]Configuration updated successfully and written to config.yaml[/bold green]"
        )
        console.print(f"[cyan]Updates Applied:[/cyan] {result}")
    else:
        console.print(
            "[bold red]Failed to get configuration updates from OpenAI[/bold red]"
        )


def predict_price(config_path: str) -> None:
    """
    Use ML engine to predict prices given a dataset.
    """
    config = load_config(Path(config_path))
    engine = EnhancedMLEngine(config)
    prediction = engine.predict_price()
    console.print(f"[bold yellow]Predicted Price: {prediction}[/bold yellow]")


def realtime_loop(config_path: str) -> None:
    """
    Run the ML engine in a real-time loop for continuous inference.
    """
    config = load_config(Path(config_path))
    engine = EnhancedMLEngine(config)
    engine.run_realtime_loop()
    console.print("[bold yellow]Realtime Loop command executed[/bold yellow]")


def tune_model(config_path: str) -> None:
    """
    Perform advanced hyperparameter tuning using ML engine methods.
    """
    config = load_config(Path(config_path))
    engine = EnhancedMLEngine(config)
    engine.tune_hyperparameters()
    console.print("[bold yellow]Tune Model command executed[/bold yellow]")


def profile_pipeline(config_path: str) -> None:
    """
    Profile the ML pipeline for bottlenecks.
    """
    config = load_config(Path(config_path))
    engine = EnhancedMLEngine(config)
    engine.profile_pipeline()
    console.print("[bold yellow]Profile Pipeline command executed[/bold yellow]")


def run_ai_assistant(config_path: str) -> None:
    config = load_config(config_path)  # removed Path() wrapper for proper string input
    # Instantiate the AI assistant using the class from ml_engine
    assistant_instance = ai_assistant(config)
    console.print("[bold blue]Launching AI Assistant...[/bold blue]")
    query = input("Enter your query: ")
    response = assistant_instance.process_query(query, use_claude=True)
    console.print(f"[bold green]Response:[/bold green] {response}")

    # Optionally use ML Engine's AI assistant for simulation tasks
    engine_instance = EnhancedMLEngine(config)
    simulation_output = engine_instance.ai_assistant.process_query("simulate")
    console.print(
        f"[bold blue]AI Assistant Simulation Output:[/bold blue]\n{simulation_output}"
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ML Engine Trading Bot CLI")
    parser.add_argument(
        "command",
        choices=[
            "train-model",
            "evaluate-model",
            "predict-price",
            "realtime-loop",
            "tune-model",
            "profile-pipeline",
            "visualize",  # New command
            "openai-tune",  # New command
            "ai-assistant",  # New sub-command
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to configuration file",
    )
    # ... add more CLI arguments as needed ...

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    command_map = {
        "train-model": train_model,
        "evaluate-model": evaluate_model,
        "predict-price": predict_price,  # New
        "realtime-loop": realtime_loop,  # New
        "tune-model": tune_model,  # New
        "profile-pipeline": profile_pipeline,  # New
        "visualize": visualize_dashboard,  # New mapping
        "openai-tune": openai_tune,  # New mapping
        "ai-assistant": run_ai_assistant,  # updated mapping
    }

    try:
        command_map[args.command](args.config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logging.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
