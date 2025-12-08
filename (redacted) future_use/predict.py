# predict.py
import time
import numpy as np
import pandas as pd
import torch
from train import EnhancedMLEngine, auto_configure
from rich.console import Console

console = Console()

def load_trained_engine(checkpoint_path: str, config: dict) -> EnhancedMLEngine:
    engine = EnhancedMLEngine(config)
    engine.load_checkpoint(checkpoint_path)
    return engine

def real_time_prediction_loop(engine: EnhancedMLEngine):
    console.print("[bold blue]Starting real-time prediction loop. Press Ctrl+C to exit.[/bold blue]")
    try:
        while True:
            # For demonstration, simulate new market data with random values.
            simulated_data = pd.DataFrame({
                'open': np.random.rand(100) * 100,
                'high': np.random.rand(100) * 100,
                'low': np.random.rand(100) * 100,
                'close': np.random.rand(100) * 100,
                'volume': np.random.rand(100) * 1e6
            })
            # Use the preprocessing function from preprocess.py if needed
            from preprocess import _create_features, preprocess_data
            sim_features = _create_features(simulated_data)
            X_new, _ = preprocess_data(sim_features)
            if X_new is not None and len(X_new) > 0:
                prediction = engine.predict(X_new[-1])
                console.print(f"[bold cyan]Real-time predicted next close price: {prediction:.4f}[/bold cyan]")
            else:
                console.print("[bold red]Insufficient data for real-time prediction.[/bold red]")
            time.sleep(5)
    except KeyboardInterrupt:
        console.print("[bold yellow]Real-time prediction loop terminated by user.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error in real-time prediction loop: {e}[/bold red]")

if __name__ == "__main__":
    # Load configuration and the trained model checkpoint.
    config = auto_configure()
    checkpoint_path = "./trained_data/models/model.pth"  # adjust as needed
    engine = load_trained_engine(checkpoint_path, config)
    # Run a sample prediction.
    sample_input = np.random.rand(60, config.get('input_features', 7))
    pred = engine.predict(sample_input)
    console.print(f"[bold green]Sample prediction: {pred:.4f}[/bold green]")
    # Start the real-time prediction loop.
    real_time_prediction_loop(engine)
