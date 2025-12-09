# predict.py
"""Real-time prediction module for stock price forecasting."""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from train import EnhancedMLEngine, auto_configure
from preprocess import preprocess_data, _create_features
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)


def load_trained_engine(
    checkpoint_path: str, config: Dict[str, Any]
) -> EnhancedMLEngine:
    """
    Load a trained ML engine from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file
        config: Engine configuration dictionary

    Returns:
        Loaded EnhancedMLEngine instance

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        engine = EnhancedMLEngine(config)
        engine.load_checkpoint(str(ckpt_path))
        console.print(f"[green]✓[/green] Model loaded from {checkpoint_path}")
        return engine
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e


def fetch_recent_data(ticker: str, days: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetch recent stock data for prediction.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data to fetch

    Returns:
        DataFrame with recent stock data or None on failure
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
        )

        if data.empty:
            logger.warning(f"No data fetched for {ticker}")
            return None

        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None


def make_prediction(engine: EnhancedMLEngine, data: pd.DataFrame) -> Optional[float]:
    """
    Make a prediction using the engine.

    Args:
        engine: Trained ML engine
        data: Input data DataFrame

    Returns:
        Predicted value or None on failure
    """
    try:
        features = _create_features(data)
        if features is None or features.empty:
            logger.warning("Feature creation failed")
            return None

        X, _ = preprocess_data(features)
        if X is None or len(X) == 0:
            logger.warning("Data preprocessing failed")
            return None

        prediction = engine.predict(X[-1])
        return float(prediction)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None


def real_time_prediction_loop(
    engine: EnhancedMLEngine,
    ticker: str = "AAPL",
    interval: int = 60,
    use_real_data: bool = True,
):
    """
    Run continuous prediction loop.

    Args:
        engine: Trained ML engine
        ticker: Stock ticker to predict (if use_real_data=True)
        interval: Sleep interval in seconds between predictions
        use_real_data: Use real market data vs simulated data
    """
    console.print("[bold blue]Starting real-time prediction loop.[/bold blue]")
    console.print(
        f"Ticker: {ticker} | Interval: {interval}s | "
        f"Mode: {'Real Data' if use_real_data else 'Simulated'}"
    )
    console.print("[dim]Press Ctrl+C to exit.[/dim]\n")

    predictions_history = []

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if use_real_data:
                # Fetch real market data
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task(f"Fetching data for {ticker}...", total=None)
                    data = fetch_recent_data(ticker)

                if data is None:
                    console.print(f"[red]✗[/red] Failed to fetch data for {ticker}")
                    time.sleep(interval)
                    continue
            else:
                # Use simulated data for testing
                data = pd.DataFrame(
                    {
                        "open": np.random.rand(100) * 100,
                        "high": np.random.rand(100) * 100,
                        "low": np.random.rand(100) * 100,
                        "close": np.random.rand(100) * 100,
                        "volume": np.random.rand(100) * 1e6,
                    }
                )

            # Make prediction
            prediction = make_prediction(engine, data)

            if prediction is not None:
                predictions_history.append(
                    {"timestamp": timestamp, "prediction": prediction}
                )

                # Display result
                table = Table(show_header=True, header_style="bold")
                table.add_column("Time", style="cyan")
                table.add_column("Prediction", style="green")
                table.add_column("Ticker", style="yellow")

                table.add_row(
                    timestamp,
                    f"${prediction:.2f}",
                    ticker if use_real_data else "SIMULATED",
                )

                console.print(table)

                # Show prediction trend if we have history
                if len(predictions_history) >= 2:
                    prev = predictions_history[-2]["prediction"]
                    curr = prediction
                    change = ((curr - prev) / prev) * 100

                    trend_icon = "↑" if change > 0 else "↓"
                    trend_color = "green" if change > 0 else "red"

                    console.print(
                        f"[{trend_color}]{trend_icon} "
                        f"{abs(change):.2f}% from last prediction"
                        f"[/{trend_color}]\n"
                    )
            else:
                console.print(f"[red]✗[/red] Prediction failed at {timestamp}\n")

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Real-time prediction loop terminated by user.[/yellow]"
        )

        # Display summary
        if predictions_history:
            console.print("\n[bold]Session Summary:[/bold]")
            console.print(f"Total predictions: {len(predictions_history)}")
            avg_pred = np.mean([p["prediction"] for p in predictions_history])
            console.print(f"Average prediction: ${avg_pred:.2f}")
    except Exception as e:
        console.print(f"[red]Error in prediction loop: {e}[/red]")
        logger.exception("Prediction loop error")


def validate_prediction(
    engine: EnhancedMLEngine, ticker: str = "AAPL", days_back: int = 30
):
    """
    Validate model predictions against historical data.

    Args:
        engine: Trained ML engine
        ticker: Stock ticker to validate
        days_back: Number of days to look back
    """
    console.print(f"\n[bold]Validating predictions for {ticker}...[/bold]")

    try:
        data = fetch_recent_data(ticker, days=days_back + 100)
        if data is None:
            console.print("[red]Failed to fetch validation data[/red]")
            return

        features = _create_features(data)
        X, y = preprocess_data(features)

        if X is None or len(X) < 10:
            console.print("[red]Insufficient data for validation[/red]")
            return

        # Make predictions on recent data
        predictions = []
        actuals = []

        for i in range(min(10, len(X))):
            pred = engine.predict(X[-(i + 1)])
            predictions.append(pred)
            actuals.append(y[-(i + 1)])

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # Display results
        table = Table(title="Validation Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("MAE", f"{mae:.4f}")
        table.add_row("RMSE", f"{rmse:.4f}")
        table.add_row("MAPE", f"{mape:.2f}%")
        table.add_row("Samples", str(len(predictions)))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        logger.exception("Validation failed")


if __name__ == "__main__":
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time stock price prediction")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./trained_data/models/model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Prediction interval in seconds",
    )
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="Use simulated data instead of real market data",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation before starting predictions",
    )

    args = parser.parse_args()

    # Display banner
    console.print("\n[bold blue]═══════════════════════════════════[/bold blue]")
    console.print("[bold blue]  Real-Time Stock Price Predictor  [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════[/bold blue]\n")

    try:
        # Load configuration and model
        config = auto_configure()
        engine = load_trained_engine(args.checkpoint, config)

        # Run validation if requested
        if args.validate:
            validate_prediction(engine, args.ticker)

        # Quick sample prediction to verify model
        console.print("\n[bold]Running quick test prediction...[/bold]")
        sample_input = np.random.rand(60, config.get("input_features", 7))
        pred = engine.predict(sample_input)
        console.print(f"[green]✓[/green] Model is working. Sample output: {pred:.4f}\n")

        # Start the real-time prediction loop
        real_time_prediction_loop(
            engine,
            ticker=args.ticker,
            interval=args.interval,
            use_real_data=not args.simulated,
        )

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Tip:[/yellow] Train a model first using train.py")
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error in main")


# Utility functions for batch predictions


def batch_predict(
    engine: EnhancedMLEngine, tickers: list, output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Make predictions for multiple tickers at once.

    Args:
        engine: Trained ML engine
        tickers: List of ticker symbols
        output_file: Optional file to save results

    Returns:
        DataFrame with predictions for all tickers
    """
    results = []

    console.print(f"\n[bold]Batch prediction for {len(tickers)} tickers[/bold]")

    for ticker in tickers:
        try:
            data = fetch_recent_data(ticker)
            if data is None:
                console.print(f"[red]✗[/red] {ticker}: Failed to fetch data")
                continue

            prediction = make_prediction(engine, data)
            if prediction is not None:
                current_price = float(data["close"].iloc[-1])
                change_pct = ((prediction - current_price) / current_price) * 100

                results.append(
                    {
                        "ticker": ticker,
                        "current_price": current_price,
                        "predicted_price": prediction,
                        "change_percent": change_pct,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                trend = "↑" if change_pct > 0 else "↓"
                color = "green" if change_pct > 0 else "red"
                console.print(
                    f"[{color}]{trend}[/{color}] {ticker}: "
                    f"${current_price:.2f} → ${prediction:.2f} "
                    f"({change_pct:+.2f}%)"
                )
            else:
                console.print(f"[red]✗[/red] {ticker}: Prediction failed")

        except Exception as e:
            console.print(f"[red]✗[/red] {ticker}: {e}")
            logger.exception(f"Error predicting {ticker}")

    if results:
        df = pd.DataFrame(results)

        # Display summary table
        table = Table(title="\nBatch Prediction Summary")
        table.add_column("Ticker", style="cyan")
        table.add_column("Current", style="yellow")
        table.add_column("Predicted", style="green")
        table.add_column("Change", style="magenta")

        for _, row in df.iterrows():
            change_color = "green" if row["change_percent"] > 0 else "red"
            table.add_row(
                row["ticker"],
                f"${row['current_price']:.2f}",
                f"${row['predicted_price']:.2f}",
                f"[{change_color}]{row['change_percent']:+.2f}%[/{change_color}]",
            )

        console.print(table)

        # Save to file if requested
        if output_file:
            df.to_csv(output_file, index=False)
            console.print(f"\n[green]✓[/green] Results saved to {output_file}")

        return df
    else:
        console.print("\n[red]No successful predictions[/red]")
        return pd.DataFrame()


def compare_predictions(
    engine: EnhancedMLEngine, ticker: str, lookback_days: int = 30
) -> None:
    """
    Compare model predictions with actual historical prices.

    Args:
        engine: Trained ML engine
        ticker: Stock ticker symbol
        lookback_days: Number of days to look back
    """
    console.print(f"\n[bold]Comparing predictions for {ticker}[/bold]")

    # Fetch more data than needed
    data = fetch_recent_data(ticker, days=lookback_days + 100)
    if data is None:
        console.print("[red]Failed to fetch data[/red]")
        return

    # Make predictions on recent days
    predictions = []
    actuals = []
    dates = []

    for i in range(lookback_days):
        # Use data up to this point
        historical_data = data.iloc[: -lookback_days + i]
        if len(historical_data) < 100:
            continue

        pred = make_prediction(engine, historical_data)
        if pred is not None:
            actual = float(data["close"].iloc[-lookback_days + i])
            predictions.append(pred)
            actuals.append(actual)
            dates.append(data.index[-lookback_days + i])

    if predictions:
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # Display results
        table = Table(title="Prediction Accuracy")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Mean Absolute Error", f"${mae:.2f}")
        table.add_row("Root Mean Squared Error", f"${rmse:.2f}")
        table.add_row("Mean Absolute % Error", f"{mape:.2f}%")
        table.add_row("Days Analyzed", str(len(predictions)))

        console.print(table)

        # Show recent predictions
        console.print("\n[bold]Recent Predictions:[/bold]")
        for i in range(min(5, len(predictions))):
            idx = -(i + 1)
            error = predictions[idx] - actuals[idx]
            error_pct = (error / actuals[idx]) * 100
            color = "green" if abs(error_pct) < 5 else "red"

            console.print(
                f"  {dates[idx].strftime('%Y-%m-%d')}: "
                f"Predicted ${predictions[idx]:.2f}, "
                f"Actual ${actuals[idx]:.2f} "
                f"[{color}]({error_pct:+.2f}%)[/{color}]"
            )
    else:
        console.print("[red]No valid predictions generated[/red]")


def save_prediction_log(
    ticker: str,
    prediction: float,
    current_price: float,
    log_file: str = "predictions.csv",
) -> None:
    """
    Save prediction to a CSV log file.

    Args:
        ticker: Stock ticker symbol
        prediction: Predicted price
        current_price: Current actual price
        log_file: Path to log file
    """
    log_path = Path(log_file)

    # Create header if file doesn't exist
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("timestamp,ticker,current_price,predicted_price,change_percent\n")

    # Append prediction
    change_pct = ((prediction - current_price) / current_price) * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a") as f:
        f.write(
            f"{timestamp},{ticker},{current_price:.2f},"
            f"{prediction:.2f},{change_pct:.2f}\n"
        )

    logger.info(f"Logged prediction for {ticker} to {log_file}")
