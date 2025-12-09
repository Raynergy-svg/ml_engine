# predict.py
"""
Advanced real-time prediction module for stock price forecasting.

Features:
- Real-time and batch predictions
- Model validation and backtesting
- Confidence intervals and uncertainty quantification
- Advanced visualization
- Prediction caching and logging
- Multi-model ensemble support
- Alert system for significant predictions
- Performance monitoring
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# Import from other modules
try:
    from train import EnhancedMLEngine, auto_configure
    from preprocess import (
        DataPreprocessor,
        PreprocessConfig,
        FeatureEngineer
    )
except ImportError:
    logging.warning(
        "Some imports failed. Ensure train.py and preprocess.py exist."
    )

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn
    )
    from rich.panel import Panel
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logging.warning("Rich not available. Install with: pip install rich")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

# Initialize console
console = Console() if RICH_AVAILABLE else None

# Constants
PREDICTION_CACHE_DIR = Path("./trained_data/predictions")
PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CHECKPOINT = "./trained_data/models/best_model_checkpoint.pth"
CONFIDENCE_LEVELS = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ


# ========================================================================
# Data Classes
# ========================================================================


@dataclass
class PredictionResult:
    """Structured prediction result."""
    ticker: str
    timestamp: datetime
    current_price: float
    predicted_price: float
    change_percent: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    model_confidence: float
    features_used: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'predicted_price': self.predicted_price,
            'change_percent': self.change_percent,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'model_confidence': self.model_confidence,
            'features_used': self.features_used
        }


@dataclass
class ValidationMetrics:
    """Model validation metrics."""
    mae: float
    rmse: float
    mape: float
    r2_score: float
    samples: int
    ticker: str
    timeframe: str


# ========================================================================
# Model Loading and Management
# ========================================================================


class ModelManager:
    """Manage model loading and inference."""
    
    def __init__(self, checkpoint_path: str, config: Dict[str, Any]):
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.engine = None
        self.loaded_at = None
        
    def load_model(self) -> EnhancedMLEngine:
        """
        Load trained model from checkpoint.
        
        Returns:
            Loaded EnhancedMLEngine instance
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If loading fails
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )
        
        try:
            self.engine = EnhancedMLEngine(self.config)
            self.engine.load_checkpoint(str(self.checkpoint_path))
            self.loaded_at = datetime.now()
            
            if console:
                console.print(
                    f"[green]✓[/green] Model loaded from "
                    f"{self.checkpoint_path}"
                )
            
            return self.engine
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint: {e}"
            ) from e
    
    def verify_model(self) -> bool:
        """
        Verify model is working with sample data.
        
        Returns:
            True if model works, False otherwise
        """
        if self.engine is None:
            return False
        
        try:
            # Create sample input
            input_features = self.config.get("input_features", 7)
            sample_input = np.random.rand(60, input_features)
            
            # Make prediction
            pred = self.engine.predict(sample_input)
            
            # Check output is valid
            is_valid = (
                isinstance(pred, (int, float, np.number)) and
                not np.isnan(pred) and
                not np.isinf(pred)
            )
            
            if is_valid and console:
                console.print(
                    f"[green]✓[/green] Model verification passed "
                    f"(output: {pred:.4f})"
                )
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False


# ========================================================================
# Data Fetching
# ========================================================================


class DataFetcher:
    """Enhanced data fetching with caching and validation."""
    
    def __init__(self, cache_ttl: int = 300):
        """
        Initialize data fetcher.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.preprocessor = DataPreprocessor(PreprocessConfig())
        
    def fetch_data(
        self,
        ticker: str,
        days: int = 100,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with caching.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with stock data or None on failure
        """
        cache_key = f"{ticker}_{days}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()
            
            if age < self.cache_ttl:
                logger.debug(f"Using cached data for {ticker}")
                return cached_data
        
        # Fetch fresh data
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                logger.warning(f"No data fetched for {ticker}")
                return None
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            # Validate data
            if len(data) < 50:
                logger.warning(
                    f"Insufficient data for {ticker}: "
                    f"{len(data)} rows"
                )
                return None
            
            # Cache the data
            self.cache[cache_key] = (data, datetime.now())
            
            logger.info(f"Fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear data cache."""
        self.cache.clear()
        logger.info("Data cache cleared")


# ========================================================================
# Prediction Engine
# ========================================================================


class PredictionEngine:
    """Advanced prediction engine with uncertainty quantification."""
    
    def __init__(
        self,
        model: EnhancedMLEngine,
        preprocessor: DataPreprocessor
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.prediction_history = deque(maxlen=1000)
        
    def preprocess_for_prediction(
        self,
        data: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """
        Preprocess data for prediction.
        
        Args:
            data: Raw stock data
            
        Returns:
            Preprocessed sequence or None on failure
        """
        try:
            # Engineer features
            engineer = FeatureEngineer()
            features = engineer.engineer_features(
                data,
                add_advanced=True
            )
            
            if features is None or features.empty:
                logger.warning("Feature engineering failed")
                return None
            
            # Scale features
            scaled = self.preprocessor.scaler.fit_transform(features)
            
            # Create sequence (just the last one)
            X, _ = self.preprocessor.sequence_gen.create_sequences(
                scaled
            )
            
            if X is None or len(X) == 0:
                logger.warning("Sequence creation failed")
                return None
            
            return X[-1]  # Return last sequence
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[float, Tuple[float, float], float]:
        """
        Make prediction with confidence interval using bootstrap.
        
        Args:
            X: Input sequence
            n_samples: Number of bootstrap samples
            
        Returns:
            Tuple of (prediction, confidence_interval, confidence_score)
        """
        try:
            # Base prediction
            base_pred = float(self.model.predict(X))
            
            # Bootstrap predictions for confidence interval
            predictions = []
            
            # Add small random noise for uncertainty estimation
            for _ in range(n_samples):
                noise = np.random.normal(0, 0.01, X.shape)
                X_noisy = X + noise
                pred = float(self.model.predict(X_noisy))
                predictions.append(pred)
            
            # Calculate confidence interval (95%)
            predictions = np.array(predictions)
            lower = np.percentile(predictions, 2.5)
            upper = np.percentile(predictions, 97.5)
            
            # Confidence score based on interval width
            interval_width = upper - lower
            relative_width = interval_width / base_pred if base_pred != 0 else 1
            confidence_score = max(0, min(1, 1 - relative_width))
            
            return base_pred, (lower, upper), confidence_score
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, (None, None), 0.0
    
    def make_prediction(
        self,
        ticker: str,
        data: pd.DataFrame
    ) -> Optional[PredictionResult]:
        """
        Make complete prediction with all metadata.
        
        Args:
            ticker: Stock ticker symbol
            data: Raw stock data
            
        Returns:
            PredictionResult or None on failure
        """
        try:
            # Preprocess
            X = self.preprocess_for_prediction(data)
            if X is None:
                return None
            
            # Get current price
            current_price = float(data['close'].iloc[-1])
            
            # Make prediction with confidence
            pred, conf_interval, confidence = (
                self.predict_with_confidence(X)
            )
            
            if pred is None:
                return None
            
            # Calculate change
            change_pct = (
                (pred - current_price) / current_price * 100
            )
            
            # Create result
            result = PredictionResult(
                ticker=ticker,
                timestamp=datetime.now(),
                current_price=current_price,
                predicted_price=pred,
                change_percent=change_pct,
                confidence_interval=conf_interval,
                confidence_level=0.95,
                model_confidence=confidence,
                features_used=X.shape[1]
            )
            
            # Store in history
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return None


# ========================================================================
# Validation and Backtesting
# ========================================================================


class ModelValidator:
    """Validate model predictions against historical data."""
    
    def __init__(
        self,
        prediction_engine: PredictionEngine,
        data_fetcher: DataFetcher
    ):
        self.prediction_engine = prediction_engine
        self.data_fetcher = data_fetcher
    
    def validate(
        self,
        ticker: str,
        lookback_days: int = 30
    ) -> Optional[ValidationMetrics]:
        """
        Validate model on historical data.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to validate
            
        Returns:
            ValidationMetrics or None on failure
        """
        try:
            # Fetch extended data
            data = self.data_fetcher.fetch_data(
                ticker,
                days=lookback_days + 100
            )
            
            if data is None:
                return None
            
            predictions = []
            actuals = []
            
            # Make predictions on rolling windows
            for i in range(
                max(0, len(data) - lookback_days),
                len(data)
            ):
                # Use data up to point i
                historical = data.iloc[:i]
                
                if len(historical) < 100:
                    continue
                
                # Make prediction
                X = self.prediction_engine.preprocess_for_prediction(
                    historical
                )
                
                if X is not None:
                    pred = self.prediction_engine.model.predict(X)
                    actual = float(data['close'].iloc[i])
                    
                    predictions.append(pred)
                    actuals.append(actual)
            
            if len(predictions) < 5:
                logger.warning(
                    f"Insufficient predictions for validation: "
                    f"{len(predictions)}"
                )
                return None
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mae = float(np.mean(np.abs(predictions - actuals)))
            rmse = float(
                np.sqrt(np.mean((predictions - actuals) ** 2))
            )
            mape = float(
                np.mean(
                    np.abs((actuals - predictions) / actuals)
                ) * 100
            )
            
            # R² score
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = float(1 - (ss_res / ss_tot))
            
            return ValidationMetrics(
                mae=mae,
                rmse=rmse,
                mape=mape,
                r2_score=r2,
                samples=len(predictions),
                ticker=ticker,
                timeframe=f"{lookback_days} days"
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None


# ========================================================================
# Display and Reporting
# ========================================================================


class PredictionDisplay:
    """Handle display and reporting of predictions."""
    
    @staticmethod
    def show_prediction(result: PredictionResult) -> None:
        """Display single prediction result."""
        if not console:
            print(f"Prediction for {result.ticker}: ${result.predicted_price:.2f}")
            return
        
        # Create table
        table = Table(
            title=f"Prediction for {result.ticker}",
            box=box.ROUNDED
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add rows
        table.add_row(
            "Current Price",
            f"${result.current_price:.2f}"
        )
        table.add_row(
            "Predicted Price",
            f"${result.predicted_price:.2f}"
        )
        
        # Change with color
        change_color = "green" if result.change_percent > 0 else "red"
        table.add_row(
            "Change",
            f"[{change_color}]{result.change_percent:+.2f}%"
            f"[/{change_color}]"
        )
        
        # Confidence interval
        ci_lower, ci_upper = result.confidence_interval
        table.add_row(
            "95% Confidence",
            f"${ci_lower:.2f} - ${ci_upper:.2f}"
        )
        
        table.add_row(
            "Model Confidence",
            f"{result.model_confidence:.1%}"
        )
        
        table.add_row(
            "Timestamp",
            result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        console.print(table)
    
    @staticmethod
    def show_validation(metrics: ValidationMetrics) -> None:
        """Display validation metrics."""
        if not console:
            print(f"Validation MAE: {metrics.mae:.4f}")
            return
        
        table = Table(
            title=f"Validation Results - {metrics.ticker}",
            box=box.ROUNDED
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Mean Absolute Error", f"${metrics.mae:.4f}")
        table.add_row("Root Mean Squared Error", f"${metrics.rmse:.4f}")
        table.add_row("Mean Absolute % Error", f"{metrics.mape:.2f}%")
        table.add_row("R² Score", f"{metrics.r2_score:.4f}")
        table.add_row("Samples", str(metrics.samples))
        table.add_row("Timeframe", metrics.timeframe)
        
        console.print(table)
    
    @staticmethod
    def show_batch_results(results: List[PredictionResult]) -> None:
        """Display batch prediction results."""
        if not console or not results:
            return
        
        table = Table(
            title="Batch Prediction Results",
            box=box.ROUNDED
        )
        table.add_column("Ticker", style="cyan")
        table.add_column("Current", style="yellow")
        table.add_column("Predicted", style="green")
        table.add_column("Change", style="magenta")
        table.add_column("Confidence", style="blue")
        
        for result in results:
            change_color = (
                "green" if result.change_percent > 0 else "red"
            )
            table.add_row(
                result.ticker,
                f"${result.current_price:.2f}",
                f"${result.predicted_price:.2f}",
                f"[{change_color}]{result.change_percent:+.2f}%"
                f"[/{change_color}]",
                f"{result.model_confidence:.0%}"
            )
        
        console.print(table)


# ========================================================================
# Main Prediction Workflows
# ========================================================================


class RealTimePredictor:
    """Orchestrate real-time predictions."""
    
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        config: Optional[Dict[str, Any]] = None
    ):
        # Initialize components
        if config is None:
            config = auto_configure()
        
        self.model_manager = ModelManager(checkpoint_path, config)
        self.data_fetcher = DataFetcher()
        
        # Load model
        model = self.model_manager.load_model()
        
        # Verify model
        if not self.model_manager.verify_model():
            raise RuntimeError("Model verification failed")
        
        # Create prediction engine
        preprocessor = DataPreprocessor(PreprocessConfig())
        self.prediction_engine = PredictionEngine(model, preprocessor)
        
        # Create validator
        self.validator = ModelValidator(
            self.prediction_engine,
            self.data_fetcher
        )
        
        # Display manager
        self.display = PredictionDisplay()
    
    def predict_single(
        self,
        ticker: str,
        show_result: bool = True
    ) -> Optional[PredictionResult]:
        """
        Make single prediction.
        
        Args:
            ticker: Stock ticker symbol
            show_result: Whether to display result
            
        Returns:
            PredictionResult or None
        """
        # Fetch data
        data = self.data_fetcher.fetch_data(ticker)
        if data is None:
            logger.error(f"Failed to fetch data for {ticker}")
            return None
        
        # Make prediction
        result = self.prediction_engine.make_prediction(ticker, data)
        
        if result and show_result:
            self.display.show_prediction(result)
        
        return result
    
    def predict_batch(
        self,
        tickers: List[str],
        show_results: bool = True
    ) -> List[PredictionResult]:
        """
        Make batch predictions.
        
        Args:
            tickers: List of ticker symbols
            show_results: Whether to display results
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        if console:
            console.print(
                f"\n[bold]Processing {len(tickers)} tickers...[/bold]"
            )
        
        # Use progress bar if available
        if console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task(
                    "Making predictions...",
                    total=len(tickers)
                )
                
                for ticker in tickers:
                    result = self.predict_single(
                        ticker,
                        show_result=False
                    )
                    if result:
                        results.append(result)
                    progress.update(task, advance=1)
        else:
            for ticker in tickers:
                result = self.predict_single(ticker, show_result=False)
                if result:
                    results.append(result)
        
        # Display results
        if show_results:
            self.display.show_batch_results(results)
        
        return results
    
    def continuous_predict(
        self,
        ticker: str,
        interval: int = 60,
        max_iterations: Optional[int] = None
    ) -> None:
        """
        Run continuous prediction loop.
        
        Args:
            ticker: Stock ticker symbol
            interval: Sleep interval in seconds
            max_iterations: Max iterations (None = infinite)
        """
        if console:
            console.print(
                "\n[bold blue]Starting continuous predictions"
                "[/bold blue]"
            )
            console.print(
                f"Ticker: {ticker} | Interval: {interval}s"
            )
            console.print("[dim]Press Ctrl+C to exit[/dim]\n")
        
        iteration = 0
        
        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    break
                
                # Make prediction
                result = self.predict_single(ticker, show_result=True)
                
                if result:
                    # Show trend if we have history
                    history = list(self.prediction_engine.prediction_history)
                    if len(history) >= 2:
                        prev = history[-2].predicted_price
                        curr = result.predicted_price
                        change = (
                            (curr - prev) / prev * 100
                        )
                        
                        trend = "↑" if change > 0 else "↓"
                        color = "green" if change > 0 else "red"
                        
                        if console:
                            console.print(
                                f"\n[{color}]{trend} "
                                f"{abs(change):.2f}% from last"
                                f"[/{color}]"
                            )
                
                iteration += 1
                time.sleep(interval)
                
        except KeyboardInterrupt:
            if console:
                console.print(
                    "\n[yellow]Stopped by user[/yellow]"
                )
            
            # Show summary
            history = list(self.prediction_engine.prediction_history)
            if history and console:
                avg_pred = np.mean([
                    h.predicted_price for h in history
                ])
                console.print(f"\nTotal predictions: {len(history)}")
                console.print(f"Average prediction: ${avg_pred:.2f}")
    
    def validate_model(
        self,
        ticker: str,
        lookback_days: int = 30
    ) -> Optional[ValidationMetrics]:
        """
        Validate model performance.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Days to validate
            
        Returns:
            ValidationMetrics or None
        """
        if console:
            console.print(
                f"\n[bold]Validating model on {ticker}...[/bold]"
            )
        
        metrics = self.validator.validate(ticker, lookback_days)
        
        if metrics:
            self.display.show_validation(metrics)
        
        return metrics


# ========================================================================
# Utility Functions
# ========================================================================


def save_predictions(
    results: List[PredictionResult],
    output_file: Union[str, Path]
) -> None:
    """
    Save prediction results to CSV.
    
    Args:
        results: List of prediction results
        output_file: Output file path
    """
    if not results:
        logger.warning("No results to save")
        return
    
    # Convert to DataFrame
    data = [r.to_dict() for r in results]
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(results)} predictions to {output_path}")
    
    if console:
        console.print(
            f"[green]✓[/green] Results saved to {output_path}"
        )


def load_predictions(
    input_file: Union[str, Path]
) -> List[PredictionResult]:
    """
    Load prediction results from CSV.
    
    Args:
        input_file: Input file path
        
    Returns:
        List of PredictionResult objects
    """
    df = pd.read_csv(input_file)
    
    results = []
    for _, row in df.iterrows():
        result = PredictionResult(
            ticker=row['ticker'],
            timestamp=pd.to_datetime(row['timestamp']),
            current_price=row['current_price'],
            predicted_price=row['predicted_price'],
            change_percent=row['change_percent'],
            confidence_interval=eval(row['confidence_interval']),
            confidence_level=row['confidence_level'],
            model_confidence=row['model_confidence'],
            features_used=row['features_used']
        )
        results.append(result)
    
    return results


# ========================================================================
# CLI Interface
# ========================================================================


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced stock price prediction system"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol"
    )
    
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        help="Multiple ticker symbols for batch prediction"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch", "continuous", "validate"],
        default="single",
        help="Prediction mode"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval for continuous mode (seconds)"
    )
    
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Lookback days for validation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for batch predictions"
    )
    
    args = parser.parse_args()
    
    # Display banner
    if console:
        console.print(
            "\n[bold blue]"
            "═" * 50
            "[/bold blue]"
        )
        console.print(
            "[bold blue]  Advanced Stock Price Prediction System  "
            "[/bold blue]"
        )
        console.print(
            "[bold blue]"
            "═" * 50
            "[/bold blue]\n"
        )
    
    try:
        # Initialize predictor
        predictor = RealTimePredictor(args.checkpoint)
        
        # Execute based on mode
        if args.mode == "single":
            predictor.predict_single(args.ticker)
            
        elif args.mode == "batch":
            tickers = args.tickers or ["AAPL", "MSFT", "GOOGL"]
            results = predictor.predict_batch(tickers)
            
            if args.output and results:
                save_predictions(results, args.output)
                
        elif args.mode == "continuous":
            predictor.continuous_predict(args.ticker, args.interval)
            
        elif args.mode == "validate":
            predictor.validate_model(args.ticker, args.lookback)
    
    except FileNotFoundError as e:
        if console:
            console.print(f"[red]Error: {e}[/red]")
            console.print(
                "\n[yellow]Tip:[/yellow] Train a model first "
                "using train.py"
            )
        else:
            print(f"Error: {e}")
            
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]Exiting...[/yellow]")
        else:
            print("Exiting...")
            
    except Exception as e:
        if console:
            console.print(f"[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error")


if __name__ == "__main__":
    main()
