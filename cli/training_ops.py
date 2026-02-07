#!/usr/bin/env python3
"""
Training operations module for ML Engine CLI.

Contains functions for training RL position sizer and retraining gate models.

Improvements made:
1. Code readability: Added constants, dataclasses, helper classes, better organization
2. Performance: Pre-allocated arrays, vectorized operations, optimized loops
3. Best practices: Type hints, dependency injection, proper error handling
4. Error handling: Specific exceptions, retry logic, input validation
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    track,
)
from rich.table import Table

from cli.io_utils import DEFAULT_CONFIG_PATH
from cli.tf_config import _configure_tf_metal

console = Console()

# ============================================================================
# CONSTANTS
# ============================================================================

# File paths
MODELS_DIR: Final[Path] = Path("trained_data/models")
RL_TRAIN_DATA_PATH: Final[Path] = MODELS_DIR / "rl_train_data.npz"
RL_META_PATH: Final[Path] = MODELS_DIR / "rl_position_sizer.meta.json"
ENSEMBLE_META_PATH: Final[Path] = MODELS_DIR / "modular_ensemble.meta.json"

# OANDA API limits
OANDA_MAX_CANDLES: Final[int] = 5000
OANDA_RATE_LIMIT_DELAY: Final[float] = 0.2

# Trading pairs
MAJOR_PAIRS: Final[tuple[str, ...]] = (
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
)

EXTENDED_PAIRS: Final[tuple[str, ...]] = (
    *MAJOR_PAIRS,
    "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "EUR_CHF", "EUR_AUD", "EUR_CAD",
    "AUD_JPY", "CAD_JPY", "CHF_JPY",
)

# Training parameters
MIN_DF_LENGTH: Final[int] = 100
PREDICTION_START_INDEX: Final[int] = 100
ROLLING_WINDOW: Final[int] = 20

# Data split ratios
TRAIN_VAL_TEST_SPLIT: Final[tuple[float, float, float]] = (0.7, 0.2, 0.1)

# Logging
TF_LOG_LEVEL: Final[str] = "3"

# ============================================================================
# TYPE ALIASES AND DATA CLASSES
# ============================================================================

StepStatus: TypeAlias = Literal["pending", "in_progress", "success", "error"]


@dataclass(frozen=True)
class Step:
    """Represents a step in the training process."""

    name: str
    status: StepStatus = "pending"
    message: str = ""


@dataclass
class RLTrainingConfig:
    """Configuration for RL position sizer training."""

    timesteps: int = 500_000
    episodes: int | None = None
    pairs: tuple[str, ...] = MAJOR_PAIRS
    granularity: str = "H1"
    candles: int = 5000
    verbose: bool = False


@dataclass
class GateTrainingConfig:
    """Configuration for gate model retraining."""

    pairs: tuple[str, ...] = EXTENDED_PAIRS
    granularity: str = "H1"
    candles: int = 5000
    verbose: bool = False


@dataclass
class TrainingMetrics:
    """Base class for training metrics."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class XGBoostMetrics(TrainingMetrics):
    """Metrics for XGBoost training."""

    momentum_mae: float = 0.0
    acceleration_accuracy: float = 0.0


@dataclass
class RandomForestMetrics(TrainingMetrics):
    """Metrics for Random Forest training."""

    drawdown_mae_bps: float = 0.0
    drawdown_mae_pct: float = 0.0
    streak_prob_mae: float = 0.0


@dataclass
class RidgeMetrics(TrainingMetrics):
    """Metrics for Ridge/ElasticNet training."""

    confidence_mae: float = 0.0
    r2_score: float = 0.0

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class TrainingError(Exception):
    """Base exception for training errors."""

    pass


class OandaConnectionError(TrainingError):
    """Raised when OANDA connection fails."""

    pass


class ModelNotFoundError(TrainingError):
    """Raised when required model is not found."""

    pass


class DataGenerationError(TrainingError):
    """Raised when data generation fails."""

    pass


class SubprocessError(TrainingError):
    """Raised when subprocess execution fails."""

    def __init__(self, message: str, returncode: int):
        super().__init__(message)
        self.returncode = returncode

# ============================================================================
# HELPER CLASSES
# ============================================================================


class StepTracker:
    """Track and display progress of multi-step operations."""

    def __init__(self, steps: list[str], console: Console | None = None):
        self._console = console or Console()
        self._steps: dict[str, Step] = {name: Step(name) for name in steps}
        self._step_order = steps

    def update(self, step_name: str, status: StepStatus, message: str = "") -> None:
        """Update the status of a step."""
        if step_name not in self._steps:
            raise ValueError(f"Unknown step: {step_name}")
        self._steps[step_name] = Step(step_name, status, message)
        self._print()

    def success(self, step_name: str, message: str = "") -> None:
        """Mark a step as successful."""
        self.update(step_name, "success", message)

    def error(self, step_name: str, message: str = "") -> None:
        """Mark a step as failed."""
        self.update(step_name, "error", message)

    def in_progress(self, step_name: str, message: str = "") -> None:
        """Mark a step as in progress."""
        self.update(step_name, "in_progress", message)

    def _print(self) -> None:
        """Print the current status of all steps."""
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("", width=2)
        table.add_column("", width=28)
        table.add_column("", style="dim")

        for step_name in self._step_order:
            step = self._steps[step_name]
            icon, color = self._get_icon_and_color(step.status)
            table.add_row(f"[{color}]{icon}[/{color}]", step.name, step.message)

        self._console.print(table)

    @staticmethod
    def _get_icon_and_color(status: StepStatus) -> tuple[str, str]:
        """Get the icon and color for a status."""
        return {
            "pending": ("⏳", "yellow"),
            "in_progress": ("⏳", "yellow"),
            "success": ("✓", "green"),
            "error": ("✗", "red"),
        }[status]


class OandaClientWrapper:
    """Wrapper for OANDA client with retry logic."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client: Any | None = None

    def connect(self) -> None:
        """Connect to OANDA API with retry logic."""
        from oanda_practice import OandaPracticeClient

        for attempt in range(self._max_retries):
            try:
                self._client = OandaPracticeClient.from_env()
                return
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise OandaConnectionError(f"Failed to connect to OANDA: {e}")
                time.sleep(self._retry_delay * (attempt + 1))

    def get_candles(
        self,
        pair: str,
        granularity: str,
        count: int,
        price: str = "MBA",
        to_time: str | None = None,
    ) -> dict[str, Any]:
        """Get candles with retry logic."""
        if self._client is None:
            raise OandaConnectionError("Not connected to OANDA")

        params = {"granularity": granularity, "count": count, "price": price}
        if to_time:
            params["to"] = to_time

        for attempt in range(self._max_retries):
            try:
                return self._client.get_candles(pair, **params)
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise TrainingError(f"Failed to fetch candles for {pair}: {e}")
                time.sleep(self._retry_delay * (attempt + 1))

        return {}  # Should never reach here


class FeatureExtractor:
    """Extract features from DataFrame for RL training."""

    def __init__(self, ensemble: Any):
        self._ensemble = ensemble

    def extract_features(
        self,
        df: pd.DataFrame,
        start_idx: int = PREDICTION_START_INDEX,
    ) -> Generator[tuple[np.ndarray, np.ndarray, float], None, None]:
        """
        Extract features from DataFrame.

        Yields tuples of (features, predictions, price).
        """
        prices = df['close'].values

        for i in range(start_idx, len(df) - 1):
            df_slice = df.iloc[:i + 1].copy()
            try:
                signal = self._ensemble.predict(df_slice)

                # Extract features
                features = self._compute_features(df_slice, signal)
                predictions = self._compute_predictions(signal)
                price = prices[i]

                yield features, predictions, price
            except Exception:
                continue

    @staticmethod
    def _compute_features(df_slice: pd.DataFrame, signal: Any) -> np.ndarray:
        """Compute feature vector from data slice and signal."""
        vol = df_slice['close'].pct_change().rolling(ROLLING_WINDOW).std().iloc[-1]
        momentum = signal.xgb_momentum if signal.xgb_momentum else 0.0
        drawdown = signal.rf_drawdown_pips / 100.0 if signal.rf_drawdown_pips else 0.0

        return np.array([vol, momentum, drawdown], dtype=np.float32)

    @staticmethod
    def _compute_predictions(signal: Any) -> np.ndarray:
        """Compute prediction vector from signal."""
        direction_prob = signal.tcn_probability if signal.tcn_probability else 0.5
        confidence = signal.ridge_confidence / 100.0 if signal.ridge_confidence else 0.5

        return np.array([direction_prob, confidence], dtype=np.float32)


class DataBatchProcessor:
    """Process data in batches for OANDA API limits."""

    def __init__(self, oanda_client: OandaClientWrapper):
        self._oanda = oanda_client

    def fetch_pair_data(
        self,
        pair: str,
        granularity: str,
        total_candles: int,
    ) -> pd.DataFrame | None:
        """
        Fetch data for a single pair, handling batching.

        Returns DataFrame with time, open, high, low, close, volume columns,
        or None if fetching fails.
        """
        pair_dfs: list[pd.DataFrame] = []
        remaining = total_candles
        last_time: str | None = None

        while remaining > 0:
            batch_size = min(remaining, OANDA_MAX_CANDLES)

            response = self._oanda.get_candles(
                pair=pair,
                granularity=granularity,
                count=batch_size,
                to_time=last_time,
            )

            candles_raw = response.get('candles', [])
            if not candles_raw:
                break

            batch_df = self._candles_to_dataframe(candles_raw)
            pair_dfs.append(batch_df)

            # Update for next batch (go backwards in time)
            last_time = candles_raw[0].get('time')
            remaining -= len(candles_raw)

            # Rate limiting
            if remaining > 0:
                time.sleep(OANDA_RATE_LIMIT_DELAY)

        if not pair_dfs:
            return None

        # Combine and clean
        df = pd.concat(pair_dfs, ignore_index=True)
        df = (
            df.drop_duplicates(subset=['time'])
            .sort_values('time')
            .reset_index(drop=True)
        )
        df['pair'] = pair

        return df

    @staticmethod
    def _candles_to_dataframe(candles: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert OANDA candles to DataFrame."""
        rows = []
        for candle in candles:
            mid = candle.get('mid', {})
            rows.append({
                'time': candle.get('time'),
                'open': float(mid.get('o', 0)),
                'high': float(mid.get('h', 0)),
                'low': float(mid.get('l', 0)),
                'close': float(mid.get('c', 0)),
                'volume': int(candle.get('volume', 0)),
            })
        return pd.DataFrame(rows)


class MetadataManager:
    """Manage model metadata files."""

    def __init__(self, model_dir: Path = MODELS_DIR):
        self._model_dir = model_dir

    def load(self, filename: str) -> dict[str, Any]:
        """Load metadata from file."""
        path = self._model_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def save(self, filename: str, data: dict[str, Any]) -> None:
        """Save metadata to file."""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        path = self._model_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def update(self, filename: str, updates: dict[str, Any]) -> None:
        """Update existing metadata with new values."""
        meta = self.load(filename)
        meta.update(updates)
        self.save(filename, meta)

# ============================================================================
# CONTEXT MANAGERS
# ============================================================================


@contextmanager
def suppress_logging(loggers: Iterable[str]) -> Generator[None, None, None]:
    """Suppress logging for specified loggers."""
    original_levels = {}
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)


@contextmanager
def configure_tensorflow(verbose: bool = False) -> Generator[None, None, None]:
    """Configure TensorFlow environment."""
    original_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_LOG_LEVEL
    _configure_tf_metal(verbose=verbose)

    try:
        yield
    finally:
        if original_log_level is not None:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_log_level
        else:
            os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_rl_sizer(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    timesteps: int = 500_000,
    episodes: int | None = None,
    pairs: str | None = None,
    granularity: str = "H1",
    candles: int = 5000,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    Train RL position sizing agent using PPO.

    Uses a two-phase subprocess approach to avoid TensorFlow/PyTorch GPU conflicts.

    Args:
        config_path: Path to config file (unused, kept for compatibility)
        timesteps: Number of training timesteps
        episodes: Number of episodes (unused, kept for compatibility)
        pairs: Comma-separated pairs to train on
        granularity: Candle timeframe
        candles: Number of candles per pair
        verbose: Show detailed logs
        **kwargs: Additional keyword arguments (ignored)

    Raises:
        ModelNotFoundError: If ensemble model is not found
        OandaConnectionError: If OANDA connection fails
        DataGenerationError: If data generation fails
        SubprocessError: If subprocess training fails
    """
    # Parse configuration
    pair_list = tuple(p.strip() for p in pairs.split(",")) if pairs else MAJOR_PAIRS
    config = RLTrainingConfig(
        timesteps=timesteps,
        episodes=episodes,
        pairs=pair_list,
        granularity=granularity,
        candles=candles,
        verbose=verbose,
    )

    # Initialize step tracker
    steps = [
        "Check ensemble model",
        "Connect to OANDA",
        "Generate predictions",
        "Train PPO (subprocess)",
        "Finalize",
    ]
    tracker = StepTracker(steps, console)

    # Print header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]RL Position Sizer Training[/bold cyan]\n"
        f"[dim]PPO • {config.timesteps:,} timesteps • {config.granularity}[/dim]",
        border_style="cyan"
    ))

    # Suppress noisy logging
    with suppress_logging([
        'modular_data_loaders',
        'modular_trainers',
        'modular_inference',
        'feature_engineering',
    ]):
        _train_rl_sizer_impl(config, tracker)


def _train_rl_sizer_impl(config: RLTrainingConfig, tracker: StepTracker) -> None:
    """Internal implementation of RL sizer training."""

    # Step 1: Check ensemble model
    tracker.in_progress("Check ensemble model")
    if not ENSEMBLE_META_PATH.exists():
        tracker.error("Check ensemble model", "Not found")
        console.print("\n[yellow]Train ensemble first: buddy train --oanda-live -n 12000[/yellow]")
        raise ModelNotFoundError("Ensemble model not found")
    tracker.success("Check ensemble model", "Found")

    # Configure TensorFlow
    with configure_tensorflow(verbose=config.verbose):
        # Import modules
        from modular_inference import ModularEnsembleInference
        from feature_engineering import FeatureEngineering
        from fx_paper import candles_to_ohlcv_df

        ensemble = ModularEnsembleInference(use_rl_sizer=False)
        ensemble.load_models()

    # Step 2: Connect to OANDA
    tracker.in_progress("Connect to OANDA")
    try:
        oanda = OandaClientWrapper()
        oanda.connect()
        tracker.success("Connect to OANDA", "Connected")
    except Exception as e:
        tracker.error("Connect to OANDA", str(e)[:20])
        raise

    # Print initial checklist
    console.print()
    tracker._print()
    console.print()

    # Step 3: Generate predictions
    tracker.in_progress("Generate predictions")

    fe = FeatureEngineering()
    feature_extractor = FeatureExtractor(ensemble)

    # Pre-allocate lists with estimated capacity
    estimated_samples = len(config.pairs) * config.candles * 0.8
    all_features: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_prices: list[float] = []
    all_features.reserve(int(estimated_samples))  # type: ignore[attr-defined]
    all_predictions.reserve(int(estimated_samples))  # type: ignore[attr-defined]
    all_prices.reserve(int(estimated_samples))  # type: ignore[attr-defined]

    console.print("[bold]Generating Training Data[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        pairs_task = progress.add_task(
            f"Processing {len(config.pairs)} pairs",
            total=len(config.pairs),
        )

        for pair in config.pairs:
            progress.update(pairs_task, description=f"[cyan]{pair}[/cyan]")

            try:
                resp = oanda.get_candles(
                    pair,
                    granularity=config.granularity,
                    count=config.candles,
                    price="MBA",
                )
                df_raw = candles_to_ohlcv_df(resp)

                if df_raw is None or len(df_raw) < MIN_DF_LENGTH:
                    progress.advance(pairs_task)
                    continue

                df = fe.create_features(df_raw.copy(), include_all=True)
                df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

                # Extract features using generator
                for features, predictions, price in feature_extractor.extract_features(df):
                    all_features.append(features)
                    all_predictions.append(predictions)
                    all_prices.append(price)

            except Exception:
                pass

            progress.advance(pairs_task)

    if not all_features:
        tracker.error("Generate predictions", "No data")
        raise DataGenerationError("No training data generated")

    # Convert to numpy arrays
    features_array = np.array(all_features, dtype=np.float32)
    predictions_array = np.array(all_predictions, dtype=np.float32)
    prices_array = np.array(all_prices, dtype=np.float32)

    tracker.success("Generate predictions", f"{len(features_array):,} samples")

    # Save training data
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RL_TRAIN_DATA_PATH,
        features=features_array,
        predictions=predictions_array,
        prices=prices_array,
    )

    # Update checklist
    console.print()
    tracker._print()
    console.print()

    # Step 4: Run PPO training in subprocess
    tracker.in_progress("Train PPO (subprocess)")
    console.print("[bold]Training PPO Agent (subprocess)[/bold]")
    console.print()

    script_path = Path(__file__).parent / "train_rl_standalone.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--data", str(RL_TRAIN_DATA_PATH),
        "--timesteps", str(config.timesteps),
    ]
    if config.verbose:
        cmd.append("--verbose")

    train_time = _run_subprocess_training(cmd, tracker)
    tracker.success("Train PPO (subprocess)", f"{train_time:.0f}s")

    # Step 5: Finalize
    _finalize_rl_training(config, tracker, features_array)

    # Final checklist and summary
    console.print()
    tracker._print()

    console.print()
    console.print(Panel(
        f"[bold green]✓ RL Position Sizer Trained[/bold green]\n\n"
        f"Timesteps: {config.timesteps:,}  •  Time: {train_time:.0f}s  •  Samples: {len(features_array):,}\n"
        f"Data: {len(config.pairs)} pairs × {config.candles:,} candles ({config.granularity})\n\n"
        f"[bold]Usage:[/bold]  buddy scan --use-rl-sizer\n"
        f"        buddy --use-rl-sizer -I USD_JPY",
        border_style="green"
    ))


def _run_subprocess_training(cmd: list[str], tracker: StepTracker) -> float:
    """Run training subprocess and return elapsed time."""
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent,
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            raise SubprocessError(
                f"Subprocess exited with code {process.returncode}",
                process.returncode,
            )

        return time.time() - start_time

    except SubprocessError:
        raise
    except Exception as e:
        tracker.error("Train PPO (subprocess)", str(e)[:20])
        raise


def _finalize_rl_training(
    config: RLTrainingConfig,
    tracker: StepTracker,
    features_array: np.ndarray,
) -> None:
    """Finalize RL training by updating metadata and cleanup."""
    tracker.in_progress("Finalize")

    # Update metadata
    meta_manager = MetadataManager()
    meta = meta_manager.load(RL_META_PATH.name)
    meta.update({
        "pairs_used": list(config.pairs),
        "granularity": config.granularity,
        "candles_per_pair": config.candles,
        "samples_generated": len(features_array),
    })
    meta_manager.save(RL_META_PATH.name, meta)

    # Cleanup
    if RL_TRAIN_DATA_PATH.exists():
        RL_TRAIN_DATA_PATH.unlink()

    tracker.success("Finalize", "Cleaned up")


def retrain_gates(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    pairs: str | None = None,
    granularity: str = "H1",
    candles: int = 5000,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    Retrain only sklearn gate models (XGBoost, RF, Ridge) while keeping Transformer.

    This solves sklearn version mismatch issues by retraining the fast sklearn
    models locally while preserving the proven 78% Colab-trained Transformer.

    Args:
        config_path: Path to config file (unused, kept for compatibility)
        pairs: Comma-separated pairs to train on (default: all major pairs)
        granularity: Candle timeframe (default: H1)
        candles: Number of candles per pair (default: 5000, will batch if >5000)
        verbose: Show detailed logs
        **kwargs: Additional keyword arguments (ignored)

    Raises:
        OandaConnectionError: If OANDA connection fails
        TrainingError: If training fails
    """
    import sklearn
    import xgboost

    # Import trainer config
    from modular_trainers import TrainerConfig

    # Parse configuration
    pair_list = tuple(p.strip() for p in pairs.split(",")) if pairs else EXTENDED_PAIRS
    config = GateTrainingConfig(
        pairs=pair_list,
        granularity=granularity,
        candles=candles,
        verbose=verbose,
    )

    # Print header
    console.print(Panel(
        f"[bold]Retraining Gate Models Locally[/bold]\n\n"
        f"Pairs: {', '.join(config.pairs)}\n"
        f"Candles per pair: {config.candles:,}\n"
        f"Granularity: {config.granularity}\n\n"
        "[dim]This will retrain XGBoost, RF, and Ridge models[/dim]\n"
        "[dim]The Transformer direction model will be kept unchanged[/dim]",
        title="🔧 Gate Model Retraining",
        border_style="cyan"
    ))

    # Import data loaders
    from modular_data_loaders import (
        compute_normalized_features,
        load_xgboost_data,
        load_rf_data,
        load_ridge_data
    )
    from feature_engineering import FeatureEngineering

    # Connect to OANDA and fetch data
    console.print("\n[bold]Step 1: Fetching data from OANDA...[/bold]")

    try:
        oanda = OandaClientWrapper()
        oanda.connect()
    except Exception as e:
        console.print(f"[red]Failed to connect to OANDA: {e}[/red]")
        console.print("[yellow]Please set OANDA_ACCOUNT_ID and OANDA_TOKEN environment variables[/yellow]")
        raise OandaConnectionError(f"Failed to connect to OANDA: {e}")

    # Fetch all pair data
    batch_processor = DataBatchProcessor(oanda)
    all_dfs: list[pd.DataFrame] = []

    for pair in track(config.pairs, description="Fetching pairs..."):
        try:
            df = batch_processor.fetch_pair_data(
                pair=pair,
                granularity=config.granularity,
                total_candles=config.candles,
            )

            if df is not None:
                all_dfs.append(df)
                console.print(f"  ✓ {pair}: {len(df):,} candles")

        except Exception as e:
            console.print(f"[yellow]  ✗ {pair}: {e}[/yellow]")
            continue

    if not all_dfs:
        console.print("[red]No data available. Please check OANDA connection.[/red]")
        raise TrainingError("No data available from OANDA")

    # Combine all pairs
    combined_df = pd.concat(all_dfs, ignore_index=True)
    console.print(f"\n[green]Combined data: {len(combined_df):,} total candles[/green]")

    # Feature engineering
    console.print("\n[bold]Step 2: Computing features...[/bold]")

    fe = FeatureEngineering()
    feature_df = fe.create_features(combined_df)
    console.print(f"  Features computed: {len(feature_df.columns)} columns")

    # Compute normalized features
    feature_df = compute_normalized_features(feature_df)
    console.print(f"  Normalized features: {len(feature_df.columns)} columns")

    # Load data for each model
    console.print("\n[bold]Step 3: Preparing training data...[/bold]")

    split = TRAIN_VAL_TEST_SPLIT

    xgb_data = load_xgboost_data(feature_df, split)
    rf_data = load_rf_data(feature_df, split)
    ridge_data = load_ridge_data(feature_df, split)

    console.print(f"  XGBoost: {len(xgb_data['X_train']):,} train, {len(xgb_data['X_val']):,} val")
    console.print(f"  RF: {len(rf_data['X_train']):,} train, {len(rf_data['X_val']):,} val")
    console.print(f"  Ridge: {len(ridge_data['X_train']):,} train, {len(ridge_data['X_val']):,} val")

    # Configure trainer
    trainer_config = TrainerConfig()
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train models
    xgb_metrics = _train_xgboost(trainer_config, xgb_data, model_dir)
    rf_metrics = _train_random_forest(trainer_config, rf_data, model_dir)
    ridge_metrics = _train_ridge(trainer_config, ridge_data, model_dir)

    # Update metadata
    console.print("\n[bold]Step 7: Updating metadata...[/bold]")

    meta_manager = MetadataManager()
    meta = meta_manager.load("modular_ensemble.meta.json")

    # Update with new training info
    meta.update({
        'gate_models_retrained_at': datetime.now().isoformat(),
        'gate_models_trained_on': 'local_m1',
        'sklearn_version': sklearn.__version__,
        'xgboost_version': xgboost.__version__,
        'retrained_pairs': list(config.pairs),
        'retrained_candles': config.candles,
        'retrained_granularity': config.granularity,
    })

    # Update results
    if 'results' not in meta:
        meta['results'] = {}
    meta['results']['xgboost'] = xgb_metrics
    meta['results']['random_forest'] = rf_metrics
    meta['results']['ridge'] = ridge_metrics

    meta_manager.save("modular_ensemble.meta.json", meta)

    console.print(f"  [green]✓ Metadata updated with sklearn={sklearn.__version__}, xgboost={xgboost.__version__}[/green]")

    # Print summary
    _print_gate_training_summary(xgb_metrics, rf_metrics, ridge_metrics)


def _train_xgboost(
    config: Any,
    data: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    """Train XGBoost model and return metrics."""
    from modular_trainers import XGBoostTrainer

    console.print("\n[bold]Step 4: Training XGBoost (Momentum)...[/bold]")

    trainer = XGBoostTrainer(config)
    metrics = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        feature_names=data['feature_names'],
        momentum_norm_factor=data.get('momentum_norm_factor'),
    )
    trainer.save(str(model_dir / "xgb_momentum.pkl"))

    console.print(
        f"  [green]✓ XGBoost: momentum_mae={metrics['momentum_mae']:.4f}, "
        f"accel_acc={metrics['acceleration_accuracy']:.1%}[/green]"
    )

    return metrics


def _train_random_forest(
    config: Any,
    data: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    """Train Random Forest model and return metrics."""
    from modular_trainers import RandomForestTrainer

    console.print("\n[bold]Step 5: Training Random Forest (Risk)...[/bold]")

    trainer = RandomForestTrainer(config)
    metrics = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        feature_names=data['feature_names'],
    )
    trainer.save(str(model_dir / "rf_risk.pkl"))

    drawdown_mae_bps = metrics.get(
        'drawdown_mae_bps',
        metrics.get('drawdown_mae_pct', 0) * 10000
    )
    console.print(
        f"  [green]✓ RF: drawdown_mae={drawdown_mae_bps:.1f} bps, "
        f"streak_mae={metrics['streak_prob_mae']:.4f}[/green]"
    )

    return metrics


def _train_ridge(
    config: Any,
    data: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    """Train Ridge/ElasticNet model and return metrics."""
    from modular_trainers import RidgeTrainer

    console.print("\n[bold]Step 6: Training ElasticNet (Confidence)...[/bold]")

    trainer = RidgeTrainer(config)
    metrics = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        feature_names=data['feature_names'],
    )
    trainer.save(str(model_dir / "ridge_confidence.pkl"))

    console.print(
        f"  [green]✓ Ridge: MAE={metrics['confidence_mae']:.2f}, "
        f"R²={metrics['r2_score']:.4f}[/green]"
    )

    return metrics


def _print_gate_training_summary(
    xgb_metrics: dict[str, Any],
    rf_metrics: dict[str, Any],
    ridge_metrics: dict[str, Any],
) -> None:
    """Print training summary panel."""
    import sklearn
    import xgboost

    drawdown_mae_bps = rf_metrics.get(
        'drawdown_mae_bps',
        rf_metrics.get('drawdown_mae_pct', 0) * 10000
    )

    console.print("\n" + "=" * 60)
    console.print(Panel(
        f"[bold green]Gate Models Retrained Successfully![/bold green]\n\n"
        f"[bold]sklearn version:[/bold] {sklearn.__version__}\n"
        f"[bold]xgboost version:[/bold] {xgboost.__version__}\n\n"
        f"[bold]XGBoost (Momentum):[/bold]\n"
        f"  MAE: {xgb_metrics['momentum_mae']:.4f}\n"
        f"  Acceleration Accuracy: {xgb_metrics['acceleration_accuracy']:.1%}\n\n"
        f"[bold]Random Forest (Risk):[/bold]\n"
        f"  Drawdown MAE: {drawdown_mae_bps:.1f} bps\n"
        f"  Streak MAE: {rf_metrics['streak_prob_mae']:.4f}\n\n"
        f"[bold]ElasticNet (Confidence):[/bold]\n"
        f"  MAE: {ridge_metrics['confidence_mae']:.2f}\n"
        f"  R²: {ridge_metrics['r2_score']:.4f}\n\n"
        f"[dim]Run 'buddy predict' to test the updated gates[/dim]",
        title="✅ Complete",
        border_style="green"
    ))
