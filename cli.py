"""
Enhanced CLI for stock market prediction model with online
data fetching capabilities.
Provides a user-friendly interface for model training, prediction,
and analysis.
"""

import os
import logging
import json
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import asyncio
import torch

# Import optimized modules
from data_processing_optimized import load_stock_data_efficient, prepare_sequences
from ml_engine_enhanced import EnhancedMLEngine
from memory_manager_enhanced import MemoryManager, memory_efficient
from reasoning_enhanced import ReasoningEngine
from neural_network_integrator import NeuralNetworkIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("stock_prediction.log")],
)
logger = logging.getLogger(__name__)


class StockPredictionCLI:
    """
    Enhanced Command Line Interface for stock market prediction
    with online data fetching.
    """

    def __init__(self):
        """Initialize the CLI."""
        self.config = {}
        self.ml_engine = None
        self.mt_engine = None
        self.mr_engine = None
        self.reasoning_engine = None
        self.integrator = None
        self.memory_manager = None

        # Set default paths
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.output_dir = self.base_dir / "output"

        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        logger.info("StockPredictionCLI initialized")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            logger.info(f"Configuration loaded from {config_path}")
            self.config = config
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Use default configuration
            self.config = self._get_default_config()
            return self.config

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "data": {
                "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                "start_date": "2020-01-01",
                "end_date": "2023-01-01",
                "features": ["open", "high", "low", "close", "volume"],
                "target": "close",
                "sequence_length": 60,
                "validation_split": 0.2,
                "test_split": 0.1,
            },
            "ml_engine": {
                "model": {
                    "type": "ensemble",
                    "base_models": [
                        {"type": "lstm", "hidden_size": 128, "num_layers": 3},
                        {"type": "attention_lstm", "hidden_size": 128, "num_layers": 2},
                        {"type": "transformer", "hidden_size": 128, "num_layers": 2},
                    ],
                    "ensemble_method": "attention",
                },
                "optimizer": {
                    "type": "adamw",
                    "learning_rate": 0.001,
                    "weight_decay": 0.01,
                },
                "scheduler": {"type": "cosine", "t_0": 10, "t_mult": 2},
                "batch_size": 32,
                "epochs": 100,
                "early_stopping_patience": 10,
                "mixed_precision": True,
            },
            "mt_engine": {
                "model": {
                    "type": "transformer",
                    "hidden_size": 128,
                    "num_layers": 3,
                    "num_heads": 8,
                },
                "optimizer": {"type": "adamw", "learning_rate": 0.001},
            },
            "mr_engine": {
                "model": {
                    "type": "gru",
                    "hidden_size": 128,
                    "num_layers": 2,
                    "bidirectional": True,
                },
                "optimizer": {"type": "adam", "learning_rate": 0.001},
            },
            "integrator": {
                "use_attention": True,
                "use_dynamic_weights": True,
                "feature_dim": 64,
                "hidden_dim": 128,
                "num_heads": 4,
            },
            "reasoning": {"confidence_threshold": 0.7, "uncertainty_threshold": 0.3},
            "paths": {
                "data_dir": str(self.data_dir),
                "model_dir": str(self.model_dir),
                "output_dir": str(self.output_dir),
            },
            "device": ("cuda" if torch.cuda.is_available() else "cpu"),
            "num_workers": 4,
            "pin_memory": True,
        }

    def initialize_engines(self):
        """Initialize all engines based on configuration."""
        # Initialize memory manager
        device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        memory_config = self.config.get("memory", {})
        self.memory_manager = MemoryManager(
            device=device,
            threshold_mb=memory_config.get("threshold_mb", 1000),
            log_level=memory_config.get("log_level", "INFO"),
            proactive_cleanup=memory_config.get("proactive_cleanup", True),
        )

        # Initialize ML engine
        self.ml_engine = EnhancedMLEngine(self.config.get("ml_engine", {}))

        # Initialize MT engine
        self.mt_engine = EnhancedMLEngine(self.config.get("mt_engine", {}))

        # Initialize MR engine
        self.mr_engine = EnhancedMLEngine(self.config.get("mr_engine", {}))

        # Initialize reasoning engine
        reasoning_config = self.config.get("reasoning", {})
        self.reasoning_engine = ReasoningEngine(reasoning_config)

        # Initialize neural network integrator
        integrator_config = self.config.get("integrator", {})
        self.integrator = NeuralNetworkIntegrator(integrator_config)
        self.integrator.set_engines(
            ml_engine=self.ml_engine,
            mt_engine=self.mt_engine,
            mr_engine=self.mr_engine,
            reasoning_engine=self.reasoning_engine,
        )

        logger.info("All engines initialized")

    async def fetch_stock_data(
        self, tickers: List[str], start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data from online sources.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1d', '1wk', '1mo', etc.)

        Returns:
            DataFrame with stock data
        """
        logger.info(
            f"Fetching stock data for {tickers} from {start_date} to {end_date}"
        )

        try:
            # Use optimized data loading function
            data = load_stock_data_efficient(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )

            if data.empty:
                logger.error("Failed to fetch stock data")
                return pd.DataFrame()

            num_tickers = len(tickers)
            logger.info(f"Successfully fetched data for {num_tickers} tickers")
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return pd.DataFrame()

    async def fetch_market_indicators(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch market indicators from online sources.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with market indicators
        """
        logger.info(f"Fetching market indicators from {start_date} to {end_date}")

        # List of market indicators (ETFs and indices)
        indicators = ["^GSPC", "^DJI", "^IXIC", "^VIX", "SPY", "QQQ", "IWM"]

        try:
            # Use optimized data loading function
            data = load_stock_data_efficient(
                tickers=indicators,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )

            if data.empty:
                logger.error("Failed to fetch market indicators")
                return pd.DataFrame()

            num_indicators = len(indicators)
            logger.info(f"Successfully fetched {num_indicators} market indicators")
            return data
        except Exception as e:
            logger.error(f"Error fetching market indicators: {e}")
            return pd.DataFrame()

    async def fetch_economic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch economic data from online sources.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with economic data
        """
        logger.info(f"Fetching economic data from {start_date} to {end_date}")

        # For demonstration, we'll use some ETFs that track economic indicators
        economic_indicators = ["TLT", "GLD", "USO", "UUP"]

        try:
            # Use optimized data loading function
            data = load_stock_data_efficient(
                tickers=economic_indicators,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )

            if data.empty:
                logger.error("Failed to fetch economic data")
                return pd.DataFrame()

            logger.info("Successfully fetched economic data")
            return data
        except Exception as e:
            logger.error(f"Error fetching economic data: {e}")
            return pd.DataFrame()

    async def prepare_data_for_prediction(
        self,
        ticker: str,
        days_back: int = 60,
        include_market_data: bool = True,
        include_economic_data: bool = True,
    ) -> Dict[str, Any]:
        """
        Prepare data for prediction.

        Args:
            ticker: Ticker symbol
            days_back: Number of days of historical data to use
            include_market_data: Whether to include market indicators
            include_economic_data: Whether to include economic data

        Returns:
            Dictionary with prepared data
        """
        # Calculate date range
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.datetime.now() - datetime.timedelta(days=days_back + 10)
        ).strftime("%Y-%m-%d")

        # Fetch data concurrently
        tasks = [self.fetch_stock_data([ticker], start_date, end_date)]

        if include_market_data:
            tasks.append(self.fetch_market_indicators(start_date, end_date))

        if include_economic_data:
            tasks.append(self.fetch_economic_data(start_date, end_date))

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Extract results
        stock_data = results[0]
        market_data = results[1] if include_market_data else pd.DataFrame()
        if include_market_data and include_economic_data:
            economic_data = results[2]
        elif include_economic_data:
            economic_data = results[1]
        else:
            economic_data = pd.DataFrame()

        if stock_data.empty:
            logger.error(f"Failed to fetch stock data for {ticker}")
            return {}

        # Prepare sequences for each engine
        try:
            # Prepare ML engine features (technical indicators)
            ml_features, ml_targets, ml_metadata = prepare_sequences(
                df=stock_data[stock_data["ticker"] == ticker],
                sequence_length=self.config["data"]["sequence_length"],
                target_column=self.config["data"]["target"],
                feature_columns=self.config["data"]["features"],
                scale_features=True,
                scale_target=True,
            )

            # Prepare MT engine features (market indicators)
            mt_features = None
            mt_targets = None
            mt_metadata = None

            if not market_data.empty:
                mt_features, mt_targets, mt_metadata = prepare_sequences(
                    df=market_data,
                    sequence_length=self.config["data"]["sequence_length"],
                    target_column="close",
                    feature_columns=["open", "high", "low", "close", "volume"],
                    scale_features=True,
                    scale_target=True,
                )

            # Prepare MR engine features (economic data)
            mr_features = None
            mr_targets = None
            mr_metadata = None

            if not economic_data.empty:
                mr_features, mr_targets, mr_metadata = prepare_sequences(
                    df=economic_data,
                    sequence_length=self.config["data"]["sequence_length"],
                    target_column="close",
                    feature_columns=["open", "high", "low", "close", "volume"],
                    scale_features=True,
                    scale_target=True,
                )

            # Create dictionary with prepared data
            prepared_data = {
                "ticker": ticker,
                "ml_features": ml_features,
                "ml_targets": ml_targets,
                "ml_metadata": ml_metadata,
                "mt_features": mt_features,
                "mt_targets": mt_targets,
                "mt_metadata": mt_metadata,
                "mr_features": mr_features,
                "mr_targets": mr_targets,
                "mr_metadata": mr_metadata,
                "stock_data": stock_data,
                "market_data": market_data,
                "economic_data": economic_data,
            }

            logger.info(f"Data prepared for prediction: {ticker}")
            return prepared_data
        except Exception as e:
            logger.error(f"Error preparing data for prediction: {e}")
            return {}

    @memory_efficient(threshold_mb=1000)
    def train_model(self, config_path: str = None):
        """
        Train the stock prediction model.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)

        # Initialize all engines
