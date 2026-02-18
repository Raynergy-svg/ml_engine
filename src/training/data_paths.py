"""
Centralized Data Path Configuration for Training Pipeline.

This module provides a single source of truth for all data paths used in the
training pipeline, eliminating path conflicts and ensuring consistent data
discovery across the codebase.

Key Features:
1. Centralized path definitions for market data, features, and models
2. Automatic path resolution with fallback strategies
3. Support for multiple data formats (CSV, NPZ, Parquet)
4. Logging of all path access attempts for debugging

Usage:
    from src.training.data_paths import DataPathResolver

    resolver = DataPathResolver(pair="EUR_USD")
    data_path = resolver.find_latest_market_data(granularity="H1")
    features_path = resolver.find_features_file()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DIRECTORY CONSTANTS
# =============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
MARKET_DATA_DIR = PROJECT_ROOT / "market_data"
TRAINED_DATA_DIR = PROJECT_ROOT / "trained_data"
CONFIG_DIR = PROJECT_ROOT / "config"

# Subdirectories
FEATURES_DIR = TRAINED_DATA_DIR / "features"
MODELS_DIR = TRAINED_DATA_DIR / "models"
REPLAY_DIR = TRAINED_DATA_DIR / "replay"
OOS_DIR = TRAINED_DATA_DIR  # Out-of-sample data stored at root level

# Data source priorities (in order of preference)
DATA_SOURCE_PRIORITIES = [
    "trained_data/features",  # Processed features (preferred)
    "market_data",            # Raw market data (fallback)
    "trained_data/replay",    # Replay buffers (alternative)
]


# =============================================================================
# SUPPORTED TRADING PAIRS
# =============================================================================

SUPPORTED_PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "GBP_JPY",
    "AUD_USD",
    "USD_CHF",
    "NZD_USD",
    "USD_CAD",
    "EUR_GBP",
    "EUR_CHF",
    "EUR_AUD",
    "AUD_JPY",
]

SUPPORTED_GRANULARITIES = ["M5", "M15", "H1", "H4", "D"]


# =============================================================================
# PATH RESOLVER
# =============================================================================

@dataclass
class DataPathResolver:
    """
    Resolves data paths for a specific trading pair.

    This class provides methods to find and validate data files for training,
    including market data, processed features, and replay buffers.

    Attributes:
        pair: Trading pair identifier (e.g., "EUR_USD")
        granularity: Candle granularity (default: "H1")
        ensure_dirs: Whether to create directories on init (default: False)

    Example:
        >>> resolver = DataPathResolver("EUR_USD", granularity="H1")
        >>> path = resolver.find_latest_market_data()
        >>> print(f"Found data at: {path}")
    """

    pair: str
    granularity: str = "H1"
    ensure_dirs: bool = False

    # Internal cache
    _cached_paths: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Validate pair and granularity after initialization."""
        self.pair = self.pair.upper()
        if self.pair not in SUPPORTED_PAIRS:
            logger.warning(
                f"Pair {self.pair} not in supported pairs list. "
                f"Supported: {SUPPORTED_PAIRS}"
            )

        if self.granularity not in SUPPORTED_GRANULARITIES:
            logger.warning(
                f"Granularity {self.granularity} not in supported list. "
                f"Supported: {SUPPORTED_GRANULARITIES}"
            )

        if self.ensure_dirs:
            self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [FEATURES_DIR, MODELS_DIR, REPLAY_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Market Data Methods
    # -------------------------------------------------------------------------

    def find_latest_market_data(
        self,
        data_type: str = "live",
        granularity: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Find the most recent market data file for this pair.

        Args:
            data_type: Type of data ("live" or "oos" for out-of-sample)
            granularity: Override granularity (default: uses instance granularity)

        Returns:
            Path to the latest data file, or None if not found
        """
        granularity = granularity or self.granularity
        pattern = f"oanda_{self.pair}_{granularity}_{data_type}_*.csv"

        logger.debug(f"Searching for market data with pattern: {pattern}")
        logger.debug(f"Search directory: {MARKET_DATA_DIR}")

        if not MARKET_DATA_DIR.exists():
            logger.warning(f"Market data directory does not exist: {MARKET_DATA_DIR}")
            return None

        # Find all matching files
        matching_files = list(MARKET_DATA_DIR.glob(pattern))

        if not matching_files:
            # Try without data_type suffix (some files don't have it)
            pattern_alt = f"oanda_{self.pair}_{granularity}_*.csv"
            matching_files = list(MARKET_DATA_DIR.glob(pattern_alt))
            # Filter out oos files if we want live
            if data_type == "live":
                matching_files = [f for f in matching_files if "_oos_" not in f.name]

        if not matching_files:
            logger.warning(
                f"No market data found for {self.pair} {granularity} {data_type}"
            )
            return None

        # Sort by modification time, return most recent
        matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_file = matching_files[0]

        logger.info(
            f"Found latest market data for {self.pair}: {latest_file.name} "
            f"({len(matching_files)} total files available)"
        )

        return latest_file

    def find_all_market_data(
        self,
        data_type: str = "live",
        granularity: Optional[str] = None,
    ) -> List[Path]:
        """
        Find all market data files for this pair.

        Args:
            data_type: Type of data ("live" or "oos")
            granularity: Override granularity

        Returns:
            List of paths to data files, sorted by modification time (newest first)
        """
        granularity = granularity or self.granularity
        pattern = f"oanda_{self.pair}_{granularity}_{data_type}_*.csv"

        if not MARKET_DATA_DIR.exists():
            return []

        matching_files = list(MARKET_DATA_DIR.glob(pattern))
        matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return matching_files

    # -------------------------------------------------------------------------
    # Features Methods
    # -------------------------------------------------------------------------

    def find_features_file(self) -> Optional[Path]:
        """
        Find the processed features file for this pair.

        Searches in order of preference:
        1. trained_data/features/{pair}_features.csv
        2. trained_data/features/{pair}_features.parquet

        Returns:
            Path to features file, or None if not found
        """
        if not FEATURES_DIR.exists():
            logger.debug(f"Features directory does not exist: {FEATURES_DIR}")
            return None

        # Check for CSV features
        csv_path = FEATURES_DIR / f"{self.pair}_features.csv"
        if csv_path.exists():
            logger.info(f"Found CSV features for {self.pair}: {csv_path}")
            return csv_path

        # Check for Parquet features
        parquet_path = FEATURES_DIR / f"{self.pair}_features.parquet"
        if parquet_path.exists():
            logger.info(f"Found Parquet features for {self.pair}: {parquet_path}")
            return parquet_path

        logger.debug(f"No features file found for {self.pair} in {FEATURES_DIR}")
        return None

    def get_features_path(self, create_dir: bool = True) -> Path:
        """
        Get the path where features should be saved.

        Args:
            create_dir: Whether to create the directory if it doesn't exist

        Returns:
            Path where features should be saved
        """
        if create_dir:
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        return FEATURES_DIR / f"{self.pair}_features.csv"

    # -------------------------------------------------------------------------
    # Replay Buffer Methods
    # -------------------------------------------------------------------------

    def find_replay_buffer(self) -> Optional[Path]:
        """
        Find the replay buffer file for this pair.

        Returns:
            Path to replay buffer (buffer.npz), or None if not found
        """
        replay_pair_dir = REPLAY_DIR / self.pair

        if not replay_pair_dir.exists():
            logger.debug(f"Replay directory does not exist: {replay_pair_dir}")
            return None

        buffer_path = replay_pair_dir / "buffer.npz"
        if buffer_path.exists():
            logger.info(f"Found replay buffer for {self.pair}: {buffer_path}")
            return buffer_path

        logger.debug(f"No replay buffer found for {self.pair}")
        return None

    # -------------------------------------------------------------------------
    # Model Methods
    # -------------------------------------------------------------------------

    def find_latest_model(self, model_type: str = "transformer") -> Optional[Path]:
        """
        Find the latest trained model for this pair.

        Args:
            model_type: Type of model ("transformer", "lgbm", "xgboost")

        Returns:
            Path to model directory, or None if not found
        """
        model_dir = MODELS_DIR / self.pair

        if not model_dir.exists():
            logger.debug(f"Model directory does not exist: {model_dir}")
            return None

        # Look for model files
        patterns = {
            "transformer": "*.keras",
            "lgbm": "*.txt",
            "xgboost": "*.json",
        }

        pattern = patterns.get(model_type, "*.keras")
        model_files = list(model_dir.glob(pattern))

        if not model_files:
            logger.debug(f"No {model_type} models found for {self.pair}")
            return None

        # Return the directory (models may have multiple files)
        logger.info(f"Found {model_type} model for {self.pair}: {model_dir}")
        return model_dir

    def get_model_dir(self, create_dir: bool = True) -> Path:
        """
        Get the directory where models should be saved.

        Args:
            create_dir: Whether to create the directory if it doesn't exist

        Returns:
            Path to model directory
        """
        model_dir = MODELS_DIR / self.pair

        if create_dir:
            model_dir.mkdir(parents=True, exist_ok=True)

        return model_dir

    # -------------------------------------------------------------------------
    # Primary Data Loading Method
    # -------------------------------------------------------------------------

    def find_training_data(self) -> Tuple[Optional[Path], str]:
        """
        Find the best available training data source.

        This method implements the fallback strategy for finding training data:
        1. First, try processed features (preferred)
        2. Then, try raw market data
        3. Finally, try replay buffers

        Returns:
            Tuple of (path, source_type) where source_type is one of:
            - "features": Processed features CSV/Parquet
            - "market_data": Raw OANDA market data
            - "replay": Replay buffer NPZ
            - (None, "none"): No data found
        """
        logger.info(f"Searching for training data for {self.pair}...")

        # 1. Try processed features first
        features_path = self.find_features_file()
        if features_path:
            logger.info(
                f"Using processed features for {self.pair} training: {features_path}"
            )
            return features_path, "features"

        # 2. Try raw market data
        market_path = self.find_latest_market_data()
        if market_path:
            logger.info(
                f"Using raw market data for {self.pair} training: {market_path}"
            )
            return market_path, "market_data"

        # 3. Try replay buffer
        replay_path = self.find_replay_buffer()
        if replay_path:
            logger.info(
                f"Using replay buffer for {self.pair} training: {replay_path}"
            )
            return replay_path, "replay"

        # No data found
        logger.error(
            f"No training data found for {self.pair} in any location. "
            f"Searched: features ({FEATURES_DIR}), market_data ({MARKET_DATA_DIR}), "
            f"replay ({REPLAY_DIR / self.pair})"
        )
        return None, "none"

    def list_available_data(self) -> dict:
        """
        List all available data for this pair.

        Returns:
            Dictionary with available data sources and their paths
        """
        available = {
            "pair": self.pair,
            "granularity": self.granularity,
            "market_data": [],
            "features": None,
            "replay_buffer": None,
            "models": [],
        }

        # Market data
        for data_type in ["live", "oos"]:
            files = self.find_all_market_data(data_type=data_type)
            for f in files:
                available["market_data"].append({
                    "path": str(f),
                    "type": data_type,
                    "size_mb": f.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })

        # Features
        features = self.find_features_file()
        if features:
            available["features"] = {
                "path": str(features),
                "size_mb": features.stat().st_size / (1024 * 1024),
            }

        # Replay buffer
        replay = self.find_replay_buffer()
        if replay:
            available["replay_buffer"] = {
                "path": str(replay),
                "size_mb": replay.stat().st_size / (1024 * 1024),
            }

        # Models
        for model_type in ["transformer", "lgbm", "xgboost"]:
            model_dir = self.find_latest_model(model_type)
            if model_dir:
                available["models"].append({
                    "type": model_type,
                    "path": str(model_dir),
                })

        return available


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_data_resolver(pair: str, granularity: str = "H1") -> DataPathResolver:
    """
    Get a DataPathResolver for a trading pair.

    Args:
        pair: Trading pair identifier
        granularity: Candle granularity

    Returns:
        DataPathResolver instance
    """
    return DataPathResolver(pair=pair, granularity=granularity)


def find_pair_data(pair: str, granularity: str = "H1") -> Tuple[Optional[Path], str]:
    """
    Find training data for a trading pair.

    This is a convenience function that creates a resolver and finds data.

    Args:
        pair: Trading pair identifier
        granularity: Candle granularity

    Returns:
        Tuple of (path, source_type)
    """
    resolver = DataPathResolver(pair=pair, granularity=granularity)
    return resolver.find_training_data()


def ensure_data_directories() -> None:
    """
    Create all necessary data directories.

    This ensures the following directories exist:
    - trained_data/features
    - trained_data/models
    - trained_data/replay
    """
    for dir_path in [FEATURES_DIR, MODELS_DIR, REPLAY_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def diagnose_data_availability(pair: str = "EUR_USD") -> dict:
    """
    Diagnose data availability for a trading pair.

    This function provides detailed information about what data is available
    and where it's located, useful for debugging "No training data found" issues.

    Args:
        pair: Trading pair to diagnose

    Returns:
        Dictionary with diagnostic information
    """
    resolver = DataPathResolver(pair=pair)
    available = resolver.list_available_data()

    # Add diagnostic info
    diagnostic = {
        "timestamp": datetime.now().isoformat(),
        "directories": {
            "market_data": {
                "path": str(MARKET_DATA_DIR),
                "exists": MARKET_DATA_DIR.exists(),
            },
            "features": {
                "path": str(FEATURES_DIR),
                "exists": FEATURES_DIR.exists(),
            },
            "models": {
                "path": str(MODELS_DIR),
                "exists": MODELS_DIR.exists(),
            },
            "replay": {
                "path": str(REPLAY_DIR),
                "exists": REPLAY_DIR.exists(),
            },
        },
        "available_data": available,
        "recommendations": [],
    }

    # Add recommendations
    if not available["market_data"] and not available["features"]:
        diagnostic["recommendations"].append(
            "No market data or features found. Run data download command first."
        )

    if not FEATURES_DIR.exists():
        diagnostic["recommendations"].append(
            f"Features directory doesn't exist. Create it at: {FEATURES_DIR}"
        )

    if available["market_data"] and not available["features"]:
        diagnostic["recommendations"].append(
            "Market data exists but no processed features. "
            "Consider running feature engineering pipeline."
        )

    return diagnostic


if __name__ == "__main__":
    # Example usage and diagnostic
    import json

    print("=" * 60)
    print("Data Path Configuration Diagnostic")
    print("=" * 60)

    for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
        print(f"\n{pair}:")
        diagnostic = diagnose_data_availability(pair)
        print(json.dumps(diagnostic, indent=2))
