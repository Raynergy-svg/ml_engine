"""
Data processing and dataset utilities with improved error handling
and validation.
"""

from functools import lru_cache, partial
from pathlib import Path
import asyncio
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import yfinance as yf
from typing import List, Tuple, Optional, Union

logger = logging.getLogger(__name__)
CACHE_DIR = Path.home() / ".trading_bot_cache"
CACHE_DIR.mkdir(exist_ok=True)


class DataValidationError(Exception):
    """Custom exception for data validation errors"""

    pass


class StockDataset(Dataset):
    """
    PyTorch Dataset for generating sequences from stock market data.
    """

    def __init__(
        self,
        features: Union[list, np.ndarray, torch.Tensor],
        targets: Union[list, np.ndarray, torch.Tensor],
        sequence_length: int = 60,
    ):
        # Validate inputs
        if features is None or targets is None:
            raise DataValidationError("Features and targets cannot be None")

        # Convert to numpy arrays for consistent handling
        if isinstance(features, list):
            features = np.array(features)
        elif isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if isinstance(targets, list):
            targets = np.array(targets)
        elif isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # Validate shapes
        if len(features) == 0 or len(targets) == 0:
            raise DataValidationError("Features and targets cannot be empty")

        if len(features) != len(targets):
            msg = (
                f"Features length ({len(features)}) must match "
                f"targets length ({len(targets)})"
            )
            raise DataValidationError(msg)

        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            msg = "Features contain NaN or infinite values, replacing with zeros"
            logger.warning(msg)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
            msg = "Targets contain NaN or infinite values, replacing with zeros"
            logger.warning(msg)
            targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

        logger.info(f"Initialized StockDataset with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Get a sample from the dataset

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (features_tensor, target_tensor)

        Raises:
            IndexError: If index is out of bounds
        """
        if idx < 0 or idx >= len(self):
            msg = f"Index {idx} out of bounds for dataset of size {len(self)}"
            raise IndexError(msg)

        try:
            seq = torch.FloatTensor(self.features[idx])
            target = torch.FloatTensor([self.targets[idx]])
            return seq, target
        except Exception as e:
            logger.error(f"Error retrieving sample at index {idx}: {e}")
            raise


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame contains required columns and has valid data

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return False

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"DataFrame missing required columns: {missing_cols}")
        return False

    # Check for excessive NaN values
    for col in required_columns:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio > 0.5:
            logger.warning(f"Column '{col}' has {nan_ratio:.2%} NaN values")

    return True


def process_multiindex_data(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Stack ticker-specific DataFrames extracted from MultiIndex data.
    """
    all_data = []
    for ticker in tickers:
        if (ticker,) in data.columns:
            df_ticker = data[ticker].copy()
            df_ticker.columns = df_ticker.columns.str.lower()
            df_ticker["ticker"] = ticker
            df_ticker.reset_index(inplace=True)
            all_data.append(df_ticker)
        else:
            logger.warning(f"Ticker {ticker} not found in data.")
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
    else:
        result = pd.DataFrame()
    return result


@lru_cache(maxsize=32)
async def async_cached_download(
    ticker: str, start: str, end: str
) -> Optional[pd.DataFrame]:
    """
    Asynchronously download and cache stock data
    to avoid repeated API calls.
    """
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
    required_cols = {"open", "high", "low", "close", "volume"}

    def fix_columns(columns):
        """Normalize column names from MultiIndex or regular columns"""
        new_cols = []
        for col in columns:
            if isinstance(col, tuple):
                new_cols.append("_".join(map(str, col)).lower())
            else:
                new_cols.append(str(col).lower())
        return new_cols

    # Try to load from cache
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file, engine="pyarrow")
            df.columns = fix_columns(df.columns)
            if "close" not in df.columns and "adj close" in df.columns:
                df["close"] = df["adj close"]
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                msg = (
                    f"Cached data missing required columns: {missing_cols}. "
                    f"Deleting cache file."
                )
                logger.error(msg)
                cache_file.unlink()
            else:
                return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            cache_file.unlink(missing_ok=True)

    # Download using yf.download
    try:
        loop = asyncio.get_event_loop()
        download_fn = partial(yf.download, ticker, start=start, end=end, progress=False)
        data = await loop.run_in_executor(None, download_fn)
    except Exception as e:
        logger.warning(f"yf.download failed: {e}")
        data = None

    # If download failed or the data is missing columns,
    # try yf.Ticker().history
    has_required = (
        data is not None
        and not data.empty
        and required_cols.issubset(set(fix_columns(data.columns)))
    )
    if not has_required:
        msg = "yf.download did not return valid data; trying yf.Ticker().history"
        logger.warning(msg)
        data = yf.Ticker(ticker).history(start=start, end=end)

    if data is None or data.empty:
        logger.error("Downloaded data is empty or None after fallback.")
        return None

    data.columns = fix_columns(data.columns)
    if "close" not in data.columns and "adj close" in data.columns:
        data["close"] = data["adj close"]

    if not required_cols.issubset(set(data.columns)):
        missing = required_cols - set(data.columns)
        msg = f"Downloaded data missing required columns: {missing}"
        logger.error(msg)
        return None

    data.to_parquet(cache_file)
    return data
