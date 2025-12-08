# preprocess.py
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from functools import lru_cache

# Define (or import) logger if needed.
logger = logging.getLogger(__name__)
CACHE_DIR = Path.home() / ".trading_bot_cache"
CACHE_DIR.mkdir(exist_ok=True)

def process_multiindex_data(data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    If data has a MultiIndex (from yfinance for multiple tickers),
    process and stack each ticker's DataFrame vertically.
    """
    all_data = []
    for ticker in tickers:
        if ticker in data.columns.levels[0]:
            df_ticker = data[ticker].copy()
            df_ticker.columns = df_ticker.columns.str.lower()
            df_ticker['ticker'] = ticker
            df_ticker.reset_index(inplace=True)
            all_data.append(df_ticker)
        else:
            logger.warning(f"Ticker {ticker} not found in data.")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

@lru_cache(maxsize=32)
def cached_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Cache downloaded data to avoid redundant API calls.
    Saves data as a Parquet file in the cache directory.
    """
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    data = yf.download(ticker, start=start, end=end, progress=False)
    data.to_parquet(cache_file)
    return data

def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features with column validation."""
    if df is None or df.empty:
        logger.error("Input DataFrame is None or empty")
        return None

    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return None

    # Convert columns to numeric and drop missing values.
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    if len(df) == 0:
        logger.error("No valid data rows after preprocessing")
        return None
    return df

def preprocess_data(data: pd.DataFrame) -> tuple:
    """
    Scale and convert DataFrame data into sequences.
    Returns (X, y) arrays or (None, None) on failure.
    """
    logger.info("Initial DataFrame info:")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")

    if isinstance(data.columns, pd.MultiIndex):
        # Flatten the MultiIndex columns.
        data.columns = ['_'.join(col).lower() for col in data.columns]
        logger.info("Flattened MultiIndex columns")
        tickers = list(set(re.findall(r'(?:open|high|low|close|volume)_([a-zA-Z]+)', ' '.join(data.columns))))
        if not tickers:
            logger.error("No ticker information found in column names")
            return None, None
        logger.info(f"Detected tickers: {tickers}")
        required_features = ['open', 'high', 'low', 'close', 'volume']
        expected_columns = [f"{feature}_{ticker}" for ticker in tickers for feature in required_features]
        missing_cols = set(expected_columns) - set(data.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None, None
        data = data[expected_columns]
        logger.info(f"Selected columns shape: {data.shape}")
        # Process only the first ticker for simplicity.
        ticker = tickers[0]
        ticker_cols = [f"{feature}_{ticker}" for feature in required_features]
        ticker_data = data[ticker_cols].copy()
        ticker_data.columns = required_features
    else:
        required_features = ['open', 'high', 'low', 'close', 'volume']
        ticker_data = data.copy()
        ticker_data.columns = ticker_data.columns.str.lower().str.strip()

    features = _create_features(ticker_data)
    if features is None:
        return None, None

    # Scale features using StandardScaler if needed (scaling may be done externally)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    # Create sequences.
    sequence_length = 60  # default or load from config
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, 3])  # Index 3 for 'close'
    if not X or not y:
        logger.error("No sequences created")
        return None, None
    return (np.array(X), np.array(y))

def _create_sequences(data: np.ndarray, sequence_length: int = 60) -> tuple:
    """Create sequences and targets from processed numeric data."""
    X, y = [], []
    if len(data) < sequence_length:
        logger.warning(f"Not enough data points (required: {sequence_length}, got: {len(data)})")
        return np.array(X), np.array(y)
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 3])
    logger.info(f"Created {len(X)} sequences")
    return np.array(X), np.array(y)
