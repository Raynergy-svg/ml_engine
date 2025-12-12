"""Data processing and dataset utilities with optimized CPU/GPU performance"""

import os
import logging
import asyncio
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import dask.array as da

# Configure logging
logger = logging.getLogger(__name__)

# Define cache directory
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(exist_ok=True, parents=True)


class StockDataset(Dataset):
    """PyTorch Dataset for generating sequences from stock market data with optimized memory usage."""

    def __init__(
        self, 
        features: Union[List, np.ndarray, da.Array], 
        targets: Union[List, np.ndarray, da.Array], 
        sequence_length: int = 60,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the StockDataset.
        
        Args:
            features: Input features (can be list, numpy array, or dask array)
            targets: Target values (can be list, numpy array, or dask array)
            sequence_length: Length of input sequences
            device: Optional device to pre-load data to (for small datasets)
        """
        # Validate inputs
        if len(features) == 0:
            raise ValueError("Features cannot be empty. Check your data loading pipeline.")
        
        if len(targets) == 0:
            raise ValueError("Targets cannot be empty. Check your data loading pipeline.")
            
        if len(features) != len(targets):
            raise ValueError(f"Features length ({len(features)}) must match targets length ({len(targets)})")
        
        # Store inputs
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.device = device
        
        # Pre-compute length for efficiency
        if isinstance(self.features, da.Array):
            self._length = self.features.shape[0]
        else:
            self._length = len(self.features)
            
        # For very small datasets, we can pre-load to device for faster access
        self.preloaded = False
        if device is not None and self._length < 10000:  # Only preload small datasets
            try:
                if isinstance(self.features, (list, np.ndarray)):
                    self.features = torch.FloatTensor(self.features).to(device)
                    self.targets = torch.FloatTensor(self.targets).to(device)
                    self.preloaded = True
                    logger.info(f"Pre-loaded dataset to {device}")
            except Exception as e:
                logger.warning(f"Failed to pre-load dataset to device: {e}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Get a sample from the dataset."""
        # If data is preloaded to device, return directly
        if self.preloaded:
            return self.features[idx], self.targets[idx]
            
        # Otherwise, convert to tensor on demand
        if isinstance(self.features, da.Array):
            # For Dask arrays, compute the specific slice
            seq = torch.FloatTensor(self.features[idx].compute())
            target = torch.FloatTensor([self.targets[idx].compute()])
        else:
            # For regular arrays/lists
            seq = torch.FloatTensor(self.features[idx])
            target = torch.FloatTensor([self.targets[idx]])
            
        # Move to device if specified
        if self.device is not None:
            seq = seq.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
        return seq, target


def create_optimized_dataloader(
    dataset: Dataset, 
    batch_size: int, 
    shuffle: bool = True,
    num_workers: int = None,
    pin_memory: bool = None,
    device: str = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = None
) -> DataLoader:
    """
    Create an optimized DataLoader based on available hardware.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (None for auto-detection)
        pin_memory: Whether to use pinned memory (None for auto-detection)
        device: Target device ('cuda', 'cpu', or None for auto-detection)
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep worker processes alive between epochs
        
    Returns:
        Optimized DataLoader
    """
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Auto-detect optimal number of workers if not specified
    if num_workers is None:
        if device == 'cuda':
            # For GPU, use CPU count but cap at 4 for diminishing returns
            num_workers = min(os.cpu_count() or 1, 4)
        else:
            # For CPU, use CPU count - 1 but at least 1
            num_workers = max((os.cpu_count() or 1) - 1, 1)
    
    # Auto-detect pin_memory if not specified
    if pin_memory is None:
        # Only use pin_memory with CUDA
        pin_memory = device == 'cuda'
    
    # Auto-detect persistent_workers if not specified
    if persistent_workers is None:
        # Only use persistent workers if we have workers
        persistent_workers = num_workers > 0
    
    # Create optimized DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Prefetch next batch while current batch is being processed
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        # Use persistent workers to avoid worker initialization overhead
        persistent_workers=persistent_workers,
        # Drop last incomplete batch for more efficient GPU utilization during training
        drop_last=shuffle,  # Only drop last when shuffling (likely training)
    )


def process_multiindex_data(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Stack ticker-specific DataFrames extracted from MultiIndex data.
    
    Args:
        data: MultiIndex DataFrame with ticker as the first level
        tickers: List of ticker symbols to extract
        
    Returns:
        Concatenated DataFrame with ticker-specific data
    """
    all_data = []
    for ticker in tickers:
        try:
            if (ticker,) in data.columns:
                df_ticker = data[ticker].copy()
                df_ticker.columns = df_ticker.columns.str.lower()
                df_ticker["ticker"] = ticker
                df_ticker.reset_index(inplace=True)
                all_data.append(df_ticker)
            else:
                logger.warning(f"Ticker {ticker} not found in data.")
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


_REQUIRED_STOCK_COLS = {"open", "high", "low", "close", "volume"}
_ADJ_CLOSE_COL = "adj close"


def _normalize_columns(columns) -> List[str]:
    """Normalize column names to lowercase and handle tuples."""
    new_cols = []
    for col in columns:
        if isinstance(col, tuple):
            new_cols.append("_".join(map(str, col)).lower())
        else:
            new_cols.append(str(col).lower())
    return new_cols


def _ensure_close_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'close' exists, mapping from adjusted close when needed."""
    if "close" not in df.columns and _ADJ_CLOSE_COL in df.columns:
        df["close"] = df[_ADJ_CLOSE_COL]
    return df


def _normalize_and_validate_stock_df(
    df: Optional[pd.DataFrame],
    required_cols: set,
) -> Optional[pd.DataFrame]:
    """Return a normalized DataFrame if valid, otherwise None."""
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = _normalize_columns(df.columns)
    df = _ensure_close_column(df)

    missing = required_cols - set(df.columns)
    if missing:
        return None

    return df


def _try_delete_file(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _read_cached_stock_df(cache_file: Path, required_cols: set) -> Optional[pd.DataFrame]:
    if not cache_file.exists():
        return None

    try:
        df = pd.read_parquet(cache_file, engine="pyarrow")
    except Exception as e:
        logger.warning(f"Failed to read cache file: {e}. Will download fresh data.")
        _try_delete_file(cache_file)
        return None

    normalized = _normalize_and_validate_stock_df(df, required_cols)
    if normalized is not None:
        return normalized

    logger.error("Cached data missing required columns. Deleting cache file.")
    _try_delete_file(cache_file)
    return None


async def _download_with_yfinance_download(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(yf.download, ticker, start=start, end=end, progress=False)
        )
    except Exception as e:
        logger.warning(f"yf.download failed: {e}")
        return None


def _download_with_yfinance_history(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        return yf.Ticker(ticker).history(start=start, end=end)
    except Exception as e:
        logger.error(f"yf.Ticker().history failed: {e}")
        return None


def _write_stock_cache(df: pd.DataFrame, cache_file: Path) -> None:
    try:
        df.to_parquet(cache_file)
    except Exception as e:
        logger.warning(f"Failed to save cache file: {e}")


@lru_cache(maxsize=32)
async def async_cached_download(
    ticker: str, start: str, end: str
) -> Optional[pd.DataFrame]:
    """
    Asynchronously download and cache stock data to avoid repeated API calls.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with stock data or None if download failed
    """
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
    required_cols = _REQUIRED_STOCK_COLS

    cached = _read_cached_stock_df(cache_file, required_cols)
    if cached is not None:
        return cached

    data = await _download_with_yfinance_download(ticker, start, end)
    normalized = _normalize_and_validate_stock_df(data, required_cols)
    if normalized is None:
        logger.warning("yf.download did not return valid data; trying yf.Ticker().history")
        normalized = _normalize_and_validate_stock_df(
            _download_with_yfinance_history(ticker, start, end),
            required_cols,
        )

    if normalized is None:
        logger.error("Downloaded data is empty/invalid after fallback.")
        return None

    _write_stock_cache(normalized, cache_file)
    return normalized


def _normalize_stock_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalize columns and ensure a 'close' column exists when possible."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = _normalize_columns(df.columns)
    return _ensure_close_column(df)


async def _download_ticker_no_cache(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download a single ticker without caching, with a fallback method."""
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            None, partial(yf.download, ticker, start=start, end=end, progress=False)
        )
        data = _normalize_stock_df(data)
        if data is not None and not data.empty:
            return data

        logger.warning(f"No data for {ticker} via yf.download, trying yf.Ticker().history")
        data = await loop.run_in_executor(None, partial(yf.Ticker(ticker).history, start=start, end=end))
        return _normalize_stock_df(data)
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None


async def _load_all_tickers_async(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool,
) -> Dict[str, pd.DataFrame]:
    """Load all tickers in parallel and return non-empty results."""
    coros = [
        async_cached_download(ticker, start_date, end_date) if use_cache
        else _download_ticker_no_cache(ticker, start_date, end_date)
        for ticker in tickers
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    ticker_data: Dict[str, pd.DataFrame] = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            logger.error(f"Error downloading {ticker}: {result}")
            continue
        if result is None or result.empty:
            logger.warning(f"No data for {ticker}")
            continue
        ticker_data[ticker] = result

    return ticker_data


def _run_async(coro):
    """Run a coroutine in a dedicated event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ticker_df_to_frame(ticker: str, data: pd.DataFrame, features: List[str]) -> Optional[pd.DataFrame]:
    """Convert a single ticker dataframe into the standardized output schema."""
    if data is None or data.empty:
        return None

    data = data.copy()
    data["ticker"] = ticker

    requested = [str(f).lower() for f in features]
    available = [f for f in requested if f in data.columns]
    if len(available) < len(requested):
        missing = set(requested) - set(available)
        logger.warning(f"Missing features for {ticker}: {missing}")

    data = data[available + ["ticker"]]
    return data.reset_index()


def load_stock_data_efficient(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    features: List[str] = None,
) -> pd.DataFrame:
    """
    Load stock data efficiently with caching and parallel downloads.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        use_cache: Whether to use cached data
        features: List of features to include (default: all)
        
    Returns:
        DataFrame with stock data
    """
    features = features or ["open", "high", "low", "close", "volume"]
    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True, parents=True)

    ticker_data = _run_async(_load_all_tickers_async(tickers, start_date, end_date, use_cache))
    if not ticker_data:
        logger.error("No data loaded for any ticker")
        return pd.DataFrame()

    dfs = [
        frame
        for ticker, data in ticker_data.items()
        for frame in [_ticker_df_to_frame(ticker, data, features)]
        if frame is not None
    ]
    if not dfs:
        logger.error("No valid data frames to combine")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int = 60,
    target_column: str = "close",
    feature_columns: List[str] = None,
    target_shift: int = 1,
    scale_features: bool = True,
    scale_target: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Prepare sequences for time series prediction with efficient processing.
    
    Args:
        df: DataFrame with time series data
        sequence_length: Length of input sequences
        target_column: Column to predict
        feature_columns: Columns to use as features
        target_shift: Number of steps to shift target (1 for next day prediction)
        scale_features: Whether to scale features
        scale_target: Whether to scale target
        
    Returns:
        Tuple of (features, targets, metadata)
        - features: numpy array of shape (n_sequences, sequence_length, n_features)
        - targets: numpy array of shape (n_sequences,)
        - metadata: dictionary with scaling information
    """
    # Set default feature columns if not provided
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in ['ticker', 'date', 'timestamp']]
    
    # Extract features and target
    feature_data = df[feature_columns].values
    target_data = df[target_column].values
    
    # Initialize scalers
    feature_scaler = None
    target_scaler = None
    
    # Scale features if requested - use RobustScaler by default for better outlier handling
    if scale_features:
        from sklearn.preprocessing import StandardScaler, RobustScaler
        # RobustScaler is more robust to outliers than StandardScaler
        # It uses median and IQR instead of mean and std
        try:
            feature_scaler = RobustScaler()
            feature_data = feature_scaler.fit_transform(feature_data)
            logger.info("Applied RobustScaler to features for better outlier handling")
        except Exception as e:
            logger.warning(f"RobustScaler failed, falling back to StandardScaler: {e}")
            feature_scaler = StandardScaler()
            feature_data = feature_scaler.fit_transform(feature_data)
    
    # Scale target if requested
    if scale_target:
        from sklearn.preprocessing import StandardScaler, RobustScaler
        try:
            target_scaler = RobustScaler()
            target_data = target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
            logger.info("Applied RobustScaler to target for better outlier handling")
        except Exception as e:
            logger.warning(f"RobustScaler failed for target, falling back to StandardScaler: {e}")
            target_scaler = StandardScaler()
            target_data = target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
    
    # Prepare sequences
    sequences = []
    targets = []
    
    for i in range(len(feature_data) - sequence_length - target_shift + 1):
        # Extract sequence
        seq = feature_data[i:i + sequence_length]
        # Extract target (shifted by target_shift)
        tgt = target_data[i + sequence_length + target_shift - 1]
        
        sequences.append(seq)
        targets.append(tgt)
    
    # Convert to numpy arrays
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Log statistics for verification (PyTorch best practice)
    logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")
    logger.info(f"Feature range: [{sequences.min():.4f}, {sequences.max():.4f}]")
    logger.info(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
    
    # Prepare metadata
    metadata = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_columns': feature_columns,
        'target_column': target_column,
        'sequence_length': sequence_length,
        'target_shift': target_shift,
        'n_sequences': len(sequences),
        'feature_shape': sequences.shape,
        'target_shape': targets.shape
    }
    
    return sequences, targets, metadata


def add_technical_indicators(
    df: pd.DataFrame,
    price_column: str = "close",
    volume_column: str = "volume"
) -> pd.DataFrame:
    """
    Add technical indicators for improved prediction (PyTorch best practice for feature engineering).
    
    Args:
        df: Input dataframe with OHLCV data
        price_column: Name of price column
        volume_column: Name of volume column
        
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Moving averages (trend indicators)
    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = df[price_column].rolling(window=window).mean()
        df[f'ema_{window}'] = df[price_column].ewm(span=window, adjust=False).mean()
    
    # Volatility indicators with improved calculations
    df['returns'] = df[price_column].pct_change()
    df['log_returns'] = np.log(df[price_column] / df[price_column].shift(1))
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    
    # Parkinson's volatility (more accurate using high-low range)
    if 'high' in df.columns and 'low' in df.columns:
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            np.log(df['high'] / df['low']) ** 2
        ).rolling(window=20).mean()
    
    # Momentum indicators with improvements
    df['rsi_14'] = compute_rsi(df[price_column], period=14)
    df['rsi_7'] = compute_rsi(df[price_column], period=7)  # Shorter period for quick signals
    df['momentum_10'] = df[price_column] - df[price_column].shift(10)
    df['momentum_5'] = df[price_column] - df[price_column].shift(5)
    
    # Rate of change
    df['roc_10'] = ((df[price_column] - df[price_column].shift(10)) / 
                    df[price_column].shift(10)) * 100
    
    # Volume indicators
    if volume_column in df.columns:
        df['volume_ma_20'] = df[volume_column].rolling(window=20).mean()
        df['volume_ratio'] = df[volume_column] / df['volume_ma_20']
    
    # MACD
    ema_12 = df[price_column].ewm(span=12, adjust=False).mean()
    ema_26 = df[price_column].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df[price_column].rolling(window=20).mean()
    bb_std = df[price_column].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Price position within bands
    df['bb_position'] = (df[price_column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Drop NaN values created by indicators
    df = df.dropna()
    
    logger.info(f"Added {len(df.columns)} technical indicators")
    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def augment_data(
    sequences: np.ndarray,
    targets: np.ndarray,
    noise_level: float = 0.01,
    augmentation_factor: int = 2,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    rng: Optional[np.random.Generator] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with multiple strategies (PyTorch best practice for robustness).
    
    Args:
        sequences: Input sequences
        targets: Target values
        noise_level: Standard deviation of Gaussian noise
        augmentation_factor: Number of augmented copies per original sample
        use_mixup: Whether to apply mixup augmentation
        mixup_alpha: Alpha parameter for Beta distribution in mixup
        rng: Optional numpy random Generator for reproducible augmentation
        seed: Seed used when creating a default RNG (only used if rng is not provided)
        
    Returns:
        Augmented sequences and targets
    """
    rng = rng or np.random.default_rng(seed)

    augmented_sequences = [sequences]
    augmented_targets = [targets]
    
    for i in range(augmentation_factor - 1):
        # Strategy 1: Add Gaussian noise
        if i % 3 == 0:
            noise = rng.normal(0, noise_level, size=sequences.shape)
            noisy_sequences = sequences + noise
            augmented_sequences.append(noisy_sequences)
            augmented_targets.append(targets)
        
        # Strategy 2: Scale perturbation (random scaling of features)
        elif i % 3 == 1:
            scale = rng.uniform(0.95, 1.05, size=(sequences.shape[0], 1, sequences.shape[2]))
            scaled_sequences = sequences * scale
            augmented_sequences.append(scaled_sequences)
            augmented_targets.append(targets)
        
        # Strategy 3: Mixup augmentation for better generalization
        elif i % 3 == 2 and use_mixup:
            # Randomly shuffle indices
            indices = rng.permutation(len(sequences))
            shuffled_sequences = sequences[indices]
            shuffled_targets = targets[indices]
            
            # Sample lambda from Beta distribution
            lam = rng.beta(mixup_alpha, mixup_alpha, size=(len(sequences), 1, 1))
            # Properly handle dimensions for mixing
            lam_targets = rng.beta(mixup_alpha, mixup_alpha, size=len(sequences))
            
            # Create mixed samples
            mixed_sequences = lam * sequences + (1 - lam) * shuffled_sequences
            mixed_targets = lam_targets * targets + (1 - lam_targets) * shuffled_targets
            
            augmented_sequences.append(mixed_sequences)
            augmented_targets.append(mixed_targets)
    
    # Concatenate all augmented data
    final_sequences = np.concatenate(augmented_sequences, axis=0)
    final_targets = np.concatenate(augmented_targets, axis=0)
    
    logger.info(f"Augmented data from {len(sequences)} to {len(final_sequences)} samples using multiple strategies")
    
    return final_sequences, final_targets
