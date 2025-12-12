"""Data processing and dataset utilities.

This module is the canonical implementation for:
- loading/caching market data
- feature normalization
- sequence creation
- PyTorch Dataset + optimized DataLoader helpers
- data augmentation

Historically there were separate legacy and optimized variants. Those have been
merged here to avoid duplication.
"""

import asyncio
import logging
import os
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yfinance as yf

try:
    import dask.array as da  # type: ignore
except Exception:  # pragma: no cover
    da = None

if TYPE_CHECKING:  # pragma: no cover
    import dask.array as dask_array  # type: ignore

    DaskArray = dask_array.Array
else:
    DaskArray = Any

logger = logging.getLogger(__name__)

# Define cache directory (env override supported)
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(Path.home() / ".trading_bot_cache")))
CACHE_DIR.mkdir(exist_ok=True, parents=True)

class StockDataset(Dataset):
    """
    PyTorch Dataset for generating sequences from stock market data.
    """

    @staticmethod
    def _is_dask_array(x: Any) -> bool:
        return da is not None and isinstance(x, da.Array)

    @classmethod
    def _coerce_to_array(cls, x: Union[list, np.ndarray, torch.Tensor, DaskArray]) -> Union[np.ndarray, DaskArray]:
        # Convert to numpy arrays for consistent handling (unless dask)
        if cls._is_dask_array(x):
            return x
        if isinstance(x, list):
            return np.array(x)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x  # type: ignore[return-value]
        if cls._is_dask_array(x):
            return x
        if isinstance(x, list):
            return np.array(x)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x  # type: ignore[return-value]

    @classmethod
    def _validate_shapes(cls, features: Any, targets: Any) -> None:
        if len(features) == 0 or len(targets) == 0:
            raise DataValidationError("Features and targets cannot be empty")

        if len(features) != len(targets):
            msg = (
                f"Features length ({len(features)}) must match "
                f"targets length ({len(targets)})"
            )
            raise DataValidationError(msg)

    @classmethod
    def _sanitize_values(cls, features: Any, targets: Any) -> Tuple[Any, Any]:
        # Check for NaN/inf values when materialized
        if cls._is_dask_array(features) or cls._is_dask_array(targets):
            return features, targets

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Features contain NaN or infinite values, replacing with zeros")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
            logger.warning("Targets contain NaN or infinite values, replacing with zeros")
            targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

        return features, targets

    @classmethod
    def _compute_length(cls, features: Any) -> int:
        # Pre-compute length for efficiency
        if cls._is_dask_array(features):
            return int(features.shape[0])
        return int(len(features))

    @classmethod
    def _maybe_preload(
        cls,
        features: Any,
        targets: Any,
        device: Optional[torch.device],
        length: int,
    ) -> Tuple[Any, Any, bool]:
        # Optional preload for small datasets
        if device is None or cls._is_dask_array(features):
            return features, targets, False
        if length >= 10000:
            return features, targets, False

        try:
            features_t = torch.as_tensor(features, dtype=torch.float32).to(device)
    def __init__(
        self,
        features: Union[list, np.ndarray, torch.Tensor, DaskArray],
        targets: Union[list, np.ndarray, torch.Tensor, DaskArray],
        sequence_length: int = 60,
        device: Optional[torch.device] = None,
    ):
    def __init__(
        self,
        features: Union[list, np.ndarray, torch.Tensor, "da.Array"],
        targets: Union[list, np.ndarray, torch.Tensor, "da.Array"],
        sequence_length: int = 60,
        device: Optional[torch.device] = None,
    ):
        # Validate inputs
        if features is None or targets is None:
            raise DataValidationError("Features and targets cannot be None")

        self.device = device
        self.preloaded = False

        features = self._coerce_to_array(features)
        targets = self._coerce_to_array(targets)

        self._validate_shapes(features, targets)
        features, targets = self._sanitize_values(features, targets)

        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

        self._length = self._compute_length(self.features)
        self.features, self.targets, self.preloaded = self._maybe_preload(
            self.features,
            self.targets,
            self.device,
            self._length,
        )

        logger.info("Initialized StockDataset with %s samples", len(self))

    def __len__(self) -> int:
        return self._length

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
            if self.preloaded:
                seq = self.features[idx]
                target_value = self.targets[idx]
                if target_value.dim() == 0:
                    target_value = target_value.unsqueeze(0)
                return seq, target_value

            if da is not None and isinstance(self.features, da.Array):
                seq = torch.as_tensor(self.features[idx].compute(), dtype=torch.float32)
                target_val = float(self.targets[idx].compute())
                target = torch.tensor([target_val], dtype=torch.float32)
            else:
                seq = torch.as_tensor(self.features[idx], dtype=torch.float32)
                target = torch.tensor([float(self.targets[idx])], dtype=torch.float32)

            if self.device is not None:
                seq = seq.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
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
    if data is None:
        raise DataValidationError("Data cannot be None")
    if not tickers:
        raise DataValidationError("Tickers list cannot be empty")

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


def normalize_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize selected columns and return (normalized_df, params)."""
    if df is None or df.empty:
        raise DataValidationError("DataFrame is None or empty")
    if not columns:
        raise DataValidationError("columns cannot be empty")

    method_l = (method or "standard").lower()
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    if method_l == "standard":
        scaler = StandardScaler()
    elif method_l == "minmax":
        scaler = MinMaxScaler()
    elif method_l == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    out = df.copy()
    missing = [c for c in columns if c not in out.columns]
    if missing:
        raise DataValidationError(f"Missing columns for normalization: {missing}")

    out[columns] = scaler.fit_transform(out[columns].values)
    params: Dict[str, Any] = {"method": method_l, "scaler": scaler, "columns": list(columns)}
    return out, params


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    target_column: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling window sequences and next-step targets."""
    if data is None:
        raise DataValidationError("data cannot be None")

    arr = np.asarray(data)
    if arr.size == 0:
        raise DataValidationError("data cannot be empty")
    if arr.ndim != 2:
        raise DataValidationError("data must be a 2D array")
    if sequence_length is None or sequence_length <= 0:
        raise DataValidationError("sequence_length must be > 0")
    if len(arr) <= sequence_length:
        raise DataValidationError("data is too short for the requested sequence_length")

    n_samples = len(arr) - sequence_length
    sequences = np.empty((n_samples, sequence_length, arr.shape[1]), dtype=float)
    targets = np.empty((n_samples,), dtype=float)

    for i in range(n_samples):
        sequences[i] = arr[i : i + sequence_length]
        targets[i] = arr[i + sequence_length, target_column]

    return sequences, targets


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    device: Optional[str] = None,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
) -> DataLoader:
    """Create a DataLoader configured for the available hardware."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if num_workers is None:
        if device == "cuda":
            num_workers = min(os.cpu_count() or 1, 4)
        else:
            num_workers = max((os.cpu_count() or 1) - 1, 1)

    if pin_memory is None:
        pin_memory = device == "cuda"

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        drop_last=shuffle,
    )


_REQUIRED_STOCK_COLS = {"open", "high", "low", "close", "volume"}


def _normalize_columns(columns) -> List[str]:
    out = []
    for col in columns:
        if isinstance(col, tuple):
            out.append("_".join(map(str, col)).lower())
        else:
            out.append(str(col).lower())
    return out


def _ensure_close_column(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]
    return df


def _normalize_and_validate_stock_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    out = df.copy()
    out.columns = _normalize_columns(out.columns)
    out = _ensure_close_column(out)
    if not _REQUIRED_STOCK_COLS.issubset(set(out.columns)):
        return None
    return out


async def _download_ticker_no_cache(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            None, partial(yf.download, ticker, start=start, end=end, progress=False)
        )
    except Exception as e:
        logger.warning("yf.download failed: %s", e)
        data = None

    normalized = _normalize_and_validate_stock_df(data)
    if normalized is not None:
        return normalized

    try:
        data2 = yf.Ticker(ticker).history(start=start, end=end)
    except Exception as e:
        logger.error("yf.Ticker().history failed: %s", e)
        return None

    return _normalize_and_validate_stock_df(data2)


async def _load_all_tickers_async(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool,
) -> Dict[str, pd.DataFrame]:
    coros = [
        async_cached_download(ticker, start_date, end_date) if use_cache else _download_ticker_no_cache(ticker, start_date, end_date)
        for ticker in tickers
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    out: Dict[str, pd.DataFrame] = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            logger.error("Error downloading %s: %s", ticker, result)
            continue
        if result is None or result.empty:
            logger.warning("No data for %s", ticker)
            continue
        out[ticker] = result
    return out


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ticker_df_to_frame(ticker: str, data: pd.DataFrame, features: List[str]) -> Optional[pd.DataFrame]:
    if data is None or data.empty:
        return None

    out = data.copy()
    out["ticker"] = ticker

    requested = [str(f).lower() for f in features]
    available = [f for f in requested if f in out.columns]
    if len(available) < len(requested):
        missing = set(requested) - set(available)
        logger.warning("Missing features for %s: %s", ticker, missing)

    out = out[available + ["ticker"]]
    return out.reset_index()


def load_stock_data_efficient(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load stock data with caching and parallel downloads."""
    if not tickers:
        raise DataValidationError("tickers cannot be empty")

    features = features or ["open", "high", "low", "close", "volume"]
    ticker_data = _run_async(_load_all_tickers_async(tickers, start_date, end_date, use_cache))
    if not ticker_data:
        logger.error("No data loaded for any ticker")
        return pd.DataFrame()

    frames = [
        frame
        for ticker, data in ticker_data.items()
        for frame in [_ticker_df_to_frame(ticker, data, features)]
        if frame is not None
    ]
    if not frames:
        logger.error("No valid data frames to combine")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int = 60,
    target_column: str = "close",
    feature_columns: Optional[List[str]] = None,
    target_shift: int = 1,
    scale_features: bool = True,
    scale_target: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare sequences for time-series prediction."""
    if df is None or df.empty:
        raise DataValidationError("df is None or empty")
    if sequence_length <= 0:
        raise DataValidationError("sequence_length must be > 0")
    if target_shift <= 0:
        raise DataValidationError("target_shift must be > 0")
    if target_column not in df.columns:
        raise DataValidationError(f"Missing target column: {target_column}")

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in {"ticker", "date", "timestamp"}]

    if not feature_columns:
        raise DataValidationError("feature_columns resolved to empty")

    feature_data = df[feature_columns].values
    target_data = df[target_column].values

    feature_scaler = None
    target_scaler = None

    if scale_features:
        from sklearn.preprocessing import RobustScaler, StandardScaler

        try:
            feature_scaler = RobustScaler()
            feature_data = feature_scaler.fit_transform(feature_data)
        except Exception as e:
            logger.warning("RobustScaler failed, falling back to StandardScaler: %s", e)
            feature_scaler = StandardScaler()
            feature_data = feature_scaler.fit_transform(feature_data)

    if scale_target:
        from sklearn.preprocessing import RobustScaler, StandardScaler

        try:
            target_scaler = RobustScaler()
            target_data = target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
        except Exception as e:
            logger.warning("RobustScaler failed for target, falling back to StandardScaler: %s", e)
            target_scaler = StandardScaler()
            target_data = target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()

    sequences: List[np.ndarray] = []
    targets: List[float] = []
    for i in range(len(feature_data) - sequence_length - target_shift + 1):
        sequences.append(feature_data[i : i + sequence_length])
        targets.append(float(target_data[i + sequence_length + target_shift - 1]))

    x = np.asarray(sequences)
    y = np.asarray(targets)

    metadata: Dict[str, Any] = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "sequence_length": sequence_length,
        "target_shift": target_shift,
        "n_sequences": int(len(x)),
        "feature_shape": tuple(x.shape),
        "target_shape": tuple(y.shape),
    }
    return x, y, metadata


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
    """Augment sequences/targets with noise, scaling, and optional mixup."""
    if sequences is None or targets is None:
        raise DataValidationError("sequences and targets cannot be None")

    seq = np.asarray(sequences)
    tgt = np.asarray(targets)
    if len(seq) == 0 or len(tgt) == 0:
        raise DataValidationError("sequences and targets cannot be empty")
    if len(seq) != len(tgt):
        raise DataValidationError("sequences and targets must have the same length")
    if augmentation_factor <= 0:
        raise DataValidationError("augmentation_factor must be > 0")

    rng = rng or np.random.default_rng(seed)

    augmented_sequences = [seq]
    augmented_targets = [tgt]

    for i in range(augmentation_factor - 1):
        if i % 3 == 0:
            noise = rng.normal(0, noise_level, size=seq.shape)
            augmented_sequences.append(seq + noise)
            augmented_targets.append(tgt)
        elif i % 3 == 1:
            scale = rng.uniform(0.95, 1.05, size=(seq.shape[0], 1, seq.shape[2]))
            augmented_sequences.append(seq * scale)
            augmented_targets.append(tgt)
        else:
            if not use_mixup:
                noise = rng.normal(0, noise_level, size=seq.shape)
                augmented_sequences.append(seq + noise)
                augmented_targets.append(tgt)
                continue

            indices = rng.permutation(len(seq))
            shuffled_sequences = seq[indices]
            shuffled_targets = tgt[indices]

            lam = rng.beta(mixup_alpha, mixup_alpha, size=(len(seq), 1, 1))
            lam_targets = rng.beta(mixup_alpha, mixup_alpha, size=len(seq))

            mixed_sequences = lam * seq + (1 - lam) * shuffled_sequences
            mixed_targets = lam_targets * tgt + (1 - lam_targets) * shuffled_targets

            augmented_sequences.append(mixed_sequences)
            augmented_targets.append(mixed_targets)

    final_sequences = np.concatenate(augmented_sequences, axis=0)
    final_targets = np.concatenate(augmented_targets, axis=0)
    logger.info(
        "Augmented data from %s to %s samples",
        len(seq),
        len(final_sequences),
    )
    return final_sequences, final_targets


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
