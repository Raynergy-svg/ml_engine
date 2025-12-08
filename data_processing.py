"""Data processing and dataset utilities"""

from functools import lru_cache
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging
from typing import List, Tuple, Optional


class StockDataset(Dataset):
    """PyTorch Dataset for generating sequences from stock market data."""

    def __init__(self, features: list, targets: list, sequence_length: int = 60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        if isinstance(self.features, da.Array):
            return self.features.shape[0]
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        seq = torch.FloatTensor(self.features[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return seq, target


def process_multiindex_data(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Stack ticker-specific DataFrames extracted from MultiIndex data."""
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
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


@lru_cache(maxsize=32)
async def async_cached_download(
    ticker: str, start: str, end: str
) -> Optional[pd.DataFrame]:
    """Asynchronously download and cache stock data to avoid repeated API calls."""
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
    required_cols = {"open", "high", "low", "close", "volume"}

    def fix_columns(columns):
        new_cols = []
        for col in columns:
            if isinstance(col, tuple):
                new_cols.append("_".join(map(str, col)).lower())
            else:
                new_cols.append(str(col).lower())
        return new_cols

    # Try reading from cache
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file, engine="pyarrow")
        except Exception as e:
            logging.warning(f"Modin read_parquet failed: {e}. Falling back to pandas.")
            import pandas as pd_normal

            df = pd_normal.read_parquet(cache_file, engine="pyarrow")
        df.columns = fix_columns(df.columns)
        if "close" not in df.columns and "adj close" in df.columns:
            df["close"] = df["adj close"]
        if not required_cols.issubset(set(df.columns)):
            logging.error(
                f"Cached data missing required columns: {missing}. Deleting cache file."
            )
            cache_file.unlink()
        else:
            return df

    # Download using yf.download
    async with aiohttp.ClientSession() as session:
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, partial(yf.download, ticker, start=start, end=end, progress=False)
            )
        except Exception as e:
            logging.warning(f"yf.download failed: {e}")
            data = None

    # If download failed or the data is missing columns, try yf.Ticker().history
    if (
        data is None
        or data.empty
        or not required_cols.issubset(set(fix_columns(data.columns)))
    ):
        logging.warning(
            "yf.download did not return valid data; trying yf.Ticker().history"
        )
        data = yf.Ticker(ticker).history(start=start, end=end)
    if data is None or data.empty:
        logging.error("Downloaded data is empty or None after fallback.")
        return None

    data.columns = fix_columns(data.columns)
    if "close" not in data.columns and "adj close" in data.columns:
        data["close"] = data["adj close"]
    if not required_cols.issubset(set(data.columns)):
        missing = required_cols - set(data.columns)
        logging.error(f"Downloaded data missing required columns: {missing}")
        return None

    data.to_parquet(cache_file)
    return data