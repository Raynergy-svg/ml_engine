"""Data processing and dataset utilities with improved error handling and validation"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Optional sklearn import for normalization features
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logger for this module
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class StockDataset(Dataset):
    """PyTorch Dataset for generating sequences from stock market data.
    
    This dataset validates input data and handles various data formats safely.
    
    Args:
        features: Array-like features data (list, numpy array, or tensor)
        targets: Array-like target data (list, numpy array, or tensor)
        sequence_length: Length of sequences (default: 60)
        
    Raises:
        DataValidationError: If input data is invalid or incompatible
    """

    def __init__(self, features: Union[list, np.ndarray, torch.Tensor], 
                 targets: Union[list, np.ndarray, torch.Tensor], 
                 sequence_length: int = 60):
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
            raise DataValidationError(
                f"Features length ({len(features)}) must match targets length ({len(targets)})"
            )
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Features contain NaN or infinite values, replacing with zeros")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
        if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
            logger.warning("Targets contain NaN or infinite values, replacing with zeros")
            targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        logger.info(f"Initialized StockDataset with {len(self)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
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
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
            
        try:
            seq = torch.FloatTensor(self.features[idx])
            target = torch.FloatTensor([self.targets[idx]])
            return seq, target
        except Exception as e:
            logger.error(f"Error retrieving sample at index {idx}: {e}")
            raise


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that a DataFrame contains required columns and has valid data
    
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
    """Stack ticker-specific DataFrames extracted from MultiIndex data.
    
    Args:
        data: MultiIndex DataFrame with ticker data
        tickers: List of ticker symbols to process
        
    Returns:
        Concatenated DataFrame with all ticker data
        
    Raises:
        DataValidationError: If data is invalid
    """
    if data is None or data.empty:
        raise DataValidationError("Input data is None or empty")
        
    if not tickers:
        raise DataValidationError("Tickers list is empty")
    
    all_data = []
    for ticker in tickers:
        try:
            if (ticker,) in data.columns:
                df_ticker = data[ticker].copy()
                df_ticker.columns = df_ticker.columns.str.lower()
                df_ticker["ticker"] = ticker
                df_ticker.reset_index(inplace=True)
                all_data.append(df_ticker)
                logger.debug(f"Successfully processed ticker: {ticker}")
            else:
                logger.warning(f"Ticker {ticker} not found in data columns")
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            continue
            
    if not all_data:
        logger.warning("No valid ticker data was processed")
        return pd.DataFrame()
        
    result = pd.concat(all_data, ignore_index=True)
    logger.info(f"Processed {len(all_data)} tickers with {len(result)} total rows")
    return result


def normalize_features(data: pd.DataFrame, columns: List[str], 
                      method: str = 'standard') -> Tuple[pd.DataFrame, dict]:
    """Normalize features using specified method
    
    Args:
        data: Input DataFrame
        columns: Columns to normalize
        method: Normalization method ('standard', 'minmax', or 'robust')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
        
    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If method is not recognized
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for normalization. "
                         "Install with: pip install scikit-learn")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized_data = data.copy()
    
    try:
        normalized_data[columns] = scaler.fit_transform(data[columns])
        logger.info(f"Normalized {len(columns)} columns using {method} scaling")
        
        # Store normalization parameters for later inverse transform
        params = {
            'method': method,
            'scaler': scaler,
            'columns': columns
        }
        return normalized_data, params
        
    except Exception as e:
        logger.error(f"Error during normalization: {e}")
        raise


def create_sequences(data: np.ndarray, sequence_length: int = 60, 
                    target_column: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences from time series data for ML training
    
    Args:
        data: Input data array of shape (n_samples, n_features)
        sequence_length: Length of each sequence
        target_column: Index of target column to predict
        
    Returns:
        Tuple of (sequences, targets) where sequences has shape 
        (n_sequences, sequence_length, n_features) and targets has shape (n_sequences,)
        
    Raises:
        DataValidationError: If inputs are invalid
    """
    if data is None or len(data) == 0:
        raise DataValidationError("Input data is None or empty")
        
    if sequence_length <= 0:
        raise DataValidationError(f"Sequence length must be positive, got {sequence_length}")
        
    if len(data) < sequence_length + 1:
        raise DataValidationError(
            f"Data length ({len(data)}) must be greater than sequence_length ({sequence_length})"
        )
    
    sequences = []
    targets = []
    
    try:
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length, target_column]
            
            # Validate sequence
            if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
                logger.warning(f"Skipping sequence {i} due to NaN/inf values")
                continue
                
            sequences.append(seq)
            targets.append(target)
        
        sequences_array = np.array(sequences)
        targets_array = np.array(targets)
        
        logger.info(f"Created {len(sequences_array)} sequences of length {sequence_length}")
        return sequences_array, targets_array
        
    except Exception as e:
        logger.error(f"Error creating sequences: {e}")
        raise DataValidationError(f"Failed to create sequences: {e}")