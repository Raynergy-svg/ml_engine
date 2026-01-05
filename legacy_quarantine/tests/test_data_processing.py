"""
Unit tests for data processing module improvements
"""

import unittest
import numpy as np
import unittest

raise unittest.SkipTest("Retired: depended on PyTorch (removed from repo).")
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing import (
    StockDataset, 
    DataValidationError,
    validate_dataframe,
    process_multiindex_data,
    normalize_features,
    create_sequences
)


class TestStockDataset(unittest.TestCase):
    """Test StockDataset class improvements"""
    
    def setUp(self):
        """Set up test data"""
        self.valid_features = np.random.randn(100, 10, 5)
        self.valid_targets = np.random.randn(100)
        
    def test_valid_initialization(self):
        """Test dataset initializes with valid data"""
        dataset = StockDataset(self.valid_features, self.valid_targets)
        self.assertEqual(len(dataset), 100)
        
    def test_none_features_raises_error(self):
        """Test that None features raise DataValidationError"""
        with self.assertRaises(DataValidationError):
            StockDataset(None, self.valid_targets)
    
    def test_none_targets_raises_error(self):
        """Test that None targets raise DataValidationError"""
        with self.assertRaises(DataValidationError):
            StockDataset(self.valid_features, None)
    
    def test_empty_features_raises_error(self):
        """Test that empty features raise DataValidationError"""
        with self.assertRaises(DataValidationError):
            StockDataset([], self.valid_targets)
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise DataValidationError"""
        with self.assertRaises(DataValidationError):
            StockDataset(self.valid_features[:50], self.valid_targets)
    
    def test_nan_handling(self):
        """Test that NaN values are handled properly"""
        features_with_nan = self.valid_features.copy()
        features_with_nan[0, 0, 0] = np.nan
        
        # Should not raise error, but should log warning
        dataset = StockDataset(features_with_nan, self.valid_targets)
        self.assertEqual(len(dataset), 100)
    
    def test_getitem(self):
        """Test __getitem__ returns proper tensors"""
        dataset = StockDataset(self.valid_features, self.valid_targets)
        features, target = dataset[0]
        
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        self.assertEqual(target.shape, (1,))
    
    def test_index_out_of_bounds(self):
        """Test that out of bounds index raises IndexError"""
        dataset = StockDataset(self.valid_features, self.valid_targets)
        
        with self.assertRaises(IndexError):
            _ = dataset[1000]


class TestValidateDataFrame(unittest.TestCase):
    """Test validate_dataframe function"""
    
    def test_valid_dataframe(self):
        """Test validation passes for valid DataFrame"""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        })
        
        result = validate_dataframe(df, ['open', 'close', 'volume'])
        self.assertTrue(result)
    
    def test_none_dataframe(self):
        """Test validation fails for None"""
        result = validate_dataframe(None, ['open', 'close'])
        self.assertFalse(result)
    
    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame"""
        df = pd.DataFrame()
        result = validate_dataframe(df, ['open', 'close'])
        self.assertFalse(result)
    
    def test_missing_columns(self):
        """Test validation fails for missing columns"""
        df = pd.DataFrame({'open': [1, 2, 3]})
        result = validate_dataframe(df, ['open', 'close', 'volume'])
        self.assertFalse(result)


class TestNormalizeFeatures(unittest.TestCase):
    """Test normalize_features function"""
    
    def setUp(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': [100, 200, 300, 400, 500]
        })
        self.columns = ['feature1', 'feature2']
    
    def test_standard_normalization(self):
        """Test standard scaling"""
        normalized, params = normalize_features(self.df, self.columns, method='standard')
        
        # Check that normalization was applied
        self.assertIn('feature1', normalized.columns)
        self.assertIn('feature2', normalized.columns)
        
        # Check that parameters were stored
        self.assertEqual(params['method'], 'standard')
        self.assertIn('scaler', params)
        self.assertEqual(params['columns'], self.columns)
    
    def test_minmax_normalization(self):
        """Test minmax scaling"""
        normalized, params = normalize_features(self.df, self.columns, method='minmax')
        
        self.assertEqual(params['method'], 'minmax')
        
        # MinMax should scale to [0, 1]
        self.assertGreaterEqual(normalized['feature1'].min(), 0)
        self.assertLessEqual(normalized['feature1'].max(), 1)
    
    def test_robust_normalization(self):
        """Test robust scaling"""
        normalized, params = normalize_features(self.df, self.columns, method='robust')
        
        self.assertEqual(params['method'], 'robust')
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError"""
        with self.assertRaises(ValueError):
            normalize_features(self.df, self.columns, method='invalid')


class TestCreateSequences(unittest.TestCase):
    """Test create_sequences function"""
    
    def setUp(self):
        """Set up test data"""
        # Create simple time series data
        self.data = np.array([[i, i*2, i*3] for i in range(100)])
        self.sequence_length = 10
    
    def test_valid_sequence_creation(self):
        """Test creating sequences from valid data"""
        sequences, targets = create_sequences(self.data, self.sequence_length)
        
        # Check shapes
        expected_n_sequences = len(self.data) - self.sequence_length
        self.assertEqual(sequences.shape[0], expected_n_sequences)
        self.assertEqual(sequences.shape[1], self.sequence_length)
        self.assertEqual(sequences.shape[2], self.data.shape[1])
        self.assertEqual(len(targets), expected_n_sequences)
    
    def test_none_data_raises_error(self):
        """Test that None data raises DataValidationError"""
        with self.assertRaises(DataValidationError):
            create_sequences(None, self.sequence_length)
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises DataValidationError"""
        with self.assertRaises(DataValidationError):
            create_sequences(np.array([]), self.sequence_length)
    
    def test_invalid_sequence_length_raises_error(self):
        """Test that invalid sequence length raises DataValidationError"""
        with self.assertRaises(DataValidationError):
            create_sequences(self.data, 0)
        
        with self.assertRaises(DataValidationError):
            create_sequences(self.data, -1)
    
    def test_data_too_short_raises_error(self):
        """Test that data shorter than sequence length raises error"""
        short_data = np.array([[1, 2, 3], [4, 5, 6]])
        
        with self.assertRaises(DataValidationError):
            create_sequences(short_data, 10)
    
    def test_custom_target_column(self):
        """Test creating sequences with custom target column"""
        sequences, targets = create_sequences(self.data, self.sequence_length, target_column=1)
        
        # Verify targets come from correct column
        self.assertEqual(len(targets), len(self.data) - self.sequence_length)


class TestProcessMultiIndexData(unittest.TestCase):
    """Test process_multiindex_data function"""
    
    def test_valid_multiindex_processing(self):
        """Test processing valid MultiIndex data"""
        # Create MultiIndex DataFrame
        dates = pd.date_range('2020-01-01', periods=10)
        data = pd.DataFrame({
            ('AAPL', 'open'): np.random.randn(10),
            ('AAPL', 'close'): np.random.randn(10),
            ('GOOGL', 'open'): np.random.randn(10),
            ('GOOGL', 'close'): np.random.randn(10)
        }, index=dates)
        
        # This would need actual MultiIndex structure to work properly
        # For now, test that function handles basic inputs
        tickers = ['AAPL', 'GOOGL']
        
        # Note: This test may fail depending on actual data structure
        # It's here to show the testing approach
    
    def test_empty_tickers_raises_error(self):
        """Test that empty tickers list raises DataValidationError"""
        df = pd.DataFrame()
        
        with self.assertRaises(DataValidationError):
            process_multiindex_data(df, [])
    
    def test_none_data_raises_error(self):
        """Test that None data raises DataValidationError"""
        with self.assertRaises(DataValidationError):
            process_multiindex_data(None, ['AAPL'])


if __name__ == '__main__':
    unittest.main()
# — Raynergy-svg —
