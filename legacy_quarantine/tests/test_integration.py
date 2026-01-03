"""
Integration test script for the stock market prediction model.
Tests the complete workflow from data fetching to prediction and analysis.
"""

import os
import sys
import logging
import unittest
import numpy as np
import unittest

raise unittest.SkipTest("Retired: depended on PyTorch (removed from repo).")
import asyncio
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import optimized modules
from data_processing import load_stock_data_efficient, prepare_sequences, StockDataset, create_optimized_dataloader
from ml_engine_enhanced import EnhancedMLEngine
from memory_manager_enhanced import MemoryManager
from models_enhanced import StockPredictor, AttentiveLSTM, GRUPredictor, TransformerPredictor, EnsemblePredictor
from reasoning_enhanced import ReasoningEngine, UncertaintyEstimator
from neural_network_integrator import NeuralNetworkIntegrator, ModelSynchronizer
from cli_enhanced import StockPredictionCLI


class TestDataIntegration(unittest.TestCase):
    """Test data integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("tests/test_data")
        self.test_dir.mkdir(exist_ok=True, parents=True)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'ticker': ['AAPL'] * 100,
            'open': np.random.randn(100) * 10 + 150,
            'high': np.random.randn(100) * 10 + 155,
            'low': np.random.randn(100) * 10 + 145,
            'close': np.random.randn(100) * 10 + 150,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Save sample data
        self.sample_data.to_csv(self.test_dir / "sample_data.csv", index=False)
    
    def test_data_to_model_integration(self):
        """Test integration from data loading to model input."""
        try:
            # Prepare sequences
            features, targets, metadata = prepare_sequences(
                df=self.sample_data,
                sequence_length=10,
                target_column="close",
                feature_columns=["open", "high", "low", "close", "volume"],
                scale_features=True,
                scale_target=True
            )
            
            # Create dataset
            dataset = StockDataset(features, targets)
            
            # Create dataloader
            dataloader = create_optimized_dataloader(
                dataset,
                batch_size=16,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            
            # Create model
            model = AttentiveLSTM(
                input_size=features.shape[2],
                hidden_size=32,
                num_layers=2,
                dropout=0.2
            )
            
            # Test forward pass with batch from dataloader
            for batch_x, batch_y in dataloader:
                output = model(batch_x)
                self.assertEqual(output.shape[0], batch_x.shape[0])
                self.assertEqual(output.shape[1], 1)
                break
            
            logger.info("Data to model integration test passed")
        except Exception as e:
            logger.error(f"Data to model integration test failed: {e}")
            self.fail(f"Data to model integration test failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        pass


class TestModelIntegration(unittest.TestCase):
    """Test model integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create sample data
        self.batch_size = 8
        self.seq_length = 10
        self.input_size = 5
        self.hidden_size = 32
        
        # Create sample inputs
        self.sample_input = torch.randn(self.batch_size, self.seq_length, self.input_size)
        
        # Create config
        self.config = {
            "model": {
                "type": "ensemble",
                "base_models": [
                    {"type": "lstm", "hidden_size": 32, "num_layers": 2},
                    {"type": "gru", "hidden_size": 32, "num_layers": 2},
                    {"type": "transformer", "hidden_size": 32, "num_layers": 2, "num_heads": 4}
                ],
                "ensemble_method": "attention"
            },
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001
            },
            "device": str(self.device)
        }
    
    def test_ml_engine_to_integrator(self):
        """Test integration from ML engine to neural network integrator."""
        try:
            # Create ML engine
            ml_engine = EnhancedMLEngine(self.config)
            
            # Create MT engine
            mt_engine = EnhancedMLEngine(self.config)
            
            # Create MR engine
            mr_engine = EnhancedMLEngine(self.config)
            
            # Create reasoning engine
            reasoning_engine = ReasoningEngine()
            
            # Create integrator
            integrator_config = {
                "use_attention": True,
                "use_dynamic_weights": True,
                "feature_dim": self.input_size,
                "hidden_dim": 64,
                "num_heads": 4,
                "device": str(self.device)
            }
            integrator = NeuralNetworkIntegrator(integrator_config)
            
            # Set engines
            integrator.set_engines(
                ml_engine=ml_engine,
                mt_engine=mt_engine,
                mr_engine=mr_engine,
                reasoning_engine=reasoning_engine
            )
            
            # Create mock prediction methods for engines
            def mock_predict(features):
                return {"prediction": torch.randn(self.batch_size, 1), "uncertainty": 0.2}
            
            ml_engine.predict = mock_predict
            mt_engine.predict = mock_predict
            mr_engine.predict = mock_predict
            
            # Create features dictionary
            features = {
                "ml_features": self.sample_input,
                "mt_features": self.sample_input,
                "mr_features": self.sample_input
            }
            
            # Make integrated prediction
            result = integrator.predict(features)
            
            # Check result
            self.assertIn("prediction", result)
            self.assertIn("uncertainty", result)
            self.assertIn("weights", result)
            self.assertIn("reasoning", result)
            
            logger.info("ML engine to integrator test passed")
        except Exception as e:
            logger.error(f"ML engine to integrator test failed: {e}")
            self.fail(f"ML engine to integrator test failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        pass


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("tests/test_data")
        self.test_dir.mkdir(exist_ok=True, parents=True)
        
        # Create output directories
        self.model_dir = Path("tests/test_models")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        self.output_dir = Path("tests/test_output")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'ticker': ['AAPL'] * 100,
            'open': np.random.randn(100) * 10 + 150,
            'high': np.random.randn(100) * 10 + 155,
            'low': np.random.randn(100) * 10 + 145,
            'close': np.random.randn(100) * 10 + 150,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Save sample data
        self.sample_data.to_csv(self.test_dir / "sample_data.csv", index=False)
        
        # Create test config
        self.config = {
            "data": {
                "tickers": ["AAPL"],
                "start_date": "2020-01-01",
                "end_date": "2020-04-01",
                "features": ["open", "high", "low", "close", "volume"],
                "target": "close",
                "sequence_length": 10,
                "validation_split": 0.2,
                "test_split": 0.1
            },
            "ml_engine": {
                "model": {
                    "type": "lstm",
                    "hidden_size": 32,
                    "num_layers": 2,
                    "dropout": 0.2
                },
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001
                },
                "batch_size": 16,
                "epochs": 2
            },
            "mt_engine": {
                "model": {
                    "type": "gru",
                    "hidden_size": 32,
                    "num_layers": 2,
                    "dropout": 0.2
                },
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001
                },
                "batch_size": 16,
                "epochs": 2
            },
            "mr_engine": {
                "model": {
                    "type": "transformer",
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout": 0.2
                },
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001
                },
                "batch_size": 16,
                "epochs": 2
            },
            "integrator": {
                "use_attention": True,
                "use_dynamic_weights": True,
                "feature_dim": 5,
                "hidden_dim": 32,
                "num_heads": 4
            },
            "reasoning": {
                "confidence_threshold": 0.7,
                "uncertainty_threshold": 0.3,
                "output_dir": str(self.output_dir)
            },
            "paths": {
                "data_dir": str(self.test_dir),
                "model_dir": str(self.model_dir),
                "output_dir": str(self.output_dir)
            },
            "device": "cpu",
            "num_workers": 0,
            "pin_memory": False
        }
        
        # Save config
        with open(self.test_dir / "test_config.json", "w") as f:
            import json
            json.dump(self.config, f, indent=2)
    
    def test_cli_workflow(self):
        """Test CLI workflow."""
        try:
            # Create CLI
            cli = StockPredictionCLI()
            
            # Load config
            cli.load_config(str(self.test_dir / "test_config.json"))
            
            # Initialize engines
            cli.initialize_engines()
            
            # Mock data fetching
            async def mock_fetch_stock_data(tickers, start_date, end_date, interval="1d"):
                return self.sample_data
            
            cli.fetch_stock_data = mock_fetch_stock_data
            
            # Mock prepare_data_for_prediction
            async def mock_prepare_data_for_prediction(ticker, days_back=60, include_market_data=True, include_economic_data=True):
                features, targets, metadata = prepare_sequences(
                    df=self.sample_data,
                    sequence_length=10,
                    target_column="close",
                    feature_columns=["open", "high", "low", "close", "volume"],
                    scale_features=True,
                    scale_target=True
                )
                
                return {
                    "ticker": ticker,
                    "ml_features": features,
                    "ml_targets": targets,
                    "ml_metadata": metadata,
                    "mt_features": features,
                    "mt_targets": targets,
                    "mt_metadata": metadata,
                    "mr_features": features,
                    "mr_targets": targets,
                    "mr_metadata": metadata,
                    "stock_data": self.sample_data,
                    "market_data": self.sample_data,
                    "economic_data": self.sample_data
                }
            
            cli.prepare_data_for_prediction = mock_prepare_data_for_prediction
            
            # Test training (limited to 1 epoch for testing)
            cli.config["ml_engine"]["epochs"] = 1
            cli.config["mt_engine"]["epochs"] = 1
            cli.config["mr_engine"]["epochs"] = 1
            
            # Mock train method to avoid actual training
            def mock_train(features, targets, validation_split=0.2, epochs=1, verbose=True, checkpoint_path=None):
                for epoch in range(epochs):
                    yield epoch, {"train_loss": 0.1, "val_loss": 0.2}
            
            cli.ml_engine.train = mock_train
            cli.mt_engine.train = mock_train
            cli.mr_engine.train = mock_train
            
            # Test prediction
            # Mock load_model to avoid loading actual model
            def mock_load_model(path):
                pass
            
            cli.ml_engine.load_model = mock_load_model
            cli.mt_engine.load_model = mock_load_model
            cli.mr_engine.load_model = mock_load_model
            
            # Mock predict method
            def mock_predict(features):
                return {"prediction": np.random.randn(features.shape[0], 1), "uncertainty": 0.2}
            
            cli.ml_engine.predict = mock_predict
            cli.mt_engine.predict = mock_predict
            cli.mr_engine.predict = mock_predict
            
            # Test analyze
            # Mock _calculate_technical_indicators
            def mock_calculate_technical_indicators(df):
                df = df.copy()
                df['SMA_20'] = df['close'].rolling(window=20).mean()
                df['SMA_50'] = df['close'].rolling(window=50).mean()
                df['SMA_200'] = df['close'].rolling(window=200).mean()
                df['RSI'] = 50  # Mock value
                df['MACD'] = 0.1  # Mock value
                df['MACD_Signal'] = 0.05  # Mock value
                df['BB_Upper'] = df['close'] + 10
                df['BB_Lower'] = df['close'] - 10
                df['ATR'] = 5  # Mock value
                return df
            
            cli._calculate_technical_indicators = mock_calculate_technical_indicators
            
            logger.info("CLI workflow test passed")
        except Exception as e:
            logger.error(f"CLI workflow test failed: {e}")
            self.fail(f"CLI workflow test failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.model_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)


class TestErrorHandling(unittest.TestCase):
    """Test error handling functionality."""
    
    def setUp(self):
        """Set up test environment."""
        pass
    
    def test_data_loading_error_handling(self):
        """Test error handling in data loading."""
        try:
           <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>
# — Raynergy-svg —
