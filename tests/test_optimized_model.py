"""
Test script for the optimized stock market prediction model.
Tests data loading, model training, prediction, and integration.
"""

import os
import sys
import logging
import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_results.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import optimized modules
from data_processing_optimized import load_stock_data_efficient, prepare_sequences, StockDataset, create_optimized_dataloader
from ml_engine_enhanced import EnhancedMLEngine
from memory_manager_enhanced import MemoryManager
from models_enhanced import StockPredictor, AttentiveLSTM, GRUPredictor, TransformerPredictor, EnsemblePredictor
from reasoning_enhanced import ReasoningEngine, UncertaintyEstimator
from neural_network_integrator import NeuralNetworkIntegrator, ModelSynchronizer


class TestDataProcessing(unittest.TestCase):
    """Test data processing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
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
    
    def test_load_stock_data_efficient(self):
        """Test efficient stock data loading."""
        try:
            # Test with sample data
            data = load_stock_data_efficient(
                tickers=["AAPL"],
                start_date="2020-01-01",
                end_date="2020-04-10",
                use_cache=False
            )
            
            # Check if data is loaded
            self.assertIsNotNone(data)
            self.assertFalse(data.empty)
            logger.info("load_stock_data_efficient test passed")
        except Exception as e:
            logger.error(f"load_stock_data_efficient test failed: {e}")
            self.fail(f"load_stock_data_efficient test failed: {e}")
    
    def test_prepare_sequences(self):
        """Test sequence preparation."""
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
            
            # Check shapes
            self.assertEqual(features.shape[0], len(self.sample_data) - 10)
            self.assertEqual(features.shape[1], 10)  # sequence length
            self.assertEqual(features.shape[2], 5)   # number of features
            self.assertEqual(targets.shape[0], len(self.sample_data) - 10)
            
            # Check metadata
            self.assertIn("feature_columns", metadata)
            self.assertIn("target_column", metadata)
            self.assertIn("sequence_length", metadata)
            self.assertIn("scalers", metadata)
            
            logger.info("prepare_sequences test passed")
        except Exception as e:
            logger.error(f"prepare_sequences test failed: {e}")
            self.fail(f"prepare_sequences test failed: {e}")
    
    def test_stock_dataset(self):
        """Test StockDataset class."""
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
            
            # Check dataset
            self.assertEqual(len(dataset), len(features))
            
            # Get an item
            x, y = dataset[0]
            self.assertEqual(x.shape, (10, 5))  # sequence length, features
            self.assertEqual(y.shape, ())       # single target value
            
            logger.info("StockDataset test passed")
        except Exception as e:
            logger.error(f"StockDataset test failed: {e}")
            self.fail(f"StockDataset test failed: {e}")
    
    def test_create_optimized_dataloader(self):
        """Test optimized dataloader creation."""
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
            
            # Check dataloader
            self.assertIsNotNone(dataloader)
            
            # Iterate through dataloader
            for batch_x, batch_y in dataloader:
                self.assertLessEqual(batch_x.shape[0], 16)  # batch size
                self.assertEqual(batch_x.shape[1], 10)      # sequence length
                self.assertEqual(batch_x.shape[2], 5)       # features
                self.assertLessEqual(batch_y.shape[0], 16)  # batch size
                break
            
            logger.info("create_optimized_dataloader test passed")
        except Exception as e:
            logger.error(f"create_optimized_dataloader test failed: {e}")
            self.fail(f"create_optimized_dataloader test failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test directory
        import shutil
        shutil.rmtree(self.test_dir)


class TestModels(unittest.TestCase):
    """Test model functionality."""
    
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
        self.sample_target = torch.randn(self.batch_size)
    
    def test_attentive_lstm(self):
        """Test AttentiveLSTM model."""
        try:
            # Create model
            model = AttentiveLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
            
            # Forward pass
            sample_input = self.sample_input.to(self.device)
            output = model(sample_input)
            
            # Check output shape
            self.assertEqual(output.shape, (self.batch_size, 1))
            
            logger.info("AttentiveLSTM test passed")
        except Exception as e:
            logger.error(f"AttentiveLSTM test failed: {e}")
            self.fail(f"AttentiveLSTM test failed: {e}")
    
    def test_gru_predictor(self):
        """Test GRUPredictor model."""
        try:
            # Create model
            model = GRUPredictor(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
            
            # Forward pass
            sample_input = self.sample_input.to(self.device)
            output = model(sample_input)
            
            # Check output shape
            self.assertEqual(output.shape, (self.batch_size, 1))
            
            logger.info("GRUPredictor test passed")
        except Exception as e:
            logger.error(f"GRUPredictor test failed: {e}")
            self.fail(f"GRUPredictor test failed: {e}")
    
    def test_transformer_predictor(self):
        """Test TransformerPredictor model."""
        try:
            # Create model
            model = TransformerPredictor(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                num_heads=4,
                dropout=0.2
            ).to(self.device)
            
            # Forward pass
            sample_input = self.sample_input.to(self.device)
            output = model(sample_input)
            
            # Check output shape
            self.assertEqual(output.shape, (self.batch_size, 1))
            
            logger.info("TransformerPredictor test passed")
        except Exception as e:
            logger.error(f"TransformerPredictor test failed: {e}")
            self.fail(f"TransformerPredictor test failed: {e}")
    
    def test_ensemble_predictor(self):
        """Test EnsemblePredictor model."""
        try:
            # Create base models
            models = [
                AttentiveLSTM(input_size=self.input_size, hidden_size=self.hidden_size),
                GRUPredictor(input_size=self.input_size, hidden_size=self.hidden_size),
                TransformerPredictor(input_size=self.input_size, hidden_size=self.hidden_size, num_heads=4)
            ]
            
            # Create ensemble model
            model = EnsemblePredictor(
                models=models,
                ensemble_method="attention",
                input_size=self.input_size
            ).to(self.device)
            
            # Forward pass
            sample_input = self.sample_input.to(self.device)
            output = model(sample_input)
            
            # Check output shape
            self.assertEqual(output.shape, (self.batch_size, 1))
            
            logger.info("EnsemblePredictor test passed")
        except Exception as e:
            logger.error(f"EnsemblePredictor test failed: {e}")
            self.fail(f"EnsemblePredictor test failed: {e}")


class TestMLEngine(unittest.TestCase):
    """Test ML engine functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("test_models")
        self.test_dir.mkdir(exist_ok=True)
        
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
        
        # Prepare sequences
        self.features, self.targets, self.metadata = prepare_sequences(
            df=self.sample_data,
            sequence_length=10,
            target_column="close",
            feature_columns=["open", "high", "low", "close", "volume"],
            scale_features=True,
            scale_target=True
        )
        
        # Create config
        self.config = {
            "model": {
                "type": "lstm",
                "input_size": 5,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.2
            },
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001
            },
            "batch_size": 16,
            "epochs": 2,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    
    def test_ml_engine_initialization(self):
        """Test ML engine initialization."""
        try:
            # Create ML engine
            engine = EnhancedMLEngine(self.config)
            
            # Check engine
            self.assertIsNotNone(engine)
            self.assertIsNotNone(engine.model)
            self.assertIsNotNone(engine.optimizer)
            
            logger.info("ML engine initialization test passed")
        except Exception as e:
            logger.error(f"ML engine initialization test failed: {e}")
            self.fail(f"ML engine initialization test failed: {e}")
    
    def test_ml_engine_training(self):
        """Test ML engine training."""
        try:
            # Create ML engine
            engine = EnhancedMLEngine(self.config)
            
            # Train for a few steps
            for epoch, metrics in engine.train(
                features=self.features,
                targets=self.targets,
                validation_split=0.2,
                epochs=2,
                verbose=True,
                checkpoint_path=str(self.test_dir / "test_model.pth")
            ):
                # Check metrics
                self.assertIn("train_loss", metrics)
                self.assertIn("val_loss", metrics)
                
                logger.info(f"Epoch {epoch}: {metrics}")
            
            # Check if model file exists
            self.assertTrue((self.test_dir / "test_model.pth").exists())
            
            logger.info("ML engine training test passed")
        except Exception as e:
            logger.error(f"ML engine training test failed: {e}")
            self.fail(f"ML engine training test failed: {e}")
    
    def test_ml_engine_prediction(self):
        """Test ML engine prediction."""
        try:
            # Create ML engine
            engine = EnhancedMLEngine(self.config)
            
            # Train for a few steps
            for epoch, metrics in engine.train(
                features=self.features,
                targets=self.targets,
                validation_split=0.2,
                epochs=1,
                verbose=True
            ):
                pass
            
            # Make prediction
            prediction = engine.predict(self.features[:1])
            
            # Check prediction
            self.assertIsNotNone(prediction)
            
            logger.info("ML engine prediction test passed")
        except Exception as e:
            logger.error(f"ML engine prediction test failed: {e}")
            self.fail(f"ML engine prediction test failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test directory
        import shutil
        shutil.rmtree(self.test_dir)


class TestNeuralNetworkIntegrator(unittest.TestCase):
    """Test neural network integrator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create sample data
        self.batch_size = 4
        self.input_dim = 64
        
        # Create dummy engine outputs
        self.ml_output = torch.randn(self.batch_size, self.input_dim)
        self.mt_output = torch.randn(self.batch_size, self.input_dim)
        self.mr_output = torch.randn(self.batch_size, self.input_<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>