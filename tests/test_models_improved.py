"""
Unit tests for improved model implementations
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import StockPredictor


class TestStockPredictorValidation(unittest.TestCase):
    """Test StockPredictor validation improvements"""
    
    def test_valid_initialization(self):
        """Test model initializes with valid parameters"""
        model = StockPredictor(
            input_size=7,
            hidden_size=128,
            num_layers=3,
            dropout=0.2
        )
        
        self.assertIsInstance(model, nn.Module)
    
    def test_invalid_input_size_raises_error(self):
        """Test that invalid input_size raises ValueError"""
        with self.assertRaises(ValueError):
            StockPredictor(input_size=0)
        
        with self.assertRaises(ValueError):
            StockPredictor(input_size=-1)
    
    def test_invalid_hidden_size_raises_error(self):
        """Test that invalid hidden_size raises ValueError"""
        with self.assertRaises(ValueError):
            StockPredictor(hidden_size=0)
        
        with self.assertRaises(ValueError):
            StockPredictor(hidden_size=-10)
    
    def test_invalid_num_layers_raises_error(self):
        """Test that invalid num_layers raises ValueError"""
        with self.assertRaises(ValueError):
            StockPredictor(num_layers=0)
        
        with self.assertRaises(ValueError):
            StockPredictor(num_layers=-1)
    
    def test_invalid_dropout_raises_error(self):
        """Test that invalid dropout raises ValueError"""
        with self.assertRaises(ValueError):
            StockPredictor(dropout=-0.1)
        
        with self.assertRaises(ValueError):
            StockPredictor(dropout=1.5)
    
    def test_forward_pass(self):
        """Test forward pass with valid input"""
        model = StockPredictor(input_size=5, hidden_size=64, num_layers=2)
        
        # Create sample input: [batch_size, sequence_length, input_size]
        batch_size = 16
        sequence_length = 60
        input_size = 5
        
        x = torch.randn(batch_size, sequence_length, input_size)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
    
    def test_bidirectional_model(self):
        """Test bidirectional LSTM model"""
        model = StockPredictor(
            input_size=7,
            hidden_size=64,
            num_layers=2,
            bidirectional=True
        )
        
        batch_size = 8
        sequence_length = 30
        x = torch.randn(batch_size, sequence_length, 7)
        
        output = model(x)
        self.assertEqual(output.shape, (batch_size, 1))
    
    def test_model_with_layer_norm(self):
        """Test model with layer normalization"""
        model = StockPredictor(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            use_layer_norm=True
        )
        
        x = torch.randn(4, 20, 5)
        output = model(x)
        
        self.assertEqual(output.shape, (4, 1))
    
    def test_model_without_layer_norm(self):
        """Test model without layer normalization"""
        model = StockPredictor(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            use_layer_norm=False
        )
        
        x = torch.randn(4, 20, 5)
        output = model(x)
        
        self.assertEqual(output.shape, (4, 1))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = StockPredictor(input_size=5, hidden_size=32, num_layers=2)
        
        x = torch.randn(8, 20, 5, requires_grad=True)
        target = torch.randn(8, 1)
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(x.grad)
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_model_output_range(self):
        """Test that model outputs are in reasonable range"""
        model = StockPredictor(input_size=5, hidden_size=64, num_layers=2)
        
        # Use reasonable input values
        x = torch.randn(16, 30, 5) * 10 + 100  # Stock price-like values
        
        output = model(x)
        
        # Check that outputs are finite
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_batch_independence(self):
        """Test that predictions for different samples are independent"""
        model = StockPredictor(input_size=5, hidden_size=32, num_layers=2)
        model.eval()
        
        # Create two different inputs
        x1 = torch.randn(1, 20, 5)
        x2 = torch.randn(1, 20, 5)
        
        # Get predictions
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        
        # Unless inputs are identical, outputs should be different
        if not torch.allclose(x1, x2):
            self.assertFalse(torch.allclose(out1, out2))
    
    def test_deterministic_output(self):
        """Test that same input produces same output (deterministic)"""
        model = StockPredictor(input_size=5, hidden_size=32, num_layers=2)
        model.eval()
        
        x = torch.randn(4, 20, 5)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        # Same input should give same output
        self.assertTrue(torch.allclose(out1, out2))


class TestModelRobustness(unittest.TestCase):
    """Test model robustness to edge cases"""
    
    def test_single_sample_batch(self):
        """Test model works with single sample"""
        model = StockPredictor(input_size=5, hidden_size=32, num_layers=2)
        
        x = torch.randn(1, 20, 5)
        output = model(x)
        
        self.assertEqual(output.shape, (1, 1))
    
    def test_large_batch(self):
        """Test model works with large batch"""
        model = StockPredictor(input_size=5, hidden_size=32, num_layers=2)
        
        x = torch.randn(256, 20, 5)
        output = model(x)
        
        self.assertEqual(output.shape, (256, 1))
    
    def test_variable_sequence_lengths(self):
        """Test model with different sequence lengths"""
        model = StockPredictor(input_size=5, hidden_size=32, num_layers=2)
        
        # Test with different sequence lengths
        for seq_len in [10, 30, 60, 120]:
            x = torch.randn(8, seq_len, 5)
            output = model(x)
            self.assertEqual(output.shape, (8, 1))
    
    def test_model_state_preservation(self):
        """Test that model state is preserved between forward passes"""
        model = StockPredictor(input_size=5, hidden_size=32, num_layers=2)
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward pass (no training)
        x = torch.randn(8, 20, 5)
        with torch.no_grad():
            _ = model(x)
        
        # Check parameters haven't changed
        for initial, current in zip(initial_params, model.parameters()):
            self.assertTrue(torch.allclose(initial, current))


if __name__ == '__main__':
    unittest.main()
