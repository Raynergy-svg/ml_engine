"""
Test script to validate ML model improvements.
"""

import torch
import torch.nn as nn
import numpy as np
from models_enhanced import StockPredictor, AttentiveLSTM, Mish, Swish
from evaluation import smooth_predictions, ensemble_predictions
from data_processing_optimized import augment_data

def test_activation_functions():
    """Test new activation functions."""
    print("Testing activation functions...")
    
    # Test Mish
    mish = Mish()
    x = torch.randn(10, 5)
    out = mish(x)
    assert out.shape == x.shape, "Mish output shape mismatch"
    assert torch.all(torch.isfinite(out)), "Mish produced non-finite values"
    print("✓ Mish activation works correctly")
    
    # Test Swish
    swish = Swish()
    out = swish(x)
    assert out.shape == x.shape, "Swish output shape mismatch"
    assert torch.all(torch.isfinite(out)), "Swish produced non-finite values"
    print("✓ Swish activation works correctly")

def test_improved_stock_predictor():
    """Test improved StockPredictor with new features."""
    print("\nTesting improved StockPredictor...")
    
    # Test with different activation functions
    for activation in ["relu", "mish", "swish", "gelu"]:
        model = StockPredictor(
            input_size=7,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            activation=activation,
            use_batch_norm=True
        )
        
        # Test forward pass
        x = torch.randn(8, 30, 7)
        output = model(x)
        
        assert output.shape == (8, 1), f"Output shape mismatch with {activation}"
        assert torch.all(torch.isfinite(output)), f"Model with {activation} produced non-finite values"
        print(f"✓ StockPredictor with {activation} activation works correctly")
    
    # Test batch normalization
    model_bn = StockPredictor(
        input_size=7,
        hidden_size=64,
        num_layers=2,
        use_batch_norm=True
    )
    x = torch.randn(16, 30, 7)
    output = model_bn(x)
    assert output.shape == (16, 1), "BatchNorm model output shape mismatch"
    print("✓ StockPredictor with BatchNorm works correctly")

def test_improved_attention():
    """Test improved AttentiveLSTM with optimized attention."""
    print("\nTesting improved AttentiveLSTM...")
    
    model = AttentiveLSTM(
        input_size=7,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        dropout=0.2,
        use_flash_attention=True
    )
    
    x = torch.randn(8, 30, 7)
    output = model(x)
    
    assert output.shape == (8, 1), "AttentiveLSTM output shape mismatch"
    assert torch.all(torch.isfinite(output)), "AttentiveLSTM produced non-finite values"
    print("✓ AttentiveLSTM with optimized attention works correctly")

def test_prediction_smoothing():
    """Test prediction smoothing utilities."""
    print("\nTesting prediction smoothing...")
    
    predictions = np.random.randn(100)
    
    # Test EMA smoothing
    smoothed_ema = smooth_predictions(predictions, method="ema", alpha=0.3)
    assert smoothed_ema.shape == predictions.shape, "EMA smoothing shape mismatch"
    assert np.all(np.isfinite(smoothed_ema)), "EMA smoothing produced non-finite values"
    print("✓ EMA smoothing works correctly")
    
    # Test SMA smoothing
    smoothed_sma = smooth_predictions(predictions, method="sma", window=5)
    assert smoothed_sma.shape == predictions.shape, "SMA smoothing shape mismatch"
    print("✓ SMA smoothing works correctly")
    
    # Test median smoothing
    smoothed_median = smooth_predictions(predictions, method="median", window=5)
    assert smoothed_median.shape == predictions.shape, "Median smoothing shape mismatch"
    print("✓ Median smoothing works correctly")

def test_ensemble_predictions():
    """Test ensemble prediction utilities."""
    print("\nTesting ensemble predictions...")
    
    pred1 = np.random.randn(100)
    pred2 = np.random.randn(100)
    pred3 = np.random.randn(100)
    predictions_list = [pred1, pred2, pred3]
    
    # Test mean ensemble
    ensemble_mean = ensemble_predictions(predictions_list, method="mean")
    assert ensemble_mean.shape == pred1.shape, "Mean ensemble shape mismatch"
    assert np.all(np.isfinite(ensemble_mean)), "Mean ensemble produced non-finite values"
    print("✓ Mean ensemble works correctly")
    
    # Test median ensemble
    ensemble_median = ensemble_predictions(predictions_list, method="median")
    assert ensemble_median.shape == pred1.shape, "Median ensemble shape mismatch"
    print("✓ Median ensemble works correctly")
    
    # Test weighted ensemble
    weights = [0.5, 0.3, 0.2]
    ensemble_weighted = ensemble_predictions(predictions_list, method="weighted", weights=weights)
    assert ensemble_weighted.shape == pred1.shape, "Weighted ensemble shape mismatch"
    print("✓ Weighted ensemble works correctly")

def test_improved_augmentation():
    """Test improved data augmentation strategies."""
    print("\nTesting improved data augmentation...")
    
    sequences = np.random.randn(50, 30, 7)
    targets = np.random.randn(50)
    
    # Test augmentation with mixup
    aug_seq, aug_tgt = augment_data(
        sequences, targets,
        noise_level=0.01,
        augmentation_factor=3,
        use_mixup=True
    )
    
    assert aug_seq.shape[0] == sequences.shape[0] * 3, "Augmented sequences count mismatch"
    assert aug_seq.shape[1:] == sequences.shape[1:], "Augmented sequences shape mismatch"
    assert aug_tgt.shape[0] == targets.shape[0] * 3, "Augmented targets count mismatch"
    assert np.all(np.isfinite(aug_seq)), "Augmented sequences contain non-finite values"
    assert np.all(np.isfinite(aug_tgt)), "Augmented targets contain non-finite values"
    print("✓ Improved data augmentation works correctly")

def test_gradient_flow():
    """Test that gradients flow properly through improved models."""
    print("\nTesting gradient flow...")
    
    model = StockPredictor(
        input_size=7,
        hidden_size=64,
        num_layers=2,
        activation="mish",
        use_batch_norm=True
    )
    
    x = torch.randn(8, 30, 7, requires_grad=True)
    target = torch.randn(8, 1)
    
    # Forward pass
    output = model(x)
    
    # Compute loss
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist and are finite
    assert x.grad is not None, "Input gradients are None"
    assert torch.all(torch.isfinite(x.grad)), "Input gradients contain non-finite values"
    
    # Check model parameter gradients
    grad_count = 0
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None, "Parameter gradient is None"
            assert torch.all(torch.isfinite(param.grad)), "Parameter gradient contains non-finite values"
            grad_count += 1
    
    assert grad_count > 0, "No gradients computed"
    print(f"✓ Gradient flow works correctly ({grad_count} parameters with gradients)")

def test_huber_loss():
    """Test Huber loss functionality."""
    print("\nTesting Huber loss...")
    
    predictions = torch.randn(32, 1)
    targets = torch.randn(32, 1)
    
    # Test Huber loss
    huber_loss = nn.HuberLoss(delta=1.0)
    loss = huber_loss(predictions, targets)
    
    assert torch.isfinite(loss), "Huber loss is not finite"
    assert loss >= 0, "Huber loss is negative"
    print("✓ Huber loss works correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing ML Model Improvements")
    print("=" * 60)
    
    try:
        test_activation_functions()
        test_improved_stock_predictor()
        test_improved_attention()
        test_prediction_smoothing()
        test_ensemble_predictions()
        test_improved_augmentation()
        test_gradient_flow()
        test_huber_loss()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
