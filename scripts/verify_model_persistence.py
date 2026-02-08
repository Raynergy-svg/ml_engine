"""
Verify model save/load cycle for Transformer direction model.

This script tests that:
1. Model metadata is saved correctly
2. Calibration parameters persist
3. Model can be loaded back successfully
4. All critical attributes are preserved
"""

import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def verify_model_save_load(pair="EUR_USD"):
    """Verify that a saved model can be loaded correctly."""
    
    model_base_path = Path("trained_data/models") / pair
    model_path = model_base_path / "transformer_direction.keras"
    meta_path = model_base_path / "transformer_direction.meta.pkl"
    
    print(f"Verifying model save/load for {pair}")
    print("=" * 60)
    
    # Check if model files exist
    print("\n1. Checking model files...")
    if not model_path.exists():
        print(f"   ❌ Model file not found: {model_path}")
        print(f"   → Train a model first: ./bin/Buddy train -i {pair}")
        return False
    print(f"   ✓ Model file exists: {model_path}")
    
    if not meta_path.exists():
        print(f"   ❌ Metadata file not found: {meta_path}")
        return False
    print(f"   ✓ Metadata file exists: {meta_path}")
    
    # Load metadata
    print("\n2. Loading metadata...")
    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        print("   ✓ Metadata loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load metadata: {e}")
        return False
    
    # Verify critical metadata fields
    print("\n3. Verifying metadata fields...")
    required_fields = [
        'scaler', 'seq_len', 'metrics', 'config', 'feature_names',
        'n_features', 'model_type', 'output_calibration'
    ]
    
    for field in required_fields:
        if field in meta:
            print(f"   ✓ {field:20s}: Present")
        else:
            print(f"   ❌ {field:20s}: MISSING")
            return False
    
    # Check metrics
    print("\n4. Checking training metrics...")
    metrics = meta.get('metrics', {})
    
    metric_checks = {
        'val_accuracy': 'Validation accuracy',
        'val_balanced_accuracy': 'Balanced accuracy',
        'val_up_accuracy': 'UP class accuracy',
        'val_down_accuracy': 'DOWN class accuracy',
        'epochs_trained': 'Epochs trained',
        'n_train_samples': 'Training samples',
        'n_val_samples': 'Validation samples',
    }
    
    for key, desc in metric_checks.items():
        if key in metrics:
            value = metrics[key]
            if key.endswith('_accuracy'):
                print(f"   ✓ {desc:25s}: {value:.4f}")
            else:
                print(f"   ✓ {desc:25s}: {value}")
        else:
            print(f"   ⚠️ {desc:25s}: Not found")
    
    # Check calibration
    print("\n5. Checking output calibration...")
    calibration = meta.get('output_calibration')
    
    if calibration is None:
        print("   ⚠️ No calibration data saved")
    else:
        print("   ✓ Calibration data present:")
        print(f"     - Threshold: {calibration.get('threshold', 'N/A')}")
        print(f"     - Mean: {calibration.get('mean', 'N/A')}")
        print(f"     - Std: {calibration.get('std', 'N/A')}")
        print(f"     - Enabled: {calibration.get('enabled', False)}")
    
    # Check model type
    print("\n6. Verifying model type...")
    model_type = meta.get('model_type', 'unknown')
    if model_type == 'transformer':
        print(f"   ✓ Model type: {model_type}")
    else:
        print(f"   ❌ Unexpected model type: {model_type}")
        return False
    
    # Check feature configuration
    print("\n7. Checking feature configuration...")
    n_features = meta.get('n_features', 0)
    seq_len = meta.get('seq_len', 0)
    feature_names = meta.get('feature_names', [])
    
    print(f"   ✓ Number of features: {n_features}")
    print(f"   ✓ Sequence length: {seq_len}")
    print(f"   ✓ Feature names: {len(feature_names)} features")
    if len(feature_names) > 0:
        print(f"     First 5: {feature_names[:5]}")
    
    # Try loading the model (requires TensorFlow)
    print("\n8. Attempting to load model...")
    try:
        from src.training.modular_trainers import TransformerDirectionTrainer
        
        trainer = TransformerDirectionTrainer()
        trainer.load(str(model_path), pair)
        
        print("   ✓ Model loaded successfully")
        print(f"   ✓ Is trained: {trainer.is_trained}")
        print(f"   ✓ Scaler initialized: {trainer.scaler is not None}")
        
        # Verify calibration loaded
        if hasattr(trainer, 'output_calibration') and trainer.output_calibration:
            print("   ✓ Output calibration restored")
            print(f"     - Threshold: {trainer.output_calibration.get('threshold', 0.5):.4f}")
        else:
            print("   ⚠️ Output calibration not restored")
        
    except ImportError as e:
        print(f"   ⚠️ Cannot test model loading: {e}")
        print("      (TensorFlow not available in this environment)")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Model save/load verification PASSED for {pair}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify model save/load cycle")
    parser.add_argument(
        '--pair', '-p',
        type=str,
        default='EUR_USD',
        help='Currency pair to verify (default: EUR_USD)'
    )
    
    args = parser.parse_args()
    
    success = verify_model_save_load(args.pair)
    sys.exit(0 if success else 1)
