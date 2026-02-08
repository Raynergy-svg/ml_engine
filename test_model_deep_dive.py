#!/usr/bin/env python
"""Deep dive into what the model receives vs outputs."""
import pickle
import numpy as np
from pathlib import Path

# Load model and meta
meta_path = Path('trained_data/models/EUR_USD/transformer_direction.meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

scaler = meta.get('scaler')
feature_names = meta.get('feature_names', [])
seq_len = meta.get('seq_len', 60)

# Load model
import keras  # noqa: E402
model_path = Path('trained_data/models/EUR_USD/transformer_direction.keras')
model = keras.models.load_model(str(model_path), compile=False)

print(f"Feature names: {len(feature_names)}")
print(f"Seq len: {seq_len}")
print(f"Scaler type: {type(scaler)}")

# Test 1: What does the model output for random normal input?
print("\n=== Test 1: Random Normal Input (no scaling) ===")
X_random = np.random.randn(100, seq_len, len(feature_names)).astype(np.float32)
preds = model.predict(X_random, verbose=0)
print(f"Random input: mean={preds.mean():.4f}, std={preds.std():.4f}")
print(f"  > 0.5: {(preds > 0.5).sum()}/{len(preds)}")

# Test 2: What does the model output for all zeros?
print("\n=== Test 2: All Zeros Input ===")
X_zeros = np.zeros((100, seq_len, len(feature_names)), dtype=np.float32)
preds = model.predict(X_zeros, verbose=0)
print(f"Zero input: mean={preds.mean():.4f}, std={preds.std():.4f}")
print(f"  > 0.5: {(preds > 0.5).sum()}/{len(preds)}")

# Test 3: What does the model output for all ones?
print("\n=== Test 3: All Ones Input ===")
X_ones = np.ones((100, seq_len, len(feature_names)), dtype=np.float32)
preds = model.predict(X_ones, verbose=0)
print(f"Ones input: mean={preds.mean():.4f}, std={preds.std():.4f}")
print(f"  > 0.5: {(preds > 0.5).sum()}/{len(preds)}")

# Test 4: What does the model output for scaled random?
print("\n=== Test 4: Random Input (after scaling) ===")
if scaler is not None:
    # Random input scaled like training data
    X_raw = np.random.randn(100, seq_len, len(feature_names)).astype(np.float32)
    X_scaled = np.array([scaler.transform(x) for x in X_raw]).astype(np.float32)
    preds = model.predict(X_scaled, verbose=0)
    print(f"Scaled random: mean={preds.mean():.4f}, std={preds.std():.4f}")
    print(f"  > 0.5: {(preds > 0.5).sum()}/{len(preds)}")
    print(f"  Input after scaling: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
else:
    print("No scaler available")

# Test 5: Check the scaler parameters
print("\n=== Test 5: Scaler Parameters ===")
if scaler is not None and hasattr(scaler, 'mean_'):
    print(f"Scaler mean: shape={scaler.mean_.shape}, mean={scaler.mean_.mean():.4f}, min={scaler.mean_.min():.4f}, max={scaler.mean_.max():.4f}")
    print(f"Scaler scale: shape={scaler.scale_.shape}, mean={scaler.scale_.mean():.4f}, min={scaler.scale_.min():.4f}, max={scaler.scale_.max():.4f}")
else:
    print("Scaler doesn't have mean_/scale_ (not StandardScaler?)")

# Test 6: Manual sigmoid bias test
print("\n=== Test 6: Testing if output layer is stuck ===")
# Get the output layer weights
output_layer = None
for layer in model.layers:
    if layer.name == 'direction':
        output_layer = layer
        break

if output_layer:
    weights, bias = output_layer.get_weights()
    print(f"Output layer weights: shape={weights.shape}, mean={weights.mean():.6f}, std={weights.std():.6f}")
    print(f"Output layer bias: {bias}")
    
    # Check what sigmoid(bias) would be
    sigmoid_bias = 1 / (1 + np.exp(-bias[0]))
    print(f"sigmoid(bias) = {sigmoid_bias:.4f}")
    
    # Check the range of possible outputs
    # If input to sigmoid is always negative, output will be < 0.5
    # Input to sigmoid = dot(prev_layer_output, weights) + bias
