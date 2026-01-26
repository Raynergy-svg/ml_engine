#!/usr/bin/env python
"""Check intermediate layer outputs."""
import pickle
import numpy as np
from pathlib import Path
import keras
import tensorflow as tf

# Load model and meta
meta_path = Path('trained_data/models/EUR_USD/transformer_direction.meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

model_path = Path('trained_data/models/EUR_USD/transformer_direction.keras')
model = keras.models.load_model(str(model_path), compile=False)

# Print model summary
print("=== Model Architecture ===")
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'output_shape'):
        print(f"{i}: {layer.name} -> {layer.output_shape}")

# Create intermediate model to see pre-sigmoid output
# Find the Dense layer before 'direction' (which applies sigmoid)
for i, layer in enumerate(model.layers):
    if layer.name == 'direction':
        # The input to this layer is what we want
        break

# Get the layer before direction (it's the dropout after the 16-unit dense)
# Let's find it
print("\n=== Finding pre-sigmoid layer ===")
direction_layer = model.get_layer('direction')
print(f"direction layer input: {direction_layer.input}")

# Create model that outputs the pre-sigmoid logits
# We'll manually compute what direction layer sees
intermediate_model = keras.Model(
    inputs=model.input,
    outputs=direction_layer.input  # The input to direction layer
)

# Test with random input
seq_len = meta.get('seq_len', 60)
n_features = len(meta.get('feature_names', []))

print("\n=== Intermediate Layer Output (pre-sigmoid input) ===")
X_random = np.random.randn(100, seq_len, n_features).astype(np.float32) * 0.1
intermediate_output = intermediate_model.predict(X_random, verbose=0)
print(f"Pre-sigmoid input shape: {intermediate_output.shape}")
print(f"Pre-sigmoid input stats: mean={intermediate_output.mean():.4f}, std={intermediate_output.std():.4f}")
print(f"Pre-sigmoid input range: [{intermediate_output.min():.4f}, {intermediate_output.max():.4f}]")

# What are the weights of direction layer?
weights, bias = direction_layer.get_weights()
print(f"\nDirection layer weights: shape={weights.shape}")
print(f"Direction layer weights: {weights.flatten()}")
print(f"Direction layer bias: {bias}")

# Manually compute pre-sigmoid value
pre_sigmoid = np.dot(intermediate_output, weights) + bias
print(f"\nManual pre-sigmoid: mean={pre_sigmoid.mean():.4f}, std={pre_sigmoid.std():.4f}")
print(f"Manual pre-sigmoid range: [{pre_sigmoid.min():.4f}, {pre_sigmoid.max():.4f}]")

# Sigmoid of this
sigmoid_output = 1 / (1 + np.exp(-pre_sigmoid))
print(f"\nSigmoid output: mean={sigmoid_output.mean():.4f}, std={sigmoid_output.std():.4f}")
print(f"Sigmoid output range: [{sigmoid_output.min():.4f}, {sigmoid_output.max():.4f}]")
print(f"> 0.5: {(sigmoid_output > 0.5).sum()}/{len(sigmoid_output)}")

# Compare to model output
model_output = model.predict(X_random, verbose=0)
print(f"\nModel output: mean={model_output.mean():.4f} (should match sigmoid output)")
