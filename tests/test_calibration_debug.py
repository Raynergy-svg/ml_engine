#!/usr/bin/env python
"""Debug calibration application."""
import pickle
import numpy as np
from pathlib import Path

# Load model and meta
meta_path = Path('trained_data/models/EUR_USD/transformer_direction.meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

# Check calibration
output_calibration = meta.get('output_calibration', {})
print('Calibration:', output_calibration)

# Load model and make a prediction
import keras  # noqa: E402
model_path = Path('trained_data/models/EUR_USD/transformer_direction.keras')
model = keras.models.load_model(str(model_path), compile=False)

# Random input
X = np.random.randn(10, meta['seq_len'], meta['n_features']).astype(np.float32) * 0.1
preds = model.predict(X, verbose=0)
print(f'Raw predictions (10 samples): {preds.flatten()}')
print(f'Raw mean: {preds.mean():.4f}')

# Apply calibration
bias = output_calibration.get('bias', 0.0)
calibrated = preds - bias
calibrated = np.clip(calibrated, 0, 1)
print(f'Calibrated predictions: {calibrated.flatten()}')
print(f'Calibrated mean: {calibrated.mean():.4f}')
print(f'Predictions > 0.5 before: {(preds > 0.5).sum()} / {len(preds)}')
print(f'Predictions > 0.5 after: {(calibrated > 0.5).sum()} / {len(preds)}')
