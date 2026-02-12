#!/usr/bin/env python
"""Debug validation predictions to see actual values."""

import pytest

pytest.skip("Debug-only script (requires keras/OANDA/local artifacts).", allow_module_level=True)

import pickle
import numpy as np
from pathlib import Path

# Load model and meta
meta_path = Path('trained_data/models/EUR_USD/transformer_direction.meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

print('Calibration:', meta.get('output_calibration', {}))
print('seq_len:', meta.get('seq_len'))
print('n_features:', meta.get('n_features'))
print('feature_names count:', len(meta.get('feature_names', [])))

# Load model
import keras  # noqa: E402
model_path = Path('trained_data/models/EUR_USD/transformer_direction.keras')
model = keras.models.load_model(str(model_path), compile=False)

# Load real data
from oanda_practice import OandaPracticeClient  # noqa: E402
from fx_paper import candles_to_ohlcv_df  # noqa: E402
from src.core.modular_data_loaders import compute_normalized_features  # noqa: E402
from src.data.feature_engineering import FeatureEngineering  # noqa: E402

client = OandaPracticeClient.from_env()
resp = client.get_candles('EUR_USD', granularity='H1', count=800, price='MBA')
df_raw = candles_to_ohlcv_df(resp)

# Compute features
df_norm = compute_normalized_features(df_raw.copy())
fe = FeatureEngineering()
df_fe = fe.create_features(df_raw.copy())

# Merge
df_feat = df_norm.copy()
for col in df_fe.columns:
    if col not in df_feat.columns:
        df_feat = df_feat.join(df_fe[[col]], how='left')

df_feat = df_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

print(f'Loaded {len(df_raw)} candles, {len(df_feat)} feature rows')
print(f'Feature columns: {len(df_feat.columns)}')

# Get feature order from meta
feature_names = meta.get('feature_names', [])
scaler = meta.get('scaler')
seq_len = meta.get('seq_len', 60)

# Make predictions on last 100 samples
predictions = []
for i in range(len(df_feat) - 100, len(df_feat)):
    if i < seq_len:
        continue
    
    # Build feature matrix in correct order
    X_list = []
    for fname in feature_names:
        if fname in df_feat.columns:
            X_list.append(df_feat[fname].iloc[i-seq_len:i].values)
        else:
            X_list.append(np.zeros(seq_len))
    X = np.column_stack(X_list)
    
    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict
    X = X.reshape(1, seq_len, -1).astype(np.float32)
    pred = model.predict(X, verbose=0)
    prob = float(pred[0, 0]) if pred.shape[-1] == 1 else float(pred[0, 1])
    predictions.append(prob)

predictions = np.array(predictions)
print(f'\nValidation predictions ({len(predictions)} samples):')
print(f'  Mean: {predictions.mean():.4f}')
print(f'  Std: {predictions.std():.4f}')
print(f'  Min: {predictions.min():.4f}')
print(f'  Max: {predictions.max():.4f}')
print(f'  > 0.5: {(predictions > 0.5).sum()} / {len(predictions)} ({100*(predictions > 0.5).mean():.1f}%)')

# Apply calibration
bias = meta.get('output_calibration', {}).get('bias', 0.0)
calibrated = predictions - bias
calibrated = np.clip(calibrated, 0, 1)
print(f'\nCalibrated (bias={bias:.4f}):')
print(f'  Mean: {calibrated.mean():.4f}')
print(f'  > 0.5: {(calibrated > 0.5).sum()} / {len(calibrated)} ({100*(calibrated > 0.5).mean():.1f}%)')
