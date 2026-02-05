#!/usr/bin/env python
"""Debug TCN feature availability."""
import pandas as pd
import pickle
import numpy as np

from src.core.modular_data_loaders import compute_normalized_features

# Create synthetic OHLCV data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'time': pd.date_range('2024-01-01', periods=n, freq='H'),
    'open': 1.0 + np.cumsum(np.random.randn(n) * 0.001),
    'high': None,
    'low': None,
    'close': None,
    'volume': np.random.randint(1000, 10000, n).astype(float),
})
df['high'] = df['open'] + np.abs(np.random.randn(n) * 0.002)
df['low'] = df['open'] - np.abs(np.random.randn(n) * 0.002)
df['close'] = df['open'] + np.random.randn(n) * 0.001

print(f'Raw columns: {list(df.columns)}')
print(f'Raw shape: {df.shape}')

# Compute features
df_feat = compute_normalized_features(df.copy())
print(f'\nFeature columns: {len(df_feat.columns)} columns')
print(f'Feature shape: {df_feat.shape}')

# Check for expected TCN features
with open('trained_data/models/joint/tcn_volatility_regime.meta.pkl', 'rb') as f:
    meta = pickle.load(f)
expected = meta['feature_names']

missing = [f for f in expected if f not in df_feat.columns]
print(f'\nMissing features: {len(missing)}')
if missing:
    print(f'  {missing}')
    
available = [f for f in expected if f in df_feat.columns]
print(f'\nAvailable expected features: {len(available)}/{len(expected)}')
print(f'All df_feat columns:\n  {list(df_feat.columns)}')
