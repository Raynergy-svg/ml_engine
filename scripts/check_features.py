import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from src.data.feature_engineering import FeatureEngineering
from src.core.modular_data_loaders import compute_normalized_features

# Create dummy OHLCV
dates = pd.date_range('2024-01-01', periods=300, freq='h')
df = pd.DataFrame({
    'open': 1.1 + np.random.randn(300) * 0.01,
    'high': 1.1 + np.random.randn(300) * 0.01 + 0.005,
    'low': 1.1 + np.random.randn(300) * 0.01 - 0.005,
    'close': 1.1 + np.random.randn(300) * 0.01,
    'volume': 1000 + np.random.randint(-100, 100, 300),
}, index=dates)

# First: compute_normalized_features
df_norm = compute_normalized_features(df.copy())
print(f"compute_normalized_features: {len(df_norm.columns)} columns")

# Second: FeatureEngineering
fe = FeatureEngineering({})
df_fe = fe.create_features(df.copy(), include_all=True)
print(f"FeatureEngineering: {len(df_fe.columns)} columns")

# Combine both (need to handle index/length differences)
# Both should have same length rows, just use column union
available_cols = set(df_norm.columns) | set(df_fe.columns)
print(f"Combined: {len(available_cols)} columns")

# Load model features
meta_path = Path('trained_data/models/EUR_USD/transformer_direction.meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

model_features = meta.get('feature_names', [])

print(f'Feature engineering produces {len(available_cols)} columns')
print(f'Model expects {len(model_features)} features\n')

missing = []
present = []
for f in model_features:
    if f in available_cols:
        present.append(f)
    else:
        missing.append(f)

print(f'PRESENT: {len(present)}/{len(model_features)}')
print(f'MISSING: {len(missing)}')
if missing:
    print('\nMissing features:')
    for m in missing[:20]:
        print(f'  - {m}')
    if len(missing) > 20:
        print(f'  ... and {len(missing) - 20} more')
