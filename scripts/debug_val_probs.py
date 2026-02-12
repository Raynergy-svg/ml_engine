#!/usr/bin/env python3
"""Debug validation probabilities."""
import numpy as np
import pickle
from pathlib import Path

# Load model
import keras
model_path = Path('trained_data/models/EUR_USD/transformer_direction.keras')
meta_path = Path('trained_data/models/EUR_USD/transformer_direction.meta.pkl')

print("Loading model...")
transformer = keras.models.load_model(str(model_path), compile=False)
meta = pickle.load(open(meta_path, 'rb'))

seq_len = meta.get('seq_len', 60)
feature_names = meta.get('feature_names', [])
scaler = meta.get('scaler')

print(f"Model expects {len(feature_names)} features, seq_len={seq_len}")
print(f"Scaler type: {type(scaler).__name__}")

# Get data
from src.utils.oanda_practice import OandaPracticeClient  # noqa: E402
from src.utils.fx_paper import candles_to_ohlcv_df  # noqa: E402

client = OandaPracticeClient.from_env()
resp = client.get_candles('EUR_USD', granularity='H1', count=300, price='MBA')
df = candles_to_ohlcv_df(resp)

# Compute features using validation's compute_features
import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parent / 'tests' / 'validation'))
from validate_model import compute_features  # noqa: E402

df_feat = compute_features(df)
print(f"Computed {len(df_feat.columns)} features")

# Check feature coverage
available = set(df_feat.columns)
missing = [f for f in feature_names if f not in available]
print(f"Missing features: {len(missing)}")
if missing:
    print(f"  {missing[:10]}...")

# Make predictions on multiple samples
probs = []
for i in range(seq_len + 10, len(df_feat) - 10, 5):
    X_list = []
    for fname in feature_names:
        if fname in available:
            X_list.append(df_feat[fname].iloc[i-seq_len:i].values)
        else:
            X_list.append(np.zeros(seq_len))
    X = np.column_stack(X_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply scaler
    if scaler is not None:
        X = scaler.transform(X)
    
    X = X.reshape(1, seq_len, -1).astype(np.float32)
    pred = transformer.predict(X, verbose=0)
    prob = float(pred[0, 0])
    probs.append(prob)

print(f"\nProbability statistics ({len(probs)} samples):")
print(f"  Min: {min(probs):.4f}")
print(f"  Max: {max(probs):.4f}")
print(f"  Mean: {np.mean(probs):.4f}")
print(f"  Std: {np.std(probs):.4f}")

# Distribution
n_long = sum(1 for p in probs if p > 0.5)
n_short = sum(1 for p in probs if p <= 0.5)
print("\nPredictions:")
print(f"  LONG (>0.5): {n_long} ({100*n_long/len(probs):.1f}%)")
print(f"  SHORT (<=0.5): {n_short} ({100*n_short/len(probs):.1f}%)")

# Show histogram
print("\nHistogram:")
bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for low, high in bins:
    count = sum(1 for p in probs if low <= p < high)
    bar = '#' * (count * 50 // len(probs)) if count > 0 else ''
    print(f"  [{low:.1f}-{high:.1f}): {count:3d} {bar}")
