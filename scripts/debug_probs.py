#!/usr/bin/env python3
"""Debug model output probabilities."""
import numpy as np
import pickle
from pathlib import Path
import keras

# Load model
model_path = Path('trained_data/models/EUR_USD/transformer_direction.keras')
meta_path = Path('trained_data/models/EUR_USD/transformer_direction.meta.pkl')

transformer = keras.models.load_model(str(model_path), compile=False)
meta = pickle.load(open(meta_path, 'rb'))

seq_len = meta.get('seq_len', 60)
feature_names = meta.get('feature_names', [])
scaler = meta.get('scaler')

print(f"Model seq_len: {seq_len}, features: {len(feature_names)}")

# Get data
from src.utils.oanda_practice import OandaPracticeClient  # noqa: E402
from src.utils.fx_paper import candles_to_ohlcv_df  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / 'tests' / 'validation'))
from validate_model import compute_features  # noqa: E402

client = OandaPracticeClient.from_env()
resp = client.get_candles('EUR_USD', granularity='H1', count=200, price='MBA')
df = candles_to_ohlcv_df(resp)
df_feat = compute_features(df)

print(f"Data: {len(df_feat)} rows, {len(df_feat.columns)} columns")

# Make predictions
probs = []
for i in range(seq_len + 10, len(df_feat) - 10, 10):
    available = df_feat.columns.tolist()
    X_list = []
    for fname in feature_names:
        if fname in available:
            X_list.append(df_feat[fname].iloc[i-seq_len:i].values)
        else:
            X_list.append(np.zeros(seq_len))
    X = np.column_stack(X_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if scaler is not None:
        X = scaler.transform(X)
    
    X = X.reshape(1, seq_len, -1).astype(np.float32)
    pred = transformer.predict(X, verbose=0)
    prob = float(pred[0, 0])
    probs.append(prob)

print()
print('Raw probability distribution:')
print(f'  Min: {min(probs):.4f}')
print(f'  Max: {max(probs):.4f}')
print(f'  Mean: {np.mean(probs):.4f}')
print(f'  Std: {np.std(probs):.4f}')
print()
print(f'All probs: {[f"{p:.3f}" for p in probs]}')
print()
longs = sum(1 for p in probs if p > 0.5)
shorts = sum(1 for p in probs if p <= 0.5)
print(f'LONG (>0.5): {longs}, SHORT (<=0.5): {shorts}')
