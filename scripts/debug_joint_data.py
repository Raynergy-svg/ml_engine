#!/usr/bin/env python3
"""Debug joint training data distribution."""

from src.utils.oanda_practice import OandaPracticeClient
from src.core.modular_data_loaders import compute_normalized_features, load_direction_data
import pandas as pd
import numpy as np

# Fetch data like joint training does
client = OandaPracticeClient.from_env()

all_rows = []
remaining = 12000
MAX = 5000
to_time = None

while remaining > 0:
    batch_size = min(remaining, MAX)
    params = {'granularity': 'H1', 'count': batch_size, 'price': 'MBA'}
    if to_time:
        params['to_time'] = to_time
    response = client.get_candles('USD_JPY', **params)
    candles = response.get('candles', [])
    if not candles:
        break
    for c in candles:
        all_rows.append({
            'time': c.get('time'),
            'open': float(c.get('mid', {}).get('o', 0)),
            'high': float(c.get('mid', {}).get('h', 0)),
            'low': float(c.get('mid', {}).get('l', 0)),
            'close': float(c.get('mid', {}).get('c', 0)),
            'volume': int(c.get('volume', 0)),
        })
    to_time = candles[0].get('time')
    remaining -= len(candles)
    if len(candles) < batch_size:
        break

df = pd.DataFrame(all_rows)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df.set_index('time', inplace=True)
df = compute_normalized_features(df)

print(f'Total rows: {len(df)}')

# Check what load_direction_data returns
result = load_direction_data(df)

print(f"X_train shape: {result['X_train'].shape}")
print(f"y_train unique values: {np.unique(result['y_train'], return_counts=True)}")

# Check train labels
y = result['y_train']
n_up = (y == 1).sum()
n_down = (y == 0).sum()
n_unclear = ((y != 0) & (y != 1)).sum()

print(f'Train labels: up={n_up} ({n_up/len(y)*100:.1f}%), down={n_down} ({n_down/len(y)*100:.1f}%), unclear={n_unclear}')

# Check val labels
y_val = result['y_val']
n_up_val = (y_val == 1).sum()
n_down_val = (y_val == 0).sum()
print(f'Val labels: up={n_up_val}, down={n_down_val}')

# Check if weights exist
if 'w_train' in result:
    w = result['w_train']
    print(f'Weights: non-zero={np.sum(w > 0)}, mean={np.mean(w):.3f}')
