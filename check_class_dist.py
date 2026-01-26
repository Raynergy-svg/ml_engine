#!/usr/bin/env python3
"""Check class distribution in training data."""
from modular_data_loaders import load_all_modular_data
from oanda_practice import OandaPracticeClient
from fx_paper import candles_to_ohlcv_df
import numpy as np

print('Loading training data...')
client = OandaPracticeClient.from_env()
resp = client.get_candles('EUR_USD', granularity='H1', count=2000, price='MBA')
df = candles_to_ohlcv_df(resp)

all_data = load_all_modular_data(df)
dir_data = all_data['direction']

y_train = dir_data['y_train']
w_train = dir_data.get('w_train')

# Filter to clear labels only (where weight > 0)
clear_mask = (w_train > 0) if w_train is not None else np.ones(len(y_train), dtype=bool)
y_clear = y_train[clear_mask]

print(f'Total samples: {len(y_train)}')
print(f'Clear samples: {len(y_clear)} ({100*len(y_clear)/len(y_train):.1f}%)')
print()
up_count = (y_clear == 1).sum()
down_count = (y_clear == 0).sum()
print(f'UP (y=1): {up_count} ({100*up_count/len(y_clear):.1f}%)')
print(f'DOWN (y=0): {down_count} ({100*down_count/len(y_clear):.1f}%)')
print()
print(f'Imbalance ratio: {max(up_count, down_count) / min(up_count, down_count):.2f}x')
