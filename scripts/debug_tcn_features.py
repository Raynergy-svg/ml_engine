#!/usr/bin/env python3
"""Debug TCN training data quality."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, '/Users/davidcertan/Desktop/ml_engine')
os.chdir('/Users/davidcertan/Desktop/ml_engine')

from src.core.modular_data_loaders import compute_normalized_features
from dotenv import load_dotenv
load_dotenv()

# Use requests directly for OANDA
import requests

def fetch_candles(instrument, granularity, count):
    """Fetch candles from OANDA."""
    token = os.getenv('OANDA_API_TOKEN')
    account = os.getenv('OANDA_ACCOUNT_ID')
    base_url = "https://api-fxpractice.oanda.com"
    
    url = f"{base_url}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"granularity": granularity, "count": count}
    
    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()
    
    rows = []
    for c in data.get('candles', []):
        if c['complete']:
            mid = c['mid']
            rows.append({
                'time': c['time'],
                'open': float(mid['o']),
                'high': float(mid['h']),
                'low': float(mid['l']),
                'close': float(mid['c']),
                'volume': int(c['volume'])
            })
    return pd.DataFrame(rows)

print("=" * 60)
print("TCN FEATURE DIAGNOSTIC")
print("=" * 60)

# Fetch some data
df = fetch_candles('EUR_USD', 'H1', 2000)
print(f'\n1. Raw candles: {len(df)}')

# Compute features
features = compute_normalized_features(df)
print(f'2. Features computed: {features.shape}')
print(f'3. Feature columns: {list(features.columns)[:15]}...')

# Check for NaN/inf
numeric_features = features.select_dtypes(include=[np.number])
nan_pct = numeric_features.isna().mean() * 100
high_nan = nan_pct[nan_pct > 10].sort_values(ascending=False)
print(f'\n4. High NaN columns (>10%): {len(high_nan)}')
if len(high_nan) > 0:
    print(high_nan.head(10))

# Check feature variance
variances = numeric_features.var()
low_var = variances[variances < 0.001].sort_values()
print(f'\n5. Low variance features (<0.001): {len(low_var)}')
if len(low_var) > 0:
    print(low_var.head(10))

# Sample feature stats
print(f'\n6. Sample feature statistics:')
numeric_cols = numeric_features.columns.tolist()
for col in numeric_cols[:8]:
    s = numeric_features[col].dropna()
    print(f'   {col:30s}: mean={s.mean():8.4f}, std={s.std():8.4f}, range=[{s.min():8.4f}, {s.max():8.4f}]')

# Check if features are predictive of future volatility
print(f'\n7. Feature correlation with ATR:')
if 'atr_pct_14' in numeric_features.columns:
    atr = numeric_features['atr_pct_14']
    for col in ['rsi_14', 'macd_line', 'bb_width', 'adx_14']:
        if col in numeric_features.columns:
            corr = numeric_features[col].corr(atr)
            print(f'   {col} vs ATR: {corr:.3f}')

# Check vol_change distribution (the regression target)
print(f'\n8. Volatility change analysis:')
high = df['high']
low = df['low']
close = df['close']
tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
atr_20 = tr.rolling(20).mean()

lookahead = 48
future_atr = atr_20.shift(-lookahead).rolling(lookahead, min_periods=1).mean()
current_atr = atr_20.rolling(lookahead).mean()
vol_change = (future_atr - current_atr) / (current_atr + 1e-10)
vol_change_clean = vol_change.dropna()

print(f'   vol_change samples: {len(vol_change_clean)}')
print(f'   vol_change range: [{vol_change_clean.min():.4f}, {vol_change_clean.max():.4f}]')
print(f'   vol_change mean: {vol_change_clean.mean():.4f}')
print(f'   vol_change std: {vol_change_clean.std():.4f}')

# Check class distribution with current percentiles
for q, s, a in [(20, 45, 70), (25, 50, 75), (15, 40, 65)]:
    quiet_t = np.percentile(vol_change_clean, q)
    stable_t = np.percentile(vol_change_clean, s)
    active_t = np.percentile(vol_change_clean, a)
    
    n_quiet = (vol_change_clean < quiet_t).sum()
    n_stable = ((vol_change_clean >= quiet_t) & (vol_change_clean < stable_t)).sum()
    n_active = ((vol_change_clean >= stable_t) & (vol_change_clean < active_t)).sum()
    n_extreme = (vol_change_clean >= active_t).sum()
    
    total = n_quiet + n_stable + n_active + n_extreme
    print(f'\n   Percentiles {q}/{s}/{a}:')
    print(f'     QUIET: {n_quiet:5d} ({100*n_quiet/total:.1f}%)')
    print(f'     STABLE: {n_stable:5d} ({100*n_stable/total:.1f}%)')
    print(f'     ACTIVE: {n_active:5d} ({100*n_active/total:.1f}%)')
    print(f'     EXTREME: {n_extreme:5d} ({100*n_extreme/total:.1f}%)')

print("\n" + "=" * 60)
