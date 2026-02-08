#!/usr/bin/env python3
"""Check if features correlate with future volatility labels."""
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from src.core.modular_data_loaders import compute_normalized_features  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
import requests  # noqa: E402
load_dotenv()


def fetch_candles(instrument, granularity, count):
    token = os.getenv('OANDA_API_TOKEN')
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
print("FEATURE PREDICTIVENESS CHECK")
print("=" * 60)

# Fetch data
df = fetch_candles('EUR_USD', 'H1', 5000)
print(f'\n1. Raw candles: {len(df)}')

# Compute features
features = compute_normalized_features(df)
print(f'2. Features: {features.shape}')

# Create forward-looking labels (same as training)
high = df['high']
low = df['low']
close = df['close']
tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
atr_20 = tr.rolling(20).mean()

lookahead = 48
future_atr = atr_20.shift(-lookahead).rolling(lookahead, min_periods=1).mean()
current_atr = atr_20.rolling(lookahead).mean()
vol_change = (future_atr - current_atr) / (current_atr + 1e-10)

# Labels using 20/45/70 percentiles
vol_change_clean = vol_change.dropna()
quiet_t = np.percentile(vol_change_clean, 20)
stable_t = np.percentile(vol_change_clean, 45)
active_t = np.percentile(vol_change_clean, 70)

labels = np.ones(len(df), dtype=int)
labels[vol_change < quiet_t] = 0
labels[(vol_change >= stable_t) & (vol_change < active_t)] = 2
labels[vol_change >= active_t] = 3
labels[-lookahead:] = -1  # Invalid

# Get numeric features
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
exclude = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [c for c in numeric_cols if c not in exclude]

# Prepare data
X = features[feature_cols].values
y = labels

# Remove invalid labels
valid_mask = (y >= 0) & (~np.isnan(X).any(axis=1)) & (~np.isinf(X).any(axis=1))
X = X[valid_mask]
y = y[valid_mask]

print(f'3. Valid samples: {len(X)}')
print(f'4. Class distribution: {np.bincount(y)}')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simple train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

print(f'\n5. Train: {len(X_train)}, Test: {len(X_test)}')

# Try Random Forest (fast, good baseline)
print('\n6. Training Random Forest baseline...')
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)
print(f'   Train accuracy: {train_acc:.1%}')
print(f'   Test accuracy:  {test_acc:.1%}')
print('   Random baseline: 25.0%')

# Feature importance
print('\n7. Top 10 most important features:')
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, imp in importances.head(10).items():
    print(f'   {feat:30s}: {imp:.4f}')

# Check per-class accuracy
from sklearn.metrics import classification_report  # noqa: E402
y_pred = rf.predict(X_test)
print('\n8. Per-class performance:')
print(classification_report(y_test, y_pred, target_names=['QUIET', 'STABLE', 'ACTIVE', 'EXTREME']))

# Try with different lookahead
print('\n9. Trying different lookahead values...')
for la in [12, 24, 48, 96]:
    future_atr_la = atr_20.shift(-la).rolling(la, min_periods=1).mean()
    current_atr_la = atr_20.rolling(la).mean()
    vol_change_la = (future_atr_la - current_atr_la) / (current_atr_la + 1e-10)
    
    vc_clean = vol_change_la.dropna()
    qt = np.percentile(vc_clean, 20)
    st = np.percentile(vc_clean, 45)
    at = np.percentile(vc_clean, 70)
    
    labels_la = np.ones(len(df), dtype=int)
    labels_la[vol_change_la < qt] = 0
    labels_la[(vol_change_la >= st) & (vol_change_la < at)] = 2
    labels_la[vol_change_la >= at] = 3
    labels_la[-la:] = -1
    
    y_la = labels_la[valid_mask[:len(labels_la)]]
    mask_la = y_la >= 0
    
    if mask_la.sum() > 100:
        X_t, X_v, y_t, y_v = train_test_split(X_scaled[mask_la], y_la[mask_la], test_size=0.2, shuffle=False)
        rf_la = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf_la.fit(X_t, y_t)
        acc = rf_la.score(X_v, y_v)
        print(f'   Lookahead {la:3d} bars: test acc = {acc:.1%}')

print("\n" + "=" * 60)
