#!/usr/bin/env python3
"""Debug gate evaluation."""

import pytest

pytest.skip("Debug-only script (requires OANDA + network + optional deps).", allow_module_level=True)

import pandas as pd
from src.scanner.gates import GateEvaluator
from src.core.modular_data_loaders import compute_normalized_features
from pathlib import Path
import os

# Use OANDA API
from dotenv import load_dotenv
load_dotenv()

import requests  # noqa: E402


def get_candles(instrument, count=100, granularity='H1'):
    """Fetch candles from OANDA."""
    api_token = os.getenv('OANDA_API_TOKEN')
    
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {api_token}"}
    params = {"count": count, "granularity": granularity}
    
    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()
    
    return pd.DataFrame([{
        'time': c['time'],
        'open': float(c['mid']['o']),
        'high': float(c['mid']['h']),
        'low': float(c['mid']['l']),
        'close': float(c['mid']['c']),
        'volume': int(c['volume']),
    } for c in data['candles']])


# Fetch some data
df = get_candles('EUR_USD', count=100)

print('Raw data shape:', df.shape)
print('Columns:', list(df.columns))

# Compute features
features = compute_normalized_features(df)
print('\nFeature columns:', len(features.columns))
print('Has ADX:', 'adx' in features.columns)

# Test gate evaluator
model_dir = Path('trained_data/models')
evaluator = GateEvaluator(model_dir)
evaluator.load_models()

# Test confidence evaluation
print('\n--- Testing Confidence Gate (ADX) ---')
conf = evaluator.evaluate_confidence(features)
print(f'Confidence (ADX): {conf}')

# Debug: check if adx column exists
if 'adx' in features.columns:
    print(f'ADX value from features: {features["adx"].iloc[-1]}')
else:
    print('ADX column not in features, computing manually...')
    adx = evaluator._compute_adx(features)
    print(f'Computed ADX: {adx}')

# Test all gates
print('\n--- Testing All Gates ---')
results = evaluator.evaluate_all_gates(
    features,
    min_confidence=50.0,
    min_momentum=0.20,
    max_drawdown_pct=0.025,
)

for key, value in results.items():
    print(f'  {key}: {value}')
