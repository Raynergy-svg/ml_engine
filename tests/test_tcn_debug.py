#!/usr/bin/env python3
"""Debug TCN volatility regime evaluation."""

import pytest

pytest.skip("Debug-only script (requires OANDA + local models).", allow_module_level=True)

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(message)s')

from src.scanner.engine import Scanner  # noqa: E402
from src.scanner.config import ScannerConfig  # noqa: E402
from src.core.modular_data_loaders import compute_normalized_features  # noqa: E402

config = ScannerConfig()
scanner = Scanner(config)

# Initialize gate evaluator
scanner._init_gate_evaluator()

# Get data from OANDA (same as scanner does)
from src.utils.oanda_practice import OandaPracticeClient  # noqa: E402
oanda = OandaPracticeClient.from_env()
raw = oanda.get_candles(instrument='EUR_USD', granularity='H1', count=500)

# Parse OANDA JSON response into DataFrame
candles = raw.get("candles", [])
data = []
for c in candles:
    if not c.get("complete", True):
        continue
    mid = c.get("mid", {})
    data.append({
        "time": c.get("time"),
        "open": float(mid.get("o", 0)),
        "high": float(mid.get("h", 0)),
        "low": float(mid.get("l", 0)),
        "close": float(mid.get("c", 0)),
        "volume": int(c.get("volume", 0)),
    })

import pandas as pd  # noqa: E402
df = pd.DataFrame(data)
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")
print(f'Fetched {len(df)} candles')

# Compute features
df_feat = compute_normalized_features(df)
print(f'Feature columns: {df_feat.shape[1]}')

# Check volatility features available
vol_features = ['atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
                'volatility_5', 'volatility_10', 'volatility_20',
                'returns_1', 'returns_5', 'returns_10',
                'volume_ratio_10', 'volume_ratio_20']
available = [f for f in vol_features if f in df_feat.columns]
missing = [f for f in vol_features if f not in df_feat.columns]
print(f'Available volatility features ({len(available)}): {available}')
print(f'Missing volatility features ({len(missing)}): {missing}')

# Call evaluate_volatility_regime directly
regime, conf, allowed = scanner._gate_evaluator.evaluate_volatility_regime(df_feat)
print(f'\nVolatility regime: {regime}')
print(f'Confidence: {conf}')
print(f'Trade allowed: {allowed}')
