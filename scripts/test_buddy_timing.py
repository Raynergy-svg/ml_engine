#!/usr/bin/env python3
"""
Test full buddy inference with timing to find bottleneck.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import time

print('Testing full buddy EUR_USD inference with timing...')
t_start = time.perf_counter()

# Import ensemble
from src.core.modular_inference import ModularEnsembleInference
print(f'[{time.perf_counter()-t_start:.1f}s] Imported modular_inference')

# Create and load
ensemble = ModularEnsembleInference(instrument='EUR_USD', use_rl_sizer=True, enable_llm_integration=True)
print(f'[{time.perf_counter()-t_start:.1f}s] Created ensemble')

ensemble.load_models()
print(f'[{time.perf_counter()-t_start:.1f}s] Loaded models')

# Fetch data
from src.utils.oanda_practice import OandaPracticeClient
from fx_paper import candles_to_ohlcv_df
client = OandaPracticeClient.from_env()
resp = client.get_candles('EUR_USD', granularity='H1', count=500, price='MBA')
df = candles_to_ohlcv_df(resp)
print(f'[{time.perf_counter()-t_start:.1f}s] Fetched {len(df)} candles')

# Feature engineering
from feature_engineering import FeatureEngineering
from src.utils import load_config
import numpy as np
cfg = load_config('config/config_intel_optimized.yaml')
fe = FeatureEngineering(cfg.get('feature_engineering', {}))
df = fe.create_features(df, include_all=True)
df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
print(f'[{time.perf_counter()-t_start:.1f}s] Features created ({len(df.columns)} cols)')

# Inference
result = ensemble.predict_verbose(df, instrument='EUR_USD')
print(f'[{time.perf_counter()-t_start:.1f}s] Inference complete')
print(f'Decision: {result.get("decision", "N/A")[:60]}...')
