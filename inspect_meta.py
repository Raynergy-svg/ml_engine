#!/usr/bin/env python3
"""Inspect meta.pkl file to see expected features vs current."""
import pickle
import os
import sys
import numpy as np
import pandas as pd

# Check meta file
meta_path = 'trained_data/models/USD_JPY/transformer_direction.meta.pkl'

print("="*60)
print("1. CHECKPOINT META FILE ANALYSIS")
print("="*60)

if not os.path.exists(meta_path):
    print(f'Meta file not found at {meta_path}')
    import glob
    metas = glob.glob('trained_data/models/**/*.meta.*', recursive=True)
    print('Found meta files:')
    for m in metas[:20]:
        print(f'  {m}')
    sys.exit(1)

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

print(f'Keys: {list(meta.keys())}')
print()

checkpoint_features = []
if 'feature_names' in meta:
    checkpoint_features = meta['feature_names']
    print(f'CHECKPOINT Feature count: {len(checkpoint_features)}')
    print('Feature names (first 20):')
    for i, f in enumerate(checkpoint_features[:20]):
        print(f'  {i}: {f}')
    if len(checkpoint_features) > 20:
        print(f'  ... and {len(checkpoint_features) - 20} more')
    print()

scaler_features = None
if 'scaler' in meta:
    scaler = meta['scaler']
    print(f'Scaler type: {type(scaler).__name__}')
    if hasattr(scaler, 'n_features_in_'):
        scaler_features = scaler.n_features_in_
        print(f'Scaler n_features_in_: {scaler_features}')
    if hasattr(scaler, 'mean_'):
        print(f'Scaler mean_ shape: {scaler.mean_.shape}')
    print()

# Print metrics
if 'metrics' in meta:
    print('Saved metrics:')
    for k, v in meta['metrics'].items():
        print(f'  {k}: {v}')
    print()

print("="*60)
print("2. CURRENT FEATURE GENERATION")
print("="*60)

# Now compute current features to compare
from src.core.modular_data_loaders import compute_normalized_features, load_direction_data

# Load sample data
csv_path = 'market_data/USD_JPY_H1.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, nrows=5000)
    print(f'Loaded {len(df)} rows from {csv_path}')
    
    # Compute features
    df_norm = compute_normalized_features(df)
    
    # Get direction data to see final feature selection
    direction_data = load_direction_data(df_norm, threshold=0.0)
    current_features = direction_data['feature_names']
    
    print(f'CURRENT Feature count: {len(current_features)}')
    print('Feature names (first 20):')
    for i, f in enumerate(current_features[:20]):
        print(f'  {i}: {f}')
    if len(current_features) > 20:
        print(f'  ... and {len(current_features) - 20} more')
    print()
else:
    print(f'No CSV at {csv_path}, trying another...')
    import glob
    csvs = glob.glob('market_data/*.csv')
    if csvs:
        csv_path = csvs[0]
        df = pd.read_csv(csv_path, nrows=5000)
        print(f'Loaded {len(df)} rows from {csv_path}')
        df_norm = compute_normalized_features(df)
        direction_data = load_direction_data(df_norm, threshold=0.0)
        current_features = direction_data['feature_names']
        print(f'CURRENT Feature count: {len(current_features)}')
    else:
        print('No CSV files found!')
        current_features = []

print("="*60)
print("3. FEATURE COMPARISON")
print("="*60)

if checkpoint_features and current_features:
    cp_set = set(checkpoint_features)
    curr_set = set(current_features)
    
    # Features in checkpoint but not current (model expects these!)
    missing_from_current = cp_set - curr_set
    # Features in current but not checkpoint (will be ignored by model)
    new_in_current = curr_set - cp_set
    # Features in both
    common = cp_set & curr_set
    
    print(f'Total in checkpoint: {len(checkpoint_features)}')
    print(f'Total in current:    {len(current_features)}')
    print(f'Common features:     {len(common)}')
    print()
    
    if missing_from_current:
        print(f'❌ MISSING from current (checkpoint expects but not provided): {len(missing_from_current)}')
        for f in sorted(missing_from_current)[:20]:
            print(f'  - {f}')
        if len(missing_from_current) > 20:
            print(f'  ... and {len(missing_from_current) - 20} more')
        print()
    
    if new_in_current:
        print(f'⚠️  NEW in current (not in checkpoint, will be passed to wrong input): {len(new_in_current)}')
        for f in sorted(new_in_current)[:20]:
            print(f'  + {f}')
        if len(new_in_current) > 20:
            print(f'  ... and {len(new_in_current) - 20} more')
        print()
    
    # Check order mismatch
    if len(checkpoint_features) == len(current_features):
        order_mismatches = []
        for i, (cp_f, curr_f) in enumerate(zip(checkpoint_features, current_features)):
            if cp_f != curr_f:
                order_mismatches.append((i, cp_f, curr_f))
        
        if order_mismatches:
            print(f'❌ ORDER MISMATCH: Features at same index have different names!')
            print(f'   First 10 mismatches:')
            for i, cp_f, curr_f in order_mismatches[:10]:
                print(f'   Index {i}: checkpoint="{cp_f}" vs current="{curr_f}"')
            print()
    
    if len(common) == len(checkpoint_features) == len(current_features):
        # Check exact order
        if checkpoint_features == current_features:
            print('✓ EXACT MATCH: Features and order are identical')
        else:
            print('❌ ORDER MISMATCH: Same features but different order!')
    
    # SUMMARY
    print("="*60)
    print("4. DIAGNOSIS")
    print("="*60)
    
    if len(checkpoint_features) != len(current_features):
        print(f'🔴 CRITICAL: Feature count mismatch!')
        print(f'   Checkpoint model expects {len(checkpoint_features)} features')
        print(f'   Current data provides {len(current_features)} features')
        print()
        print('   This WILL cause warm-start degradation because:')
        print('   - If current < checkpoint: model receives incomplete input')
        print('   - If current > checkpoint: extra features go to wrong neurons')
        print()
        print('   FIX: The training code must align features before warm-start!')
    elif checkpoint_features != current_features:
        print(f'🟠 WARNING: Feature names/order mismatch')
        print(f'   Same count ({len(checkpoint_features)}) but different features')
        print()
        print('   This causes warm-start degradation because neurons')
        print('   trained on feature X now receive feature Y')
    else:
        print('✓ Features match perfectly - warm-start should work')
else:
    print('Cannot compare - missing features from one source')
