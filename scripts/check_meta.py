#!/usr/bin/env python3
"""Quick check of model metadata."""
import pickle
from pathlib import Path

meta_path = Path('trained_data/models/transformer_direction.meta.pkl')
if meta_path.exists():
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    print('=== SAVED MODEL METADATA ===')
    print(f"n_features: {meta.get('n_features', 'NOT FOUND')}")
    print(f"seq_len: {meta.get('seq_len', 'NOT FOUND')}")
    
    if 'scaler' in meta:
        scaler = meta['scaler']
        print(f"scaler n_features_in_: {getattr(scaler, 'n_features_in_', 'NOT FOUND')}")
    
    if 'feature_names' in meta:
        print(f"feature_names count: {len(meta['feature_names'])}")
    
    if 'metrics' in meta:
        print(f"metrics: {meta['metrics']}")
    
    if 'lineage' in meta:
        lineage = meta['lineage']
        print(f"lineage instrument: {lineage.get('instrument', 'NOT FOUND')}")
else:
    print('No meta.pkl found')

ewc_path = Path('trained_data/models/transformer_direction.ewc.pkl')
if ewc_path.exists():
    with open(ewc_path, 'rb') as f:
        ewc = pickle.load(f)
    print(f"\nEWC State: n_tasks={ewc.get('n_tasks', 0)}, lambda={ewc.get('ewc_lambda', 0)}")
else:
    print("\nNo EWC state found")
