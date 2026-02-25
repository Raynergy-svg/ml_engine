#!/usr/bin/env python3
"""Fetch best parameters from wandb sweeps."""

import wandb
import json

api = wandb.Api()

entity = 'tencylinder8310-smartdebtflow-com'
project = 'ml_engine_fx'
sweeps = {
    'AUD_JPY': 'q4ufao2c',
    'AUD_NZD': 'rivyiff6',
    'GBP_JPY': 'eksbr3ub',
    'EUR_GBP': 'zn7e8fop',
    'USD_CAD': 'm738sn9x',
    'AUD_USD': 'f3t6673s',
    'GBP_AUD': 'ldjkqfhk'
}

results = {}

for pair, sweep_id in sweeps.items():
    print(f'\nFetching {pair}...')
    try:
        sweep = api.sweep(f'{entity}/{project}/{sweep_id}')
        
        # Get all runs and find best by validation accuracy
        best_run = None
        best_acc = -1
        
        for run in sweep.runs:
            # Try different metric names
            acc = (run.summary.get('val_accuracy') or 
                   run.summary.get('val_acc') or 
                   run.summary.get('best_val_accuracy') or 
                   run.summary.get('accuracy') or
                   run.summary.get('final_val_accuracy'))
            
            if acc is not None:
                try:
                    acc_val = float(acc)
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best_run = run
                except (ValueError, TypeError):
                    pass
        
        if best_run:
            params = {k: v for k, v in best_run.config.items() if not k.startswith('_')}
            results[pair] = {
                'run_id': best_run.id,
                'run_name': best_run.name,
                'best_accuracy': best_acc,
                'parameters': params
            }
            print(f'  Best accuracy: {best_acc:.4f}')
            print(f'  Run: {best_run.name}')
        else:
            print(f'  No runs with accuracy metrics found')
            # Get first run to see available metrics
            runs = list(sweep.runs)
            if runs:
                print(f'  Available metrics: {list(runs[0].summary.keys())}')
            
    except Exception as e:
        print(f'  Error: {e}')

# Output results
print('\n' + '='*60)
print('BEST PARAMETERS SUMMARY')
print('='*60)

for pair, data in results.items():
    print(f'\n### {pair} ###')
    print(f"Best Accuracy: {data['best_accuracy']:.4f}")
    print(f"Run ID: {data['run_id']}")
    print('Parameters:')
    for k, v in sorted(data['parameters'].items()):
        print(f'  {k}: {v}')

# Save to JSON
with open('best_sweep_params.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved to best_sweep_params.json')
