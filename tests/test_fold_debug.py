#!/usr/bin/env python3
"""Debug why folds fail in candle_optimizer."""
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.utils.utils import load_config  # noqa: E402
from src.core.modular_data_loaders import compute_normalized_features, load_direction_data  # noqa: E402
import pandas as pd  # noqa: E402
import glob  # noqa: E402

cfg = load_config('config/config_intel_optimized.yaml')

csvs = sorted(glob.glob('market_data/oanda_EUR_USD_H1_live_*.csv'))
if csvs:
    df = pd.read_csv(csvs[-1])
    print(f'Loaded {len(df)} rows from {csvs[-1]}')
else:
    print('No CSV found')
    sys.exit(1)

if 'returns_1' not in df.columns:
    df = compute_normalized_features(df)

direction_cfg = cfg.get('direction', {})
lookahead = int(direction_cfg.get('lookahead', cfg.get('direction_lookahead', 24)))
threshold = float(direction_cfg.get('threshold', cfg.get('direction_threshold', 0.003)))
data = load_direction_data(df, lookahead=lookahead, threshold=threshold)

x_all = np.concatenate([data['X_train'], data['X_val']], axis=0)
y_all = np.concatenate([data['y_train'], data['y_val']], axis=0)
w_train = data.get('w_train')
w_val_data = data.get('w_val', np.ones(len(data['y_val'])))
w_all = np.concatenate([w_train, w_val_data], axis=0) if w_train is not None else None

print(f'x_all shape: {x_all.shape}, y_all shape: {y_all.shape}')

from src.training.walkforward_validation import WalkForwardValidator  # noqa: E402
validator = WalkForwardValidator(n_splits=3, train_size=0.60, val_size=0.15, test_size=0.10, gap=60, mode='expanding', min_train_size=200)

for fold_idx, (train_idx, val_idx, test_idx) in enumerate(validator.split(x_all)):
    print(f'\nFold {fold_idx}: train={len(train_idx)} [{train_idx[0]}..{train_idx[-1]}], val={len(val_idx)} [{val_idx[0]}..{val_idx[-1]}], test={len(test_idx)} [{test_idx[0]}..{test_idx[-1]}]')
    print(f'  max val_idx={val_idx.max()}, x_all size={len(x_all)}')

    x_train_fold = x_all[train_idx]
    y_train_fold = y_all[train_idx]
    w_train_fold = w_all[train_idx] if w_all is not None else None
    x_val_fold = x_all[val_idx]
    y_val_fold = y_all[val_idx]

    print(f'  x_train_fold: {x_train_fold.shape}, x_val_fold: {x_val_fold.shape}')

    try:
        from src.training.modular_trainers import TransformerDirectionTrainer, TrainerConfig
        from dataclasses import replace
        import tempfile
        trainer_cfg = TrainerConfig(
            epochs=5, batch_size=64, learning_rate=0.0005, patience=3, min_epochs=2, verbose=0,
            transformer_d_model=32, transformer_num_heads=4, transformer_num_layers=2, transformer_dropout=0.2,
            use_focal_loss=True, use_ema=False, use_ewc=False, use_replay_buffer=False,
        )
        tmp_dir = tempfile.mkdtemp(prefix='candle_opt_test_')
        trainer_cfg = replace(trainer_cfg, checkpoint_dir=tmp_dir)
        trainer = TransformerDirectionTrainer(trainer_cfg)
        metrics = trainer.train(x_train_fold, y_train_fold, x_val_fold, y_val_fold,
                                feature_names=data.get('feature_names', []), w_train=w_train_fold)
        print(f'  Fold {fold_idx} SUCCESS: bal_acc={metrics.get("val_balanced_accuracy", "N/A")}')
    except Exception as exc:
        print(f'  Fold {fold_idx} FAILED: {type(exc).__name__}: {exc}')
        import traceback
        traceback.print_exc()
    break
