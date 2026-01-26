# ML Engine - Copilot Instructions

## Project Overview
An FX trading bot using multi-model ensemble ML (TCN/Transformer + XGBoost + RandomForest + Ridge) with OANDA API integration, optimized for Apple Silicon (M1/M2/M3 Metal).

## Architecture (Key Flows)

```
main.py (CLI) → buddy_training_helpers.py → tensorflow_engine.py → models_enhanced.py
                                          ↓
                              modular_inference.py (gated decisions)
                                          ↓
                              fx_guardrails.py → oanda_practice.py (execution)
```

### Critical Components
- **Ensemble Gate System**: All 4 gates must pass before trade execution (see [modular_inference.py](modular_inference.py#L1-L25))
- **Triple Barrier Labeling**: Professional trade outcome labels, not raw price direction ([triple_barrier.py](triple_barrier.py))
- **Walk-Forward Validation**: Time-series CV to prevent look-ahead bias ([walkforward_validation.py](walkforward_validation.py))

## CLI Commands

```bash
# Primary training (uses config_m1_optimized.yaml by default)
python main.py train-buddy --csv market_data/PAIR_TIMEFRAME.csv

# Quick inference
python main.py buddy --instrument USD_JPY --execute

# Retrain gate models (XGBoost, RF, Ridge) without retraining TCN
python main.py retrain-gates

# Train RL position sizer manually (if auto-training is disabled)
python main.py train-rl-sizer --timesteps 500000

# Run tests
pytest tests/
```

## M1 Metal Critical Settings
When modifying training code, preserve these optimizations in `BuddyTrainingOptions`:
- `model_type: "tcn"` - 2-3x faster than LSTM on Metal
- `batch_size: 128` - optimal for Metal GPU
- `mixed_precision: True` - 1.5-2x speedup
- `recurrent_dropout: 0.0` - **CRITICAL**: non-zero causes massive slowdown on Metal
- `steps_per_execution: 10` - reduces Python overhead
- `train_rl_sizer: True` - automatically trains RL position sizer after ensemble

## Code Patterns

### Feature Engineering
Always use **normalized features** from `modular_data_loaders.compute_normalized_features()` - models are instrument-agnostic (train on GBP_USD, works on EUR_USD).

### Model Loading
Models save as `.keras` with companion `.meta.json` containing:
- Scaler parameters (`feature_scaler`, `target_scaler`)
- Feature list (`feature_names`)
- Tier-2 calibration data

```python
# Correct pattern:
meta_path = Path("trained_data/models") / "buddy_tf.meta.json"
model_path = Path("trained_data/models") / "buddy_tf.keras"
```

### Custom Keras Layers
All custom layers in [tensorflow_models.py](tensorflow_models.py) use `@tf.keras.utils.register_keras_serializable()` for model serialization. Maintain this decorator on new layers.

### Inference Gate Checks
The ensemble uses a gated architecture ([modular_inference.py](modular_inference.py#L43-L70)):
```python
# All gates must pass:
# 1. TCN probability > 0.55
# 2. Ridge confidence > 45 (0-100 scale)
# 3. XGBoost momentum > 0.15 OR accelerating
# 4. RandomForest drawdown < 2.5%
```

## Testing Conventions
- Tests in `tests/` use pytest
- Some tests are ignored in CI (see [pytest.ini](pytest.ini))
- Integration tests require OANDA credentials in `.env`

## Configuration Files
- [config_m1_optimized.yaml](config_m1_optimized.yaml) - Production config for Apple Silicon
- `trained_data/models/` - Model artifacts (`.keras`, `.meta.json`, `.pkl`)
- `market_data/` - CSV price data

## Common Pitfalls
1. **NaN in features**: Always clean with `df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)`
2. **Sequence length**: Must match between training and inference (`seq_len` in config)
3. **Model promotion**: New models save as `buddy_tf_candidate.keras`; promote with `python main.py promote-model`
4. **Tier-2 calibration**: Stored in meta.json under `tier2.calibration`; recomputed on training

---

# IMPROVEMENT RECOMMENDATIONS

## 1. MODEL ARCHITECTURE UPGRADES

### 1.1 Add Mamba/State Space Models (SSM)
Current TCN has fixed receptive field. Mamba-style SSMs handle variable-length dependencies better for FX regime changes.
```python
# Add to tensorflow_models.py - S4/Mamba layer
pip install mamba-ssm  # or implement custom S4 layer
```

### 1.2 Implement Temporal Fusion Transformer (TFT) Properly
The existing TFT in [tensorflow_models.py](tensorflow_models.py) is incomplete. Add:
- **Static covariates**: instrument ID, session type
- **Known future inputs**: time-of-day, day-of-week embeddings
- **Interpretable attention**: visualize which lags matter

### 1.3 Replace Ridge with LightGBM
Ridge in [ensemble_model.py](ensemble_model.py#L192-L230) uses only last timestep - wasteful.
```python
# Better: LightGBM with lag features + rolling stats
import lightgbm as lgb
model = lgb.LGBMClassifier(boosting_type='dart', n_estimators=500)
```

### 1.4 Add Uncertainty Quantification
Current models output point estimates. Add:
```python
# Monte Carlo Dropout at inference
def predict_with_uncertainty(model, X, n_samples=30):
    preds = [model(X, training=True) for _ in range(n_samples)]
    return np.mean(preds, axis=0), np.std(preds, axis=0)
```

## 2. MULTI-PAIR TRAINING

### 2.1 Implement Cross-Pair Pre-training
Train a **foundation model** on ALL pairs, then fine-tune per-pair:
```python
# Step 1: Concatenate normalized features from EUR_USD, GBP_USD, USD_JPY, etc.
# Step 2: Pre-train TCN encoder with contrastive loss
# Step 3: Fine-tune classification heads per instrument
```

### 2.2 Add Pair Embeddings
Inject instrument identity into the model:
```python
# In TFTransformerPredictor
pair_embedding = Embedding(num_pairs, 16)(pair_id)
x = Concatenate()([sequence_features, pair_embedding])
```

### 2.3 Correlation-Aware Training
USD strength affects multiple pairs. Add cross-pair features:
```python
# DXY proxy = weighted avg of USD pairs
# Add correlation features between pairs
df['usd_strength'] = compute_dxy_proxy(df_eur_usd, df_gbp_usd, df_usd_jpy)
```

## 3. CHECKPOINTING & TRAINING PIPELINE

### 3.1 Implement Gradient Checkpointing
Reduce memory for longer sequences:
```python
# In tensorflow_models.py
tf.recompute_grad(attention_layer)  # Recompute instead of store
```

### 3.2 Add Model EMA (Exponential Moving Average)
Smoother inference, better generalization:
```python
# Add to training loop
ema = tf.train.ExponentialMovingAverage(decay=0.999)
ema.apply(model.trainable_variables)
# Use EMA weights for inference
```

### 3.3 Checkpoint Best Per-Metric
Currently saves only best `val_loss`. Save multiple checkpoints:
```python
# Save best direction_accuracy, best confidence_mae, best combined
ModelCheckpoint('best_direction.keras', monitor='val_direction_accuracy')
ModelCheckpoint('best_confidence.keras', monitor='val_confidence_mae')
```

### 3.4 Add Warm Restart Learning Rate
```yaml
# In config_m1_optimized.yaml
lr_schedule:
  type: cosine_warm_restart
  T_0: 10  # Initial period
  T_mult: 2  # Period multiplier
  eta_min: 1e-6
```

## 4. DATA PIPELINE OPTIMIZATION

### 4.1 Memory-Mapped Datasets
For large multi-pair training:
```python
# In tensorflow_data_pipeline.py
import numpy as np
X_mmap = np.memmap('features.dat', dtype='float32', mode='r', shape=(n, seq, feat))
dataset = tf.data.Dataset.from_generator(lambda: iter(X_mmap), output_signature=...)
```

### 4.2 Online Data Augmentation
Current augmentation is pre-computed. Move to training loop:
```python
# Add to tf.data pipeline
def augment(x, y):
    x = x + tf.random.normal(tf.shape(x), stddev=0.01)  # Noise
    if tf.random.uniform(()) > 0.5:
        x = x[::-1]  # Time reversal (for symmetric patterns)
    return x, y
dataset = dataset.map(augment)
```

### 4.3 Hard Example Mining
Focus training on difficult samples:
```python
# After each epoch, identify samples with highest loss
hard_indices = np.argsort(sample_losses)[-1000:]
# Oversample hard examples in next epoch
```

## 5. MATHEMATICAL IMPROVEMENTS

### 5.1 Replace Categorical Crossentropy with Focal Loss
Better for imbalanced direction labels:
```python
# In custom_losses.py
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow(1 - p_t, gamma) * ce
    return loss
```

### 5.2 Add Label Smoothing
Prevent overconfidence:
```python
# In training config
label_smoothing: 0.1  # Soft labels: 0.1/0.9 instead of 0/1
```

### 5.3 Implement Proper Kelly Criterion
Current position sizing is heuristic. Use:
```python
# True Kelly with edge and odds
kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
kelly_fraction = max(0, kelly_fraction * 0.25)  # Quarter Kelly for safety
```

### 5.4 Add Regime-Conditional Thresholds
Different market regimes need different gate thresholds:
```python
# In modular_inference.py
if regime == 'trend':
    min_tcn_probability = 0.52  # Lower threshold in trends
elif regime == 'chop':
    min_tcn_probability = 0.65  # Higher threshold in chop
```

## 6. DEPENDENCY UPGRADES

### 6.1 Upgrade to TensorFlow 2.16+ with Keras 3
```bash
pip install tensorflow>=2.16 keras>=3.0
# Benefits: Faster Metal backend, better mixed precision
```

### 6.2 Add Optuna for Hyperparameter Tuning
```python
# Add optuna_hpo.py
import optuna
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    # Train and return validation metric
```

### 6.3 Add PyArrow for Faster Data Loading
Already in requirements. Use for feature caching:
```python
import pyarrow.parquet as pq
df.to_parquet('features_cache.parquet', compression='zstd')
df = pq.read_table('features_cache.parquet').to_pandas()
```

### 6.4 Add ONNX Export for Production
```python
# For faster inference without TF overhead
pip install tf2onnx onnxruntime
python -m tf2onnx.convert --saved-model trained_data/models/buddy_tf.keras --output buddy.onnx
```

## 7. MEMORY & PERFORMANCE

### 7.1 Implement Sequence Bucketing
Avoid padding waste:
```python
# Group sequences by length, batch within buckets
dataset = dataset.bucket_by_sequence_length(
    element_length_func=lambda x, y: tf.shape(x)[0],
    bucket_boundaries=[30, 60, 90],
    bucket_batch_sizes=[256, 128, 64]
)
```

### 7.2 Use TensorFlow Data Service
For multi-GPU or distributed training:
```python
dispatcher = tf.data.experimental.service.DispatchServer()
dataset = dataset.apply(tf.data.experimental.service.distribute(...))
```

### 7.3 Profile and Optimize Critical Paths
```python
# Add to training
tf.profiler.experimental.start('logdir')
model.fit(...)
tf.profiler.experimental.stop()
# Analyze with: tensorboard --logdir logdir
```

## 8. RL POSITION SIZING (Now Integrated)

The [rl_position_sizing.py](rl_position_sizing.py) is now **automatically trained** after ensemble training completes.

### Automatic Training (Default)
RL training happens automatically after ensemble training when:
- `train_rl_sizer: True` in `BuddyTrainingOptions` (default)
- `training.auto_train_rl: true` in config YAML (default)

### Manual Training
```bash
# Train RL agent manually (if auto-training is disabled)
python main.py train-rl-sizer --timesteps 500000
```

### Disable Auto-Training
In `config_m1_optimized.yaml`:
```yaml
training:
  auto_train_rl: false  # Disable automatic RL training
```

Or via CLI:
```bash
python main.py train-buddy --skip-rl --csv market_data/USD_JPY_H1.csv
```

### Use in Inference
```bash
python main.py buddy --instrument USD_JPY --use-rl-sizer --execute
```

## QUICK WINS (Implement Today)

1. **Enable LightGBM** - `pip install lightgbm` and swap Ridge wrapper
2. **Add EMA** - 10 lines in training loop, better test performance
3. **Label smoothing** - 1 line config change, reduces overconfidence
4. **Focal loss** - Replace BCE, better imbalanced handling
5. **Warm restart LR** - Use `CosineDecayRestarts` scheduler
