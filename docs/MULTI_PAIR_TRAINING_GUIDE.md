# Multi-Pair Training Guide

## Overview

The `JointMultiPairTrainer` enables training models across multiple currency pairs simultaneously using transfer learning. This is especially useful for Phase 4's smaller per-pair datasets with stricter direction thresholds.

## Benefits

1. **Transfer Learning**: Models learn common patterns across pairs
2. **Better Generalization**: Reduces overfitting with limited data per pair
3. **Shared Knowledge**: Cross-pair patterns improve individual pair performance
4. **Foundation Model**: Pre-train on all pairs, then fine-tune per pair

## Usage

### Basic Multi-Pair Training

```python
from src.training.trainers.joint_trainer import JointMultiPairTrainer
from src.training.trainers.config import TrainerConfig
import pandas as pd

# Initialize trainer with config
config = TrainerConfig()
trainer = JointMultiPairTrainer(config)

# Prepare data for multiple pairs
dfs = {
    "EUR_USD": eur_usd_df,  # DataFrames with OHLCV + features
    "GBP_USD": gbp_usd_df,
    "USD_JPY": usd_jpy_df,
}
instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]

# Train joint models (saves to trained_data/models/joint/)
results = trainer.train(dfs, instruments)

print(f"Joint training results: {results}")
```

### CLI Usage

```bash
# Train joint models for major pairs
python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY

# Or use Buddy wrapper
./bin/Buddy train-joint --pairs EUR_USD,GBP_USD,USD_JPY
```

### Fine-Tuning for Specific Pairs

After joint training, fine-tune for specific pairs:

```python
# Fine-tune for EUR_USD
fine_tuned_results = trainer.fine_tune_for_pair(
    df=eur_usd_df,
    instrument="EUR_USD",
    save_dir="trained_data/models",
)

print(f"Fine-tuned results: {fine_tuned_results}")
```

## Architecture

### Models Trained

The joint trainer trains 4 models:

1. **TransformerDirectionTrainer**: Direction prediction (Keras)
2. **LightGBMMomentumTrainer**: Momentum analysis (Gate 3)
3. **LightGBMRiskTrainer**: Risk assessment (Gate 4)
4. **RidgeTrainer**: Confidence scoring (Gate 2)

### Instrument Encoding

Each pair gets one-hot encoded features:

```python
# Example for 3 pairs:
EUR_USD: [1, 0, 0]
GBP_USD: [0, 1, 0]
USD_JPY: [0, 0, 1]
```

This allows models to learn:
- Common patterns across all pairs
- Pair-specific behaviors via instrument embeddings

## Data Preparation

### Input Format

Each DataFrame must have:
- OHLCV columns: `open`, `high`, `low`, `close`, `volume`
- Feature engineering applied (RSI, MACD, ATR, etc.)
- Direction labels (if training direction model)
- Momentum/risk labels (if training gate models)

### Example

```python
import pandas as pd
from src.data.feature_engineering import FeatureEngineering
from src.utils.oanda_practice import OandaPracticeClient

# Fetch data
client = OandaPracticeClient.from_env()
resp = client.get_candles("EUR_USD", granularity="H1", count=25000)

# Convert to DataFrame
df = candles_to_ohlcv_df(resp)

# Apply feature engineering
fe = FeatureEngineering()
df_features = fe.create_features(df, include_all=True)
```

## Model Saving

### Joint Models

Saved to `trained_data/models/joint/`:
```
trained_data/models/joint/
├── transformer_direction.keras
├── transformer_direction.meta.pkl
├── lgbm_momentum.pkl
├── lgbm_risk.pkl
└── ridge_confidence.pkl
```

### Fine-Tuned Models

Saved to `trained_data/models/{PAIR}/`:
```
trained_data/models/EUR_USD/
├── transformer_direction.keras
├── transformer_direction.meta.pkl
├── lgbm_momentum.pkl
├── lgbm_risk.pkl
└── ridge_confidence.pkl
```

## Phase 4 Strategy

With stricter direction thresholds (75 bps vs 30 bps), training data per pair is reduced. Multi-pair training compensates by:

1. **Pre-training**: Train on all 7 major pairs (EUR_USD, GBP_USD, USD_JPY, AUD_USD, NZD_USD, USD_CAD, USD_CHF)
2. **Transfer Learning**: Shared encoder learns common FX patterns
3. **Fine-tuning**: Adapt to each pair's specific characteristics
4. **Data Augmentation**: Combined with time-series augmentation for better generalization

## Configuration

Add to `config/config_improved_H1.yaml`:

```yaml
joint_training:
  enabled: true
  pairs:
    - EUR_USD
    - GBP_USD
    - USD_JPY
    - AUD_USD
    - NZD_USD
    - USD_CAD
    - USD_CHF
  fine_tune_threshold: 0.05  # Fine-tune if joint performance < 5% below pair-specific
```

## Best Practices

### 1. Data Quality

- Ensure all pairs have similar time ranges
- Clean data (handle NaN, infinity values)
- Apply consistent feature engineering

### 2. Training Order

```bash
# Step 1: Train joint models
python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY

# Step 2: Fine-tune for each pair
python main.py train-buddy --instrument EUR_USD --warm-start trained_data/models/joint/transformer_direction.keras
python main.py train-buddy --instrument GBP_USD --warm-start trained_data/models/joint/transformer_direction.keras
```

### 3. Monitoring

Check per-instrument metrics:

```python
# Evaluate joint model performance per pair
metrics = trainer.evaluate_per_instrument(dfs, instruments)

for pair, pair_metrics in metrics.items():
    print(f"{pair}: {pair_metrics}")
```

### 4. Fine-Tuning Decision

The trainer automatically decides when to fine-tune:

```python
# Check if fine-tuning is recommended
decisions = trainer.should_fine_tune(performance_threshold=0.05)

for pair, should_ft in decisions.items():
    if should_ft:
        print(f"Fine-tuning recommended for {pair}")
```

## Advanced Usage

### Custom Config

```python
from src.training.trainers.config import TrainerConfig

config = TrainerConfig(
    epochs=200,
    batch_size=64,
    learning_rate=0.0003,
    use_augmentation=True,  # Enable data augmentation
    augmentation_noise_std=0.01,
    augmentation_scale_range=(0.98, 1.02),
)

trainer = JointMultiPairTrainer(config)
```

### Loading Saved Models

```python
# Load joint models
success = trainer.load(load_dir="trained_data/models/joint")

if success:
    # Use for inference or fine-tuning
    results = trainer.fine_tune_for_pair(df, "EUR_USD")
```

## Troubleshooting

### Issue: Feature Dimension Mismatch

**Error**: `Feature dimension mismatch: saved model has X features, but only Y available`

**Solution**: Ensure all pairs use the same feature engineering pipeline:

```python
# Use consistent feature names across all pairs
feature_engineering = FeatureEngineering()
for pair, df in dfs.items():
    dfs[pair] = feature_engineering.create_features(df, include_all=True)
```

### Issue: Memory Errors

**Error**: `OOM when allocating tensor`

**Solution**: Reduce batch size or train pairs sequentially:

```python
config = TrainerConfig(batch_size=32)  # Reduce from 64
trainer = JointMultiPairTrainer(config)
```

### Issue: Poor Joint Performance

**Symptom**: Joint model underperforms single-pair models

**Solution**: Fine-tune for each pair:

```python
for pair in instruments:
    trainer.fine_tune_for_pair(dfs[pair], pair)
```

## References

- Implementation: `src/training/trainers/joint_trainer.py`
- Base Config: `src/training/trainers/config.py`
- CLI Integration: `main.py` (train-joint command)
- Tests: `tests/test_joint_trainer.py`

## See Also

- [Walk-Forward Validation Guide](WALKFORWARD_VALIDATION_GUIDE.md)
- [Training Troubleshooting](TRAINING_TROUBLESHOOTING.md)
- [Copilot Instructions](.github/copilot-instructions.md)
