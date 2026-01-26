# ML Engine - Copilot Instructions

## Project Overview
An FX trading bot using a **4-model gated ensemble** (Transformer + XGBoost + RandomForest + Ridge) with OANDA API integration, optimized for Apple Silicon (M1/M2/M3 Metal) and Intel Macs. Default timeframe is **H1 (Hourly)**.

## Architecture (Key Flows)

```
main.py (CLI)
     │
     ├── ./bin/Buddy (shell wrapper)
     │
     └── src/training/buddy_training_helpers.py
              │
              ├── src/models/tensorflow_engine.py
              │        └── src/models/tensorflow_models.py (Transformer, TCN, TFT)
              │
              └── src/training/modular_trainers.py
                       │
                       └── src/core/modular_inference.py (gated decisions)
                                │
                                ├── src/risk/fx_guardrails.py
                                └── src/utils/oanda_practice.py (execution)
```

### Critical Components
- **Ensemble Gate System**: All 4 gates must pass before trade execution
- **Triple Barrier Labeling**: Professional trade outcome labels (src/risk/triple_barrier.py)
- **Walk-Forward Validation**: Time-series CV to prevent look-ahead bias (src/training/walkforward_validation.py)
- **Market Intelligence**: News sentiment via FinBERT (market_intelligence.py)

## Project Structure

```
ml_engine/
├── bin/
│   └── Buddy                          # Main CLI script (shell wrapper)
├── config/
│   ├── config_improved_H1.yaml        # H1 timeframe config (DEFAULT)
│   └── config_m1_optimized.yaml       # Apple Silicon optimized
├── src/
│   ├── core/
│   │   ├── modular_inference.py       # Gated ensemble inference
│   │   └── modular_data_loaders.py    # Feature preparation
│   ├── models/
│   │   ├── tensorflow_models.py       # Transformer, TCN, TFT architectures
│   │   ├── tensorflow_engine.py       # Training pipeline
│   │   └── ensemble_model.py          # Ensemble stacking
│   ├── training/
│   │   ├── modular_trainers.py        # Model trainers
│   │   ├── buddy_training_helpers.py  # Training orchestration
│   │   └── walkforward_validation.py  # Time-series CV
│   ├── risk/
│   │   ├── fx_guardrails.py           # Trading rules
│   │   ├── position_sizing.py         # Kelly-based sizing
│   │   └── triple_barrier.py          # Trade labeling
│   └── utils/
│       ├── oanda_practice.py          # OANDA API client
│       └── trade_journal.py           # Trade logging
├── main.py                            # CLI entry point
├── market_data/                       # Downloaded price data (gitignored)
└── trained_data/models/               # Model artifacts (gitignored)
```

## CLI Commands

```bash
# === BUDDY SCRIPT (Recommended) ===
./bin/Buddy EUR_USD              # Predict (dry run)
./bin/Buddy EUR_USD -x           # Predict + execute trade
./bin/Buddy train -i EUR_USD     # Train model for pair
./bin/Buddy scan                 # Scan all pairs for opportunities
./bin/Buddy status               # Show model status
./bin/Buddy journal              # View trade journal

# === MAIN.PY COMMANDS ===
# Training (fetches 15k H1 candles from OANDA)
python main.py train-buddy --instrument EUR_USD --oanda-live

# Training from local CSV
python main.py train-buddy --instrument EUR_USD --csv market_data/EUR_USD_H1.csv

# Inference
python main.py buddy --instrument EUR_USD --execute

# Scan pairs
python main.py scan --pairs EUR_USD,GBP_USD,USD_JPY

# Retrain gate models only (XGBoost, RF, Ridge)
python main.py retrain-gates

# Train RL position sizer manually
python main.py train-rl-sizer --timesteps 500000

# Run tests
pytest tests/ -v
```

## Gate Thresholds (H1 Config)

The ensemble uses a gated architecture. **ALL gates must pass** for trade execution:

```python
# From src/core/modular_inference.py InferenceConfig:
min_tcn_probability: float = 0.60    # Transformer direction >= 60%
min_confidence: float = 50.0         # Ridge ADX score >= 50/100
min_momentum: float = 0.20           # XGBoost percentile >= 0.20
max_drawdown_pct: float = 0.025      # RF drawdown <= 2.5%

# Additional gates:
min_meta_confidence: float = 0.55    # Meta-labeler success probability
sentiment_block_threshold: float = 0.60  # Block on strong contrary sentiment
```

## H1 Timeframe Settings

Default configuration uses **H1 (Hourly)** candles. Key settings in `config/config_improved_H1.yaml`:

```yaml
# Timeframe
fx:
  granularity: H1                    # Hourly candles

# Direction labeling
direction_lookahead: 24              # 24 hours (24 bars) lookahead
direction_threshold: 0.003           # 0.3% min move for clear label

# Model architecture
transformer:
  d_model: 32
  num_heads: 4
  num_layers: 2
  dropout: 0.4

# Training
training:
  epochs: 200
  early_stopping_patience: 40
  batch_size: 64
  learning_rate: 0.0003

# Risk
buddy:
  stop_loss_pips: 15.0               # 15 pip SL
  take_profit_pips: 30.0             # 30 pip TP
  risk_per_trade_pct: 0.02           # 2% risk per trade
```

## Apple Silicon (M1/M2/M3) Critical Settings

When modifying training code, preserve these optimizations:

```yaml
# CRITICAL for Metal GPU performance:
model:
  recurrent_dropout: 0.0             # NON-ZERO CAUSES MASSIVE SLOWDOWN
  
training:
  batch_size: 64                     # Optimal for Metal GPU
  mixed_precision: false             # Metal doesn't fully support FP16
  steps_per_execution: 10            # Reduces Python overhead
  jit_compile: false                 # Avoid Metal issues
```

## Intel Mac Settings

Uses `ml_engine_py312` conda environment. Auto-detected in `bin/Buddy`:

```bash
# Auto-detection in bin/Buddy:
if [[ "$(uname -m)" == "x86_64" ]]; then
    ENV_NAME="${BUDDY_CONDA_ENV:-ml_engine_py312}"
else
    ENV_NAME="${BUDDY_CONDA_ENV:-tf-metal}"
fi
```

## Code Patterns

### Feature Engineering
Always use **normalized features** from `src/core/modular_data_loaders.compute_normalized_features()` - models are instrument-agnostic.

### Model Loading
Models save as `.keras` with companion `.meta.pkl` containing:
- Scaler parameters
- Feature list (`feature_names`)
- Tier-2 calibration data

```python
# Correct pattern:
from pathlib import Path
model_path = Path("trained_data/models") / "transformer_direction.keras"
meta_path = Path("trained_data/models") / "transformer_direction.meta.pkl"

# Pair-specific models:
pair_model_path = Path("trained_data/models/EUR_USD") / "transformer_direction.keras"
```

### Custom Keras Layers
All custom layers in `src/models/tensorflow_models.py` use `@tf.keras.utils.register_keras_serializable()` for model serialization. Maintain this decorator on new layers.

## Model Files

After training, models are saved to `trained_data/models/`:

| File | Description |
|------|-------------|
| `transformer_direction.keras` | Transformer direction model |
| `transformer_direction.meta.pkl` | Scalers and metadata |
| `transformer_direction.ema.pkl` | EMA weights |
| `xgb_momentum.pkl` | XGBoost momentum gate |
| `ridge_confidence.pkl` | Ridge confidence gate |
| `rf_risk.pkl` | RandomForest risk gate |
| `modular_ensemble.meta.json` | Ensemble configuration |
| `rl_position_sizer.zip` | RL position sizing agent |

Pair-specific models stored in `trained_data/models/{PAIR}/`.

## Testing Conventions
- Tests in `tests/` use pytest
- Some tests are ignored in CI (see pytest.ini)
- Integration tests require OANDA credentials in `.env`

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_buddy_intelligent_mode.py -v
```

## Common Pitfalls

1. **NaN in features**: Always clean with `df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)`
2. **Sequence length**: Must match between training and inference (`seq_len: 60` in H1 config)
3. **Path changes**: Source files now in `src/` subfolders, not root
4. **Config path**: Default config is `config/config_improved_H1.yaml`, not root
5. **recurrent_dropout**: Keep at `0.0` on Metal - non-zero causes 10x slowdown

## Environment Variables

Create `.env` file in project root:

```bash
OANDA_API_TOKEN=your_practice_api_token
OANDA_ACCOUNT_ID=your_account_id
```

## Gitignored Files

These are machine-specific and NOT tracked:

```
trained_data/          # All model artifacts
market_data/*.csv      # Downloaded price data
*.pkl, *.keras, *.h5   # Model files
*.log                  # Log files
```

---

# IMPROVEMENT RECOMMENDATIONS

## Quick Wins

1. **Enable LightGBM** - `pip install lightgbm` and swap Ridge wrapper
2. **Label smoothing** - 1 line config change, reduces overconfidence
3. **Focal loss** - Already implemented, use `direction_loss: focal` in config
4. **Warm restart LR** - Use `CosineDecayRestarts` scheduler

## Model Architecture Upgrades

### Add Mamba/State Space Models
```python
# Mamba-style SSMs handle variable-length dependencies better
pip install mamba-ssm  # or implement custom S4 layer
```

### Uncertainty Quantification
```python
# Monte Carlo Dropout at inference
def predict_with_uncertainty(model, X, n_samples=30):
    preds = [model(X, training=True) for _ in range(n_samples)]
    return np.mean(preds, axis=0), np.std(preds, axis=0)
```

## Multi-Pair Training

Currently models are trained per-pair. Future improvement:

```python
# Foundation model approach:
# 1. Pre-train on ALL pairs with contrastive loss
# 2. Fine-tune classification heads per instrument
# 3. Add pair embeddings for instrument identity
```

## RL Position Sizing

The `rl_position_sizing.py` is integrated and can be trained:

```bash
# Train RL agent
python main.py train-rl-sizer --timesteps 500000

# Use in inference
python main.py buddy --instrument EUR_USD --use-rl-sizer --execute
```

## Dependency Upgrades

```bash
# Upgrade TensorFlow for better Metal support
pip install tensorflow>=2.16 keras>=3.0

# Add Optuna for hyperparameter tuning
pip install optuna

# ONNX export for faster inference
pip install tf2onnx onnxruntime
```
