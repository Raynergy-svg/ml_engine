# ML Engine

**Multi-Model Ensemble FX Trading System**

A professional-grade machine learning system for Forex trading with a 4-model gated ensemble architecture, optimized for Apple Silicon (M1/M2/M3) and Intel Macs.

---

## ⚠️ Disclaimer

> **WARNING: This is experimental software. Not financial advice.**
>
> Trading Forex involves substantial risk of loss. Use only capital you can afford to lose. Always test on demo accounts first. The authors are not responsible for any financial outcomes.

---

## System Overview

| Component | Value |
|-----------|-------|
| **Timeframe** | H1 (Hourly) - default |
| **Primary Model** | Transformer (direction prediction) |
| **Gate Models** | XGBoost, RandomForest, Ridge |
| **Default Pairs** | EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, USD_CAD, NZD_USD |
| **Risk Per Trade** | 2-5% configurable |
| **SL/TP** | 15 pip SL / 30 pip TP (default) |
| **Account Integration** | OANDA v20 API (practice & live) |

---

## How It Works

### 4-Model Gated Ensemble

The system uses **four independent models** that must ALL pass before executing a trade:

```
                    ┌─────────────────────────────────┐
                    │   TRANSFORMER (Direction)       │
                    │   Predicts: Long/Short          │
                    │   Threshold: ≥60% probability   │
                    └─────────────┬───────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│    RIDGE      │       │   XGBOOST     │       │ RANDOMFOREST  │
│  (Confidence) │       │  (Momentum)   │       │    (Risk)     │
│  ADX-based    │       │  Percentile   │       │  ATR-based    │
│  Min: 50/100  │       │  Min: 0.20    │       │  Max: 2.5%    │
└───────────────┘       └───────────────┘       └───────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
                    ┌─────────────────────────────────┐
                    │     ALL GATES MUST PASS         │
                    │                                 │
                    │  ✅ Transformer prob ≥ 60%      │
                    │  ✅ Ridge confidence ≥ 50       │
                    │  ✅ XGBoost momentum ≥ 0.20     │
                    │  ✅ RF drawdown ≤ 2.5%          │
                    │                                 │
                    │        ↓ TRADE EXECUTED ↓       │
                    └─────────────────────────────────┘
```

### Gate Thresholds (Default)

| Gate | Model | Metric | Threshold | Purpose |
|------|-------|--------|-----------|---------|
| **Direction** | Transformer | Probability | ≥ 60% | Primary trend signal |
| **Confidence** | Ridge | ADX Score | ≥ 50/100 | Trend strength filter |
| **Momentum** | XGBoost | Percentile | ≥ 0.20 | Fresh/accelerating moves |
| **Risk** | RandomForest | ATR Drawdown | ≤ 2.5% | Max expected pullback |

### Additional Filters

- **Meta-Labeling**: ML model predicting if trade will succeed (min 55% confidence)
- **Market Intelligence**: News sentiment via FinBERT (blocks conflicting sentiment)
- **FX Guardrails**: Session windows, spread filters, daily limits

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Raynergy-svg/ml_engine.git
cd ml_engine

# Create conda environment
conda create -n tf-metal python=3.11
conda activate tf-metal

# Install TA-Lib (macOS)
brew install ta-lib

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure OANDA API

Create `.env` file:
```bash
OANDA_API_TOKEN=your_practice_api_token
OANDA_ACCOUNT_ID=your_account_id
```

### 3. Train a Model

```bash
# Train on EUR_USD H1 data (fetches 15k candles from OANDA)
./bin/Buddy train -i EUR_USD

# Train from local CSV
./bin/Buddy train -i EUR_USD --csv market_data/EUR_USD_H1.csv
```

### 4. Run Inference

```bash
# Predict on EUR_USD (dry run)
./bin/Buddy EUR_USD

# Execute trade if gates pass
./bin/Buddy EUR_USD -x

# Scan all pairs for opportunities
./bin/Buddy scan
```

---

## CLI Commands

### Buddy Script (`./bin/Buddy`)

| Command | Description | Example |
|---------|-------------|---------|
| `./bin/Buddy [PAIR]` | Predict direction (dry run) | `./bin/Buddy EUR_USD` |
| `./bin/Buddy [PAIR] -x` | Predict + execute trade | `./bin/Buddy USD_JPY -x` |
| `./bin/Buddy train -i PAIR` | Train model for pair | `./bin/Buddy train -i GBP_USD` |
| `./bin/Buddy scan` | Scan all pairs | `./bin/Buddy scan` |
| `./bin/Buddy status` | Show model status | `./bin/Buddy status` |
| `./bin/Buddy journal` | View trade journal | `./bin/Buddy journal` |

### Main.py Commands

```bash
# Training
python main.py train-buddy --instrument EUR_USD --candles 15000 --oanda-live

# Inference
python main.py buddy --instrument EUR_USD --execute

# Scan pairs
python main.py scan --pairs EUR_USD,GBP_USD,USD_JPY

# Retrain gate models only (XGBoost, RF, Ridge)
python main.py retrain-gates

# Model status
python main.py model-status
```

---

## Configuration

### Key Settings (`config/config_improved_H1.yaml`)

```yaml
# Timeframe
fx:
  granularity: H1              # Hourly candles

# Direction labeling (training)
direction_lookahead: 24        # 24 hours lookahead
direction_threshold: 0.003     # 0.3% min move for label

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

# Risk management
buddy:
  stop_loss_pips: 15.0         # 15 pip stop loss
  take_profit_pips: 30.0       # 30 pip take profit
  risk_per_trade_pct: 0.02     # 2% risk per trade

# Scanning (aggressive mode)
scan:
  risk_per_trade_pct: 0.05     # 5% risk for larger lots
  min_confidence: 0.53         # 53%+ confidence filter
  aggressive_mode: true
```

---

## Project Structure

```
ml_engine/
├── bin/
│   └── Buddy                    # Main CLI script
├── config/
│   ├── config_improved_H1.yaml  # H1 timeframe config (default)
│   └── config_m1_optimized.yaml # Apple Silicon optimized
├── src/
│   ├── core/
│   │   ├── modular_inference.py # Gated ensemble inference
│   │   └── modular_data_loaders.py
│   ├── models/
│   │   ├── tensorflow_models.py # Transformer, TCN, TFT
│   │   ├── tensorflow_engine.py # Training pipeline
│   │   └── ensemble_model.py    # Ensemble stacking
│   ├── training/
│   │   ├── modular_trainers.py  # Model trainers
│   │   ├── buddy_training_helpers.py
│   │   └── walkforward_validation.py
│   ├── risk/
│   │   ├── fx_guardrails.py     # Trading rules
│   │   ├── position_sizing.py   # Kelly-based sizing
│   │   └── triple_barrier.py    # Trade labeling
│   └── utils/
│       ├── oanda_practice.py    # OANDA API client
│       └── trade_journal.py     # Trade logging
├── main.py                      # CLI entry point
├── market_data/                 # Downloaded price data
└── trained_data/models/         # Trained model artifacts
```

---

## Performance Metrics

### Training Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `val_direction_accuracy` | Direction prediction accuracy | > 55% |
| `val_confidence_mae` | Confidence calibration error | < 0.15 |
| `val_combined` | Weighted direction + confidence | > 0.60 |

### Inference Metrics

| Metric | Description |
|--------|-------------|
| `tcn_probability` | Transformer direction confidence (0-1) |
| `ridge_confidence` | ADX-based trend strength (0-100) |
| `xgb_momentum` | Momentum percentile (0-1) |
| `rf_drawdown_pct` | Expected drawdown (%) |
| `meta_confidence` | Meta-labeler success probability |

### Example Output

```
╭──────────────────────────────────────────────────────────────────╮
│ EUR_USD Prediction                                               │
├──────────────────────────────────────────────────────────────────┤
│ Direction: LONG                                                  │
│ Transformer Prob: 67.3%                                          │
│ Ridge Confidence: 72/100                                         │
│ XGBoost Momentum: 0.45 (accelerating)                            │
│ RF Drawdown: 1.8%                                                │
│ Meta Confidence: 62%                                             │
├──────────────────────────────────────────────────────────────────┤
│ ✅ ALL GATES PASSED                                              │
│ Position Size: 0.5 lots                                          │
│ Stop Loss: 15 pips                                               │
│ Take Profit: 30 pips                                             │
╰──────────────────────────────────────────────────────────────────╯
```

---

## Platform Support

### Apple Silicon (M1/M2/M3)

Automatically optimized with TensorFlow Metal:

```bash
# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

| Setting | Optimized Value | Impact |
|---------|-----------------|--------|
| `model_type` | `transformer` | Best for H1 patterns |
| `batch_size` | `64-128` | Optimal GPU utilization |
| `recurrent_dropout` | `0.0` | **CRITICAL**: Non-zero causes massive slowdown |
| `mixed_precision` | `false` | Metal doesn't fully support FP16 |

### Intel Mac

Uses `ml_engine_py312` conda environment:

```bash
conda env create -f environment_intel_mac.yml
conda activate ml_engine_py312
```

---

## Model Files

After training, models are saved to `trained_data/models/`:

| File | Description |
|------|-------------|
| `transformer_direction.keras` | Transformer direction model |
| `transformer_direction.meta.pkl` | Scalers and metadata |
| `xgb_momentum.pkl` | XGBoost momentum gate |
| `ridge_confidence.pkl` | Ridge confidence gate |
| `rf_risk.pkl` | RandomForest risk gate |
| `modular_ensemble.meta.json` | Ensemble configuration |

Pair-specific models are stored in `trained_data/models/{PAIR}/`.

---

## Advanced Features

### Market Intelligence

```bash
# Enable news sentiment blocking
python main.py buddy --intelligent --instrument EUR_USD
```

- Fetches forex news via web scraping
- Analyzes sentiment with FinBERT
- Blocks trades conflicting with strong sentiment

### RL Position Sizer

```bash
# Train RL agent for position sizing
python main.py train-rl-sizer --timesteps 500000

# Use in inference
python main.py buddy --use-rl-sizer --instrument EUR_USD
```

### Walk-Forward Validation

```bash
# Time-series cross-validation
python -c "from src.training.walkforward_validation import run_walk_forward; run_walk_forward()"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `conda not found` | Install Miniforge: `brew install miniforge` |
| GPU not detected | Verify TF-Metal: `pip install tensorflow-metal` |
| OANDA errors | Check `.env` credentials and account status |
| Missing model | Train first: `./bin/Buddy train -i EUR_USD` |
| Slow training | Set `recurrent_dropout: 0.0` in config |

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md) | System architecture |
| [docs/CONFIDENCE_SYSTEM_DOCUMENTATION.md](docs/CONFIDENCE_SYSTEM_DOCUMENTATION.md) | Calibration details |
| [docs/FX_TIER1_GUARDRAILS_PLAN.md](docs/FX_TIER1_GUARDRAILS_PLAN.md) | Trading rules |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Development guide |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_buddy_intelligent_mode.py -v
```

---

## License

This project is for educational and research purposes. See LICENSE for details.

---

<p align="center">
  <strong>⚠️ Always test on demo accounts first. Trade responsibly.</strong>
</p>
