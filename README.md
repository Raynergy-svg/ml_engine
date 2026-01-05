<p align="center">
  <h1 align="center">ML Engine</h1>
  <p align="center">
    <strong>Modular Ensemble FX Trading Bot</strong>
  </p>
  <p align="center">
    A professional-grade machine learning system for Forex trading with multi-model ensemble architecture, optimized for Apple Silicon.
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#documentation">Documentation</a>
</p>

---

## ⚠️ Disclaimer

> **WARNING: This is experimental software. Not financial advice.**
>
> Trading Forex involves substantial risk of loss. Use only capital you can afford to lose. Always test on demo accounts first. The authors are not responsible for any financial outcomes.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Ensemble** | TCN/Transformer + XGBoost + RandomForest + Ridge working together |
| **Market Regime Detection** | Automatic classification of trend, chop, and mean-revert conditions |
| **Confidence Calibration** | Platt/Isotonic scaling aligns model confidence with actual win rates |
| **Dynamic Risk Management** | Confidence-based SL/TP adjustments and position sizing |
| **Walk-Forward Validation** | Time-series cross-validation preventing look-ahead bias |
| **Triple Barrier Labeling** | Professional trade outcome classification |
| **M1/M2/M3 Metal Optimization** | Native TensorFlow Metal GPU acceleration |
| **OANDA Integration** | End-to-end trade execution via OANDA v20 API |
| **Real-time Dashboard** | Rich CLI visualization for monitoring |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main.py (CLI)                                  │
│                         Command Dispatcher & Orchestrator                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    Data Layer       │   │    Model Layer      │   │   Training Layer    │
├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│ • oanda_practice    │   │ • tensorflow_models │   │ • modular_trainers  │
│ • data_loader       │   │ • custom_losses     │   │ • tensorflow_engine │
│ • feature_engineering│  │ • ensemble_model    │   │ • walkforward_valid │
│ • modular_data_loaders│ │ • xgboost_model     │   │ • dynamic_thresholds│
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      ▼
                    ┌─────────────────────────────────┐
                    │        Inference & Risk         │
                    ├─────────────────────────────────┤
                    │ • modular_inference (gated)     │
                    │ • fx_guardrails (Tier-1 rules)  │
                    │ • risk_management (SL/TP)       │
                    │ • position_sizing (Kelly-based) │
                    │ • confidence_calibration        │
                    └─────────────────────────────────┘
```

### Ensemble Components

| Model | Task | Framework | Purpose |
|-------|------|-----------|---------|
| **TCN/Transformer** | Direction | TensorFlow | Primary trend prediction |
| **XGBoost** | Momentum | XGBoost | Strength and acceleration |
| **RandomForest** | Risk | scikit-learn | Drawdown probability |
| **Ridge** | Confidence | scikit-learn | Trade confidence scoring |
| **HistGradientBoosting** | Hybrid | scikit-learn | Voting ensemble member |

### Gated Decision Logic

Trades are only executed when **all gates pass**:
1. ✅ Ridge confidence > threshold (default: 75%)
2. ✅ XGBoost momentum is fresh or accelerating
3. ✅ RandomForest expected drawdown < risk tolerance
4. ✅ FX guardrails (session window, spread filter, daily limits)

---

## Installation

### Prerequisites

- **Python**: 3.9 or later
- **Conda**: Recommended (Miniforge for Apple Silicon)
- **TA-Lib**: Required for technical indicators

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Raynergy-svg/ml_engine.git
cd ml_engine

# Create and activate environment (Conda recommended for M1/M2/M3)
conda create -n tf-metal python=3.11
conda activate tf-metal

# Install TA-Lib (macOS)
brew install ta-lib

# Install dependencies
pip install -r requirements.txt
```

### Apple Silicon Optimization

For M1/M2/M3 Macs, TensorFlow Metal is automatically installed:

```bash
# Verify Metal GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Environment Variables

Create a `.env` file for OANDA credentials:

```bash
OANDA_API_TOKEN=your_api_token
OANDA_ACCOUNT_ID=your_account_id
```

---

## Usage

### Training

```bash
# Train with default settings (H1 timeframe)
python main.py buddy train

# Train with custom config
python main.py buddy train --config config_tuned.yaml

# Train for specific pair
python main.py buddy train --pair EUR_USD

# Train with OANDA live data fetch
python main.py buddy train --oanda-fetch
```

### Inference & Trading

```bash
# Scan for trading opportunities
python main.py buddy scan

# Check bot status
python main.py buddy status

# Run interactive mode
python main.py buddy --config config_tuned.yaml
```

### Walk-Forward Validation

```bash
# Run time-series cross-validation
python scripts/run_walkforward_ci.py --folds 5 --window expanding
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Configuration

### Key Configuration Files

| File | Purpose |
|------|---------|
| `config_tuned.yaml` | Production-ready settings |
| `config_m1_optimized.yaml` | Apple Silicon optimized |
| `config_improved_H1.yaml` | H1 timeframe specific |

### Example Configuration

```yaml
# Model Architecture
model:
  type: tcn                    # tcn | lstm | transformer | tft
  hidden_size: 32
  num_layers: 2
  dropout: 0.4

# Training (M1 Optimized)
training:
  batch_size: 128              # Optimal for Metal GPU
  epochs: 200
  learning_rate: 0.0005
  early_stopping_patience: 15
  mixed_precision: true        # FP16 acceleration

# Sequence Settings
data:
  sequence_length: 60
  target_shift: 12
  direction_lookahead: 12

# Risk Management
risk:
  max_daily_loss_pct: 2.0
  max_position_risk_pct: 1.0
  confidence_threshold: 0.75

# OANDA Integration
oanda:
  environment: practice        # practice | live
  default_instrument: EUR_USD
  granularity: H1
```

---

## Project Structure

```
ml_engine/
├── 📂 Core
│   ├── main.py                      # CLI entry point (~8800 lines)
│   ├── modular_trainers.py          # Specialized model trainers
│   ├── modular_inference.py         # Gated inference pipeline
│   └── modular_data_loaders.py      # Feature preparation
│
├── 📂 Models
│   ├── tensorflow_models.py         # Neural architectures (TCN, LSTM, TFT)
│   ├── tensorflow_engine.py         # Training pipeline
│   ├── custom_losses.py             # BinaryFocalLoss, MADL
│   ├── xgboost_model.py             # Gradient boosting wrapper
│   └── ensemble_model.py            # Ensemble stacking
│
├── 📂 Risk & Trading
│   ├── fx_guardrails.py             # Tier-1 trading rules
│   ├── risk_management.py           # Dynamic SL/TP
│   ├── position_sizing.py           # Kelly-based sizing
│   └── triple_barrier.py            # Trade labeling
│
├── 📂 Validation
│   ├── walkforward_validation.py    # Time-series CV
│   ├── confidence_calibration.py    # Probability calibration
│   └── dynamic_thresholds.py        # Adaptive thresholds
│
├── 📂 Features
│   ├── feature_engineering.py       # Technical indicators
│   ├── data_loader.py               # Data preprocessing
│   └── candle_smoothing.py          # Noise reduction
│
├── 📂 Integration
│   ├── oanda_practice.py            # OANDA v20 API client
│   ├── openai_integration.py        # LLM reasoning (optional)
│   └── tracing_setup.py             # Observability
│
├── 📂 Config
│   ├── config_tuned.yaml
│   ├── config_m1_optimized.yaml
│   └── requirements.txt
│
├── 📂 tests/                        # 18+ test modules
├── 📂 scripts/                      # Utility scripts
├── 📂 docs/                         # Additional documentation
└── 📂 trained_data/                 # Model artifacts
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) | Detailed system architecture |
| [CONFIDENCE_SYSTEM_DOCUMENTATION.md](CONFIDENCE_SYSTEM_DOCUMENTATION.md) | Confidence calibration system |
| [FX_TIER1_GUARDRAILS_PLAN.md](FX_TIER1_GUARDRAILS_PLAN.md) | Trading safety rules |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [docs/CACHE_SYSTEM.md](docs/CACHE_SYSTEM.md) | Caching implementation |

---

## Technical Stack

### Core Dependencies

```
tensorflow>=2.15.0          # Deep learning
tensorflow-metal>=1.0.0     # Apple Silicon GPU
xgboost                     # Gradient boosting
scikit-learn>=1.3.0         # Traditional ML
pandas>=2.0.0               # Data manipulation
numpy>=1.24.0               # Numerical computing
ta-lib>=0.4.26              # Technical analysis
rich>=13.4.0                # CLI visualization
wandb>=0.15.0               # Experiment tracking
```

### Model Architectures

- **TCN**: Temporal Convolutional Network (2-3x faster than LSTM on Metal)
- **Transformer**: Self-attention for sequence modeling
- **TFT**: Temporal Fusion Transformer for multi-horizon forecasting
- **LSTM**: Legacy support with attention mechanism

---

## Performance Optimizations

### Apple Silicon (M1/M2/M3)

| Setting | Optimized Value | Impact |
|---------|-----------------|--------|
| `model_type` | `tcn` | 2-3x faster than LSTM |
| `batch_size` | `128` | Better GPU utilization |
| `mixed_precision` | `true` | 1.5-2x speedup |
| `steps_per_execution` | `10` | Reduced Python overhead |

### Training Tips

1. **Use TCN** for fastest training on Apple Silicon
2. **Enable mixed precision** for memory efficiency
3. **Batch size 128** is optimal for Metal GPU
4. **Early stopping** with patience 15-20 prevents overfitting

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests before submitting PR
pytest tests/ -v

# Check code style
ruff check .
```

---

## License

This project is for educational and research purposes. See LICENSE for details.

---

<p align="center">
  <strong>⚠️ Always test on demo accounts first. Trade responsibly.</strong>
</p>