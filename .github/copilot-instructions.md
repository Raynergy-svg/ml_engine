# ML Engine - Copilot Instructions

> **Version**: 2.1.0 | **Last Updated**: 2025-02-07 | **Status**: Active

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [CLI Commands](#cli-commands)
- [Gate Thresholds](#gate-thresholds)
- [Configuration](#configuration)
- [Platform-Specific Settings](#platform-specific-settings)
- [Code Patterns](#code-patterns)
- [Model Files](#model-files)
- [Testing](#testing)
- [Best Practices](#best-practices)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Improvement Roadmap](#improvement-roadmap)

---

## Quick Start

```bash
# Install dependencies (Apple Silicon)
conda env create -f environment_tf_metal.yml

# Train a model for EUR/USD
./bin/Buddy train -i EUR_USD

# Run inference (dry run)
./bin/Buddy EUR_USD

# Execute a trade
./bin/Buddy EUR_USD -x
```

---

## Project Overview

An FX trading bot using a **4-model gated ensemble** (Transformer + XGBoost + RandomForest + Ridge) with OANDA API integration, optimized for Apple Silicon (M1/M2/M3 Metal) and Intel Macs.

### Key Specifications

| Property | Value |
|----------|-------|
| **Default Timeframe** | H1 (Hourly) |
| **Ensemble Size** | 4 models (Transformer, XGBoost, RF, Ridge) |
| **Gate Count** | 8+ gates (all must pass) |
| **Meta-Labeling** | XGBoost trade success predictor |
| **Labeling Method** | Triple Barrier (professional trade outcomes) |
| **Validation** | Walk-Forward (prevents look-ahead bias) |
| **Supported Platforms** | Apple Silicon (M1/M2/M3), Intel Mac |

### Critical Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Ensemble Gate System** | 8+ gates must pass before trade execution | [`src/core/modular_inference.py`](src/core/modular_inference.py) |
| **Meta-Labeling** | 5th gate predicts trade success probability | [`src/training/meta_labeling.py`](src/training/meta_labeling.py) |
| **Triple Barrier Labeling** | Professional trade outcome labels | [`src/risk/triple_barrier.py`](src/risk/triple_barrier.py) |
| **Walk-Forward Validation** | Time-series CV to prevent look-ahead bias | [`src/training/walkforward_validation.py`](src/training/walkforward_validation.py) |
| **Market Intelligence** | News sentiment via FinBERT, drift detection, auto-retrain | [`market_intelligence.py`](market_intelligence.py) |
| **Drift Detection** | Auto-triggers model retraining when performance degrades | [`market_intelligence.py`](market_intelligence.py) |
| **LLM Integration** | Optional reasoning layer for dynamic threshold adjustment | [`buddy_intelligent_mode.py`](buddy_intelligent_mode.py) |
| **Permissive Mode** | Graceful degradation when sklearn models have version mismatches | [`src/core/modular_inference.py`](src/core/modular_inference.py) |

---

## Architecture

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              MAIN ENTRY POINT                            │
│                          main.py (CLI) / ./bin/Buddy                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        ┌───────────▼───────────┐       ┌───────────▼───────────┐
        │   Training Pipeline   │       │   Inference Pipeline   │
        └───────────┬───────────┘       └───────────┬───────────┘
                    │                               │
    ┌───────────────▼───────────────┐   ┌───────────▼───────────┐
    │ buddy_training_helpers.py    │   │ modular_inference.py   │
    └───────────────┬───────────────┘   └───────────┬───────────┘
                    │                               │
    ┌───────────────▼───────────────┐   ┌───────────▼───────────┐
    │  Model Training Components   │   │  Gated Decision Logic  │
    ├───────────────────────────────┤   ├────────────────────────┤
    │ • tensorflow_engine.py       │   │ • Gate 1: TCN Prob      │
    │   └─ tensorflow_models.py    │   │ • Gate 2: Ridge Conf   │
    │     (Transformer, TCN, TFT)  │   │ • Gate 3: XGBoost Mom  │
    │ • modular_trainers.py        │   │ • Gate 4: RF Risk      │
    │ • walkforward_validation.py  │   │ • Gate 5: Meta-Label   │
    │ • meta_labeling.py           │   │ • Gate 6: Sentiment    │
    │ • triple_barrier.py          │   │ • Gate 7: RSI Extreme   │
    │ • position_sizing.py         │   │ • Gate 8: Trend Check  │
    └───────────────────────────────┘   └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │   Risk & Execution    │
                                    ├────────────────────────┤
                                    │ • fx_guardrails.py    │
                                    │ • oanda_practice.py   │
                                    │ • trade_journal.py    │
                                    └────────────────────────┘
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OANDA API     │───▶│  Raw Price Data │───▶│  Feature Eng.   │
│  (Live/CSV)     │    │  (H1 candles)   │    │  (Normalized)   │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                    ┌──────────────────▼──────────────────┐
                                    │      Model Predictions               │
                                    ├──────────────────────────────────────┤
                                    │ • Transformer (Direction)            │
                                    │ • XGBoost (Momentum)                │
                                    │ • RandomForest (Risk)                │
                                    │ • Ridge (Confidence)                │
                                    │ • Meta-Labeler (Success Prob)       │
                                    └──────────────────┬───────────────────┘
                                                       │
                                    ┌──────────────────▼──────────────────┐
                                    │      Gate Evaluation                │
                                    │   (All 8 gates must pass)           │
                                    └──────────────────┬───────────────────┘
                                                       │
                                    ┌──────────────────▼──────────────────┐
                                    │      Risk Guardrails                │
                                    │   • Position Sizing                 │
                                    │   • Stop Loss / Take Profit         │
                                    │   • Market Regime Check             │
                                    └──────────────────┬───────────────────┘
                                                       │
                                    ┌──────────────────▼──────────────────┐
                                    │      Trade Execution                │
                                    │   (OANDA API)                       │
                                    └──────────────────┬───────────────────┘
                                                       │
                                    ┌──────────────────▼──────────────────┐
                                    │      Trade Journaling               │
                                    │   • Performance Tracking            │
                                    │   • Model Monitoring                │
                                    └──────────────────────────────────────┘
```

---

## Project Structure

```
ml_engine/
├── .github/
│   └── copilot-instructions.md     # This file
│
├── bin/
│   └── Buddy                       # Main CLI script (shell wrapper)
│
├── config/
│   ├── config_improved_H1.yaml     # H1 timeframe config (DEFAULT)
│   ├── config_m1_optimized.yaml    # Apple Silicon optimized
│   └── config_intel_optimized.yaml # Intel Mac optimized
│
├── src/
│   ├── core/
│   │   ├── modular_inference.py    # Gated ensemble inference
│   │   └── modular_data_loaders.py # Feature preparation
│   │
│   ├── models/
│   │   ├── tensorflow_models.py    # Transformer, TCN, TFT architectures
│   │   ├── tensorflow_engine.py    # Training pipeline
│   │   └── ensemble_model.py       # Ensemble stacking
│   │
│   ├── training/
│   │   ├── modular_trainers.py     # Model trainers
│   │   ├── buddy_training_helpers.py # Training orchestration
│   │   ├── walkforward_validation.py # Time-series CV
│   │   └── meta_labeling.py       # Meta-labeling implementation
│   │
│   ├── risk/
│   │   ├── fx_guardrails.py        # Trading rules
│   │   ├── position_sizing.py      # Kelly-based sizing
│   │   └── triple_barrier.py       # Trade labeling
│   │
│   └── utils/
│       ├── oanda_practice.py       # OANDA API client
│       └── trade_journal.py        # Trade logging
│
├── cli/
│   ├── commands.py                 # CLI command definitions
│   ├── config.py                   # Config management
│   ├── models.py                   # CLI model utilities
│   └── training.py                 # CLI training commands
│
├── main.py                         # CLI entry point
├── market_intelligence.py          # News sentiment & drift detection
├── buddy_intelligent_mode.py       # LLM integration
├── rl_position_sizing.py           # RL position sizer
│
├── tests/                          # Test suite
│   ├── test_prediction_collapse.py
│   ├── test_buddy_intelligent_mode.py
│   └── ...
│
├── docs/                           # Documentation
│   ├── PREDICTION_COLLAPSE_SYSTEM.md
│   ├── TRAINING_TROUBLESHOOTING.md
│   └── ...
│
├── market_data/                    # Downloaded price data (gitignored)
└── trained_data/models/            # Model artifacts (gitignored)
```

---

## CLI Commands

### Buddy Script (Recommended)

The `./bin/Buddy` script is the preferred interface with auto-detection for Apple Silicon vs Intel Macs.

```bash
# === TRAINING ===
./bin/Buddy train -i EUR_USD              # Train model for pair (fetches from OANDA)
./bin/Buddy train -i EUR_USD --csv path/to/data.csv  # Train from local CSV

# === INFERENCE ===
./bin/Buddy EUR_USD                       # Predict (dry run)
./bin/Buddy EUR_USD -x                    # Predict + execute trade
./bin/Buddy EUR_USD -x --dry-run          # Simulate execution without trading

# === SCANNING ===
./bin/Buddy scan                           # Scan all pairs for opportunities
./bin/Buddy scan --pairs EUR_USD,GBP_USD  # Scan specific pairs

# === MONITORING ===
./bin/Buddy status                        # Show model status
./bin/Buddy journal                       # View trade journal
./bin/Buddy logs                          # View recent logs

# === MAINTENANCE ===
./bin/Buddy retrain-gates                 # Retrain gate models only (XGBoost, RF, Ridge)
./bin/Buddy train-rl-sizer --timesteps 500000  # Train RL position sizer
```

### main.py Commands

Direct Python access for advanced usage.

```bash
# === TRAINING ===
# Training (fetches 15k H1 candles from OANDA)
python main.py train-buddy --instrument EUR_USD --oanda-live

# Training from local CSV
python main.py train-buddy --instrument EUR_USD --csv market_data/EUR_USD_H1.csv

# === INFERENCE ===
# Inference
python main.py buddy --instrument EUR_USD

# Execute trade
python main.py buddy --instrument EUR_USD --execute

# === SCANNING ===
# Scan pairs
python main.py scan --pairs EUR_USD,GBP_USD,USD_JPY

# === MAINTENANCE ===
# Retrain gate models only (XGBoost, RF, Ridge)
python main.py retrain-gates

# Train RL position sizer manually
python main.py train-rl-sizer --timesteps 500000

# === TESTING ===
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_buddy_intelligent_mode.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Command Reference

| Command | Purpose | Required Args | Optional Args |
|---------|---------|---------------|---------------|
| `train -i PAIR` | Train model for pair | `PAIR` (e.g., EUR_USD) | `--csv PATH` |
| `PAIR` | Predict (dry run) | `PAIR` | None |
| `PAIR -x` | Execute trade | `PAIR` | `--dry-run` |
| `scan` | Scan all pairs | None | `--pairs LIST` |
| `status` | Show model status | None | None |
| `journal` | View trade journal | None | `--limit N` |
| `retrain-gates` | Retrain gate models | None | None |

---

## Gate Thresholds

The ensemble uses a gated architecture with **8+ gates**. **ALL gates must pass** for trade execution.

### Gate Configuration

```python
# From src/core/modular_inference.py InferenceConfig:

# === CORE 4-MODEL GATES ===
min_tcn_probability: float = 0.60        # Gate 1: Transformer direction >= 60% confidence
min_confidence: float = 50.0             # Gate 2: Ridge ADX score >= 50/100
min_momentum: float = 0.20               # Gate 3: XGBoost percentile >= 0.20
max_drawdown_pct: float = 0.025          # Gate 4: RF drawdown <= 2.5%

# === ADDITIONAL GATES ===
min_meta_confidence: float = 0.55        # Gate 5: Meta-labeler trade success >= 55%
sentiment_block_threshold: float = 0.60  # Gate 6: Block on strong contrary sentiment
# Gate 7: RSI extremes (RSI < 10 for LONG, RSI > 90 for SHORT)
# Gate 8: Trend contradiction (ADX > 35, block counter-trend trades)
```

### Gate System Details

| Gate # | Name | Source Model | Purpose | Default Threshold | Failure Action |
|--------|------|--------------|---------|-------------------|----------------|
| 1 | **TCN Probability** | Transformer/TCN | Direction confidence | ≥60% (not 50/50) | Block trade |
| 2 | **Confidence** | Ridge | Trend strength (ADX-based) | ≥50/100 | Block trade |
| 3 | **Momentum** | XGBoost | Momentum percentile or acceleration | ≥0.20 OR accelerating | Block trade |
| 4 | **Risk/Drawdown** | RandomForest | Expected drawdown | ≤2.5% of equity | Block trade |
| 5 | **Meta-Labeling** | Meta-Labeler (XGBoost) | Trade success probability | ≥55% confidence | Block trade |
| 6 | **Sentiment** | Market Intelligence | News sentiment alignment | No strong contrary sentiment (>60%) | Block trade |
| 7 | **RSI Extreme** | Technical indicator | Avoid extremes | RSI 10-90 range | Block trade |
| 8 | **Trend Contradiction** | ADX + direction | Don't fight strong trends | ADX ≤35 or aligned | Block trade |

### Decision Logic

```python
all_gates_passed = (
    tcn_probability_gate_passed      # Gate 1: Direction confidence
    and confidence_gate_passed       # Gate 2: Trend strength
    and momentum_gate_passed         # Gate 3: Momentum check
    and risk_gate_passed             # Gate 4: Drawdown limit
    and meta_gate_passed             # Gate 5: Meta-labeler (if trained)
    and sentiment_gate_passed        # Gate 6: Sentiment alignment
    and rsi_gate_passed              # Gate 7: RSI range check
    and trend_gate_passed            # Gate 8: Trend alignment
)

# Trade execution only if all gates pass
if all_gates_passed:
    execute_trade()
else:
    log_blocked_trade()
```

### Permissive Mode

When `permissive_mode=True`, gates with version mismatches are bypassed with warnings. This allows graceful degradation when sklearn models have compatibility issues.

```python
# Enable permissive mode in config:
inference:
  permissive_mode: true
```

### Gate Bypass Conditions

| Condition | Behavior |
|-----------|----------|
| Meta-labeler not trained | Gate 5 skipped with warning |
| Sklearn version mismatch | Gate bypassed with warning (permissive mode) |
| Missing market data | Gate skipped with error |
| API rate limit | Trade delayed, not blocked |

---

## Configuration

### H1 Timeframe Settings (Default)

Default configuration uses **H1 (Hourly)** candles. Key settings in [`config/config_improved_H1.yaml`](config/config_improved_H1.yaml):

```yaml
# === TIMEFRAME ===
fx:
  granularity: H1                    # Hourly candles

# === DIRECTION LABELING ===
direction_lookahead: 24              # 24 hours (24 bars) lookahead
direction_threshold: 0.003           # 0.3% min move for clear label

# === MODEL ARCHITECTURE ===
transformer:
  d_model: 32
  num_heads: 4
  num_layers: 2
  dropout: 0.4

# === TRAINING ===
training:
  epochs: 200
  early_stopping_patience: 40
  batch_size: 64
  learning_rate: 0.0003

# === RISK ===
buddy:
  stop_loss_pips: 15.0               # 15 pip SL
  take_profit_pips: 30.0             # 30 pip TP
  risk_per_trade_pct: 0.02           # 2% risk per trade
```

### Configuration Hierarchy

Configurations are loaded in the following priority order (higher priority overrides lower):

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **User config file** (`~/.ml_engine/config.yaml`)
4. **Project config file** (`config/config_improved_H1.yaml`)
5. **Default values** (lowest priority)

### Environment Variables

Create a `.env` file in the project root:

```bash
# === OANDA API CREDENTIALS ===
OANDA_API_TOKEN=your_practice_api_token
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice  # or 'live'

# === MODEL PATHS ===
MODEL_PATH=trained_data/models
DATA_PATH=market_data

# === LOGGING ===
LOG_LEVEL=INFO
LOG_PATH=logs

# === CONDA ENVIRONMENT ===
BUDDY_CONDA_ENV=tf-metal  # or ml_engine_py312 for Intel
```

---

## Platform-Specific Settings

### Apple Silicon (M1/M2/M3)

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

#### Apple Silicon Setup

```bash
# Create conda environment
conda env create -f environment_tf_metal.yml

# Activate environment
conda activate tf-metal

# Verify Metal support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Performance Tips

| Setting | Value | Reason |
|---------|-------|--------|
| `recurrent_dropout` | `0.0` | Non-zero causes 10x slowdown on Metal |
| `batch_size` | `64` | Optimal for Metal GPU memory |
| `mixed_precision` | `false` | Metal doesn't fully support FP16 |
| `steps_per_execution` | `10` | Reduces Python overhead |
| `jit_compile` | `false` | Avoids Metal compatibility issues |

### Intel Mac

Uses `ml_engine_py312` conda environment. Auto-detected in [`bin/Buddy`](bin/Buddy):

```bash
# Auto-detection in bin/Buddy:
if [[ "$(uname -m)" == "x86_64" ]]; then
    ENV_NAME="${BUDDY_CONDA_ENV:-ml_engine_py312}"
else
    ENV_NAME="${BUDDY_CONDA_ENV:-tf-metal}"
fi
```

#### Intel Mac Setup

```bash
# Create conda environment
conda env create -f environment_intel_mac.yml

# Activate environment
conda activate ml_engine_py312

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### Performance Tips

| Setting | Value | Reason |
|---------|-------|--------|
| `recurrent_dropout` | `0.1-0.2` | Can use non-zero on CPU |
| `batch_size` | `32-64` | Adjust based on available RAM |
| `mixed_precision` | `true` | Can use FP16 for speedup |
| `steps_per_execution` | `1` | Not needed for CPU |
| `jit_compile` | `true` | Can use XLA compilation |

---

## Code Patterns

### Feature Engineering

Always use **normalized features** from [`src/core/modular_data_loaders.compute_normalized_features()`](src/core/modular_data_loaders.py) - models are instrument-agnostic.

```python
# Correct pattern:
from src.core.modular_data_loaders import compute_normalized_features

# Compute normalized features (instrument-agnostic)
features = compute_normalized_features(
    df=price_data,
    seq_len=60,
    feature_names=['returns', 'volume', 'rsi', 'macd']
)
```

### Model Loading

Models save as `.keras` with companion `.meta.pkl` containing:
- Scaler parameters
- Feature list (`feature_names`)
- Tier-2 calibration data

```python
# Correct pattern:
from pathlib import Path
import pickle

# Load model and metadata
model_path = Path("trained_data/models") / "transformer_direction.keras"
meta_path = Path("trained_data/models") / "transformer_direction.meta.pkl"

with open(meta_path, 'rb') as f:
    metadata = pickle.load(f)

model = tf.keras.models.load_model(model_path)

# Pair-specific models:
pair_model_path = Path("trained_data/models/EUR_USD") / "transformer_direction.keras"
pair_meta_path = Path("trained_data/models/EUR_USD") / "transformer_direction.meta.pkl"
```

### Custom Keras Layers

All custom layers in [`src/models/tensorflow_models.py`](src/models/tensorflow_models.py) use `@tf.keras.utils.register_keras_serializable()` for model serialization.

```python
# Correct pattern:
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Layer logic here
        return inputs
```

### Data Cleaning Pattern

Always clean data to handle NaN and infinity values:

```python
import numpy as np

# Standard data cleaning pattern
def clean_dataframe(df):
    """Clean DataFrame by handling NaN and infinity values."""
    return (
        df
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )

# Usage:
clean_df = clean_dataframe(df)
```

### Configuration Loading Pattern

```python
import yaml
from pathlib import Path

def load_config(config_path: str = "config/config_improved_H1.yaml"):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Usage:
config = load_config()
```

---

## Model Files

### Model Artifacts

After training, models are saved to `trained_data/models/`:

| File | Description | Type | Gate |
|------|-------------|------|------|
| `transformer_direction.keras` | Transformer direction model | Keras | Gate 1 |
| `transformer_direction.meta.pkl` | Scalers and metadata | Pickle | - |
| `transformer_direction.ema.pkl` | EMA weights | Pickle | - |
| `xgb_momentum.pkl` | XGBoost momentum gate | Pickle | Gate 3 |
| `ridge_confidence.pkl` | Ridge confidence gate | Pickle | Gate 2 |
| `rf_risk.pkl` | RandomForest risk gate | Pickle | Gate 4 |
| `meta_labeler.pkl` | Meta-labeler trade success predictor | Pickle | Gate 5 |
| `modular_ensemble.meta.json` | Ensemble configuration | JSON | - |
| `rl_position_sizer.zip` | RL position sizing agent | Zip | - |

### Pair-Specific Models

Pair-specific models are stored in `trained_data/models/{PAIR}/`:

```
trained_data/models/
├── EUR_USD/
│   ├── transformer_direction.keras
│   ├── transformer_direction.meta.pkl
│   ├── xgb_momentum.pkl
│   ├── ridge_confidence.pkl
│   ├── rf_risk.pkl
│   └── meta_labeler.pkl
├── GBP_USD/
│   └── ...
└── USD_JPY/
    └── ...
```

**Note:** Meta-labeler is trained by default during `train-buddy` and saves to both pair-specific and generic paths.

### Model Versioning

Models include version information in their metadata:

```python
# Model metadata structure
metadata = {
    'version': '2.1.0',
    'trained_at': '2025-02-07T12:00:00Z',
    'instrument': 'EUR_USD',
    'config': config_dict,
    'feature_names': [...],
    'scaler_params': {...},
    'calibration_data': {...}
}
```

---

## Testing

### Test Conventions

- Tests in `tests/` use pytest
- Some tests are ignored in CI (see [`pytest.ini`](pytest.ini))
- Integration tests require OANDA credentials in `.env`

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_buddy_intelligent_mode.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run with verbose output
pytest tests/ -vv

# Run specific test function
pytest tests/test_prediction_collapse.py::test_collapse_detection -v
```

### Test Structure

```
tests/
├── test_prediction_collapse.py      # Prediction collapse detection
├── test_buddy_intelligent_mode.py    # LLM integration tests
├── test_buddy_commands.py           # CLI command tests
├── test_buddy_scan.py               # Scanner tests
├── test_calibration_integration.py   # Calibration tests
├── test_market_intel.py            # Market intelligence tests
├── test_rl_training.py              # RL training tests
└── ...
```

### Writing Tests

```python
import pytest
from src.core.modular_inference import InferenceConfig

def test_gate_thresholds():
    """Test that gate thresholds are properly configured."""
    config = InferenceConfig()
    assert config.min_tcn_probability == 0.60
    assert config.min_confidence == 50.0
    assert config.min_momentum == 0.20
    assert config.max_drawdown_pct == 0.025
```

---

## Best Practices

### Code Style

Follow these conventions for consistent code style:

| Category | Convention | Example |
|----------|-----------|---------|
| **File naming** | `snake_case` | `modular_inference.py` |
| **Class naming** | `PascalCase` | `InferenceConfig` |
| **Function naming** | `snake_case` | `compute_normalized_features()` |
| **Constants** | `UPPER_SNAKE_CASE` | `DEFAULT_BATCH_SIZE` |
| **Private members** | `_leading_underscore` | `_internal_method()` |

### Documentation

- Use docstrings for all public functions and classes
- Include type hints for function signatures
- Add inline comments for complex logic

```python
def compute_normalized_features(
    df: pd.DataFrame,
    seq_len: int,
    feature_names: list[str]
) -> np.ndarray:
    """
    Compute normalized features for model input.

    Args:
        df: Input DataFrame with price data
        seq_len: Sequence length for time series
        feature_names: List of feature names to extract

    Returns:
        Normalized feature array with shape (n_samples, seq_len, n_features)

    Raises:
        ValueError: If required columns are missing from DataFrame
    """
    # Implementation here
    pass
```

### Error Handling

Always handle potential errors gracefully:

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_model(model_path: Path):
    """Load model with error handling."""
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
```

### Logging

Use structured logging for better debugging:

```python
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use structured logging for important events
logger.info(json.dumps({
    'event': 'trade_executed',
    'instrument': 'EUR_USD',
    'direction': 'LONG',
    'size': 1000,
    'confidence': 0.75
}))
```

### Git Conventions

Follow these Git conventions:

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat: add Mamba model support` |
| `fix` | Bug fix | `fix: resolve prediction collapse issue` |
| `docs` | Documentation | `docs: update gate thresholds` |
| `refactor` | Code refactoring | `refactor: simplify feature engineering` |
| `test` | Test changes | `test: add collapse detection tests` |
| `chore` | Maintenance | `chore: update dependencies` |

---

## Error Handling

### Common Pitfalls

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **NaN in features** | Model predictions are NaN | Always clean with `df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)` |
| **Sequence length mismatch** | Shape errors during inference | Ensure `seq_len: 60` matches between training and inference |
| **Path changes** | File not found errors | Source files now in `src/` subfolders, not root |
| **Config path** | Config not found errors | Default config is `config/config_improved_H1.yaml`, not root |
| **recurrent_dropout** | 10x slowdown on Metal | Keep at `0.0` on Metal - non-zero causes massive slowdown |
| **Prediction Collapse** | All predictions in one class | See [`docs/PREDICTION_COLLAPSE_SYSTEM.md`](docs/PREDICTION_COLLAPSE_SYSTEM.md) for detection/recovery |

### Error Recovery Strategies

#### Prediction Collapse Detection

The training system includes a comprehensive prediction collapse detection and recovery system:

**Graduated Detection Levels**:
- **80-85%**: Early warning (informational)
- **85-90%**: Moderate imbalance (warning)
- **>90%**: Severe collapse (intervention triggered)

**Progressive Recovery Strategies** (5 attempts):
1. **Attempt 1-2**: Restore best balanced weights (LR × 0.5, 0.3)
2. **Attempt 3**: Perturb output layer (noise 0.15, LR × 0.4)
3. **Attempt 4**: Perturb all layers (noise 0.05-0.2, LR × 0.2)
4. **Attempt 5**: Reinitialize output layer (LR × 0.6)

**Key Features**:
- Prediction history tracking (last 10 checks)
- Balance metric (0.0-1.0 scale)
- Best weight checkpointing (threshold 0.25)
- Detailed failure logging

**Documentation**: See [`docs/PREDICTION_COLLAPSE_SYSTEM.md`](docs/PREDICTION_COLLAPSE_SYSTEM.md) and [`docs/TRAINING_TROUBLESHOOTING.md`](docs/TRAINING_TROUBLESHOOTING.md)

**Tests**: [`tests/test_prediction_collapse.py`](tests/test_prediction_collapse.py)

#### API Rate Limiting

```python
import time
from functools import wraps

def rate_limit(max_calls: int, period: float):
    """Decorator to rate limit API calls."""
    calls = []
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls older than period
            calls[:] = [c for c in calls if now - c < period]
            if len(calls) >= max_calls:
                sleep_time = period - (now - calls[0])
                time.sleep(sleep_time)
                calls[:] = []
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

#### Model Loading Fallback

```python
def load_model_with_fallback(model_path: Path, fallback_path: Path = None):
    """Load model with fallback to default if pair-specific model not found."""
    try:
        return tf.keras.models.load_model(model_path)
    except (FileNotFoundError, OSError) as e:
        if fallback_path and fallback_path.exists():
            logger.warning(f"Pair-specific model not found, using fallback: {e}")
            return tf.keras.models.load_model(fallback_path)
        raise
```

### Debugging Guidelines

#### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use debug logging for detailed information
logger.debug(f"Feature shape: {features.shape}")
logger.debug(f"Gate results: {gate_results}")
```

#### Model Profiling

```python
import time

def profile_model_inference(model, input_data, n_runs=100):
    """Profile model inference time."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = model.predict(input_data)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time * 1000:.2f}ms")
    return avg_time
```

#### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile a function
python -m memory_profiler your_script.py
```

---

## Performance Optimization

### Training Optimization

#### Batch Size Selection

| Platform | Recommended Batch Size | Reason |
|----------|----------------------|--------|
| Apple Silicon (M1/M2/M3) | 64 | Optimal for Metal GPU memory |
| Intel Mac | 32-64 | Adjust based on available RAM |
| Cloud GPU | 128-256 | Larger batch for faster training |

#### Learning Rate Scheduling

```python
# Use warm restarts for better convergence
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

initial_learning_rate = 0.001
decay_steps = 1000
alpha = 0.0

lr_schedule = CosineDecayRestarts(
    initial_learning_rate,
    first_decay_steps=decay_steps,
    t_mul=2.0,
    m_mul=1.0,
    alpha=alpha
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

#### Mixed Precision Training

**Apple Silicon**: Not supported (keep `mixed_precision: false`)

**Intel Mac / Cloud GPU**: Can enable for speedup

```python
# Enable mixed precision (only on supported platforms)
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
except Exception as e:
    logger.warning(f"Mixed precision not available: {e}")
```

### Inference Optimization

#### Model Quantization

```python
# Convert model to TensorFlow Lite for faster inference
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### Batch Inference

```python
# Process multiple samples at once for efficiency
def batch_predict(model, data, batch_size=64):
    """Predict in batches for efficiency."""
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        pred = model.predict(batch, verbose=0)
        predictions.append(pred)
    return np.concatenate(predictions, axis=0)
```

#### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_technical_indicators(symbol: str, timeframe: str):
    """Cache technical indicator computations."""
    # Expensive computation here
    return indicators
```

### Data Optimization

#### Efficient Data Loading

```python
# Use tf.data for efficient data pipeline
def create_dataset(features, labels, batch_size=64, shuffle=True):
    """Create efficient tf.data dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

#### Feature Caching

```python
import pickle
from pathlib import Path

def cache_features(features, cache_path: Path):
    """Cache computed features to disk."""
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)

def load_cached_features(cache_path: Path):
    """Load cached features from disk."""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)
```

---

## Improvement Roadmap

### Quick Wins

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| High | Enable LightGBM | Low | High |
| High | Label smoothing | Low | Medium |
| Medium | Focal loss tuning | Low | Medium |
| Medium | Warm restart LR | Low | Medium |

#### Enable LightGBM

```bash
# Install LightGBM
pip install lightgbm

# Swap Ridge wrapper in config
models:
  confidence:
    type: lightgbm  # instead of ridge
    params:
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 100
```

#### Label Smoothing

```yaml
# Add to config
training:
  label_smoothing: 0.1  # Reduces overconfidence
```

#### Focal Loss

```yaml
# Already implemented, enable in config
training:
  direction_loss: focal
  focal_alpha: 0.25
  focal_gamma: 2.0
```

### Model Architecture Upgrades

#### Mamba/State Space Models

```python
# Mamba-style SSMs handle variable-length dependencies better
# pip install mamba-ssm  # or implement custom S4 layer

class MambaLayer(tf.keras.layers.Layer):
    """Mamba-style State Space Model layer."""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # Implementation here
```

#### Uncertainty Quantification

```python
# Monte Carlo Dropout at inference
def predict_with_uncertainty(model, X, n_samples=30):
    """
    Predict with uncertainty using Monte Carlo Dropout.
    
    Returns:
        mean: Mean prediction
        std: Standard deviation (uncertainty estimate)
    """
    preds = [model(X, training=True) for _ in range(n_samples)]
    preds = np.stack(preds, axis=0)
    return np.mean(preds, axis=0), np.std(preds, axis=0)
```

### Multi-Pair Training

Currently models are trained per-pair. Future improvement:

```python
# Foundation model approach:
# 1. Pre-train on ALL pairs with contrastive loss
# 2. Fine-tune classification heads per instrument
# 3. Add pair embeddings for instrument identity

class FoundationModel(tf.keras.Model):
    """Foundation model for multi-pair training."""
    def __init__(self, d_model, num_pairs):
        super().__init__()
        self.encoder = TransformerEncoder(d_model)
        self.pair_embeddings = tf.keras.layers.Embedding(num_pairs, d_model)
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x, pair_id):
        # Encode features
        encoded = self.encoder(x)
        # Add pair embedding
        pair_emb = self.pair_embeddings(pair_id)
        combined = encoded + pair_emb
        # Classify
        return self.classifier(combined)
```

### RL Position Sizing

The [`rl_position_sizing.py`](rl_position_sizing.py) is integrated and can be trained:

```bash
# Train RL agent
python main.py train-rl-sizer --timesteps 500000

# Use in inference
python main.py buddy --instrument EUR_USD --use-rl-sizer --execute
```

### Dependency Upgrades

```bash
# Upgrade TensorFlow for better Metal support
pip install tensorflow>=2.16 keras>=3.0

# Add Optuna for hyperparameter tuning
pip install optuna

# ONNX export for faster inference
pip install tf2onnx onnxruntime

# Add Weights & Biases for experiment tracking
pip install wandb
```

### Advanced Features

#### Hyperparameter Optimization with Optuna

```python
import optuna

def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
    }
    
    # Train model with params
    model = train_model(params)
    accuracy = evaluate_model(model)
    
    return accuracy

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### Experiment Tracking with Weights & Biases

```python
import wandb

# Initialize wandb
wandb.init(project="ml-engine", config=config)

# Log metrics during training
wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'learning_rate': current_lr
})

# Finish run
wandb.finish()
```

---

## Appendix

### Gitignored Files

These are machine-specific and NOT tracked:

```
# Model artifacts
trained_data/
*.pkl
*.keras
*.h5
*.tflite

# Data files
market_data/*.csv
*.csv

# Log files
*.log
logs/

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### Useful Commands

```bash
# Check model file sizes
du -sh trained_data/models/*

# Find large files
find . -type f -size +100M -exec ls -lh {} \;

# Count lines of code
find src -name "*.py" | xargs wc -l

# Run linter
flake8 src/

# Run formatter
black src/

# Type checking
mypy src/
```

### Resources

| Resource | URL |
|----------|-----|
| TensorFlow Metal | https://developer.apple.com/metal/tensorflow-plugin/ |
| OANDA API Docs | https://developer.oanda.com/ |
| Keras Documentation | https://keras.io/ |
| Pytest Documentation | https://docs.pytest.org/ |

---

## Changelog

### Version 2.1.0 (2025-02-07)
- Added comprehensive documentation structure
- Improved error handling guidelines
- Added performance optimization section
- Enhanced best practices and patterns
- Added improvement roadmap

### Version 2.0.0 (2025-01-XX)
- Implemented prediction collapse detection system
- Added meta-labeling gate
- Improved walk-forward validation
- Enhanced market intelligence features

---

**End of Document**
