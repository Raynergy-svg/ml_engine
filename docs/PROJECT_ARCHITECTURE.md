# ML Engine Project Architecture

> A comprehensive guide to the FX Trading Bot ML Engine, its components, and how they integrate with `main.py`.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Entry Point: main.py](#entry-point-mainpy)
4. [Core Module Breakdown](#core-module-breakdown)
5. [Data Flow Pipeline](#data-flow-pipeline)
6. [Training Pipeline](#training-pipeline)
7. [Configuration System](#configuration-system)
8. [File-by-File Reference](#file-by-file-reference)
9. [Dependency Graph](#dependency-graph)
10. [M1 Metal Optimizations](#m1-metal-optimizations)

---

## Project Overview

This ML Engine is a sophisticated **Forex (FX) trading bot** that uses deep learning models (LSTM, Transformer, TCN, TFT) to predict:
- **Direction**: Whether price will go up, down, or stay neutral
- **Magnitude**: Expected price movement magnitude
- **Volatility**: Market volatility prediction
- **Risk**: Risk assessment for position sizing

The system is optimized for **Apple M1/M2/M3 Silicon** using TensorFlow Metal acceleration.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main.py (CLI)                                  │
│                         Command Dispatcher & Orchestrator                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Data Layer     │       │  Model Layer    │       │  Training Layer │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ oanda_practice  │       │ models_enhanced │       │ tensorflow_     │
│ data_loader     │       │ tensorflow_     │       │   engine        │
│ feature_        │       │   models        │       │ tensorflow_     │
│   engineering   │       │ custom_layers   │       │   data_pipeline │
│ buddy_training_ │       │ m1_metal_       │       │ walkforward_    │
│   helpers       │       │   optimizer     │       │   validation    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                                      ▼
                          ┌─────────────────────┐
                          │   Config Layer      │
                          ├─────────────────────┤
                          │ config.yaml         │
                          │ config_m1_optimized │
                          │ utils.py            │
                          └─────────────────────┘
```

---

## Entry Point: main.py

`main.py` is the **central orchestrator** of the entire ML engine. It's a comprehensive CLI application with ~6,200 lines that handles:

### Command Structure

```python
# Main commands available via CLI:
python main.py <command> [options]

# Key commands:
train-buddy      # Primary training command for the ML model
evaluate         # Model evaluation on test data
predict          # Generate predictions
backtest         # Run backtesting simulations
dashboard        # Launch interactive Streamlit dashboard
```

### Key Components in main.py

#### 1. **BuddyTrainingOptions** (Dataclass)
```python
@dataclass(frozen=True)
class BuddyTrainingOptions:
    seq_len: int = 60           # Sequence length for LSTM/Transformer
    epochs: int = 100           # Training epochs
    batch_size: int = 128       # Batch size (M1 optimized)
    lr: float = 0.0005          # Learning rate
    patience: int = 15          # Early stopping patience
    model_type: str = "tcn"     # Model architecture
    mixed_precision: bool = True # FP16 acceleration
    # ... more options
```

#### 2. **Command Dispatch System**
```python
def main():
    args = parse_arguments()
    command_map = {
        "train-buddy": train_buddy,
        "evaluate": evaluate_model,
        "predict": generate_predictions,
        # ...
    }
    _dispatch_command(args, command_map)
```

#### 3. **Training Flow** (`_train_buddy_impl`)
This is the core training function that:
1. Loads configuration from YAML
2. Fetches/loads data (OANDA or CSV)
3. Initializes the TensorFlow engine
4. Builds the model
5. Runs training with callbacks
6. Saves artifacts (model, scalers, metrics)

---

## Core Module Breakdown

### Data Acquisition & Processing

#### `oanda_practice.py`
**Purpose**: OANDA v20 REST API client for live/practice data fetching.

```python
class OandaPracticeClient:
    @classmethod
    def from_env(cls):
        """Load credentials from environment variables"""
        # Reads OANDA_API_TOKEN and OANDA_ACCOUNT_ID
        
    def fetch_candles(self, instrument, granularity, count):
        """Fetch OHLCV candle data"""
        
    def create_order(self, instrument, units, side):
        """Execute trades"""
```

**Integration with main.py**:
```python
# In main.py
from oanda_practice import OandaPracticeClient

def _oanda_fetch_to_csv(oanda_fetch: OandaFetch) -> Path:
    client = OandaPracticeClient.from_env()  # Uses env vars
    candles = client.fetch_candles(...)
    # Saves to CSV for training
```

---

#### `data_loader.py`
**Purpose**: Enhanced data loading with preprocessing, feature engineering, and sequence creation.

```python
class DataLoader:
    def load_csv(self, path: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing pipeline"""
        
    def create_sequences(self, df, seq_len) -> Tuple[np.ndarray, dict]:
        """Create time-series sequences for LSTM/Transformer"""
```

**Integration with main.py**:
```python
# Called from buddy_training_helpers.py
from data_loader import DataLoader

loader = DataLoader(config)
df = loader.load_csv(csv_path)
df = loader.preprocess(df)
```

---

#### `feature_engineering.py`
**Purpose**: Technical indicator calculation and advanced feature generation.

```python
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, MACD, Bollinger Bands, etc."""
    
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR, historical volatility, GARCH-like features"""
    
def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Add lagged versions of key features"""
```

**Integration**:
```python
# Called from data_loader.py and tensorflow_data_pipeline.py
from feature_engineering import add_technical_indicators, add_volatility_features

df = add_technical_indicators(df)
df = add_volatility_features(df)
```

---

#### `buddy_training_helpers.py`
**Purpose**: Helper functions specifically for the `train-buddy` command workflow.

```python
def _resolve_training_csv_path(csv_path, oanda_fetch) -> Path:
    """Resolve CSV path - either use existing or fetch from OANDA"""
    if oanda_fetch:
        return oanda_fetch_to_csv(oanda_fetch)  # Calls main.py function
    return Path(csv_path)

def _buddy_load_and_validate_csv(csv_path, oanda_fetch, config) -> pd.DataFrame:
    """Load CSV with validation and preprocessing"""
```

**Integration with main.py**:
```python
# In main.py's _train_buddy_impl()
from buddy_training_helpers import _buddy_load_and_validate_csv

df = _buddy_load_and_validate_csv(
    csv_path=args.csv,
    oanda_fetch=oanda_fetch_config,
    config=config
)
```

---

### Model Architecture

#### `models_enhanced.py`
**Purpose**: State-of-the-art TensorFlow/Keras model implementations.

```python
# Available model types:
ENHANCED_MODEL_REGISTRY = {
    'lstm': build_lstm_model,
    'attentive_lstm': build_attentive_lstm_model,
    'transformer': build_transformer_model,
    'tft': build_tft_model,           # Temporal Fusion Transformer
    'tcn': build_tcn_model,           # Temporal Convolutional Network
}

def build_tcn_model(input_shape, config):
    """
    TCN is recommended for M1 Metal as it:
    - Uses 1D convolutions (Metal-optimized)
    - No recurrent dropout issues
    - Parallelizable computations
    """
```

**M1 Optimization Note**:
```python
# Conditional recurrent dropout for M1
import platform
is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
DEFAULT_RECURRENT_DROPOUT = 0.0 if is_apple_silicon else 0.15
```

**Integration with main.py**:
```python
# Called from tensorflow_engine.py
from models_enhanced import ENHANCED_MODEL_REGISTRY

model = ENHANCED_MODEL_REGISTRY[model_type](input_shape, config)
```

---

#### `tensorflow_models.py`
**Purpose**: Additional TensorFlow model implementations including the Temporal Fusion Transformer.

```python
class TFTemporalFusionTransformer(tf.keras.Model):
    """
    Full TFT implementation with:
    - Variable Selection Networks
    - Gated Residual Networks
    - Interpretable Multi-Head Attention
    - Multi-task output heads
    """
```

---

#### `custom_layers.py`
**Purpose**: Custom Keras layers for advanced architectures.

```python
@tf.keras.utils.register_keras_serializable()
class GatedLinearUnit(tf.keras.layers.Layer):
    """GLU activation for TFT"""

@tf.keras.utils.register_keras_serializable()
class GatedResidualNetwork(tf.keras.layers.Layer):
    """GRN building block for TFT"""

@tf.keras.utils.register_keras_serializable()
class VariableSelectionNetwork(tf.keras.layers.Layer):
    """Feature importance weighting"""

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding for Transformers"""
```

---

### Training Infrastructure

#### `tensorflow_engine.py`
**Purpose**: Core TensorFlow training engine with callbacks, loss functions, and optimization.

```python
class TensorFlowEngine:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.callbacks = []
        
    def build_model(self, input_shape: tuple):
        """Build model based on config['model']['type']"""
        model_type = self.config['model']['type']
        self.model = ENHANCED_MODEL_REGISTRY[model_type](input_shape, self.config)
        
    def compile_model(self):
        """Compile with multi-task losses and M1 optimizations"""
        losses = {
            'direction': 'categorical_crossentropy',
            'magnitude': 'huber',
            'volatility': 'mse',
            'risk': 'huber',  # Huber for numerical stability
        }
        
    def setup_callbacks(self):
        """Setup training callbacks"""
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=self.config['patience']),
            tf.keras.callbacks.ModelCheckpoint(...),
            tf.keras.callbacks.ReduceLROnPlateau(...),
            tf.keras.callbacks.TensorBoard(...),
        ]
        
    def train(self, train_ds, val_ds, epochs):
        """Execute training loop"""
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=self.callbacks,
        )
```

**Integration with main.py**:
```python
# In _train_buddy_impl()
from tensorflow_engine import TensorFlowEngine

engine = TensorFlowEngine(config)
engine.build_model(input_shape)
engine.compile_model()
engine.setup_callbacks()
history = engine.train(train_ds, val_ds, epochs=config['epochs'])
```

---

#### `tensorflow_data_pipeline.py`
**Purpose**: TensorFlow-compatible data pipeline with `tf.data.Dataset` generation.

```python
class TensorFlowDataPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.scalers = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Full pipeline:
        1. Feature engineering
        2. Target generation (direction, magnitude, volatility, risk)
        3. Scaling (StandardScaler for features, MinMaxScaler for risk)
        4. Sequence creation
        5. tf.data.Dataset creation with optimization
        """
        
    def create_optimized_dataset(self, X, y, batch_size, shuffle=True):
        """Create M1-optimized tf.data.Dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.cache()              # Cache in memory
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Overlap compute/IO
        return dataset
```

**Integration with main.py**:
```python
# In _train_buddy_impl()
from tensorflow_data_pipeline import TensorFlowDataPipeline

pipeline = TensorFlowDataPipeline(config)
train_ds, val_ds = pipeline.prepare_data(df)
```

---

#### `m1_metal_optimizer.py`
**Purpose**: Apple Silicon-specific TensorFlow optimizations.

```python
def configure_tf_metal():
    """Configure TensorFlow for optimal Metal performance"""
    # Enable memory growth
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
def apply_m1_regularization(model):
    """
    Apply M1-compatible regularization:
    - GaussianNoise instead of recurrent dropout
    - SpatialDropout1D for sequence models
    """
    
def create_optimized_dataset(dataset, batch_size):
    """Apply M1-optimized dataset configuration"""
    return dataset.cache().prefetch(tf.data.AUTOTUNE)
```

**Integration with main.py**:
```python
# Called at startup in main.py or tensorflow_engine.py
from m1_metal_optimizer import configure_tf_metal

configure_tf_metal()  # Set up Metal optimizations
```

---

#### `walkforward_validation.py`
**Purpose**: Time-series cross-validation without look-ahead bias.

```python
class WalkForwardValidator:
    """
    Walk-forward validation for time-series:
    
    |----Train----|--Val--|----Test----|
                  |----Train----|--Val--|----Test----|
                                |----Train----|--Val--|----Test----|
    
    Ensures no future data leaks into training.
    """
    
    def __init__(self, n_splits=5, train_ratio=0.7, val_ratio=0.15):
        self.n_splits = n_splits
        
    def split(self, X, y):
        """Generate train/val/test splits"""
        for fold in range(self.n_splits):
            yield train_idx, val_idx, test_idx
```

**Integration**:
```python
# In train_optimized_m1.py or via --walkforward flag in main.py
from walkforward_validation import WalkForwardValidator

validator = WalkForwardValidator(n_splits=5)
for train_idx, val_idx, test_idx in validator.split(X, y):
    # Train on this fold
```

---

### Utility & Configuration

#### `utils.py`
**Purpose**: Shared utility functions across the project.

```python
def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(level: str = 'INFO'):
    """Configure logging for the application"""
    
def ensure_directory(path: str):
    """Create directory if it doesn't exist"""
    
def save_artifact(obj, path: str):
    """Save model artifacts (scalers, encoders, etc.)"""
```

**Integration with main.py**:
```python
from utils import load_config, setup_logging

config = load_config('config_m1_optimized.yaml')
setup_logging(config.get('log_level', 'INFO'))
```

---

## Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                         │
└─────────────────────────────────────────────────────────────────────────────┘

1. DATA ACQUISITION
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │ OANDA API    │     │ CSV File     │     │ Other        │
   │ (Live/Demo)  │     │ (Historical) │     │ Sources      │
   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
          │                    │                    │
          └──────────────────┬─┴────────────────────┘
                             │
                             ▼
2. RAW DATA LOADING    ┌──────────────────┐
                       │ data_loader.py   │
                       │ load_csv()       │
                       └────────┬─────────┘
                                │
                                ▼
3. FEATURE ENGINEERING ┌──────────────────────────────────────┐
                       │ feature_engineering.py               │
                       │ - Technical indicators (RSI, MACD)   │
                       │ - Volatility features (ATR)          │
                       │ - Lag features                       │
                       │ - Custom features                    │
                       └────────┬─────────────────────────────┘
                                │
                                ▼
4. TARGET GENERATION   ┌──────────────────────────────────────┐
                       │ tensorflow_data_pipeline.py          │
                       │ - Direction (up/down/neutral)        │
                       │ - Magnitude (price change)           │
                       │ - Volatility (future vol)            │
                       │ - Risk (rolling std)                 │
                       └────────┬─────────────────────────────┘
                                │
                                ▼
5. SCALING             ┌──────────────────────────────────────┐
                       │ tensorflow_data_pipeline.py          │
                       │ - StandardScaler for features        │
                       │ - MinMaxScaler for risk target       │
                       └────────┬─────────────────────────────┘
                                │
                                ▼
6. SEQUENCE CREATION   ┌──────────────────────────────────────┐
                       │ tensorflow_data_pipeline.py          │
                       │ - Create rolling windows             │
                       │ - Shape: (samples, seq_len, features)│
                       └────────┬─────────────────────────────┘
                                │
                                ▼
7. TF.DATA DATASET     ┌──────────────────────────────────────┐
                       │ tensorflow_data_pipeline.py          │
                       │ - tf.data.Dataset creation           │
                       │ - Batching, caching, prefetching     │
                       │ - Train/Val/Test splits              │
                       └────────┬─────────────────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Ready for Model  │
                       └──────────────────┘
```

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

python main.py train-buddy --config config_m1_optimized.yaml

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION (main.py)                                                 │
│    - Parse CLI arguments                                                    │
│    - Load YAML configuration                                                │
│    - Configure TF Metal (m1_metal_optimizer.py)                            │
│    - Set mixed precision policy                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. DATA LOADING (buddy_training_helpers.py → data_loader.py)               │
│    - Resolve CSV path or fetch from OANDA                                   │
│    - Validate data structure                                                │
│    - Initial preprocessing                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. PIPELINE SETUP (tensorflow_data_pipeline.py)                            │
│    - Feature engineering                                                    │
│    - Target generation                                                      │
│    - Scaling and normalization                                              │
│    - Sequence creation                                                      │
│    - tf.data.Dataset creation                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. MODEL BUILDING (tensorflow_engine.py → models_enhanced.py)              │
│    - Select model architecture (TCN recommended for M1)                     │
│    - Build model with config parameters                                     │
│    - Apply M1 regularization                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. COMPILATION (tensorflow_engine.py)                                       │
│    - Multi-task loss configuration                                          │
│    - Loss weights balancing                                                 │
│    - Optimizer setup (Adam with learning rate)                              │
│    - Metrics configuration                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. CALLBACKS SETUP (tensorflow_engine.py)                                   │
│    - EarlyStopping (patience=15)                                            │
│    - ModelCheckpoint (save best)                                            │
│    - ReduceLROnPlateau                                                      │
│    - TensorBoard logging                                                    │
│    - CSVLogger                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. TRAINING LOOP (tensorflow_engine.py)                                     │
│    - model.fit() with train/val datasets                                    │
│    - Epoch-by-epoch optimization                                            │
│    - Callback execution                                                     │
│    - Progress monitoring                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. ARTIFACT SAVING (main.py)                                                │
│    - Save trained model (.keras)                                            │
│    - Save scalers (joblib)                                                  │
│    - Save training history                                                  │
│    - Save configuration snapshot                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 9. EVALUATION (optional)                                                    │
│    - Test set evaluation                                                    │
│    - Metrics calculation                                                    │
│    - Walk-forward validation results                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration System

### Configuration Files

#### `config.yaml` (Default)
Base configuration with all parameters.

#### `config_m1_optimized.yaml` (M1 Optimized)
```yaml
# Model Architecture
model:
  type: tcn              # TCN recommended for M1 (convolutions are Metal-optimized)
  units: 128
  dropout: 0.2
  recurrent_dropout: 0.0  # CRITICAL: Must be 0.0 for M1 Metal

# Training
training:
  batch_size: 128         # Larger batches for M1 efficiency
  epochs: 100
  learning_rate: 0.0005
  patience: 15
  mixed_precision: true   # FP16 acceleration

# Data Pipeline
data:
  seq_len: 60
  cache: true             # Cache dataset in memory
  prefetch: auto          # Prefetch for pipelining

# Multi-task Loss Weights (balanced)
unified_head_loss_weights:
  direction: 5.0          # Reduced from 20.0
  magnitude: 3.0
  volatility: 2.0
  risk: 1.0
```

### Configuration Loading Chain

```
main.py
    │
    ├── utils.load_config('config_m1_optimized.yaml')
    │       │
    │       └── Returns dict with all settings
    │
    ├── BuddyTrainingOptions(config)
    │       │
    │       └── Dataclass with typed defaults + overrides
    │
    ├── TensorFlowEngine(config)
    │       │
    │       └── Uses config for model/training setup
    │
    └── TensorFlowDataPipeline(config)
            │
            └── Uses config for data processing
```

---

## File-by-File Reference

### Core Files (Connected to main.py)

| File | Purpose | Integration Point |
|------|---------|-------------------|
| `main.py` | CLI entry point, orchestrator | Entry point |
| `buddy_training_helpers.py` | Training helper functions | Called by `_train_buddy_impl()` |
| `tensorflow_engine.py` | TensorFlow training engine | Called by `train_buddy()` |
| `tensorflow_data_pipeline.py` | Data pipeline for TF | Called by `TensorFlowEngine` |
| `models_enhanced.py` | Model architectures | Called by `TensorFlowEngine.build_model()` |
| `tensorflow_models.py` | Additional TF models | Imported by `models_enhanced.py` |
| `custom_layers.py` | Custom Keras layers | Imported by model files |
| `data_loader.py` | Data loading utilities | Called by `buddy_training_helpers.py` |
| `feature_engineering.py` | Feature generation | Called by data pipeline |
| `oanda_practice.py` | OANDA API client | Called for `--oanda-live` flag |
| `utils.py` | Shared utilities | Used throughout |
| `m1_metal_optimizer.py` | M1 optimizations | Called at initialization |
| `walkforward_validation.py` | Time-series CV | Called with `--walkforward` flag |

### Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Default configuration |
| `config_m1_optimized.yaml` | M1-optimized settings |
| `.env.local` | Environment variables (OANDA credentials) |

### Training Scripts (Alternative Entry Points)

| File | Purpose |
|------|---------|
| `train_optimized_m1.py` | Standalone M1-optimized training |
| `quick_test_training.py` | Quick validation of setup |
| `run_optimized_training.sh` | Shell script wrapper |

### Diagnostic Files

| File | Purpose |
|------|---------|
| `diagnose_nan.py` | Debug NaN issues in training |

---

## Dependency Graph

```
main.py
├── utils.py
├── buddy_training_helpers.py
│   ├── data_loader.py
│   │   └── feature_engineering.py
│   └── oanda_practice.py
├── tensorflow_engine.py
│   ├── tensorflow_data_pipeline.py
│   │   └── feature_engineering.py
│   ├── models_enhanced.py
│   │   ├── tensorflow_models.py
│   │   │   └── custom_layers.py
│   │   └── custom_layers.py
│   └── m1_metal_optimizer.py
└── walkforward_validation.py
```

### Import Flow Example

```python
# When you run: python main.py train-buddy --config config_m1_optimized.yaml

# 1. main.py starts
import tensorflow as tf
from utils import load_config
from buddy_training_helpers import _buddy_load_and_validate_csv
from tensorflow_engine import TensorFlowEngine

# 2. TensorFlowEngine imports
from models_enhanced import ENHANCED_MODEL_REGISTRY
from tensorflow_data_pipeline import TensorFlowDataPipeline
from m1_metal_optimizer import configure_tf_metal

# 3. models_enhanced imports
from custom_layers import GatedLinearUnit, GatedResidualNetwork
from tensorflow_models import TFTemporalFusionTransformer

# 4. tensorflow_data_pipeline imports
from feature_engineering import add_technical_indicators
from data_loader import DataLoader
```

---

## M1 Metal Optimizations

### Key Optimizations Applied

1. **Disable Recurrent Dropout**
   - `recurrent_dropout=0.0` in all LSTM/GRU layers
   - Prevents fallback to CPU-only paths

2. **Use Metal-Compatible Models**
   - TCN (Temporal Convolutional Network) recommended
   - Transformers work well
   - LSTMs work but slower than TCN

3. **Mixed Precision Training**
   - `mixed_precision: true` enables FP16
   - 2-3x speedup on M1/M2/M3

4. **Optimized Data Pipeline**
   - `dataset.cache()` - stores in memory
   - `dataset.prefetch(AUTOTUNE)` - overlaps compute/IO
   - Large batch sizes (128+)

5. **Memory Growth**
   - `tf.config.experimental.set_memory_growth(gpu, True)`
   - Prevents OOM errors

### Files with M1 Optimizations

| File | Optimization |
|------|--------------|
| `models_enhanced.py` | Conditional `recurrent_dropout=0.0` |
| `m1_metal_optimizer.py` | All M1 helper functions |
| `tensorflow_data_pipeline.py` | Optimized `tf.data` pipeline |
| `config_m1_optimized.yaml` | M1-tuned parameters |
| `main.py` | M1 defaults in `BuddyTrainingOptions` |

---

## Quick Start

### 1. Set Environment Variables
```bash
# Create .env.local file
export OANDA_API_TOKEN="your-oanda-token"
export OANDA_ACCOUNT_ID="your-account-id"
```

### 2. Run Training
```bash
# With OANDA live data
python main.py train-buddy \
    --config config_m1_optimized.yaml \
    --oanda-live \
    --candles 25000

# With existing CSV
python main.py train-buddy \
    --config config_m1_optimized.yaml \
    --csv data/training_data.csv
```

### 3. Monitor with TensorBoard
```bash
tensorboard --logdir logs/
```

### 4. Evaluate Model
```bash
python main.py evaluate --model models/best_model.keras
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow training between epochs | `recurrent_dropout > 0` | Set `recurrent_dropout: 0.0` |
| `risk_loss: nan` | Unscaled risk target | Use MinMaxScaler for risk |
| `OANDA env vars missing` | No credentials | Create `.env.local` file |
| Low direction accuracy | Loss weight imbalance | Reduce `direction` weight to 5.0 |
| OOM errors | Memory not growing | Enable `memory_growth` in TF config |
| `mixed_precision=False` despite config | Config not read properly | **FIXED**: Now reads from `config.mixed_precision` |
| Model type ignored | `model.type` not wired | **FIXED**: Now uses `model.type` (tcn/lstm) |
| TCN not used on M1 | Hardcoded LSTM | **FIXED**: Set `model.type: tcn` in config |

---

---

## Recent Improvements (Dec 2025)

### Config Integration Fixes

The following issues were identified and fixed in `main.py`:

#### 1. **Mixed Precision Not Applied**

**Before**: `mixed_precision` was hardcoded to read from CLI args only with `False` default:
```python
mixed_precision=bool(getattr(args, "mixed_precision", False))  # ❌ Ignores config
```

**After**: Now properly reads from config file first:
```python
mixed_precision_eff = (
    bool(args.mixed_precision)
    if getattr(args, "mixed_precision", None) is not None
    else bool(_cfg_get("mixed_precision", True))  # ✓ Reads from config
)
```

#### 2. **Model Type Ignored**

**Before**: Only LSTM shared encoder was available, `model.type: tcn` was ignored.

**After**: Added full TCN support with new builder function:
```python
def _build_buddy_model_tcn(...):
    """TCN model - 2-3x faster than LSTM on M1 Metal"""
    # Uses dilated causal convolutions
    # Metal-compatible (no recurrent_dropout issues)
```

#### 3. **Config Lookup Enhanced**

**Before**: `_cfg_get` only searched `buddy.train_defaults`:
```python
def _cfg_get(name, default):
    return buddy_train_cfg.get(name, default)  # ❌ Only one place
```

**After**: Now searches multiple config sections:
```python
def _cfg_get(name, default):
    # 1. Check buddy.train_defaults
    # 2. Check root config
    # 3. Check training section
    # 4. Fall back to default
```

### New Model Builders

| Function | Description | M1 Performance |
|----------|-------------|----------------|
| `_build_buddy_model_tcn()` | TCN with dilated causal convolutions | **Fastest** |
| `_build_buddy_model_for_type()` | Factory for any model type | Selects best |
| `_build_buddy_model_shared_encoder()` | LSTM shared encoder | Good |

### Updated Default Values

| Setting | Old Default | New Default | Reason |
|---------|-------------|-------------|--------|
| `model_type` | `lstm` | `tcn` | **2-3x faster on M1** (parallelizable) |
| `mixed_precision` | `False` | `True` | 1.5-2x speedup on M1 |
| `batch_size` | `32` | `128` | Better GPU utilization |
| `shared_encoder` | `True` | `False` | TCN doesn't need it |
| `cache_val` | `False` | `True` | Faster validation |

---

## Summary

The ML Engine is a comprehensive FX trading bot built around `main.py` as the central orchestrator. Key architectural decisions:

1. **Modular Design**: Each component (data, models, training) is separated
2. **Configuration-Driven**: YAML configs control all parameters
3. **M1 Optimized**: Specific optimizations for Apple Silicon
4. **Multi-Task Learning**: Predicts direction, magnitude, volatility, risk
5. **Production-Ready**: Includes walk-forward validation, OANDA integration

The flow is: **CLI → Config → Data Pipeline → Model → Training → Artifacts**

All roads lead back to `main.py`, which coordinates the entire workflow.

