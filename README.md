# Buddy: ML Engine for Financial Market Prediction

**Buddy** is an integrated command-line utility for training, validating, and auditing predictive models tailored specifically for financial markets. Built on TensorFlow with optimized support for Apple Silicon (M1/M2) via TF-Metal, Buddy provides end-to-end workflows for developing and deploying directional prediction models for forex and other financial instruments.

---

## Core Features

- **Multi-head LSTM architecture** with 5 specialized encoding heads (ML, MR, MT, MS, MX)
- **Direction + confidence prediction** with Tier-2 TP/SL calibration for win probability estimation
- **Live OANDA integration** for both training data fetching and practice trading execution
- **Feature engineering pipeline** with 100+ technical indicators, smoothing, and curriculum learning
- **Checkpoint replay system** for reproducible evaluation and out-of-sample validation
- **Interactive REPL mode** for live trading experimentation with buy/sell/close commands
- **Optimized for Apple Silicon** with TF-Metal GPU acceleration and platform-specific tuning

---

## Quick Start

### 1. Environment Setup (Apple M1/M2 Recommended)

Buddy is optimized for Apple Silicon using **Miniforge** and **TensorFlow-Metal**:

```bash
# Install Miniforge (if not already installed)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate

# Create conda environment from provided spec
conda env create -f environment_tf_metal.yml
conda activate tf-metal
```

**Alternative manual setup:**
```bash
conda create -n ml_engine python=3.11
conda activate ml_engine
pip install -r requirements_tf_metal.txt
```

**Verify TF-Metal GPU acceleration:**
```python
import tensorflow as tf
print(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
# Expected output: GPUs Available: 1 (on M1/M2 systems)
```

### 2. Configure OANDA API Credentials

Create a `.env` file in the repository root:

```bash
OANDA_PRACTICE_TOKEN=your_practice_api_token_here
OANDA_PRACTICE_ACCOUNT_ID=your_practice_account_id_here
```

Buddy fetches live market data and executes practice trades via the OANDA API.

---

## Usage

### Training Buddy

Train a new model using live OANDA data (default: 5000 candles of USD_JPY M5):

```bash
./Buddy train-buddy
```

**Common training options:**

```bash
# Train with custom instrument and granularity
python main.py train-buddy --instrument EUR_USD --granularity M15 --candles 10000

# Train from local CSV instead of live fetch
python main.py train-buddy --csv market_data/historical_usdjpy.csv

# Enable feature curriculum (progressive feature selection)
python main.py train-buddy --feature-curriculum

# Warm-start from existing checkpoint
python main.py train-buddy --warm-start

# Optimize for speed (shared encoder + mixed precision)
python main.py train-buddy --shared-encoder --mixed-precision --steps-per-execution 10

# Disable Tier-2 calibration for faster training
python main.py train-buddy --no-tier2-calibrate
```

**Training output example:**
```
Feature engineering: done rows=4950 cols=187 in 2.34s
Direction label positive rate (train/val): 50.23% / 49.87%
Tier-2 calibration: 3960 train samples processed (stride=5)
Epoch 1/300: val_direction_accuracy=0.5234 val_combined=0.5421
...
Epoch 45/300: val_direction_accuracy=0.6523 val_combined=0.6891
Early stopping triggered. Best epoch: 35
Model saved: trained_data/models/buddy_tf.keras
```

### Running Inference with Buddy

**Single-shot prediction (dry-run):**
```bash
./Buddy
```

Launches interactive wizard. Select mode 1 for single-shot inference.

**Direct command-line execution:**
```bash
python main.py buddy --instrument USD_JPY --granularity M5 --candles 300 --dry-run
```

**Execute live practice trades:**
```bash
python main.py buddy --instrument EUR_USD --execute --force-execute
```

**Continuous trading loop:**
```bash
python main.py buddy --loop --max-trades 5 --execute
```

**Output example:**
```
Buddy: buy tier2_p_win=0.687 candle=2024-01-03T12:35:00.000000Z
Order sizing: computed_units=2847 final_submitted_units=2847
Order submitted: {'id': '12345', 'units': 2847, 'price': 143.256}
```

### Interactive Buddy REPL

Launch the interactive trading console:

```bash
python main.py buddy --repl
```

**Available REPL commands:**
- `buy` / `sell` - Execute market orders with stop-loss and take-profit
- `close` - Close all positions for the instrument
- `status` - Display current predictions and open positions
- `help` - Show command reference

---

## Advanced Workflows

### Checkpoint Replay and Validation

Replay a saved checkpoint on out-of-sample data to verify reproducibility:

```bash
python scripts/replay_buddy_checkpoint.py \
  --meta trained_data/models/buddy_tf.meta.json \
  --csv market_data/oos_data.csv
```

**Output:**
```
meta: trained_data/models/buddy_tf.meta.json
csv:  market_data/oos_data.csv
val_direction_accuracy: 0.6312
val_avg_confidence: 0.7245
val_combined_score: 0.6591
```

This script is essential for:
- Verifying model performance on unseen data
- Debugging training reproducibility issues
- Evaluating checkpoint quality before deployment

### Buddy Audit Script

Analyze prediction quality and calibration metrics:

```bash
python scripts/buddy_audit.py --checkpoint trained_data/models/buddy_tf.keras
```

### Confidence Calibration Tuning

Fine-tune Tier-2 probability calibration post-training:

```bash
python scripts/calibrate_fx_confidence.py \
  --meta trained_data/models/buddy_tf.meta.json \
  --csv market_data/calibration_data.csv
```

---

## Key Configuration Options

Edit `config_tuned.yaml` to customize Buddy's behavior:

**Training defaults (`buddy.train_defaults`):**
- `seq_len`: Lookback window length (default: 50)
- `epochs`: Maximum training epochs (default: 300)
- `batch_size`: Training batch size (default: 32)
- `es_monitor`: Early stopping metric (choices: `direction`, `combined`, `val_loss`)
- `shared_encoder`: Use single LSTM vs 5 parallel heads (default: false)

**Trading parameters (`buddy`):**
- `stop_loss_pips`: Stop-loss distance in pips (default: 20.0)
- `take_profit_pips`: Take-profit distance in pips (default: 60.0)
- `risk_per_trade_pct`: Risk fraction per trade (default: 0.005 = 0.5%)
- `equity`: Paper trading equity for position sizing (default: 10000.0)

**Tier-2 calibration (`buddy.tier2`):**
- `enabled`: Enable calibrated win probability (default: false)
- `p_win_threshold`: Minimum P(win) to trigger trades (default: 0.60)
- `kelly_fraction`: Fractional Kelly sizing multiplier (default: 0.5)

---

## Platform-Specific Notes

### Apple M1/M2 (TF-Metal)

**Configuration applied automatically:**
- Meta optimizer disabled to prevent Metal plugin crashes
- Memory growth enabled for efficient GPU allocation
- Mixed precision partially supported (use `--mixed-precision` cautiously)

**Environment variables:**
- `BUDDY_DISABLE_META_OPTIMIZER=1` - Force disable TF meta optimizer
- `BUDDY_GPU_MEMORY_LIMIT_MB=4096` - Cap GPU memory allocation
- `TF_CPP_MIN_LOG_LEVEL=3` - Reduce TensorFlow logging noise

### Linux/Windows (CUDA)

Use standard TensorFlow installation:
```bash
pip install tensorflow==2.15.1
```

Buddy detects available GPUs automatically. Override with `--device cpu` for CPU-only training.

---

## Command Reference

### `train-buddy`
Train a new Buddy model.

**Key arguments:**
- `--instrument PAIR` - Trading pair (default: USD_JPY)
- `--granularity G` - Candle timeframe: M5, M15, H1, etc. (default: M5)
- `--candles N` - Number of candles to fetch for training (default: 5000)
- `--csv PATH` - Use local CSV instead of OANDA live fetch
- `--oanda-live` - Force OANDA fetch even when --csv provided
- `--epochs N` - Training epochs (default: 300)
- `--patience N` - Early stopping patience (default: 10)
- `--warm-start` - Initialize from existing checkpoint
- `--feature-curriculum` - Enable progressive feature masking
- `--shared-encoder` - Use single LSTM instead of 5-head architecture
- `--no-tier2-calibrate` - Disable TP/SL calibration (faster training)

### `buddy`
Run inference and optionally execute trades.

**Key arguments:**
- `--instrument PAIR` - Trading pair
- `--granularity G` - Candle timeframe
- `--candles N` - Lookback window size (default: 300)
- `--execute` / `--dry-run` - Enable/disable live order placement
- `--force-execute` - Bypass training accuracy gate
- `--loop` - Continuous trading mode
- `--max-trades N` - Stop after N trades in loop mode
- `--repl` - Launch interactive trading console
- `--equity AMT` - Paper equity for sizing (default: 10000)
- `--risk PCT` - Risk per trade as fraction (default: 0.005)

---

## Troubleshooting

**Issue:** `Conda not found` when running `./Buddy`
- **Solution:** Install Miniforge and activate environment:
  ```bash
  export BUDDY_CONDA_ENV=ml_engine
  conda activate ml_engine
  ```

**Issue:** Training is slow or GPU not detected
- **Solution (M1/M2):** Verify TF-Metal installation:
  ```bash
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  ```
- **Solution (General):** Try `--shared-encoder --steps-per-execution 10` for faster training

**Issue:** OANDA API errors
- **Solution:** Verify `.env` credentials and network connectivity. Check practice account status at OANDA portal.

**Issue:** `FileNotFoundError: Missing Buddy metadata`
- **Solution:** Train a model first: `python main.py train-buddy`

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

See repository license file for terms.