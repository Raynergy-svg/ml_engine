# Training Commands for Optimal Performance

This guide provides optimized training commands for the ML Engine's TensorFlow models.

---

## 🚀 Quick Start

**Defaults (so you can omit flags):** `train_visual.py` defaults to `--framework tensorflow` and `--model tft`.

### Basic Multi-Task Training (Synthetic Data)
```bash
python3 train_visual.py --multi-task
```

### Multi-Task Training with Real Data (Recommended)
```bash
python3 train_visual.py --multi-task \
    --data-dir trained_data/data --state-classes 3
```

---

## ⚡ Optimized Commands

### Standard Optimized Training
```bash
python3 train_visual.py --multi-task \
    --data-dir trained_data/data \
    --state-classes 3 \
    --epochs 100 \
    --batch-size 32 \
    --dropout 0.35 \
    --lr 0.0003 \
    --patience 25 \
    --tensorboard
```

Note: For better direction learning (avoid ~0.50 coin-flip), tune `unified_head_loss_weights` in `config.yaml` to give **direction** the highest weight.

### 🔄 Resume Training from Best Checkpoint (RECOMMENDED)
```bash
python3 train_visual.py --framework tensorflow --model tft --multi-task \
    --data-dir trained_data/data \
    --state-classes 3 \
    --epochs 100 \
    --batch-size 32 \
    --dropout 0.35 \
    --lr 0.0003 \
    --patience 25 \
    --resume \
    --tensorboard
```

**How Resume Works:**
1. Loads weights from `trained_data/checkpoints/tensorflow/tf_model_best.keras`
2. Evaluates the loaded model to get the **previous best val_loss**
3. Sets `initial_value_threshold` so checkpoint **only saves if val_loss improves**
4. Uses a **lower learning rate** (1/10th of `--lr`) for fine-tuning by default
5. Your previous 65-epoch progress is preserved - only improvements get saved!

**Override the resume learning rate:**
```bash
python3 train_visual.py ... --resume --resume-lr 0.0001
```

### Maximum Accuracy (~93%+ State Accuracy)
```bash
python3 train_visual.py --framework tensorflow --model tft --multi-task \
    --data-dir trained_data/data \
    --state-classes 3 \
    --epochs 150 \
    --batch-size 32 \
    --dropout 0.35 \
    --lr 0.0003 \
    --patience 30 \
    --ensemble-size 5 \
    --cyclic-lr \
    --tensorboard
```

### Fast Training with TCN (Faster than LSTM)
```bash
python3 train_visual.py --framework tensorflow --model tcn --multi-task \
    --data-dir trained_data/data \
    --state-classes 3 \
    --epochs 100 \
    --batch-size 64 \
    --dropout 0.3 \
    --lr 0.0005 \
    --patience 20 \
    --tensorboard
```

### Ensemble Training (+0.5-1% Accuracy Boost)
```bash
python3 train_visual.py --framework tensorflow --model tft --multi-task \
    --data-dir trained_data/data \
    --state-classes 3 \
    --epochs 100 \
    --batch-size 32 \
    --dropout 0.35 \
    --lr 0.0003 \
    --patience 25 \
    --ensemble-size 3 \
    --cyclic-lr \
    --seed 42 \
    --tensorboard
```

---

## 📊 Parameter Reference

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--framework` | tensorflow | **tensorflow** | ML framework (tensorflow/pytorch) |
| `--model` | tft | **tft** or **tcn** | Model architecture |
| `--multi-task` | off | **on** | Enable 5-head output (price, trend, direction, risk, state) |
| `--state-classes` | 3 | **3** | Market state classes (3 matches PyTorch 93%+ accuracy) |
| `--epochs` | 50 | **100-150** | Training epochs |
| `--batch-size` | 64 | **32** | Smaller batches = better generalization |
| `--dropout` | 0.3 | **0.35** | Regularization strength |
| `--lr` | 0.0003 | **0.0003** | Learning rate |
| `--patience` | 20 | **25-30** | Early stopping patience |
| `--resume` | off | **on** | Load from best checkpoint and continue training |
| `--resume-lr` | 1/10th of `--lr` | varies | Learning rate for resumed training (lower = fine-tune) |
| `--checkpoint` | `trained_data/checkpoints/tensorflow/tf_model_best.keras` | default | Path to checkpoint file |
| `--ensemble-size` | 1 | **3-5** | Number of models for ensemble |
| `--cyclic-lr` | off | **on** | Cosine annealing for snapshot ensemble |
| `--seed` | 42 | 42 | Random seed for reproducibility |
| `--tensorboard` | off | **on** | Auto-launch TensorBoard |

---

## 🎯 Model Selection Guide

| Model | Speed | Memory | Best For |
|-------|-------|--------|----------|
| **tft** | Medium | Higher | Multi-feature forecasting, interpretability |
| **tcn** | Fastest | Low | Real-time inference, long sequences |
| **attention_lstm** | Slower | Higher | Long-range dependencies |
| **transformer** | Fast (GPU) | High | Parallel training |
| **lstm** | Medium | Medium | Baseline, short sequences |

### Recommendation
- **For accuracy**: Use `tft` with ensemble training
- **For speed**: Use `tcn` (parallelizable, faster than LSTM)
- **For production**: Use `tcn` (fastest inference) or `tft` (best accuracy)

---

## 📁 Data Options

### Single Instrument
```bash
python3 train_visual.py --framework tensorflow --model tft --multi-task \
    --data-file trained_data/data/oanda_EUR_USD_M5.csv
```

### Multi-Instrument (Recommended - Better Generalization)
```bash
python3 train_visual.py --framework tensorflow --model tft --multi-task \
    --data-dir trained_data/data
```

Multi-instrument training combines data from multiple FX pairs (~15,000+ samples), significantly improving generalization and directional accuracy.

---

## 📈 TensorBoard Monitoring

After training starts (or completes), view metrics:

```bash
tensorboard --logdir=trained_data/tensorboard
```

Then open: http://localhost:6006

### Key Metrics to Watch
- **loss**: Overall training loss
- **val_loss**: Validation loss (watch for overfitting)
- **direction_dir_acc** (and **val_direction_dir_acc**): Binary up/down direction-head accuracy
- **state_accuracy**: Market regime classification accuracy
- **price_mae**: Price prediction error

Note: Direction is a hard target on noisy FX data. It’s normal to see **~0.50** in the first few epochs.
For balanced labels (~50/50 up/down), the engine now auto-uses **BCE** (instead of focal loss) so the model moves off 0.5 predictions faster.

---

## 🔧 Troubleshooting

### Out of Memory
Reduce batch size:
```bash
--batch-size 16
```

### Overfitting (val_loss increasing)
Increase regularization:
```bash
--dropout 0.4 --patience 15
```

### Slow Convergence
Increase learning rate slightly:
```bash
--lr 0.0005
```

### Underfitting (high loss)
Reduce regularization:
```bash
--dropout 0.2
```

---

## 🏆 Best Practices

1. **Always use `--resume`** to continue from your best checkpoint (builds on previous training)
2. **Always use `--multi-task`** for the 5-head model (price, trend, direction, risk, state)
3. **Use `--data-dir`** with multiple instruments for better generalization
4. **Use ensemble training** (`--ensemble-size 3-5`) for +0.5-1% accuracy
5. **Enable TensorBoard** (`--tensorboard` or `-t`) for monitoring
6. **Use 3 state classes** to match PyTorch 93%+ accuracy targets
7. **Start with smaller batch size** (32) for better generalization
8. **Use cyclic LR** (`--cyclic-lr`) with ensemble for snapshot ensemble benefits

---

## 📋 Complete Example Session

```bash
# 1. Fetch fresh data (if using OANDA)
python main.py train-oanda-unified --config config.yaml \
    --instrument EUR_USD,GBP_USD,USD_JPY --candles 5000 --all-features

# 2. First training run (no checkpoint exists yet)
python3 train_visual.py --framework tensorflow --model tft --multi-task \
    --data-dir trained_data/data \
    --state-classes 3 \
    --epochs 100 \
    --batch-size 32 \
    --dropout 0.35 \
    --lr 0.0003 \
    --patience 25 \
    --tensorboard

# 3. Subsequent training runs - ALWAYS use --resume to build on best checkpoint
python3 train_visual.py --framework tensorflow --model tft --multi-task \
    --data-dir trained_data/data \
    --state-classes 3 \
    --epochs 100 \
    --batch-size 32 \
    --dropout 0.35 \
    --lr 0.0003 \
    --patience 25 \
    --resume \
    --tensorboard

# 4. View TensorBoard (if not auto-launched)
tensorboard --logdir=trained_data/tensorboard
```

---

## 🔄 Checkpoint Management

### Default Checkpoint Location
```
trained_data/checkpoints/tensorflow/tf_model_best.keras
```

### How It Works
1. Each training run saves the **best model** (lowest val_loss) to this file
2. Using `--resume` loads this checkpoint at the start of training
3. Training continues from where the best model left off
4. If the new training improves, it overwrites with the new best model

### Custom Checkpoint Path
```bash
python3 train_visual.py --framework tensorflow --model tft --multi-task \
    --data-dir trained_data/data \
    --resume \
    --checkpoint /path/to/custom/checkpoint.keras
```

---

## 🎯 Target Metrics

With optimal settings, expect:
- **Direction Accuracy**: 55-60%+ (significant for trading)
- **State Accuracy**: 90-93%+ (market regime classification)
- **Price MAE**: Dataset-dependent (lower is better)
- **Risk Correlation**: 0.7+ with actual volatility


python3 train_visual.py --framework tensorflow --model tft --multi-task --fetch-data --ensemble-size 3 --epochs 50