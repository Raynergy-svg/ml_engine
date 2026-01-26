# Training Investigation Report: 20K Candles vs Production

## Key Findings

### 1. **Early Stopping Issue**
- **Best Epoch: 3** (extremely early!)
- **Trained Epochs: 28** (stopped after 25 epochs of no improvement)
- **Patience: 25** (too high for early stopping at epoch 3)
- **Monitor: combined** (direction + confidence)

**Problem**: Model peaked at epoch 3, then degraded. This suggests:
- Learning rate too high (0.0005)
- Model overfitting immediately
- Validation set may have distribution shift with 20K candles

### 2. **Meta-Labeler Overfitting**
- **Train Accuracy: 79.2%**
- **Val Accuracy: 49.9%**
- **Gap: 29.3%** (severe overfitting!)

**Problem**: Meta-labeler memorized training data but failed on validation.

### 3. **Data Quality Issues**
- **20K candles** may include:
  - Older data with different market regimes
  - Data distribution shift
  - More noise than signal

### 4. **Performance Comparison**

| Metric | 20K Model | Production | Difference |
|--------|-----------|------------|------------|
| Direction Accuracy | 51.69% | 54.00% | **-2.31%** |
| Combined Score | 0.5115 | 0.5281 | **-0.0166** |
| Confidence | 0.4678 | 0.5575 | **-0.0897** |

## Root Causes

1. **Too Much Data**: 20K candles includes older, less relevant data
2. **Hyperparameters Not Optimized**: Fixed LR (0.0005) may be too high for larger dataset
3. **Early Overfitting**: Model capacity may be insufficient for 20K samples
4. **Meta-Labeler Issues**: Needs regularization or simpler architecture

## Recommendations

1. **Use 15K candles** (optimal amount from production model)
2. **Auto-tune hyperparameters**:
   - Learning rate: [0.0001, 0.0003, 0.0005, 0.001]
   - Batch size: [64, 128, 256]
   - Patience: [15, 20, 25]
   - Dropout: [0.3, 0.4, 0.5]
3. **Reduce meta-labeler complexity** or disable if overfitting
4. **Use walk-forward validation** to ensure robustness

