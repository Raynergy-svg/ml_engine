#!/bin/bash
# Optimized M1 Training Script
# Run this from terminal with tf-metal environment activated

set -e

cd /Users/mirelacertan/Desktop/ml_engine

echo "============================================================"
echo "FX Bot 2025 - Optimized M1 Training"
echo "============================================================"

# Best data file (25,000 candles)
DATA_FILE="market_data/oanda_USD_JPY_M5_live_20251230_221406.csv"

# Check data
echo ""
echo "📊 Data Summary:"
wc -l "$DATA_FILE"
head -2 "$DATA_FILE"

echo ""
echo "🚀 Starting optimized training..."
echo "   Model: TCN (Temporal Convolutional Network)"
echo "   Epochs: 50"
echo "   Batch Size: 128"
echo "   Mixed Precision: ON"
echo "   Steps Per Execution: 20"
echo ""

# Start TensorBoard in background
tensorboard --logdir=trained_data/logs --bind_all --port=6006 &
TB_PID=$!
echo "📈 TensorBoard started at http://localhost:6006 (PID: $TB_PID)"

# Run training
python train_optimized_m1.py \
    --data "$DATA_FILE" \
    --epochs 50 \
    --model tcn

echo ""
echo "✅ Training complete!"
echo ""
echo "📊 View results in TensorBoard: http://localhost:6006"
echo "   - Watch 'direction_binary_accuracy' (target: >52%)"
echo "   - Loss curves should be smooth"
echo "   - Check learning rate schedule"

# Cleanup
kill $TB_PID 2>/dev/null || true

