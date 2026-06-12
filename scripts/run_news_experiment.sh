#!/usr/bin/env bash
# Run the news-fused transformer experiment for EUR_USD M15.
#
# Measures whether FinBERT + ForexFactory calendar news pushes val_accuracy
# above the 52% price-only ceiling (dad8624, 2026-06-10).
#
# Runtime: ~15-25 min on M1 (first run includes ~4 min ForexFactory scrape).
# Subsequent runs skip the scrape via parquet cache.
#
# Usage:
#   cd ~/Documents/ml_engine
#   bash scripts/run_news_experiment.sh
#
# Output to watch:
#   RESULT:{"val_acc":..., "gap":..., "news_lift_pp":..., "quarantined":...}
#
# Ship criteria: val_acc >= 0.55 AND quarantined=false
#   → bot can be unhalted after backtest passes.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "================================================================"
echo "  News-Fused Transformer Experiment — EUR_USD M15"
echo "  $(date)"
echo "================================================================"

# Activate conda/venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python scripts/train_news_experiment_eur_usd.py 2>&1 | tee logs/news_experiment_$(date +%Y%m%dT%H%M%S).log

echo ""
echo "Log saved. Check the RESULT: line above for val_acc, gap, news_lift_pp."
