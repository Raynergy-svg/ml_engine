#!/usr/bin/env bash
# Gap Hunt — four experiments to find a shippable direction model.
#
# Rationale: We've proven the Transformer can't generalize on M15. That's ONE
# architecture. These experiments test different architectures (HistGB) and
# timeframes (Daily) where evidence suggests the gap may close.
#
# Evidence-based ordering (highest expected value first):
#
#  Exp 1 — HistGB on AUD_USD Daily
#    AUD_USD D already showed val=59.34% with Transformer (above 55% ship threshold)
#    but gap=17.33% (above 10% hard gate). HistGB regularizes differently from
#    neural nets — the gap may collapse while signal stays above 55%. This is
#    the single most likely path to a shippable model today.
#
#  Exp 2 — HistGB on EUR_USD M15 (PENDING from docs/experiments-log.md 2026-05-18)
#    Pending experiment blocked on disk space. New defaults (max_iter=100,
#    max_depth=4, lr=0.03, l2=1.0) have NEVER been measured. Old defaults
#    produced gap=47% — the new shrunk defaults are designed to collapse this.
#
#  Exp 3 — Transformer Daily on GBP_USD
#    strategy.md claims "GBP_USD/USD_JPY daily ~62%" but there is NO
#    training_status.json to back this up. Verify the claim.
#
#  Exp 4 — Transformer Daily on USD_JPY
#    Same unverified 62% claim. Verify independently.
#
# Decision rules (same as experiments-log.md):
#   gap < 0.10 AND val_acc >= 0.55 → SHIPPABLE (unhalt after backtest passes)
#   gap < 0.10 AND val_acc 0.51-0.54 → Gap closed, but no signal. HistGB baseline only.
#   gap > 0.10 → Still overfitting. Try --model histgb_direction on the daily pairs.
#
# Usage:
#   cd ~/Documents/ml_engine
#   bash scripts/run_gap_hunt.sh 2>&1 | tee logs/gap_hunt_$(date +%Y%m%dT%H%M%S).log
#
# Runtime per experiment: ~2-10 min on M1 Metal (daily much faster than M15/65k).
# Watch for: RESULT: lines. Gap < 0.10 + val_acc >= 0.55 = ship it.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_TS=$(date +%Y%m%dT%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then source venv/bin/activate
fi

run_exp() {
    local name="$1"; shift
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: $name"
    echo "  $(date)"
    echo "  Command: python $*"
    echo "================================================================"
    python "$@"
    echo "  ↑ check the RESULT: line above — gap < 0.10 AND val_acc >= 0.55 = SHIPPABLE"
}

# ─── Exp 1: HistGB on AUD_USD Daily ─────────────────────────────────────────
# AUD_USD D transformer produced val=59.34% / gap=17.33% on 2026-05-27.
# Signal is there; the gap is the blocker. HistGB may close it.
run_exp "HistGB AUD_USD Daily" \
    scripts/train_single_model_m1.py \
    --pair AUD_USD \
    --model histgb_direction \
    --granularity D \
    --candles 6000

# ─── Exp 2: HistGB on EUR_USD M15 (PENDING from experiments-log.md) ─────────
# New tighter defaults (never measured). Old gap=47% was with old defaults.
run_exp "HistGB EUR_USD M15 (pending exp from 2026-05-18)" \
    scripts/train_single_model_m1.py \
    --pair EUR_USD \
    --model histgb_direction \
    --granularity M15 \
    --candles 65000

# ─── Exp 3: Transformer Daily on GBP_USD (verify ~62% claim) ────────────────
# strategy.md claims ~62% but NO training_status.json backs this up.
run_exp "Transformer GBP_USD Daily (verify ~62% claim)" \
    scripts/train_single_model_m1.py \
    --pair GBP_USD \
    --model transformer \
    --granularity D \
    --candles 6000

# ─── Exp 4: Transformer Daily on USD_JPY (verify ~62% claim) ────────────────
run_exp "Transformer USD_JPY Daily (verify ~62% claim)" \
    scripts/train_single_model_m1.py \
    --pair USD_JPY \
    --model transformer \
    --granularity D \
    --candles 6000

echo ""
echo "================================================================"
echo "  GAP HUNT COMPLETE — $(date)"
echo "  Search for: RESULT: lines above"
echo "  Ship criteria: val_acc >= 0.55 AND quarantined=false"
echo "================================================================"
