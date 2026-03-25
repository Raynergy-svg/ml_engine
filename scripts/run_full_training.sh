#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Buddy ML Engine — Full Training Pipeline (M1 Mac with tf-metal)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 5-Tier Training Architecture:
#   Tier 1: Core Ensemble (8 models × 5 master pairs)
#   Tier 2: Transfer Learning (master → target pairs per correlation group)
#   Tier 3: RL Suite (PPO Position Sizer + SAC Gates + PPO Exits + Reward Model)
#   Tier 4: Meta-Labeler (XGBoost trade signal filter)
#   Tier 5: Confidence Calibrator (Platt scaling)
#   Final:  Validation Report (PASS/FAIL per model, <6% gap enforcement)
#
# Usage:
#   chmod +x scripts/run_full_training.sh
#   ./scripts/run_full_training.sh                    # Full pipeline
#   ./scripts/run_full_training.sh --tier 1           # Only core ensemble
#   ./scripts/run_full_training.sh --tier 2           # Only transfer learning
#   ./scripts/run_full_training.sh --tier 3           # Only RL suite
#   ./scripts/run_full_training.sh --pair GBP_JPY     # Only one master pair
#   ./scripts/run_full_training.sh --skip-tier 3      # Skip RL
#
# Requirements:
#   pip install tensorflow-macos tensorflow-metal
#   pip install lightgbm xgboost scikit-learn pandas numpy
#   pip install stable-baselines3 gymnasium  # For RL suite
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
CANDLES="${CANDLES:-25000}"
GRANULARITY="${GRANULARITY:-H1}"
RL_TIMESTEPS="${RL_TIMESTEPS:-50000}"

# Master pairs for correlation groups
MASTER_PAIRS=("GBP_JPY" "EUR_GBP" "AUD_USD" "USD_CAD" "AUD_NZD")

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Parse arguments
TIER_FILTER=""
PAIR_FILTER=""
SKIP_TIERS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier)
            TIER_FILTER="$2"
            shift 2
            ;;
        --pair)
            PAIR_FILTER="$2"
            shift 2
            ;;
        --skip-tier)
            SKIP_TIERS+=("$2")
            shift 2
            ;;
        --candles)
            CANDLES="$2"
            shift 2
            ;;
        --rl-timesteps)
            RL_TIMESTEPS="$2"
            shift 2
            ;;
        --help|-h)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

should_run_tier() {
    local tier=$1
    # If tier filter set, only run that tier
    if [[ -n "$TIER_FILTER" ]] && [[ "$TIER_FILTER" != "$tier" ]]; then
        return 1
    fi
    # If tier is in skip list, skip it
    for skip in "${SKIP_TIERS[@]+"${SKIP_TIERS[@]}"}"; do
        if [[ "$skip" == "$tier" ]]; then
            return 1
        fi
    done
    return 0
}

log_header() {
    echo ""
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

log_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

TOTAL_START=$(date +%s)

log_header "BUDDY ML ENGINE — FULL TRAINING PIPELINE"
echo "  Candles:      ${CANDLES} per pair"
echo "  Granularity:  ${GRANULARITY}"
echo "  RL Timesteps: ${RL_TIMESTEPS}"
echo "  Python:       $($PYTHON --version 2>&1)"
echo "  Project:      ${PROJECT_ROOT}"
echo ""

# ── Verify tf-metal ──────────────────────────────────────────────────────────
log_step "Checking tf-metal GPU..."
$PYTHON -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  Metal GPU: {gpus[0]}')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
else:
    print('  WARNING: No Metal GPU detected — will use CPU')
" 2>/dev/null || echo "  WARNING: TensorFlow check failed"

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: Core Ensemble (8 models × 5 master pairs)
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_tier "1"; then
    log_header "TIER 1: Core Ensemble Training"

    pairs_to_train=("${MASTER_PAIRS[@]}")
    if [[ -n "$PAIR_FILTER" ]]; then
        pairs_to_train=("$PAIR_FILTER")
    fi

    for pair in "${pairs_to_train[@]}"; do
        PAIR_START=$(date +%s)
        log_step "Training all 8 models for ${pair}..."

        $PYTHON "${SCRIPT_DIR}/train_single_model_m1.py" \
            --instrument "$pair" \
            --model all \
            --candles "$CANDLES" \
            --granularity "$GRANULARITY" \
            2>&1 | tee "${PROJECT_ROOT}/trained_data/logs/tier1_${pair}.log" || {
                log_error "Training failed for ${pair}"
                # Continue to next pair instead of exiting
                continue
            }

        PAIR_END=$(date +%s)
        PAIR_ELAPSED=$((PAIR_END - PAIR_START))
        log_success "${pair} complete — ${PAIR_ELAPSED}s"
    done

    log_success "Tier 1 complete!"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: Transfer Learning (master → target pairs)
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_tier "2"; then
    log_header "TIER 2: Transfer Learning"

    TRANSFER_ARGS="--candles $CANDLES --granularity $GRANULARITY"
    if [[ -n "$PAIR_FILTER" ]]; then
        # Find the group for this pair
        TRANSFER_ARGS="--master $PAIR_FILTER $TRANSFER_ARGS"
    fi

    log_step "Transferring knowledge from masters to target pairs..."

    $PYTHON "${SCRIPT_DIR}/train_transfer_learning.py" \
        $TRANSFER_ARGS \
        2>&1 | tee "${PROJECT_ROOT}/trained_data/logs/tier2_transfer.log" || {
            log_error "Transfer learning had errors (see log)"
        }

    log_success "Tier 2 complete!"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: RL Suite (PPO + SAC)
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_tier "3"; then
    log_header "TIER 3: RL Suite"

    log_step "Training RL components (position sizer, gate thresholds, exits, reward model)..."

    $PYTHON "${SCRIPT_DIR}/train_rl_suite.py" \
        --timesteps "$RL_TIMESTEPS" \
        --candles "$CANDLES" \
        2>&1 | tee "${PROJECT_ROOT}/trained_data/logs/tier3_rl.log" || {
            log_error "RL suite had errors (see log)"
        }

    log_success "Tier 3 complete!"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4+5: Meta-Labeler + Confidence Calibrator
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_tier "4" || should_run_tier "5"; then
    log_header "TIER 4+5: Meta-Labeler + Confidence Calibrator"

    META_ARGS=""
    if ! should_run_tier "4"; then
        META_ARGS="--skip-meta"
    fi
    if ! should_run_tier "5"; then
        META_ARGS="$META_ARGS --skip-calibrator"
    fi

    $PYTHON "${SCRIPT_DIR}/train_meta_calibrator.py" \
        $META_ARGS \
        2>&1 | tee "${PROJECT_ROOT}/trained_data/logs/tier4_5_meta.log" || {
            log_error "Meta/calibrator had errors (see log)"
        }

    log_success "Tier 4+5 complete!"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL: Validation Report
# ═══════════════════════════════════════════════════════════════════════════════
log_header "FINAL: Validation Report"

$PYTHON "${SCRIPT_DIR}/generate_validation_report.py" \
    --output "${PROJECT_ROOT}/trained_data/validation_report.json" || {
        log_error "Validation report generation failed"
    }

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo ""
log_header "PIPELINE COMPLETE"
echo -e "  Total time: ${BOLD}${TOTAL_MIN}m ${TOTAL_SEC}s${NC}"
echo -e "  Report:     ${PROJECT_ROOT}/trained_data/validation_report.json"
echo -e "  Logs:       ${PROJECT_ROOT}/trained_data/logs/"
echo ""
