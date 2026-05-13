#!/usr/bin/env bash
# Pair expansion at M15 — CLAUDE.md modernization priority #2.
# Trains USD_JPY, USD_CHF, AUD_USD, USD_CAD, NZD_USD as per-pair masters
# at M15 / 65000 candles. Wandb logging auto-enabled via
# BUDDY_WANDB_REGISTRY_ENABLED=1.
#
# Launched: 2026-05-12 via "B using wandb" operator directive.
# Log: trained_data/logs/m15_pair_expansion.log

set -uo pipefail  # no -e: continue on per-pair failure
cd "$(dirname "$0")/.."

LOG=trained_data/logs/m15_pair_expansion.log
mkdir -p trained_data/logs

PAIRS=(USD_JPY USD_CHF AUD_USD USD_CAD NZD_USD)

{
  echo "════════════════════════════════════════════════════════════════"
  echo "M15 PAIR EXPANSION — start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pairs: ${PAIRS[*]}"
  echo "granularity: M15  candles: 65000  model: all"
  echo "wandb: BUDDY_WANDB_REGISTRY_ENABLED=${BUDDY_WANDB_REGISTRY_ENABLED:-unset}"
  echo "════════════════════════════════════════════════════════════════"
} >> "$LOG"

for pair in "${PAIRS[@]}"; do
  start=$(date +%s)
  {
    echo
    echo "── START $pair @ $(date -u +%H:%M:%SZ) ─────────────────────────"
  } >> "$LOG"

  python scripts/train_single_model_m1.py \
    --instrument "$pair" \
    --model all \
    --candles 65000 \
    --granularity M15 \
    >> "$LOG" 2>&1
  rc=$?

  elapsed=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    echo "── END   $pair OK   ${elapsed}s ────────────────────────────────" >> "$LOG"
  else
    echo "── END   $pair FAIL rc=$rc ${elapsed}s ─────────────────────────" >> "$LOG"
  fi
done

{
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "M15 PAIR EXPANSION — done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "verify: ls trained_data/models/{USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD}/transformer_direction.keras"
  echo "════════════════════════════════════════════════════════════════"
} >> "$LOG"
