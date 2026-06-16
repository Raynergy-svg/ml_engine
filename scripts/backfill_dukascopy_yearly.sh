#!/usr/bin/env bash
PAIR="${1:-EUR_USD}"
START_YEAR="${2:-2020}"
END_YEAR="${3:-2025}"
LOG="trained_data/dukascopy/backfill_yearly.log"

mkdir -p trained_data/dukascopy/yearly

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Starting yearly backfill for $PAIR $START_YEAR-$END_YEAR" >> "$LOG"

for YEAR in $(seq "$START_YEAR" "$END_YEAR"); do
    OUTDIR="trained_data/dukascopy/yearly/$YEAR"
    mkdir -p "$OUTDIR"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$PAIR] Backfilling year $YEAR ..." >> "$LOG"
    python scripts/backfill_dukascopy.py \
        --pair "$PAIR" \
        --start "${YEAR}-01-01" \
        --end "${YEAR}-12-31" \
        --output "$OUTDIR" \
        --convert-to-harvest >> "$LOG" 2>&1 || true
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$PAIR] Year $YEAR done." >> "$LOG"
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) All years done. Concatenating M15 parquets..." >> "$LOG"
python scripts/merge_dukascopy_yearly.py >> "$LOG" 2>&1 || true

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Yearly backfill finished." >> "$LOG"
