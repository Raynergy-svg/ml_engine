#!/bin/bash
# =============================================================================
# Run Training Diagnostics for Buddy FX Model
# =============================================================================
# Usage:
#   ./run_diagnostics.sh                    # Use default CSV
#   ./run_diagnostics.sh path/to/data.csv   # Use custom CSV
#   ./run_diagnostics.sh --help             # Show help
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default CSV path
DEFAULT_CSV="market_data/USD_JPY_M5.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Buddy FX Training Diagnostics${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
CSV_PATH="${1:-$DEFAULT_CSV}"

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [CSV_PATH] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  CSV_PATH      Path to training CSV (default: $DEFAULT_CSV)"
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with default CSV"
    echo "  $0 market_data/EUR_USD_H1.csv         # Run with custom CSV"
    echo ""
    exit 0
fi

# Check if CSV exists
if [[ ! -f "$CSV_PATH" ]]; then
    echo -e "${RED}Error: CSV file not found: $CSV_PATH${NC}"
    echo ""
    echo "Available CSV files in market_data/:"
    ls -la market_data/*.csv 2>/dev/null | head -20 || echo "  No CSV files found"
    echo ""
    exit 1
fi

# Detect instrument from filename
INSTRUMENT="USD_JPY"
if [[ "$CSV_PATH" == *"EUR_USD"* ]]; then
    INSTRUMENT="EUR_USD"
elif [[ "$CSV_PATH" == *"GBP_USD"* ]]; then
    INSTRUMENT="GBP_USD"
elif [[ "$CSV_PATH" == *"USD_JPY"* ]]; then
    INSTRUMENT="USD_JPY"
fi

echo -e "${GREEN}CSV:${NC} $CSV_PATH"
echo -e "${GREEN}Instrument:${NC} $INSTRUMENT"
echo ""

# Create output directory
OUTPUT_DIR="trained_data/diagnostics"
mkdir -p "$OUTPUT_DIR"

# Generate output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/diagnostic_report_${TIMESTAMP}.json"

# Run diagnostics
echo -e "${YELLOW}Running diagnostics...${NC}"
echo ""

python3 diagnose_training.py \
    --csv "$CSV_PATH" \
    --instrument "$INSTRUMENT" \
    --seq-len 60 \
    --output "$OUTPUT_FILE" \
    --tensorboard "trained_data/tensorboard" \
    --training-log "cli.log"

EXIT_CODE=$?

echo ""
echo -e "${BLUE}========================================${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ Diagnostics completed: No critical issues${NC}"
elif [[ $EXIT_CODE -eq 1 ]]; then
    echo -e "${YELLOW}⚠ Diagnostics completed: Warnings found${NC}"
else
    echo -e "${RED}✗ Diagnostics completed: Critical issues found${NC}"
fi

echo -e "${BLUE}Report saved to:${NC} $OUTPUT_FILE"
echo -e "${BLUE}========================================${NC}"
echo ""

# Quick summary of next steps
echo -e "${GREEN}Next Steps:${NC}"
echo "  1. Review the recommendations above"
echo "  2. Use the improved config: config_improved.yaml"
echo "  3. Re-run training with:"
echo "     python main.py train-buddy --config config_improved.yaml"
echo ""

exit $EXIT_CODE

