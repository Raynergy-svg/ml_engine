#!/usr/bin/env bash
# Coverage runner for the ML engine test suite.
#
# Configuration lives in pyproject.toml under [tool.coverage.*].
#
# Usage:
#   scripts/coverage.sh                  # Run on the curated baseline suite
#   scripts/coverage.sh tests/test_X.py  # Run on a specific file or dir
#   scripts/coverage.sh --all            # Run on the full tests/ tree
#                                        # (warning: requires pandas + full deps)
#   scripts/coverage.sh --html           # Open the HTML report after running
#
# The "baseline suite" is the curated set of tests known to run with only
# stdlib + numpy installed. New zero-coverage tests should be added here.

set -euo pipefail

cd "$(dirname "$0")/.."

# Curated baseline — pure-Python and numpy-only tests added under the
# 2026-05-01 zero-coverage audit. Keep this list in sync with the file
# tree under tests/.
BASELINE_TESTS=(
    tests/test_dynamic_sl_tp.py
    tests/test_dynamic_drawdown.py
    tests/test_entropy_sizer.py
    tests/test_kelly_portfolio.py
    tests/test_model_bandit.py
    tests/test_model_calibration.py
    tests/test_agent_lifecycle.py
    tests/test_affinity_portfolio.py
)

OPEN_HTML=0
RUN_ALL=0
TARGETS=()

for arg in "$@"; do
    case "$arg" in
        --html)
            OPEN_HTML=1
            ;;
        --all)
            RUN_ALL=1
            ;;
        *)
            TARGETS+=("$arg")
            ;;
    esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    if [[ $RUN_ALL -eq 1 ]]; then
        TARGETS=(tests/)
    else
        TARGETS=("${BASELINE_TESTS[@]}")
    fi
fi

echo "▶ Running coverage on: ${TARGETS[*]}"
echo

pytest \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    "${TARGETS[@]}"

echo
echo "✓ HTML report: htmlcov/index.html"
echo "✓ XML report:  coverage.xml"

if [[ $OPEN_HTML -eq 1 ]]; then
    if command -v open >/dev/null 2>&1; then
        open htmlcov/index.html
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open htmlcov/index.html
    fi
fi
