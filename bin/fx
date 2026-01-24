#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

PY="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

# Super-short usage:
#   ./fx                 # EUR_USD M5 dry-run
#   ./fx GBP_USD         # instrument only
#   ./fx GBP_USD M15     # instrument + granularity
#   ./fx GBP_USD M15 -n 400 -r 0.005
#   ./fx GBP_USD M15 -x  # execute (places PRACTICE order)

instrument=""
granularity=""

if [ "${1:-}" != "" ] && [ "${1#-}" = "$1" ]; then
  instrument="$1"
  shift
fi

if [ "${1:-}" != "" ] && [ "${1#-}" = "$1" ]; then
  granularity="$1"
  shift
fi

args=""
if [ -n "$instrument" ]; then
  args="$args -I $instrument"
fi
if [ -n "$granularity" ]; then
  args="$args -g $granularity"
fi

# shellcheck disable=SC2086
exec "$PY" "$ROOT_DIR/main.py" fx $args "$@"
