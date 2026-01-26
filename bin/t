#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="conda"
elif [ -x "${HOME}/miniforge3/bin/conda" ]; then
  CONDA_BIN="${HOME}/miniforge3/bin/conda"
elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi

if [ -n "$CONDA_BIN" ] && [ -d "${HOME}/miniforge3/envs/tf-metal" ]; then
  exec "$CONDA_BIN" run -n tf-metal --no-capture-output python "$ROOT_DIR/main.py" predict "$@"
fi

PY="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

# Default to the unified predictor.
exec "$PY" "$ROOT_DIR/main.py" predict "$@"
