#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="conda"
elif [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniforge3/bin/conda"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi

if [[ -z "${CONDA_BIN}" ]]; then
  echo "Conda not found. Install Miniforge/Miniconda and create the 'tf-metal' env (see environment_tf_metal.yml)." >&2
  exit 1
fi

exec "${CONDA_BIN}" run -n tf-metal --no-capture-output python "${ROOT_DIR}/main.py" train-buddy "$@"
