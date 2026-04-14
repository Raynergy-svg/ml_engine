#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/.claude/litellm.openrouter.yaml"
PORT="${LITELLM_PORT:-4000}"
MODEL_ALIAS="${CLAUDE_QWEN_MODEL_ALIAS:-qwen36-plus}"
LITELLM_BIN="/opt/homebrew/Caskroom/miniforge/base/bin/litellm"

if [[ -f "$ROOT_DIR/.env.local" ]]; then
  set -a
  source "$ROOT_DIR/.env.local"
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set."
  echo "Add it to $ROOT_DIR/.env.local and rerun this script."
  echo "Current default model alias: $MODEL_ALIAS"
  exit 1
fi

if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  "$LITELLM_BIN" --config "$CONFIG_PATH" --port "$PORT" >/tmp/qwen36-litellm.log 2>&1 &

  for _ in {1..20}; do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "LiteLLM did not come up on http://127.0.0.1:${PORT}."
  echo "Check /tmp/qwen36-litellm.log for details."
  exit 1
fi

exec claude --model "$MODEL_ALIAS" "$@"
