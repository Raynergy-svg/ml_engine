#!/usr/bin/env bash
# Map the bot's mutable state onto ONE persistent volume so a container restart never
# loses it. The code uses fixed repo-relative paths (.claude/, trained_data/); we point
# those at $STATE_DIR (the mounted volume) via symlinks — non-invasive, no code change.
#
# $STATE_DIR defaults to /data (Fly volume mount) and is overridable (compose sets it to
# the same named-volume mount). Works identically for compose and Fly.
set -euo pipefail

STATE_DIR="${STATE_DIR:-/data}"
APP_DIR="${APP_DIR:-/app}"

mkdir -p "$STATE_DIR/.claude" "$STATE_DIR/trained_data"

# Symlink repo-relative state dirs to the volume (idempotent). If a real dir exists in
# the image (shouldn't — .dockerignore excludes them), move its contents over once.
link_state() {
  local name="$1"
  local target="$STATE_DIR/$name"
  local link="$APP_DIR/$name"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    cp -an "$link/." "$target/" 2>/dev/null || true
    rm -rf "$link"
  fi
  ln -sfn "$target" "$link"
}
link_state ".claude"
link_state "trained_data"

# Practice pin is immutable — refuse to start on anything else (defense in depth; the
# app re-asserts this too). The cloud bot is as practice-locked as the local one.
if [ "${OANDA_ENVIRONMENT:-practice}" != "practice" ]; then
  echo "FATAL: OANDA_ENVIRONMENT=${OANDA_ENVIRONMENT} — AXIOM is PRACTICE-ONLY. Refusing to start." >&2
  exit 1
fi

exec "$@"
