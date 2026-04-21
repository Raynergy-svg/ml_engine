#!/usr/bin/env bash
# ensure_buddy_tmux.sh — idempotently create the "buddy" tmux session with Layout A.
#
# Runs on every Claude Code SessionStart (wired in .claude/settings.json).
# Silent no-op if:
#   • tmux isn't installed
#   • the session already exists
#
# Otherwise creates a detached session with four named panes:
#   tui · scanner · logs · shell
#
# Does NOT run the startup commands (those live in ~/.local/bin/buddy-start).
# Just guarantees the named panes exist so Claude can peek/send to them
# the moment you ask. You still `tmux attach -t buddy` to see them yourself.

set -euo pipefail

command -v tmux >/dev/null || exit 0

SESSION=buddy
REPO=/Users/buddy/Documents/ml_engine

if tmux has-session -t "$SESSION" 2>/dev/null; then
  exit 0
fi

tmux new-session  -d -s "$SESSION" -n main -c "$REPO"
tmux select-pane  -t "$SESSION:main.1" -T "tui"
tmux split-window -t "$SESSION:main" -h -c "$REPO" ; tmux select-pane -T "scanner"
tmux split-window -t "$SESSION:main" -v -c "$REPO" ; tmux select-pane -T "logs"
tmux select-pane  -t "$SESSION:main.1"
tmux split-window -t "$SESSION:main" -v -c "$REPO" ; tmux select-pane -T "shell"
tmux select-layout -t "$SESSION:main" tiled >/dev/null

echo "[buddy-tmux] created session '$SESSION' with panes: tui, scanner, logs, shell" >&2
