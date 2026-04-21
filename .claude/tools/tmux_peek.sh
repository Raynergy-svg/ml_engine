#!/usr/bin/env bash
# tmux_peek.sh — read-only pane capture for Claude.
#
# Claude uses a stable command pattern here so the permission prompt appears
# once and then this becomes a pre-approved read. Does NOT send keys.
#
# Usage:
#   tmux_peek.sh <session> <pane-name> [lines=200]
#   tmux_peek.sh --list                       list all sessions:panes
set -euo pipefail

command -v tmux >/dev/null || { echo "tmux not installed"; exit 1; }

if [[ "${1:-}" == "--list" ]]; then
  tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index}  #{pane_title}  [#{pane_current_command}]' 2>/dev/null || echo "(no sessions)"
  exit 0
fi

session=${1:?usage: tmux_peek.sh <session> <pane-name> [lines]}
name=${2:?usage: tmux_peek.sh <session> <pane-name> [lines]}
lines=${3:-200}

target=$(tmux list-panes -s -t "$session" -F '#{session_name}:#{window_index}.#{pane_index} #{pane_title}' 2>/dev/null \
  | awk -v n="$name" '$2==n {print $1; exit}')

if [[ -z "${target:-}" ]]; then
  echo "pane '$name' not found in session '$session'" >&2
  echo "available panes:" >&2
  tmux list-panes -a -F '  #{session_name}:#{window_index}.#{pane_index}  #{pane_title}' >&2
  exit 1
fi

tmux capture-pane -p -t "$target" -S "-$lines"
