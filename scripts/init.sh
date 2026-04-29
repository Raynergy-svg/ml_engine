#!/usr/bin/env bash
# Buddy environment bootstrap — sourced by ./buddy.
#
# Single source of truth for runtime env vars. The Python equivalent is
# src/bootstrap/env.py — keep both in lockstep by always calling the
# Python path inside this script (see "DUMP_PYTHON_DEFAULTS" below).
#
# Usage (from buddy launcher):
#   source "$SCRIPT_DIR/scripts/init.sh"

# Don't `set -e` here — we're being sourced, and a non-fatal env probe
# shouldn't abort the parent script.

# Resolve project root — works whether script is sourced from buddy/
# or invoked directly.
__BUDDY_INIT_SCRIPT="${BASH_SOURCE[0]:-$0}"
__BUDDY_SCRIPT_DIR="$(cd "$(dirname "$__BUDDY_INIT_SCRIPT")" && pwd)"
__BUDDY_PROJECT_ROOT="$(cd "$__BUDDY_SCRIPT_DIR/.." && pwd)"

# 1. .env.local — secrets (OANDA_*, etc.). Existing convention.
if [ -f "$__BUDDY_PROJECT_ROOT/.env.local" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$__BUDDY_PROJECT_ROOT/.env.local"
    set +a
fi

# 2. .env.local.toggles — class-C BUDDY_* behavior toggles. NEW.
if [ -f "$__BUDDY_PROJECT_ROOT/.env.local.toggles" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$__BUDDY_PROJECT_ROOT/.env.local.toggles"
    set +a
fi

# 3. Activate venv if present (preserves existing buddy launcher behavior).
if [ -f "$__BUDDY_PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$__BUDDY_PROJECT_ROOT/.venv/bin/activate"
fi

# 4. Defer to Python for the rest — keeps macOS thread caps and
#    behavior defaults in lockstep with src/bootstrap/env.py. The
#    Python emits `export KEY=value` lines ONLY for vars not already
#    set by the operator's shell or .env.local files.
__BUDDY_PY_DEFAULTS="$(
    cd "$__BUDDY_PROJECT_ROOT" && python -c 'from src.bootstrap.env import dump_shell_exports; dump_shell_exports()' 2>/dev/null
)"
if [ -n "$__BUDDY_PY_DEFAULTS" ]; then
    eval "$__BUDDY_PY_DEFAULTS"
fi

# 5. Stderr-log rotation — preserved from prior buddy launcher.
mkdir -p "$__BUDDY_PROJECT_ROOT/logs"
if [ -f "${BUDDY_TUI_STDERR_LOG:-logs/buddy_tui.stderr.log}" ]; then
    LOG_BYTES=$(
        stat -f%z "$BUDDY_TUI_STDERR_LOG" 2>/dev/null \
            || stat -c%s "$BUDDY_TUI_STDERR_LOG" 2>/dev/null \
            || echo 0
    )
    if [ "${LOG_BYTES:-0}" -gt 2097152 ]; then
        mv "$BUDDY_TUI_STDERR_LOG" "${BUDDY_TUI_STDERR_LOG}.1"
    fi
fi

if [ "${BUDDY_DEBUG:-}" = "1" ]; then
    printf "  \033[1;33m▸ BUDDY_DEBUG=1 — stderr stays on terminal (no redirect)\033[0m\n"
fi

# 6. Validate live-mode prerequisites. If first arg is --live and no
#    OANDA token is set, fail-loud with a friendly message rather than
#    letting the TUI fail mid-startup.
for __arg in "$@"; do
    case "$__arg" in
        --live)
            if [ -z "${OANDA_API_TOKEN:-}${OANDA_API_KEY:-}" ]; then
                printf "  \033[1;31m✗ --live requested but OANDA_API_TOKEN is not set.\033[0m\n" >&2
                printf "    Add it to %s/.env.local and re-run.\n" "$__BUDDY_PROJECT_ROOT" >&2
                return 1 2>/dev/null || exit 1
            fi
            ;;
    esac
done

unset __BUDDY_INIT_SCRIPT __BUDDY_SCRIPT_DIR __BUDDY_PROJECT_ROOT __BUDDY_PY_DEFAULTS
