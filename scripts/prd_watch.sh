#!/bin/bash
# PRD Agent Chain — Event-driven autonomous pipeline
#
# Watches .claude/ralph/prd*.json for completion.
# When a PRD completes (all stories pass), fires the chain:
#
#   1. Gap Wirer → detects unwired modules → generates NEW prd.json
#   2. Ralph     → spawns claude/amp to IMPLEMENT the gap stories
#   3. Reviewer  → spawns claude to REVIEW the code changes
#   4. (if blockers) Ralph → fixes the blockers from review
#
# Usage:
#   ./scripts/prd_watch.sh                          # Full auto: watch → wire → implement → review → fix
#   ./scripts/prd_watch.sh --no-spawn               # Just generate PRDs, don't run Ralph
#   ./scripts/prd_watch.sh --mode manual --prd .claude/ralph/prd_phase27.json
#   ./scripts/prd_watch.sh --ralph-tool amp          # Use amp instead of claude
#   ./scripts/prd_watch.sh --ralph-iters 5           # Max 5 Ralph iterations per PRD
#   ./scripts/prd_watch.sh --no-review               # Skip code review step
#   ./scripts/prd_watch.sh -v                        # Debug logging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

mkdir -p .claude/ralph/reports

echo "═══════════════════════════════════════════════════════"
echo "  PRD Agent Chain"
echo "  Project: $PROJECT_ROOT"
echo ""
echo "  Chain: PRD Complete → Gap PRD → Ralph → Review → Fix"
echo "  Each step spawns real Claude agent instances"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check for watchdog (optional)
if python3 -c "import watchdog" 2>/dev/null; then
    echo "  ✓ watchdog (inotify backend)"
else
    echo "  ⚠ watchdog not installed — polling fallback"
fi

# Check for claude CLI
if command -v claude &>/dev/null; then
    echo "  ✓ claude CLI available"
else
    echo "  ⚠ claude CLI not found — will use static analysis fallback"
fi

echo ""

exec python3 -m src.scanner.automation.prd_agent_chain \
    --project-root "$PROJECT_ROOT" \
    "$@"
