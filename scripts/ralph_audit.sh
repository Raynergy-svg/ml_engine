#!/bin/bash
# Ralph completion-discipline audit
#
# For every story marked passes=true in prd.json, independently verify:
#   1. Ralph logged a "Marked story X as complete" line in progress.txt
#   2. A per-story output log exists in .claude/ralph/story_outputs/
#   3. That log contains direct evidence of validation: test pass lines,
#      explicit VALIDATION: PASS markers, or successful pytest/jq/grep output.
#
# Flags false positives: stories closed without captured validation evidence.
#
# Usage: ./scripts/ralph_audit.sh [--verbose]
#
# Exit code: 0 if no false positives, 1 if any detected.

set -eu

VERBOSE=0
[ "${1:-}" = "--verbose" ] && VERBOSE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RALPH_DIR="$PROJECT_ROOT/.claude/ralph"
PRD_FILE="$RALPH_DIR/prd.json"
PROGRESS_FILE="$RALPH_DIR/progress.txt"
STORY_OUTPUT_DIR="$RALPH_DIR/story_outputs"

if [ ! -f "$PRD_FILE" ]; then
  echo "FATAL: PRD file not found at $PRD_FILE" >&2
  exit 2
fi

echo "======================================================================="
echo "  Ralph Completion-Discipline Audit"
echo "  PRD: $PRD_FILE"
echo "  Phase: $(jq -r '.phase // "unknown"' "$PRD_FILE") — $(jq -r '.title // "unknown"' "$PRD_FILE" | head -c 60)"
echo "======================================================================="

TOTAL=$(jq -r '.userStories | length' "$PRD_FILE")
COMPLETED=$(jq -r '[.userStories[] | select(.passes==true)] | length' "$PRD_FILE")
PENDING=$(jq -r '[.userStories[] | select(.passes==false)] | length' "$PRD_FILE")

echo "Total: $TOTAL  Completed: $COMPLETED  Pending: $PENDING"
echo ""

if [ "$COMPLETED" -eq 0 ]; then
  echo "No completed stories yet — nothing to audit."
  exit 0
fi

COMPLETED_IDS=$(jq -r '.userStories[] | select(.passes==true) | .id' "$PRD_FILE")
FP_COUNT=0
FP_LIST=""

for ID in $COMPLETED_IDS; do
  # Check 1: progress.txt Mark line
  MARKED=$(grep -c "Marked story $ID as complete" "$PROGRESS_FILE" 2>/dev/null || true)

  # Check 2: per-story log exists
  LOG_COUNT=$(ls "$STORY_OUTPUT_DIR/${ID}_iter"*.log 2>/dev/null | wc -l | tr -d ' ')

  # Check 3: validation evidence in per-story log
  # Accepted evidence patterns (pytest/unittest/jq/grep/custom "VALIDATION: PASS")
  EVIDENCE=0
  if [ "$LOG_COUNT" -gt 0 ]; then
    for LOG in "$STORY_OUTPUT_DIR/${ID}_iter"*.log; do
      [ -f "$LOG" ] || continue
      # Patterns that indicate validation actually ran and passed:
      #   - pytest "X passed" summary
      #   - "VALIDATION: PASS" explicit marker
      #   - "OK" on its own line (unittest)
      #   - "exit 0" markers
      MATCHES=$(grep -c -E \
        "([0-9]+ passed|VALIDATION: PASS|^OK$|PASS\b|exit code: 0|passed in [0-9])" \
        "$LOG" 2>/dev/null || true)
      EVIDENCE=$((EVIDENCE + MATCHES))
    done
  fi

  # Check 4: completion signal in per-story log
  SIGNAL_FOUND=0
  if [ "$LOG_COUNT" -gt 0 ]; then
    for LOG in "$STORY_OUTPUT_DIR/${ID}_iter"*.log; do
      [ -f "$LOG" ] || continue
      if grep -q "<promise>COMPLETE_STORY_${ID}</promise>" "$LOG"; then
        SIGNAL_FOUND=1
      fi
    done
  fi

  # Classify
  STATUS="OK"
  REASON=""
  if [ "$MARKED" -eq 0 ]; then
    STATUS="INCONSISTENT"
    REASON="prd.json says passes=true but no Mark line in progress.txt"
  elif [ "$LOG_COUNT" -eq 0 ]; then
    STATUS="FP_CANDIDATE"
    REASON="no per-story log in $STORY_OUTPUT_DIR — completed before capture patch? or log wiped"
  elif [ "$SIGNAL_FOUND" -eq 0 ]; then
    STATUS="FP_CANDIDATE"
    REASON="no <promise>COMPLETE_STORY_${ID}</promise> signal in captured log"
  elif [ "$EVIDENCE" -eq 0 ]; then
    STATUS="FP_CANDIDATE"
    REASON="completion signal present but NO validation evidence (no 'X passed', 'VALIDATION: PASS', 'OK', or exit-0 markers)"
  fi

  if [ "$STATUS" != "OK" ]; then
    FP_COUNT=$((FP_COUNT + 1))
    FP_LIST="$FP_LIST $ID"
  fi

  printf "  %-6s %-15s  marked=%d logs=%d signal=%d evidence=%d\n" \
    "$ID" "$STATUS" "$MARKED" "$LOG_COUNT" "$SIGNAL_FOUND" "$EVIDENCE"
  if [ "$STATUS" != "OK" ]; then
    printf "         reason: %s\n" "$REASON"
  fi
  if [ "$VERBOSE" -eq 1 ] && [ "$LOG_COUNT" -gt 0 ]; then
    echo "         logs:"
    for LOG in "$STORY_OUTPUT_DIR/${ID}_iter"*.log; do
      [ -f "$LOG" ] || continue
      printf "           %s (%d bytes)\n" "$LOG" "$(wc -c < "$LOG" | tr -d ' ')"
    done
  fi
done

echo ""
echo "======================================================================="
echo "  Summary"
echo "======================================================================="
echo "  Stories marked complete:     $COMPLETED"
echo "  False-positive candidates:   $FP_COUNT"
if [ "$FP_COUNT" -gt 0 ]; then
  echo "  FP IDs:                     $FP_LIST"
  echo ""
  echo "  ACTION: Re-run validation_commands out-of-band for each FP candidate."
  echo "  Pull the exact commands per story with:"
  echo "    jq -r '.userStories[] | select(.id==\"US-XXX\") | .validation_commands[]' $PRD_FILE"
  exit 1
fi

echo "  All completed stories have captured validation evidence."
exit 0
