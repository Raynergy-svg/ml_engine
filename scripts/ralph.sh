#!/bin/bash
# Ralph Wiggum - Long-running AI agent loop with PRD-aware story processing
# Usage: ./ralph.sh [--tool amp|claude] [max_iterations]
#
# Reads prd.json, extracts pending stories (passes=false), sends each to Claude
# one at a time with proper implementation prompts, marks stories complete when
# Claude emits the <promise>COMPLETE_STORY_{id}</promise> signal, and exits only
# when all stories pass.

set -e

# Parse arguments
TOOL="claude"  # Default to claude (non-interactive, --dangerously-skip-permissions)
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case $1 in
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --tool=*)
      TOOL="${1#*=}"
      shift
      ;;
    *)
      # Assume it's max_iterations if it's a number
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

# Validate tool choice
if [[ "$TOOL" != "amp" && "$TOOL" != "claude" ]]; then
  echo "Error: Invalid tool '$TOOL'. Must be 'amp' or 'claude'."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RALPH_DIR="$PROJECT_ROOT/.claude/ralph"
PRD_FILE="$RALPH_DIR/prd.json"
PROGRESS_FILE="$RALPH_DIR/progress.txt"
ARCHIVE_DIR="$RALPH_DIR/archive"
LAST_BRANCH_FILE="$RALPH_DIR/.last-branch"

# Trap to ensure temp files are cleaned up on exit
trap 'rm -f /tmp/ralph_prompt_*.md 2>/dev/null' EXIT

# Helper: Get count of pending stories (passes=false)
get_pending_count() {
  jq -r '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE" 2>/dev/null || echo "0"
}

# Helper: Get count of completed stories (passes=true)
get_completed_count() {
  jq -r '[.userStories[] | select(.passes == true)] | length' "$PRD_FILE" 2>/dev/null || echo "0"
}

# Helper: Get total count of stories
get_total_count() {
  jq -r '.userStories | length' "$PRD_FILE" 2>/dev/null || echo "0"
}

# Helper: Get first pending story ID (sorted by priority ascending)
get_first_pending_story_id() {
  jq -r '[.userStories[] | select(.passes == false)] | sort_by(.priority)[0].id // empty' "$PRD_FILE" 2>/dev/null || echo ""
}

# Helper: Get story field by ID
get_story_field() {
  local story_id="$1"
  local field="$2"
  jq -r --arg id "$story_id" '.userStories[] | select(.id == $id) | '"$field"' // empty' "$PRD_FILE" 2>/dev/null || echo ""
}

# Helper: Get acceptance criteria as newline-separated list
get_story_criteria() {
  local story_id="$1"
  jq -r --arg id "$story_id" '.userStories[] | select(.id == $id) | .acceptanceCriteria[]' "$PRD_FILE" 2>/dev/null || echo ""
}

# Helper: Mark a story as complete in prd.json (atomic write)
mark_story_complete() {
  local story_id="$1"
  local tmp_file
  tmp_file=$(mktemp)

  jq --arg id "$story_id" \
    '.userStories = [
      .userStories[] |
      if .id == $id then .passes = true | .notes = ("Completed by Ralph at " + (now | todate))
      else . end
    ]' "$PRD_FILE" > "$tmp_file"

  if [ $? -eq 0 ]; then
    mv "$tmp_file" "$PRD_FILE"
    echo "[INFO] Marked story $story_id as complete" | tee -a "$PROGRESS_FILE"
    return 0
  else
    rm -f "$tmp_file"
    echo "[ERROR] Failed to mark story $story_id as complete" >&2
    return 1
  fi
}

# Helper: Check if all stories are complete
check_all_complete() {
  local pending
  pending=$(get_pending_count)
  [ "$pending" -eq 0 ]
}

# Helper: Log progress to progress file
log_progress() {
  local message="$1"
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $message" >> "$PROGRESS_FILE"
}

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")

  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    # Archive the previous run
    DATE=$(date +%Y-%m-%d)
    # Strip "ralph/" prefix from branch name for folder
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"

    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    echo "   Archived to: $ARCHIVE_FOLDER"

    # Reset progress file for new run
    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

# Track current branch
if [ -f "$PRD_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ]; then
    echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
  fi
fi

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

# Guard: Check if PRD file exists
if [ ! -f "$PRD_FILE" ]; then
  echo "No PRD file found at $PRD_FILE. Nothing to implement."
  echo "<promise>COMPLETE</promise>"
  exit 0
fi

echo "Starting Ralph - Tool: $TOOL - Max iterations: $MAX_ITERATIONS"
log_progress "Ralph started with tool=$TOOL, max_iterations=$MAX_ITERATIONS"

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "==============================================================="

  # Check if all stories are already complete
  if check_all_complete; then
    TOTAL=$(get_total_count)
    echo ""
    echo "All $TOTAL stories complete!"
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    log_progress "All stories complete. Ralph exiting successfully."
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  # Get pending count and progress
  PENDING=$(get_pending_count)
  DONE=$(get_completed_count)
  TOTAL=$(get_total_count)
  echo "Progress: $DONE/$TOTAL stories complete. $PENDING pending."
  log_progress "Iteration $i: Progress $DONE/$TOTAL complete, $PENDING pending"

  # Get first pending story by priority
  STORY_ID=$(get_first_pending_story_id)

  if [ -z "$STORY_ID" ]; then
    echo "No pending stories found."
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  # Extract story details
  STORY_TITLE=$(get_story_field "$STORY_ID" ".title")
  STORY_DESC=$(get_story_field "$STORY_ID" ".description")
  STORY_CRITERIA=$(get_story_criteria "$STORY_ID")

  echo ""
  echo "Processing story: $STORY_ID"
  echo "Title: $STORY_TITLE"
  log_progress "Processing story: $STORY_ID - $STORY_TITLE"

  # Build prompt in temp file (avoids shell escaping nightmares)
  PROMPT_FILE=$(mktemp /tmp/ralph_prompt_XXXXXX.md)

  cat > "$PROMPT_FILE" << 'PROMPT_END'
You are implementing a user story for the ML Engine FX trading bot.

Your job: Implement the story fully, verify each acceptance criterion passes,
then write a <promise>COMPLETE_STORY_{id}</promise> tag when done.

STORY DETAILS:
─────────────────────────────────────────
PROMPT_END

  echo "STORY ID: $STORY_ID" >> "$PROMPT_FILE"
  echo "TITLE: $STORY_TITLE" >> "$PROMPT_FILE"
  echo "" >> "$PROMPT_FILE"
  echo "DESCRIPTION:" >> "$PROMPT_FILE"
  echo "$STORY_DESC" >> "$PROMPT_FILE"
  echo "" >> "$PROMPT_FILE"
  echo "ACCEPTANCE CRITERIA:" >> "$PROMPT_FILE"
  echo "$STORY_CRITERIA" | sed 's/^/- /' >> "$PROMPT_FILE"
  echo "" >> "$PROMPT_FILE"
  echo "─────────────────────────────────────────" >> "$PROMPT_FILE"
  echo "" >> "$PROMPT_FILE"
  echo "Implement this story fully. Verify each acceptance criterion." >> "$PROMPT_FILE"
  echo "When complete, emit: <promise>COMPLETE_STORY_${STORY_ID}</promise>" >> "$PROMPT_FILE"

  # Run Claude with the story prompt
  OUTPUT=$(claude --dangerously-skip-permissions --print < "$PROMPT_FILE" 2>&1 | tee /dev/stderr) || true

  # Clean up temp prompt file
  rm -f "$PROMPT_FILE"

  # Check if story was marked complete by Claude
  if echo "$OUTPUT" | grep -q "<promise>COMPLETE_STORY_${STORY_ID}</promise>"; then
    mark_story_complete "$STORY_ID"
    echo "Story $STORY_ID marked complete."
  else
    echo "Story $STORY_ID not marked complete in this iteration."
    log_progress "Story $STORY_ID attempt made, not yet complete"
  fi

  echo "Iteration $i complete. Continuing..."
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
echo "Check $PROGRESS_FILE for status."
log_progress "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
exit 1
