#!/bin/bash
# Ralph Wiggum -- Local Ollama Backend
# Usage: ./ralph_kimi.sh [--model MODEL] [max_iterations]

set -e

OLLAMA_MODEL="${RALPH_OLLAMA_MODEL:-llama3.2:3b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    --model=*)
      OLLAMA_MODEL="${1#*=}"
      shift
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RALPH_DIR="$PROJECT_ROOT/.claude/ralph"
PRD_FILE="$RALPH_DIR/prd.json"
PROGRESS_FILE="$RALPH_DIR/progress.txt"
STORY_OUTPUT_DIR="$RALPH_DIR/story_outputs"

trap 'rm -f /tmp/ralph_prompt_*.md /tmp/ralph_ollama_*.json 2>/dev/null' EXIT

get_pending_count() {
  jq -r '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE" 2>/dev/null || echo "0"
}

get_completed_count() {
  jq -r '[.userStories[] | select(.passes == true)] | length' "$PRD_FILE" 2>/dev/null || echo "0"
}

get_total_count() {
  jq -r '.userStories | length' "$PRD_FILE" 2>/dev/null || echo "0"
}

get_first_pending_story_id() {
  jq -r '[.userStories[] | select(.passes == false)] | sort_by(.priority)[0].id // empty' "$PRD_FILE" 2>/dev/null || echo ""
}

get_story_field() {
  local id="$1"
  local field="$2"
  jq -r --arg id "$id" '.userStories[] | select(.id == $id) '"$field" "$PRD_FILE" 2>/dev/null || echo ""
}

get_story_criteria() {
  local id="$1"
  jq -r --arg id "$id" '.userStories[] | select(.id == $id) | .acceptanceCriteria[]' "$PRD_FILE" 2>/dev/null || echo ""
}

mark_story_complete() {
  local id="$1"
  local tmp=$(mktemp)
  jq --arg id "$id" '(.userStories[] | select(.id == $id)).passes = true' "$PRD_FILE" > "$tmp" && mv "$tmp" "$PRD_FILE"
  echo "Story $id marked complete."
}

log_progress() {
  echo "[$(date -Iseconds)] $1" >> "$PROGRESS_FILE"
}

if [ ! -f "$PRD_FILE" ]; then
  echo "No PRD file found at $PRD_FILE. Nothing to implement."
  echo "<promise>COMPLETE</promise>"
  exit 0
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama is not installed or not in PATH."
  exit 1
fi

if ! ollama list | grep -q "$OLLAMA_MODEL"; then
  echo "Error: Ollama model '$OLLAMA_MODEL' not found."
  echo "Pull it with: ollama pull $OLLAMA_MODEL"
  exit 1
fi

echo "==============================================================="
echo "  Ralph Wiggum -- Local Ollama Edition"
echo "  Model: $OLLAMA_MODEL"
echo "  Max iterations: $MAX_ITERATIONS"
echo "==============================================================="

for i in $(seq 1 "$MAX_ITERATIONS"); do
  TOTAL=$(get_total_count)
  DONE=$(get_completed_count)
  PENDING=$(get_pending_count)

  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS (ollama)"
  echo "==============================================================="
  echo "Progress: $DONE/$TOTAL stories complete. $PENDING pending."
  log_progress "Iteration $i: Progress $DONE/$TOTAL complete, $PENDING pending"

  if [ "$PENDING" -eq 0 ]; then
    echo ""
    echo "All $TOTAL stories complete!"
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    log_progress "All stories complete. Ralph exiting successfully."
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  STORY_ID=$(get_first_pending_story_id)

  if [ -z "$STORY_ID" ]; then
    echo "No pending stories found."
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  STORY_TITLE=$(get_story_field "$STORY_ID" ".title")
  STORY_DESC=$(get_story_field "$STORY_ID" ".description")
  STORY_CRITERIA=$(get_story_criteria "$STORY_ID")

  echo ""
  echo "Processing story: $STORY_ID  [model: $OLLAMA_MODEL]"
  echo "Title: $STORY_TITLE"
  log_progress "Processing story: $STORY_ID - $STORY_TITLE [model: $OLLAMA_MODEL]"

  PROMPT_FILE=$(mktemp /tmp/ralph_prompt_XXXXXX.md)

  cat > "$PROMPT_FILE" << PROMPT_EOF
You are implementing a user story for the ML Engine FX trading bot.

Your job: Implement the story fully, verify each acceptance criterion passes,
then write a <promise>COMPLETE_STORY_{id}</promise> tag when done.

STORY DETAILS:
─────────────────────────────────────────
PROMPT_EOF

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

  PROMPT_TEXT=$(cat "$PROMPT_FILE")

  STORY_LOG="$STORY_OUTPUT_DIR/${STORY_ID}_iter${i}.log"
  echo "# Ralph story output -- $STORY_ID iteration $i -- model: $OLLAMA_MODEL -- $(date -Iseconds)" > "$STORY_LOG"

  echo "Calling Ollama API ($OLLAMA_MODEL)..."
  RESP_FILE=$(mktemp /tmp/ralph_ollama_XXXXXX.json)
  HTTP_CODE=$(curl -s -o "$RESP_FILE" -w "%{http_code}" \
    --max-time 180 \
    "$OLLAMA_HOST/api/generate" \
    -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":$(echo "$PROMPT_TEXT" | jq -Rs .),\"stream\":false,\"options\":{\"num_ctx\":8192,\"temperature\":0.2}}")

  if [ "$HTTP_CODE" = "200" ]; then
    OUTPUT=$(jq -r '.response // ""' "$RESP_FILE" 2>/dev/null)
    echo "$OUTPUT" >> "$STORY_LOG"
    echo "# Exit marker -- $STORY_ID iteration $i -- $(date -Iseconds)" >> "$STORY_LOG"
    rm -f "$RESP_FILE"
  else
    echo "Ollama API error (HTTP $HTTP_CODE)." >> "$STORY_LOG"
    cat "$RESP_FILE" >> "$STORY_LOG" 2>/dev/null || true
    echo "Error: Ollama API returned HTTP $HTTP_CODE" | tee -a "$STORY_LOG"
    rm -f "$RESP_FILE"
    continue
  fi

  rm -f "$PROMPT_FILE"

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
