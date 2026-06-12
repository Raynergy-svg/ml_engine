#!/usr/bin/env bash
# agent-loop.sh (ml_engine) — supervised, dev-time autonomous loop.
#
# Aligned with Anthropic engineering standards (researched + flag-verified 2026-06-11):
#   - Permission allowlist (`--permission-mode dontAsk` + `--allowedTools`) instead of
#     --dangerously-skip-permissions. Docs (code.claude.com/docs/en/permission-modes):
#     bypass mode "offers no protection against prompt injection" — not for unattended loops.
#   - `--output-format json`: failure detected via is_error, spend via total_cost_usd.
#     Cumulative budget cap is the runaway guard (--max-turns absent on this CLI version).
#   - Fresh context per iteration (best-practices: a clean session with a better prompt
#     beats a long session with accumulated corrections).
#   - Independent fresh-context reviewer audits the diff before each checkpoint
#     (harness-design post: generators confidently praise their own work).
#   - Tests as ground-truth verify gate; failures fed into ONE bounded repair step.
#
# Distinct from scripts/ralph.sh and the Tier 7 runtime control loop. Works the SAFE
# surface only, on a throwaway branch, and HARD-BLOCKS any working-tree change touching
# Danger Zone paths at the script level (defense in depth on top of the allowlist).
#
# Usage:  ./agent-loop.sh /path/to/ml_engine  [max_iterations]
# Env:    CLAUDE_LOOP_MODEL   (default claude-fable-5 — operator directive)
#         LOOP_MAX_COST_USD   (default 20 — hard cumulative spend cap)
#         LOOP_REVIEW=0       disable the reviewer pass
#
# Flags verified against `claude --help` on this machine 2026-06-11:
# --permission-mode (incl. dontAsk), --allowedTools, --output-format, --bare, --model.

set -uo pipefail
REPO="${1:-$(pwd)}"
MAX_ITERS="${2:-8}"
MODEL="${CLAUDE_LOOP_MODEL:-claude-fable-5}"
MAX_COST_USD="${LOOP_MAX_COST_USD:-20}"
REVIEW="${LOOP_REVIEW:-1}"
cd "$REPO" || { echo "Cannot cd to $REPO"; exit 1; }
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p .loop

# --- Tool allowlists -----------------------------------------------------------
# dontAsk denies everything not listed, without prompting — the agent sees the
# denial and must adapt or write a REVIEW-QUEUE proposal. No git commit/push (the
# runner checkpoints), no network tools, no arbitrary shell.
WORKER_TOOLS="Read,Glob,Grep,Edit,Write,TodoWrite,Bash(python -m pytest:*),Bash(python -m flake8:*),Bash(git status:*),Bash(git diff:*),Bash(git log:*),Bash(ls:*)"
REVIEWER_TOOLS="Read,Glob,Grep,Bash(git diff:*),Bash(git log:*),Bash(git status:*)"

# --- DANGER ZONE denylist (egrep patterns vs working-tree paths) ----------------
# Includes the loop's own harness, contract, and CI — the worker must not be able
# to edit the thing that constrains it.
DANGER='^src/scanner/|^src/training/|^src/risk/|^\.claude/|^trained_data/|(^|/)\.env|scripts/ralph\.sh|scripts/run_full_training\.sh|^scripts/agent-loop/|^CLAUDE\.loop\.md|^CLAUDE\.md|^AGENTS\.md|^\.github/|^\.flake8|^pyproject\.toml'

# --- Precheck: the ONLY hard line is practice vs live. On a demo account,
# dry_run/live and halted/running carry no real-money risk, so we don't gate them.
python - <<'PY' || { echo "PRECHECK FAILED — refusing to start: not on the OANDA practice (demo) environment."; exit 1; }
import re, sys
try:
    cfg = open("src/scanner/config.py").read()
    if not re.search(r'oanda_environment[^\n]*"practice"', cfg):
        print("oanda_environment is not 'practice' — refusing to run a dev loop against live capital.")
        sys.exit(1)
except Exception as e:
    print("cannot read config.py:", e); sys.exit(1)
sys.exit(0)
PY

# Optional: bail if a live OANDA token looks present. Adjust var name to yours.
if [ -n "${OANDA_LIVE_TOKEN:-}" ] || [ "${OANDA_ENV:-}" = "live" ]; then
  echo "Live OANDA credentials detected in env — refusing to start."; exit 1
fi

# Refuse to run while the scanner process is alive — danger_check's revert is
# repo-wide and runtime state is tracked in git; a live writer would be clobbered.
python - <<'PY' || { echo "Scanner appears ALIVE (heartbeat fresh) — refusing to start."; exit 1; }
import json, sys
from datetime import datetime, timezone
try:
    h = json.load(open(".claude/heartbeat.json"))
    ts = datetime.fromisoformat(h.get("ts_iso", "1970-01-01T00:00:00+00:00"))
    fresh = (datetime.now(timezone.utc) - ts).total_seconds() < 60
    sys.exit(1 if (h.get("scanner_alive") and fresh) else 0)
except Exception:
    sys.exit(0)  # unreadable heartbeat = dead process; safe to proceed
PY

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree dirty. Commit or stash first."; exit 1
fi

BRANCH="agent/auto-$(date -u '+%Y%m%d-%H%M')"
git switch -c "$BRANCH" || exit 1
echo "Branch: $BRANCH (main untouched, scanner never started)"
echo "Model: $MODEL · budget cap: \$$MAX_COST_USD · reviewer: $REVIEW"

# Bootstrap the files the prompt references (committed at first checkpoint).
[ -f CLAUDE.loop.md ] || cp "$DIR/CLAUDE.loop.ml_engine.md" CLAUDE.loop.md
[ -f REVIEW-QUEUE.md ] || printf '# Review queue — Danger Zone proposals for the human\n\n' > REVIEW-QUEUE.md
[ -f TASKS.md ] || cat > TASKS.md <<'EOF'
# TASKS — safe surface only (see CLAUDE.loop.md for the Danger Zone)
# Done-criteria are immutable once written: build to the criterion, never edit it to fit.
- [ ] Golden/regression tests pinning current behavior of Danger Zone modules (no edits to them)
- [ ] TUI display polish: read-only panels, layout, formatting (no control wiring)
- [ ] Docs: bring docs/ in line with current fail-closed reality
EOF

# --- claude -p wrapper: JSON output, error + cost accounting ---------------------
# Anthropic headless guidance: detect failure via is_error (not exit code alone),
# track spend via total_cost_usd. Raw JSON kept per call in .loop/ for audit.
run_claude() {  # $1=label  $2=allowed tools  $3=prompt   → prints result text
  claude -p "$3" --model "$MODEL" --bare \
    --permission-mode dontAsk --allowedTools "$2" \
    --output-format json > ".loop/${1}.json" 2>>loop.log
  python - "$1" <<'PY'
import json, sys, pathlib
label = sys.argv[1]
try:
    d = json.loads(pathlib.Path(f".loop/{label}.json").read_text())
except Exception as e:
    print(f"[{label}] unparseable claude output: {e}", file=sys.stderr); sys.exit(1)
cost = float(d.get("total_cost_usd") or 0)
ledger = pathlib.Path(".loop/cost_usd")
total = (float(ledger.read_text()) if ledger.exists() else 0.0) + cost
ledger.write_text(f"{total:.4f}")
print(f"[{label}] cost=${cost:.4f} cumulative=${total:.4f} turns={d.get('num_turns', '?')}",
      file=sys.stderr)
if d.get("is_error"):
    print(f"[{label}] is_error=true: {str(d.get('result'))[:400]}", file=sys.stderr)
    sys.exit(1)
print(d.get("result") or "")
PY
}

budget_ok() {
  python - "$MAX_COST_USD" <<'PY'
import pathlib, sys
ledger = pathlib.Path(".loop/cost_usd")
spent = float(ledger.read_text()) if ledger.exists() else 0.0
sys.exit(0 if spent < float(sys.argv[1]) else 1)
PY
}

verify() {
  python -m pytest tests/ -q --tb=line > /tmp/_v 2>&1 \
    && python -m flake8 src/ --config=.flake8 >> /tmp/_v 2>&1 || return 1
  # Backtest only when a live transformer artifact exists; the runtime is
  # currently fail-closed (all quarantined) and the CLI errors with none loaded.
  local art pair
  art="$(ls trained_data/models/*/transformer_direction.keras 2>/dev/null | head -1 || true)"
  if [ -n "$art" ]; then
    pair="$(basename "$(dirname "$art")")"
    python scripts/backtest_harness.py --instrument "$pair" \
      --out /tmp/_loop_backtest.json >> /tmp/_v 2>&1 || return 1
  else
    echo "backtest skipped: no live transformer artifact (fail-closed state)" >> /tmp/_v
  fi
}

# Hard block: abort + revert if ANY working-tree change (tracked OR untracked)
# touched a Danger Zone path. `git diff` alone misses new untracked files.
danger_check() {
  local hits
  hits="$(git status --porcelain | sed 's/^ *[A-Z?!]* *//; s/.* -> //' | grep -E "$DANGER" || true)"
  if [ -n "$hits" ]; then
    echo "DANGER ZONE TOUCHED — reverting and stopping:"; echo "$hits"
    git checkout -- . ; git clean -fd
    echo "Reverted. Branch left at last clean checkpoint: $BRANCH"
    exit 3
  fi
}

# Independent fresh-context reviewer (read-only tools). REJECT = pause for human
# with the tree intact — Anthropic's pause-on-blocker checkpoint, not silent retry.
review_diff() {
  [ "$REVIEW" = "1" ] || return 0
  [ -n "$(git status --porcelain)" ] || return 0
  local out verdict
  out="$(run_claude "review_$1" "$REVIEWER_TOOLS" "You are an independent reviewer in a fresh context — you did NOT write this change.
Read CLAUDE.loop.md (the contract), then run: git diff && git status --porcelain
Judge ONLY the working-tree change against the contract: zero Danger Zone paths, no
skipped/deleted/weakened tests, no mocks, no TUI control wiring, one small coherent
unit. Evidence over assertion — cite the diff lines that support your verdict.
Your FINAL line must be exactly 'VERDICT: APPROVE' or 'VERDICT: REJECT — <one-line reason>'." \
    | tee -a loop.log)" || { echo "Reviewer call errored — pausing for human."; exit 4; }
  verdict="$(printf '%s' "$out" | grep -Eo 'VERDICT: (APPROVE|REJECT.*)' | tail -1)"
  case "$verdict" in
    "VERDICT: APPROVE"*) echo "Reviewer approved." ;;
    *)
      printf '\n## Reviewer rejection — %s iteration %s\n%s\n' "$BRANCH" "$1" "${verdict:-no verdict emitted}" >> REVIEW-QUEUE.md
      echo "REVIEWER REJECTED iteration $1 — tree left uncommitted for human review."
      echo "${verdict:-no verdict emitted}"
      exit 4 ;;
  esac
}

checkpoint() {  # $1 = commit message
  git add -A -- ':(exclude)STATE.md' ':(exclude).loop' && git commit -q -m "$1"
}

for i in $(seq 1 "$MAX_ITERS"); do
  echo "===================== ITERATION $i / $MAX_ITERS ====================="
  if ! budget_ok; then
    echo "Budget cap \$$MAX_COST_USD reached (.loop/cost_usd) — stopping for human."; break
  fi
  bash "$DIR/state-snapshot.ml_engine.sh" "$REPO"

  PROMPT="Read CLAUDE.loop.md (your contract), STATE.md (current state incl. safety
posture), and TASKS.md. Session-start protocol: if STATE.md shows pytest or flake8
FAIL, fixing that IS this iteration's task — never build new work on a red base.
Otherwise do ONE coherent unit of SAFE work: pick the highest-value task NOT in the
Danger Zone, implement it, run pytest + flake8, and fix what you broke. Evidence over
assertion: end your reply with the actual final pytest and flake8 summary lines you
observed. Never skip, delete, or weaken a test (or edit a TASKS.md done-criterion) to
go green. No mocks. Do NOT git commit or push — the runner checkpoints for you. You
run under a permission allowlist; a denied tool call is signal — adapt or append a
proposal to REVIEW-QUEUE.md. If the best task is Danger Zone, do NOT edit it —
propose in REVIEW-QUEUE.md and pick safe work. Touch only src/tui (display), tests,
docs, notebooks. Keep changes small."

  run_claude "worker_$i" "$WORKER_TOOLS" "$PROMPT" | tee -a loop.log \
    || echo "(worker reported is_error — continuing to script-level checks)"

  danger_check   # script-level guard BEFORE we even verify

  if verify; then
    review_diff "$i"
    checkpoint "agent($i): checkpoint — verify green, review passed" \
      && echo "Iteration $i committed."
  else
    echo "Verify RED. One repair attempt..."
    run_claude "repair_$i" "$WORKER_TOOLS" "Verify failed:
$(tail -n 40 /tmp/_v)
Fix with the smallest change. No new work, no mocks, no Danger Zone edits, never
weaken a test to go green, no git commit — the runner checkpoints. End with the
actual final pytest/flake8 summary lines." | tee -a loop.log \
      || echo "(repair reported is_error — continuing to script-level checks)"
    danger_check
    if verify; then
      review_diff "$i"
      checkpoint "agent($i): checkpoint after repair — verify green, review passed"
    else
      echo "STILL RED. Stopping for human. Branch intact: $BRANCH"; exit 2
    fi
  fi
done

echo
echo "Loop complete on $BRANCH. Total spend: \$$(cat .loop/cost_usd 2>/dev/null || echo 0)"
echo "HUMAN next steps:"
echo "  git log --oneline main..$BRANCH"
echo "  cat REVIEW-QUEUE.md      # Danger-zone items + reviewer rejections"
echo "  .loop/*.json             # full per-call transcripts (cost, turns, errors)"
echo "  review the diff before anything merges. Engine stays halted until YOU decide."
