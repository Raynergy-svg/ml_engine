---
name: stall-diagnosis
description: "Diagnose zero-trade stalls, chronic gate rejections, or sudden throughput collapse in the ML Engine trading loop. Use when the system stops producing trades, when tradeable pairs drop to zero, or when a user asks why the scanner is blocked. Focus on virtual trades, gate failures, threshold interactions, adaptive floors, and recent learnings/rules before changing code."
---

# Stall Diagnosis

Use this skill for production-like trading stalls where the scanner runs but executions stop or approved setups collapse.

## Primary evidence order

1. Read `.claude/learnings.md` for recent stall-related patterns before forming a theory.
2. Check `virtual_trades.jsonl` or equivalent recent virtual trade output first.
3. Use `gate_attribution.json` only as a secondary aggregate, not the source of truth.
4. Inspect recent threshold or config changes only after confirming the kill stage.

## Workflow

### 1. Confirm the symptom

- Determine whether the issue is:
  - zero tradeable pairs
  - pairs scanned but blocked pre-execution
  - executions approved but not sent
  - portfolio/risk lockout
- Capture the concrete time window and the last known good date if available.

### 2. Identify the dominant kill stage

- Read recent virtual-trade records and count which gate fails most often.
- Separate:
  - confidence floor failures
  - momentum gate failures
  - risk gate failures
  - agent consensus failures
  - execution/routing failures after approval
- If one stage dominates, treat it as the first bottleneck.

### 3. Check for compound suppression

- Audit interactions across penalty and multiplier systems, not just single thresholds.
- Look for stacked confidence reductions, disagreement penalties, drift penalties, regime adjustments, and adaptive floors combining into full suppression.
- If three or more mechanisms all push in the same direction, quantify the compounded effect.

### 4. Verify adaptive coverage across all gates

- If some gates are adaptive, enumerate all remaining static gates.
- Confirm the system is not bottlenecked by one hardcoded threshold after the others became dynamic.
- Pay special attention to `weighted_vote_threshold` and agent consensus behavior.

### 5. Distinguish data absence from true rejection

- If aggregates are empty, confirm whether the tracker failed to record data or whether no qualifying events occurred.
- Treat empty derived files as instrumentation problems until raw evidence says otherwise.

### 6. Recommend the smallest safe fix

- Prefer:
  - instrumentation repair
  - threshold visibility
  - adaptive mechanism completion
  - one bottleneck fix at a time
- Avoid broad loosening of multiple gates in one change.

## Repo-specific anchors

- `.claude/learnings.md`
- `.claude/rules/improvement.md`
- `src/scanner/automation/continuous.py`
- `src/scanner/automation/orchestrator.py`
- `src/scanner/automation/qa_pipeline.py`
- Any `virtual_trades*.jsonl` and gate attribution outputs under project runtime data

## Output contract

Return:

1. Symptom summary
2. Dominant kill stage with direct evidence
3. Ranked contributing causes
4. Smallest safe fix
5. Verification plan

## Guardrails

- Do not trust aggregate attribution over raw virtual trade evidence.
- Do not recommend relaxing risk controls without showing the exact bottleneck and expected effect.
- Do not treat “no data” as “no problem.”
