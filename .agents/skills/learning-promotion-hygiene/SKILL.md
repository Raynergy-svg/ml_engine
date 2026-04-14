---
name: learning-promotion-hygiene
description: "Maintain the learning-to-rule pipeline in ML Engine. Use when learnings are malformed, promotions miscount patterns, archives bloat, mixed markdown formats conflict, or rules/learnings drift out of sync. Focus on compatibility across learnings producers, promotion safety, deduplication, and archive integrity."
---

# Learning Promotion Hygiene

Use this skill for `.claude/learnings.md`, promotion logic, rule emission, consolidation, and archive maintenance.

## What this skill protects

- mixed learnings formats from multiple writers
- safe promotion of repeated patterns
- deduplication of promoted rules
- archive integrity
- prevention of parser warnings caused by legitimate markdown variants

## Workflow

### 1. Inventory learnings producers

- Identify every module that writes to `.claude/learnings.md`.
- Compare their output formats before changing promotion logic.
- Do not assume one canonical markdown shape unless the codebase enforces it.

### 2. Parse before normalizing

- Accept legitimate historical formats where practical.
- Normalize promotion extraction narrowly so it does not start counting unrelated bullets.
- Avoid “match everything” regex fixes.

### 3. Audit promotion safety

- Confirm repeated patterns are counted correctly.
- Confirm promoted lines are marked once.
- Confirm promoted rules dedupe against prior rules.
- Confirm promotion does not rewrite unrelated narrative notes.

### 4. Check archive and consolidation behavior

- Ensure `[PROMOTED]` entries move to archive cleanly.
- Ensure active learnings remain in place.
- Watch for filename drift such as `learnings-archive` vs `learnings_archive`.

### 5. Preserve historical signal

- Do not destroy valuable markdown notes from prior phases.
- If multiple formats must coexist, make the parser compatible instead of mass-rewriting the file.

## Repo-specific anchors

- `.claude/learnings.md`
- `.claude/learnings-archive.md`
- `.claude/rules/`
- `src/scanner/automation/learning_engine.py`
- `src/scanner/automation/counterfactual_learner.py`
- `src/scanner/automation/orchestrator.py`
- `tests/test_learning_engine.py`

## Output contract

Return:

1. Producer map
2. Exact format mismatch or hygiene issue
3. Safe parser or pipeline fix
4. Tests added or needed
5. Residual risk if mixed formats remain

## Guardrails

- Do not silence warnings without explaining the real format mismatch.
- Do not rewrite learnings history just to satisfy one parser.
- Prefer compatibility plus tests over manual cleanup.
