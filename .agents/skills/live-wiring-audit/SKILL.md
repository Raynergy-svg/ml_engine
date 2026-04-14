---
name: live-wiring-audit
description: "Verify that a module, tracker, agent, or automation feature is truly wired into live production paths in ML Engine. Use when code exists but may be dead, when a phase claims completion, or when tests pass while runtime behavior suggests the feature is inert. Focus on config flags, production call sites, persistence writes, and runtime consumers."
---

# Live Wiring Audit

Use this skill when a feature may exist in code but not in the live path.

## Core rule

Presence is not wiring. A class, test, or config flag does not prove runtime integration.

## Workflow

### 1. Find the claimed feature boundary

- Identify the producer, the wiring site, the persisted artifact, and the downstream consumer.
- Name the exact files expected to participate.

### 2. Prove production call sites

- Grep production files for actual method calls, not just imports or class names.
- Distinguish:
  - class definition
  - initialization
  - method invocation
  - persistence/readback
  - decision impact

### 3. Check config gating

- Verify the feature flag exists in config.
- Verify it is enabled in the relevant profile.
- Verify the guarded block still executes when enabled.

### 4. Check persistence and readback

- If a module writes state, confirm the state file/log is created and later consumed.
- If a tracker records metrics, confirm something reads those metrics for a decision or report.

### 5. Validate impact on live behavior

- A feature is only live if it changes one of:
  - gating
  - scoring
  - execution
  - sizing
  - monitoring
  - learnings/rules

## Repo-specific anchors

- `src/scanner/engine.py`
- `src/scanner/execution.py`
- `src/scanner/agents/`
- `src/scanner/automation/continuous.py`
- `src/scanner/automation/orchestrator.py`
- `.claude/rules/improvement.md`
- runtime logs and persisted JSON/JSONL artifacts

## Output contract

Return:

1. Wiring verdict: live, partial, or dead
2. Evidence for each required stage
3. Missing link if not fully wired
4. Exact patch location to complete wiring
5. Verification method that proves runtime use

## Guardrails

- Unit tests that mock boundaries do not prove live wiring.
- Imports are not enough.
- If a feature writes but nothing reads it, call it partial wiring.
