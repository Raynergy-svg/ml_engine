---
name: execution-safety-audit
description: "Review changes to ML Engine's execution path, broker integration, and risk enforcement. Use when modifying `execution.py`, OANDA adapters, sizing, routing, or trade approval logic. Focus on preserving gate integrity, R:R minimums, correlation/risk controls, retry safety, state writes, and observability in a live trading environment."
---

# Execution Safety Audit

Use this skill before approving changes that can alter real trade behavior.

## Non-negotiables

- Do not bypass confidence, momentum, or risk gates.
- Do not approve R:R below 1.2:1.
- Do not ignore correlation or portfolio-risk controls.
- Do not introduce silent failures on the execution path.

## Workflow

### 1. Map the execution path

- Identify where approval becomes intent, where intent becomes broker request, and where outcomes sync back into learning/RL state.
- Name all touched files across execution, routing, sizing, and automation.

### 2. Check preflight safety

- Validate:
  - account/NAV availability
  - max concurrent trades
  - trade parameter sanity
  - spread/slippage controls
  - portfolio exposure and correlation checks
  - stop-loss / take-profit validity

### 3. Check risk invariants

- Confirm ATR-based SL/TP remains intact.
- Confirm regime-aware sizing still uses current equity/NAV correctly.
- Confirm retries cannot duplicate intent or corrupt state.

### 4. Check persistence and failure handling

- Look for non-atomic writes on critical state.
- Flag swallowed exceptions, especially around broker I/O, journaling, and RL sync.
- Confirm failures are logged with enough context to reconstruct what happened.

### 5. Check post-trade feedback

- A real execution change is incomplete if outcome logging, RL updates, or learnings extraction break.
- Confirm trade-close data still feeds:
  - trade journal
  - agent weights
  - learnings/promotions
  - execution quality profiling

## Repo-specific anchors

- `src/scanner/execution.py`
- `src/scanner/automation/execution_router.py`
- `src/scanner/automation/execution_quality_optimizer.py`
- `src/risk/position_sizing.py`
- `src/scanner/automation/orchestrator.py`
- `trained_data/trade_journal_rl.json`
- `trained_data/models/agent_weights.json`

## Output contract

Return:

1. Execution-risk summary
2. Broken invariant or verified invariant list
3. Highest-severity findings first
4. Safe patch recommendation
5. Verification steps with broker-safe boundaries

## Guardrails

- Treat broker credentials, order duplication, and stale state as high severity by default.
- Do not recommend wide threshold loosening as an execution fix.
- If testing against live broker behavior is not possible, say so explicitly.
