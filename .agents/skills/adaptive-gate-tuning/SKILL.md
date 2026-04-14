---
name: adaptive-gate-tuning
description: "Tune or audit ML Engine gate thresholds as a coordinated adaptive system. Use when confidence, momentum, risk, or agent-consensus thresholds are being changed; when one static gate bottlenecks an otherwise adaptive pipeline; or when the scanner alternates between over-trading and total suppression. Focus on coupled gate behavior, regime awareness, drawdown state, virtual trades, and safe one-change-at-a-time tuning."
---

# Adaptive Gate Tuning

Use this skill when adjusting gates that decide whether a setup becomes a trade.

## Core principle

No gate is independent in production. Confidence, momentum, risk, and agent-consensus thresholds interact, and the system must be tuned as a coupled control loop.

## Goals

- identify the true bottleneck gate
- prevent stacked suppression across multiple gates
- avoid over-loosening in response to a single dry spell
- preserve regime-aware and drawdown-aware safety

## Workflow

### 1. Establish the current control surface

- Identify every active gate and threshold involved in approval.
- Separate:
  - static thresholds
  - adaptive floors
  - drawdown-dependent tightening
  - regime-dependent adjustments
  - execution-quality penalties

### 2. Use raw evidence before tuning

- Read recent virtual trade outputs to see where setups are dying.
- Confirm whether the dominant failure mode is:
  - confidence
  - momentum
  - risk
  - agent consensus
  - post-gate execution rejection

### 3. Check for asymmetric adaptation

- If some gates are adaptive and one remains static, test whether the static gate became the bottleneck.
- Explicitly audit `weighted_vote_threshold` and any other hardcoded consensus rules when adaptive floors exist elsewhere.

### 4. Quantify interaction effects

- Measure how multiple penalties and floors combine.
- Look for:
  - confidence penalties stacked with disagreement penalties
  - drift or calibration penalties
  - regime multipliers
  - drawdown tightening
  - execution-quality adjustments
- Treat total suppression as a systems problem first, not a single-threshold bug.

### 5. Apply the smallest safe tuning change

- Prefer one bounded change at a time.
- Prefer completing missing adaptive logic over broadly lowering thresholds.
- Preserve hard safety invariants around portfolio risk and minimum R:R.

### 6. Verify by regime and recent outcomes

- Check whether the proposed tuning behaves differently in LOW, NORMAL, HIGH, and EXTREME regimes where applicable.
- Confirm the change would have helped recent false negatives without obviously admitting low-quality setups.

## Repo-specific anchors

- `src/scanner/config.py`
- `src/scanner/agents/`
- `src/scanner/automation/continuous.py`
- `src/scanner/automation/orchestrator.py`
- `src/scanner/automation/qa_pipeline.py`
- `.claude/learnings.md`
- `.claude/rules/improvement.md`
- virtual trade and gate-attribution artifacts

## Output contract

Return:

1. Active gate map
2. Dominant bottleneck with evidence
3. Interaction analysis across gates
4. Recommended bounded tuning change
5. Verification plan by regime and recent trades

## Guardrails

- Do not loosen multiple gates at once unless the evidence clearly shows coupled suppression.
- Do not weaken risk controls to compensate for confidence or consensus issues.
- Do not propose tuning without naming the exact gate and expected directional effect.
