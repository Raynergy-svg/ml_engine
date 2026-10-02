# Axiom 2.0 Adversarial Repairs Implementation Plan

**Goal:** Repair the demonstrated Tasks 2-11 evidence, point-in-time, economic-measurement, promotion, integration-proof, and portfolio-state gaps without changing Axiom's approved authority architecture, model family, promotion thresholds, genuine holdout state, or production execution posture.

**Base:** 26738b9c6758847322a5b733c75e6d558079c793
**Authority:** 2026-09-29 equities-first spec, 2026-10-01 execution-kernel plan, and the approved three-pass adversarial review.

## Constraints
- Do not access the genuine final holdout.
- Do not search additional model families.
- Do not weaken development/promotion gates.
- Do not reconnect Axiom to legacy FX/OANDA authority.
- Do not enable broker write capability, place/review/cancel orders, merge, deploy, reset, amend, force-push, or overwrite unrelated work.
- Every behavior repair is RED -> GREEN; update pinned research/portfolio source digests exactly when reviewed source changes.
- Keep Tasks 1-5 controls that already passed.

## Repair 1 — PIT features and declared-family truth
Tests first for NaT availability, row-time availability, reference availability, exact family projection, and revision/future invariance. Implement per-row knowability and exact selected outputs without changing admissible families.

## Repair 2 — Research evaluation semantics
Tests first for finite costs, validated disjoint/purged splits, global rebalance schedule, no-overlap policy consistency, complete top-K cohorts, per-session cross-sectional IC, initial-wealth drawdown, baseline/date alignment, and weight-drift turnover. Preserve top-5/equal/5-session policy.

## Repair 3 — Campaign/accountability and candidate evidence
Tests first proving prediction receives feature-only frames, approved adapter implementation identity is bound, failed experiments cannot run, changed manifests require a new attempt, each configuration produces durable attempt evidence, and frozen candidate construction verifies signed finite campaign evidence plus actual artifact bytes and preregistered policy/gate identity.

## Repair 4 — Promotion evidence resolution
Add a registry-backed promotion entrypoint that resolves candidate, holdout, counts, result identity, campaign evidence, immutable rule identity, and signer role before a PROMOTED decision. Keep pure scalar evaluation as non-authoritative/internal compatibility behavior but ensure it cannot be mistaken for verified promotion evidence.

## Repair 5 — Real research-kernel E2E proof
Replace the completion claim with a real proposal -> temporal/universe -> features -> labels -> splits -> baseline -> campaign -> registered freeze -> sacrificial/synthetic sealed holdout -> verified promotion proof. Add poisoned-stage negatives. Do not use the genuine holdout.

## Repair 6 — Task 11 operational hardening
Represent zero-valued/delisted holdings for reconciliation, bind external cash-flow-normalized peak/equity state, preserve risk-management requirement when routing is closed, and add authenticated/recomputed RiskDecision ingress requirements to Task 12/16 plan. Do not add execution authority.

## Completion
After every repair: focused tests, three review passes, full Axiom dependency/safety gate, research isolation artifact. Final whole-branch adversarial self-review if no independent reviewer tool is available. Commit clean checkpoints only.
