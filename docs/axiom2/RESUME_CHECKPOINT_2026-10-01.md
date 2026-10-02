# Axiom Mac continuation checkpoint — 2026-10-01

Initial checkout: `/Users/mirelagherasim/ml_engine-task-5`, branch `codex/axiom-2-adversarial-repairs`, clean at `a8ae582f151fcb74434e6c73f165a14d8e7aeb49`. Task 12 exists in source and tests despite stale unchecked execution-plan/document status. Baseline scoped dependency/safety gate reproduced: 1014 passed.

## Duplicate account inventory correction

AccountObservationRef never updated its uniqueness set: duplicate rows could generate duplicate shadow sell intents. Two regressions first failed with DID NOT RAISE, while 12 existing intent tests passed. Move seen.add outside the rejection branch. Construction and reflected mutation at build_order_intents ingress now reject duplicates. Update the exact reviewed source digest; no privilege or threshold change.

Three inline review passes: (1) position uniqueness and sell-inventory semantics; (2) reflected frozen-record mutation and fail-closed ingress; (3) source inventory, dependency regression, research bundle exclusion. These are author self-reviews, not independent agents.

Verified using the existing repo-suite.Gi7lHb Mac virtualenv with bytecode disabled, plugin autoload disabled, warnings errors and pytest cache disabled:
- Focused order-intent tests: 14 passed in 0.36s.
- Full approved Axiom/evidence dependency gate: 1016 passed in 16.38s.
- Both synthetic research integration suites: 10 passed in 2.48s. This proves their scoped synthetic workflows, not live or genuine-holdout end-to-end readiness.
- Standalone research isolation artifact: PASS, SHA256 `1ecf03d59e2219afa150958b62ed333dd46d7807c1eeda678ffaf468b30c3b74`; research_kernel_complete=false, execution_enabled=false.

Task13 remains LIVE UNVERIFIED: no Robinhood tools/protected connection exposed to this task. Tasks14–18 remain pending. All capital, deployment, genuine-holdout and real-order gates remain closed.
