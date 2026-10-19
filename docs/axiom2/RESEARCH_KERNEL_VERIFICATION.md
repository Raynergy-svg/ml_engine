# Axiom 2.0 Research Kernel Verification

Verified on the isolated Mac worktree through the repaired Task 10 proof. The research kernel covers proposal registration, temporal availability, point-in-time universe contracts, Phase-1 features, forward labels, purged walk-forward validation, cost-aware baselines, preregistered LightGBM ranking through a model adapter, candidate freezing, single-use sealed holdout evidence, and deterministic verified signed promotion decisions.

Safety boundary: execution remains disabled. This plan does not implement Robinhood order placement, broker-neutral order submission, shadow fills, or live reconciliation. The separate Task 11 portfolio/risk core remains non-capital.

## Adversarial correction to the original Task 10 claim

The original test_research_kernel_e2e.py proved the signed registry -> freeze -> holdout -> promotion lifecycle plus separate temporal/universe negatives, but it did not call the feature, label, split, baseline or registered-campaign stages. The earlier description of that test as proof of the complete research chain was therefore too broad.

The repaired proof adds tests/axiom2/integration/test_research_pipeline_full_e2e.py. Its deterministic synthetic path actually executes:

PIT market/universe -> features -> forward labels -> purged splits -> cost-aware baseline -> registered LightGBM campaign -> verified frozen candidate -> sacrificial sealed holdout -> registry-backed verified promotion.

The holdout in this integration proof is explicitly synthetic/sacrificial. It is not the genuine final holdout.
The integration proof also poisons each required callable in turn—feature construction, label construction, split construction, baseline evaluation, registered campaign, candidate freeze, and verified promotion—and requires the pipeline proof to fail at that exact stage. This closes the adversarial finding where the earlier research-stage functions could be replaced by exceptions while the old Task 10 tests still passed.

## Research corrections preceding the repaired proof

The repaired chain also incorporates the adversarial fixes committed before this document: per-row feature decision cutoffs and exact declared-family projection; finite cost assumptions; validated purged/disjoint split admission; global non-overlapping rebalance semantics; complete top-K cohorts; per-session cross-sectional IC; initial-wealth drawdown; baseline/evaluation alignment; durable campaign trial evidence; exact campaign-manifest binding; feature-only prediction input; exact authorized LightGBM adapter implementation identity; verified candidate bytes/metrics/signature/trial identity; and registry-backed promotion evidence with role authorization and immutable promotion-rule identity.

These repairs do not silently preserve the old historical development numbers. The previously reported 36-equity campaign remains historical exploratory evidence until development data is replayed through the corrected pipeline under frozen semantics. The genuine final holdout remains sealed.

## Verification posture

The research isolation receipt deliberately retains research_kernel_complete=false and execution_enabled=false. That conservative field is not changed into capital readiness. Completion here means the locally tested research/evidence implementation and its integration proof are internally coherent; it does not mean an investable edge, live-data readiness, broker readiness, or authorization to trade.

Exact repaired-suite counts and source hashes are recorded in the adversarial-repair ledger/checkpoints. The unrelated legacy FX/OANDA suite is not made a prerequisite for Axiom research correctness.

Remaining execution work follows the execution-kernel plan: broker-neutral intents, Robinhood read-only observations, durable shadow execution, reconciliation, sole execution authority, operational privilege denial, and the separately approved controlled broker proof.
