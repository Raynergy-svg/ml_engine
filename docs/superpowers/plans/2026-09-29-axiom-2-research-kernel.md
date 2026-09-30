# Axiom 2.0 Research Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Axiom 2.0 research-to-promotion kernel that can accept replaceable research proposals, enforce point-in-time evidence, run the minimal equity ranking campaign, and promote or reject a frozen candidate without exposing production order authority.

**Architecture:** Codex is the initial replaceable research engineer. Axiom owns immutable evidence and deterministic promotion; execution remains disabled in this plan. Historical research data is provider-independent and every training value must prove when it became knowable.

**Tech Stack:** Python 3.11, pytest, pandas/numpy, LightGBM, existing `src/evidence` canonical/signing/store infrastructure.

**Spec:** `docs/superpowers/specs/2026-09-29-axiom-2-equities-first-design.md`

## Global Constraints

- No LLM/research agent in the trading hot path and no production order authority in this plan.
- Phase-1 universe is liquid US equities/ETFs, long-only.
- Phase-1 features are only proven point-in-time price/return, volume/liquidity, cross-sectional relative strength, and market/sector regime.
- News, fundamentals, earnings, L2, LLM sentiment, transformers, multi-agent consensus, options, and crypto remain quarantined.
- Current S&P 500 membership may never be projected backward.
- Final promotion requires preregistration, walk-forward OOS, selection-bias accounting, a frozen candidate, and a one-time sealed holdout.
- Existing FX/OANDA evidence remains reproducible and unchanged.

## Review Focus

1. Research-agent privilege escalation: research code must have no production broker-order dependency.
2. Temporal leakage: unknown historical availability makes a value non-trainable.
3. Holdout reuse: consumed holdouts cannot be represented as untouched.
4. Experiment cherry-picking: failed attempts remain registered and experiment count travels with evidence.
5. Survivorship: unknown historical membership/delisting state fails the affected dataset build.

---

### Task 1: Freeze FX baseline and create Axiom 2.0 namespace

**Files:** Create `docs/axiom2/FX_BASELINE_MANIFEST.md`, `src/axiom2/__init__.py`, `src/axiom2/contracts/__init__.py`; test `tests/axiom2/test_namespace_boundary.py`.

**Interfaces:** Produces importable `src.axiom2`; importing it must not import OANDA runtime modules.

- [ ] Write a failing test asserting the namespace imports and `src.brokers.oanda` is absent from `sys.modules`.
- [ ] Run `pytest tests/axiom2/test_namespace_boundary.py -v`; expect failure.
- [ ] Add focused package initializers and baseline manifest containing the pre-migration commit/evidence locations.
- [ ] Re-run the test; expect PASS.
- [ ] Commit: `chore: freeze Axiom FX baseline`.

### Task 2: Research Proposal Contract

**Files:** Create `src/axiom2/contracts/research_proposal.py`; test `tests/axiom2/contracts/test_research_proposal.py`.

**Interfaces:** Produce immutable `ResearchProposal`, `DataDependency`, and `validate_research_proposal(proposal) -> None`. Required proposal fields: experiment ID, hypothesis, universe ID, feature families, label ID, holding horizon, benchmark ID, cost-model ID, model ID, baseline IDs, primary metrics, promotion-rule ID, code commit, data dependencies.

- [ ] Write failing tests for a valid proposal, missing required fields, duplicate/empty baselines, quarantined Phase-1 features, and provider-specific agent objects.
- [ ] Run the test; expect failure.
- [ ] Implement provider-neutral frozen dataclasses and validation.
- [ ] Re-run; expect PASS.
- [ ] Commit: `feat: add Axiom research proposal contract`.

### Task 3: Temporal-availability contract

**Files:** Create `src/axiom2/data/temporal.py`, `src/axiom2/data/__init__.py`; test `tests/axiom2/data/test_temporal_contract.py`.

**Interfaces:** Produce `TemporalRecord` and `assert_trainable(record, feature_cutoff) -> None`. Fields: event time, optional published time, available-to-Axiom time, ingested time, source, revision, schema version.

- [ ] Write failing tests for valid availability, availability after cutoff, unknown availability, attempted ingestion-time substitution, timezone normalization, and revised records.
- [ ] Run tests; expect failure.
- [ ] Implement conservative validation; unknown availability is non-trainable.
- [ ] Re-run; expect PASS.
- [ ] Commit: `feat: enforce point-in-time data availability`.

### Task 4: Point-in-time universe

**Files:** Create `src/axiom2/data/universe.py`; test `tests/axiom2/data/test_universe.py`.

**Interfaces:** Produce `UniverseMembership`, `UniverseManifest`, `build_universe_as_of(timestamp, manifest) -> tuple[str, ...]`.

- [ ] Write failing tests proving current members do not appear before membership start, removed/delisted securities remain historical members, unknown intervals fail closed, and ETF exceptions require explicit manifests.
- [ ] Run; expect failure.
- [ ] Implement interval-based membership with no current-registry fallback.
- [ ] Re-run; expect PASS.
- [ ] Commit: `feat: add point-in-time equity universe`.

### Task 5: Experiment registry and sealed holdout

**Files:** Create `src/evidence/equity_research/experiment_registry.py`, `src/evidence/equity_research/holdout.py`; test corresponding files under `tests/evidence/equity_research/`. Reuse `src/evidence/store.py`, `signing.py`, and `hashing.py`.

**Interfaces:** Produce `register_experiment(proposal)`, `freeze_candidate(experiment_id, artifact_hash, code_commit)`, and `consume_holdout(candidate_id, holdout_id, result_hash)`.

- [ ] Write failing tests for append-only attempts, retention of failed experiments, frozen-candidate immutability, one-time holdout consumption, and rejection of reused holdouts as untouched.
- [ ] Run; expect failure.
- [ ] Implement using existing canonical/signing primitives; do not create a parallel evidence database.
- [ ] Re-run; expect PASS.
- [ ] Commit: `feat: add equity experiment and holdout evidence`.

### Task 6: Minimal Phase-1 feature factory

**Files:** Create `src/axiom2/research/features.py`, `src/axiom2/research/__init__.py`; test `tests/axiom2/research/test_features.py`.

**Interfaces:** `build_phase1_features(...) -> FeatureFrame` consumes point-in-time OHLCV, benchmark/sector series, and universe manifest.

- [ ] Write failing tests asserting only approved families are emitted, every value is cutoff-safe, quarantined families are rejected, and historical results are invariant when future rows are appended.
- [ ] Run; expect failure.
- [ ] Implement minimal vectorized features without importing the legacy feature-engineering class wholesale.
- [ ] Re-run; expect PASS.
- [ ] Commit: `feat: add minimal Axiom equity features`.

### Task 7: Labels, purged walk-forward splits, and naive baselines

**Files:** Create `src/axiom2/research/labels.py`, `splits.py`, `baselines.py`; tests `test_labels.py`, `test_splits.py`, `test_baselines.py`.

**Interfaces:** Produce benchmark/sector-adjusted forward-return ranking labels, purged/embargoed split manifests, and cost-adjusted naive momentum/factor baseline results.

- [ ] Write failing tests for label cutoff direction, overlapping-horizon purging, embargo boundaries, benchmark alignment, and conservative cost subtraction.
- [ ] Run the three test files; expect failure.
- [ ] Implement minimal functions with cost assumptions as explicit inputs/evidence IDs.
- [ ] Re-run; expect PASS.
- [ ] Commit: `feat: add equity labels validation splits and baselines`.

### Task 8: Preregistered LightGBM ranking campaign

**Files:** Create `src/axiom2/research/ranker.py`, `scripts/axiom2_run_phase1_campaign.py`; test `tests/axiom2/research/test_ranker_campaign.py`.

**Interfaces:** Consume a registered `ResearchProposal`, immutable feature/label manifests, and walk-forward splits; produce signed results with ranking metrics, portfolio-after-cost metrics, regime slices, turnover, drawdown, attempted-search count, and artifact hash.

- [ ] Write failing deterministic-fixture tests: unregistered proposal cannot run; schema mismatch fails; results include baseline comparison and experiment count; campaign cannot access sealed holdout.
- [ ] Run; expect failure.
- [ ] Implement the runner with LightGBM as the only ML candidate; preregister and record the finite hyperparameter search space.
- [ ] Re-run; expect PASS and deterministic manifest.
- [ ] Commit: `feat: add preregistered Axiom equity ranking campaign`.

### Task 9: Promotion authority

**Files:** Create `src/axiom2/promotion/authority.py`, `src/axiom2/promotion/__init__.py`; test `tests/axiom2/promotion/test_authority.py`.

**Interfaces:** Consume frozen candidate, baseline/walk-forward/selection-bias evidence, and sealed-holdout result; produce signed immutable `PromotionDecision(PROMOTED|REJECTED)`.

- [ ] Write failing tests rejecting missing baselines, missing costs, missing experiment-count accounting, candidate mutation after freeze, reused holdout, and failed promotion rule; include one complete passing synthetic package.
- [ ] Run; expect failure.
- [ ] Implement deterministic promotion evaluation with no LLM/free-form override.
- [ ] Re-run; expect PASS.
- [ ] Commit: `feat: add deterministic Axiom promotion gate`.

### Task 10: Research-kernel end-to-end proof

**Files:** Create `tests/axiom2/integration/test_research_kernel_e2e.py`; update `docs/axiom2/RESEARCH_KERNEL_VERIFICATION.md`.

**Interfaces:** Exercise proposal -> registry -> temporal/universe validation -> features -> labels/splits -> baseline/ranker -> frozen candidate -> sealed holdout -> promotion decision.

- [ ] Write the end-to-end test with a deterministic synthetic multi-equity fixture and a separate hidden holdout fixture.
- [ ] Add negative assertions for future-data mutation, unknown membership, and second holdout consumption.
- [ ] Run `pytest tests/axiom2 tests/evidence/equity_research -v`; expect all PASS.
- [ ] Run the existing relevant evidence tests to prove no regression.
- [ ] Record exact commands/results and remaining production-data prerequisites in the verification doc.
- [ ] Commit: `test: prove Axiom 2.0 research kernel end to end`.

## Plan Completion Gate

Stop after Task 10. Do **not** build Robinhood order placement merely because the research kernel compiles. The next plan begins only after this kernel passes and will cover:

1. deterministic equity portfolio/risk authority;
2. broker-neutral equity order contracts;
3. Robinhood minimum-data read-only adapter;
4. shadow execution and fill/cost reconciliation;
5. sole execution authority and negative privilege tests;
6. tightly controlled broker execution proof.

