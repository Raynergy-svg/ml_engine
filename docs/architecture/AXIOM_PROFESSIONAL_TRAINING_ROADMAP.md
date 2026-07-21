# AXIOM Professional-Grade Training Roadmap

This is the master training roadmap for the whole AXIOM OS. It treats training as a governed lifecycle across every strategy lane, the hedge and risk layers, the learning loops, and the combined portfolio.

## 1. Mission

Build AXIOM into a complete, professional-grade research, training, verification, and promotion system where:

- Every strategy and learning component trains from immutable, traceable data.
- Every experiment is pre-registered or explicitly labeled exploratory.
- Remote compute can produce evidence but cannot control capital.
- Every artifact is independently verified before AXIOM can load it.
- Negative results are preserved as first-class evidence.
- Every lane accumulates honest forward performance after training.
- Portfolio-level risk and correlation determine final promotion—not standalone model metrics.
- Local AXIOM policy and operator authority remain the final control plane.

The final operating loop is:

```text
Data capture
→ validation
→ immutable snapshot
→ hypothesis or training specification
→ signed job
→ training/research
→ independent evaluation
→ immutable EvidencePackage
→ local import verification
→ quarantine
→ shadow-forward testing
→ portfolio gate
→ operator approval
→ champion pointer
→ lane loading
→ monitoring
→ recalibration, retraining, retirement, or rollback
```

---

## 2. Current AXIOM baseline

AXIOM already has several professional-grade foundations.

### Existing strengths

#### Lane-oriented safety

The harvester branch defines independently haltable lanes:

- `oanda_fx`
- `equity`
- `brain`
- `crypto_momentum`
- `track_b`
- `crypto_carry`

The global halt remains the master override, while missing or malformed configured lane state fails closed.

#### Strong research methodology

The current research system already uses:

- Pre-registration.
- Frozen parameters.
- Untouched out-of-sample data.
- Deflated Sharpe Ratio.
- Bonferroni-adjusted block bootstrap.
- Maximum-drawdown gates.
- Minimum-history gates.
- Separate adversarial verification.

The audit correctly identifies this harness as one of AXIOM’s strongest assets.

#### Modern but legacy-heavy ML estate

The older desktop pipeline contains:

- Tier 1: 8-model core ensemble across 5 FX master pairs.
- Tier 2: transfer learning.
- Tier 3: PPO/SAC RL suite.
- Tier 4: XGBoost meta-labeling.
- Tier 5: confidence calibration.
- Final: validation report.

It still runs sequentially through a shell pipeline designed for an M1 Mac and local TensorFlow Metal execution.

#### Headless learning now exists

The offline learning cycle has already repaired a major architectural weakness. It now:

- Consumes pending retraining markers.
- Applies agent-weight updates.
- Trains risk calibration.
- Uses a chronological holdout.
- Rejects candidates that fail Brier-score comparison.
- Prevents risk multipliers above 1.0.
- Runs without the TUI.
- Never touches orders, halts, or broker state.

#### Online retraining now has a gate

The online retrainer is limited to lightweight sklearn gate models and reserves the temporal tail of replay data as a holdout. It refuses undersized or degraded candidates and limits retraining frequency.

#### Risk-target training is properly isolated

The new risk-target trainer predicts:

- Forward realized volatility.
- Drawdown-stress probability.

It is structurally decoupled from the live directional ensemble and emits risk estimates rather than trade direction.

### Current limitations

The system is not yet professionally complete because:

- Training data and evidence use multiple unrelated storage conventions.
- The full legacy training pipeline is desktop-sequential.
- Research harness logic remains duplicated across experiment scripts.
- Remote execution contracts do not yet exist.
- There is no immutable evidence-package implementation.
- There is no signed disposition chain.
- There is no universal local candidate importer.
- The resident agent does not observe every lane and evidence family.
- Crypto exposure is not fully modeled in the hedge layer.
- Crypto carry is not integrated across all portfolio-wide services.
- Training and promotion remain inconsistent between old FX components and newer lane-contract components.
- Portfolio-level promotion is not yet the universal final gate.

---

## 3. Training scope

Professional-grade AXIOM training includes nine distinct programs.

| Program | Purpose | Training status |
|---|---|---|
| FX trend and execution | Risk posture, sizing, fill quality and regime behavior | Active operational lane |
| Legacy FX ensemble | Historical benchmark and controlled research reference | Retired from primary alpha search |
| Equity harvester and sleeves | Equity beta/risk-premium allocation and future orthogonal sleeves | Active shadow research |
| Crypto momentum | Forward-test frozen cross-sectional momentum | Shadow, significance not cleared |
| Crypto carry | Funding/basis risk-premium research | Shadow, cost/tail gate not cleared |
| Track B | Point-in-time SEC filing-text research | Shadow, temporal evidence incomplete |
| Hedge and exposure | Cross-lane concentration, hedge selection and raw-vs-hedged testing | Analysis-only |
| Risk-target ML | Volatility and drawdown-state estimation | Active training candidate |
| Learning/calibration | Agent weights, confidence, execution and risk calibration | Local bounded updates |

The brain loop and resident AXIOM operator orchestrate these programs but are not themselves predictive models.

---

## 4. Governing principles

### 4.1 Train only admissible targets

AXIOM must not treat all possible predictions as equally legitimate.

#### Active targets

- Volatility.
- Drawdown or stress probability.
- Liquidity and fill quality.
- Expected transaction cost.
- Exposure concentration.
- Calibration.
- Abstention and gate quality.
- Regime-conditioned risk posture.
- Strategy-level and portfolio-level risk.
- New-input alpha hypotheses.

#### Frozen or retired targets

Price-only free-bar FX direction remains a benchmark, not an active research priority. The audit found that real leakage and methodology defects had existed, but after those were fixed, direction accuracy remained near the market-information limit.

Professional training does not mean automatically retraining every old model. It means refusing to spend compute on targets that AXIOM has already falsified.

### 4.2 No self-attestation

The training process that produces a candidate cannot be the sole process that approves it.

Every promotion requires:

```text
producer result
+ independent verifier result
+ local policy verdict
```

### 4.3 Negative results are permanent assets

A failed hypothesis must generate an immutable evidence package containing:

- The frozen hypothesis.
- Data lineage.
- Trial count.
- Evaluation metrics.
- Gate failures.
- Verifier result.
- Rejection reason.

That prevents AXIOM from repeatedly rediscovering the same dead strategy.

### 4.4 Remote computation has no authority

Remote workers may read approved snapshots and write job-scoped results.

They may never:

- Read broker credentials.
- Place or cancel orders.
- Read operator signing keys.
- Change lane halt state.
- Change LiveGate state.
- Write a champion pointer.
- Modify local model directories.
- Approve their own evidence.
- Promote a package to shadow or live.

---

## 5. The canonical training lifecycle

### Stage 1 — Data acquisition

Every data source gets a dedicated ingestion process.

#### FX

- OANDA bars.
- Daily FX history.
- Tick capture.
- Position and order-book snapshots.
- Realized fills.
- Spread and slippage.
- Macro and rates data where appropriate.

#### Equity

- Adjusted prices.
- Corporate actions.
- Survivorship-aware universes.
- Point-in-time SEC fundamentals.
- FINRA short-volume data.
- Sector and market mappings.
- Rebalance and shadow-forward records.

#### Crypto

- Perpetual klines.
- Funding history.
- Spot prices.
- Spot-perpetual basis.
- Delisted-contract history.
- Venue metadata.
- Contract specification and margin rules.
- Liquidity and cost data.

#### Track B

- Original SEC documents.
- Filing metadata.
- Model-cutoff dates.
- Blinded document batches.
- Score artifacts.
- Price panels used for forward evaluation.

#### Portfolio and learning

- Per-lane books.
- Exposure snapshots.
- Hedge decisions.
- Raw-vs-hedged returns.
- Closed-trade outcomes.
- Agent votes and gate decisions.
- Calibration inputs.
- Risk-target labels.

### Stage 2 — Data validation

Every ingestion job must produce a data-quality report.

Required checks include:

- Schema conformity.
- Timestamp monotonicity.
- Duplicate detection.
- Missing-partition detection.
- Point-in-time correctness.
- Survivorship treatment.
- Corporate-action handling.
- Staleness.
- Outlier and bad-tick detection.
- Instrument identity consistency.
- Time-zone normalization.
- Data-source and license metadata.
- Look-ahead and cutoff checks.

A failed data-quality gate prevents snapshot creation.

### Stage 3 — Immutable snapshot creation

Validated data is normalized into partitioned Parquet or equivalent immutable storage.

Each snapshot receives a `DatasetManifest` containing:

- Dataset ID.
- Asset class and domain.
- Provider and retrieval time.
- Coverage dates.
- Instruments.
- Schema version.
- Point-in-time rules.
- Exclusions.
- Partition hashes.
- Full manifest digest.
- Signature.

Training must never define its dataset by making ad hoc live broker calls during the training process.

### Stage 4 — Research or training specification

Every run begins with either a `StrategyManifest` or a model training specification.

#### StrategyManifest

For model-free or strategy-rule research:

- Equity harvester.
- Crypto momentum.
- Crypto carry.
- Track B portfolio construction.
- Multi-asset sleeves.
- Hedge overlays.

#### Model training specification

For fitted models:

- Risk target.
- Execution-cost model.
- Calibration models.
- Gate models.
- Legacy benchmark models.

The specification freezes:

- Targets.
- Features.
- Causal lag.
- Universe.
- Cost model.
- Evaluation windows.
- Search space.
- Trial budget.
- Metrics.
- Promotion criteria.

Changing a frozen field creates a new experiment identity.

### Stage 5 — Signed job creation

AXIOM creates a `JobManifest` containing:

- Exact git commit.
- Container digest.
- Dataset-manifest digests.
- Strategy-manifest digest.
- Feature-pipeline version.
- Configuration digest.
- Random seeds.
- Assigned fold, target or document batch.
- Resource class.
- Capability profile.
- Trial budget.
- Expected outputs.

The job manifest is signed before compute begins.

### Stage 6 — Controlled execution

The worker:

1. Verifies the manifest signature.
2. Confirms that requested data hashes exist.
3. Loads only declared partitions.
4. Executes the declared workload.
5. Writes only under its job-specific namespace.
6. Produces metrics, logs and artifacts.
7. Emits an immutable `EvidencePackage`.
8. Creates only the remote-authorized `null → CREATED` event.

It cannot alter the job after signature verification.

### Stage 7 — Evaluation

Evaluation is separate from fitting.

Every relevant workload receives:

- Temporal holdout.
- Purging and embargo.
- Walk-forward validation.
- Cost-adjusted backtest.
- Baseline comparison.
- Regime-stratified reporting.
- Stress testing.
- Parameter-sensitivity testing.
- Multiple-testing adjustment.
- Data and causality checks.
- Incumbent comparison.

CPCV should be added where path dependence and overlapping observations justify it. The existing audit already identifies this as a useful methodology upgrade, unlike further loss-function experimentation on signal-free targets.

### Stage 8 — Evidence packaging

The immutable package contains:

```text
EvidencePackage
├── package identity and digest
├── lineage
├── manifests
├── evaluation reports
├── model or strategy artifacts
├── logs
├── safety assertions
├── checksums
└── producer signature
```

The package never changes after creation.

### Stage 9 — Local import and replay

The local importer verifies:

- Producer identity.
- Signature.
- Package digest.
- Artifact hashes.
- Dataset lineage.
- Git and container identities.
- Capability compliance.
- Search/trial count.
- Evaluation completeness.
- Strategy parameters.
- No forbidden code or credentials.
- Metric reproduction.
- Gate verdict reproduction.

It then creates signed local disposition events.

### Stage 10 — Quarantine

Imported artifacts first enter quarantine.

Quarantine means:

- Not loadable by a lane.
- Not a champion.
- Not visible to execution code.
- Available for replay, inspection and comparison.
- Preserved even if rejected.

### Stage 11 — Shadow-forward qualification

A package that passes offline gates must prove itself forward.

Each lane records:

- Actual decision date.
- Frozen parameters.
- Would-be positions.
- Expected costs.
- Realized forward returns.
- Drawdown.
- Turnover.
- Data freshness.
- Exposure.
- Abstentions.
- Gate decisions.

No backfill may be presented as forward evidence.

### Stage 12 — Portfolio gate

Before championing, AXIOM evaluates the candidate inside the full portfolio.

The portfolio gate considers:

- Standalone expectancy.
- Correlation to current lanes.
- Marginal Sharpe contribution.
- Drawdown contribution.
- Tail dependence.
- Currency, sector and market concentration.
- Hedge cost.
- Turnover and liquidity.
- Capacity.
- Stress-regime behavior.
- Whether the candidate improves the incumbent combined book.

A candidate can pass alone and still fail the portfolio gate.

### Stage 13 — Operator promotion

The allowed sequence is:

```text
CREATED
→ RECEIVED
→ HASH_VERIFIED
→ POLICY_VERIFIED
→ METRIC_REPLAYED
→ QUARANTINED
→ SHADOW
→ OPERATOR_APPROVED
→ CHAMPION
→ RETIRED
```

Only the operator may authorize the transition into the approved promotion path.

### Stage 14 — Lane loading

A lane may load an artifact only when:

- The local champion pointer references the exact package digest.
- The artifact hash matches.
- The lane ID matches.
- The feature and data contracts match.
- The package is not retired.
- The lane is not halted.
- Staleness rails pass.
- All local safety requirements remain intact.

Remote MLflow, W&B or cloud registry aliases cannot grant load permission.

### Stage 15 — Monitoring and retirement

Every champion gets a frozen promotion baseline.

Monitoring compares actual behavior against:

- Expected return.
- Expected volatility.
- Drawdown.
- Turnover.
- Cost.
- Calibration.
- Exposure.
- Feature distribution.
- Regime distribution.
- Prediction confidence.
- Abstention frequency.
- Data freshness.

Breaches can produce:

- Warning.
- Tighter risk.
- Shadow fallback.
- Retraining request.
- Lane halt proposal.
- Immediate retirement.

Retirement is a signed event, not deletion.

---

## 6. Foundation program

### Phase A — Freeze the contract layer

**Implementation status (2026-07-11): COMPLETE.** Strict contracts, canonical
hashing, Ed25519 envelopes, migration rules, transition authority, and tamper/
reconstruction tests are implemented under `src/evidence/`.

**QA correction (2026-07-12):** the code and tests are genuinely correct — all 36
tests in `tests/test_evidence_contracts.py`/`tests/test_evidence_store.py` pass,
including real tamper-detection (one-byte payload change → `SignatureVerificationError`)
and real index-reconstruction (delete `indexes/`, rebuild, byte-equal) tests. But
"COMPLETE" here means implemented-and-tested-in-isolation, not wired: only
`DatasetManifest` plus `canonical.py`/`hashing.py`/`signing.py` have production
callers (via `src/data_platform/`). `EvidenceStore` (Phase B, `store.py`),
`event_store.py`, `indexes.py`, `importer.py`, `transition_policy.py`, and 7 of the
8 contract types have zero callers outside `src/evidence/` and `tests/` — no
training or capture pipeline constructs an `EvidencePackage`, writes a champion
pointer, or appends a `DispositionEvent` yet. Treat this phase as "foundation built
and tested" rather than "in the loop."

Implement the eight formal contracts:

- `DatasetManifest`
- `StrategyManifest`
- `JobManifest`
- `EvaluationReport`
- `EvidencePackage`
- `CapabilityProfile`
- `DispositionEvent`
- `LocalImportVerdict`

#### Deliverables

```text
src/evidence/contracts/
src/evidence/canonical.py
src/evidence/hashing.py
src/evidence/signing.py
src/evidence/transition_policy.py
src/evidence/event_store.py
src/evidence/indexes.py
src/evidence/importer.py
```

#### Requirements

- Strict Pydantic or equivalent schemas.
- Canonical serialization.
- SHA-256 content addressing.
- Ed25519 or equivalent signatures.
- Key rotation and signer identity.
- Schema-version migration rules.
- Unknown-field policy.
- Golden serialization tests.
- Tamper-detection tests.
- Event-chain reconstruction tests.
- Transition-authority tests.

#### Exit gate

A one-byte change to any signed object must invalidate verification, and the current package state must be reproducible solely from the disposition ledger.

### Phase B — Build the evidence store

**Implementation status (2026-07-11): COMPLETE.** Immutable package writes,
append-only signed dispositions, lock+CAS heads, rebuildable indexes, and exact
champion pointers are implemented in `src/evidence/store.py`.

Recommended local structure:

```text
trained_data/evidence/
├── packages/
├── dispositions/
├── verdicts/
├── indexes/
├── quarantine/
└── champions/
```

#### Required behavior

- Package writes use temporary paths and atomic rename.
- Disposition events are append-only.
- Each event references the prior event digest.
- Event sequence must be monotonic.
- Indexes are atomically replaced.
- Indexes can be deleted and rebuilt from history.
- Concurrent head updates use locking or compare-and-swap.
- Forked event histories are detected and rejected.
- Every champion pointer references exact package and artifact hashes.

#### Exit gate

Destroy the indexes, reconstruct them from packages and events, and obtain byte-equivalent current state.

---

## 7. Data-platform program

### Phase C — Standardize data storage

**Implementation status (2026-07-11): COMPLETE.** The canonical local-compatible
object-store API and scoped ingestion/training/feature/control capabilities are
implemented under `src/data_platform/`.

Use partitioned object storage or a local-compatible object-store layout:

```text
axiom-data/
├── fx/
├── equity/
├── crypto/
├── filings/
├── multi_asset/
├── portfolio/
├── outcomes/
└── evidence/
```

#### Storage principles

- Raw data remains immutable.
- Normalized data is versioned.
- Training snapshots are immutable.
- Historical records use Parquet, JSONL or a database—not repeatedly rewritten JSON arrays.
- Small control state remains local atomic JSON.
- Feature outputs are keyed by snapshot and feature-pipeline digest.
- No training job writes into source-data paths.

### Phase D — Activate missing forward-data capture

**Implementation status (2026-07-11): COMPLETE.** All seven capture families
write through the canonical forward/backfill contract; tick and exposure have
resident launchd definitions, daily Track B/hedge/P2 capture and monthly crypto
refresh schedules are included in `scripts/axiom_launchd/load.sh`. These new
LaunchAgents are installable but were not bootstrapped into the operator's user
session by this implementation change. Risk-target P2 remains correctly blocked
until the canonical readiness command observes 60 forward weekdays for exposure
and every required pair.

**QA correction (2026-07-12):** "all seven families write through the canonical
contract" is technically true (each has a `capture_*`/`capture_best_effort()`
call site) but overstates completeness — in 5 of 7 families (tick/spread,
exposure, trade-entry context, crypto klines/funding, and — worst — trade-entry
context) the canonical write is a secondary, exception-swallowed mirror bolted
onto a pre-existing legacy path that remains the actual source of record read by
the rest of the system. One of those legacy paths
(`src/scanner/execution.py`'s `trained_data/trade_journal_rl.json` writes) is a
whole-file JSON-array read-modify-rewrite on every trade — the exact anti-pattern
§7's storage principles forbid — not yet migrated because the file is live
trading state read by the TUI, RL trainer, and dashboards; migrating its format
needs its own scoped, coordinated change, not a QA-pass edit. The 60-forward-weekday
P2 readiness number and computation (`src/data_platform/forward_capture.py`) are
real and correctly computed, but nothing currently *consumes* that readiness
state to block anything — acceptable today only because Risk-target P2 training
itself doesn't exist yet to be blocked. Bootstrap status has also drifted since
2026-07-11: `com.buddy.exposure_history` is now loaded and running; the other
three (`tick_capture`, `forward_daily`, `crypto_refresh`) remain unloaded.

Priority capture pipelines:

1. Tick and spread history.
2. Exposure history.
3. Full trade-entry context.
4. Fill-book ladders and slippage.
5. Updated crypto klines and funding.
6. New Track B filings and rebalance dates.
7. Hedge raw-vs-hedged forward records.

The training audit found that only a small portion of journal records contained complete agent/gate context, while real order-fill ladders existed but were not being consumed by training.

Risk-target P2 should remain blocked until enough forward-only exposure and tick history exists. Its current gate calls for roughly 60 trading days of both before the added feature family is frozen and evaluated.

---

## 8. Research-methodology program

### Phase E — Extract one canonical gated harness

Create:

```text
src/research/gated_harness/
├── preregistration.py
├── temporal_splits.py
├── purging.py
├── cpcv.py
├── backtest.py
├── costs.py
├── significance.py
├── regimes.py
├── stress.py
├── verifier.py
└── report.py
```

#### Mandatory capabilities

- Frozen hypothesis hash.
- Untouched OOS.
- Trial-count accounting.
- Deflated Sharpe.
- Block bootstrap.
- Bonferroni or declared correction.
- CPCV where applicable.
- Regime-stratified results.
- Minimum effective sample size.
- Minimum history.
- Cost scenarios.
- Drawdown gate.
- Placebo tests.
- Causality tests.
- Survivorship/PIT declaration.
- Independent replay.

#### Migration process

Port one already completed experiment and require byte-identical results before moving additional experiments.

#### Exit gate

All new research workloads call the same harness library rather than copying evaluation code.

---

## 9. Local performance program

### Phase F — Repair algorithmic bottlenecks before cloud migration

Cloud compute must not be used to conceal inefficient algorithms.

#### Legacy RL data generation

The old RL path must be redesigned to:

- Compute features once.
- Eliminate growing-prefix DataFrame copies.
- Batch model inference.
- Use fixed windows.
- Materialize training matrices once.
- Store arrays as Parquet, Arrow or memory-mapped files.
- Separate sample creation from RL fitting.

#### Equity and Track B

- Predicate-pushdown date and ticker selection.
- Partition by rebalance period.
- Reuse PIT universe snapshots.
- Batch SEC parsing and scoring.
- Never load the full filing corpus into one worker.

#### Crypto

- Partition klines and funding by symbol and month.
- Materialize eligibility separately.
- Construct only required cross-sections.
- Separate momentum and carry datasets.

#### Risk target

- Compute trailing features per pair.
- Cache immutable feature partitions.
- Pool only final valid rows.
- Preserve feature-window invariance tests.

#### Exit gate

Each training family has a baseline report with:

- Wall time.
- Peak RSS.
- CPU utilization.
- GPU utilization.
- I/O volume.
- Dataset size.
- Feature-computation time.
- Fit time.
- Evaluation time.
- Artifact size.

---

## 10. Container and environment program

### Phase G — Build workload-specific images

Do not create one giant AXIOM image.

Recommended images:

- `axiom-risk-target`
- `axiom-execution-cost`
- `axiom-fx-benchmark`
- `axiom-equity-research`
- `axiom-crypto-momentum`
- `axiom-crypto-carry`
- `axiom-track-b-ingestion`
- `axiom-track-b-scoring`
- `axiom-track-b-harness`
- `axiom-hedge-evaluation`
- `axiom-multi-sleeve`
- `axiom-independent-verifier`

#### Every image must have

- Locked dependencies.
- Immutable digest.
- Non-root user.
- Read-only root filesystem.
- Health check.
- Resource limits.
- No broker SDK credentials.
- No local control files.
- Minimal network policy.
- Software bill of materials.
- Vulnerability scan.
- Reproducible build.
- Signed image.

#### Exit gate

The same signed job produces equivalent evidence locally and in the isolated worker environment.

---

## 11. Distributed research fabric

### Phase H — Introduce a batch job graph

Start with ordinary batch orchestration rather than a permanently running cluster.

```text
snapshot validation
→ feature generation
→ fold workers
→ result aggregation
→ independent verification
→ evidence packaging
```

#### Natural units of parallelism

| Workload | Parallel unit |
|---|---|
| FX benchmark | pair × model × fold |
| Risk target | target head × fold |
| Equity research | hypothesis × universe snapshot × fold |
| Crypto momentum | construction × fold × stress |
| Crypto carry | venue set × cost model × regime |
| Track B scoring | filing batch × scorer version |
| Track B evaluation | score artifact × rebalance date × fold |
| Hedge layer | strategy × hedge type × forward window |
| Sleeve book | sleeve combination × allocation method × fold |

#### Deterministic aggregation

Workers never determine the final verdict.

A separate aggregator:

- Confirms all expected folds arrived.
- Refuses missing or duplicate folds.
- Applies trial-count corrections.
- Calculates combined metrics.
- Produces an evaluation report.
- Hands results to an independent verifier.

#### Resource classes

- `cpu-small`
- `cpu-memory`
- `gpu-small`
- `gpu-large`
- `document-processing`
- `verification`

Resource selection is part of the signed job manifest.

#### Cost controls

- Per-job cost ceiling.
- Per-research-campaign trial ceiling.
- Daily and monthly quotas.
- Automatic termination for idle or stuck jobs.
- Spot/preemptible use only for restartable workers.
- Checkpointing for long jobs.
- No open-ended hyperparameter sweep.
- Cost included in the evidence report.

---

## 12. First vertical slice

### Phase I — Risk-target end-to-end proof

Risk target should be the first complete implementation because it is:

- Offline.
- Decoupled from execution.
- Model-bearing.
- Gated per head.
- Useful to multiple lanes.
- Small enough to replay locally.

The current trainer already evaluates forward volatility with QLIKE, pinball loss, R² and MAE, and evaluates drawdown probability with AUC and Brier score.

**QA status note (2026-07-12):** the diagram below is not implemented yet — none
of its stages are built (`cli/risk_target_training.py`/`RiskTargetTrainer` never
touch `src/evidence/` contracts; there is no remote/local split, no
`DatasetManifest`/`JobManifest`/`EvidencePackage`, no dashboard surface). What
exists today is a real, gated, offline LightGBM trainer (`src/training/trainers/risk_target_trainer.py`)
with a regression gate before overwriting the incumbent artifact
(`cli/risk_target_training.py`), tested in `tests/test_risk_target_training_gate.py`.
The forward-vol head is genuinely learnable (OOS QLIKE 0.0505 vs 0.0712 naive,
19/19 pairs, per `docs/prereg-risk-target-vol-drawdown-2026-07-08.md`); the
drawdown-state head honestly failed its own pre-registered bar (OOS AUC 0.625
clears 0.55, but OOS Brier 0.232 is worse than the 0.143 base-rate baseline).
Until this QA pass, the gate only checked regression against the incumbent, so a
not-learnable drawdown candidate could be (and was) written and marked PASSED on
first deploy — fixed in `cli/risk_target_training.py`/`risk_target_trainer.py` to
independently enforce the same threshold/formula as the pre-registered bar
(AUC>=0.55 AND Brier beats p(1-p) baseline), evaluated on each retrain's own
validation split — a per-retrain proxy for the pre-reg doc's one-time frozen OOS
verdict, not a re-derivation of it; see the "IMPORTANT" note on
`DRAWDOWN_LEARNABLE_MIN_AUC` in `risk_target_trainer.py`
(`tests/test_risk_target_training_gate.py::test_first_deploy_refuses_drawdown_head_that_fails_absolute_prereg_bar`).
Zero production code currently consumes risk-target predictions, so this gap was
latent, not live.

#### Vertical slice

```text
FX daily snapshot
→ DatasetManifest
→ risk-target feature snapshot
→ signed JobManifest
→ isolated training
→ per-head EvaluationReport
→ EvidencePackage
→ remote CREATED event
→ local RECEIVED event
→ hash verification
→ policy verification
→ local metric replay
→ LocalImportVerdict
→ QUARANTINED or REJECTED
→ dashboard display
```

#### Acceptance tests

- Remote worker cannot read `.claude/state.json`.
- Remote worker has no broker credentials.
- Candidate artifact cannot overwrite incumbent.
- Missing fold causes failure.
- Changed dataset partition causes hash failure.
- Changed model bytes cause signature failure.
- Worse head is rejected independently.
- Good head does not rescue a failed head automatically.
- Local replay reproduces metrics within declared tolerance.
- External registry outage does not stop local verification.

---

## 13. Lane-specific training programs

### Phase J1 — FX program

#### Active goals

- Trend-risk estimation.
- Volatility forecasting.
- Drawdown-state forecasting.
- Regime-conditioned sizing.
- Fill-quality modeling.
- Spread and slippage forecasting.
- Currency-exposure analysis.
- Execution and abstention quality.

#### Legacy benchmark estate

Preserve the older ensemble for:

- Regression testing.
- Training-infrastructure testing.
- Feature-contract testing.
- Historical comparisons.

Do not automatically promote it back into the active alpha search.

#### RL prerequisites

The RL suite may only advance after:

- Training states reflect the current trend lane.
- Entry context is complete.
- Outcome labels are reliable.
- Rejected setups are recorded with full features.
- The sample size is adequate.
- Off-policy evaluation is available.
- A deterministic baseline beats the RL policy.
- Risk constraints are encoded structurally.
- RL artifacts use the same evidence package and shadow process.

Until then, RL remains research-only.

### Phase J2 — Equity harvester and sleeve program

#### Data requirements

- Survivorship-aware universe.
- Corporate actions.
- PIT fundamentals.
- Delisted names.
- Sector classifications.
- Transaction-cost estimates.
- Capacity and liquidity.

#### Training/research process

```text
universe snapshot
→ causal target weights
→ cost-aware portfolio simulation
→ standalone gate
→ independent verifier
→ shadow-forward book
→ combined-book evaluation
```

#### Required reporting

Always report:

- Curated-universe result.
- Wide survivorship-corrected result.
- Full-history result.
- OOS result.
- Drawdown.
- Turnover.
- Concentration.
- Beta.
- Marginal portfolio contribution.

The branch itself documents that the attractive curated-universe result and the defensible wide-universe result are materially different.

#### Future sleeve factory

Candidate sleeves include:

- Multi-asset trend.
- Quality, after PIT-data acquisition.
- Defensive or low-vol overlays, properly classified.
- Carry only with crash-aware controls.
- Other genuinely orthogonal strategies.

Every sleeve needs its own gate and the combined book needs a separate gate.

### Phase J3 — Crypto momentum program

#### Preserve the frozen construction

The current construction includes:

- 14-day cross-sectional momentum.
- Long/short quintiles.
- Weekly rebalance.
- Funding-aware P&L.
- Volatility targeting.
- Maximum leverage.
- Explicit costs.

It remains shadow-only because historical significance did not clear the required bar.

#### Professional upgrades

- Automated monthly/daily data refresh.
- Data-freshness manifest.
- Venue and contract normalization.
- Delisted-symbol retention.
- Liquidity and capacity limits.
- Funding and basis stress.
- Separate forward and backtest ledgers.
- Effective-N monitoring.
- Cost and leverage sensitivity.
- Exchange outage scenarios.

#### Promotion gate

No promotion until:

- Adequate forward duration.
- Adequate effective N.
- Stable cost-adjusted expectancy.
- Drawdown within limits.
- No dependence on one market episode.
- Crypto exposure and hedge semantics are supported.

### Phase J4 — Crypto carry program

Crypto carry remains separate from momentum.

#### Required data model

- Spot instrument.
- Perpetual instrument.
- Venue.
- Funding timestamp.
- Basis.
- Margin requirement.
- Liquidation threshold.
- Borrow/transfer cost.
- Fees.
- Liquidity.
- Counterparty-risk classification.

#### Required evaluation

- Gross funding capture.
- Net capture after both legs.
- Rebalance and turnover.
- Spot-perp tracking error.
- Margin requirements.
- Basis blowout.
- Liquidation stress.
- Venue failure.
- Withdrawal suspension.
- Stablecoin depeg.
- Cross-venue settlement risk.
- Capacity.

The current lane explicitly acknowledges exchange-solvency and liquidation tails that a basic daily backtest cannot fully model.

#### Promotion gate

A carry strategy cannot pass from Sharpe alone. Tail and counterparty-risk evidence are mandatory.

### Phase J5 — Track B program

#### Pipeline

```text
SEC filing retrieval
→ document hash
→ PIT cutoff
→ blinding
→ scoring
→ score validation
→ cross-sectional construction
→ portfolio backtest
→ placebo
→ forward shadow record
```

#### Required lineage

Every score records:

- Filing accession.
- Document hash.
- Filing date.
- As-of date.
- Model cutoff.
- Scorer model and version.
- Prompt digest.
- Parameters.
- Rationale.
- Extracted spans.
- Score validation.
- Batch identity.

#### Professional upgrades

- Automated SEC ingestion.
- Explicit scoring queue.
- Idempotent scoring.
- Retry and rate-limit handling.
- Manual-review queue for malformed or extreme scores.
- Model-cutoff enforcement.
- Contamination tests.
- Rebalance-date tracking.
- Forward evaluation independent of the scoring worker.

The current Track B result is constrained primarily by forward time and valid rebalance dates, not merely by adding more names.

### Phase J6 — Hedge and exposure program

#### Expand exposure taxonomy

Current coverage must be expanded to include:

- FX currency buckets.
- Equity market beta.
- Equity sectors.
- Crypto beta.
- Stablecoin exposure.
- Exchange/venue exposure.
- Funding/basis exposure.
- Carry liquidation exposure.
- Correlation clusters.
- Shared liquidity risk.

#### Training targets

- Hedge effectiveness.
- Hedge cost.
- Residual exposure.
- Concentration.
- Hedge slippage.
- Drawdown reduction.
- Tail-risk reduction.
- Whether hedging improves after-cost expectancy.

#### Required comparison

```text
raw strategy
vs
hedged strategy
vs
combined portfolio
```

The existing hedge layer already reads each strategy’s own book and produces raw-vs-hedged forward records, but crypto is currently unsupported as a complete exposure class.

### Phase J7 — Learning, calibration and agent weights

#### Local-only by default

Keep these close to the journals and control kernel:

- Agent-weight synchronization.
- Confidence calibration.
- Risk calibration.
- Lightweight gate-model retraining.
- Drift detection.
- Alert routing.
- Retraining requests.

#### Promotion requirements

- Chronological holdout.
- Minimum class balance.
- Minimum sample size.
- Incumbent comparison.
- Calibration metrics.
- Risk-increasing prohibition where applicable.
- Atomic candidate write.
- Full local event trail.
- Rollback to prior incumbent.
- No in-place silent overwrite.

#### Future remote threshold

Move fitting remotely only when dataset size or computational complexity justifies it. The final import and load decision remains local.

---

## 14. Combined portfolio training

### Phase K — Build the AXIOM portfolio allocator

Every lane produces a standardized strategy-return and exposure contract.

```text
lane book
→ net return
→ volatility
→ drawdown
→ turnover
→ exposure
→ liquidity
→ tail indicators
```

The allocator evaluates:

- Equal-risk contribution.
- Inverse volatility.
- Correlation-penalized allocation.
- HRP across sleeves.
- Drawdown budgets.
- Tail-correlation limits.
- Lane capacity.
- Minimum and maximum allocation.
- Cash reserve.

#### Book-level evidence package

The combined portfolio produces its own package containing:

- Included strategy-package digests.
- Allocation policy.
- Return series.
- Correlation matrix.
- Exposure history.
- Drawdown.
- Turnover.
- Cost.
- Stress scenarios.
- Incremental contribution by sleeve.
- Portfolio gate.
- Verifier result.

No lane becomes capital-active solely from a standalone gate.

---

## 15. Promotion, deployment and rollback

### Phase L — Local promotion service

The service performs:

1. Resolve disposition head.
2. Verify package and artifact digest.
3. Confirm operator approval.
4. Confirm lane and feature compatibility.
5. Write candidate into a versioned local artifact store.
6. Run load smoke test.
7. Run dry inference or weight calculation.
8. Atomically change champion pointer.
9. Record promotion event.
10. Notify dashboard and Tier 7.
11. Preserve prior champion for rollback.

#### Deployment stages

```text
quarantine
→ offline replay
→ shadow
→ canary/paper
→ approved champion
→ retired
```

For lanes without an execution path, “champion” means the approved frozen research or shadow construction—not permission to trade.

#### Automatic rollback triggers

- Artifact mismatch.
- Feature-contract mismatch.
- Abnormal output.
- Cost explosion.
- Drawdown breach.
- Data staleness.
- Calibration collapse.
- Missing heartbeat.
- Disposition-chain inconsistency.
- Operator halt.
- Tier-7 safety event.

---

## 16. Observability and dashboard

### Phase M — Training and evidence cockpit

The AXIOM dashboard should expose:

#### Data

- Freshness by source.
- Snapshot versions.
- Quality failures.
- Tick/exposure-history accumulation.
- Missing partitions.

#### Jobs

- Submitted.
- Running.
- Failed.
- Completed.
- Resource usage.
- Cost.
- Manifest digest.
- Container digest.

#### Evidence

- Packages by lane.
- Current disposition.
- Gate results.
- Verification status.
- Negative results.
- Derived-from relationships.
- Current champion.
- Prior champion.

#### Forward monitoring

- Shadow duration.
- Forward Sharpe.
- Drawdown.
- Turnover.
- Cost.
- Exposure.
- Deviation from promotion baseline.
- Retirement warnings.

#### Controls

The dashboard may request local actions but must not bypass:

- Signature checks.
- Transition policy.
- Operator authority.
- LiveGate.
- Lane halts.
- Champion-pointer verification.

---

## 17. Reliability, security and recovery

### Phase N — Professional operational hardening

#### Failure injection

Test:

- Corrupt manifest.
- Missing partition.
- Wrong dataset hash.
- Forged signature.
- Missing fold.
- Duplicate worker output.
- Stale data.
- Worker termination.
- Partial upload.
- Event-chain fork.
- Index corruption.
- Registry outage.
- Object-store outage.
- Dashboard outage.
- Model-load failure.
- Local disk full.
- Clock skew.

#### Recovery

- Rebuild indexes from events.
- Recreate candidate cache from packages.
- Recover prior champion.
- Resume restartable jobs.
- Detect incomplete packages.
- Preserve evidence through process failure.
- Export signed backups.
- Periodically test restoration on a clean environment.

#### Key management

- Separate producer and local-authority keys.
- No private signing keys in repositories.
- Key IDs in signatures.
- Revocation list.
- Key rotation process.
- Offline recovery key.
- Audit of every signing operation.

#### Supply-chain security

- Pinned dependencies.
- Signed containers.
- SBOM.
- Vulnerability scanning.
- Restricted package sources.
- Reproducible builds.
- Secret scanning.
- Provenance records.

---

## 18. Testing standard

Every training family needs the following test coverage.

### Unit tests

- Features.
- Targets.
- Metrics.
- Serialization.
- Gates.
- Transitions.

### Property tests

- No future data affects past features.
- Weight constraints always hold.
- Risk multipliers remain bounded.
- Halt failures resolve safely.
- Package hashes change when content changes.

### Regression tests

- Existing frozen experiment reproduces.
- Current strategy construction remains unchanged.
- Backtest and shadow math match.
- Local and isolated-worker outputs agree.

### Integration tests

- Snapshot through package.
- Package through importer.
- Import through quarantine.
- Shadow through portfolio gate.
- Promotion through lane load.

### Adversarial tests

- Leakage.
- Survivorship.
- Data contamination.
- Trial inflation.
- Metric selection.
- Degenerate predictions.
- Cost underestimation.
- False liveness.
- Forged evidence.
- Unsafe capabilities.

No phase is complete from unit tests alone.

---

## 19. Execution order

The dependency order is:

1. Contract schemas.
2. Canonical hashing and signing.
3. Event store and disposition policy.
4. Evidence indexes and local importer.
5. Dataset manifests.
6. Risk-target vertical slice.
7. Data-platform normalization.
8. Canonical gated harness.
9. Hedge evaluation vertical slice.
10. Equity research workers.
11. Crypto momentum workers.
12. Crypto carry workers.
13. Track B ingestion/scoring/evaluation.
14. Execution-cost model.
15. Portfolio allocator and book gate.
16. Legacy FX benchmark containerization.
17. RL redesign after data prerequisites.
18. Dashboard evidence cockpit.
19. Failure-injection and recovery certification.
20. Full lane-by-lane migration to the contract.

Do not begin with Kubernetes, Ray, or a complete migration of the old full-training shell script.

The first success criterion is not “more models trained.” It is:

```text
One exact dataset
→ one signed job
→ one isolated run
→ one immutable package
→ one independently reproduced verdict
→ one locally governed disposition
```

---

## 20. Definition of complete

AXIOM reaches professional-grade training when all of the following are true.

### Reproducibility

- Every active artifact resolves to exact data, code, container and config digests.
- Every run can be replayed.
- Randomness is controlled and recorded.
- No model depends on mutable latest inputs.

The current W&B control plane automatically pulls `<head>_training_config:latest`; professional remote jobs should instead resolve that configuration before signing the job and record the exact artifact digest.

### Data integrity

- All active data sources have manifests.
- PIT and survivorship rules are tested.
- Forward evidence cannot be backfilled.
- Data freshness is visible.

### Methodology

- One canonical harness is used.
- Trial counts are recorded.
- Costs are realistic.
- OOS is untouched.
- Independent verification is mandatory.

### Governance

- Evidence is immutable.
- Disposition is append-only.
- Remote workers cannot promote.
- Local champion pointers are content-addressed.
- Operator approval is explicit.

### Multi-lane completeness

- FX, equity, crypto momentum, crypto carry and Track B use the same evidence system.
- Hedge, risk-target and learning systems use compatible packages.
- The brain loop can observe all lanes and evidence.
- Combined portfolio gating is mandatory.

### Operational safety

- Failed imports cannot affect incumbents.
- Rollback is tested.
- Index reconstruction is tested.
- Recovery from a clean machine is tested.
- Broker and state credentials never enter the research fabric.

### Honest performance

- Negative results remain visible.
- Curated and defensible results are distinguished.
- Shadow records are not described as live.
- Risk premia are not mislabeled as alpha.
- No lane is promoted because of one attractive headline metric.

---

## 21. Final architecture

```text
                         AXIOM LOCAL CONTROL KERNEL
┌─────────────────────────────────────────────────────────────────┐
│ Lane halts · LiveGate · operator authority · Tier 7             │
│ Headless learning · online calibration · brain registry         │
│ Local evidence importer · disposition ledger · champion store   │
│ Portfolio gate · execution · dashboard controls                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │ signed manifests
                               ▼
                       RESEARCH/TRAINING FABRIC
┌─────────────────────────────────────────────────────────────────┐
│ Data validation · feature generation · fold workers             │
│ FX benchmark · equity research · crypto momentum                │
│ crypto carry · Track B · hedge evaluation · risk targets        │
│ sleeve combinations · independent verifier                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ immutable evidence only
                               ▼
                       LOCAL EVIDENCE AUTHORITY
┌─────────────────────────────────────────────────────────────────┐
│ Verify hashes · replay metrics · enforce policy · quarantine    │
│ shadow forward · portfolio gate · operator approval             │
│ atomic champion pointer · rollback · retirement                 │
└─────────────────────────────────────────────────────────────────┘
```

### North-star rule

AXIOM does not trust a model because training completed. It trusts an exact artifact only after its data, construction, evaluation, capabilities, lineage and lifecycle have been independently proven—and even then, only local policy and the operator may permit a lane to load it.

The next concrete move is to implement the contract and risk-target vertical slice before touching distributed orchestration.
