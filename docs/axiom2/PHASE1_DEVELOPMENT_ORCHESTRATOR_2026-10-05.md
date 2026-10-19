# Qualified Phase-1 development orchestrator

Status: implementation and synthetic verification, **not empirical qualification**.

This change composes the existing Axiom feature factory, forward-label helper,
walk-forward splitter, momentum baseline, LightGBM adapter and signed experiment
registry. P, PV, PR, PM and FULL each receive one configuration. Only FULL tests
the primary hypothesis. No holdout reader, evaluator, candidate-freeze, capital
or broker capability is added to the service request interface.

## Admission contract

The research service supplies an explicit mapping of snapshot IDs to dedicated,
immutable development bundles. Requests contain only `action` and `snapshot_id`.
There is no requester-selected filesystem path, import, proposal, authority, key,
feature matrix, label matrix, seed or hyperparameter input.

Each bundle contains these fixed names:

- `admission.json`: an existing-authority SignedEnvelope over `SourceAudit`, signed
  by a trusted, separately authorized INDEPENDENT_VERIFIER.
- `panel.json`: the exact raw development snapshot.
- `timestamps.txt`, `revisions.txt`, `membership.txt`, `sectors.txt`,
  `corporate_actions.txt`, `calendar.txt`, `liquidity.txt`, `identity.txt`:
  original evidence reviewed by the independent source auditor. These may contain
  source documents, publication/delivery records and audit explanations, but
  must not contain out-of-window market observations or genuine holdout data.

The audit binds SHA-256 identities for the panel and each evidence file. The
loader verifies actual bounded bytes, trusted signature and auditor role. It
rejects symlink files, unknown fields, missing evidence and unknown/holdout
classification **before opening the panel**. Synthetic development is denied by
service default; only test composition explicitly enables it.

**A trusted audit is a human/service trust boundary, not an automatic historical
truth detector.** The independent verifier must review actual source publication
and delivery timestamp semantics; PIT S&P 500 membership completeness, removed
and delisted identities; contemporaneous liquidity eligibility and sector
mapping; calendar completeness including holidays; opening-price coverage;
revision, split/dividend and total-return adjustment compatibility; and terminal
outcomes. A researcher cannot replace these with `verified=true`, self-signed
hashes, current-membership backfill or invented next-day timestamps. Ingestion
today is not historical availability proof. No auditor or signing key was
provisioned by this implementation.

`SourceAudit` pins the exact `2022-01-03`–`2024-06-28` development window,
calendar/adjustment identities, source availability basis, per-decision membership
basis, terminal policy and explicit contemporaneous liquidity policy. No new
market-data acquisition or period is authorized.

The panel schema is visible in the synthetic fixture
`tests/axiom2/research/test_phase1_development.py`:

- `sessions`: ordered actual exchange-session `open`, `close` and `decision`
  instants. Opening is 09:30 and decision is 09:25 America/New_York, including DST.
  Early closes are supported by the audited calendar.
- `bars`: `timestamp` (session close), stable `instrument_id`, `open`, `close`,
  `volume`, `available`, historical `sector`, `adjustment`, resolved `outcome`.
- `references`: close timestamp, availability, separate benchmark close and
  `benchmark_open` total-return valuation proxies, historical sector reference
  levels, adjustment identity. Future opening proxies are labels, never features.
- `eligibility`: one publication-eligible decision snapshot of the complete
  covered membership inventory, with stable instrument ID, sector, eligibility
  and reason. Future removal fields are not admitted. Excluded source identities
  remain in the inventory with their contemporaneous reasons.

The loader checks every declared timestamp and adjustment, historical feature
coverage and all required families. It constructs features from source history,
then applies membership separately at each decision through the existing PIT
selector. Future membership changes do not erase earlier cohorts. Labels enter
at that decision session's opening proxy, exit five actual exchange sessions
later, and rank only that decision's eligible names. Missing outcomes or missing
critical inputs abort the comparison; no outcome-dependent row dropping occurs.
The first six source sessions are a fixed feature warmup, and the last five
decisions have exits outside the permitted window and are excluded in advance.
These exclusions and membership reasons are in the frozen manifest and reports.

## Finite registration and execution

Admission produces one common paired row index and source snapshot, labels,
actual label-end timeline and five immutable folds. Each fold has exactly 252
training sessions after purging, 63 test sessions and five-session embargo.
Training exits must precede the first test decision, not merely its entry.
The existing splitter supplies calendar folds; the orchestrator filters by
actual exits and retains the last 252 eligible training sessions. Incomplete
folds or cohorts are rejected.

Before fitting, the service persists a signed manifest containing actual audited
source hashes, Python/package runtime identity, complete effective LightGBM
parameters, all five exact feature schemas/digests, labels, folds, source audit,
protocol, top-5 equal-weight five-session policy, unchanged development gate,
last-fold candidate-policy identity, all fixed cost scenarios, bootstrap protocol
and existing experiment/trial history. The five exact proposals are then admitted
to the existing registry **before the first fit**.

Parameters remain lambdarank, num_leaves=7, learning_rate=0.05, n_estimators=20,
random_state=0, n_jobs=1. There is no early-stop sweep, seed search or alternate
model. A durable exclusive campaign claim prevents silent refits after a crash
or successful run. Recovery is inspection of retained evidence, not automatic
retry. Previous failed experiments remain in the same registry.

Every fold has signed STARTED then SUCCEEDED/FAILED records. Success retains its
actual serialized model bytes and effective parameters; the existing registry
retains each trial's status and result digest. Campaign computation failure marks
the experiment failed and stops fitting subsequent arms; their registered rows
are reported NOT_RUN. This does not hide failed trials or silently run substitutes.
A failed metric gate remains an explicit signed result and cannot authorize a
candidate through the unchanged verified-freeze gate.

The ranker now binds session IC, fold IC, paired book rows and fold-artifact hashes
in separately signed companion evidence bound to the original campaign and signed
trial result. The existing promotion/execution campaign wire contract is unchanged.
The report verifier checks historical attempts,
registered trial identity, complete fold chains, artifacts and diagnostic digests.
This is source-level research composition, not deployed operational isolation.

## Reporting and diagnostics

Every computed arm reports the unchanged IC >= 0.01, net excess return >= 0,
excess-return drawdown >= -0.15 and positive-IC-fold fraction >= 0.60 gates. The
unchanged gate identity is
`3117d82f92a2ba707fd4d155af4a6391e364cbcc555dc17fa8841caae161eb0b`.

Reports include per-session and fold IC; aligned momentum baseline; raw-return
weight-drift turnover; fixed equal-weight concentration; membership/coverage
exclusions; and separate raw wealth and excess-return wealth/drawdown curves at
10, 25 and 50 bps. The 10-bps round-trip convention is a development-comparability
assumption, not verified broker costs or live account drawdown. Model predictions
are reused across cost scenarios. No additional model fits are made.

FULL-minus-P and FULL-minus-momentum use paired, fold-stratified circular block
resampling: 9,999 replicates, seed 0, four rebalance observations per block,
97.5% one-sided marginal lower bounds. Blocks cannot cross fold boundaries.
Before fitting, fixed synthetic zero/positive controls and 40 seed-11 IID-null
series calibrate the diagnostic; more than four null rejections blocks execution.
This small calibration is not universal coverage under arbitrary dependence.
The bounds neither weaken the development gate nor repair source leakage.
`primary_hypothesis_pass` additionally requires both lower bounds above zero and
FULL passing all unchanged gates. Secondary arms cannot replace FULL.

## Service and CLI use

The configured research service must supply its **existing** registry and trusted
source-verifier identities. It must derive `code_commit` from its clean, verified
checkout and preserve the owner's complete research history; a fresh empty
registry must not be represented as the historical registry. The implementation
checks actual source bytes against the reviewed research artifact inventory.

The service embeds either `handle_development_request(...)` or
`serve_development_stdio(reader, writer, registry=existing_registry,
bundles=configured_allowlist, code_commit=verified_checkout_commit)`.
Stdio handles one bounded request on an already-established service pipe. A native
service may alternatively route the same bounded handler over its owner-private
Unix socket. There is no registry/key bootstrapping in the CLI.

Client invocation:

```sh
python scripts/axiom2_run_phase1_campaign.py \
  --service-socket /configured/owner-private/research.sock \
  --snapshot-id qualified-development-snapshot --action preflight
```

Replace `preflight` with `run` only after successful admission. `--service-stdio`
is supported for a parent-managed pipe: the client emits one request line,
consumes one service reply on stdin and emits that reply. The signed report is
content-addressed under the existing evidence store's `development/` directory;
`verify_comparison_report` revalidates it. No native endpoint or real signing
context has been deployed by this change. Stdio handoff is tested with real
subprocess pipes; Unix-socket integration is not claimed in this restricted
executor.

## Repository data decision

The tracked repository inventory contains price/fundamental and universe files,
but no qualifying `admission.json`/`panel.json` bundle or independently signed
historical publication/delivery, volume, sector, adjustment and calendar audit
for this protocol. The existing corrected-replay and next-experiment documents
also explicitly record missing volume/sector inputs and unresolved availability.
Data admission therefore remains **BLOCKED**. No empirical five-arm campaign was
registered or fitted. No genuine holdout market observations were opened. The
verification suite uses sacrificial synthetic fixtures only.

The next admissible input is an independently audited development-only bundle
and the existing service authority/history configuration. Do not fabricate these
to make an empirical table appear complete. Synthetic passing metrics are only
controls; they establish no market edge or architecture-level profitability.

## Verification at the implementation snapshot

The full scoped command completed with **1350 passed in 182.81s**, exit 0:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests/axiom2 tests/evidence/equity_research \
  tests/test_evidence_contracts.py tests/test_evidence_store.py \
  tests/test_equity_research_evidence_slice.py \
  tests/test_equity_research_evidence_worker_no_authority.py -q --tb=short
```

The 31 new controls are included in this count. The final standalone artifact
isolation check returned PASS, execution_enabled=false and
research_kernel_complete=false. Its bundle SHA-256 is `483414965c480191e4b1491d081cce64947d198c8d7eedd47da5887c32f45bb0`.
`git diff --check` passed. This is the required scoped Axiom/evidence gate, not
a repository-wide legacy-suite result.

Review caught and corrected exact post-purge training size, decision-eligible
label ranks, diagnostic/report identity binding, compatibility with the existing
promotion/execution wire contract, and the isolation fixture's explicit inventory.
Admission, failure retention and report poisoning controls now pass through the
existing kernel stages. No independent reviewer or live-data claim is made.

`PHASE1_SYNTHETIC_CONTROL_2026-10-05.json` records the actual software-control
report: every arm computed and passed the unchanged metric gate, but FULL did
not improve over paired P, so primary_hypothesis_pass=false. These invented
fixture observations are not market results. Source and package identities,
verification limits and log hash are in `PHASE1_VERIFICATION_2026-10-05.json`.
