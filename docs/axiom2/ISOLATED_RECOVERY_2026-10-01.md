# Axiom isolated recovery — 2026-10-01

Plan: docs/superpowers/plans/2026-10-01-axiom-2-execution-kernel.md

Tasks1–12 remain at verified efc9554622a98c4f91a59f79afb0f67f90a9453c; no research restart or genuine holdout consumption.
Original checkout <original-checkout> was read only. Independent no-hardlink clone and linked recovery worktree use a separate Git common directory and index. Recovery branch: codex/axiom-2-isolated-recovery.

Snapshot: before/after source HEAD, branch, tracked/untracked inventory, diff bytes, content bytes and modes matched on attempt1. One tracked file plus exactly22 task-scoped untracked files captured. Patch SHA256 c04b34a855ecb4b725d056dce875a0c0b4fe817429c11907fae8f129662d1155. Applied recovered contents verified against all23 file hashes. Capture manifest retained externally in recovery-snapshot/manifest.json. No original file was deleted or changed.

Ruling: independent clone before worktree creation — protects original Git metadata from concurrent writers. No source index or object store is shared.
Ruling: preserve recovered implementation rather than delete it under a generic TDD instruction — explicit recovery authorization requires exact preservation. New fixes use RED→GREEN.
Ruling: exact dependency reconciliation adds scikit-learn==1.7.2 immediately after lightgbm==4.7.0, preserving published PR59 correction. Connector remote head is parent-managed; local base intentionally remains efc9554.
Pre-flight: Task14 produces hypothetical signed journal receipts; Task15 cannot authenticate normalized provider inputs. Live qualification must remain false.
Pre-flight: Task16 recovered methods deny all actions; provider request identity and immutable capital artifacts are unavailable. Do not label denial stubs completed execution authority.
Pre-flight: Task17 requires actual deployment identity/secret/egress denials; shared development Mac cannot prove them. Task18 preparation cannot substitute for this or a human gate.

Initial recovered shadow gate: 8 inventory failures from stale execution source hashes; all other shadow tests passed. Exact results are recorded after reviewed hash repair.

Task13 LIVE UNVERIFIED. Deployment, capital, broker side effects and genuine holdout gates CLOSED.

## Reviewed recovery checkpoint

Three independent read-only review passes used journal_review, execution_review_two and execution_review_three. All reviewed Task14–18 code; each also reviewed the new offline authorization and normalized reconciliation additions. These were genuine separate agent contexts, not author self-review. Findings fixed: account-revision reuse by distinct decisions; observed-account/hypothetical-ledger mixing; unmatched equity claims; missing correction ancestor escaping fail-closed handling; conflicting superseded fill history disappearing during normalization. Receipt-time tampering is rejected by signed outer ShadowJournalReceipt and separate receipt/operator/producer trust stores. Exact scope is signed; offline nonces are scoped by operator key plus nonce text, not globally across key rotation.

RED→GREEN witnessed for account-revision reuse, mixed-mode MATCHED, correction-chain handling, missing corporate-action implementation, dangling correction ancestry, conflicting superseded history, and missing offline operator scope/nonce implementation. Valuation mismatch regression was added after the review-driven equation fix; do not claim that regression was observed RED. Recovered pre-existing tests were retained rather than represented as newly authored TDD.

Final recovery verification (existing Mac repo-suite.Gi7lHb virtualenv, bytecode and plugin autoload disabled, warnings errors, pytest cache disabled):
- Full approved dependency/safety gate: 1167 passed in19.86s.
- Both synthetic research integration suites:10 passed in1.78s.
- Standalone research isolation:PASS; research.zip SHA256 1ecf03d59e2219afa150958b62ed333dd46d7807c1eeda678ffaf468b30c3b74, unchanged from the checkpoint. research_kernel_complete=false and execution_enabled=false remain visible.
- Offline operator signatures/nonce tests:5 passed, including real four-process reservation race and signed receipt tampering.

Initial verification failures were stale recovered execution hashes and undeclared imports; fixed only exact reviewed digests/imports. The first integration command named a nonexistent test_verified_research_kernel_e2e.py and ran no tests; corrected to the actual test_research_pipeline_full_e2e.py before recording the10 passing tests.

## Acceptance boundaries

Task14 locally implemented: signed receipt-time authority history; retirement-safe replay; immutable intent/completion identities; atomic/fsynced storage under EvidenceStore lock; real multiprocess reservation/whole-engine races; crash before/after append; conservative terminal hypothetical fills; per-revision distinct-decision fence; zero broker transport. This is offline evidence only. Continuous live shadow qualification and external rollback checkpointing are absent.

Task15 locally implemented normalized cash/quantity/fee/turnover and equity-mark equations; native-ID deduplication; out-of-order fill matching; correction chains with raw-history conflict detection; explicit period-opening whole-share splits and declared dividend cash; settlement, incomplete observations, unexpected orders and unsupported/ambiguous actions fail closed. Corporate-action facts must agree exactly with journal expectations. Prices/fees are cents; valuation marks are microdollars with aggregate mark value floored to cents. Provider authenticity/freshness, order lifecycle/settlement ordering and price-adjustment semantics are not proven. Ledger is a normalized caller contract, not yet an authenticated reconstruction of production journal facts. Acceptance preregistration requires20 regular sessions and4 non-overlapping five-session cycles, zero forbidden effects and unresolved breaks; qualified() deliberately remains false without an authenticated live verifier.

Task16 partial: public review/submit/cancel deny before caller inspection and contain no transport/activation path. Offline operator declarations verify dedicated operator signatures, receipt-time trust, exact account/candidate/model/policy/intent/build/capability scope, time window and integer limits. Signed offline nonce reservations survive restart and concurrent process races. Production evidence/artifact resolution, risk signature role admission, integrated review/submission/cancel state machine, provider ambiguity matching and distributed gateway leases remain pending; these are not fulfilled by passing denial tests.

Task17 partial: exact source inventory excludes all execution code from research.zip; local denial diagnostics never authenticate operational readiness. Actual separately deployed identities, secret mounts, gateway/tool identity and provider-egress denials are absent. A shared developer Mac identity does not prove hostile-agent separation. No security settings, accounts, credentials or deployment were changed.

Task18 safe preparation: exact proposal field validation, bounded descriptive principal and UTC expiry, human-readable kill plan, default BLOCKED, zero broker review/approval/submission/cancellation. No real order is selected or requested. Genuine holdout, live shadow, production deployment, trading enablement and exact order each retain separate human gates.

Task13 protected official Robinhood Trading MCP is not exposed in this task. Parent discovery identifies https://agent.robinhood.com/mcp/trading and official https://robinhood.com/us/en/support/articles/agentic-trading-overview/ . This is intended provider route, not Crypto API. Required fixed reads: capabilities(), market_snapshot(instruments), account_snapshot(account_alias), order_observations(account_alias,cursor). Acceptance requires schema/capability pinning, dedicated-account allowlist/redaction, timestamp/session truth, complete cash/position/order pages, revocation/drift and denied side effects. No guessed endpoint or broad token forwarding is permitted. Provider schema and live read-only probe remain LIVE UNVERIFIED until authorized tools become exposed. External-agent trade approvals must be verified ON before any future human-gated execution proof; connection success alone is insufficient.

All financial/live/holdout/deployment gates remain closed. Original snapshot files and HEAD were read-only rechecked after the implementation and remained unchanged.

## Offline lifecycle continuation after recovery commit

Recovery checkpoint commit8489cf3ad73122621af6f0599b5fd84dbb5e6da5, tree68792ea3eb4bb77d167e385d1ff1839032e07247. First26-file Library transfer libfile_cbda7e878a58819188f25c92d08ac97f, SHA25664febfd8d453cf9b42204de790633df351535af40ff3b6078fb3ff5f3395514d. This records the intermediate checkpoint; final transfer contains all scoped changes from efc9554 and supersedes it for publication.

Task16 now additionally implements durable OFFLINE_PREPARATION lifecycle receipts for PROPOSED, APPROVAL_PENDING, REVIEWED, SUBMISSION_RESERVED, SUBMISSION_UNKNOWN, ACKNOWLEDGED, PARTIALLY_FILLED, FILLED, CANCEL_REQUESTED, CANCELED, EXPIRED and REJECTED. Separate configured GATEWAY/OBSERVER key roles and actor bindings are checked at signed receipt time. Proposal/approval/review/acknowledgement cannot report fills; cancellation requests cannot add fills; observed late fills remain visible after cancel acknowledgement. Exact scope-to-order fencing prevents renamed-identity retries; same-account ambiguity denies both new and previously queued proposals/reviews/reservations. Observer reconciliation and safe cancellation/expiry transitions remain possible.

Integrated reserve_proposal verifies canonical exact whole-share SHADOW intent batch bytes and dedicated signed RiskIntentReceipt identity/metadata/freshness/trust, binds selected intent/account/capability/principal to the signed offline operator declaration, persists the nonce before lifecycle progression, and revalidates both risk freshness/key trust and operator scope/time/key trust at final durable admission. The receipt retains signed risk/authorization envelopes and the nonce request digest. Crash/CAS failure may leave an offline nonce reserved without a proposal; exact retry recovers it without a second identity. These receipts attest configured gateway-verified offline ingress; replay independently verifies lifecycle signatures/transitions, not genuine production artifact/readiness truth or all upstream provenance against a separate resolver.

Ten lifecycle tests cover valid transitions, wrong role, CAS, monotonic fill bounds, renamed-identity and prequeued-order ambiguity, integrated signed risk/operator ingress, delayed risk becoming stale at admission, crash before/after commit and restart, key retirement, tampering and four-process append races. RED→GREEN observed for missing lifecycle/proposal implementation and both ambiguity defects. Delayed freshness regression was added after the reviewer-driven final-lock fix; no claim it ran RED before that fix.

Three independent lifecycle reviews (same three named reviewers in fresh review turns) found renamed-identity and prequeued-order ambiguity, plus admission-time freshness; all addressed. Final whole-branch read-only review by journal_review found no new blocking offline defect and confirmed actual remaining prerequisites. No independent reviewer edited code or executed tests. Exact source inventory expanded only for reviewed lifecycle.py; explicit import allowlist and source hash added; no research bundle expansion or wildcard exemption.

Isolated shadow artifact:37 audited source files, excluding research/training implementations; fresh -I process imports offline shadow/lifecycle/gateway from zip, rejects network/process side effects via audit hooks, proves public gateway review/submit/cancel all BLOCKED, and imports no research ranker, scanner, LightGBM, pandas, NumPy or broker SDK. Bundle SHA256 f7d0091f3fed438a8d847bd9f5a46bdca8d5a6bf1d8185cfb75bffd607e733f1. This is source/import proof, NOT Task17 OS/deployment isolation. Exact per-file artifact hashes retained outside repository.

Task17 harmless native identity observation: development parent UID501 and child UID501; separate operational identities not verified. No users, security settings, secret permissions, gateway deployments, credential scopes or network policies were changed. Actual required evidence: independently trusted verifier role/key; exact deployed build and gateway identity; separate research/LLM/development/healing versus gateway service identities; production-secret mount/credential reads denied from those non-gateway identities; direct broker/provider egress and broad MCP forwarding denied; exact side-effect-tool allowlist and unknown-tool denial; alternate-gateway/deployment-hook denial; revocation/kill/restart/fault probes. Tests or chmod on a developer-owned dummy file would not satisfy that operational requirement. Provisioning intended identities/policies is outside this authorized no-deployment/no-security-settings-change session and would need action-time approval.

Task16 remains PARTIAL, despite the completed offline controls: genuine artifact/readiness resolver, full authenticated RiskDecision input reconstruction at production ingress, immutable promoted/calibration pointer admission, live account/quote/kill/provider rechecks, real broker semantics/transport and distributed deployment fencing are absent. These software/integration omissions are stated separately from provider/operational blockers, not labeled complete. Current Task12 EquityOrderIntent and risk receipt contracts admit only SHADOW mode; this continuation does not silently broaden them to LIVE. A production adapter cannot be invented from unverified schema or simulated observations. Task13 provider exposure and actual deployment-boundary decisions are necessary before completing that production integration without guessing interfaces.

Task14 offline journal/fill/recovery controls are locally verified; continuous shadow account/portfolio evolution and live operational qualification remain unverified. Task15 normalized reconciliation controls are locally verified; live provider ordering/settlement/price-adjustment semantics, authenticated journal-derived ledger and20-session/four-cycle qualification remain partial. Task18 safe preparation remains implemented only; all separate human financial/deployment/holdout gates remain closed.

Final continuation verification: shadow suite161 passed in4.60s; full approved Axiom/evidence dependency/safety gate1177 passed in22.18s; standalone research isolation PASS with unchanged research.zip SHA2561ecf03d59e2219afa150958b62ed333dd46d7807c1eeda678ffaf468b30c3b74; isolated shadow artifact PASS with zero broker actions and operational_proof=false. No bare legacy suite run was made an Axiom prerequisite. All final code tests used the same Mac virtualenv, disabled bytecode/plugin autoload/cache, warnings-as-errors. Documentation-only updates followed that final code gate.

## Signed full risk reconstruction continuation

The isolated branch now reconstructs exact canonical policy, request, snapshot,
candidates and decision and reruns deterministic portfolio risk. Signed risk
service packets bind complete normalized account and capability observations.
Reconstructed offline proposal ingress compares the complete intent batch,
checks source TTLs and the session against the trusted receipt clock, binds the
operator's policy/model scope, and repeats packet validation under the durable
lifecycle lock. Receipts retain the packet, normalized inputs, signed intent
receipt, operator envelope and nonce request digest. The separate historical
risk reconstruction audit repeats risk receipt time/trust/scope and proposal
bindings; operator authorization/nonce verification remains a separate audit.

Validation: resolution tests 19 passed; full Axiom/evidence gate 1196 passed in
22.60 seconds with warnings treated as errors and plugin autoload disabled.
Research isolation PASS with unchanged SHA256
`1ecf03d59e2219afa150958b62ed333dd46d7807c1eeda678ffaf468b30c3b74`.
Offline execution artifact PASS, 38 sources, SHA256
`8613fbda2f5d4ae25006d9aecce2f6729733bfe63cd89c6295577229b928a8cd`;
no broker actions or capital authority and no operational isolation claim.
Three independent read-only reviewers found historical audit scope/freshness
omissions; these were fixed and all three final inspections found no additional
concrete defect in the offline risk reconstruction scope. Tests observed RED
for source freshness, missing full-batch verifier, missing lifecycle ingress and
altered proposal audit; the initial strict reconstruction draft was preserved,
and some review-driven regressions followed their corresponding fixes.

Continuous shadow portfolio evolution, actual artifact/readiness resolution,
production observation truth, provider request semantics, deployment fencing
and actual service identity isolation remain separate unfinished requirements.
No genuine holdout was accessed and every live/capital/deployment gate is closed.

## Continuous hypothetical shadow portfolio continuation

An authenticated `shadow-portfolio` stream now shares the existing EvidenceStore
lock/fsync and time-bounded shadow actor bindings. The ShadowEngine can consume
and produce ledger-backed account/snapshot state. Replay reconstructs whole-share
inventory, cash, fees, cohort entry, peak NAV and pending sale credits from exact
risk inputs and authenticated execution checkpoints. Sale proceeds settle only
at the next consecutive nonoverlapping modeled session; session/calendar drift
and skipped indices fail. Marking requires available fresh quotes; zero-valued,
delisted, corporate-action or external-flow changes need future explicit
normalized events rather than silently disappearing holdings. Baselines must
be clear and fully settled. No production source truth is asserted.

Exact replay is idempotent, changed/stale account inputs fail before new
reservations, and unfinished execution batches block new observations until
recovered. Final source/quote freshness and monotonic receipt clocks are checked
before durable portfolio writes. Checkpoints cannot precede their execution
receipts. Operator retirement is fenced against both execution and portfolio
history. Three independent reviewers identified session, clock, retirement,
market-binding and quote/checkpoint timing defects; these were fixed with
regressions, and all three final inspections found no further concrete blocker
in the inspected offline scope.

Validation: 16 continuous-ledger tests passed in 5.40 seconds, including
buy/restart/hold/maturity sell/T+1 settlement/next scheduled rebalance, crashes
mid-batch and after portfolio commit, tamper, negative freshness/session/policy
checks and a real four-process initializer race with exactly one commit.
All shadow tests: 196 passed in 9.97 seconds. Full Axiom/evidence gate:
1212 passed in 27.41 seconds, warnings as errors, no plugin autoload.
Research isolation PASS retains the original bundle hash. Standalone offline
execution import probe remains capital-disabled and is not operational proof.
`qualified=False` is permanent for this simulated ledger: compressed fixtures
cannot establish twenty live sessions or four actual rebalance cycles.

## Offline shared-store epoch fencing continuation

A signed `shadow-fences` stream now records bounded leases, monotonic epochs,
renewal/release and immutable hypothetical effect reservations. Late, expired,
released and superseded tokens fail. A successor cannot blindly reuse another
epoch's recorded effect. Fenced reconstructed lifecycle ingress reserves first,
then checks the exact token/reservation realm-owner-epoch pair and current lease
under the same EvidenceStore commit lock. Final admission cannot predate fence
or authority receipts. Operator retirement includes fence receipt ordering;
shared journal timing now also rejects future-created signatures at receipt.
Owner labels are signed shadow-service metadata, not authenticated deployment
nodes. This proves only single shared POSIX-store behavior; cross-host locks,
whole-store rollback protection and provider-side epoch/idempotency enforcement
remain unverified and cannot establish capital eligibility.

Validation: eight fencing tests pass, including four-process acquisition race,
crash after effect commit, restart, retirement, expiry, supersession and an
integrated lease-expiry denial at final lifecycle commit. Focused fencing,
journal and continuous-portfolio gate: 58 passed in 7.83 seconds. Full
Axiom/evidence gate: 1220 passed in 28.42 seconds with warnings as errors.
Research isolation PASS retains the original research bundle SHA256; the
standalone execution artifact remains disabled with no operational proof.
All three independent final read-only reviewers found no additional concrete
blocker in the inspected offline shared-store fencing scope after fixes.

## Actual artifact resolution and frozen offline admission

The execution profile now resolves content-addressed proposal/model/campaign,
registered trial/search, previously consumed result, signed classification,
promotion decision, economic calibration, risk policy and disabled build bytes.
It hashes opaque model/build bytes without executing them and never opens a
sealed dataset. Explicit service role/key bindings and current trust apply.
Campaign validation reconstructs configuration identity, actual summary metrics,
return-curve drawdown and holding-horizon purge from hash-bound CSV date facts.
Unsupported frame facts fail closed. Economic edge is recomputed from typed
already-after-cost results and a declared conservative buffer; this is not a
probability calibration or independent market/source truth claim.

Signed account session pointers are immutable and overlapping renamed sessions
are rejected. Sources are read through a bounded no-follow/nonblocking descriptor,
rejecting symlinks/nonregular files and wrong SHA256 bytes. Authenticated session
receipts use the existing store lock/fsync. Final current trust/session checks
occur after source reads. Resolved lifecycle ingress binds source classification,
exact session interval, model/promotion/economic/portfolio identities, calibrated
edge and operator candidate/model/policy/build scope, then combines complete risk
reconstruction, nonce reservation and persistent fencing. The resolver and
lifecycle share the exact final receipt timestamp. Durable receipts retain the
artifact pointer and resolution alongside the risk/auth/fence evidence.

Validation: 20 artifact tests passed in 3.04 seconds, including real combined
artifact/risk/operator/fence durable admission, negative mixed classification and
session bounds, false but consistently signed economic/drawdown/purge metadata,
retired/wrong-role keys, actual byte drift, FIFO rejection, immutable replay and
expiry during first/second resolution. Two freeze regressions were observed RED
before fixes; semantic fixes were review-driven and their regressions followed.
Full Axiom/evidence gate: 1240 passed in 30.75 seconds with warnings as errors and
plugin autoload disabled. Three independent read-only reviewers cleared the final
offline scope after fixes. Research isolation PASS retains bundle SHA256
`1ecf03d59e2219afa150958b62ed333dd46d7807c1eeda678ffaf468b30c3b74`.
The standalone execution ZIP explicitly imports the resolver, loads no training
or broker implementation and reports zero broker actions/capital disabled.

Task 14 durable hypothetical shadow/restart behavior is implemented and locally
verified. Task 16 offline artifact/risk/operator/fence admission is implemented
and locally verified. Full production Task 16 acceptance remains blocked: the
private Robinhood provider adapter, review/request identity mapping and
instrumented provider-contract fake require the official Task 13 authenticated
schema; quote/account/kill observations require authenticated live normalized
source facts. Genuine-holdout classification is service-signed metadata, not
independent provenance verification. Live twenty-session/four-cycle shadow
qualification, cross-host/provider fencing, external rollback checkpoints and
actual deployed identity/secret/tool/network denial proof remain unverified.
Every public review/submit/cancel route stays BLOCKED. No broker credential,
genuine holdout, capital, settings change, repository publication or deployment
was performed in this increment.

## Official read-only adapter continuation (2026-10-02)

The protected connector became available to the parent. A scoped typed adapter
was implemented from six exact official declared read contracts and the sanitized
parent verification packet, with no provider calls from this recovery worker.
See `ROBINHOOD_READONLY_CONTRACT_2026-10-02.md` for implemented semantics and
separate observed/synthetic/unverified evidence. Sixty focused synthetic tests
and 1300 full-gate tests pass; three read-only reviews and both artifact-isolation
probes pass. Task 13 remains incomplete because provider settlement/snapshot,
approval/calendar/advanced coverage and live branch/revocation guarantees are
unavailable/unverified. Connection success does not relax any execution gate.

## Offline service composition continuation (2026-10-02)

Fixed read-host composition and a persistent private fake execution service now
exercise actual artifact/risk/operator/fence admission through journal-derived
hypothetical reconciliation. See `OFFLINE_SERVICE_COMPOSITION_2026-10-02.md` for
scope, recovery behavior and remaining gates. This extends the safe offline
implementation; no live execution or connection/authentication grant is added.
