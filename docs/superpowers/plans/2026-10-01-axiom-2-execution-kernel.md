# Axiom 2.0 Execution Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline on the connected Mac), or subagent-driven-development only when genuine agents are available. Steps use checkbox syntax. Do not restart completed Tasks 1-10.

**Goal:** Extend the completed research kernel through deterministic equity portfolio/risk, broker-neutral contracts, read-only Robinhood observations, shadow execution, reconciliation, and a single gated execution authority.

**Architecture:** Axiom retains exactly four authorities: evidence integrity, promotion, portfolio/risk, and execution/reconciliation. Research agents remain replaceable and cannot acquire production order capability. A signed promotion is evidence, never permission to trade.

**Tech Stack:** Python 3.11, pytest, immutable records, integer money/basis-point arithmetic, existing canonical JSON / Ed25519 / EvidenceStore primitives. No new training stack, broker SDK, database, or paid service is required by Task 11.

**Spec:** `docs/superpowers/specs/2026-09-29-axiom-2-equities-first-design.md`.
**Predecessor:** `docs/superpowers/plans/2026-09-29-axiom-2-research-kernel.md`, completed through Task 10.
**Checkpoint:** `a348bed38c45e6bba066d04820b22ba841c2e057`; Mac worktree `~/ml_engine-task-5`.

## Global Constraints

- US listed liquid equities and highly liquid broad/sector ETFs; long-only; regular market hours.
- Preserve the preregistered top K = 5, equal weighting, five-trading-session horizon, five-session rebalance interval, and non-overlapping cohorts. No refill from rank six, silent weight optimization, or extra model search.
- ModelResearchAdapter remains the model boundary; LightGBM remains the only authorized Phase-1 adapter. The execution-side kernel must not import training or model-provider implementations.
- No research, LLM, development, or healing process receives a production credential, unrestricted MCP client, execution signing key, or alternative order path.
- No genuine final holdout is opened by this plan automatically. Its use requires separate explicit approval. Synthetic and development evidence remain visibly non-capital evidence.
- No real order, trade-approval request, cancel, account mutation, merge, or production deployment is authorized by this planning/implementation session.
- Preserve unrelated work and legacy FX/OANDA reproducibility. Do not reset, amend, force-push, or make unrelated legacy failures Axiom prerequisites.
- Every task follows RED -> GREEN -> three review passes -> fixes -> focused retest -> full Axiom dependency/safety gate -> isolated artifact proof -> clean commit.

## Review Focus

1. A negative test without a signer can pass because signing is absent, not because its intended risk/evidence rule works. Exercise negative promotion cases with real ephemeral signing authority (Task 11).
2. Caller-supplied hashes, booleans, account aliases, and valid signatures are not proof of truth or actor authority. Resolve artifacts, roles, freshness, account state, and exact content at privileged ingress (Tasks 13, 16, 17).
3. Decimal context, rounding, duplicate instruments, stale/future clocks, unknown corporate actions, pending sells, and unsettled cash must not create buying power (Tasks 11, 12, 15).
4. A crash, late fill, cancel/fill race, truncated pagination, stale lease, or empty eventual-consistency response must not produce a duplicate order or falsely reconciled account (Tasks 14-16).
5. A Python wrapper is not a hostile-code sandbox. Prove credential/filesystem/egress denial under the actual deployment identities before any capital eligibility (Task 17).

## Authority and privilege map

| Component | Accepts | May produce | Must never receive |
|---|---|---|---|
| Research/model adapter | registered research inputs | candidates, scores, evidence | broker credentials, order tools, production signer |
| Evidence/promotion | verified immutable evidence | signed decisions, references | broker order capability |
| Portfolio/risk | authenticated model/evidence identity, policy, normalized observations | desired positions; later signed bounded intents | broker session or submission client |
| Read-only broker service | explicit dedicated-account allowlist | minimized immutable market/account snapshots | downstream exposure of bearer tokens or raw account identifiers |
| Shadow execution | non-capital risk outputs and observation snapshots | explicitly hypothetical fills, costs, journal | submit/review/cancel transport |
| Execution/reconciliation gateway | authenticated risk intent plus explicit operator authorization | broker-reviewed/submitted orders, reconciled receipts | arbitrary agent instructions or model callbacks |

Only the eventual gateway may hold broker-facing write capability. It is a separate process/identity from research, portfolio, and healing. Where the provider cannot mint read-only scopes, a trusted credential-owning read service exposes only fixed read methods and redacted projections; giving a broad token to research and setting `read_only=True` is forbidden. Account discovery and full raw responses stay inside that boundary.

## Evidence and deterministic transitions

Every downstream receipt binds schema/version, candidate/model artifact, promotion decision, campaign portfolio policy, risk policy, signal/universe/calendar/quote/account snapshots, decision time, account alias, previous state, and input/output hashes. Hashes bind bytes, not their truth. Production ingress verifies signatures, role/key trust at receipt, actual artifact bytes, frozen registry identity, genuine holdout classification, and current policy before accepting these references.

Task 11 outputs are **shadow-only plans**, not executable orders. Integer cents, microdollar quotes, and integer basis points avoid ambient floating/Decimal context. Thresholds are explicit policy inputs, not inferred from development results. An explicit deployment capital sleeve may retain cash while keeping the selected five names equal within the sleeve. Its risk/cost overlays must be versioned before shadow evaluation; the old development return must not be represented as the performance of the new overlay. A material strategy/selection change requires a new registered experiment.

Portfolio states: `READY_SHADOW`, `REJECTED`, `HOLD`, `HALTED`, `RECONCILE`, `EXIT_REQUIRED`. Only READY_SHADOW contains new target positions. Empty targets on another state mean **no executable instruction**, not liquidation. Unknown positions/orders -> RECONCILE; kill -> HALTED; known held cohort -> HOLD until maturity; maturity/risk-off/exposure breach -> EXIT_REQUIRED. Exits are subsequently revalidated and routed through the same gateway, never a bypass. No replacement cohort may enter until a fresh reconciled flat snapshot and a scheduled rebalance session.

Order states: `INTENT_RECORDED -> RISK_ACCEPTED -> REVIEWED -> SUBMISSION_RESERVED -> ACKNOWLEDGED -> PARTIALLY_FILLED -> FILLED`; rejection/expiry/cancel are explicit terminal alternatives. A transport ambiguity moves to `SUBMISSION_UNKNOWN`, not back to READY. Cancellation is a request, never evidence that an order stopped filling. Restarts reconstruct state from durable receipts.

Idempotency identity binds account alias, strategy/decision digest, instrument, side, revision, and mode. Persist a unique reservation and full request digest before side effects. Exact replay returns prior state; changed content under the same identity is rejected. Persist broker IDs and fills by broker-native event identity. Atomic single-writer fencing and post-restart reconciliation prevent concurrent execution. Do not claim end-to-end exactly-once unless provider semantics prove it. If an uncertain request cannot be matched authoritatively, halt new exposure and require resolution; an empty query is not proof of absence.

## Task 11: Deterministic equity portfolio/risk core (no order capability)

**Files:** Create `src/axiom2/portfolio/{__init__,contracts,authority}.py`, `tests/axiom2/portfolio/{test_portfolio_authority,test_boundary}.py`, `tests/axiom2/promotion/test_signed_negative_cases.py`, and `docs/axiom2/EXECUTION_KERNEL_VERIFICATION.md`. Narrow safety correction: `src/axiom2/promotion/authority.py` and its exact source digest in `config/axiom2/research_boundary.json`.

**Interfaces:** Frozen `RiskPolicy`, `Position`, `PortfolioSnapshot`, `RankedOpportunity`, `PortfolioRequest`, `TargetPosition`, and `RiskDecision`; `evaluate_portfolio(request: PortfolioRequest, policy: RiskPolicy) -> RiskDecision`; `risk_policy_digest(policy) -> str`.

RiskPolicy explicitly supplies allocation, per-name/sector/correlation-group/gross caps, cash reserve, one-way turnover cap, liquidity participation, after-cost edge floor, spread/cost buffers, drawdown cap, freshness limits, and rebalance anchor. PortfolioSnapshot contains NAV, total/settled/reserved cash, complete valued positions, open/unresolved-order state, reconciliation status, peak NAV, and active-cohort session. Opportunities carry exact ranks 1-5, equity/ETF identity, sector/correlation group, conservative economic evidence, bid/ask, liquidity, eligibility/corporate-action state, and timestamps. PortfolioRequest binds all evidence digests and a sourced regular-session interval/index. These are normalized inputs, not live attestations; Task 16 must resolve them authoritatively.

- [x] Reproduce signed NaN/-Inf baseline promotion using synthetic fixtures; add signer-present negatives for every metric and existing rejection case, with specific reasons and a valid positive control.
- [x] Run `python -m pytest tests/axiom2/promotion/test_signed_negative_cases.py -q -W error`; require intended failures before the narrow fix. No real holdout is involved.
- [x] Reject non-finite baseline values at their source without changing thresholds; preserve deterministic signed rejection. Update only the changed reviewed source digest. Rerun focused promotion tests and the full gate.
- [x] Write portfolio tests before implementation. Assert five equal targets within the declared sleeve, cash residuals after conservative costs, integer-safe cap comparisons, deterministic permutation/replay, explicit economic evidence (ranking is not expected return), and policy hash binding.
- [x] Add negatives for floats/bools used as integer money, NaN/Inf, mutable containers, duplicate ranks/instruments, bad hashes, stale/future quotes/snapshot/signals/metadata, unknown membership, corporate actions, crossed/wide quotes, insufficient settled cash, reserved cash, and each concentration/liquidity/turnover/edge cap.
- [x] Pin schedule/state tests: no new entry outside regular session; no overlapping cohort; maturity requires exit/reconciliation; open or ambiguous orders block entry; risk-off and kill cannot create targets; existing positions are explicitly referred for management, not forgotten.
- [x] Run `python -m pytest tests/axiom2/portfolio -q -W error`; require RED for missing functionality, not setup/import mistakes.
- [x] Implement a pure deterministic evaluator. Sort semantic collections before hashing; floor equal target cents and retain remainder as cash; round estimated cash costs upward. Never optimize around a breached cap. Every result has `execution_enabled=False` and `capital_authorized=False`.
- [x] Prove fresh-process imports never load research/model/legacy/broker code or perform network/process side effects. Research bundle excludes portfolio/execution modules. Prove malicious mutations are revalidated at entry; do not claim Python records sandbox hostile code.
- [x] Run three review passes and fix findings; execute a seeded portfolio invariant probe on the Mac; rerun all focused/dependency/isolation gates. Record exact environment, commands, results and limitations. Commit `feat: add deterministic Axiom equity portfolio risk core`.

## Research-derived execution constraints

Deep research and external-drive archaeology do not change the execution authority model. They add these explicit constraints:

- Risk intents must consume evidence-bound expected-net-edge values; rank position or arbitrary model score is insufficient.
- Shadow receipts must bind the exact promoted ranker/model identity and, when introduced, the exact promoted calibration/edge-estimator identity.
- Reconciliation/attribution must separate alpha-model error, calibration error, portfolio-overlay effects, modeled costs, and observed execution differences.
- Production admission must resolve the exact promoted model and any required calibration artifact; missing/mismatched artifacts fail closed.
- Execution tasks may not silently introduce portfolio optimization, volatility scaling, sector/beta hedging, ensembles, RL sizing, or other strategy changes that were not separately research-registered.
- Legacy FX/OANDA LightGBM momentum/risk trainers, scanner voting, W&B auto-training, and RL sizing remain outside Axiom 2.0 authority.

## Task 12: Broker-neutral equity order and risk-intent contracts

**Files:** Create `src/axiom2/contracts/equity_orders.py`, `src/axiom2/portfolio/intents.py`, and corresponding contract/portfolio tests.
**Interfaces:** `EquityOrderIntent`, `BrokerCapabilities`, `RiskIntentReceipt`, `AccountObservationRef`; `build_order_intents(decision, snapshot, capabilities, *, mode) -> tuple[EquityOrderIntent, ...]`.

- [ ] RED: reject short/oversell, negative/non-finite quantity, simultaneous quantity and notional, unsupported order type/TIF, invalid tick/lot, after-hours, expired evidence, stale account revision, wrong policy/account, unsigned or altered risk receipt, and cross-mode replay. Recompute `RiskDecision.decision_digest` from canonical contents at ingress before signature/role checks; a frozen Python object or caller-supplied digest is not integrity evidence.
- [ ] Implement explicit equity-native side/type/TIF/quantity-or-notional contracts. SL/TP are optional policy concepts, never mandatory FX parameters. Broker IDs stay adapter-private. Advertised support does not automatically enable a feature: initial execution profile is whole-share regular-session DAY limit orders; other forms remain structurally representable but denied unless separately enabled and tested.
- [ ] Round buys down conservatively, recheck actual projected exposure/cash including costs, reserve cash/inventory, and never fund a buy from an unconfirmed sale. EXIT_REQUIRED may generate only verified risk-reducing intents; kill still denies submission. No submit client exists here.
- [ ] Verify deterministic idempotency and immutable signing using ephemeral authority fixtures; run reviews, full gate and isolation proof; commit.

## Task 13: Robinhood minimum-data read-only adapter

**Files:** Create `src/axiom2/brokers/{__init__,contracts,robinhood_readonly}.py`, `scripts/axiom2_verify_readonly.py`, translation/privilege tests.
**Interfaces:** `ReadOnlyBrokerPort.capabilities()`, `.market_snapshot(instruments)`, `.account_snapshot(account_alias)`, `.order_observations(account_alias, cursor)`; immutable `CapabilitySnapshot`, `MarketSnapshot`, `AccountSnapshot`, `OrderObservationPage`.

- [ ] RED with recorded/minimized fixtures: wrong account, missing/changed schema/version/capability, missing timestamps, unknown corporate actions, stale/closed-session quote, pagination gaps, redaction failures, missing tradability, unavailable read scope, and any attempted side-effect tool.
- [ ] Discover the actual official provider schemas read-only. Pin supported tool semantics and hashes; do not copy guessed SDK endpoints. Expose fixed typed read methods only. Deny arbitrary tool forwarding, trade review/placement/cancellation, watchlist/account mutations, options and crypto.
- [ ] Keep raw account IDs and credentials solely in the trusted service. Log aliases/reason codes, not raw provider errors. Normalize zero-valued/delisted positions, reserved inventory, unsettled cash, and signed external cash flows explicitly so reconciliation and flow-adjusted drawdown remain possible. Unsupported semantics are explicit blocked prerequisites, never a fallback to unofficial endpoints.
- [ ] Run a live **read-only** probe only through an authorized, protected connection: account allowlist/redaction, quotes/session/timezones, cash/positions/orders completeness, revocation, and capability drift. No default approval-setting assumption; record verified setting. If a protected connection is unavailable, report LIVE UNVERIFIED and leave later capital gates closed.
- [ ] Review/fix, full gate, source artifact and read-only call audit; commit. No broker side effects.

## Task 14: Durable shadow execution and recovery

**Files:** Create `src/axiom2/shadow/{__init__,engine,fills}.py`, `src/axiom2/execution/journal.py`, typed execution evidence extension under `src/evidence/`, and shadow/journal integration tests.
**Interfaces:** `ShadowEngine.step(risk_decision, market_snapshot, account_snapshot) -> ShadowReceipt`; `ExecutionJournal.reserve(intent)`, `.append(event, expected_head)`, `.replay()`; typed immutable `ExecutionEvent`.

- [ ] RED: duplicate decision, changed replay, concurrent reservation, crash before/after journal append, restart, stale observation, closed market, crossed quotes, partial/unfilled limit, halted instrument, volume cap, and hidden submit client.
- [ ] Extend the existing EvidenceStore durability/locking/signing primitives without a parallel authoritative database or research-holdout mutation. Add time-bounded authority-binding history so routine key/role retirement blocks new privileged events without making previously valid signed history unreadable; registration/revocation times must be authenticated rather than inferred from a current allowlist. Atomic reservations and fsync precede state progression. Source streams remain separate by schema/mode.
- [ ] Implement deterministic conservative hypothetical fill rules, finite liquidity, spread/fees/slippage, no fill before quote availability, and explicit uncertainty. A quote is not proof a real order would fill. Reconstruct shadow state on restart.
- [ ] Exercise real filesystem and multiprocess races plus zero-submit audited shadow E2E; review/fix; full gate/isolation proof; commit.

## Task 15: Fill, cost, cash, position and corporate-action reconciliation

**Files:** Create `src/axiom2/execution/{reconciliation,costs}.py` and reconciliation/property/race tests.
**Interfaces:** `reconcile(journal_state, broker_observations, account_snapshot) -> ReconciliationDecision`; `CostReceipt`, `ReconciliationDecision(MATCHED|PENDING|HALTED)`.

- [ ] RED: duplicate/out-of-order/corrected fills, partial fill then cancel, late fill after cancel, split/dividend, account mismatch, unknown manual order, timeout, incomplete pages, lost IDs, quantity/fee discrepancy, unsettled cash, and incompatible price-adjustment semantics.
- [ ] Deduplicate by authoritative broker event identity, not timestamps alone; recompute settled cash/inventory and actual turnover/costs. Normalize deposits/withdrawals separately from P&L and maintain a flow-adjusted equity peak. Preserve zero-valued/delisted holdings until authoritative closure. Do not silently replace missing with zero. Separate hypothetical outcomes from observed broker facts. Unresolved state blocks new exposure while reconciliation continues.
- [ ] Register shadow operational acceptance BEFORE accumulation: at least 20 regular sessions and four completed five-session cycles, zero forbidden side effects, zero unresolved reconciliation breaks, complete audited transitions, and cost/risk tolerances bound to the frozen policy. These are engineering minimums, not evidence of profitable edge; do not shorten them after seeing results.
- [ ] Reconcile live read-only observations against shadow assumptions, explicitly marking unobservable fills/costs. Review/fix, full gate/isolation, commit; do not claim calendar-time shadow gates from accelerated fixtures.

## Task 16: Sole execution authority (implemented disabled)

**Files:** Create `src/axiom2/execution/{authority,admission,authorization}.py`, isolated `src/axiom2/brokers/robinhood_execution.py`, negative ingress and crash/race tests.
**Interfaces:** `ExecutionAuthority.review(intent, authorization)`, `.submit(reviewed_intent, authorization)`, `.cancel(order_identity, authorization)`; `CapitalAuthorization`, `AdmissionReceipt`. The broker transport is private to this gateway.

- [ ] RED: signed-but-wrong-role, forged promotion status, mutated frozen model/policy, unresolved evidence hash, reused approval nonce, expired authorization, wrong account, mixed synthetic/live evidence, changed capability/approval setting, stale risk snapshot, active kill, double reservation, split brain, and ambiguous response retry.
- [ ] Verify actual candidate/registry/promotion/economic/shadow artifacts and signatures with explicit role trust and time-bounded authority validity at each receipt. Current revocation must deny new actions while historical receipts remain verifiable only if their actor/key/role binding was valid at receipt time. The current promotion summary/hash alone is not enough. Reject a missing genuine holdout result; never open it here. Freeze the promoted pointer for the session; no mid-session agent mutation.
- [ ] Verify a separate signed operator authorization binding account, candidate, risk policy, capital/order limits, time window, gateway build, capability digest and single-use nonce. No environment flag, `PROMOTED` string, research signer, or agent approval substitutes for it.
- [ ] Persist and fence reservations, perform broker review, recheck quote/account/kill/authorization immediately before side effects, and reconcile ambiguous outcomes without blind retries. Recompute and authenticate the complete RiskDecision and signed intent at privileged ingress. Model `PROPOSED`, `APPROVAL_PENDING`, `ACKNOWLEDGED`, partial/final fills and expiry as distinct evidence states; an approval or proposal is never a fill. Require demonstrated provider request-identity semantics; otherwise refuse capital execution.
- [ ] Test with an instrumented fake transport only. Real submission is disabled in shipped defaults. No credential acquisition, deployment or real order is part of this task. Review/fix, full gate, isolated execution artifact proof; commit.

## Task 17: Negative production-authority and deployment-boundary proof

**Files:** Create `tests/axiom2/security/`, `scripts/axiom2_verify_execution_boundary.py`, and `docs/axiom2/EXECUTION_PRIVILEGE_MATRIX.md`.
**Interfaces:** signed `ExecutionBoundaryReceipt` tied to exact build, service identities, secret mounts, tool allowlists, network policy and gateway configuration.

- [ ] RED: research/LLM/development/healing actors attempt gateway calls, forged roles, direct provider calls, credential reads, broad MCP forwarding, accidental deployment hook import, stale authorization, alternate gateway, and bypass via new tool names.
- [ ] Enforce separate OS/service identities and credential/egress isolation at the actual deployment boundary. Research has no production secrets or broker egress. The sole gateway cannot execute arbitrary research/LLM code. Pin capabilities and default-deny all tools outside the exact allowlist.
- [ ] Prove real cross-process/filesystem/network denials in the intended environment; source scans and monkeypatch tests alone are insufficient. One shared administrator Mac account is not proof of hostile-agent isolation. Keep deployment/capital BLOCKED until operational isolation is demonstrated.
- [ ] Run fault injection, revocation/kill rehearsal, restart reconciliation and all three reviews. Save denial evidence without secrets; full gates and clean commit. A receipt is invalidated by material build/policy/capability changes.

## Task 18: Tightly controlled broker execution proof -- HARD HUMAN GATE

**Files:** Create `scripts/axiom2_controlled_execution_proof.py`, `docs/axiom2/CONTROLLED_EXECUTION_RUNBOOK.md`, and proof-runner denial tests.
**Interfaces:** `ControlledExecutionProof` links operator authorization, readiness receipts, reviewed intent, submission identity, broker fills/costs, reconciliation and final kill/revocation receipt.

- [ ] RED: by default the runner refuses without ALL evidence, genuine holdout promotion, completed live shadow/reconciliation, Task 17 operational denial proof, approved build/deployment, allowlisted account, approval setting ON, and explicit bounded operator authorization. Fixture/development evidence never satisfies these checks.
- [ ] Implement a dry-run preparation path that produces a human-readable exact order/account/capital cap/expiry and rollback/kill plan, without creating a broker review or approval request.
- [ ] STOP for separate approvals before genuine holdout consumption, production deployment, enabling real trading, and the exact real broker order. Prior coding authorization or "finish the plan" is not that approval.
- [ ] Only after approval, use the sole gateway for a specifically bounded regular-session proof. No size/symbol/order is preselected by this plan. Retain broker review, acknowledgement, actual fills/fees, cancellation races, final holdings/cash, and reconciliation; then disable/revoke proof authorization.
- [ ] Unknown state or any unmet requirement aborts and blocks new exposure. Do not auto-scale capital after a successful proof. Review/fix, complete evidence and clean checkpoint; no autonomous merge/deploy.

## Verification commands and reporting

The existing dependency gate remains mandatory (warnings treated as errors):

```sh
python -m pytest tests/axiom2 tests/evidence/equity_research \
  tests/test_evidence_contracts.py tests/test_evidence_store.py \
  tests/test_equity_research_evidence_slice.py \
  tests/test_equity_research_evidence_worker_no_authority.py -q -W error
python -m pytest tests/axiom2/integration/test_research_kernel_e2e.py -q -W error
python scripts/axiom2_verify_isolation.py --bundle <external-review-directory>/research.zip
```

Set `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `PYTHONWARNINGS=error`. Use the existing complete Mac environment at `/Users/mirelagherasim/.local/share/axiom/repo-suite.Gi7lHb/venv/bin/python`; the older Task 5 review environment lacks pandas. Do not install/reconfigure dependencies merely to rediscover this environment.

Each checkpoint distinguishes IMPLEMENTED, LOCALLY VERIFIED, LIVE READ-ONLY VERIFIED, SHADOW QUALIFIED, CAPITAL BLOCKED. Record exact commands/counts and reviewer scope; never invent independent subagents. Existing research receipt lists Tasks 1-10 and `execution_enabled=false`; leave its conservative `research_kernel_complete=false` metadata visible rather than treating it as capital authorization. New execution completion needs its own scoped receipt, not a broadened research artifact.

Fresh predecessor verification on 2026-10-01: 795 passed in 10.37s; dedicated E2E 2 passed in 0.19s; standalone research artifact PASS; clean checkout. A signed synthetic baseline probe subsequently exposed PROMOTED for NaN and -Inf: Task 11 explicitly repairs this safety defect, without restarting Task 9 or consuming real data.

## Current provider facts and unresolved live gates (checked 2026-10-01)

Robinhood's official Trading with your agent documentation lists separate equity read, review, place and cancel tools, and a trade-approval-setting read. It states external MCP trade approvals default to OFF. Therefore the gateway must verify approvals ON rather than infer it from connection success. The actual account-specific schemas, scoped credential options, request-ID/idempotency, pagination and fill/cancel semantics remain LIVE UNVERIFIED until Task 13/18 evidence proves them.

Source: https://robinhood.com/us/en/support/articles/trading-with-your-agent/

**Capital prohibition:** Absolute through Tasks 11-17 and the Task 18 dry run. Passing tests, signed synthetic promotion, connecting read-only, or completing this implementation plan does not remove it. Only the separately approved, evidence-qualified Task 18 proof may exercise the sole production order path.

## Task 11 implementation review rulings

- The normalized request explicitly binds an account alias, and the account snapshot explicitly reports corporate-action reconciliation health. No raw broker account ID is exposed.
- The existing research gate inventories every Axiom source file. Coexistence therefore also modifies `scripts/axiom2_verify_isolation.py` and creates `config/axiom2/portfolio_boundary.json` plus `tests/axiom2/portfolio/test_inventory_boundary.py`. Exactly the three reviewed portfolio source files are pinned and import-audited but excluded from the research bundle. Missing policies, incomplete inventories, new files, hash drift, privilege flags, network imports and research-to-portfolio imports fail. No wildcard directory exemption or research artifact expansion is permitted.
- An entry cost buffer below the observable buy-side half-spread fails with `COST_BUFFER_BELOW_SPREAD`; the kernel does not silently alter costs. This is only a lower-bound consistency check, not proof of future fees, slippage or fills. Actual broker costs and realized turnover remain Tasks 12-15 obligations.
- New review coverage lives in `tests/axiom2/portfolio/test_adversarial.py`. It includes integer rounding, cash reserves funded only from available settled funds, exact freshness boundaries, UTC/DST elapsed time, independently recomputed decision hashes, reflected mutations and no-capital behavior across all accepted evidence classes.

**Execution progress:** Task 11 is implemented and locally verified; see `docs/axiom2/EXECUTION_KERNEL_VERIFICATION.md`. Tasks 12-18 remain unchecked and unimplemented.
