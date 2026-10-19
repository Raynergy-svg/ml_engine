# Nautilus operational bindings — bounded plan, 2026-10-07

Historical planning receipt at `65694bf`. David approved its persistent preparation
scopes at 04:07 UTC. The current implementation, test/review status, source
coverage and remaining blockers are in
[NAUTILUS_OBSERVATION_PREPARATION_2026-10-07.md](NAUTILUS_OBSERVATION_PREPARATION_2026-10-07.md).
Statements below saying "plan only" or "unapplied" describe that earlier checkpoint.

Decision: reuse the existing owner transport, Python research worker and process
supervisor. Add narrow adapters around the reviewed observation runtime; do not
create a provider, credentials, a new scheduler/framework, a trading route or UI.
This branch contains offline tests and unapplied proposals only. Runtime code
remains byte-identical to cloud-tested `933753a9`; the green handoff `3c0ca52c`
and original replay `479d868` remain preserved.

## Verified current contracts

| Existing binding | Evidence | Reuse and limit |
|---|---|---|
| Site v71, owner access, published MCP | Project `appgprj_6abf2d73fac48191a2167664e6b15093`; source `6633d32a3b9884c7f8fa64ae6722bcee827f9b5d`; deployment succeeded | Current Site exists. No Site mutation or new credential was requested. |
| Native status | Existing `read_native_read_status`, checked 2026-10-07 03:57:54 UTC: READ_VERIFIED, market_state VERIFIED, sequence 4134 | Owner/session-scoped native reader exists. This is a server report; we did not independently refresh its provider or verify its signature implementation. |
| Signed native publisher | Existing `publish_native_read_status(payload, signature)` declaration | Explicitly accepts sanitized native-read status and **never research heartbeats or account contents**. Do not extend/relabel this channel as Nautilus health. |
| Accepted native market reads | Existing `read_market_data(require_fresh, max_age_ms)` declaration | Reads already-delivered signed batches; default requires every quote field within 1,000ms. It does not refresh the broker. No market-data call was made in this task. |
| Analytical / research worker | `axiom_runtime_status`: ON_DEMAND, CALLER_DELIVERED, trained model UNAVAILABLE; `axiom_research_status`: recent heartbeat at 03:57:46 UTC, engine digest `e9550669…` | A Python worker and research status exist. Its reported heartbeat does not prove a Nautilus node, supervisor configuration or signed promotion. No research job was submitted. |
| Collector health | Existing `axiom_research_health`: REPORTED; nested collector PROVIDER_UNAVAILABLE, last receipt 2026-10-04 13:42:28 UTC | Reported metadata availability is separate from collector availability, source freshness and observation-worker health. Preserve its negative state. |
| Bounded research pipe/socket | `src/axiom2/research/development.py:566,609`, `scripts/axiom2_run_phase1_campaign.py` | Exactly action + configured snapshot_id; preflight/run only, 64KiB newline request, existing registry/allowlist/verified checkout injected by owner. No observation/status/restart action exists here. Reuse transport infrastructure, not its action semantics or historical-data admission as live admission. |
| Finite work queue | `dashboard/server/training_jobs.py` | One worker thread, durable submitted/running/failed/completed journal; caller supplies already-authorized callable. No leases/heartbeat/restart reconciliation. Do not occupy it with the infinite observation runner or infer current liveness from a running row. |
| Producer / evidence authority | `src/evidence/equity_research/worker.py`, `local_import.py`, `src/axiom2/promotion/authority.py` | Worker consumes signed job/manifests and injected bytes, creates evidence only; Axiom separately verifies/imports/promotes. Existing default registry mirror is best effort; native observation host must not invoke provider or mirror side effects. |
| Owner read-only display | `dashboard/server/training_cockpit.py`, `data_sources.py`, Next authenticated GET proxy | Readback routes and projections already exist. `connected`, `available` and persisted job state are not native live-handle proof. No new UI is in scope. |
| Process restart mechanisms | `scripts/axiom_launchd/README.md`, `com.axiom.api.plist`, `deploy/supervisord.bot.conf` | launchd/KeepAlive and supervisord already exist. Repo files do not prove current installation. Legacy scanner supervisor reaches execution; API launch config enables control and its lifespan builds an OANDA client. Do not start/reconfigure these as a credential-free native host. |

The Site source archive is declared as SHA256
`4d71bfbcf802cddc8e8ff8eec6c12cd9ed766919d940fffa5d3fd71e68149d73`
(115 files). Its download returned `file could not be authorized or resolved`.
Current MCP declarations and read-only statuses were inspected, but native
publisher verification, worker ingress, persistent-store schema and its installed
supervisor cannot be claimed source-reviewed. Required handoff: Site owner supplies
its existing source checkout/handoff at the above commit, including those four
contracts; no new write credential should be created to resolve this.

## Minimal operational chain and ownership

```text
existing admitted owner records / retained verified native batches
  -> Axiom admission + candidate/thesis/version/freshness binding
  -> existing owner-private transport, bounded observation adapter
  -> existing Python worker's observation child / isolated native host mode
  -> NautilusContinuousRuntime.publish(record, generation=current)
  -> CandidateJournal + material-event outbox
  -> existing governed finite research worker, only if separately authorized
  -> owner-verified result -> runtime.complete(result, generation=current)
  -> owner-only status projection through existing status transport
```

1. **Source/admission adapter.** The already-authorized owner producer supplies a
   retained source receipt and immutable evidence reference. Resolve and verify
   them in the authority-owning service before constructing CandidateObservation.
   Bind candidate_id/version to the owner's actual thesis; confirmation,
   invalidation and freshness are Axiom policy outputs, never caller flags or a
   broker status label. A CandidateObservation constructor validates shape only.
   Reject arbitrary paths/imports/URLs/accounts/keys and unknown fields; use the
   existing configured allowlist. No provider refresh, new symbol enrollment or
   daily research bundle is implicit in this binding.

   Reuse native receipt identity for idempotency, with owner/candidate/version
   binding; do not generate a fresh identity merely because a process restarted.
   Retain raw_observation_digest of the actual retained record and evidence_digest
   of the resolved Axiom evidence. Do not substitute receipt time for source time.
   A missing source time or missing admission receipt remains UNVERIFIED.
   Effective freshness is the earliest applicable source-field deadline, delivery
   deadline, enrolled-session expiry, Axiom thesis/evidence deadline and research
   result deadline. Keep the existing 1,000ms quote bound. READ_VERIFIED or a
   session with minutes remaining cannot make older quotes fresh. Gaps between
   existing deliveries are expected stale periods; this plan adds no fetch loop.

2. **Native host and input transport.** Reuse the existing native Python worker's
   process manager and owner-private transport after its handoff confirms them.
   Use a separately isolated observation child/host mode, not the finite research
   queue, broker collector or legacy execution supervisor. Only the existing
   admitted-record adapter and health channel are available to this child. Its
   environment is explicitly allowlisted, with no broker tokens, signer keys,
   account lookup, assistant connector session or provider network access.
   CPython/OS/architecture and extension must match its independently verified
   pinned artifact; Linux bytes are not usable proof for Mac.

   The host constructs exactly the reviewed zero-client SANDBOX runtime, with
   interval 100ms and mechanics lease 500ms. No strategies/plugins/factories or
   caller-supplied LiveNodeConfig. In-process entrypoints are start, publish,
   pending, complete, status and shutdown. An external adapter stays bounded to
   existing framed records (proposal: at most 64KiB each); its generation is the
   current native instance's generation, not a session ID or research-job epoch.
   The original preflight/run handler remains unchanged. Adapter action dispatch
   must be distinct and owner-only before any mutation; no restart control is
   added to a browser or public MCP caller.

3. **Research/result adapter.** The outbox wakeup ID, bound evidence digest and
   candidate/version resolve an existing authorized job, never an invented new
   research budget. Reuse the existing finite worker and preserve its manifest,
   trial limits, history, holdout policy and authority context. The runtime itself
   performs no research or inference. Deliver only a result whose immutable
   evidence and lineage were verified by Axiom; a plain `qualifies=true` input is
   not a signature, promotion or trade authorization. Keep claim/result records
   durable and idempotent. IN_PROGRESS is recoverable delivery state, not proof a
   fitting process is alive; resuming computation needs its original job policy.
   Stale/superseded/future-dated results cannot promote the monitored candidate.

4. **Status adapter.** Reuse owner-only `axiom_runtime_status` readback with a
   distinct proposed `observation_runtime` object. Research heartbeat, collector
   status and native broker-read status retain their original fields/semantics.
   The proposed write lives in the existing owner-scoped metadata store under a
   separate observation-runtime namespace; its physical schema/handler must be
   confirmed by the current Site handoff. This scope is unapplied.

   A status receipt binds owner scope, engine/source digest, native process
   instance, generation, monotonic publication sequence, emitted_at,
   heartbeat_ns, lease_until_ns, source/evidence deadlines and the already-approved
   publisher identity. Runtime.status currently provides local health; transport
   fields above would need a separately reviewed adapter/envelope. Do not reuse
   the native-read signing scope or key for a new type without owner approval.
   Receipt time, source time and heartbeat time remain separate.

   Only the native host can attest its current handle. A remote view displays
   the reported state plus its age; it cannot infer a current running handle
   from persisted bytes. Expired lease/session, future time, replayed generation/
   sequence, missing publisher proof, disconnected transport or unreadable state
   clears readiness and yields STALE/UNVERIFIED/FAULTED. Never refresh timestamps
   from the act of reading. Effective candidate readiness requires current native
   generation/lease/handle plus valid source, evidence and Axiom candidate state.
   Monitoring readiness remains distinct from execution readiness. Every status
   retains execution_enabled=false and capital_authorized=false.

5. **Persistence, retention and restart.** Proposed writer scope is only
   `${AXIOM_EVIDENCE_ROOT}/observation_runtime/candidates.sqlite` and its SQLite
   WAL/SHM; producer receipts remain in their existing immutable source store.
   Do not let this process write packages, indexes, champions, authorization
   journals, risk state or broker ledgers. This new production writer scope is
   unapplied, even if the parent directory already exists.

   Reuse the original source store's retention for source bytes. Preserve the
   coupled candidate raw/event/outbox journal across restarts; no TTL deletion,
   truncation, fresh empty-history substitution or pruning of only one table.
   Proposed bounded resource policy: combined DB/WAL/SHM <=64MiB; deny new
   ingestion and mark storage fault when budget is exhausted, retain existing
   state, require owner export/retention decision. No automatic compactor/archive
   framework is proposed. Current runtime has no quota/production-retention
   enforcement; these values are an explicit unapplied owner proposal.

   Reuse the existing OS supervisor. Child death clears current-process proof;
   restart waits for the retained lease, replays/verifies the journal, acquires a
   new generation, and reports UNVERIFIED until a real native timer and new fresh
   admitted source data arrive. A returned job result is delivered idempotently,
   never causes another broker action. Native 3s startup lease and old-generation
   write fencing remain unchanged. Clean stop calls shutdown and cancels timers.
   Retained cancellation tracebacks can delay same-thread node replacement;
   recover with a fresh child process rather than modifying upstream lifetimes.
   Supervisor restart restores availability only; it cannot unhalt, enable
   execution, consume an authorization nonce or bypass the final gateway.

## Apply sequence and required decisions

| Stage | Current authorization permits | Remaining prerequisite / action |
|---|---|---|
| Read existing Site/repo contracts | Done, harmless current-state reads | Source owner supplies existing v71 checkout/handoff; verify actual ingress, signatures, store and installed supervision |
| Offline composition | Done; five tests, no native build/provider/model | Retained result fixtures are synthetic; they do not prove signed admission |
| In-memory already-admitted delivery | Existing runtime interface is tested | Actual owner producer must resolve policy/receipt bindings before operational use |
| Add observation host/transport mode | Plan only | Review current worker's interpreter/lifetime; approve isolated child scope, exact framed actions and sanitized environment; keep deployment disabled |
| Add journal/status persistence | Plan only | Approve the exact candidate SQLite writer and owner-scoped status namespace, current publisher identity/type binding and retention/quota policy; apply no broader source/profile grant |
| Install/start under current supervisor | Not authorized by this task | Separate production activation decision after Mac qualification and reviewed end-to-end denial/restart proof |

Concrete preparation approval request (unapplied): approve only an observation
child of the existing native Python worker, receiving already-admitted owner
records on the existing private transport; allow writes only to the candidate
SQLite path/WAL/SHM above and a distinct owner-scoped observation-runtime status
namespace through the existing authorized publisher; approve the 64KiB record and
64MiB journal limits and retain-before-owner-export policy. Require the current
Site handoff to bind the exact service identity, signature scope and physical
status storage before applying it. No broker/provider access, new credentials,
risk/capital changes, source-profile expansion, trade routes or production start
is included in this preparation request.

## Verification and acceptance

Five offline tests passed in 0.54s. Together with the existing research isolation
and native inventory boundary checks, **122 cases passed in 3.59s** with warnings
as errors. All 64 registered source files and all four policies remain unchanged
from `3c0ca52c`. No native build or full research campaign was repeated for these
test/documentation-only additions. The offline tests use existing interfaces: observation serialization/outbox,
finite worker result delivery/idempotency, gateway denial at READY, old research
action rejection, and persisted job state without runtime-health proof. Run:

```sh
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/axiom2/nautilus_runtime/test_operational_bindings.py -q -W error
```

Before any operational activation, use a sacrificial owner-private test scope:
reject raw/unadmitted/future/stale/superseded records and wrong owner/session;
verify source/delivery/heartbeat ages independently; reject replayed sequence and
old generations; interrupt native child and transport; deny all gateway methods
throughout; preserve journal/results on fresh-process restart; expire READY
without source delivery; deny storage growth at the approved budget; and confirm
status reader performs no source fetch or timestamp refresh. Do not fit a new
model/campaign or submit real orders for these checks. Axiom evidence integrity,
promotion, thesis/candidate policy, risk/capital/human/final command authorization,
signed journals/fences, reconciliation truth and the sole gateway stay unchanged.

| Artifact | Implemented | Tested | Reviewed | Mac |
|---|---|---|---|---|
| Continuous PoC at 933753a9 | Yes | 1,465 Linux cloud tests | Independent approval retained | Owner qualifying separately |
| Existing-interface offline binding tests | Tests only | Five passed; 122 including isolation | Self-checked; no production adapter approval implied | No native extension required |
| Operational adapters / persistent access / activation | Proposal only | Acceptance plan above | Requires concrete source/scope review | Pending owner |

Supporting current-state receipts and declarations are in
`nautilus-operational-evidence-2026-10-07/`. Financial values, account/session
identifiers, retained datasets/results, credentials and signing keys are omitted
from these sanitized receipts. Site status is a point-in-time report, not an
ongoing availability or signature-verification attestation by this worker.
