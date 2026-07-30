# Phase G / H / N Infrastructure & Security Audit

**Date:** 2026-07-30 · **Auditor:** Security Engineer sub-agent · **Mode:** read-only, no code changed
**HEAD:** `074758d` (`research: per-lane halt + shadow lanes ... (#52)`)
**Scope:** `AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md` §4.4 (capability enforcement), §10 Phase G
(containers), §11 Phase H (batch job graph), §17 Phase N (operational hardening), plus credential
handling and CI.

**Method.** Every claim below was re-derived by reading files on disk in this session. The
roadmap's own `Implementation status: COMPLETE` markers (lines 577, 638, 676, 706) were **not**
used as evidence and are not repeated. Confidence is tagged per finding: HIGH = read the file and
the enforcing code path; MEDIUM = read the file, inferred one hop; LOW = grep-level only.

**Classification scale**
| Label | Meaning |
|---|---|
| ABSENT | No artifact on disk implements it. |
| ISOLATED | Implemented + tested, but no production call site outside `tests/`. |
| PARTIAL | Implemented and reachable, but the enforcement is incomplete or advisory. |
| LIVE | Structurally enforced in a code path that production actually executes. |

---

## 0. Headline findings

1. **No private keys, no hardcoded secrets anywhere in the repository.** Verified four ways
   (§4.1). This is the one unambiguous pass.
2. **`JobManifest.container_digest` is fabricated.** Five signed slices hash a literal ASCII
   string instead of an image digest — e.g. `sha256_bytes(b"axiom-execution-cost:local-inprocess:v1")`
   at `src/evidence/execution_cost/slice.py:100`. The signed provenance record asserts a container
   identity that does not exist. HIGH.
3. **Zero of the 12 recommended images exist.** Three unrelated deployment images do
   (`deploy/Dockerfile.{api,bot,web}`), satisfying ~1.5 of the 13 hardening requirements between them.
4. **There is no remote worker.** Every "producer / importer / independent verifier / operator /
   promotion service" runs inside one Python interpreter, holding ephemeral keys it generated itself
   (`build_slice_identities`, `src/evidence/execution_cost/slice.py:53`,
   `src/evidence/risk_target/slice.py:84`). Five of the nine §4.4 prohibitions therefore have no
   runtime boundary behind them (§3).
5. **None of the 22 evidence/AXIOM test files run in CI.** `.github/workflows/code-quality.yml:113-131`
   runs a hand-curated 15-file list; `grep -c "evidence\|axiom"` over the `tests/` lines in that
   workflow returns **0**. 5,482 lines of governance tests are unenforced on push. HIGH.

---

## 1. Phase G — Container and environment program

### 1.1 Inventory

`find` for `Dockerfile*`, `*.dockerfile`, `Containerfile*`, `docker-compose*` over the tree returns
exactly three files, all under `deploy/`:

| File | Purpose (from its own header) | Base |
|---|---|---|
| `deploy/Dockerfile.api` | FastAPI data/control layer | `python:3.11-slim` (`:3`) |
| `deploy/Dockerfile.bot` | OANDA trend loop + Tier 7 supervisor | `python:3.11-slim` (`:3`) |
| `deploy/Dockerfile.web` | Next.js operator cockpit | `node:22-slim` (`:4`, `:13`) |

No Kubernetes manifests, no Helm chart, no `docker-compose.yml`, no `fly.toml` — despite
`Dockerfile.api:31` and `entrypoint.sh:6` both referring to "compose / Fly volume". The runtime
topology those comments describe is **not in this repository**, so none of its claims (private
network, unpublished `:8888`, volume mounts, resource limits) can be verified here. MEDIUM.

### 1.2 The 12 recommended images

| Image | Status |
|---|---|
| `axiom-risk-target` | ABSENT |
| `axiom-execution-cost` | ABSENT |
| `axiom-fx-benchmark` | ABSENT |
| `axiom-equity-research` | ABSENT |
| `axiom-crypto-momentum` | ABSENT |
| `axiom-crypto-carry` | ABSENT |
| `axiom-track-b-ingestion` | ABSENT |
| `axiom-track-b-scoring` | ABSENT |
| `axiom-track-b-harness` | ABSENT |
| `axiom-hedge-evaluation` | ABSENT |
| `axiom-multi-sleeve` | ABSENT |
| `axiom-independent-verifier` | ABSENT |

**0 / 12.** HIGH. The names appear as ASCII literals inside `container_digest` computations
(`execution_cost/slice.py:100`, `crypto_momentum/slice.py:231`, `hedge_eval/slice.py:182`,
`track_b/slice.py:130` and `:186`, `equity_research/slice.py:216`) — i.e. the image names exist only
as strings being hashed to fill a required contract field.

### 1.3 The 13 requirements, assessed against the 3 images that do exist

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 1 | Locked dependencies | PARTIAL | `deploy/requirements-deploy.txt` pins 5 of 6 with `==`; `numpy>=1.26.4` floats (line 16). `requirements.txt`: 22 of 159 lines are non-`==`. No hashes, no `requirements.lock`. Web is the best case — `npm ci` against a committed lockfile (`Dockerfile.web:7-8`). |
| 2 | Immutable digest | ABSENT | All three `FROM` lines use mutable tags, not `@sha256:` (`Dockerfile.api:3`, `.bot:3`, `.web:4,13`). |
| 3 | Non-root user | ABSENT | No `USER` directive in any of the three files. `deploy/supervisord.bot.conf:8` is explicitly `user=root`. |
| 4 | Read-only root filesystem | ABSENT | No directive, and structurally incompatible: `deploy/entrypoint.sh:17-28` mutates `/app` at start-up (`rm -rf`, `ln -sfn`) on every boot. |
| 5 | Health check | ABSENT | No `HEALTHCHECK` in any Dockerfile. An `/api/health` endpoint exists and is polled externally (`scripts/axiom_safety_monitor.py:92`), but the container declares no liveness contract. |
| 6 | Resource limits | ABSENT | No orchestration manifest exists to carry them (§1.1). |
| 7 | No broker SDK credentials | PARTIAL | `.dockerignore:8-13` excludes `.env`, `.env.*`, `**/.env.local`; no secret is baked in (§4). But `Dockerfile.bot` *is* the trading bot — it requires OANDA credentials at runtime by design. The roadmap's requirement is about *research worker* images; since none exist, the property is untested rather than met. |
| 8 | No local control files | FAIL | `Dockerfile.api:22` and `Dockerfile.bot:18` both `COPY scripts/`; `entrypoint.sh:27` symlinks `.claude/` (which holds `state.json`'s `halted` flag) into `/app`. Both images carry the full local control surface. |
| 9 | Minimal network policy | ABSENT | No policy artifact. `Dockerfile.api:11` sets `AXIOM_BIND_HOST=0.0.0.0`; lines 30-31 assert `:8888` "is never published publicly" — an assertion with no enforcing file in the repo. |
| 10 | SBOM | ABSENT | Zero references to `syft`, `sbom`, `cyclonedx`, `spdx` anywhere in `scripts/`, `.github/`, `deploy/`. |
| 11 | Vulnerability scan | ABSENT | Zero references to `trivy`, `grype`, `snyk`, `pip-audit`, `safety`, `bandit`. |
| 12 | Reproducible build | ABSENT | Floating base tags (#2) + unpinned pip resolution (#1) + no build script. |
| 13 | Signed image | ABSENT | Zero references to `cosign`, `notation`, `notary`. |

**Score: ~1.5 / 13** (partial credit on #1 and #7). **Phase G exit gate — "the same signed job
produces equivalent evidence locally and in the isolated worker environment" — is
unreachable**: there is no isolated worker environment, and the digest that would bind the two is a
literal-string hash. HIGH.

**Phase G classification: ABSENT.**

---

## 2. Phase H — Distributed research fabric

### 2.1 Is there a batch job graph?

**No orchestration layer of any kind exists.** Grep for `ray`, `celery`, `dask`, `boto3`,
`kubernetes`, `modal.`, `runpod`, `paramiko` across `src/` and `scripts/` returns four hits, all
false positives (substring matches in `src/tui/app.py`, `trade_search_modal.py`,
`trades_screen.py`, `src/equity/live_gate.py`). No queue, no scheduler, no submission API. HIGH.

The roadmap's six stages exist as **ordered function calls inside a single process**. For the
execution-cost lane, `run_execution_cost_evidence_slice`
(`src/evidence/execution_cost/slice.py:117-130`) calls `produce_worker_output` → `run_worker`
(`:105`) → `import_model`, sequentially, in one interpreter.

| Roadmap stage | On disk | Class |
|---|---|---|
| snapshot validation | `worker._verify_partitions` (`execution_cost/worker.py:34-41`) | PARTIAL (in-process) |
| feature generation | inside `evaluate_partitions` per lane | PARTIAL |
| fold workers | `run_worker` — one call, all cells | PARTIAL |
| result aggregation | per-lane `evaluate_partitions` (see §2.2) | PARTIAL |
| independent verification | `local_import._preflight` replay (`execution_cost/local_import.py:37-99`) — real recomputation, but same process, verifier key held by the caller | PARTIAL |
| evidence packaging | `EvidenceStore.write_package` (`src/evidence/store.py:212`) | LIVE |

**Job-graph classification: ABSENT** as a fabric; the *stages* are PARTIAL.

### 2.2 Aggregator that refuses missing / duplicate folds — the strongest part of Phase H

This genuinely exists and is genuinely enforced, per lane, at evaluation time:

- `src/evidence/crypto_momentum/evaluation.py:205-220` — raises on an undeclared construction
  producing cells, and on declared constructions with no supplied cells.
- `src/evidence/equity_research/evaluation.py:240-244` — "that is an incomplete aggregation —
  refuse rather than silently omit a head"; raises on declared tiers with no folds.
- `src/evidence/track_b/evaluation.py:152-155` — raises on `duplicate expected evaluation cells`
  and on an evaluation-cell set mismatch, reporting both missing and extra.
- `src/evidence/execution_cost/evaluation.py:70-77` — refuses fill records missing fields and
  duplicate `order_id`.
- The expected fold set is **signed**: `crypto_momentum/local_import.py:81` reads "the signed
  expected-fold-set-per-construction declared in the strategy manifest", so the aggregator cannot be
  told a smaller expectation after the fact.

Tested in `tests/test_{equity_research,hedge,risk_target}_evidence_slice.py` (missing/incomplete
folds) and `tests/test_{crypto_carry,crypto_momentum,equity_research,execution_cost}_evidence_slice.py`
(duplicates). HIGH.

**Caveat:** workers "never determine the final verdict" holds structurally *only because* the
aggregator is a separate function whose inputs are hash-bound — not because it is a separate
principal. In the wired path the same process runs both. **Classification: PARTIAL** (LIVE logic,
no isolation).

### 2.3 Resource classes

`ResourceClass` (`src/evidence/contracts/models.py:39-46`) enumerates **exactly** the six roadmap
classes and is a **required** field of the signed `JobManifest` (`models.py:211`). That is real
contract-level modelling.

But every production call site hardcodes `CPU_SMALL` (`execution_cost/manifests.py:112`,
`crypto_momentum/manifests.py:229`, `hedge_eval/manifests.py:185`, `equity_research/manifests.py:214`)
with the single exception of `track_b/slice.py:134` (`DOCUMENT_PROCESSING`). Nothing reads the field
to select hardware — grep for `resource_class` outside `src/evidence/` returns nothing.

**Classification: PARTIAL** — declared and signed, never consumed. HIGH.

### 2.4 Cost controls

| Control | Status | Evidence |
|---|---|---|
| Per-job cost ceiling | ABSENT | No `cost` field on `JobManifest` or `EvaluationReport`. |
| Per-campaign trial ceiling | PARTIAL | `JobManifest.trial_budget: int = Field(ge=1)` (`models.py:213`); `execution_cost/manifests.py:70,100` hard-refuse `trial_budget != 1`. That is a *per-job* cap of one, not a campaign budget. |
| Daily / monthly quota | ABSENT | Grep `quota` over `src/evidence/` → zero hits. |
| Idle / stuck termination | ABSENT | No timeout, no watchdog on job execution. |
| Spot/preemptible only for restartable workers | ABSENT | N/A — no compute layer. |
| Checkpointing for long jobs | ABSENT | Grep `checkpoint` over `src/evidence/` → zero hits. |
| No open-ended sweep | LIVE-by-accident | `trial_budget != 1` refusal makes sweeps impossible in the two lanes that check it. |
| Cost in the evidence report | ABSENT | `EvaluationReport` (`models.py:227-258`) has no cost field. |

**Cost controls: 1 of 8 present, 1 partial.** HIGH.

**Phase H classification: ABSENT** overall; **aggregation logic PARTIAL(strong)**, **resource
classes PARTIAL(declarative)**, **cost controls ABSENT**.

---

## 3. §4.4 — Capability enforcement (the security core)

### 3.1 The structural situation that governs every row below

`CapabilityProfile` (`src/evidence/contracts/models.py:176-188`) declares eight booleans, each typed
`Literal[False]`. This means pydantic **rejects at parse time** any profile asserting the authority —
a genuine schema-level guarantee that a *signed declaration* can never claim these powers.

It also means the runtime check that validates them is **tautological**.
`local_import._authority_free` (`execution_cost/local_import.py:26-36`) iterates the eight fields
asserting each `is False`; since no other value can be constructed, the only load-bearing clause in
that function is `profile.network_endpoints == ()`. The `no_forbidden_capabilities` import check
(`local_import.py:69`, `:93`) therefore certifies a promise, not a behaviour.

Second, and more important: **the worker is not a separate principal**.
`build_slice_identities` (`execution_cost/slice.py:52-59`, `risk_target/slice.py:77-92`) executes
`Ed25519Signer.generate()` for *every* role — producer, local importer, independent verifier,
operator, promotion service — inside the calling process, then registers all of them in one
`TrustStore` and one `AuthorityRegistry`. Separation of duties is enforced *between key IDs* that a
single process simultaneously holds in memory. A compromised or buggy worker in that process is,
by construction, also the operator and the promotion service.

Third, only **one** lane is wired to anything outside `tests/`: `risk_target`, via
`scripts/run_risk_target_evidence_slice.py:90` and `dashboard/server/axiom_evidence_control.py:477`.
The other six lanes (execution_cost, crypto_momentum, crypto_carry, equity_research, hedge_eval,
track_b) are **ISOLATED** — tests only. HIGH.

### 3.2 The nine prohibitions

| # | Prohibition | Enforcement | Class |
|---|---|---|---|
| 1 | Read broker credentials | **Convention + CI-time static test.** `may_read_broker_credentials: Literal[False]` is a declaration. The only real check is an AST import-graph test forbidding `src.brokers` in producer files (`tests/test_execution_cost_evidence_worker_no_authority.py:29-51`). No process, container, uid, or env boundary: an in-process worker can read `os.environ["OANDA_API_TOKEN"]` directly, and nothing prevents it. | PARTIAL |
| 2 | Place or cancel orders | **Convention + static test.** Same AST test bans `src.scanner.execution` / `src.trading.execution`; a source-string scan rejects `place_order(`, `cancel_order(` (`:54-60`). Both are grep-strength — an `importlib` call or a string-built attribute defeats them. Mitigating: `grep` for `requests`/`urllib`/`http` across all of `src/evidence/` returns **zero** hits, so no evidence-lane code performs network I/O today. | PARTIAL |
| 3 | Read operator signing keys | **Convention.** The operator key lives at `~/.config/axiom/operator_ed25519.pem`, mode `0600`, and reads refuse if group/other bits are set (`scripts/axiom_operator_credential.py:66-67`). That stops *other users*, not a same-user research process. It is also written with `serialization.NoEncryption()` (`:100`) — unencrypted at rest. | PARTIAL |
| 4 | Change lane halt state | **Two soft gates + a static test.** `effects.execute_safe_action` refuses `halt_lane`/`stop_loop` unless `AXIOM_CONTROL_ENABLED` is truthy (`src/axiom_operator/effects.py:35-36`); the dashboard router requires an `X-AXIOM-CONFIRM` header (`dashboard/server/control.py:70`); the AST test bans importing `src.scanner.automation.state_engine` (test `:32`). Note the direction: halting is *permitted* auto-action policy (`policy.py:14-20`), unhalting is human-required (`:22-32`) — fail-safe. But an in-process worker can still `import` and call `StateEngine` directly; only the test forbids it. | PARTIAL |
| 5 | Change LiveGate state | **Convention only.** `may_change_live_gate` is a `Literal[False]` declaration with no counterpart anywhere: `src/equity/live_gate.py` and the three sibling gates are **not** in the AST forbidden-import list (test `:30-34`), and no capability check guards them. The only enforcement is a CI *review-evidence* rule — `.ci/policy/rego/live_arm_review_evidence.rego:20-26` requires a checklist file or a `policy:live-arm-reviewed` label when a PR touches those files. That gates human commits, not running code. | CONVENTION |
| 6 | Write a champion pointer | **Structurally enforced.** `EvidenceStore._validate_champion_unlocked` (`src/evidence/store.py:551-559`) calls `authorities.authorize_identity(role=PROMOTION_SERVICE, key_id=pointer.signer_key_id)` and raises `ChampionPointerError` when the signer is not registered for that role; it further requires the pointer to resolve to exact package bytes (`:566-570`), the reconstructed ledger state to be `CHAMPION` (`:577-578`), and the pointer to reference the current disposition head (`:579-580`). Compare-and-swap on `expected_pointer_digest` (`:589`, `:603-607`). Producer files are additionally AST-banned from importing `src.evidence.store`. | LIVE* |
| 7 | Modify local model directories | **Convention.** AST ban on `src.evidence.store`; `run_worker` returns bytes and never opens a file for write. No filesystem sandbox. A real boundary *does* exist — `DataPlatform.export_training_root` (`src/data_platform/platform.py:171-188`) copies a replica and `chmod`s every path to `0o555`/`0o444` — but grep shows **zero production call sites**; it is exercised only in `tests/test_data_platform.py:61`. | PARTIAL (boundary exists, ISOLATED) |
| 8 | Approve its own evidence | **Structurally enforced.** `enforce_separation_of_duties` (`src/evidence/transition_policy.py:125-164`) rejects any ledger where two of {producer, local importer, independent verifier, operator, promotion service} share an `actor_id` **or** a `signer_key_id`, and requires producer + independent verifier + operator to be present before `OPERATOR_APPROVED`. Tested: `tests/test_evidence_contracts.py:424` `test_one_identity_cannot_self_promote_across_a_complete_ledger`, `:404` `test_transition_authority_is_not_self_attested`. | LIVE* |
| 9 | Promote a package to shadow or live | **Structurally enforced.** `_TRANSITION_AUTHORITIES` (`transition_policy.py:15-41`) permits `QUARANTINED→SHADOW` only for `OPERATOR` and `OPERATOR_APPROVED→CHAMPION` only for `PROMOTION_SERVICE`; unlisted transitions are refused outright (`enforce_transition`, `:101-113`). `store._require_promotion_evidence_unlocked` (`:396-410`) additionally requires a `QUARANTINED` event carrying an exact `verdict_digest` backed by a durable accepted verdict. **Gap:** `CapabilityProfile` has no `may_promote` field — the 9th prohibition is the only one with no declarative counterpart (8 booleans for 9 prohibitions). | LIVE* |

\* **LIVE within the ledger's trust model.** All three depend on the assumption that role keys are
held by different principals. §3.1 shows they are not: one process generates and holds all five. The
cryptographic machinery is correct and well-tested; the deployment does not yet give it anything to
protect against.

### 3.3 Plain answer

**Convention-only or static-test-only (no runtime boundary): prohibitions 1, 2, 3, 5, 7 — five of
nine.** Prohibition 4 is gated by an env flag and an HTTP header but not by process isolation.
Prohibitions 6, 8, 9 are cryptographically enforced in the evidence store and disposition ledger and
are the strongest security work in this repository — but their separation-of-duties premise is
currently simulated inside one interpreter.

An AST import test and a `Literal[False]` type annotation are **not** capability enforcement. They
are lint. Saying otherwise would be the failure mode this audit exists to catch.

---

## 4. Credential handling

### 4.1 Secret exposure — clean

Four independent checks, all negative:

1. `git ls-files | grep -iE '\.(pem|key|p12|pfx|jks|asc|gpg)$|id_rsa|id_ed25519|private'` → **empty**.
2. `git grep -lI "BEGIN .*PRIVATE KEY"` → **empty**.
3. `find` for `*.pem`, `*.key`, `id_rsa*`, `*.p12` on the working tree → **empty**.
4. Pattern scans for `sk-ant-`, `sk-…`, `ghp_…`, `AKIA…`, and OANDA's `32hex-32hex` token shape →
   one hit only, `tests/test_axiom_operator.py:44` `monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")`,
   a test literal.

`.gitignore:9-11` blocks `.env` and `.env.*` with a single allowlisted example file.
`.dockerignore:8-13` blocks the same from images. **No secret or credential exposure found.** HIGH.

### 4.2 How credentials load

Uniformly via environment: `os.environ` / `os.getenv` at
`src/scanner/automation/broker_transport.py:279-280`, `outcome_backfill.py:296-297`,
`state_reconciler.py:277-278`, `trend_journal_sync.py:657`, `src/data/tick_capture.py:85-86`,
`src/tui/screens/mode_modal.py:23-24`. `src/brokers/factory.py:53-55` refuses construction without
both `oanda_account_id` and `oanda_api_key`. No file-based credential store, no config-file secret.
HIGH.

### 4.3 Can a training/research process read them? — Yes, except in one place

Any process launched from an operator shell inherits `OANDA_API_TOKEN` / `OANDA_ACCOUNT_ID`.
`scripts/run_risk_target_evidence_slice.py` performs no env scrubbing.

The **one** genuine credential-containment boundary in the repo is
`src/axiom_operator/runner.py:38-52` `_build_child_env`: the Claude CLI subprocess receives an
explicit allowlist (`HOME`, `USER`, `LANG`, `LC_ALL`, `TMPDIR`, `SHELL`, plus a resolved `PATH`),
and `ANTHROPIC_API_KEY` only when the caller opted into API billing. Broker credentials are
structurally invisible to it. This is the pattern the research workers need and do not have.
**Classification: LIVE (operator subprocess only).** HIGH.

**Latent leak surface.** `ScannerConfig.oanda_api_key: str = ""` (`src/scanner/config.py:741`) is a
credential-shaped dataclass field with **no writer and no reader** anywhere in `src/`, `dashboard/`
or `scripts/` (grep for `config.oanda_api_key` and `.oanda_api_key =` → empty). Harmless today, but
`asdict()`-style config dumps are common in this codebase (e.g. `src/training/wandb_observatory.py:662`,
`src/training/trainers/*.py`), and a future `ScannerConfig` dump would exfiltrate it to W&B or a
JSON artifact. Delete the field. MEDIUM.

### 4.4 Secret scanning in CI

**Present.** `.github/workflows/code-quality.yml:10-25` runs `gitleaks/gitleaks-action@v2` with
`fetch-depth: 0` (full history) and `persist-credentials: false`. No `.gitleaks.toml` — it runs on
default rules, so project-specific patterns (OANDA token shape, the `ed25519:` key-id format) are
not detected. Triggers on push to `main`/`develop`/`copilot/**` and PRs to `main`/`develop`
(`:3-7`) — pushes to other branches are unscanned. MEDIUM.

---

## 5. Phase N — Operational hardening

### 5.1 Failure injection — 17 scenarios

| # | Scenario | Status | Evidence |
|---|---|---|---|
| 1 | Corrupt manifest | PARTIAL | Covered generically by `tests/test_evidence_contracts.py:257` (one-byte payload change invalidates the signed object) and `:271` (one-byte signature change). No manifest-specific corruption test. |
| 2 | Missing partition | LIVE | `worker._verify_partitions` raises `DatasetHashError` on set mismatch (`execution_cost/worker.py:36-37`); exercised by the slice tests. |
| 3 | Wrong dataset hash | LIVE | Same function, `:40-41`; re-checked independently at import (`local_import.py:47-50`). |
| 4 | Forged signature | LIVE | `test_evidence_contracts.py:271`, `:286` (signature metadata cryptographically bound), `:404` (authority is not self-attested), `:470` (backdated revoked key cannot append). |
| 5 | Missing fold | LIVE | §2.2; tests in `test_{equity_research,hedge,risk_target}_evidence_slice.py`. |
| 6 | Duplicate worker output | LIVE | §2.2; tests in `test_{crypto_carry,crypto_momentum,equity_research,execution_cost}_evidence_slice.py`. |
| 7 | Stale data | PARTIAL | Key-validity intervals and `require_trusted_at_receipt` (`signing.py:74-82`) cover *stale signers*. Stale *dataset* injection is not tested. |
| 8 | Worker termination | ABSENT | No test kills or interrupts a worker mid-run. |
| 9 | Partial upload | LIVE | `test_evidence_store.py:209` `test_partial_package_failure_leaves_no_visible_or_temporary_package`. |
| 10 | Event-chain fork | LIVE | `test_evidence_store.py:332` `test_injected_fork_is_detected_during_rebuild`; `test_evidence_contracts.py:389` (broken link, duplicate, fork). |
| 11 | Index corruption | LIVE | `test_evidence_store.py:650` (destroyed indexes rebuild byte-equivalent), `:798` (artifact corruption blocks reconstruction), `:827` (ledger reconstructing to a different package is rejected). |
| 12 | Registry outage | PARTIAL | `run_worker` wraps `registry_publish` and logs a warning — deliberately **fail-open**, "mirror only" (`execution_cost/worker.py:134-137`). Correct for a mirror; means an outage is untested as a *failure*. |
| 13 | Object-store outage | PARTIAL | `test_evidence_store.py:778` `test_failed_atomic_index_replace_preserves_previous_index`; `:477`/`:506`/`:712` cover committed-then-stale projection repair. No full-unavailability injection. |
| 14 | Dashboard outage | ABSENT | No test. |
| 15 | Model-load failure | PARTIAL | `test_evidence_store.py:812` (member reread rechecks declared bytes), `:798`. No test of a corrupt/unloadable model artifact at inference. |
| 16 | Local disk full | PARTIAL (LOW confidence) | `OSError`-path tests exist in `test_evidence_store.py` / `test_data_platform.py`; no explicit `ENOSPC` injection found. |
| 17 | Clock skew | **CODE, NO TEST** | `MAX_SIGNATURE_EVENT_SKEW = 5s` / `MAX_EVENT_FUTURE_SKEW = 30s` and `verify_disposition_timing` (`signing.py:24-25, 240-259`) are wired into `event_store.py:53`. Grep for `MAX_SIGNATURE_EVENT_SKEW`, `MAX_EVENT_FUTURE_SKEW`, `verify_disposition_timing` across `tests/` returns **zero hits**. A safety-critical timing rail with no regression test. |

**8 of 17 have real dedicated tests; 6 PARTIAL; 3 effectively ABSENT (8, 14, 17-as-tested).** HIGH.

### 5.2 Recovery — 8 capabilities

| Capability | Status | Evidence |
|---|---|---|
| Rebuild indexes from events | LIVE | `store.rebuild_indexes` (`:726`) / `_rebuild_indexes_unlocked` (`:703`); tested `test_evidence_store.py:650`. |
| Recreate candidate cache from packages | ABSENT | No candidate cache exists. |
| Recover prior champion | PARTIAL | CAS on `expected_pointer_digest` prevents lost updates (`store.py:589,603-607`) and the ledger is replayable, but there is no "restore the previous champion" operation. |
| Resume restartable jobs | ABSENT | No checkpointing (§2.4). |
| Detect incomplete packages | LIVE | Atomic create + temp-file discipline (`store.py:122-176`); tested `:209`. |
| Preserve evidence through process failure | LIVE | `_fsync_directory` (`:114-120`), atomic replace, fsync of first ledger creation; tested `:752`. |
| Export signed backups | ABSENT | No backup routine. `DataPlatform.export_training_root` is an unsigned read-only training replica and is ISOLATED (§3.2 row 7). |
| Periodic restoration test on a clean environment | ABSENT | No such job in CI or `scripts/`. |

**3 LIVE, 1 PARTIAL, 4 ABSENT.** HIGH.

### 5.3 Key management — 7 requirements

| Requirement | Status | Evidence |
|---|---|---|
| Separate producer vs local-authority keys | PARTIAL | Modelled: `AuthorityRole` (`models.py:62-69`), `AuthorityRegistry` binding actor+role→key IDs (`transition_policy.py:54-91`) with an explicit docstring that "merely writing a privileged role into an event never grants that role". Deployed: `build_slice_identities` generates **all five role keys in one process** (`execution_cost/slice.py:53`, `risk_target/slice.py:84`). Correct design, no deployment separation. |
| No private signing keys in repositories | **LIVE** | §4.1 — verified four ways, clean. |
| Key IDs in signatures | LIVE | `SignatureRecord.key_id`, derived as `ed25519:<sha256(pubkey)[:32]>` (`signing.py:121-124`), bound into the signed statement (`_signature_message`, `:195-203`) so the key ID cannot be swapped. |
| Revocation list | PARTIAL | `TrustStore.revoke` (`signing.py:84-95`) is **in-memory only**, rebuilt per process. The on-disk trust anchor written by `scripts/axiom_operator_credential.py:107-112` carries only `actor_id`, `key_id`, `public_key_b64`, `valid_from` — **no `revoked_at`, no CRL file**. Revoking the operator key across restarts is not expressible. |
| Key rotation process | PARTIAL | Overlap is supported and tested in the library (`TrustedKey.permits`, `signing.py:46-52`; `test_evidence_contracts.py:297` `test_key_rotation_overlap_and_revocation`). But `axiom_operator_credential.py init --force` (`:92-102`) overwrites the private key and rewrites the anchor with a **single** key — a hard cutover with no overlap window, invalidating verification of previously signed history. |
| Offline recovery key | ABSENT | No second key, no escrow, no shard. Compounding: the operator key is written with `serialization.NoEncryption()` (`:100`) — unencrypted, protected only by mode `0600`. Loss or theft of that one file is unrecoverable / total. |
| Audit of every signing operation | ABSENT | `Ed25519Signer.sign` / `sign_digest` (`signing.py:150-192`) write no log. Two adjacent ledgers exist but do not cover signing: `trained_data/axiom/operator_nonces.jsonl` records credential *consumption* (`axiom_evidence_control.py:221-274`, single-use + replay guard) and `trained_data/axiom/safety_monitor.jsonl` records monitor runs (`axiom_safety_monitor.py:21,201-205`). |

**1 LIVE, 4 PARTIAL, 2 ABSENT.** HIGH.

**Positive note:** the operator credential design is strong where it exists — Ed25519, action- and
subject-bound, `MAX_CREDENTIAL_WINDOW = 300s` (`axiom_evidence_control.py:61`), a durable
single-use nonce ledger with file locking, `0600` private key with a permission pre-check, and a
single-execution guard on the authorized work item (`axiom_evidence_control.py:471-476`).

### 5.4 Supply-chain security — 8 requirements

| Requirement | Status |
|---|---|
| Pinned dependencies | PARTIAL (§1.3 #1) |
| Signed containers | ABSENT |
| SBOM | ABSENT |
| Vulnerability scanning | ABSENT |
| Restricted package sources | ABSENT (no index pinning, no internal mirror) |
| Reproducible builds | ABSENT |
| Secret scanning | **LIVE** (gitleaks, §4.4) |
| Provenance records | PARTIAL — `JobManifest` carries `git_commit`, `configuration_digest`, `feature_pipeline_version`, `random_seeds` (`models.py:201-215`), all signed. But `container_digest` is fabricated (§0.2), so the provenance record contains a **false attestation**. |

**1 of 8.** HIGH.

**Phase N classification: PARTIAL** — the evidence-store durability and event-chain work is real and
well-tested; key management, recovery, and the entire supply chain are not.

---

## 6. CI — what actually runs

Six workflows in `.github/workflows/`. All trigger on push to `main`/`develop`/`copilot/**` and PRs
to `main`/`develop`.

| Workflow | Jobs | Enforces a roadmap gate? |
|---|---|---|
| `code-quality.yml` | `secret-scan` (gitleaks), `lint` (flake8), `syntax-check`, `test` (curated list), `import-validation`, `p0-safety-gates` (pre-commit all files), `policy` (conftest/OPA) | Only Phase N "secret scanning". |
| `coverage.yml` | coverage baseline over the same 8-file curated list (`:44-51`) | No |
| `e2e.yml` | supervisor console e2e | No |
| `code-graph.yml` | codebase-memory graph build; degrades gracefully on install failure (`:200`) | No |
| `brain-caps.yml` | `.claude/` cap check | No |
| `skills-lock.yml` | skills drift check | No |

**Finding — the governance core is untested in CI.** `code-quality.yml:113-131` runs a hand-curated
15-file pytest list, and the header comment concedes "the full `tests/` tree has ~10+ pre-existing
stale tests on main". None of the 22 `test_*evidence*` / `test_*axiom*` files (5,482 lines) appear in
that list, nor in `coverage.yml`. Every guarantee in §3 — separation of duties, promotion authority,
champion-pointer binding, fold-set refusal, fork detection — is verified **only on a developer's
machine**. HIGH.

**Positive — policy-as-code exists and is real.** The `policy` job (`code-quality.yml:208-238`)
installs conftest v0.68.2 with a **checksum-verified** download (`CONFTEST_SHA256`, `:226-233`) — the
only pinned-by-digest supply-chain step in the entire repo — then runs `conftest verify` plus fixture
acceptance tests. Three Rego policies (`.ci/policy/rego/`): `broker_execution_test_coverage`,
`live_arm_review_evidence` (requires a checklist file or `policy:live-arm-reviewed` label on any PR
touching the four live gates, `state_engine.py`, or `agent_runtime/policy.py`), and
`model_promotion_gate_evidence`. These gate **human commits**, not runtime capability — a useful and
correctly-scoped control, but not a substitute for §4.4 enforcement.

**Absent from CI:** container build, image scan, SBOM generation, image signing, dependency
vulnerability audit, evidence-suite execution, restoration drill.

---

## 7. Classification summary

| Deliverable | Class |
|---|---|
| **Phase G** — 12 workload images | ABSENT (0/12) |
| Phase G — 13 hardening requirements on the 3 shipped images | ABSENT (~1.5/13) |
| Phase G — exit gate (local ≡ isolated worker) | ABSENT (unreachable; digest fabricated) |
| **Phase H** — batch job graph / orchestration | ABSENT |
| Phase H — deterministic aggregator (missing/duplicate fold refusal) | PARTIAL (logic LIVE, no isolation) |
| Phase H — resource classes | PARTIAL (signed, never consumed) |
| Phase H — cost controls | ABSENT (1/8) |
| **§4.4** — prohibitions 1, 2, 3, 5, 7 | CONVENTION / static-test only |
| §4.4 — prohibition 4 (halt state) | PARTIAL |
| §4.4 — prohibitions 6, 8, 9 (champion pointer, self-approval, promotion) | LIVE within ledger trust model, simulated principals |
| §4.4 — `may_promote` capability field | ABSENT (8 booleans for 9 prohibitions) |
| **Credentials** — hardcoded secrets / keys in repo | NONE FOUND (pass) |
| Credentials — env-allowlist containment | LIVE (operator subprocess only) |
| Credentials — research-process env isolation | ABSENT |
| Credentials — CI secret scanning | LIVE (default rules only) |
| **Phase N** — failure injection | PARTIAL (8/17 tested) |
| Phase N — recovery | PARTIAL (3/8) |
| Phase N — key management | PARTIAL (1 LIVE, 4 PARTIAL, 2 ABSENT) |
| Phase N — supply chain | ABSENT (1/8) |
| **CI** — roadmap gate enforcement | ABSENT (evidence suite unrun) |
| Evidence lanes wired outside `tests/` | 1 of 7 (`risk_target` only); other 6 ISOLATED |

---

## 8. GAP REGISTER

Ranked by **security severity first**, then dependency order. Effort: S ≤ 1 day · M ≤ 1 week ·
L > 1 week.

### Tier 1 — Security-critical

| # | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **G1** | **No offline recovery key; operator private key unencrypted at rest.** Loss or theft of `~/.config/axiom/operator_ed25519.pem` is total and unrecoverable (`axiom_operator_credential.py:100` `NoEncryption()`). | Add a passphrase-protected second key generated offline, recorded in the trust anchor as a recovery principal. Change `init` to `BestAvailableEncryption` with a prompted passphrase (`Ed25519Signer.from_private_pem` already accepts one, `signing.py:111-115`). Document the offline procedure. | S | — |
| **G2** | **No persistent revocation list.** `TrustStore.revoke` is in-memory; the on-disk anchor has no `revoked_at` (`axiom_operator_credential.py:107-112`). A leaked key cannot be revoked across a restart. | Add `revoked_at` / `valid_until` to the trust-anchor schema and a `trained_data/axiom/revocations.jsonl` CRL; load it in `load_operator_trust` and in every `TrustStore` construction. Add a `revoke` subcommand. | S | G1 |
| **G3** | **Hard-cutover key rotation.** `init --force` replaces the key and rewrites the anchor with one entry — previously signed history stops verifying. | Make the trust anchor a **list** of keys with overlapping validity windows (the library already supports this — `TrustedKey.permits`, `signing.py:46-52`). Add `rotate` that appends a new key with an overlap window and schedules the old one's `valid_until`. | S | G2 |
| **G4** | **No signing audit log.** No record of what was signed, by which key, when. | Wrap `Ed25519Signer.sign` / `sign_digest` to append `{key_id, payload_type, payload_digest, created_at}` to an append-only, fsync'd `signing_audit.jsonl`, reusing the atomic-write discipline in `store.py:122-176`. | S | — |
| **G5** | **`container_digest` is a fabricated attestation.** Five slices sign a hash of an ASCII literal (§0.2). Signed provenance asserting a container that does not exist is worse than an absent field — it will read as verified once real containers arrive. | Immediate: make the placeholder *explicit and refusable* — resolve from a `AXIOM_CONTAINER_DIGEST` env var and **fail closed** when unset outside a `--local-inprocess` flag that stamps the manifest with an unambiguous `local-inprocess` marker the importer treats as non-promotable. | S | — |
| **G6** | **Prohibitions 1/2/3/5/7 are convention.** No process, uid, container, filesystem, or env boundary between a research worker and broker credentials, order placement, signing keys, LiveGate, or model dirs. | Phase 1 (S): apply the `_build_child_env` pattern (`runner.py:38-52`) — the one working boundary in the repo — to every research entrypoint, so workers launch with an env allowlist that excludes `OANDA_*` and `ANTHROPIC_API_KEY`. Phase 2 (M): run workers as a separate uid with the read-only training root from `export_training_root` (which already exists but is ISOLATED). Phase 3 (L): actual containers — see G10. | S → M → L | G10 for full |
| **G7** | **Evidence/governance test suite does not run in CI.** 5,482 lines covering every §4.4 guarantee are unenforced on push. | Add an `evidence-governance` job running `pytest tests/test_*evidence* tests/test_axiom_* tests/test_data_platform.py tests/test_evidence_*` and make it required. These are self-contained (`tmp_path`, no external services) — cheap and fast. | S | — |
| **G8** | **Clock-skew rails untested.** `MAX_SIGNATURE_EVENT_SKEW` / `MAX_EVENT_FUTURE_SKEW` / `verify_disposition_timing` (`signing.py:24-25,240-259`) have zero test coverage despite being wired into `event_store.py:53`. | Add three tests: signature-vs-event skew beyond 5s rejected; event beyond 30s ahead of receipt rejected; a key untrusted at receipt time rejected. | S | G7 |
| **G9** | **`ScannerConfig.oanda_api_key` — dead credential-shaped field** (`config.py:741`), no reader, no writer, at risk of capture by a future `asdict()` dump. | Delete the field. If a broker key ever needs to reach `ScannerConfig`, pass it at call time as `factory.py:53-55` already does. | S | — |

### Tier 2 — Structural prerequisites

| # | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **G10** | **Zero of the 12 workload images.** No isolation substrate exists, so G6 phase 3 and the Phase G exit gate are both blocked. | Build **one** image first — `axiom-risk-target`, the only wired lane. Requirements in order: digest-pinned base, non-root `USER`, hash-pinned deps (`pip-compile --generate-hashes`), no `scripts/`, no `.claude/`, no broker env, `HEALTHCHECK`, read-only rootfs with an explicit writable `tmpfs`. Then templatize. | M (first) → L (all 12) | G5 |
| **G11** | **Ephemeral, co-located role keys.** `build_slice_identities` generates producer + importer + verifier + operator + promotion-service keys in one process; separation of duties is nominal at runtime. | Split identity provisioning from execution: producer key supplied to the worker (env/file), importer + verifier + operator + promotion keys held only by the local authority process. Refuse to construct a `SliceIdentities` holding more than one role's private key in a worker context. | M | G1-G3, G10 |
| **G12** | **No supply-chain controls** — no SBOM, no vuln scan, no image signing, no reproducible build, no restricted index. | Add to CI: `pip-audit` on `requirements*.txt`; `syft` SBOM + `trivy` image scan on every built image; `cosign sign` with a keyless or KMS identity; hash-pinned requirements. Mirror the checksum-verification discipline already used for conftest (`code-quality.yml:226-233`). | M | G10 |
| **G13** | **`may_promote` capability field missing.** Eight booleans for nine prohibitions. | Add `may_promote_to_shadow_or_live: Literal[False]` to `CapabilityProfile` (`models.py:176-188`) and include it in `_authority_free` across all six lanes. Note this is completeness, not enforcement — see G14. | S | — |
| **G14** | **`_authority_free` is tautological.** Eight `Literal[False]` fields cannot be anything else, so the check certifies a promise. | Re-point the check at observable facts: assert the worker process's effective uid, that `OANDA_*` is absent from its env, that its filesystem root is read-only, and that `network_endpoints == ()` matches an actual egress policy. Record those as `SafetyAssertion`s derived from measurement, not declaration — today `worker.py:80-84` emits three hardcoded `passed=True` assertions. | M | G6, G10 |
| **G15** | **LiveGate files absent from the producer import ban** (test `:30-34`), the weakest of the nine. | Add `src.equity.live_gate`, `src.crypto.crypto_live_gate`, `src.crypto.crypto_carry_live_gate`, `src.equity.track_b_live_gate`, `src.agent_runtime.policy` to `forbidden` in all six `*_worker_no_authority.py` tests. | S | — |

### Tier 3 — Phase H / N completeness

| # | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **G16** | **No cost controls** (1/8). No ceiling, quota, idle termination, checkpointing, or cost in evidence. | Add `cost_ceiling_usd` to `JobManifest` and `observed_cost_usd` to `EvaluationReport`; a per-campaign trial ledger; an idle watchdog. Defer checkpointing until real long jobs exist. | M | G10, G17 |
| **G17** | **No job graph.** Six stages are in-process function calls. | Do **not** start with Kubernetes or Ray (roadmap §19 says so explicitly). Start with a signed-manifest queue on the local filesystem plus a worker that consumes one manifest and writes one package — the six stages already exist as functions and mostly need a process boundary, not a rewrite. | L | G10, G11 |
| **G18** | **Resource classes never consumed.** Perfect enum, hardcoded `CPU_SMALL` at every site. | Have the G17 dispatcher read `JobManifest.resource_class` and refuse a job whose declared class the executor cannot honour. | S | G17 |
| **G19** | **Recovery 3/8** — no candidate cache rebuild, no job resume, no signed backups, no restoration drill. | Add `export_signed_backup` / `restore_from_backup` to `EvidenceStore` (its atomic-write and rebuild machinery is already the hard part), plus a scheduled CI job that restores into a clean container and verifies byte-equivalence against `rebuild_indexes`. | M | G7, G10 |
| **G20** | **Failure injection 8/17.** Missing: worker termination, dashboard outage; partial: corrupt manifest, stale data, registry outage, object-store outage, model-load failure, disk full. | Add the nine missing/partial injections against the real `EvidenceStore` and real workers — the no-mock policy makes these straightforward with `tmp_path`, `os.kill`, and a full-`tmpfs` fixture. | M | G7 |
| **G21** | **`export_training_root` is ISOLATED.** A real `0o555`/`0o444` filesystem boundary (`platform.py:171-188`) with zero production call sites. | Route the wired `risk_target` slice through an exported read-only training root instead of raw partition bytes. Cheapest available upgrade from convention to structure for prohibition 7. | S | — |
| **G22** | **No `.gitleaks.toml`.** Default rules only; project-specific shapes (OANDA `32hex-32hex`, `ed25519:` key IDs, the trust-anchor `public_key_b64` format) undetected. Non-`main`/`develop`/`copilot/**` branches unscanned. | Add a `.gitleaks.toml` with those patterns; widen the push trigger to `**`. | S | — |

### Dependency order for execution

```
G1 → G2 → G3 ─┐
G4, G9, G13, G15, G21, G22 (independent, all S)
G7 → G8, G20
G5 → G10 → G11 → G17 → G18
        └→ G12, G14, G19
G6 phase 1 (S, independent) → phase 2 (M, needs G21) → phase 3 (L, needs G10)
G16 ← G10, G17
```

**Recommended first sprint (all S, no dependencies, highest security return):**
G7 (run the tests), G5 (stop the false attestation), G6-phase-1 (env allowlist for research
entrypoints), G1+G2 (recovery key + persistent CRL), G9 (delete the dead credential field),
G15 (close the LiveGate import hole), G22 (project secret patterns).
