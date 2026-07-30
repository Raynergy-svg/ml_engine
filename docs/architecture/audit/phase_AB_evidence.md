# Phase A / Phase B evidence audit — wiring truth vs roadmap claims

**Audited:** 2026-07-30 · **Method:** read-only. Every claim below was re-derived from files
read in this session; no roadmap status line was accepted as evidence. Integration greps
excluded `src/evidence/` and `tests/` before anything was called "wired".

**Checkout audited:** `/home/user/ml_engine` (Linux). The production launchd definitions in
`scripts/axiom_launchd/*.plist` target `/Users/buddy/Documents/ml_engine` (macOS). This audit
can verify that the *wiring exists and is correct*; it cannot verify from here that the
operator's LaunchAgents are loaded. That distinction is tagged per claim.

---

## 0. Test results — REAL output

The environment's default interpreter has neither `pytest` nor `pydantic` installed
(`requirements.txt:88` pins `pytest==9.0.2`, `:112` pins `pydantic==2.12.5`), and the system
`cryptography` (Debian `/usr/lib/python3/dist-packages`) is broken — importing it raises
`pyo3_runtime.PanicException: ModuleNotFoundError: No module named '_cffi_backend'`. I
installed `pydantic==2.12.5`, `pytest==9.0.2`, `cryptography`, `cffi` into a scratchpad
`--target` dir and ran with `PYTHONPATH` prefixed. No project file was modified.

```
$ python -m pytest tests/test_evidence_contracts.py tests/test_evidence_store.py -q
......................................                                   [100%]
38 passed in 2.89s
```

**38 passed, 0 failed.** (The roadmap's 2026-07-12 QA note says 36; the suite has grown by 2.)
Confidence **HIGH** — output produced in this session.

### Exit gate A — one-byte tamper invalidates verification: **PASS**

```
tests/test_evidence_contracts.py::test_one_byte_payload_change_invalidates_signed_object PASSED
tests/test_evidence_contracts.py::test_one_byte_signature_change_is_detected PASSED
```

Both the payload and the signature bytes are covered independently, plus
`test_signature_metadata_is_cryptographically_bound` (key_id / payload_type / digest are
inside the signed preimage, so metadata cannot be swapped). Real Ed25519 via
`cryptography`, no mocks. Confidence **HIGH**.

### Exit gate B — destroy indexes → rebuild → byte-equal: **PASS**

```
tests/test_evidence_store.py::test_destroyed_indexes_rebuild_to_byte_equivalent_state PASSED
tests/test_evidence_store.py::test_injected_fork_is_detected_during_rebuild PASSED
tests/test_evidence_contracts.py::test_event_chain_reconstructs_current_state_from_ledger_only PASSED
```

The rebuild test (`tests/test_evidence_store.py`, body read this session) advances a package
to CHAMPION, writes a champion pointer, captures `current_index_bytes()`, then does a real
`shutil.rmtree(store.root / "indexes")` and asserts `rebuild_indexes() == before` **and** that
no `.tmp-*` residue survives. This is genuine byte-equality against real disk via `tmp_path`.
Confidence **HIGH**.

**Both roadmap exit gates are empirically satisfied at the library level.** Neither gate says
anything about production wiring, which is where the real gap is.

---

## 1. Structural finding: the roadmap's own QA correction is now wrong in both directions

The 2026-07-12 QA correction (`AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md:581-592`) claims:

> "only `DatasetManifest` plus `canonical.py`/`hashing.py`/`signing.py` have production
> callers (via `src/data_platform/`). `EvidenceStore` (Phase B, `store.py`), `event_store.py`,
> `indexes.py`, `importer.py`, `transition_policy.py`, and 7 of the 8 contract types have zero
> callers outside `src/evidence/` and `tests/`"

Both halves are stale:

- **Understates the store.** `EvidenceStore` now has a real production caller —
  `dashboard/server/axiom_evidence_control.py:46` imports it, constructs it at `:426`, and
  calls `append_disposition` at `:556`. `transition_policy.AuthorityRegistry` is imported at
  `:47`. Seven read-only cockpit views are registered at
  `dashboard/server/training_cockpit.py:71-77`. Since 2026-07-12 the repo also grew **seven
  domain evidence lanes** (`src/evidence/{risk_target,hedge_eval,equity_research,
  crypto_momentum,crypto_carry,track_b,execution_cost}/`, 8-9 modules each) that did not
  exist when that note was written.
- **Overstates `DatasetManifest`.** The named production consumer is
  `src/data_platform/platform.py:329,540` — but `DataPlatform`, `IngestionWriter`,
  `TrainingDataView` and every other `platform.py` class have **zero importers outside
  `src/data_platform/` and `tests/`** (verified: the only hits are
  `tests/test_data_platform.py:12,18,209` and `tests/test_forward_capture_clock_skew.py:11`).
  The `src/data_platform` module that *is* live is `forward_capture.py`, and it imports only
  `StrictContract`, `content_digest`, `sha256_bytes`, `TrustStore` — **not** `DatasetManifest`.
- **`indexes.py` does not exist.** `ls src/evidence/indexes.py` → No such file. The
  functionality lives in `store.py:627-731` (`_index_payload_unlocked`, `_rebuild_indexes_
  unlocked`, `rebuild_indexes`, `current_index_bytes`). Behaviourally complete; the roadmap
  deliverable path is absent.

Confidence **HIGH** on all three (direct greps + `ls` this session).

---

## 2. The one production path that reaches the evidence layer

```
scripts/axiom_launchd/com.axiom.api.plist  (KeepAlive, RunAtLoad, AXIOM_CONTROL_ENABLED=1 at :24-25)
  → uvicorn dashboard.server.app:app             (app.py:39 includes training_router)
    → dashboard/server/training_api.py
       ├─ GET  /  → training_cockpit()            :177-180  — NO _require_enabled(), always on
       │    → training_cockpit.read_training_cockpit()  :868
       │      → 7 read-only lane views             :71-77   (store indexes/current.json + verdicts/)
       ├─ POST /run     → run_training()           :185-186 — _require_enabled() + signed operator credential
       │    → axiom_evidence_control.EvidenceControlPlane.prepare_run()  :441
       │      → src.evidence.risk_target.slice.run_risk_target_evidence_slice  (imported :49)
       │         → manifests.py {CapabilityProfile:55, DatasetManifest:97, JobManifest:144}
       │         → worker.py {EvaluationReport:122, EvidencePackage:196, DispositionEvent:219}
       │         → local_import.py → importer.build_import_verdict:403 → store.write_verdict
       │         → store.write_package / append_disposition → event_store.reconstruct_disposition
       └─ POST /promote → relay_promotion()        :247-248 — relays operator-signed DispositionEvent
            → EvidenceControlPlane :556 → store.append_disposition
```

Secondary manual entry point: `scripts/run_risk_target_evidence_slice.py:35,90` (CLI, documented
at `README.md:205-206`, **not** referenced by any launchd plist or shell script — verified).

Three fail-closed gates sit on that path:
1. `training_api.py:53-59` — `AXIOM_CONTROL_ENABLED` must be truthy. Set to `1` only in
   `com.axiom.api.plist:24-25`; absent from every other launcher.
2. `axiom_evidence_control.py:65,418` — `trained_data/axiom/operator_trust.json` must exist or
   `OperatorTrustUnavailable` → HTTP 503 (`training_api.py:66-68`). This path is **gitignored**
   (`.gitignore:268`), so its absence here proves nothing about the operator machine.
3. Per-request signed operator credential with a ≤300s window and single-use nonce
   (`axiom_evidence_control.py:60-62,229`).

**Nothing writes evidence automatically.** There is no scheduled producer, no training-run hook,
no post-trade hook. Every EvidencePackage in existence must have been minted by an operator
clicking a dashboard button or running the one CLI script. Confidence **HIGH**.

**Terminal state is QUARANTINED.** `risk_target/local_import.py:456` is the last transition;
the run cannot reach CHAMPION. `write_champion_pointer` (`store.py:584`) has **zero callers
anywhere outside `store.py` and `tests/test_evidence_store.py:634`** — the seven
`local_import.py` hits are `capability.may_write_champion_pointer is False` assertions, i.e.
the *denial* of that authority, not a use of it. Confidence **HIGH**.

---

## 3. Contract classification (8)

| Contract | Class | Constructed at | Reached by | Gap |
|---|---|---|---|---|
| `DatasetManifest` | **PARTIAL** | `risk_target/manifests.py:97` | dashboard `POST /run`; CLI script | Only the risk-target lane. `platform.py:329,540` — the roadmap's cited consumer — is itself dead (no importer outside `src/data_platform/`+tests). No dataset in the real training pipeline has a manifest. |
| `StrategyManifest` | **ISOLATED** | `crypto_momentum/manifests.py:170` only | nothing | The one lane that builds it (`crypto_momentum`) has zero production callers. Verified: no `StrategyManifest` reference outside `src/evidence/` + `tests/`. |
| `JobManifest` | **PARTIAL** | `risk_target/manifests.py:144` (+6 other lanes, all isolated) | dashboard `POST /run` | No real training job (`scripts/run_full_training.sh`, `transformer_trainer.py`) emits one. |
| `EvaluationReport` | **PARTIAL** | `risk_target/worker.py:122` | dashboard `POST /run` | Name collision handled: `src/research/gated_harness/report.py:133` aliases a *different* `ResearchEvaluationReport`; the two are unrelated (`report.py:24-30`). The gated harness does **not** emit the formal contract. |
| `EvidencePackage` | **PARTIAL** | `risk_target/worker.py:196` | dashboard `POST /run` | Only reference outside the module is a docstring (`gated_harness/report.py:30`). No trained model artifact is packaged. |
| `CapabilityProfile` | **PARTIAL** | `risk_target/manifests.py:55` | dashboard `POST /run` | All six `may_*` fields are `Literal[False]` (`models.py:181-188`) — structurally correct, but only asserted inside the evidence lanes; no runtime component checks a profile before loading a model. |
| `DispositionEvent` | **PARTIAL** | `risk_target/worker.py:219`, `risk_target/local_import.py:313` | dashboard `POST /run` and `POST /promote` | Ledger never advances past QUARANTINED. No SHADOW / OPERATOR_APPROVED / CHAMPION / RETIRED event is ever produced by any code path. |
| `LocalImportVerdict` | **PARTIAL** | `importer.py:16` via `risk_target/local_import.py:403` | dashboard `POST /run` | Six other lanes' `local_import.py` call `build_import_verdict` but are unreachable. |

**Contracts: 0 LIVE · 1 ISOLATED · 7 PARTIAL · 0 ABSENT.**

Every PARTIAL shares the identical gap shape: *reachable, but only through one operator-triggered
route, for one of seven lanes, terminating at QUARANTINED, with no automatic producer.*

---

## 4. Deliverable-file classification (8)

| File | Class | Production caller (file:line) | Entry point |
|---|---|---|---|
| `canonical.py` | **LIVE** | `src/crypto/research_store.py:12`; `src/equity/research/partitioned_store.py:12`; `src/training/performance/{metrics.py:19,rl_dataset.py:16,partition_store.py:16}`; `src/research/gated_harness/{report.py:11,preregistration.py:10,temporal_splits.py:10}`; `src/training/risk_target_cache.py:13` | `src/crypto/data_layer.py:197,247,262,309` (live crypto data layer); `src/equity/research/pit_text_loader.py:433`; `scripts/profile_phase_f_training.py:25,27,37` |
| `hashing.py` | **LIVE** | `src/data_platform/forward_capture.py:19` (used at :207,214,259,286,330,356,373); `dashboard/server/{training_cockpit.py:23,training_jobs.py:15,training_api.py:18}` | `scripts/run_tick_capture.py:33` ← `com.buddy.tick_capture.plist:11`; `scripts/run_exposure_history_capture.py:29` ← `com.buddy.exposure_history.plist:17`; dashboard `GET /` (unflagged) |
| `store.py` | **PARTIAL** | `dashboard/server/axiom_evidence_control.py:46` → ctor `:426`, `append_disposition` `:556`; `training_api.py:250` (`EvidenceStoreError`) | dashboard `POST /run` + `POST /promote` (flag + credential gated). Read side is live-unflagged via `training_cockpit.py:71-77`. `write_champion_pointer:584` / `load_champion_pointer:616` have **zero** non-test callers. |
| `signing.py` | **PARTIAL** | `src/data_platform/forward_capture.py:160` constructs a **default-empty** `TrustStore()` — no signature is verified on that path; `axiom_evidence_control.py:45` (`Ed25519Signer`, `TrustedKey`, `TrustStore`); `platform.py:25` (dead module); `scripts/axiom_operator_credential.py:37` | Real signing only via the credential-gated dashboard routes and a manual CLI. The always-on capture path exercises the type, not the crypto. |
| `transition_policy.py` | **PARTIAL** | `dashboard/server/axiom_evidence_control.py:47` (`AuthorityRegistry`) — sole external caller | dashboard `POST /run`, `POST /promote` only |
| `event_store.py` | **PARTIAL** | Zero direct external callers. Transitive only: `store.py:31` → invoked at `store.py:439,474,491,573,649` | Same as `store.py`. `reconstruct_disposition` never runs outside a dashboard-initiated request. |
| `importer.py` | **PARTIAL** | Zero direct external callers. Transitive via `*/local_import.py` (7 lanes, `:13`/`:44`/`:15`/`:47`) | Only the `risk_target` lane is reachable (`axiom_evidence_control.py:49`, `scripts/run_risk_target_evidence_slice.py:35`). |
| `indexes.py` | **ABSENT** | — | File does not exist. Behaviour implemented in `store.py:627-731` and covered by the passing exit-gate B test. Deliverable path unfulfilled; functionally satisfied. |

**Files: 2 LIVE · 0 ISOLATED · 5 PARTIAL · 1 ABSENT.**

**Overall (16 items): 2 LIVE · 1 ISOLATED · 12 PARTIAL · 1 ABSENT.**

Note on the operator's "shadow" framing: strictly, only `StrategyManifest` and `indexes.py`
fail outright. But the 12 PARTIALs are *thin* — a single operator-triggered route, a single
lane, terminating before promotion. Judged against "is the evidence layer governing anything
the trading system actually does," the honest answer is **no**, and the operator's instinct is
correct even though the literal caller count is non-zero.

---

## 5. Six unreachable lanes

`hedge_eval`, `equity_research`, `crypto_momentum`, `crypto_carry`, `track_b`, `execution_cost`
each ship a full 8-9 module lane (`manifests / worker / evaluation / local_import / slice /
dashboard / models`) with a dedicated slice test and a `*_evidence_worker_no_authority` test.
Their **only** production reference is the read-only cockpit view registered at
`training_cockpit.py:72-77` — which reads `indexes/current.json` and returns
`{"available": false, "reason": "no evidence index on disk"}` when nothing has run
(`risk_target/dashboard.py:22-23`). No `*/slice.py` from those six is imported outside
`src/evidence/` and `tests/` (verified by grep). Confidence **HIGH**.

## 6. On-disk state

`trained_data/evidence/` and `trained_data/forward_capture/` do **not** exist in this checkout
and are **not** gitignored (`git check-ignore` returns nothing for either), while
`trained_data/` is otherwise substantially committed (156 entries, incl. an 85 KB
`trade_journal_rl.json`). `trained_data/axiom/` **is** gitignored (`.gitignore:268`).

Interpretation, **MEDIUM confidence**: this is consistent with the evidence store never having
been populated, but an untracked directory on the operator's Mac would look identical from
here. The cheap disambiguator is `GET /axiom/training` on the operator's box — if the seven
lane views report `available: false`, nothing has ever run. I did not have access to resolve it.

---

## 7. GAP REGISTER

Effort: **S** ≤ 1 day · **M** 2-5 days · **L** > 1 week. Ordering is a strict dependency chain
within each tier; tiers may be parallelised.

### Tier 0 — unblock (nothing below matters until these land)

| # | Gap | Work to make it LIVE | Effort |
|---|---|---|---|
| G1 | No automatic producer. Every contract is PARTIAL for the same reason: the only producer is an operator-clicked route. | Add a post-training hook in the real training path — `src/training/trainers/transformer_trainer.py` (or `scripts/run_full_training.sh`'s completion step) must construct `DatasetManifest` + `JobManifest` + `CapabilityProfile` from the run it just finished, emit an `EvaluationReport` from `trainer.metrics`, package the `.keras`/`.meta.pkl`/`.arch.json` artifacts into an `EvidencePackage`, sign it, and call `EvidenceStore.write_package`. Consumer: `local_import` → QUARANTINED. Entry point: the existing `run_full_training.sh` invocation. | **L** |
| G2 | Operator trust material absent/unverified — `trained_data/axiom/operator_trust.json` gates every write route (`axiom_evidence_control.py:418`). | Run `scripts/axiom_operator_credential.py` to mint and install the operator key; verify `GET /axiom/training` stops reporting `available: false`. Then record on-disk proof in NOTES. | **S** |
| G3 | `indexes.py` ABSENT as a deliverable. | Either (a) extract `store.py:627-731` into `src/evidence/indexes.py` and re-import, or (b) amend the roadmap deliverable list to state the index projection lives in `store.py`. (b) is correct — the behaviour is tested and byte-deterministic; the file path is bookkeeping. | **S** |

### Tier 1 — close the lifecycle (depends on G1+G2)

| # | Gap | Work to make it LIVE | Effort |
|---|---|---|---|
| G4 | `DispositionEvent` never leaves QUARANTINED; `store.write_champion_pointer` has zero callers. Phase L is entirely unbuilt. | Build the local promotion service (roadmap §15 steps 1-11). Producer: a `src/evidence/promotion/` module that resolves the disposition head, re-verifies package+artifact digests, requires an operator-signed `OPERATOR_APPROVED` event, runs a load smoke test, then calls `write_champion_pointer`. Consumer: `gates._load_transformer` must resolve its artifact **through** `load_champion_pointer(lane_id)` instead of globbing `trained_data/models/{PAIR}/`. Entry point: a new `POST /axiom/training/promote-champion` behind the same credential gate. **This is the single highest-value item — until a runtime component *reads* a champion pointer, the evidence layer governs nothing.** | **L** |
| G5 | `CapabilityProfile` is asserted only inside the evidence lanes; no runtime component checks one. | Consumer: `src/scanner/gates.py` `_load_transformer` refuses any artifact whose package lacks a `CapabilityProfile` with all `may_*` false. Pairs naturally with G4 (same read path). | **M** |
| G6 | `signing.py` runs with a default-empty `TrustStore()` on the only always-on path (`forward_capture.py:160`) — the type is exercised, the crypto is not. | Load the real producer trust store in `ForwardCaptureService`, so forward-capture partitions are signature-verified rather than merely digested. Depends on G2 for key material. | **M** |

### Tier 2 — breadth (depends on G1; independent of G4)

| # | Gap | Work to make it LIVE | Effort |
|---|---|---|---|
| G7 | Six lanes unreachable (`hedge_eval`, `equity_research`, `crypto_momentum`, `crypto_carry`, `track_b`, `execution_cost`). | Generalise `axiom_evidence_control.py:48-52` from a hard-coded `risk_target` import to a lane registry keyed by `lane_id`, mirroring the `ReaderSpec` table already at `training_cockpit.py:71-77`. Add one `scripts/run_<lane>_evidence_slice.py` per lane. This also promotes `StrategyManifest` from ISOLATED (crypto_momentum is the only lane that builds one). | **M** |
| G8 | `src/data_platform/platform.py` (`DataPlatform`/`IngestionWriter`/`TrainingDataView`) — the roadmap's cited `DatasetManifest` consumer — has zero non-test importers. Phase C is built and unused. | Route one real ingestion path (start with tick capture, which is already scheduled) through `IngestionWriter` so `DatasetManifest` describes production data rather than slice fixtures. Prerequisite for G1 producing honest manifests. | **L** |
| G9 | `src/research/gated_harness` emits `ResearchEvaluationReport`, not the formal `EvaluationReport` contract (`report.py:24-30,133`). Two parallel evaluation schemas. | Have the gated harness additionally emit the formal contract, or make `ResearchEvaluationReport` a documented projection of it. Blocks any claim that "one canonical harness is used" (§20). | **M** |
| G10 | No evidence dir on disk, no scheduled slice run — the exit gates are proven in `tmp_path`, never against the real store. | Add `com.axiom.evidence_slice.plist` (or extend `com.buddy.forward_daily`) to run one lane nightly, and add a startup assertion that `rebuild_indexes()` byte-matches `current_index_bytes()` on the *live* store. Converts both exit gates from unit-tested to continuously-verified. | **S** |

### Recommended order

`G3 → G2 → G8 → G1 → G10 → G4 → G5 → G7 → G6 → G9`

G3/G2 are cheap and unblock everything. G8 before G1 so the first real packages carry honest
dataset manifests rather than fixtures. G4 is the load-bearing item — it is what converts the
evidence layer from a ledger nobody reads into the thing that decides which artifact loads.

---

## 8. Confidence summary

| Claim | Confidence | Basis |
|---|---|---|
| 38/38 tests pass; both exit gates pass | **HIGH** | Run in this session; test bodies read |
| `indexes.py` absent; behaviour in `store.py` | **HIGH** | `ls` + grep this session |
| `write_champion_pointer` has zero production callers | **HIGH** | Exhaustive repo grep |
| Only `risk_target` of 7 lanes is reachable | **HIGH** | Exhaustive `*.slice import` grep |
| `platform.py` classes have zero production importers | **HIGH** | Exhaustive grep; only test hits |
| No automatic/scheduled evidence producer exists | **HIGH** | plist + shell + grep review |
| `canonical.py` / `hashing.py` are LIVE | **MEDIUM** | Call chain to a scheduled plist is verified; that the LaunchAgents are loaded on the operator's Mac is not verifiable from this checkout |
| Evidence store has never been populated | **MEDIUM** | Dir absent and not gitignored, but an untracked dir on the operator machine is indistinguishable from here |
