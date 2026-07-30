# Phase C / Phase D data-platform audit — evidence-based

**Audited:** 2026-07-30 · **Commit:** `074758d` · **Auditor:** Data Engineer sub-agent
**Scope:** `docs/architecture/AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md` §5 Stages 1–3 (L225–320),
§7 Phase C (L674–702), §7 Phase D (L704–742), including the QA correction at L715–732.
**Mode:** read-only. No code was changed.

---

## 0. Method and the one caveat that bounds every claim in this document

Every claim below was re-derived by reading files in this session and is cited `file:line`.
Neither the roadmap's `**Implementation status (2026-07-11): COMPLETE.**` lines
(`AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md:676`, `:706`) nor the QA correction (`:715`) was taken
on trust; both are graded against disk below, and both turn out to be partly wrong in
*different directions*.

**Environment caveat (HIGH confidence, load-bearing for §3 and §6).** This audit ran in a Linux
container at `/home/user/ml_engine` holding a **git checkout at `074758d`**, not the operator's
macOS host. Consequences that must not be read as findings about the operator's machine:

| Observation here | What it does / does not prove |
|---|---|
| all file mtimes are `Jul 30 15:18` (clone time) | filesystem mtime is **useless** for freshness — every freshness claim in §6 comes from timestamps *inside* file content |
| `trained_data/ticks/` absent | gitignored (`.gitignore:303`) — **absence proves nothing** |
| `trained_data/hedge/exposure_history.jsonl` absent | **NOT** gitignored (verified `git check-ignore` → rc=1) and not tracked — absent from the repo, unknown on host |
| `axiom-data/` absent | **NOT** gitignored — absent from the repo, unknown on host |
| `.claude/state.json` says `halted:false`, `last_updated 2026-05-05` | this is the *committed* copy; CLAUDE.md says live state is `halted:true`. Do not read runtime state from this checkout |
| `launchctl` does not exist here | LaunchAgent load state is **unverifiable from this environment**; §3 therefore grades *definitions and their referenced targets*, which is verifiable, and reports load state only as a secondary source |

Everything about **code wiring** (does a producer exist, does a consumer exist, is an exception
swallowed) is fully verifiable here and is stated at HIGH confidence.

---

## 1. Headline verdict

| Classification | Count | Items |
|---|---|---|
| **LIVE** (real producer AND real consumer in the running system) | **0** | — |
| **PARTIAL** (producer wired, no canonical consumer; legacy path still the source of record) | **7** | tick/spread capture, exposure capture, crypto capture, Track B filing capture, `DataPlatform`+`IngestionWriter`, `contracts.py`, `io.py` |
| **ISOLATED** (exists, zero production consumers) | **6** | `trade_journal_events.py`, `TrainingDataView`, `export_training_root`, `FeatureWriter`, `ControlStateStore`, P2 readiness state |
| **ABSENT** (deliverable does not exist or has no call site at all) | **7** | trade-entry-context capture, fill-ladder capture, Track B rebalance capture, hedge forward capture, `scripts/capture_track_b_new_filings.py`, `scripts/check_risk_target_p2_readiness.py`, `axiom-data/` tree |

**Biggest blocker (HIGH):** nothing anywhere in the repo *reads* `axiom-data/` for training or
gating. `TrainingDataView`, `export_training_root` and `FeatureWriter` — the entire read side of
Phase C — have zero callers outside `tests/test_data_platform.py`. Until a training job consumes a
canonical snapshot, every capture family is a write-only limb and every "canonical is the
authority" comment in the code is aspirational, not descriptive.

**Second blocker (HIGH, and not mentioned anywhere in the roadmap):** the canonical append path is
**O(N²)** in stream length. `ForwardCaptureService._capture` re-reads and strict-validates the
*entire* prior stream before every append (`src/data_platform/forward_capture.py:232`, via
`records()` → `platform.iter_history` → `_iter_history_root`), and `IngestionWriter.append_history`
writes **one file per record** (`src/data_platform/platform.py:589`). Tick capture flushes every
30 s per plist (`scripts/axiom_launchd/com.buddy.tick_capture.plist`, `--flush-interval 30`),
i.e. ~2,880 captures/pair/day × 19 pairs. Day 1 alone costs ≈ Σ(1..2880) ≈ 4.1 M file reads +
SHA-256 + canonicalisation *per pair*. The 60-forward-weekday P2 gate is not reachable through
this code path. This is the load-bearing engineering fact of the audit.

---

## 2. `src/data_platform/` — inventory and real production consumers

Grep basis: `grep -rn "data_platform" --include=*.py --include=*.sh --include=*.plist .`
excluding `src/data_platform/` itself and `tests/`.

| Module | LOC | What it provides | Production consumers (outside `src/data_platform/`, `tests/`) | Class |
|---|---|---|---|---|
| `contracts.py` | 120 | `StorageDomain`, `StorageFormat`, `RawObjectReceipt`, `NormalizedVersion`, `FeatureBundle` (pydantic strict) | none directly; consumed by `platform.py` | PARTIAL |
| `io.py` | 86 | `atomic_create` / `atomic_replace` / `exclusive_lock` / `publish_directory` (fsync + flock, correct) | none directly; consumed by `platform.py` | PARTIAL |
| `platform.py` | 967 | `DataPlatform`, `IngestionWriter`, `TrainingDataView`, `FeatureWriter`, `ControlStateStore` | **zero direct**. Reached only transitively via `ForwardCaptureService` | see split below |
| `forward_capture.py` | 700 | `ForwardCaptureService` (8 `capture_*` methods), `P2ReadinessReport`, `capture_best_effort` | `src/crypto/data_layer.py:115`, `src/equity/research/pit_text_loader.py:377`, `scripts/run_tick_capture.py:33`, `scripts/run_exposure_history_capture.py:29`, `dashboard/server/training_cockpit.py:488` | PARTIAL |
| `trade_journal_events.py` | 160 | Hash-chained UPSERT/DELETE event log intended to become the authority behind `trade_journal_rl.json` | **zero — including zero tests** | ISOLATED |

Split of `platform.py` by capability:

| Capability | Symbol | Reached in production? | Class |
|---|---|---|---|
| Immutable raw put | `IngestionWriter.put_raw` (`platform.py:379`) | yes, via `_publish_parquet_version` / `capture_track_b_filing` | PARTIAL |
| Normalized publish | `IngestionWriter.publish_normalized` (`:430`) | yes, via `_publish_parquet_version` (`forward_capture.py:273`) | PARTIAL |
| Append-only history | `IngestionWriter.append_history` (`:568`) | yes, via `_capture` (`forward_capture.py:236`) | PARTIAL |
| Signed snapshot manifests (Stage 3 `DatasetManifest`) | `publish_snapshot` / `_manifest_path` (`:316`, `:546`) | **no caller** | ISOLATED |
| Read-only training root | `TrainingDataView` (`:594`), `export_training_root` (`:174`) | **no caller** | ISOLATED |
| Feature outputs keyed by snapshot+pipeline digest | `FeatureWriter` (`:798`) | **no caller** | ISOLATED |
| Small atomic control state | `ControlStateStore` (`:909`) | **no caller** — notably *not* used to write `p2_readiness.json` | ISOLATED |

**Roadmap grade — Phase C (`:676` "COMPLETE").** The *write* half is real, well-built and
well-tested (15 tests in `tests/test_data_platform.py`, 2 in `tests/test_forward_capture_clock_skew.py`;
real fsync, real `fcntl`, real digest verification, path-traversal guards at `platform.py:230`/`:633`).
The *read* half — the half that makes Stage 3 snapshots mean anything — is unreferenced. "COMPLETE"
is not a fair description of a storage layer nothing reads. **Grade: PARTIAL, not COMPLETE.**

**Roadmap grade — Stage 3 (`:290–306`, `DatasetManifest`).** No production code publishes or
consumes a signed snapshot manifest. The Stage-3 exit condition ("training must never define its
dataset by ad hoc live broker calls") is unmet: `scripts/backtest_harness.py` and the trainers
still read `market_data/` and `trained_data/` directly. **ABSENT in production.**

---

## 3. The seven capture families — canonical write vs. actual source of record

Legend: **Canonical write** = a `capture_*` call reaching `ForwardCaptureService`.
**Source of record** = the path the rest of the system actually reads.

| # | Family | Canonical write exists? | Call site | Failure mode of canonical write | Legacy path (the ACTUAL source of record) | Legacy readers | Canonical readers | Class |
|---|---|---|---|---|---|---|---|---|
| 1 | Tick + spread | yes | `src/data/tick_capture.py:178` | **fail-closed** — not wrapped; `_do_flush` re-buffers and re-raises (`tick_capture.py:436-441`) | `trained_data/ticks/<pair>/<y>/<m>/<d>.parquet` (`tick_capture.py:43,153`) | `src/data/tick_aggregate.py:28`, `src/data/execution_cost_model.py:50,233`, `src/hedge/exposure_history.py:67,166`, `src/data/tick_post_flush.py:166` | only `evaluate_p2_readiness` (never invoked — §5) | PARTIAL |
| 2 | Exposure history | yes | `src/hedge/exposure_history.py:366` | **fail-closed** — `except Exception → log + return None` (`:367-369`); refuses the legacy write too | `trained_data/hedge/exposure_history.jsonl` (`exposure_history.py:68`) | `dashboard/server/data_sources.py:1134,1197`, `dashboard/server/training_cockpit.py:41` | only `evaluate_p2_readiness` | PARTIAL |
| 3 | Full trade-entry context | **NO** | — | — | `trained_data/trade_journal_rl.json`, written whole-file in `src/scanner/execution.py` (§4) | **81 non-test Python files** reference it (top: `execution.py` ×12, `automation/continuous.py` ×6, `training/labels/journal_loader.py` ×5, `automation/trend_journal_sync.py` ×5, `training/rl/offline_sizer.py` ×4, plus 4 TUI screens/caches) | none | **ABSENT** |
| 4 | Fill-book ladders + slippage | **NO** | — | `capture_fill_transaction` (`forward_capture.py:406`) is defined and never called | OANDA transaction handling in `execution.py`; `fullPrice` ladders are **not persisted at all** — 0/18 journal records carry `full_price` or `fill_ladder` | — | none | **ABSENT** |
| 5 | Crypto klines + funding | yes | `src/crypto/data_layer.py:117` (4 call sites pass `capture_metadata`: `:239, :301, :349, :413`) | **fail-closed** — unlinks tmp and raises `RuntimeError` before the cache projection (`data_layer.py:128-133`) | `crypto_cache/binance/{funding,klines}/*.parquet` (`data_layer.py:41,194,259`) | `scripts/refresh_crypto_forward_data.py:26`, all `src/crypto` strategy loaders | none | PARTIAL |
| 6a | Track B **filings** | yes | `src/equity/research/pit_text_loader.py:378` | **double-swallowed** — `capture_best_effort` swallows internally (`forward_capture.py:699`), *and* the call site wraps it in a second `except Exception → logger.warning` (`pit_text_loader.py:389`) | `_write_cache(cache_dir, ...)` text cache (`pit_text_loader.py:362`) | Track B harness / scoring scripts | none | PARTIAL |
| 6b | Track B **rebalance dates** | **NO** | — | `capture_track_b_rebalance` (`forward_capture.py:584`) defined, never called | `trained_data/research/track_b_shadow_ledger.jsonl` via `src/equity/track_b_shadow.py:380 record_shadow_cycle` → plain `_append_jsonl` | `scripts/run_track_b_shadow.py:62`, `src/hedge/hedged_shadow_lane.py:262` | none | **ABSENT** |
| 7 | Hedge raw-vs-hedged forward | **NO** | — | `capture_hedge_forward` (`forward_capture.py:600`) defined, never called | `trained_data/hedge/raw_vs_hedged_ledger.jsonl` + `hedge_decision_log.jsonl` via `hedged_shadow_lane.py:814-815` | `src/hedge/hedge_scorecard.py`, dashboards | none | **ABSENT** |

### 3.1 The roadmap's QA correction is itself wrong

`AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md:715-717` states: *"'all seven families write through the
canonical contract' is technically true (each has a `capture_*`/`capture_best_effort()` call site)"*.

**That is false.** Verified by repo-wide grep for the eight method names, excluding
`forward_capture.py` itself:

- `capture_trade_entry_context` — 0 call sites
- `capture_fill_transaction` — 0 call sites
- `capture_track_b_rebalance` — 0 call sites
- `capture_hedge_forward` — 0 call sites

`git log -S` on each of those four names returns exactly one commit — `ae685fa`, the commit that
*defined* them. They have never had a caller. So the true count is **4 of 8 methods wired, 4 dead**;
by roadmap family, **4 of 7 partially wired, 3 with no canonical write at all**.

The QA correction is also right about something the "COMPLETE" line denies (trade-entry context is
the worst gap), and its "5 of 7 are secondary exception-swallowed mirrors" framing is too harsh in
the other direction: three of the four wired families (tick, exposure, crypto) are genuinely
**fail-closed** — the legacy projection is refused when canonical publication fails. Only Track B
filings is a true swallowed mirror. Both roadmap status blocks should be replaced with §3's table.

### 3.2 Two scheduled jobs point at scripts that do not exist

`scripts/run_forward_capture_daily.py:16-21` shells out to four commands. Two of the four targets
are absent from the repo and absent from all git history (`git log -S` returns nothing for either):

| Job | Command | Target exists? | Accepted return codes | Effect |
|---|---|---|---|---|
| `track_b_filings` | `scripts/capture_track_b_new_filings.py` | **NO** | `{0}` | `python` exits 2 → `hard_failure = True` |
| `track_b_rebalance` | `scripts/run_track_b_shadow.py --refresh-prices` | yes | `{0,1}` | runs |
| `hedge_forward` | `python -m src.hedge.hedged_shadow_lane` | yes | `{0}` | runs, but writes **only** legacy JSONL |
| `p2_readiness` | `scripts/check_risk_target_p2_readiness.py` | **NO** | `{0,3}` | exits 2 → `hard_failure = True`; **no `p2_readiness.json` is ever produced** |

So `com.buddy.forward_daily`, even if bootstrapped and running, exits non-zero every night, never
captures a new Track B filing through the canonical contract, and never writes the readiness file
the dashboard reads. Confidence: **HIGH** (verified by `ls`, `find`, and `git log -S`).

---

## 4. `scripts/axiom_launchd/` — defined vs. loaded vs. effective

`load.sh:41` enumerates 10 labels. All 10 `.plist` files exist. Every plist pins
`WorkingDirectory` to `/Users/buddy/Documents/ml_engine` and interpreter
`/opt/homebrew/Caskroom/miniforge/base/bin/python3` — macOS-only, so **none of them can run in
this environment**, and no `launchctl` state is inspectable here.

Capture-relevant agents:

| Label | Defined | In `load.sh` LABELS | Schedule | Target script exists | On-disk proof of running found here | Effective status |
|---|---|---|---|---|---|---|
| `com.buddy.exposure_history` | yes | yes | `RunAtLoad` + `KeepAlive`, `--loop 900` | yes (`scripts/run_exposure_history_capture.py`) | `trained_data/axiom/` **does not exist** → no `exposure_history_capture.out`; `exposure_history.jsonl` absent | claimed loaded 2026-07-12 (`.claude/NOTES.md:23`) — **unverified from disk** |
| `com.buddy.tick_capture` | yes | yes | `RunAtLoad` + `KeepAlive`, flush 30 s, `ALL_FX` | yes (`scripts/run_tick_capture.py`) | no `tick_capture.out`, no `trained_data/ticks/` | **not verifiable; NOTES says unloaded** |
| `com.buddy.forward_daily` | yes | yes | `StartCalendarInterval` 23:10, `RunAtLoad=false` | yes, **but 2 of its 4 sub-jobs target missing scripts (§3.2)** | no `forward_daily.out` | **not verifiable; and would hard-fail if it ran** |
| `com.buddy.crypto_refresh` | yes | yes | `StartCalendarInterval` day 2 03:20, `RunAtLoad=false` | yes (`scripts/refresh_crypto_forward_data.py`) | no `crypto_refresh.out`; `crypto_cache/` absent | **not verifiable; NOTES says unloaded** |
| `com.buddy.learning_loop` | yes | **deliberately excluded** (`load.sh:26-33`) | Saturday batch | yes | — | intentionally not installed |

Two independent secondary sources (not disk in this session, so **MEDIUM** confidence) agree:
`AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md:730-732` and `.claude/NOTES.md:23` both say
`com.buddy.exposure_history` is bootstrapped and `tick_capture` / `forward_daily` /
`crypto_refresh` are not. Nothing in this checkout contradicts or confirms that.

**Consequence for the P2 gate.** The 60-forward-weekday gate requires ≥60 eligible forward weekdays
for exposure **and** for every required pair's ticks (`forward_capture.py:653-659`). If
`com.buddy.tick_capture` is unloaded, the tick clock has not started for any of the 19 pairs, so the
earliest possible P2 date is ≥60 weekdays (≈12 calendar weeks) after the day it is bootstrapped —
and even then only if §1's O(N²) problem is solved first.

---

## 5. Storage-principle compliance — whole-file JSON-array read-modify-rewrite

Phase C principle (`:698`): *"Historical records use Parquet, JSONL or a database — not repeatedly
rewritten JSON arrays."*

### 5.1 Unbounded violations (the real ones)

| File | Rewrite sites | Bounded? | Readers |
|---|---|---|---|
| `trained_data/trade_journal_rl.json` | `src/scanner/execution.py:3677-3682` (operator veto), `:4045-4062` (per-trade entry, `safe_json_write` + fallback), `:5542-5547` (post-close sync), `:5695-5699` (RL weight re-persist), `:5900-5905` (`_persist_journal`), `:6791-6794` (`close_trade`) | **NO cap** — every write serialises the full array | **81 non-test Python files** (see §3 row 3). Writers: 6 sites in `execution.py` + `automation/trend_journal_sync.py` + `automation/continuous.py` |
| `trained_data/confidence_calibration.json` → `trade_history` | `src/scanner/confidence_calibration.py:392` append; whole-object rewrite via `automation/confidence_calibrator.py:417` | **NO cap found** — grep for a `[-N:]` slice on `_trade_history` returns nothing; currently **690 entries / 37 KB** on disk | `confidence_calibration.py:430-443`, isotonic/Platt refit |

The `trade_journal_rl.json` writes *are* atomic and locked (`src/scanner/automation/safe_json.py`
— real `fcntl` + tmp + `fsync` + `os.rename`), so this is a scalability/contention violation, not a
corruption risk. But 6 writers × 81 readers on one growing JSON array is exactly the coupling that
makes the roadmap's own note ("migrating its format needs its own scoped, coordinated change")
correct. `src/data_platform/trade_journal_events.py` is the *designed* fix for precisely this file
— and it has zero callers **and zero tests**.

### 5.2 Bounded ring buffers — technically the same anti-pattern, materially harmless

| File | Cap | Site |
|---|---|---|
| `trained_data/episodic_memory.json` (313 entries / 145 KB) | `max_episodes=500` | `src/scanner/automation/episodic_memory.py:85,133-141` |
| `trained_data/attention_feedback_log.json` (150 / 36 KB) | `[-150:]` | `automation/attention_feedback.py:182,271` |
| `trained_data/regime_transitions.json` (100 / 24 KB) | `[-100:]` | `automation/regime_broadcaster.py:145,233` |
| `trained_data/health_registry_log.json` (100 / 19 KB) | 100 | health registry |

These are constant-size rewrites. Flag for tidiness, not for remediation priority.

### 5.3 Other Phase C principle checks

| Principle (`:694-702`) | Status | Evidence |
|---|---|---|
| Raw data immutable | **PASS in code** | `atomic_create` uses `O_EXCL` + `os.link`, raising `ImmutableWriteConflict` (`io.py:47-66`) |
| Normalized data versioned | **PASS in code** | `version_id=f"content-{digest}"` (`forward_capture.py:276`) |
| Training snapshots immutable | **N/A** | no snapshot is ever published (§2) |
| Historical records not rewritten JSON arrays | **FAIL** | §5.1 |
| Small control state local atomic JSON | **PASS in code, unused** | `ControlStateStore.write` is CAS-guarded (`platform.py:947-963`); zero callers |
| Feature outputs keyed by snapshot + pipeline digest | **ABSENT** | `FeatureWriter` has zero callers; the live feature contract is still `FEATURE_PIPELINE_VERSION` in `src/core/modular_data_loaders.py` |
| No training job writes into source-data paths | **unverified** — not in scope of this pass | — |

### 5.4 A retention asymmetry that will make canonical and legacy diverge

`TickPersister.prune_older_than` (`src/data/tick_capture.py:206-...`, driven by `--retention-days`)
deletes **legacy** day-partitions only. Nothing prunes or compacts the canonical
`axiom-data/fx/history/...` stream. So the "compatibility projection" is retention-bounded while the
declared "authority" grows without bound — the reverse of the intended relationship, and it
guarantees the two stores disagree about coverage after the first retention window. Confidence: HIGH.

---

## 6. The 60-forward-weekday P2 readiness computation — real, and dead

**Is the computation real?** Yes. `ForwardCaptureService.evaluate_p2_readiness`
(`src/data_platform/forward_capture.py:620-673`) is correct and non-trivial:

- eligible day = `training_eligible` **and** `mode is FORWARD` **and** `observed_at_utc.weekday() < 5`
  (`:629-638`) — so backfilled and weekend rows cannot inflate the gate;
- `training_eligible` is itself invariant-enforced at construction: it must equal
  `FORWARD and context_complete` or the model rejects the record (`:109-111`);
- exposure is read from one explicit source key `"oanda_fx_trend_lane"` (`:645`) with a comment
  saying why, so unrelated portfolio history cannot count;
- the join key is stated explicitly in the report (`:670`);
- default `minimum_trading_days = 60` (`:36`), per-pair blocking reasons emitted (`:653-659`).

**Who consumes it?** Trace, end to end:

| Step | Reality |
|---|---|
| Producer of `trained_data/forward_capture/p2_readiness.json` | **none.** The only intended writer is `scripts/check_risk_target_p2_readiness.py` — **the file does not exist** (§3.2). `ControlStateStore` is not used for it either. Repo-wide grep for `p2_readiness` outside tests returns only the missing-script reference and readers. |
| Reader | `dashboard/server/training_cockpit.py:483-499` — reads the file, validates it into `P2ReadinessReport`, else emits `{"available": false, "ready": false, "error": "P2 readiness report is absent"}` |
| What the reader does with `ready` | `read_control_readiness` (`training_cockpit.py:788-855`) maps it to a **display string only**: `eligibility: "ready_for_implementation" / "accumulating_forward_evidence" / "readiness_unavailable"` (`:842-848`). The sibling field `"trainable": False` (`:840`) is a **hard-coded literal** and does not depend on readiness at all. |
| Any gate, halt, promotion, or training entrypoint that reads it | **none** |

**Plain statement:** the 60-forward-weekday readiness state is **dead telemetry**. It has no
producer, and its single consumer renders a label on a dashboard card next to a hard-coded
`trainable: false`. It blocks nothing, because it *can* block nothing — even a `ready: true` report
would change one string. The roadmap's framing at `:725-728` ("nothing currently consumes that
readiness state to block anything — acceptable today because Risk-target P2 training doesn't exist
yet") is accurate as far as it goes, but understates it: there is also **no writer**, so today the
dashboard permanently displays `readiness_unavailable`. Class: **ISOLATED**.

---

## 7. Actual data on disk

Read from file *content* (mtimes are worthless here — §0).

| Store | Path | Present | Volume | Coverage (from content) | Staleness vs. 2026-07-30 |
|---|---|---|---|---|---|
| **Canonical Phase C tree** | `axiom-data/` | **NO** (not gitignored) | 0 | — | never written in this checkout |
| Forward-capture control state | `trained_data/forward_capture/` | **NO** (not gitignored) | 0 | — | no `p2_readiness.json` ever |
| FX daily factor panels | `market_data/factor/*_D.csv` | yes | 20 pairs + 8 `rates_*.csv`, 4.1 MB | 2014-01-01 → **2026-06-11** (EUR_USD 3,231 rows; USD_JPY 3,248) | **49 days** |
| Risk-target dataset descriptor | `market_data/factor/axiom_risk_target_dataset.json` | yes | — | dict | referenced by `training_cockpit.py:795` |
| Equity PIT | `market_data/equity/` | yes | 26 MB (`sp500_prices.parquet`, 2 PIT fundamentals, universe snapshot) | not sampled this pass | unknown |
| FX ticks (legacy) | `trained_data/ticks/` | **NO** (gitignored) | — | — | unknown on host |
| Crypto cache (legacy) | `crypto_cache/` | **NO** | — | — | unknown on host |
| Exposure history (legacy) | `trained_data/hedge/exposure_history.jsonl` | **NO** (not gitignored) | 0 | — | absent from repo |
| Hedge raw-vs-hedged ledger | `trained_data/hedge/raw_vs_hedged_ledger.jsonl` | yes | **3 rows**, 12.7 KB | all three at `2026-07-08T02:32:38.779–.781Z` — a **single 2 ms run**, not forward history | **22 days**, and only 1 cycle ever |
| Hedge decision log | `trained_data/hedge/hedge_decision_log.jsonl` | yes | 3 rows, 15 KB | same single instant | 22 days |
| Track B shadow ledger | `trained_data/research/track_b_shadow_ledger.jsonl` | yes | **1 row** | `cycle_ts 2026-07-03T01:07:23Z`, `forward_cycle_seq 1` | **27 days**, 1 cycle |
| Crypto momentum shadow | `trained_data/crypto/shadow_momentum_ledger.jsonl` | yes | **1 row** | `2026-07-02T23:18:05Z` | 28 days |
| Crypto carry shadow | `trained_data/crypto_carry/shadow_carry_ledger.jsonl` | yes | **1 row** | `2026-07-07T10:38:22Z` | 23 days |
| Equity cycle ledger | `trained_data/equity/cycle_ledger.jsonl` | yes | 141 rows, 64 KB | `asof` 2026-06-24 → **2026-07-01** | 29 days |
| Trade journal (legacy authority) | `trained_data/trade_journal_rl.json` | yes (tracked) | **18 records**, 85 KB | `2026-04-03T19:40Z` → **2026-05-05T00:42Z** | **86 days** |
| Trained model dirs | `trained_data/models/<PAIR>/` | yes | 17 pairs + joint | — | per CLAUDE.md all transformer artifacts are gap-quarantined |

**Trade-journal context completeness (measured, all 18 records):** `gates` 18/18, `agents` 18/18,
`ridge_features` 18/18, `spread_pips` 18/18, `expected_price` 18/18, `outcome` 18/18 — so on *this*
sample the roadmap's "only a small portion of journal records contained complete agent/gate
context" (`:740`) no longer holds. But `full_price` / `fill_ladder` **0/18** and
`rl_weights_applied` **0/18**. The fill-ladder half of that roadmap sentence — *"real order-fill
ladders existed but were not being consumed by training"* — is confirmed: OANDA `fullPrice`
ladders are not persisted anywhere in this repo. Confidence HIGH on this sample; MEDIUM as a claim
about the operator's larger live journal.

**Forward-capture clock status.** Not one of the seven families has more than a handful of forward
records anywhere in this checkout, and the three shadow lanes that *do* write legacy ledgers have
**1–3 rows each from single one-off runs**. Against a 60-weekday requirement, the honest reading is
that the forward clock has effectively **not started** for any family.

---

## 8. Swallowed-exception register (project-rule violations)

Project rule (`CLAUDE.md` "Code quality non-negotiables", `.claude/rules/improvement.md`
"Silent Exception Prevention"): *no bare `except:` or `except Exception: pass`; always log; always
re-raise or return an error status — callers must know something failed; never swallow errors in
financial paths.*

| # | `file:line` | Handler | Logged? | Caller informed? | Severity | Note |
|---|---|---|---|---|---|---|
| **V1** | `src/data/tick_capture.py:189` | `except Exception: merged = group` | **NO — silent** | no | **HIGH** | On a corrupt/unreadable existing day-parquet, the **entire prior day of ticks for that pair is silently discarded** and overwritten with only the current 30 s batch. Data loss with zero signal. Hardest violation found. |
| **V2** | `src/data_platform/forward_capture.py:699` | `except Exception: logger.exception(...)` in `capture_best_effort` | yes | **no** — returns `None` | MEDIUM | By design ("without changing the calling lane's control flow"), but it means a canonical write can fail forever while the caller reports success. Violates "always re-raise or return error status". |
| **V3** | `src/equity/research/pit_text_loader.py:389` | `except Exception as capture_error: logger.warning(...)` | yes | no | MEDIUM | **Second** swallow wrapped around V2 — belt-and-braces silence on the only Track B canonical path. |
| **V4** | `src/hedge/hedged_shadow_lane.py:199-200` | `except OSError: logger.error(...)` in `_append_jsonl`, then returns normally | yes | **no** | MEDIUM | A failed ledger append is indistinguishable from success. `exposure_history.py:370-381` explicitly compensates by re-reading and verifying; **`hedged_shadow_lane.py:814-815` and `track_b_shadow.py record_shadow_cycle` do not** — families 6b and 7 can silently lose rows. |
| **V5** | `src/hedge/hedged_shadow_lane.py:841` | `except Exception: logger.exception(...); results[strategy]=None` | yes | partially (`None` sentinel) | LOW | Acceptable — a per-strategy sentinel is a real error status. |
| **V6** | `src/scanner/execution.py:3682` | `except Exception as _jw_err: logger.warning(...)` around the veto-journal write | yes | no | MEDIUM | Financial path: a veto that fails to journal is invisible to RL. |
| **V7** | `src/data/tick_capture.py:202` | `except Exception as _hook_err: logger.warning(...)` (post-flush aggregation hook) | yes | no | LOW | Hook is auxiliary. |
| **V8** | `src/data/tick_capture.py:453` | `except Exception as prune_err: logger.warning(...)` | yes | no | LOW | Pruning is best-effort by design. |
| **V9** | `scripts/refresh_crypto_forward_data.py:59` | `except Exception as exc:` → counted into `failed` + `errors` map, non-zero exit | yes | **yes** | none | Correct pattern — cited as the model the others should follow. |

**Clean:** `src/data/tick_capture.py:178` (canonical tick write — deliberately unwrapped, propagates),
`src/hedge/exposure_history.py:366-369` (fail-closed, refuses legacy write),
`src/crypto/data_layer.py:128-133` (fail-closed, unlinks tmp and raises).
**No bare `except:`** was found in any Phase C/D path.

---

## 9. GAP REGISTER

Dependency order is strict: an item cannot be usefully done before its listed dependency.

| ID | Gap | Concrete work — who must write, who must read, what must be scheduled | Effort | Depends on |
|---|---|---|---|---|
| **G0** | **Canonical append is O(N²) and one file per record**; the 60-day gate is unreachable through it | Change `IngestionWriter.append_history` to append lines to a **rolling segment file** (seal at N records / T bytes) with a per-stream head-digest sidecar; replace `_capture`'s "read the whole stream then check membership" (`forward_capture.py:232`) with a **dedupe index** (`capture_id` → segment offset) read under the existing `exclusive_lock`. Keep `_iter_history_root`'s digest verification as an offline `verify` path, not a per-append path. Add a benchmark test asserting ≥10k appends complete in bounded time. **Writer:** `platform.py`. **Reader:** unchanged. | **L** | — |
| **G1** | **Nothing reads `axiom-data/`** — the whole read side is ISOLATED | Pick ONE real consumer and wire it end to end: the cheapest is `src/data/execution_cost_model.py` (already reads tick parquets) reading spreads from `TrainingDataView` instead of `trained_data/ticks/`. That single wire turns tick capture from PARTIAL to LIVE and gives `TrainingDataView`/`export_training_root` their first caller. **Writer:** existing tick capture. **Reader:** `execution_cost_model.py`. **Schedule:** none. | **M** | G0 |
| **G2** | **P2 readiness has no producer**; `scripts/check_risk_target_p2_readiness.py` is referenced by `run_forward_capture_daily.py:20` and does not exist | Write the script: instantiate `ForwardCaptureService.default()`, call `evaluate_p2_readiness(required_pairs=<19 P2 pairs>)`, persist via `ControlStateStore.write("p2_readiness", ...)` (gives it its first caller and CAS safety) into `trained_data/forward_capture/`, exit `0` when ready / `3` when accumulating — matching the `{0,3}` accepted codes already declared. **Writer:** new script. **Reader:** `training_cockpit.py:483` (exists). **Schedule:** already in `com.buddy.forward_daily`. | **S** | G0 |
| **G3** | **Readiness gates nothing** — dead telemetry; `trainable` is a hard-coded `False` at `training_cockpit.py:840` | Make the P2 program's `trainable` a function of `p2.get("ready")`, and add a fail-closed refusal at whatever risk-target-P2 entrypoint is built (refuse when the readiness file is absent, stale > 48 h, or `ready:false`). Until that entrypoint exists, at minimum surface `blocking_reasons` and the per-pair day counts in the cockpit so the gap is visible. **Reader:** `read_control_readiness`. | **S** | G2 |
| **G4** | **V1 silent data loss** in `tick_capture.py:189` | Log the exception with `pq_path`, quarantine the unreadable file to `<path>.corrupt-<ts>` instead of overwriting it, and increment a `parquet_merge_failures` stat surfaced through `_write_health`. **No behaviour change to the happy path.** | **S** | — |
| **G5** | **Family 3 — trade-entry context has no canonical write**; `trade_journal_rl.json` is an unbounded whole-file JSON array with 6 writers and 81 reader files | Two-step, in order. (a) Call `capture_trade_entry_context(entry_record)` at `execution.py:4045` (the single per-trade entry point) — fail-open first, matching the Track B pattern, to bound blast radius. (b) Make `trade_journal_events.append_snapshot_events` the authority behind the file: every one of the 6 writers goes through it, and `rebuild_projection` regenerates the JSON array as a read-only projection so all 81 readers stay untouched. Add the tests `trade_journal_events.py` currently lacks entirely. **Writer:** `execution.py`. **Reader:** unchanged (projection). | **L** | G0 |
| **G6** | **Family 4 — fill ladders never persisted anywhere** (0/18 journal records carry `full_price`); the roadmap's "real order-fill ladders existed but were not being consumed" is still true | Capture the OANDA `ORDER_FILL` transaction where it is already fetched (transaction-ledger path in `execution.py`) and call `capture_fill_transaction(tx)`. The method already computes `slippage_vs_ladder_mid_price` and `slippage_vs_requested_price` (`forward_capture.py:452-462`) — no new modelling needed, just a call site plus persistence of `fullPrice`. **Reader:** execution-cost model / meta-labeler, once G1 establishes the read pattern. | **M** | G0, G1 |
| **G7** | **Families 6b + 7 — no canonical write, and their legacy writer can lose rows silently (V4)** | Add `capture_track_b_rebalance(row)` in `track_b_shadow.record_shadow_cycle` (`track_b_shadow.py:405`) and `capture_hedge_forward(row)` in `hedged_shadow_lane.run_cycle_for_strategy` (`hedged_shadow_lane.py:814`), both **fail-closed before** the legacy `_append_jsonl` — copying the `exposure_history.py:364-381` pattern verbatim, including its post-write read-back verification. Fixes V4 for both lanes at the same time. | **M** | G0 |
| **G8** | **`scripts/capture_track_b_new_filings.py` missing**; `com.buddy.forward_daily` hard-fails nightly | Either write the script (enumerate new EDGAR filings since the last captured accession, call `load_pit_filing`, which already hooks `capture_track_b_filing`) or remove the job from `run_forward_capture_daily.py:17` so the exit code stops lying. **Decide, don't leave a broken reference.** | **S** | — |
| **G9** | **V2/V3 double swallow** on the only Track B canonical path | Drop the redundant outer `try` at `pit_text_loader.py:388-390`, and change `capture_best_effort` to return a `bool` so callers can at minimum count failures (`forward_capture.py:692-700`). Keep it non-raising — but make failure *countable*, per the improvement-rules "write side AND read side" gate. | **S** | — |
| **G10** | **Retention asymmetry** — legacy ticks pruned, canonical history unbounded (§5.4) | Give the canonical stream a retention/compaction policy that is **never shorter** than `--retention-days`, or document explicitly that canonical is permanent and legacy is a cache. Currently neither is stated and the two will silently disagree. | **M** | G0 |
| **G11** | **Stage 3 snapshots + `DatasetManifest` never published**; training still defines its dataset by direct file reads | After G1 proves a read path, have one trainer (risk-target baseline is the natural first, it already has a descriptor at `market_data/factor/axiom_risk_target_dataset.json`) consume a signed snapshot via `TrainingDataView` and record the manifest digest in its model meta. This is the actual Stage-3 exit gate. | **L** | G1 |
| **G12** | **`confidence_calibration.json:trade_history` is an uncapped rewritten array** (690 entries) | Either cap it (`[-N:]`, matching the other ring buffers) or move it to JSONL. Low blast radius, single writer. | **S** | — |
| **G13** | **Roadmap says COMPLETE where §3.1 shows 4 of 8 methods dead** | Replace `:676` and `:706` status blocks with §1/§3's table. The current text will cause the next reader to skip exactly the work that is missing. | **S** | this audit |

**Suggested execution order:** G4, G8, G9, G12 (all S, independent, no dependencies) →
**G0** (the unlock for everything canonical) → G2 → G3 → G1 → G7 → G6 → G5 → G10 → G11 → G13.

---

## 10. Confidence summary

| Claim | Confidence | Basis |
|---|---|---|
| 4 of 8 `capture_*` methods have zero call sites, ever | **HIGH** | repo-wide grep + `git log -S` per name |
| `TrainingDataView` / `FeatureWriter` / `ControlStateStore` / snapshots / `trade_journal_events.py` have zero production callers | **HIGH** | repo-wide grep incl. `dashboard/`, `scripts/`, `cli/` |
| P2 readiness has no producer and its only consumer is a display string | **HIGH** | grep `p2_readiness`; read `training_cockpit.py:483-508, 788-855` |
| Canonical append is O(N²) per stream | **HIGH** | code read of `forward_capture.py:231-245` + `platform.py:568-591` + `_iter_history_root:122-137` |
| `capture_track_b_new_filings.py` / `check_risk_target_p2_readiness.py` do not exist | **HIGH** | `ls`, `find`, `git log -S` (both empty) |
| V1 silently discards a day of ticks | **HIGH** | `tick_capture.py:183-193` read in full |
| LaunchAgent load state on the operator's host | **MEDIUM** | two secondary sources agree; not verifiable from this environment |
| Journal context-completeness generalises beyond these 18 records | **MEDIUM** | small committed sample; host journal not visible |
| Freshness figures for gitignored stores (`trained_data/ticks/`, `crypto_cache/`) | **UNKNOWN** | not present in this checkout |
