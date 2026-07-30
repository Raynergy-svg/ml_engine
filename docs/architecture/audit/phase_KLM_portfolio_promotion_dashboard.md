# Phase K / L / M audit — portfolio allocator, promotion service, evidence cockpit

**Date:** 2026-07-30 · **Scope:** read-only evidence audit. No code changed.
**Method:** every claim below was re-derived from disk in this session — file read,
integration grep, or live execution of the reader. Claims carry an explicit
confidence tag. Where a grep returned nothing, the grep is quoted so the negative
is checkable.

**Roadmap sections audited:** `docs/architecture/AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md`
§5 Stages 10–15 (lines 455–570), §14 Phase K (1335–1379), §15 Phase L (1385–1427),
§16 Phase M (1432–1487).

**Environment caveat:** this container lacks `fastapi`, `pyarrow`, and `defusedxml`.
That blocks 5 of 10 dashboard test modules from collecting (§7). It does not affect
any file-read or grep finding.

---

## 0. Classification summary

| Deliverable | Class | One-line basis |
|---|---|---|
| **K** — cross-lane portfolio allocator | **ABSENT** | no module; `grep "class .*Allocator"` over `src/` returns one hit, `DynamicRiskAllocator` (a per-regime risk *multiplier*) |
| **K** — standardized per-lane return/exposure contract | **ABSENT** | no lane emits a net-return / vol / DD / turnover / exposure / liquidity / tail tuple; `/api/lanes` returns halt booleans (`data_sources.py:457`) |
| **K** — 9 allocation methods/limits | **2 of 9, ISOLATED** | HRP-across-sleeves + inverse-variance in `src/equity/sleeve_combiner.py`; offline-only, called by experiment scripts |
| **K** — book-level evidence package (12 fields) | **ABSENT** | `grep -i "book.level\|portfolio.gate"` over `src/ dashboard/ scripts/ tests/` hits only the roadmap `.md` |
| **K** — "no lane capital-active from a standalone gate" | **VIOLATED BY DESIGN** | 4 independent per-lane LiveGates; none consults a portfolio gate |
| **L** — promotion service (11 steps) | **ABSENT as a service; 5/11 steps exist as primitives** | `EvidenceStore` has the primitives; nothing composes them; `write_champion_pointer` has zero production callers |
| **Stages 10–15** state machine | **PARTIAL (ISOLATED)** | 10-state machine fully implemented in `src/evidence/transition_policy.py`; no CANARY/PAPER state; zero packages on disk |
| **Rollback triggers (11)** | **0 implemented in the AXIOM path; 4 analogues exist in the legacy path** | no retirement/rollback monitor exists over `src/evidence/` |
| **M** — Data cockpit section | **LIVE (code) / EMPTY (data)** | real readers, real disk paths; all 8 domains report `available: False` (executed this session) |
| **M** — Jobs cockpit section | **LIVE (code) / EMPTY (data)** | real journal + package lineage merge; counts all 0 |
| **M** — Evidence cockpit section | **LIVE (code) / EMPTY (data)** | reads the store's own signed index; `available: False`, 0 packages, 0 champions |
| **M** — Forward-monitoring section | **PARTIAL** | reader is real and honest; **no writer exists anywhere** — `grep forward_monitoring` finds only readers + tests |
| **M** — Controls (no-bypass) | **LIVE and correct** | Ed25519 operator credential + nonce ledger + relay-only promotion; no bypass found |

**Counts:** 13 deliverables assessed → **4 ABSENT**, **1 ABSENT-with-primitives**,
**3 PARTIAL/ISOLATED**, **4 LIVE-code-empty-data**, **1 LIVE**.
**Zero AXIOM evidence packages exist on disk.** (`ls trained_data/evidence` →
`No such file or directory`.)

---

## 1. Phase K — portfolio allocator

### 1.1 Is there any portfolio allocator? — **NO** (confidence: HIGH)

`grep -rn "class .*Allocator\|def allocate\|capital_active\|cash_reserve\|lane_capacity\|tail_correlation\|drawdown_budget" --include=*.py src/`
returns exactly one hit: `src/scanner/automation/dynamic_risk_allocator.py:53`
`class DynamicRiskAllocator`. Reading it (lines 1–110): it records closed-trade
P/L per volatility regime and returns a scalar risk multiplier in `[0.50, 1.30]`
from streak detection and a half-Kelly cap. It has no concept of a lane, a
correlation matrix, or a capital split. **Not a portfolio allocator.**

Adjacent, genuinely-live but out-of-scope machinery:

- `src/scanner/automation/kelly_portfolio.py` — `KellyPortfolioOptimizer`,
  instantiated live at `src/scanner/engine.py:1045-1046`. Portfolio-level Kelly with a
  15%-of-NAV VaR budget (`MAX_PORTFOLIO_RISK_PCT = 0.15`, line 38) and 5% max single
  position. This is **intra-lane** (FX pairs inside the oanda_fx lane), not cross-lane.
- `src/scanner/config.py:1190` `correlation_filter_threshold = 0.70` — a per-trade
  double-exposure filter, not an allocation policy.

### 1.2 Standardized per-lane return/exposure contract — **ABSENT** (HIGH)

The roadmap requires `lane book → net return → volatility → drawdown → turnover →
exposure → liquidity → tail indicators` (§14, lines 1339–1348).

No such contract type exists. The evidence contract layer (`src/evidence/contracts/models.py`)
defines `EvidencePackage`, `EvaluationReport`, `GateResult`, `ChampionPointer` — none
carries a return series or exposure history. `EvaluationReport.metrics` is a free-form
`dict[Identifier, MetricValue]` (line 238), which is a bag, not a contract.

What lanes actually publish is heterogeneous and per-lane:
`/api/lanes` → `read_lane_status` → `{lane: bool}` halt flags (`data_sources.py:457,468`);
crypto/track_b/hedge lanes each write their own bespoke JSONL ledger
(`trained_data/hedge/raw_vs_hedged_ledger.jsonl`, `trained_data/crypto_carry/…`).
`AssetClass.PORTFOLIO` exists in the enum (`models.py:35`) but nothing emits it —
`grep AssetClass.PORTFOLIO` over `src/` outside the enum definition: no hits.

Notably absent: `trained_data/hedge/exposure_history.jsonl` is referenced as a data
source by the cockpit (`training_cockpit.py:41`) but **does not exist on disk**
(`wc -l` → No such file).

### 1.3 The 9 allocation methods/limits

Each grepped individually over `src/` (case-insensitive, excluding tests):

| # | Method | Status | Evidence |
|---|---|---|---|
| 1 | Equal-risk contribution | **ABSENT** | `grep -i "equal_risk\|risk_contribution"` → 0 hits |
| 2 | Inverse volatility | **PARTIAL / ISOLATED** | inverse-**variance** at `src/equity/sleeve_combiner.py:52-53,64-65`; correct family, wrong exponent, offline |
| 3 | Correlation-penalized allocation | **ABSENT** | `grep -i "correlation_penal\|corr_penalty"` → 0 hits |
| 4 | HRP across sleeves | **ISOLATED** | real López-de-Prado HRP: `src/equity/hrp.py:27-84` (quasi-diag + recursive bisection), applied across sleeve return streams at `sleeve_combiner.py:54-61`. Module docstring self-declares `running:NO` (line 12) |
| 5 | Drawdown budgets | **ABSENT** | `grep -i "drawdown_budget\|dd_budget"` → 0 hits |
| 6 | Tail-correlation limits | **ABSENT** | `grep -i "tail_corr\|tail_dependence"` → 0 hits |
| 7 | Lane capacity | **ABSENT** | `grep -i capacity` hits only training-capacity knobs (`wandb_sweep_config.py:541`, `trainers/config.py:38`) |
| 8 | Min/max allocation | **ABSENT at portfolio level** | `MIN_WEIGHT/MAX_WEIGHT` hits are agent-vote weights (`recursive_intelligence/weight_learner.py:31-32`). The only capital-side cap is intra-lane Kelly's 5%/15% |
| 9 | Cash reserve | **ABSENT** | `grep -i "cash_reserve\|cash_buffer"` → 0 hits |

**2 of 9 present, both offline.** Confidence HIGH.

Integration grep for the HRP combiner
(`grep -rn "SHIP_GATE_book\|combine_sleeves\|run_sleeve_bakeoff" src/ dashboard/ scripts/ tests/`):
callers are `src/equity/multi_asset_trend.py:73` (an offline variant evaluator),
`scripts/experiment_sleeve_combinations.py:83`, and `tests/`. **No runtime caller,
no promotion-path caller.** Class: ISOLATED.

### 1.4 Book-level evidence package (12 required fields) — **ABSENT** (HIGH)

Required (§14, 1364–1377): included strategy-package digests · allocation policy ·
return series · correlation matrix · exposure history · drawdown · turnover · cost ·
stress scenarios · incremental contribution by sleeve · portfolio gate · verifier result.

`grep -rn "portfolio.gate\|book.level\|book_gate" -i` over `src/ dashboard/ scripts/
tests/ docs/ .claude/` returns **only** the roadmap `.md` itself (lines 32, 487, 491,
505, 1362, 1376, 1583, 1712, 1728) plus one unrelated hedge-leg comment
(`src/hedge/hedged_shadow_lane.py:733,741`, a synthetic candidate label).

The nearest artifact is `trained_data/backtests/SHIP_GATE_book.json`, produced by
`sleeve_combiner.run_sleeve_bakeoff`. Read this session: it contains
`generated_at: 2026-06-25T00:45:12Z`, `running: "NO (offline backtest evaluation)"`,
and a 2-sleeve comparison (`combined_book.net_sharpe 0.873` vs
`harvester_alone 0.840`, margin 0.033, `avg_sleeve_corr 0.487`). That is **4 of 12
fields** (return-derived stats, one pairwise correlation, drawdown, a gate verdict) and
it is an offline research artifact, not a signed package. It carries no digests, no
allocation policy record, no exposure history, no stress scenarios, and no verifier result.

### 1.5 THE KEY RULE — "No lane becomes capital-active solely from a standalone gate"

**Status: the portfolio gate is ABSENT, and the architecture actively contradicts the
rule.** Confidence HIGH.

There are **four independent per-lane live gates**, each of which arms its own lane
from its own standalone artifact, with no cross-lane term anywhere in the decision:

| Lane | Gate module | Standalone artifact | On disk? |
|---|---|---|---|
| equity harvester | `src/equity/live_gate.py:78` | `trained_data/backtests/SHIP_GATE.json` | **YES — `gate_pass: true`** |
| track_b | `src/equity/track_b_live_gate.py:42` | `trained_data/equity/track_b/SHIP_GATE.json` | no |
| crypto momentum | `src/crypto/crypto_live_gate.py:41` | `trained_data/crypto/SHIP_GATE.json` | no |
| crypto carry | `src/crypto/crypto_carry_live_gate.py:50` | `trained_data/crypto_carry/SHIP_GATE.json` | no |

`src/equity/live_gate.py` lines 6–33 enumerate the five arm preconditions: ship-gate
`gate_pass` + universe-hash match, a typed `LIVE` token, an NAV-fraction bound, a
constructible kill switch, and a constructible drawdown guardian. **None of the five is
a portfolio-level check.** Its own docstring (line 43) states: *"nothing else may flip
dry_run → live without going through `LiveGate.arm`"* — i.e. `LiveGate` is, by design,
the sole authority, and it is standalone.

Mitigating facts (verified this session): `trained_data/equity/live_gate_state.json`
does not exist, so the equity gate has never been armed; the other three lanes have no
`gate_pass` artifact and will refuse. The exposure is architectural, not currently
realized.

**Cross-check worth flagging:** `.claude/state.json` currently reads
`halted: false, mode: "live", scan_cycle_count: 2`, while `CLAUDE.md` Hard-NO #2
documents `halted: true`. Reported as a factual disk read; resolving it is outside this
audit's scope.

---

## 2. Phase L — the 11-step local promotion service

**There is no promotion service.** `grep -rn "PROMOTION_SERVICE" src/ dashboard/ tests/`
returns 5 hits in `src/evidence/` (the role enum, the transition table, the store's
authority check) and 14 in `tests/`. **No module named, or acting as, a promotion
service exists.** The decisive integration grep:

> `grep -rn "write_champion_pointer\|load_champion_pointer" --include=*.py src/ dashboard/ tests/ scripts/`
> → definitions at `src/evidence/store.py:584,616`; **all call sites are in `tests/`**
> (`tests/test_evidence_store.py:634,635,644,666,694,743,744`,
> `tests/test_hedge_evidence_slice.py:193`, `tests/test_risk_target_evidence_slice.py:201`).
> The remaining hits are the `may_write_champion_pointer: Literal[False]` capability
> flag being *asserted false* by each lane's importer.

Every lane slice terminates before promotion by construction. `src/evidence/risk_target/slice.py:5-10`:
*"importing every produced head into the local evidence store, ending each at
QUARANTINED or REJECTED. A candidate never overwrites an incumbent — the slice never
promotes a champion."* Lines 43–47 declare only three roles (PRODUCER, LOCAL_IMPORTER,
INDEPENDENT_VERIFIER) — *"champion/operator authority is out of scope"*.

### Step-by-step

| # | Step | Status | Evidence |
|---|---|---|---|
| 1 | Resolve disposition head | **PRIMITIVE, no service** | `store.append_disposition(..., expected_head_digest=)` `store.py:460-486`; `ConcurrentHeadError` on mismatch (481–486); head reconstructed by `reconstruct_disposition` (473–478) |
| 2 | Verify package + artifact digest | **PRIMITIVE** | `_validate_champion_event` `store.py:527-540` (lane + exact artifact digest); `_validate_champion_unlocked` `store.py:546-582`; per-file digest re-verification on read `store.py:316-350` |
| 3 | Confirm operator approval | **PRIMITIVE, strong** | `transition_policy.py:30-34` requires `OPERATOR` for QUARANTINED→SHADOW and SHADOW→OPERATOR_APPROVED, and `PROMOTION_SERVICE` for OPERATOR_APPROVED→CHAMPION; `enforce_separation_of_duties` `transition_policy.py:125-164` requires distinct actors **and** distinct keys across the 5 privileged roles, and refuses a CHAMPION ledger missing PRODUCER/INDEPENDENT_VERIFIER/OPERATOR/PROMOTION_SERVICE |
| 4 | Confirm lane + feature compatibility | **HALF** | lane_id equality checked (`store.py:535-536`, `563-564`). **Feature/data-contract compatibility is NOT checked at champion time** — no `feature_pipeline_version` comparison in `_validate_champion_unlocked`. `JobManifest.feature_pipeline_version` (`models.py:207`) is recorded but never re-asserted against the runtime constant on promotion |
| 5 | Write candidate to versioned artifact store | **PRESENT** | `store.write_package(envelope, files)` `store.py:212-262`; content-addressed `packages/<digest>/`; path-traversal guard `_safe_package_path` `store.py:159`; `_write_new_file` is create-only (169) |
| 6 | Load smoke test | **ABSENT** | `grep -rn "smoke_test\|load_smoke" -i src/evidence/ dashboard/server/` → 0 hits |
| 7 | Dry inference / weight calculation | **ABSENT** | `grep -rn "dry_inference\|dry inference" -i src/evidence/ dashboard/server/` → 0 hits |
| 8 | Atomic champion-pointer swap | **PRIMITIVE** | `write_champion_pointer` `store.py:584-614`; `_atomic_write_bytes` (122) = tmp + fsync + `os.replace`; CAS via `expected_pointer_digest` (602–606). **Zero production callers** |
| 9 | Record promotion event | **PRIMITIVE** | `append_disposition` `store.py:460`; append-only, filename `{sequence:020d}-{digest}.json` (517), `ImmutableConflictError` on rewrite (518–519) |
| 10 | Notify dashboard + Tier 7 | **ABSENT** | `grep -rn "notify" src/evidence/ dashboard/server/` → 0 hits; `grep -rn "tier7" src/evidence/` → 0 hits. The dashboard *polls* the index; there is no notification and no Tier-7 hook |
| 11 | Preserve prior champion for rollback | **WEAK / DERIVED** | the store is append-only so prior packages survive, but there is no `prior_champion` pointer. The cockpit *infers* it as "the most recent RETIRED package in this lane" (`training_cockpit.py:664,670`) — a display heuristic, not a rollback target |

**Score: 5 primitives present, 1 half, 5 absent, 0 composed into a service.**
Confidence HIGH (each row grep- or read-verified).

### Additional structural gap: ephemeral signing identities

`slice.py:80` `signers = {role: Ed25519Signer.generate() for role in _SLICE_ROLES}` —
each slice run mints **fresh** producer/importer/verifier keys held only in memory.
Consequence: a ledger written by run *N* cannot be re-verified by run *N+1* (the trust
store no longer contains the key). The dashboard control plane fixes this for its own
three roles by persisting PEMs (`axiom_evidence_control.py:364-386`
`_load_or_create_signers`, mode `0o600`, `O_EXCL`), but the standalone slices do not.
This must be resolved before any package can survive a process boundary — it is a
prerequisite for Phase L, not a Phase-L step. Confidence HIGH.

---

## 3. The 6 deployment stages, and the competing promotion systems

### 3.1 Is the state machine implemented? — **YES, in one place, isolated**

`src/evidence/contracts/models.py:48-59` defines all 11 disposition states.
`src/evidence/transition_policy.py:15-51` defines the transition table exactly as
roadmap Stage 13 (line 511–522) specifies, plus REJECTED edges. `store.py:446-453,499-506`
requires an accepted import verdict before any state at or beyond QUARANTINED.

**Gap vs the roadmap's own deployment-stage list (§15, 1403–1410):** the roadmap names
six stages `quarantine → offline replay → shadow → canary/paper → approved champion →
retired`. The implemented enum maps offline-replay to `METRIC_REPLAYED` (which precedes
`QUARANTINED` in the transition table — inverted order vs the prose) and **has no
CANARY or PAPER state at all** (`grep -rn "CANARY\|PAPER" src/evidence/` → 0 hits).
Class: PARTIAL. Confidence HIGH.

**And it holds nothing.** `ls trained_data/evidence` → `No such file or directory`.
Zero packages, zero ledgers, zero champion pointers, no index. Executing
`read_training_cockpit()` this session returned
`evidence.available: False, packages: 0, champions: {}`, with all 7 lane readers
reporting `"no evidence index on disk"`. Class of the AXIOM promotion path as a whole:
**ISOLATED** (tested in isolation, never exercised).

### 3.2 Competing promotion systems — **CONFIRMED, and there are more than two** (HIGH)

Six distinct promotion mechanisms coexist. Only the last three are running.

| # | System | Stages | Governs | Running? |
|---|---|---|---|---|
| 1 | **AXIOM evidence** `src/evidence/` | CREATED→…→CHAMPION→RETIRED (10) | signed research artifacts | **NO** — no store on disk, no service |
| 2 | **Meta-cybernetic** `StagedDeployer` + `MetaManager` | shadow→canary→live (+rollback) | `ScannerConfig` adjustments | **NO** — `MetaManager(` constructed only at `orchestrator.py:451` and in `scripts/cybernetic_*.py`; `Orchestrator` is library-only per CLAUDE.md; `.claude/meta/changes/` is **empty** |
| 3 | **Brain-loop proposals** `src/brain_loop/promotion.py` | shadow auto / live → `PENDING_OPERATOR` | hypotheses | partially — called from `src/brain_loop/cycle.py:109` |
| 4 | **Training registry** `src/training/promotion_policy.should_promote` | `:staging` → `:production` alias | trained model artifacts | **YES** — `src/training/training_events.py:152,194` on every training completion |
| 5 | **Champion-challenger** `ModelManager` | shadow scans → promote/retire | live model versions | **YES** — `src/scanner/engine.py:1254` |
| 6 | **Four LiveGates** (§1.5) | shadow → armed live | lane capital | **YES** (as the arming authority) |

The sharpest conflict: `src/scanner/engine.py:315-316` calls
`register_model_reload_handler(self)`, which subscribes the running scanner to
`MODEL_PROMOTED` and **hot-reloads models in-process**
(`src/scanner/model_reload_handler.py:31,48`). So a model can go from trained to
live-in-the-scanner via systems #4 + #5 with **zero** interaction with the AXIOM
evidence layer: no signature, no disposition ledger, no operator authority, no champion
pointer, no separation of duties.

**Finding:** the roadmap's evidence-based promotion path is a parallel, unpopulated
system built beside a live promotion path it neither gates nor observes. Nothing in
`src/evidence/` can block, and nothing in `src/training/` or `src/scanner/` consults it.
Confidence HIGH — established by the `write_champion_pointer` zero-caller grep plus the
`engine.py:315` / `training_events.py:194` positive call sites.

---

## 4. The 11 automatic rollback triggers

**In the AXIOM path: 0 of 11 implemented, 0 monitored.** There is no retirement monitor,
no champion watchdog, and no writer of forward telemetry over `src/evidence/`
(`grep -rn "forward_monitoring" src/` → 0 hits outside the dashboard reader).
`SAFETY_MONITOR` exists as an authority role permitted to drive CHAMPION→RETIRED
(`transition_policy.py:38-40`) but **nothing implements that role** —
`grep SAFETY_MONITOR src/` outside the enum and transition table: 0 hits.

Legacy-path analogues that *do* run, none of which can act on a disposition chain:

| # | Trigger | AXIOM path | Legacy analogue |
|---|---|---|---|
| 1 | Artifact mismatch | ABSENT (checked only inside the uncalled `_validate_champion_unlocked`) | `store.py:316-350` verifies digests on read — but only when read |
| 2 | Feature-contract mismatch | ABSENT | **PARTIAL/LIVE** — `FEATURE_PIPELINE_VERSION` refuses cross-version artifacts (`src/core/modular_data_loaders.py`, per `.claude/rules/improvement.md`) |
| 3 | Abnormal output | ABSENT | none found |
| 4 | Cost explosion | ABSENT | `grep -i "cost_explosion\|cost_breach"` → 0 hits repo-wide |
| 5 | Drawdown breach | ABSENT | **LIVE** — drawdown guardian every scan cycle; `src/equity/kill_switch.py` for the equity lane |
| 6 | Data staleness | ABSENT | **LIVE** — model-freshness rails (`get_model_freshness_for_pairs`), uncertainty hard-block at `max_component_age_days > 7` |
| 7 | Calibration collapse | ABSENT | `grep -i "calibration_collapse\|calibration_drift"` → 0 hits. `src/risk/confidence_calibration.py` recalibrates but does not trigger rollback |
| 8 | Missing heartbeat | ABSENT | **LIVE for the TUI** — `.claude/heartbeat.json` + `src/tui/heartbeat.py`; not bound to any champion |
| 9 | Disposition-chain inconsistency | **PRIMITIVE** | `reconstruct_disposition` raises `EventChainError`→`StoreCorruptionError` (`store.py:444-445`); detection only, never scheduled |
| 10 | Operator halt | ABSENT in AXIOM | **LIVE** — `StateEngine().get_halted_strict()` fail-closed at `src/scanner/execution.py:2099`; `.claude/tools/stop_gate.sh` + `risk_monitor.sh` at every turn end |
| 11 | Tier-7 safety event | ABSENT | **LIVE** — `src/scanner/automation/alert_manager.py:48` `AlertManager` |

**Documentation-only in the AXIOM sense: all 11.** The 5 live analogues protect the
legacy trading path and have no channel into a disposition ledger. Confidence HIGH.

The only rollback machinery that is *coded* end-to-end is the meta pipeline's —
`StagedDeployer.rollback` (`staged_deployer.py:133-144`) delegating to
`ConfigAdjuster.revert_by_id` (385–393), driven by `PostDeployCritic`. It is not
running (§3.2, row 2) and it governs config values, not artifacts.

---

## 5. Phase M — the cockpit's four sections

Backend: `dashboard/server/training_cockpit.py` (933 lines, read in full for the
sections below). Frontend: `dashboard/web/components/TrainingEvidenceCockpit.tsx`
(361 lines). API: `dashboard/server/training_api.py`, router mounted unconditionally at
`app.py:56`, GET `/api/axiom_training`.

**Hardcoded-sample-data check (the requested top-finding hunt): NEGATIVE.**
`grep -in "demo\|sample\|placeholder\|fake\|mock\|hardcode\|TODO\|stub"` over
`dashboard/server/data_sources.py` (1814 lines) returns **one** hit — line 316, a
comment explaining that the label comes from a rule file *"never the repo dirname or a
fake label"*. `grep -rn "random\.\|np.random\|faker\|lorem"` over
`dashboard/server/*.py`, `dashboard/web/components/*.tsx`, `dashboard/web/lib/*.ts`
(excluding tests) returns **0 hits**. `SentimentPlaceholder.tsx` is named "placeholder"
but reads `/api/sentiment` → `sentiment_snapshot.json` and renders `NotConnected` when
absent (`data_sources.py:579-623`). **No dashboard surface is backed by fabricated data.**
Confidence HIGH.

I executed `read_training_cockpit()` in this session; the real output is quoted per
section below.

### 5.1 Data — freshness / snapshots / quality / tick accumulation / missing partitions

**Code: LIVE. Data: EMPTY.**

All five sub-surfaces are implemented against real disk:
`read_data_status` (`training_cockpit.py:368`) walks 8 domains × 5 tiers
(`DATA_DOMAINS` line 28, `DATA_TIERS` line 29), resolves manifests via
`TIER_MANIFEST_PATTERNS` (30–34) and `_inspect_manifest` (316) — which is what produces
the **missing-partition** list — and inventories raw sources through
`DOMAIN_SOURCE_PATTERNS` (35–47). Freshness is `_age_seconds` off real `st_mtime` (96–101).
Tick accumulation is `read_live_capture_status` (193), deliberately held **outside** the
30-second cache (comment at 898–900) so a page refresh cannot report a dead writer from a
stale snapshot — a correctness detail worth preserving.

Executed output: all 8 domains `available: False`. Root cause:
`AXIOM_DATA_ROOT` defaults to `<repo>/axiom-data` (line 873) and that directory does not
exist. `dataset_configured: True` — the risk-target descriptor
`market_data/factor/axiom_risk_target_dataset.json` is present.

### 5.2 Jobs — submitted/running/failed/completed/resources/cost/manifest digest/container digest

**Code: LIVE. Data: EMPTY.**

`read_jobs` (`training_cockpit.py:701-742`) merges two real sources: the dashboard job
journal (`_journal_jobs`, 691) and job lineage reconstructed from immutable packages
(715–734), keyed by `job_manifest_digest`. All eight required fields are carried:
`manifest_digest`, `git_commit`, `container_digest`, `configuration_digest`,
`feature_pipeline_version`, `resource_class`, `resource_usage`, `cost`. Counts are
computed over the four required statuses (741).

Cost is honest rather than invented: `_completion_fields` (`training_api.py:148-174`)
derives cost from measured `wall_seconds` × a configured rate, and stamps
`cost_basis: "local_unmetered_no_incremental_provider_charge"` when the rate is 0.

Executed output: `{'submitted': 0, 'running': 0, 'failed': 0, 'completed': 0}, total 0`.

### 5.3 Evidence — packages by lane / disposition / gates / verification / negative results / derived-from / champion / prior champion

**Code: LIVE (7 of 8 fields first-class). Data: EMPTY.**

`read_evidence_status` (`training_cockpit.py:597-688`) reads the store's own committed
index (`indexes/current.json`, line 600) — i.e. the same byte-deterministic projection
the store rebuilds from events, not a side-channel. Per package it emits `lane_id`,
`state`, `sequence`, `disposition_head_digest`, `artifact_digests`, `verification`
(from signed `LocalImportVerdict`s via `_verdicts`, 572), `gate_results`,
`negative_result` (650), `is_champion` (651), plus `_package_metadata` (520).
Lane summaries add `champion` and `prior_champion` (663–673).

Two soft spots:
- **prior champion is inferred**, not recorded — `retired[0]` where `retired` is
  packages in state RETIRED (664, 670). A lane that retires a non-champion package will
  mislabel it.
- **derived-from** is only surfaced if `_package_metadata` lifts
  `EvidencePackage.derived_from_package_digests` (`models.py:268`); it is not an
  explicit top-level cockpit field or a rendered column in the TSX.

Reader failures degrade honestly: a throwing lane reader is caught per-spec and recorded
in `reader_health` rather than blanking the page (624–625).

Executed output: `available: False`, 0 packages, `champions: {}`, `state_counts: {}`,
7/7 readers reporting `"no evidence index on disk"`.

### 5.4 Forward monitoring — **PARTIAL, and this is the biggest Phase-M gap**

`read_forward_monitoring` (`training_cockpit.py:745-785`) reads `*.json` from
`AXIOM_FORWARD_MONITOR_ROOT` (default `trained_data/evidence/forward_monitoring`) and
computes `shadow_age_seconds` from a real `shadow_started_at`. For any evidence lane with
no monitor file it emits an explicit `available: False, status: "not_reporting"` row with
every metric `None` (766–783) — **it refuses to fabricate**, which is exactly right.
All eight roadmap fields are carried through to the UI: shadow duration, forward Sharpe,
drawdown, turnover, cost, exposure, `baseline_deviation`, `retirement_warnings`
(`TrainingEvidenceCockpit.tsx:246-258`).

**But no writer exists.** `grep -rn "forward_monitoring\|AXIOM_FORWARD_MONITOR_ROOT"
--include=*.py --include=*.ts --include=*.tsx src/ dashboard/ scripts/ tests/` returns
**only** the reader (`training_cockpit.py:745,872,919,930`), the TSX
(`TrainingEvidenceCockpit.tsx:355-357`), the type (`types.ts:1201`), and two tests
(`test_training_cockpit.py:22,412` — the latter literally named
`test_forward_monitoring_is_explicitly_absent_until_artifact_lands`).

Executed output: `0 / 0` lanes reporting. Because Stage 15 monitoring is the only
mechanism that could ever fire a rollback trigger (§4), this absence is what makes §4
score zero. Confidence HIGH.

---

## 6. Controls — can the dashboard bypass policy?

**No policy-bypassing endpoint was found.** Confidence: HIGH for the AXIOM training and
proposals routers (read line-by-line); MEDIUM for `control.py`/`control_safety.py`
(read at the call-site level only, not exhaustively).

### 6.1 The six no-bypass requirements (§16, 1479–1486)

| Requirement | Verdict | Evidence |
|---|---|---|
| Signature checks | **ENFORCED** | `EvidenceControlPlane.promote` **relays only** — it validates the envelope shape then hands it to `store.append_disposition`, which re-derives the signature, transition policy and authority (`axiom_evidence_control.py:536-562`, docstring 539–545). The web tier signs nothing |
| Transition policy | **ENFORCED** | edges come from `_TRANSITION_AUTHORITIES` (`transition_policy.py:15-51`). Skipping SHADOW is structurally impossible: `(QUARANTINED, OPERATOR_APPROVED)` is not a key, so `enforce_transition` raises `"transition … is forbidden"` (101–106) |
| Operator authority | **ENFORCED, with a real primitive** | Ed25519 operator request credential bound to exact `action` + `subject` digest, domain-separated (`OPERATOR_REQUEST_DOMAIN`, line 58), ≤300 s window (61), ±30 s skew (62), signature verified (325), and a **durable single-use nonce ledger under `flock`** consumed *last* so only a fully-valid credential burns a nonce (220–243, 328–329). No operator private key on the server (`_SERVER_ROLES` excludes OPERATOR, 69–73; `operator_private_key_on_server: False` reported at `training_cockpit.py:831`) |
| LiveGate | **NOT REACHABLE from the dashboard** | `axiom_proposals.py:54` `_NOT_YET_WIRED = {"arm_live_gate", "promote_model"}` → accept returns **501** with a written rationale (191–200). No route touches `LiveGate.arm` |
| Lane halts | **ENFORCED via the existing path** | `unhalt_lane` accepts only by dispatching to the pre-existing `control._run("unhalt", …)` (`axiom_proposals.py:74-80`), keeping its ARM + eligibility gate |
| Champion-pointer verification | **N/A but correctly declared** | no endpoint writes a pointer; `training_cockpit.py:862` states the invariant *"the dashboard cannot sign a disposition or write a champion pointer"* — verified true by the zero-caller grep in §2 |

### 6.2 Write-endpoint inventory

`app.py` mounts **GET-only** routes plus:

- `POST /api/axiom_operator/run` (`app.py:190`) — **mounted unconditionally**, not behind
  `AXIOM_CONTROL_ENABLED`. It runs `AxiomOperator.run_once`. Assessed **not a security
  finding**: the action space is fail-closed at `src/axiom_operator/policy.py:14-38` —
  `unhalt`, `set_gross_leverage`, `raise_leverage`, `promote_model`, `promote_strategy`,
  `promote_code`, `disable_safety_gate`, `flatten` are all `HUMAN_REQUIRED` and return
  `allowed=False` (89–99); unknown actions default to refused (101–107). The only
  auto-allowed mutations are risk-*reducing* (`halt_lane`, `stop_loop`) and
  `effects.py:34-36` additionally refuses them unless `AXIOM_CONTROL_ENABLED` is set,
  routing through `control._run` so the practice pin and audit trail still apply.
  A concurrency guard (`app.py:187,198`) prevents threadpool exhaustion.
- `POST /api/axiom_training/run` and `/promote` (`training_api.py:185,247`) — router
  mounted unconditionally but **both call `_require_enabled()` first** (53–59) → 503 when
  the flag is off, then `_control()` → 503 `"operator authority not configured"` when the
  trust anchor is missing (62–68). Currently both paths 503: `AXIOM_CONTROL_ENABLED` is
  unset and `trained_data/axiom/operator_trust.json` does not exist.
- `POST /api/control/*` and `/api/axiom_proposals/*` — mounted **only** when
  `AXIOM_CONTROL_ENABLED` is truthy (`app.py:123-133`), default OFF → 404.

Supporting hardening observed: CORS `allow_origins=[]`, `allow_methods=["GET"]`
(`app.py:63,65`); the browser reaches the API only through the authenticated Next.js
server-side proxy. Request-path traversal is blocked by `_safe_partition`
(`training_api.py:71-81`), and the signed request must bind the exact dataset digest
(107–108).

### 6.3 Minor code-quality findings (not security)

- `dashboard/server/data_sources.py:607-608` — `except Exception: pass` around the
  `ScannerConfig`/`ScannerAgentTeam` import, silently falling back to hardcoded
  `order_flow_enabled = True` / `order_flow_weight = 0.95` (600–601). Violates the
  project's "no bare except / no silent failure" rule. Low impact (display only), but it
  is the one place in the dashboard where a constant can be presented as if it were read
  from config.
- Several readers swallow `ImportError` and return `None`, which is what turns the three
  test failures in §7 into failures rather than skips.

---

## 7. `python -m pytest dashboard/server/ -q --tb=short` — real output

```
1 skipped, 5 errors in 1.02s
ERROR dashboard/server/test_axiom_operator_api.py       — ModuleNotFoundError: No module named 'fastapi'
ERROR dashboard/server/test_control_arm.py              — ModuleNotFoundError: No module named 'fastapi'
ERROR dashboard/server/test_control_loop_state.py       — ModuleNotFoundError: No module named 'fastapi'
ERROR dashboard/server/test_dashboard_robustness.py     — ModuleNotFoundError: No module named 'fastapi'
ERROR dashboard/server/test_training_cockpit.py         — ModuleNotFoundError: No module named 'pyarrow'
!!!!!!!!!!!!!!!!!!! Interrupted: 5 errors during collection !!!!!!!!!!!!!!!!!!!
```

Collection aborts, so the suite never runs. Re-running with those 5 modules ignored:

```
3 failed, 40 passed, 1 skipped in 1.76s
FAILED test_data_sources_ledger.py::test_read_crypto_momentum_populated_ledger_surfaces_last_cycle
FAILED test_data_sources_ledger.py::test_read_crypto_carry_empty_ledger_does_not_raise
FAILED test_data_sources_ledger.py::test_read_crypto_carry_populated_ledger_surfaces_last_cycle
```

All three failures share one cause, visible in the captured log:
`WARNING axiom.data:data_sources.py:932 read_crypto_momentum: forward_oos_summary
unavailable (No module named 'defusedxml')` and the same at `:1092` for
`carry_shadow`. The reader catches the `ImportError`, returns `None`, and the assertions
(`assert out["risk_premium_note"] is not None`, `assert out["forward_sharpe_annualized"]
is not None`) fail. **Environment dependency gap, not a logic defect** — confidence
MEDIUM-HIGH (I did not install `defusedxml` to confirm the tests then pass).

**Test-environment finding:** the cockpit's own test module — the one that would prove
Phase M — cannot even be collected here. `dashboard/server/requirements.txt` exists; the
audit environment does not satisfy it.

---

## 8. GAP REGISTER

Effort: **S** ≤ 1 day · **M** 2–5 days · **L** > 1 week.
Ordered by dependency: each gap's blockers appear above it.

### Tier 0 — prerequisites (nothing downstream works without these)

| # | Gap | Work | Effort |
|---|---|---|---|
| **G1** | Evidence store has no persistent identity — slices mint ephemeral keys (`slice.py:80`), so no ledger survives a process | Extract `_load_or_create_signers` (`axiom_evidence_control.py:364-386`) into `src/evidence/keystore.py`; have all 7 lane slices load persistent PEMs; add an operator trust-anchor bootstrap script (extend `scripts/axiom_operator_credential.py`) | **S** |
| **G2** | Zero evidence packages exist (`trained_data/evidence` absent) — every Phase-L/M surface is empty | Run one lane slice end-to-end (risk_target is the most complete) against G1's persistent keys; commit the resulting store layout as the reference fixture | **S** |
| **G3** | Dashboard test suite cannot be collected (`fastapi`, `pyarrow`, `defusedxml` missing) | Pin `dashboard/server/requirements.txt` into the CI/dev image; make optional-dependency readers `pytest.skip` instead of returning `None` so absence is a skip, not a failure | **S** |

### Tier 1 — close the promotion path (blocked by G1, G2)

| # | Gap | Work | Effort |
|---|---|---|---|
| **G4** | **No promotion service** — `write_champion_pointer` has zero production callers | New `src/evidence/promotion_service.py` composing the 11 steps against `EvidenceStore`. Steps 1,2,3,5,8,9 are existing primitives; author steps 6 (load smoke), 7 (dry inference), 10 (notify), 11 (explicit prior-champion record) | **M** |
| **G5** | Step 4 is half-done: feature/data-contract compatibility is never re-asserted at champion time | In `_validate_champion_unlocked`, compare the package's `JobManifest.feature_pipeline_version` against the runtime `FEATURE_PIPELINE_VERSION` and refuse on mismatch. This is the existing `.claude/rules/improvement.md` train↔inference rail, applied at the promotion boundary | **S** |
| **G6** | Prior champion is *inferred* from RETIRED rows (`training_cockpit.py:670`) — not a rollback target | Record a signed `prior_champion` alongside every pointer swap; make the cockpit read it instead of guessing | **S** |
| **G7** | Roadmap names a canary/paper stage; `DispositionState` has none | Decide explicitly: add `CANARY` + its two transition-authority edges, or amend §15 to the 10-state reality. Do not leave prose and enum disagreeing | **S** |

### Tier 2 — make promotion observable and reversible (blocked by G4)

| # | Gap | Work | Effort |
|---|---|---|---|
| **G8** | **No forward-monitoring writer** — the reader, types and UI are all done and starved | A `forward_monitor` job per SHADOW/CHAMPION lane writing the 8 fields the reader already expects (`training_cockpit.py:766-783`) into `trained_data/evidence/forward_monitoring/<lane>.json`. Cheapest high-value item in this register: it lights up an entire finished cockpit section | **M** |
| **G9** | **0 of 11 rollback triggers** — `SAFETY_MONITOR` is an unimplemented role | A retirement monitor consuming G8's telemetry, authorized as `SAFETY_MONITOR` for CHAMPION→RETIRED (the edge already exists, `transition_policy.py:38-40`). Start with the 5 that already have live legacy analogues (drawdown, staleness, operator halt, Tier-7 alert, feature-contract) by bridging their existing signals; then add cost, calibration, abnormal-output, heartbeat, chain-inconsistency, artifact-mismatch | **L** |
| **G10** | Step 10 (notify dashboard + Tier 7) absent — the cockpit polls, Tier 7 is unaware | Emit a promotion event onto the existing Tier-7 bus (the same bus `MODEL_PROMOTED` already uses, `src/scanner/model_reload_handler.py:80`) and append to the dashboard activity feed | **S** |

### Tier 3 — Phase K (blocked by G2 for packaging, G8 for return series)

| # | Gap | Work | Effort |
|---|---|---|---|
| **G11** | **No per-lane strategy-return/exposure contract** — the allocator's only possible input | Add a `LaneBook` contract to `src/evidence/contracts/models.py` (net return series, vol, drawdown, turnover, exposure, liquidity, tail indicators) and make each of the 7 lane slices emit it. Everything in Phase K depends on this — do it first | **M** |
| **G12** | **No portfolio allocator** — 7 of 9 methods absent | New `src/portfolio/allocator.py`. Reuse `src/equity/hrp.py` (real, correct HRP) for the HRP and inverse-vol legs; author ERC, correlation penalty, drawdown budgets, tail-correlation limits, lane capacity, min/max allocation, cash reserve | **L** |
| **G13** | **No book-level evidence package** (12 fields) | Emit an `AssetClass.PORTFOLIO` package from G12's output. The enum value already exists (`models.py:35`) and the store will accept it unchanged | **M** |
| **G14** | **THE KEY RULE is violated: 4 standalone LiveGates arm lanes with no portfolio term** | Add a portfolio-gate precondition to `LiveGate.arm` and its 3 siblings: refuse unless a current book-level package (G13) marks this lane capital-eligible. This is the single change that makes §14's closing sentence true. **Operator decision required** — it changes arming semantics for a lane whose `SHIP_GATE.json` already reads `gate_pass: true` | **M** |

### Tier 4 — reconcile the competing systems (blocked by G4, G9)

| # | Gap | Work | Effort |
|---|---|---|---|
| **G15** | Six promotion mechanisms; the three that run bypass the evidence layer entirely | Decide the target topology, then subordinate: either route `promotion_policy.should_promote` + `ModelManager` through the evidence store, or formally scope `src/evidence/` to research lanes and document that FX model promotion is out of scope. **Do not leave both live and mutually blind** | **L** |
| **G16** | Meta-cybernetic pipeline (`MetaManager`/`StagedDeployer`) is fully built, has rollback, and runs nowhere (`.claude/meta/changes/` empty) | Either wire it into the live driver (`EmbeddedScanner`, not `Orchestrator`) or mark it dormant in `docs/tier7-architecture.md`. A shadow→canary→live state machine that no process drives is a maintenance liability and a source of exactly this kind of audit confusion | **S** |
| **G17** | `data_sources.py:607-608` `except Exception: pass` → hardcoded `order_flow` constants presented as config | Narrow the catch, log, and report `available: False` instead of a constant | **S** |

**Suggested first slice (one week, unblocks the most):** G1 → G2 → G3 → G8 → G5.
That yields a persistent store with one real package, a green test suite, a live
forward-monitoring section, and the train↔inference rail enforced at the promotion
boundary — before any of the larger G4/G9/G12 builds begin.
