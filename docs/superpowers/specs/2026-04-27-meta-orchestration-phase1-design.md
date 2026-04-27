# Meta-Orchestration Phase 1 — Closed-Loop Cybernetics

**Date:** 2026-04-27
**Status:** Design (approved, pending spec self-review and operator sign-off)
**Author:** brainstorming pass on smoke #7 outcomes
**Implementation budget:** ~12h, ~420 LOC across 6 existing files, 0 new files
**Phased decision:** Phase 2 (G2 architectural regime scoping) deferred 14 days pending regime-tagged outcome data

---

## 1. Context

The 9-stage meta-cybernetic change pipeline (`src/scanner/automation/meta_manager.py` and adjacent modules) circulates end-to-end as of smoke #7 (2026-04-27 06:37): incidents flow through analyst → Code Surgeon → scorecard → constitution → policy auditor → deployment arbiter → approval queue → staged deployer (shadow → canary → live).

A code-grounded gap audit on 2026-04-27 surfaced 10 architectural gaps. The gaps are not implementation errors — every stage is wired and tested. The gaps are **connection topology**: feedback edges that exist as components but were never connected end-to-end.

**The principle for this phase:** *match the cybernetic frame instead of just labeling itself with it.* Phase 1 wires the dead lines. Phase 2 only adds new architecture if Phase 1's measurements demand it.

## 2. Problem

Specifically, the pipeline currently has three categories of unclosed loops:

1. **Episodic memory is write-only.** `record_setup()` and `record_outcome()` execute on every incident and trade close, but no stage in the pipeline calls `query_similar()` to consult prior outcomes when evaluating a new proposal. The system *remembers* but does not *condition on memory*.
2. **Soak windows are wall-clock proxies.** `staged_deployer.py` advances shadow → canary → live based on `cycles_elapsed`, not on closed trade counts or trade quality. A 20-cycle weekend with zero trades validates nothing.
3. **No regime tagging on outcomes.** Trade outcomes feed back into `model_bandit.json` and episodic memory without recording which regime the deployment was active in. This makes any future regime-scoping decision (G2) impossible to evaluate, because the data needed to evaluate it doesn't exist.

A fourth category — **unfiltered intake** — admits duplicate proposals (G8) and orphan-key proposals (G6) into the pipeline, wasting validation cycles on packages that should be rejected before stage 1.

## 3. Goals

- Close the episodic-memory feedback ring: queries on intake, vetoes at constitution stage
- Replace cycle-based soak with closed-trade-count soak gated on R-multiple
- Capture `regime_at_deploy` in `DeploymentRecord`, episodic memory, and post-deploy critic slices
- Prevent duplicate and orphan-key proposals from entering the pipeline
- Wire post-deploy critic findings back into scorecard tolerance calibration (`model_bandit.json`)
- Ship inside ~12h with no new architectural surfaces

## 4. Non-Goals

The following are deliberately deferred and **must not be implemented in Phase 1**:

- **G2: Architectural regime scoping** of `config_delta` at runtime. ScannerConfig keeps its current shape (legacy regime-suffixed keys remain). Decision deferred 14 days pending data from regime-tagged outcomes.
- **G5: Approval queue ergonomics** (TUI, batch ops, summary tables). Operator continues to review filesystem JSON. Revisit if queue depth exceeds 10 packages.
- **G10: Canary rollback hygiene** (ConfigAdjuster history cleanup on canary failure). Cosmetic; revisit on first real rollback.
- **Bayesian soak promotion.** R-floor (count + threshold) ships first; upgrade later if R-floor proves too crude.
- **Schema migration.** No changes to `ScannerConfig` field structure, profile dicts, or existing config keys.

## 5. Architecture

No new modules. All changes are additive to existing files. The architectural shape stays exactly as it is — the diff is purely in the *edges* of the graph.

### 5.1 Closed-loop data flow

```
Incident emitted (cycle_autonomy.py / performance_prd_generator.py)
        │
        ▼
MetaManager.intake()
        │
        ├─► dedup_hash check ──(duplicate)──► drop, log to ledger
        │
        ├─► episodic_memory.query_similar(setup_signature)
        │       └─► attaches historical_outcomes to incident
        │
        └─► route_incident()
                │
                ▼
        Code Surgeon prompt (now includes ScannerConfig field whitelist)
                │
                ▼
        Constitution.evaluate()
                │
                ├─► existing clauses: bounds, monotonic_non_increasing, diff
                └─► NEW clause: historical_loss_rate(setup, n_min=5) ≤ 0.7
                │
                ▼
        Scorecard ──► Approval Queue ──(operator approves at LIVE only)──► StagedDeployer
                                                                                  │
                                                                                  ▼
                                                          DeploymentRecord {
                                                              ...,
                                                              regime_at_deploy,
                                                              closed_trade_count_at_deploy
                                                          }
                                                                                  │
                          shadow phase: wait for 15 closed trades                  │
                                  │                                                │
                                  ▼                                                │
                          R_observed ≥ R_baseline − 0.5R ?                         │
                                  │                                                │
                          ┌───────┴───────┐                                        │
                         yes              no                                        │
                          │                │                                        │
                          ▼                ▼                                        │
                  promote to canary    revert (existing path)                       │
                                  │                                                │
                          canary phase: wait for 30 closed trades                   │
                                  │                                                │
                                  ▼                                                │
                          R_observed ≥ R_baseline − 0.5R ? + operator approval     │
                                  │                                                │
                                  ▼                                                │
                          live ──► Post-Deploy Critic                              │
                                       │                                            │
                                       ├─► tolerance_overshoot ──► model_bandit.json (calibration feedback)
                                       │
                                       └─► episodic_memory.record_outcome(
                                               change_id, regime_at_deploy, R, win
                                           )
                                                       │
                                                       ▼
                                       (next incident's query_similar reads this)
```

The single new edge that matters most is the bottom one: post-deploy outcomes flow into episodic memory keyed by `regime_at_deploy`, and the next incident's constitution check consults that memory. That's the difference between *recording* and *learning*.

## 6. Component Changes

All changes additive. Detailed per file:

### 6.1 `src/scanner/automation/meta_types.py`

- **`ChangePackage._dedup_hash() -> str`**: SHA1 of `(incident_kind, sorted_config_delta_items)`. Excludes timestamps and incident metadata so genuinely identical proposals collide; includes `incident_kind` so different kinds with the same delta don't collapse.
- **`DeploymentRecord.regime_at_deploy: Optional[str]`**: snapshot at promotion-to-shadow time. Optional because pre-Phase-1 records won't have it.
- **`DeploymentRecord.closed_trade_count_at_deploy: int`**: snapshot of `len(trade_journal_rl.json)` at promotion-to-shadow. Used by soak window arithmetic.

### 6.2 `src/scanner/automation/meta_manager.py`

- **`MetaManager.intake(incident)`**:
  1. compute `dedup_hash(incident)`; check against in-flight (`changes/`) and recent ledger; drop on collision with structured log entry
  2. call `episodic_memory.query_similar(incident.setup_signature, lookback_n=50)`; attach results to incident as `incident.historical_outcomes`
  3. proceed to `route_incident()` as today
- **`_build_surgeon_prompt()`**: insert a fixed list of valid `ScannerConfig` field names (introspected at module import via `dataclasses.fields(ScannerConfig)`). Surgeon's response parser rejects any `config_delta` key not in the whitelist; rejection routes back to surgeon as a retry hint, not a hard failure (surgeon should self-correct from the whitelist).

### 6.3 `src/scanner/automation/constitution.py`

- New clause: **`historical_loss_rate_clause`**.
  - Reads `incident.historical_outcomes` (attached by intake).
  - If `len(historical_outcomes) >= 5` and `loss_rate(historical_outcomes) > 0.7`, vetoes the proposal with verdict `"historical_pattern_loss_rate_exceeds_threshold"` and attaches the matching outcome IDs.
  - If `len < 5`: clause abstains (returns `passes=True` with reason `"insufficient_history"`). Cold-start safe.
  - Threshold (`0.7`) and `n_min` (`5`) are constants in `constitution.py`; surface as config fields if Phase 2 needs to tune.

### 6.4 `src/scanner/automation/staged_deployer.py`

- **Soak gate replaced.** `_should_promote_shadow_to_canary()` now reads:
  - `closed_trades_since_deploy = current_trade_count - record.closed_trade_count_at_deploy`
  - returns False unless `closed_trades_since_deploy >= 15`
  - **soak_window** = the 15 (or 30 for canary) closed trades from the journal whose index falls in `[record.closed_trade_count_at_deploy, record.closed_trade_count_at_deploy + N]`
  - returns False unless `R_mean(soak_window) >= R_baseline - 0.5`
  - `R_baseline` = mean R-multiple of the 50 trades **immediately preceding** `record.closed_trade_count_at_deploy` (fallback `R_baseline = 0.0` if fewer than 50 prior trades exist)
- **Canary → live** uses the same logic with `closed_trades_since_canary >= 30` and a soak_window of the next 30 trades after canary promotion.
- **Regime capture:** at promotion-to-shadow, `record.regime_at_deploy = regime_detector.current()`. Captured once at shadow start; not updated mid-deploy. Phase 2 (if it happens) revisits this.

### 6.5 `src/scanner/automation/post_deploy_critic.py`

- **`_default_metrics_slicer()` widening:** sample includes trades from `[deployed_at, now]` (existing) **plus** trades from `[deployed_at, deployed_at + soak_window]` even if soak is still running — fixes G9.
- **Tolerance feedback:** when actual diverges from predicted beyond `WIN_RATE_TOLERANCE` or `PNL_DELTA_TOLERANCE_R`, write a calibration entry to `model_bandit.json` under a top-level `calibration_feedback` key. Schema:
  ```json
  {
    "calibration_feedback": {
      "<change_id>": {
        "ts": "2026-04-27T12:00:00Z",
        "predicted_win_rate": 0.55,
        "actual_win_rate": 0.41,
        "win_rate_overshoot": -0.14,
        "predicted_r_mean": 0.8,
        "actual_r_mean": 0.3,
        "r_overshoot": -0.5,
        "regime_at_deploy": "LOW",
        "n_trades": 30
      }
    }
  }
  ```
  Bandit consumers read these entries to tighten future scorecard thresholds (e.g., shrink prediction interval if multiple change_ids overshot in the same direction). Entries are append-only; no delete on rollback.
- **Regime slicing:** outcomes filtered by `record.regime_at_deploy` when present. Slices reported separately in `PostDeployVerdict`.

### 6.6 `src/scanner/automation/episodic_memory.py`

- **`record_outcome` signature gains `regime: Optional[str]`** parameter.
- **`query_similar(setup_signature, lookback_n=50)`** returns list of dicts including `regime_at_deploy`. Called by `intake()` (G3) and `historical_loss_rate_clause` (G7).
- No new persistence schema beyond an optional regime field on existing outcome rows.

## 7. New Behavior (External-Facing)

From outside the pipeline, the changes are observable as:

- **Incidents that duplicate a recently-proposed delta no longer reach Code Surgeon.** Logged as `meta.dedup.dropped` events.
- **Code Surgeon never emits orphan keys.** When it tries, the response parser rejects and the surgeon retries with the whitelist as a prompt suffix. Maximum two retries before incident is dead-lettered.
- **Constitution rejects proposals matching historical loss patterns.** Visible in `meta/changes/<id>.json` as a new verdict reason.
- **Soak windows take longer in low-volume periods.** A weekend with 3 trades will not advance soak. A high-volume Monday session will advance multiple soaks. **Operators must understand this** — it's a deliberate behavior change, not a bug.
- **`DeploymentRecord` JSON gains two fields.** Backward-compatible reads via `.get()`.

## 8. Error Handling & Edge Cases

- **Empty episodic memory (cold start):** `query_similar` returns empty; constitution clause abstains; system falls back to existing constitution behavior. Acceptable degradation.
- **Trade journal corruption:** existing `try/except` wrapping in `staged_deployer.py` already handles this; soak math falls back to "0 closed trades since deploy" which prevents promotion. Fail-safe.
- **Regime detector returns None:** `record.regime_at_deploy = None`; post-deploy critic doesn't slice by regime for that record. Logged as warning.
- **Dedup hash collision on legitimately distinct incidents:** mitigated by including `incident_kind` in the hash. If still problematic, second-line mitigation is to widen the hash to include first 3 fields of `incident_metadata`.
- **R_baseline cold-start:** when fewer than 50 prior trades exist, `R_baseline = 0.0` (i.e., R-floor becomes "R ≥ -0.5"). Soft floor that admits anything not actively losing money. Tightens automatically as trade history accumulates.
- **Surgeon retry exhaustion:** after two retries with whitelist hint, incident is dead-lettered with reason `"surgeon_orphan_key_unrecoverable"`. Logged for operator review; does not crash pipeline.

## 9. Testing Strategy

### 9.1 Unit tests (added to existing test files)

- `test_meta_manager.py`:
  - `test_intake_dedup_drops_duplicate_proposal`
  - `test_intake_attaches_historical_outcomes`
  - `test_surgeon_prompt_contains_field_whitelist`
  - `test_surgeon_orphan_key_rejected_and_retried`
- `test_constitution.py`:
  - `test_historical_loss_rate_clause_vetoes_above_threshold`
  - `test_historical_loss_rate_clause_abstains_on_insufficient_history`
- `test_meta_pipeline_components.py`:
  - `test_staged_deployer_soak_uses_real_trade_count`
  - `test_staged_deployer_r_floor_blocks_promotion_below_baseline`

### 9.2 Smoke (extends `cybernetic_smoke.py`)

- New scenario: `dup_drop` — fire same incident twice, assert second is dropped
- New scenario: `historical_veto` — pre-populate episodic memory with 8 losing outcomes for a setup, fire matching incident, assert constitution vetoes
- New scenario: `r_floor_promotion` — simulate 15 trades at R=0.6 vs R=-0.6, assert one promotes and one reverts

### 9.3 Regression budget

- Existing 45 passing tests must still pass
- The one currently-failing test (`ConfigAdjuster.apply_adjustments` bound validation, unrelated to meta pipeline) is out of scope for this phase but tracked separately

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| R-floor too strict on small samples | M | M | Log promotion failures with R distribution; review at 14-day mark |
| Episodic memory cold-start neutralizes G7 clause | H | L | Acceptable; clause abstains gracefully; ramps up with trade history |
| Dedup hash false positive | L | M | Hash includes `incident_kind`; widen to incident metadata if surfaces |
| Soak window stalls during low-volume period | H | L | Documented as expected behavior; operator runbook updated |
| Tolerance feedback to model_bandit creates oscillation | L | M | Bandit consumer must clamp updates; revisit if observed |
| Regime detector unstable around regime boundaries | M | M | Capture once at shadow start, don't re-snapshot; accept staleness |

## 11. Out of Scope — Deferred With Intent

### 11.1 G2: Architectural regime scoping

**Defer rationale:** The most expensive change in the gap audit (touches `ScannerConfig.apply_profile()`, runtime hot path, every regime-suffixed profile key, and the regime detector). Phase 1 ships *measurement* (regime tagging on outcomes) so the Phase 2 decision can be data-driven instead of speculative.

### 11.2 G5: Approval queue ergonomics, G10: canary cleanup

Cosmetic / minor. Revisit when the underlying pain surfaces (queue depth or actual rollback, respectively).

## 12. 2-Week Review Mechanism

A scheduled background agent runs at T+14 days from Phase 1 deployment. It reads:

- `episodic_memory.json` outcomes filtered by `regime_at_deploy IS NOT NULL`
- Per-regime R-multiple distributions for each `change_id`
- Promotion success/failure rates by regime
- Number of constitution vetoes triggered by `historical_loss_rate_clause`
- Bandit calibration feedback entries from post-deploy critic

The agent answers a single question: **"Does the data justify G2?"** Concrete decision criteria:

- **YES → proceed with G2 PRD** if any of:
  - Distinct R-multiple distributions per regime exceed 0.5R difference for ≥ 3 deployed change_ids
  - ≥ 2 deployed changes show inverted sign (positive R in one regime, negative in another)
  - Operator observation of regime-mismatched degradation
- **NO → close the question for 30 days** if:
  - Per-regime distributions are within 0.3R of each other
  - No deployed change shows inverted sign
  - Constitution `historical_loss_rate_clause` is doing the work G2 would have done

The 14-day timer matches the project's existing `.claude/ralph/archive/` phase cadence. The decision agent prompt and trigger are defined as part of the implementation plan.

## 13. Success Criteria

Phase 1 is complete when:

1. All 8 gap-targeted unit tests pass
2. Three new smoke scenarios pass end-to-end
3. Existing 45 tests still pass
4. A staged shadow deployment in a live or simulated environment demonstrates:
   - Real trade count gating (not cycle-based)
   - At least one constitution veto driven by `historical_loss_rate_clause` (synthesized from injected memory)
   - At least one dedup drop (synthesized via repeated incident)
   - `regime_at_deploy` populated on `DeploymentRecord`
5. The 2-week review agent is scheduled and the trigger conditions are documented

## 14. Open Questions

- **Soak window numbers (15 / 30 trades):** picked from intuition; revisit at 14-day mark with R-distribution data.
- **Historical loss-rate clause threshold (0.7) and `n_min` (5):** same — start strict (high threshold, low n_min); loosen if too aggressive.
- **R_baseline lookback (50 trades):** same — fewer if the trading frequency is too low to gather 50 in a reasonable window.

These knobs are intentionally constants in code, not config fields, for Phase 1. Promote to config in Phase 2 if tuning becomes frequent.
