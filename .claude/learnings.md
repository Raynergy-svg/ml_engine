# Buddy Trading Learnings

Date-stamped insights extracted from trade outcomes, scan analysis, and system behavior. Patterns that repeat 3+ times get promoted to `.claude/rules/`. Archive: `.claude/learnings-archive.md`.

---

### Self-Heal #4 (Cycle 1): 14-Loss Streak Deep Diagnostic — Post-Retrain — 2026-04-16

- [2026-04-16] **PATTERN/rl_position_sizer_not_retrained_with_ensemble**: Ensemble models retrained today (age=0d, 2026-04-16T12:20). BUT rl_position_sizer.meta.json still shows trained_at=2026-03-23 (age=23.7d, 858351s). rl_train_data.npz age=28.2d. The autonomous trainer retrained transformer/TCN/Ridge/LightGBM but SKIPPED the RL position sizer entirely. The sizer was trained on 9995 samples with 47.4% win rate; live is 0/14 (0%). **Immediate: retrain RL position sizer on fresh data. Add rl_position_sizer to autonomous_trainer model group list.**

- [2026-04-16] **PATTERN/config_adjuster_pending_keys_now_valid_5_of_5**: Previous cycle found 5/9 orphan keys. Corrected proposals now use exact ScannerConfig field names: atr_sl_multiplier (field line 485), max_model_disagreement (line 619), max_uncertainty_score (line 618), min_confidence (line 405), weighted_vote_threshold (line 591). All 5 will setattr() to live fields. _load_state() fix confirmed at config_adjuster.py:186. Next process restart should consume all 5 pending -> total_adjustments should go from 0 to 5. **Verify after next restart.**

- [2026-04-16] **PATTERN/risk_per_trade_must_reduce_during_cold_start**: 14 consecutive losses ($-3,527) prove the system has no directional edge during model transitions. Fresh models (val_accuracy=0.556, barely above coin flip) need 5-10 live trades to validate before full risk exposure. During cold-start (first 10 trades on fresh models), risk_per_trade should halve from 5% to 2.5% max.

- [2026-04-16] **PATTERN/autonomous_trainer_model_group_incomplete**: autonomous_trainer retrained modular_ensemble (age=0d) and joint_gates (age=0d) but not rl_position_sizer (age=23.7d) or per_pair_models (age=unknown). Split-freshness state: gates pass on fresh predictions but position sizing uses stale reward distributions. **Fix: add rl_position_sizer and per_pair_models to the trainer retrain manifest.**

### Self-Heal #3 (Cycle 3): Config Orphan Keys + RL Staleness — 2026-04-16

- [2026-04-16] **PATTERN/config_adjuster_orphan_keys_5_of_9**: _load_state() fix confirmed (line 186 loads pending). BUT 5/9 pending keys are ORPHANS that don't match ScannerConfig field names: min_confidence_threshold (should be min_confidence), min_adx_for_trend_trade (no field), optimized_confidence_threshold (no field), optimized_momentum_threshold (no field), atr_sl_multiplier_low_regime (should be atr_sl_multiplier). setattr() silently creates orphan attributes nobody reads. Only 4 of 9 proposals (max_model_disagreement, max_uncertainty_score, weighted_vote_threshold, optimized_rr_ratio_threshold) would actually reach live config. This is the 2nd layer of the open-circuit bug. Fix: pending proposals MUST use exact ScannerConfig field names. Re-proposing with corrected keys below.

- [2026-04-16] **PATTERN/rl_position_sizer_stale_23d_reconfirmed**: rl_position_sizer.meta.json trained_at=2026-03-23 (age=23.7d, 854146s). Trained on 9995 samples with 47.4% win rate. Current live: 0/14 (0%). rl_train_data.npz also stale (23.7d). The ensemble was retrained today (joint models age=0d) but RL sizer was NOT included. Immediate: retrain RL position sizer.

- [2026-04-16] **PATTERN/mfe_zero_14_of_14_directional_failure**: All 14 closed trades had MFE=0.0 pips. 12/14 entered before today's retrain. Fresh transformer val_accuracy=0.556, barely above coin flip. Need 5+ trades on fresh models before diagnosing further.

- [2026-04-16] **PATTERN/model_disagreement_0.5_universal**: 10/14 closed trades had model_disagreement=0.5 (maximum divergence). 4/14 had 0.0 (full agreement, wrong direction). The 0.5 signal is structural, not episodic.

### Self-Heal #3 (Prior Entries): 14-Loss Streak (0W/14L, $-3,527), Critical Model Staleness — 2026-04-16

- [2026-04-16] **PATTERN/config_adjuster_load_state_ignores_pending**: ConfigAdjuster._load_state() (config_adjuster.py:174-193) loads history and last_applied but NEVER loads pending from disk. Every process restart starts with self._pending = {}, discarding all proposals written by self-heal, threshold_optimizer, and drift_monitor. This is the root cause of the open-circuit feedback loop: 7 CRITICAL pending adjustments (max_disagreement=0.25, max_uncertainty=0.40, WVS=0.85, min_confidence=0.68) proposed across 2 self-heal cycles were written to config_adjustments.json but never read back. total_adjustments=0 and history=[] after 14 consecutive losses proves the consumer pipeline has NEVER applied a single adjustment. **Fix: add self._pending = data.get("pending", {}) to _load_state().** This is a 1-line code fix with $3,527 impact.

- [2026-04-16] **PATTERN/14_loss_streak_all_mfe_zero**: 14/14 closed trades (1124-1255) have MFE=0.0 pips — price NEVER moved in the predicted direction on ANY trade. All 14 hit SL. Total PL=-$3,527.25. modular_ensemble age=28.5d, RL position sizer age=23.5d. This is not bad luck or tight stops — the directional model is systematically wrong. The ensemble trained 2026-03-18 on March directional regime cannot predict April reversed/rotated regime.

- [2026-04-16] **PATTERN/model_disagreement_0.5_on_new_trades**: Most recent trades (1249 USD_CAD, 1255 USD_CHF, 1261 USD_CAD) all show model_disagreement=0.50 and uncertainty soft penalty applied (-0.05 confidence delta). 0.50 disagreement = models split 50/50 on direction = no edge. Yet default max_model_disagreement=0.65 allows this through. The pending tightening to 0.25 would have blocked these — but the open-circuit config adjuster never applied it.

- [2026-04-16] **PATTERN/rl_position_sizer_stale_23d**: RL position sizer trained 2026-03-23 (age=23.5d, 849,156s). Trained on 9,995 samples with 47.4% win rate. Current live win rate: 0% (0/14). The sizer state distribution and reward function no longer match market conditions. **Immediate action: retrain all models.**

- [2026-04-16] **PATTERN/rejection_correlation_filter_cycle_7**: No trained_data/correlation_state.json exists — correlation filter is in-memory only. Cannot diagnose which groups block execution. Filter needs to persist active-group state for post-hoc diagnosis.
- [2026-04-16] **PATTERN/rejection_gate_thresholds_cycle_30**: config_adjustments.json has 7 pending threshold tightenings (min_confidence->0.68, max_uncertainty->0.40, WVS->0.85, max_disagreement->0.25) but total_adjustments=0 and history=[]. The apply pipeline is broken -- pending proposals are never consumed into live config. Root cause of rejection streak is the unapplied pipeline, not the threshold values themselves.
- [2026-04-16] **PATTERN/losing_exit_pattern_cycle_2**: 3 consecutive losses (1220, 1232, 1236) all in LOW regime — 2/3 sl_hit, 1/3 manual_close (pre-emptive cut). No time_stop exits. Dominant pattern: wrong-direction entry in ranging/low-volatility conditions, not tight SL placement. Price moved against entry immediately on all three. Implication: LOW-regime directional signals lack edge; require higher confidence floor when regime=LOW and ADX weak.
### Self-Heal #1: 10-Loss Streak Root Cause — 2026-04-15

- [2026-04-16] **PATTERN/rejection_gate_thresholds_cycle_11**: config_adjustments.json has 7 pending tighter thresholds (min_confidence=0.68, max_uncertainty=0.40, WVS=0.85) but total_adjustments=0 and last_applied={}. None consumed by runtime. Rejections use trading.md defaults. 3-cycle streak is NOT from over-tightened gates — setups genuinely fail defaults with stale 28d models. Pending self-heal proposals are write-only dead code with no consumer.
- [2026-04-16] **PATTERN/rejection_correlation_filter_cycle_8**: Neither trained_data/correlation_state.json nor correlation.json exists on disk. Correlation filter rejects in-memory only — no persisted state to audit which groups block which pairs. Verify if filter persists state or operates ephemerally.
- [2026-04-16] **PATTERN/quiet_cycle_N12**: scanned 15 pairs, 2 tradeable, 0 executed — post-rule-tightening gates correctly filtering with stale models; 4th consecutive zero-execution cycle confirms hard blocks are working as intended- [2026-04-15] **PATTERN/stale_model_total_directional_failure**: 10/10 closed trades lost via sl_hit with MFE=0 (price NEVER moved in predicted direction). modular_ensemble age=28d, joint_gates age=22d. Models trained on March market regime are predicting directions in an April regime that has fundamentally shifted. The RL agent weight layer adapted (updated today) but cannot fix wrong directional predictions from stale upstream models — RL tunes *which agents to trust*, not *what the models predict*. Total PL=-$2,637. **Root cause: model staleness, not agent miscalibration.**


### Self-Heal #2: 3 More Losses (Cycle 2, Trades 1220/1199/1195) — 2026-04-15

- [2026-04-15] **PATTERN/pending_config_adjustments_never_applied**: Cycle 1 proposed 4 CRITICAL config adjustments (max_model_disagreement=0.25, max_uncertainty_score=0.40, weighted_vote_threshold=0.85, min_confidence raise). ALL remain in "pending" state with total_adjustments=0. The self-heal loop diagnosed the problem correctly but the config consumption pipeline is broken — proposals sit in config_adjustments.json but nothing reads and applies them. **Implication: the entire self-heal feedback loop is open-circuit. Fix the consumer side of config_adjustments.json or the reflection system is write-only dead code.**


- [2026-04-15] **PATTERN/model_staleness_persists_through_partial_retrain**: transformer_direction_best.keras was updated at 19:03 UTC today, but modular_ensemble.meta.json still shows trained_at=2026-03-18. The autonomous trainer partially ran (transformer only?) but did not retrain the full ensemble. Per-pair models (EUR_USD, GBP_USD, etc.) all show mtimes from March 19-24. The system is trading on 22-28 day old ensemble predictions. **Implication: autonomous_trainer must verify ALL model groups are refreshed, not just the transformer checkpoint.**


- [2026-04-15] **PATTERN/autonomous_trainer_status_unknown_after_completion**: Retrain spawned at 19:01 UTC, ran 22.3min, completed at 19:24 UTC, but last_status=unknown. The trainer process likely completed but the status-polling mechanism did not capture success/failure. Without verified retraining success, the system may continue trading on stale models indefinitely. **Implication: autonomous_trainer must verify model meta.json timestamps changed post-retrain, not just that the subprocess exited.**

## Promotion Log
  - JSON Safety Gates (31 observations across Phases 4-34)
  - Retry & Robustness Gates (27 observations across Phases 4-34)
  - State Persistence Gates (8 observations across Phases 17-34)
  - Test Coverage Gates (8 observations across Phases 28-34)
  - Config Validation Gates (6 observations across Phases 12-34)
  - Silent Exception Prevention (4 observations across Phases 29-34)

### Phase 90 Execution Pipeline Root Cause Analysis — 2026-03-31
- [2026-03-31] **PATTERN/default_chain_override**: `ScannerConfig.max_data_age_seconds = 60.0` was the THIRD layer of a default chain — ExecutionConfig had `600.0` as its class default, `_init_executor()` passed `getattr(scanner_config, "max_data_age_seconds", 300.0)` as fallback, but since `ScannerConfig` DID have the attribute (set to 60.0), the ExecutionConfig got 60.0. The fix was NOT at ExecutionConfig (the destination) but at ScannerConfig (the source). **Rule**: When a getattr fallback isn't triggering, check if the SOURCE config already has the attribute with the wrong value — fix the source, not the fallback.

- [2026-03-31] **PATTERN/cache_overwrite_with_stale_history**: `_trade_sl_tp_cache` was built with `dict[key] = value` in a loop over `get_trades(state="ALL", count=500)`. OANDA returns newest-first, so the final iteration for each instrument was the OLDEST historical trade. The current open USD/JPY had SL=158.644 but the oldest historical trade had SL=155.326, which made portfolio risk compute 230% (354 pips) instead of 14.6% (22.5 pips), blocking all subsequent trades. Fix: `setdefault()` so first (newest) write wins. **Rule**: Never use `dict[key] = value` in a loop over ordered-newest-first API results — the last write wins, which is the oldest record. Use `setdefault()` when you want the most recent value.

### Phase 79 Confidence Penalty Scale Fix — 2026-03-30
- [2026-03-30] **PATTERN/scale_mismatch_across_system_layers_needs_end_to_end_audit**: The confidence penalty system had THREE scale mismatches: (1) overconfidence penalty -3.0 designed for 0-100 applied to 0-1, (2) ceiling floor 40.0 compared to 0-1 values making headroom always 0, (3) drift proxy dividing already-0-1 confidence by 100. Each was masked by a different safety net. The fix required tracing the full penalty path from penalty source → ceiling check → application → output. **Rule**: When fixing a scale mismatch, don't just fix the one you found — trace the full data flow and audit every constant on the path. Scale bugs travel in packs because they were introduced when the system's scale was different.

- [2026-03-30] **PATTERN/safety_nets_masking_scale_bugs_create_fragile_systems**: The penalty ceiling with floor=40.0 produced headroom=0 for all 0-1 confidence values, which accidentally blocked all penalties. The CalibrationGuard blocked penalties until 20+ predictions, adding another layer of masking. Both safety nets produced "correct" outcomes (no penalty applied) for the "wrong" reason (scale mismatch, not insufficient data). **Rule**: When a safety net always fires, investigate whether it's protecting against a real condition or masking a bug. If `ceiling_active=True` for 100% of checks, the ceiling is either correctly protecting every pair (expected) or misconfigured (bug). Add a metric for ceiling activation rate to distinguish.

### Phase 76 Disagreement Gate Quantization Fix — 2026-03-30
- [2026-03-30] **PATTERN/heuristic_count_must_propagate_to_adaptive_mechanism**: Phase 76 expanded heuristics from 3 to 5 (added SMA_50, MACD_histogram) in `_team.py`, but `AdaptiveDisagreementFloor` still used `num_heuristics=3`. The adaptive mechanism computed boundaries at [0.0, 0.333, 0.667, 1.0] while actual values were [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]. **Rule**: When changing the number of inputs to a quantized metric, grep all consumers that depend on the quantization boundaries. The source (heuristic computation) and all sinks (adaptive floor, gap analyzer, recommendations) must agree on N.

- [2026-03-30] **PATTERN/5_heuristic_expansion_resolves_coarse_quantization**: With 3 heuristics, floor=0.30 blocked on single-indicator disagreement (0.33 > 0.30). With 5 heuristics, 1/5=0.20 passes naturally — no adaptive mechanism needed to unblock. The expansion from 3→5 heuristics simultaneously solved two problems: (1) finer quantization steps (0.20 vs 0.33), and (2) natural pass-through for single-indicator disagreement. **Rule**: Before tuning a threshold, check if adding more inputs to the quantized metric would naturally resolve the quantization dead zone. Sometimes N+2 inputs is a cleaner fix than threshold gymnastics.

- [2026-03-30] **PATTERN/feature_column_name_mismatch_creates_dead_heuristic**: Phase 76 read `macd_histogram` from df_feat but the column is named `macd_hist` in feature_engineering.py. `_last_value()` silently returned the default `0.0`, and the `abs(0.0) >= 1e-6` guard prevented the heuristic from being appended. The MACD heuristic was dead code for all scans — only 4 of 5 heuristics ever contributed. **Rule**: When adding a new feature column reference, always grep feature_engineering.py for the exact column name. The `_last_value()` default-fallback pattern masks column name mismatches silently. Consider adding a debug log when `_last_value` returns the default.

- [2026-03-30] **PATTERN/default_fallback_masks_column_name_bugs_across_modules**: Column audit found 2 mismatches: `volume_ratio_20` vs actual `volume_ratio` (execution quality agent), `macd_histogram` vs actual `macd_hist` (uncertainty agent). Both used `_last_value(df, "wrong_name", default)` which silently returned defaults. Agents functioned but with constant/dead inputs — no crash, no error, no log. **Rule**: After any feature column reference change, audit all `_last_value` calls against feature_engineering.py. Default-fallback patterns hide mismatches. A debug log on default-hit would reveal dead features immediately.

- [2026-03-30] **PATTERN/quantization_aware_adaptive_mechanism_must_match_actual_domain**: The adaptive mechanism was made quantization-aware (Phase 72 pattern) but initialized with wrong boundaries. It would have jumped to 0.34 (above the 0.333 boundary that no longer exists with 5 heuristics) instead of 0.41 (above the actual 0.40 boundary). The mechanism would have appeared to work (it still raised the threshold) but would have landed between boundaries with no behavioral effect — a silent regression. **Rule**: Quantization-aware mechanisms are only correct when their boundary computation matches the actual domain. Always parameterize boundary computation from the authoritative source of N.

### Phase 72 Sub-Inference Gate Unblock — 2026-03-30
- [2026-03-30] **PATTERN/latent_scale_mismatch_masked_by_safety_nets**: The confidence penalty system subtracts 3.0 (designed for 0-100 scale) from confidence values on 0-1 scale. Two safety nets accidentally prevent damage: (1) CalibrationGuard blocks penalty when < 20 predictions, (2) ConfidencePenaltyCeiling floor=40.0 computes headroom=max(0, 0.43-40.0)=0, so applied_penalty=0. The penalty is fully blocked, but for the WRONG reason. **Rule**: When a safety net produces correct outcomes for incorrect reasons, document it explicitly. The system is fragile — removing either safety net or fixing the scale would expose the underlying bug. Tag for fix when confidence scale is unified.

- [2026-03-30] **PATTERN/config_profile_audit_finds_class_of_bugs_not_individual_bugs**: The sub_inference_vote_threshold dead zone was not just in the balanced profile — it was also in conservative (0.72 requires 3/3) and futures_live (0.65 misaligned). Finding one quantization bug should trigger a systematic audit of ALL profiles for the same parameter. **Rule**: When fixing a config threshold bug, grep all profiles for the same parameter and verify each value against the quantized domain boundaries. Threshold values are often copy-edited across profiles without re-checking the math.

- [2026-03-30] **PATTERN/unreachable_threshold_creates_dead_gate**: Balanced profile set sub_inference_min_confidence=0.60, but all models output 0.20-0.49. This made votes=0/3 permanently, which made agent_passed=False for 100% of 2585 virtual trades. The adaptive mechanism (Phase 67) could not fix this because it adapts the vote_threshold (0.66), not the min_confidence (0.60). **Rule**: When a gate has a "per-item threshold" (min_confidence for each vote) AND a "aggregate threshold" (vote_count required), both must be calibrated against the actual model output distribution. If the per-item threshold exceeds the model's maximum output, the aggregate threshold is irrelevant — the gate is dead regardless of how low you set it.

- [2026-03-30] **PATTERN/ceil_quantization_kills_adaptive_mechanisms_for_small_agent_teams**: With 3 window checks, math.ceil(3 * threshold) produces only 4 distinct values: 0 (t=0), 1 (t<=0.33), 2 (0.34<=t<=0.66), 3 (0.67<=t<=1.0). An adaptive mechanism that steps by 0.05 within the [0.51, 0.66] band produces 3 adaptations that all yield the same vote_required=2. The mechanism consumes its adaptation budget with zero behavioral effect. **Rule**: For any adaptive mechanism that targets a ceil()/floor()/round() formula, always verify that the step size crosses at least one integer boundary for the actual team size. If it doesn't, the mechanism is dead code. Make the mechanism quantization-aware: compute the boundary where the integer changes and jump directly to it.

- [2026-03-30] **PATTERN/compound_dead_mechanisms_mask_root_cause**: The 0-trade stall had two independent root causes that masked each other: (1) sub_inference_min_confidence=0.60 made votes=0 always, and (2) even if votes were non-zero, the adaptive mechanism couldn't adjust the vote threshold effectively. Fixing only one would still produce 0 trades — both needed simultaneous fixes. **Rule**: When diagnosing a multi-gate stall, trace each gate independently to determine if it can EVER pass. If gate A's output is always 0, gate B's threshold is irrelevant. Fix from the innermost bottleneck (per-vote confidence) outward (aggregate vote threshold).

> Aura (Eve) is a separate system. Her learnings now live in `P-90/.claude/learnings.md`.
> Phase 1 and Phase 2 findings migrated on 2026-03-23.

### Finding: Gap resolution PRDs become stale quickly
- Ralph's gap-resolution-phase11 PRD (30 stories, 0 passed) was generated from Phase 10 analysis
- By Phase 33, most gaps had already been fixed in Phases 19-32
- Fresh analysis found only real remaining gaps: profile parity, shutdown coverage, dispatcher holes
- **Lesson**: Run fresh gap analysis rather than working through stale PRDs

### Finding: Mock-based testing strategy for OANDA-dependent code
- ExecutionManager.__init__ requires OANDA credentials — mock it with __new__ + manual attribute setup
- Pure calculation methods (_calculate_base_tp_pips, _determine_exit_reason) can be tested directly
- Gate methods (can_trade, portfolio_risk) need MagicMock on fetch methods
- Position sizing has multiple fallback paths — test each independently
- **Lesson**: Separate pure calculation logic from API-dependent code for testability

### Phase 36: Test Coverage Expansion for Core Modules (2026-03-23)
- **Lesson**: Always test with `predicted_direction` not `direction` for ContinuousScanner — the attribute name matters. Use `os.chdir(tmpdir)` when code uses local `Path()` calls that can't be easily patched. Pre-set `orch.scanner = None` before calling get_system_status() on default-constructed Orchestrator.

### Phase 37: Critical Path Hardening (2026-03-23)
- **Lesson**: ScannerConfig has no `dry_run` field — use `enable_execution=False` instead. Modular ensemble lazy-inits via `_init_modular_ensemble()` — must patch the init method, not just set `_modular_ensemble=None`. Profile-applied `blocked_pairs` override config-level empty lists — explicitly set `scanner.config.blocked_pairs=[]` in tests.

1. **Fractional Kelly is essential for FX**: Full Kelly is too aggressive; f=0.33 (third-Kelly) balances growth vs drawdown. Rolling 50-trade window prevents stale estimates from early losing streaks.
2. **Confidence sigmoid mapping prevents extreme positions**: σ(x,k=4) maps [0,1]→[~0.02,~0.98] — confidence 0.3 gives 0.27x multiplier, 0.7 gives 0.73x. Steepness k=4 gives good discrimination without cliff effects.
3. **Drawdown-recovery is critical for survival**: During 10% drawdown (of 15% max), position size drops to 63% of normal. Floor at 30% ensures we don't freeze out during recovery.
4. **Regime multipliers stack multiplicatively with other factors**: LOW=1.3× allows bigger positions in calm markets, EXTREME=0.4× reduces to 40% in crisis. Combined with drawdown factor, crisis sizing can be as low as 12% of normal.

5. **Thompson Sampling naturally balances exploration/exploitation**: No need for complex UCB calculations. Beta(α,β) with online updates and ε-greedy provides robust exploration.
6. **Regime-conditional Beta distributions are key**: Agent "trend" may be α=10,β=5 in NORMAL regime but α=3,β=8 in LOW regime — TS naturally adapts weights per regime without manual tuning.
7. **Weight decay (0.995 every 20 updates) prevents stale beliefs**: Without decay, agents that performed well 200 trades ago dominate despite market regime shifts. Decay forgets at ~50% per 139 updates.
8. **Partial credit via score-proportional updates outperforms binary**: An agent scoring 0.8 on a winning trade gets α+=0.8, not α+=1. This captures calibration quality, not just direction.

9. **Lazy imports with try/except are the right pattern for optional modules**: All three wiring points (engine.py, execution.py, _team.py) use lazy init — system works identically without Phase 43/44 modules.
10. **Gate_details dict is the right vehicle for regime metadata**: Adding `regime_detector` sub-dict doesn't break any downstream consumers; they just check `if "regime_detector" in gate_details`.
11. **Calibration overlay pattern preserves backward compatibility**: Compute raw weighted_vote_score first, then optionally override with calibrated score. If calibrator fails, raw score flows through unchanged.
12. **Per-trade exception handling in exit evaluation prevents cascading failures**: One bad trade context (e.g., zero entry price) doesn't block exit evaluation for other trades.

### Cumulative Stats
- Phase 36-44: **1,508 passing tests** across 9 phases
- Phase 43-44: 5 new production modules (~3,000 lines source) + 3 pipeline integrations
- Key upgrade: 38% win rate should improve as regime detection + calibrated confidence + adaptive exits filter bad entries and improve exit timing

---

1. **Unwired modules are dead code**: Phase 44 created `adaptive_position_sizing.py` (627 lines) and `bayesian_agent_weights.py` (554 lines) but they were only called by tests. Production pipeline still used `DynamicPositionSizer` and flat-file weights. Gap analysis by dedicated research agents caught this immediately.
2. **Fallback-first wiring pattern is essential**: Every new module wired as "try adaptive → fall back to legacy" — this means zero risk of production regression. The `calculate_position_size()` method tries `_calculate_adaptive_position_size()` first, falls through to `DynamicPositionSizer` on any failure.
3. **Blend ratios smooth transitions**: Thompson Sampling weights blended 70/30 with flat weights (not 100% Bayesian). This prevents sudden weight shifts while the Beta distributions warm up. Same principle applies to position sizing — adaptive sizer is preferred but fallback is seamless.

4. **λ=0.94 is the RiskMetrics standard for FX daily correlation**: Higher λ (0.97) is too slow for FX regime changes; lower λ (0.90) overreacts to noise. 0.94 gives ~17-day effective window.
5. **Diversification multiplier D = 1/√(1 + avg_corr) is elegant**: When avg correlation to portfolio is 0, D=1.0 (full size). When avg corr is 0.7, D≈0.77. When corr is 1.0, D≈0.71. Natural concave decay.
6. **RISK_OFF correlation regime (all pairs moving together) is the most dangerous**: Average off-diagonal correlation > 0.6 signals contagion risk — reducing portfolio risk limit from 15% to ~10% is a critical safety valve.
7. **Union-Find stays as hard block, EWMA as soft adjustment**: The static correlation filter prevents double exposure. EWMA adjusts position size for partially correlated pairs. Two layers, different purposes.

8. **Anti-martingale with caps prevents Kelly's ruin**: Raw Kelly with winning streaks could suggest 3-5x normal size. Capping at 2x and flooring at 0.3x keeps sizing within safe bounds while still exploiting momentum.
9. **Regime-change resets are essential**: A 5-win streak in LOW volatility doesn't mean you should bet big entering HIGH volatility. The streak tracker resets on regime transitions, starting fresh in the new environment.
10. **Exponential streak multipliers are aggressive — use sparingly**: 1.3^3 = 2.197 (hits cap). 1/1.8^2 = 0.31 (near floor). Just 2-3 consecutive results produce significant sizing changes. This is by design — streak detection should react quickly.

11. **Parallel agent pattern works brilliantly for independent modules**: US-282, US-283, US-284 were developed simultaneously by 3 parallel agents. No conflicts because each touched different files/methods. Integration agent (US-287) verified they compose correctly.
12. **State persistence at sync_closed_trades_rl is the right place**: All module state (adaptive sizer, Bayesian weights, EWMA correlation) is persisted atomically after RL updates. Single sync point prevents partial state on crash.

### Cumulative Stats
- Phase 36-45: **~1,960 passing tests** across 10 phases
- Phase 45: 1 new module (EWMA correlation, 604 lines) + 5 production wiring edits + streak overlay (~180 lines)
- Phase 45 test breakdown: 392 Phase 45 tests + 60 Phase 44 regression tests = 452 total, 0 failures
- Production pipeline now fully adaptive: regime→sizing→correlation→execution→exit→bayesian learning

---

1. **Regime-conditional gates are composable**: Each regime profile is a self-contained dataclass that can override any combination of thresholds. The `apply_regime_gates()` function merges regime adjustments onto base thresholds, making it trivial to add new regimes or tweak existing ones without touching gate evaluation logic.

2. **Elder Triple Screen works well as a filter, not a signal generator**: The MTF confluence module scores existing signals rather than generating new ones. This design means it can be layered on top of any agent's output without changing the agent itself — just multiply confluence score into confidence.

3. **Session detection must handle midnight-crossing**: Tokyo session (23:00-08:00 UTC) wraps past midnight. The SessionDetector handles this by checking `hour >= start OR hour < end` for wrapping sessions vs `start <= hour < end` for normal ones. This edge case bit us in initial testing.

4. **Graduated penalty curves beat binary thresholds**: The ensemble conflict resolver uses linear interpolation between 5 penalty points instead of hard cutoffs. This gives the system smooth degradation — a 0.35 disagreement gets -12% vs the old flat -15%, while 0.50 correctly blocks entirely.

5. **Expectancy reframes optimization correctly**: Win rate alone is misleading. An agent with 30% win rate but 3:1 avg win/loss has positive expectancy ($0.30×3 - $0.70×1 = $0.20). The ExpectancyTracker captures this per agent per regime, enabling data-driven weight adjustments that optimize for returns, not just accuracy.

6. **Synthetic OHLCV data with controlled indicators**: For MTF confluence tests, we generate DataFrames where SMA slopes, RSI values, and ADX strength are predictable from the input prices. This makes tests deterministic without needing real market data.

7. **Integration tests verify module composition, not individual logic**: Phase 46 integration tests verify that regime gates → MTF confluence → session filter → ensemble conflict compose correctly. Each module's unit tests already verify internal logic — integration tests verify the contracts between modules.

### Cumulative Statistics
- Phase 36-46: ~2,270 passing tests across 11 phases
- Phase 46: 5 new modules, 310 new tests, 6/6 stories complete
- New capabilities: regime-adaptive gates, Elder confluence, session awareness, graduated ensemble penalties, expectancy tracking

1. **Lazy-init with fallback is the canonical wiring pattern**: Every Phase 46 module was wired using: try/import/init in `__init__()`, wrapped call in the hot path, except→fallback to legacy. This pattern (established in Phase 45) proved reliable again — 0 production crashes from integration failures.

2. **Legacy hard blocks interact with new graduated penalties**: The existing `_evaluate_uncertainty()` has a hard block at `model_disagreement > 0.30` computed from heuristic indicators. The new EnsembleConflictResolver operates on TCN/Ridge/RF model scores independently. Both must pass — the legacy check is a safety floor that the graduated penalty can't override.

3. **Position sizing chain order matters**: The final chain is: base → adaptive (Kelly+confidence+drawdown+regime) → regime multiplier → session multiplier → EWMA diversification → streak overlay. Each layer compounds multiplicatively, so a 0.65x regime × 0.75x session × 0.8x EWMA = 0.39x total. The floor protections in each layer prevent sizing from going below minimum lot.

4. **Test data must match heuristic expectations**: Random OHLCV data creates unpredictable heuristic values (close vs SMA, RSI direction, return sign). For deterministic tests of uncertainty/disagreement, use synthetic data with controlled monotonic trends so heuristics agree with the direction under test.

### Cumulative Statistics
- Phase 36-47: ~2,324 passing tests across 12 phases
- Phase 47: 0 new modules, 5 pipeline integrations, 54 new tests, 6/6 stories
- Full pipeline now active: regime gates → MTF confluence → session → ensemble conflict → expectancy → adaptive sizing chain

---

1. **Spread normalization is universally better than absolute pip thresholds**: Normalizing spread as a percentage of ATR (spread_pips / atr_pips) automatically adapts to any pair and any volatility condition. A 2-pip spread on EUR_USD (ATR=30) is very different from 2 pips on GBP_NZD (ATR=60). The 20% threshold works well as a default but JPY crosses and exotics need overrides (25%).

2. **Economic calendar currency-pair mapping catches more risks than direct pair matching**: A USD event affects EUR_USD, GBP_USD, USD_JPY, AUD_USD and 4 more pairs. Building the mapping once and checking it per-event is much cheaper than N×M pair-event comparisons.

3. **Test cache isolation is critical for calendar/fitness tests**: The EconomicCalendarFilter loads from a default cache file on init. Tests that write cache can pollute later tests that expect empty state. Always pass isolated cache_file paths in test configs.

4. **Strategy fitness composite scoring works better than any single metric**: Profit factor alone misses high-disagreement environments. Prediction correlation alone misses negative-PF scenarios. The weighted composite (PF 0.4 + corr 0.3 + disagree 0.3) catches both failure modes and degrades gracefully.

5. **Walk-forward efficiency (WFE) is the only reliable overfitting detector**: A high IS profit factor means nothing if OOS collapses. WFE = OOS_PF / IS_PF below 0.30 reliably identifies curve-fitted parameters. The optimizer explicitly writes recommendations WITHOUT auto-applying — manual review prevents silent degradation.

6. **SHAP-lite attribution with alignment scoring is surprisingly informative**: Simple direction alignment (did the agent vote the right direction relative to outcome?) combined with confidence weighting produces actionable weight recommendations. Agents below 45% accuracy over 50 trades should get 0.75x weight — this is cheaper and more interpretable than full SHAP.

7. **Auto-blacklisting needs minimum trade counts**: Without a minimum (10 trades), noise-driven blacklisting punishes pairs that simply haven't had enough observations. The cooldown cycle system (2 scan cycles) prevents permanent exclusion while still protecting against genuine underperformers.

### Cumulative Statistics
- Phase 36-48: ~2,457 passing tests across 13 phases
- Phase 48: 6 new modules, 133 new tests, 6/6 stories
- New modules NOT yet wired into production — Phase 49 will do pipeline integration
- Next priority: Wire spread_filter, economic_calendar, strategy_fitness, pair_blacklist, trade_attribution, walkforward_optimizer into execution.py and engine.py

---

**System reached full autonomous self-improvement architecture.**

Key closures this session:
## 2026-03-27 — Tier 5 Certification
- Episodic memory suppression gate wired into `execute_trade()` before Gate 1 — the last gap between Tier 4.9 and Tier 5. Buddy now blocks repeated losing patterns without human input.
- Aura brain stripped out of `src/aura/` — only bridge protocol remains (`signals.py`, `rules_engine.py`). ~210 lines of dead dispatch code removed from orchestrator + continuous.
- All 6 architectural fixes from Phase 48 audit verified clean: dispatch table, config flag regression test, attribution loop closed, FilterChain wired, Ralph reading PRD stories, Aura shutdown handling.

**Episodic memory full lifecycle confirmed:**
1. `execute_trade()` → `get_suppression_signal()` (earliest gate, fail-open)
2. `_append_journal_entry()` → `record_setup()` → episode_id stored in journal
3. `sync_closed_trades_rl()` → `record_outcome()` → memory learns win/loss

**Ralph is live but waiting:** `PerformancePRDGenerator` needs 30 days of trade outcomes for gap analysis. Not a failure — a dependency on production data. First analysis fires automatically once window is full.

**Architecture is now self-sustaining:** trade → outcome → RL weights + episodic memory + drift detection → config adjustment → Ralph PRD generation → code improvement → repeat. No human in the loop required.

**US-343 (DisagreementTracker):** Coefficient of variation (std/mean) is the correct disagreement metric because it normalizes spread relative to magnitude — allowing comparison across different confidence scales. Toxic pair detection via above/below-mean camp split is more stable than raw pair-score comparison. The 70% loss-rate threshold with 5-observation minimum prevents premature flagging.

**US-344 (FeatureHealthMonitor):** PSI bins must use the combined (recent + baseline) range to ensure consistent bucketing. Flooring bin counts at 1 (not 0) prevents log(0) in the PSI formula. The `feature_health_score = mean(1 - min(psi/0.50, 1.0))` formula maps directly to [0,1] and feeds WFEStabilityMonitor without any custom alert pathway.

**US-345 wiring lesson:** The "live wiring" test pattern (grep for `self._tracker.method()` in production files) is more reliable than unit tests that mock boundaries. Both should exist.

### Auto-extracted 2026-04-03
- **[2026-04-03]** `agent_accuracy` | uncertainty_was_warning for USD_JPY: score=0.80, lost $64.60 → *Lower max_uncertainty_score threshold*
- **[2026-04-03]** `pair_behavior` | EUR_AUD SHORT lost: 19.0p (conf=57%) → *Track EUR_AUD directional accuracy*
- **[2026-04-03]** `pair_behavior` | USD_CAD LONG won: 20.8p (conf=64%) → *Track USD_CAD directional accuracy*
- **[2026-04-03]** `sl_tp` | tp_could_be_wider for GBP_USD: TP hit in 5min → *Increase atr_tp_multiplier for GBP_USD*
- **[2026-04-03]** `pair_behavior` | GBP_USD SHORT won: 35.1p (conf=60%) → *Track GBP_USD directional accuracy*
- **[2026-04-03]** `pair_behavior` | EUR_JPY SHORT won: 20.0p (conf=56%) → *Track EUR_JPY directional accuracy*
- **[2026-04-03]** `agent_accuracy` | uncertainty_was_warning for USD_JPY: score=0.80, lost $64.60 → *Lower max_uncertainty_score threshold*
- **[2026-04-03]** `pair_behavior` | USD_JPY LONG lost: 12.9p (conf=68%) → *Track USD_JPY directional accuracy*
- **[2026-04-03]** `pair_behavior` | EUR_AUD SHORT lost: 19.0p (conf=57%) → *Track EUR_AUD directional accuracy*
- **[2026-04-03]** `pair_behavior` | USD_JPY LONG lost: 29.8p (conf=62%) → *Track USD_JPY directional accuracy*
- **[2026-04-03]** `pair_behavior` | USD_CAD LONG won: 20.8p (conf=64%) → *Track USD_CAD directional accuracy*
- **[2026-04-03]** `pair_behavior` | GBP_JPY SHORT lost: 16.2p (conf=56%) → *Track GBP_JPY directional accuracy*

### Auto-extracted 2026-04-04
- **[2026-04-04]** `sl_tp` | low_rr_ratio_loss for USD_CHF: R:R=1.07 (<1.2), lost $426.95 → *Reject trades with R:R < 1.2 — gate is enforced but log for pattern tracking*
- **[2026-04-04]** `exit_accountability` | exit_accountability_correct_reject: mean_reversion voted NO, USD_CHF LONG lost via sl_hit (-13.6p) — agent was right → *Maintain or boost mean_reversion weight for rejection accuracy*

### Auto-extracted 2026-04-06
- **[2026-04-06]** `pair_behavior` | EUR_GBP SHORT lost: 11.5p (conf=69%) → *Track EUR_GBP directional accuracy*
- **[2026-04-06]** `pair_behavior` | EUR_GBP SHORT lost: 11.5p (conf=69%) → *Track EUR_GBP directional accuracy*
- **[2026-04-06]** `pair_behavior` | NZD_USD LONG lost: 11.6p (conf=69%) → *Track NZD_USD directional accuracy*
- **[2026-04-06]** `exit_accountability` | exit_accountability_correct_reject: mean_reversion voted NO, NZD_USD LONG lost via sl_hit (-11.6p) — agent was right → *Maintain or boost mean_reversion weight for rejection accuracy*

### Auto-extracted 2026-04-08
- **[2026-04-08]** `pair_behavior` | AUD_JPY LONG lost: 17.7p (conf=63%) → *Track AUD_JPY directional accuracy*
- **[2026-04-08]** `exit_accountability` | exit_accountability_correct_reject: mean_reversion voted NO, AUD_JPY LONG lost via sl_hit (-17.7p) — agent was right → *Maintain or boost mean_reversion weight for rejection accuracy*

### Auto-extracted 2026-04-09
- **[2026-04-09]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $534.57 → *Lower max_model_disagreement threshold*
- **[2026-04-09]** `pair_behavior` | USD_CHF LONG lost: 16.8p (conf=65%) → *Track USD_CHF directional accuracy*
- **[2026-04-09]** `exit_accountability` | exit_accountability_correct_reject: mean_reversion voted NO, USD_CHF LONG lost via sl_hit (-16.8p) — agent was right → *Maintain or boost mean_reversion weight for rejection accuracy*

### Auto-extracted 2026-04-12
- **[2026-04-12]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $381.50 → *Lower max_model_disagreement threshold*
- **[2026-04-12]** `pair_behavior` | EUR_GBP SHORT lost: 11.5p (conf=69%) → *Track EUR_GBP directional accuracy*
- **[2026-04-12]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $290.00 → *Lower max_model_disagreement threshold*
- **[2026-04-12]** `pair_behavior` | NZD_USD LONG lost: 11.6p (conf=69%) → *Track NZD_USD directional accuracy*
- **[2026-04-12]** `exit_accountability` | exit_accountability_correct_reject: mean_reversion voted NO, NZD_USD LONG lost via sl_hit (-11.6p) — agent was right → *Maintain or boost mean_reversion weight for rejection accuracy*
- **[2026-04-12]** `pair_behavior` | AUD_JPY LONG lost: 17.7p (conf=63%) → *Track AUD_JPY directional accuracy*
- **[2026-04-12]** `exit_accountability` | exit_accountability_correct_reject: mean_reversion voted NO, AUD_JPY LONG lost via sl_hit (-17.7p) — agent was right → *Maintain or boost mean_reversion weight for rejection accuracy*
- **[2026-04-12]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $534.57 → *Lower max_model_disagreement threshold*
- **[2026-04-12]** `pair_behavior` | USD_CHF LONG lost: 16.8p (conf=65%) → *Track USD_CHF directional accuracy*
- **[2026-04-12]** `exit_accountability` | exit_accountability_correct_reject: mean_reversion voted NO, USD_CHF LONG lost via sl_hit (-16.8p) — agent was right → *Maintain or boost mean_reversion weight for rejection accuracy*
- **[2026-04-12]** `pair_behavior` | EUR_AUD SHORT lost: 27.6p (conf=59%) → *Track EUR_AUD directional accuracy*
- **[2026-04-12]** `exit_accountability` | exit_accountability_correct_reject: mean_reversion voted NO, EUR_AUD SHORT lost via sl_hit (-27.6p) — agent was right → *Maintain or boost mean_reversion weight for rejection accuracy*
- **[2026-04-12]** `sl_tp` | low_rr_ratio_loss for EUR_JPY: R:R=1.08 (<1.2), lost $114.34 → *Reject trades with R:R < 1.2 — gate is enforced but log for pattern tracking*
- **[2026-04-12]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $114.34 → *Lower max_model_disagreement threshold*
- **[2026-04-12]** `pair_behavior` | EUR_JPY SHORT lost: 20.8p (conf=55%) → *Track EUR_JPY directional accuracy*

- [2026-04-15] **PATTERN/per_pair_models_not_retrained_despite_global_retrain**: transformer_direction_best.keras updated today (Apr 15) but ALL per-pair models (EUR_GBP, USD_CHF, EUR_JPY, AUD_JPY, EUR_AUD, NZD_USD) still show Mar 24 timestamps. The autonomous trainer may only retrain the global model, not per-pair models. Per-pair ridge_confidence, lgbm_risk, lgbm_momentum, histgb_direction all stale at 22d. **Implication: retrain command must include per-pair model refresh or the ensemble still runs on stale sub-models.**

- [2026-04-15] **PATTERN/rr_gate_violated_on_eur_jpy_trade_6**: EUR_JPY trade #6 (2026-04-09) closed with rr_ratio=1.08, below the hard 1.2:1 minimum R:R gate. Either ATR shifted between gate evaluation and order placement, or the gate check uses pre-execution values. 1 of 10 trades violated a supposedly hard gate. **Implication: R:R must be re-validated at execution time.**

- [2026-04-15] **PATTERN/mfe_zero_on_all_10_proves_directional_model_failure**: Every closed trade (10/10) has MFE=0.0 pips. Price never moved even 1 pip in predicted direction before SL hit. This is not risk management failure. The models predict wrong direction with 100% frequency at 0.549-0.702 confidence. No SL/TP tuning or agent weight adjustment fixes wrong-direction predictions.

- [2026-04-16] **PATTERN/feedback_loop_closed**: Reflection write → staging → merge pipeline verified end-to-end.

- [2026-04-16] **PATTERN/quiet_cycle_N12**: scanned 30 pairs, 0 tradeable, 0 executed

### Trade 1243 Reflection — EUR_GBP SHORT Loss (-$216, -11.8p) — 2026-04-16

- [2026-04-16] **PATTERN/low_regime_sl_mult_0.8_still_in_code_3rd_observation**: Trade 1243 EUR_GBP SHORT in LOW regime used sl_mult=0.8 (config.py:492). SL=11.8p on ATR=4.77p (SL/ATR=2.47x raw, but regime multiplied 0.8*base). Trading rule promoted 2026-04-15 mandates LOW sl_mult>=1.2, but code default unchanged. This is the 3rd LOW-regime loss caused by this (1220, 1236, 1243). **The rule exists but the code doesn't enforce it.**

- [2026-04-16] **PATTERN/eur_gbp_short_low_regime_0pct_winrate**: EUR_GBP SHORT in LOW regime: Trade 1124 (-$381, -11.5p, MFE=0), Trade 1236 (-$301, -11.6p, MFE=0), Trade 1243 (-$216, -11.8p, MFE=0). 0/3 wins, -$898 total, all MFE=0. This pair+direction+regime has no edge. Consider pair-level blocking when per-pair win rate < 20% over 3+ trades.

- [2026-04-16] **PATTERN/ensemble_consensus_HOLD_traded_SHORT**: Trade 1243 had ensemble_consensus_direction=HOLD but executed SHORT. Model disagreement=0.50 (max divergence), uncertainty=0.80, confidence_variance=0.68. The ensemble literally said "don't take a side" and the system took one anyway. WVS=0.83 masked the signal because risk_sentinel (1.0) and execution_quality (0.78) inflated the score without directional insight.

- [2026-04-16] **PATTERN/uncertainty_0.80_soft_penalty_only_minus_0.05**: Trade 1243 uncertainty_score=0.80 (extreme), but confidence_delta was only -0.05 (conf 0.727→0.677). The soft penalty function is near-constant regardless of uncertainty magnitude: 0.45→-0.05, 0.80→-0.05. There is no proportional scaling. An uncertainty of 0.80 should produce a much larger penalty than 0.45.

- [2026-04-16] **PATTERN/15th_consecutive_mfe_zero**: Trade 1243 is the 15th consecutive trade with MFE=0.0 pips. The directional model has not produced a single correct directional prediction in 15 trades. Total loss: ~$3,743. This is no longer a "streak" — it's systematic directional failure.


### Self-Heal Cycle #1 — 16th consecutive loss (Trade 1261 USD_CAD LONG -$136, -7.4p) — 2026-04-16T16:03Z

- [2026-04-16] **PATTERN/16th_consecutive_mfe_zero_streak**: Trade 1261 USD_CAD LONG (entered 2026-04-16T03:03, closed 14:05 sl_hit, MFE=0.0p, MAE=7.4p) is the 16th consecutive losing trade with MFE=0. Cumulative drawdown across the streak (1124->1261): -$3,879 realized, all from 16 trades, 0 wins, every single one with MFE=0.0 pips. The directional ensemble has produced zero correct first-bar predictions in the entire streak.

- [2026-04-16] **PATTERN/fresh_global_model_still_lost_disproves_staleness_only_theory**: Trade 1255 USD_CHF LONG ran on the global ensemble (USD_CHF has no per-pair model directory under trained_data/models/). transformer_direction_best.keras was retrained today (Apr 16 11:24 mtime); even within the same calendar-day retrain window the ensemble still produced a wrong-direction prediction with MFE=0. Implication: per-pair staleness is one cause but not the sole cause -- the directional pipeline itself produces inverted-sign predictions. Investigate feature normalization mismatch between training and inference, or label-sign convention drift in the latest training run.

- [2026-04-16] **PATTERN/per_pair_retrain_skipped_8_of_14_pairs_on_apr16_run**: trained_data/models/ shows only 6 pair dirs touched today (EUR_AUD, EUR_CHF, EUR_JPY, GBP_AUD, GBP_CHF, USD_JPY all Apr 16 06:42-07:22). 8 pairs still on Mar 24 (22d stale): AUD_NZD, AUD_USD, EUR_GBP, EUR_USD (Mar 18, 29d), GBP_JPY, GBP_USD, USD_CAD. EUR_GBP and USD_CAD account for 5 of the 16 streak losses (1124, 1236, 1243, 1249, 1261). The autonomous trainer global retrain does not refresh per-pair sub-models. Direct cause of recurring EUR_GBP SHORT and USD_CAD LONG losses on stale ridge/lgbm/histgb component models.

- [2026-04-16] **PATTERN/rl_position_sizer_train_data_24d_old**: trained_data/rl_train_data.npz mtime is 2026-03-23 16:02 (24d old). Diagnostic flagged rl_model_staleness=868681s (~10d) exceeds critical threshold. All 16 losing trades sized at exactly 2.5 lots -- RL sizer no longer adapting to recent drawdown context (kept 2.5 lots even on the 16th consecutive loss with account ~$3.7k down from the streak). Next action: retrain the RL position sizer or fall back to a hard fixed-fraction cap at 1.0 lot during the post-loss recovery window.

- [2026-04-16] **PATTERN/trend_agent_misreads_strong_adx_as_neutral**: Trade 1261 trend agent reason: trend neutral (ADX 54) -- ADX 54 is a STRONG trend reading (Wilder threshold for very strong is 50+). Trade 1255 also got trend neutral (ADX 56). Trade 1243 got trend neutral (ADX 64). The trend agent neutral threshold is mis-set: it should classify ADX>=40 as trending and provide directional confirmation, but it returns neutral through ADX 64. This systematically masks counter-trend entries from the gate. Source: 3 consecutive trades 1243, 1255, 1261 -- meets the 3+ promotion threshold.

- [2026-04-16] **PATTERN/self_heal_cycle1_levers_maxed**: Cycle 1 history applied all 5 supported config keys to their safe ceilings (atr_sl_multiplier 1.0->1.2, max_model_disagreement 0.65->0.25, min_confidence 50->68, weighted_vote_threshold 0.45->0.85, risk_per_trade_pct 0.02->0.025). Pending block is empty. Further config tightening will freeze trading without addressing the directional-model root cause. The actionable corrective is OUTSIDE the config plane: per-pair retrain coverage + RL sizer retrain + trend agent ADX threshold fix. This reflection deliberately skips stacking a 6th config adjustment.

<!-- BUDDY_OUTCOMES_BEGIN -->
## Recent Trade Outcomes (auto-rendered)

_This section is machine-managed — do not edit. Source: `.claude/learnings_outcomes.jsonl`._

| ts_utc | trade_id | pair | dir | conf | pnl_pips | exit | regime | wvs | dissent |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04-03T20:10:35.490907987Z | 1124 | EUR_GBP | SHORT | 0.69 | -11.5 | sl_hit | LOW | 0.82 |  |
| 2026-04-06T15:10:54.795532270Z | 1132 | NZD_USD | LONG | 0.69 | -11.6 | sl_hit | LOW | 0.79 |  |
| 2026-04-07T01:39:03.200109448Z | 1136 | AUD_JPY | LONG | 0.63 | -17.7 | sl_hit | LOW | 0.82 |  |
| 2026-04-09T00:07:27.312160245Z | 1149 | EUR_AUD | SHORT | 0.59 | -27.6 | sl_hit | NORMAL | 0.84 |  |
| 2026-04-09T13:33:56.560058146Z | 1145 | USD_CHF | LONG | 0.65 | -16.8 | sl_hit | HIGH | 0.83 |  |
| 2026-04-09T19:26:33.512391421Z | 1157 | EUR_JPY | SHORT | 0.55 | -20.8 | sl_hit | NORMAL | 0.90 |  |
| 2026-04-13T23:51:19.772609364Z | 1189 | AUD_JPY | LONG | 0.68 | -15.5 | sl_hit | NORMAL | 0.87 |  |
| 2026-04-14T12:26:39.900044035Z | 1195 | USD_CHF | LONG | 0.70 | -11.3 | sl_hit | NORMAL | 0.90 |  |
| 2026-04-14T15:00:02.492657245Z | 1199 | EUR_JPY | SHORT | 0.64 | -16.5 | sl_hit | NORMAL | 0.90 |  |
| 2026-04-15T02:46:03.058288459Z | 1220 | EUR_AUD | LONG | 0.64 | -19.7 | sl_hit | LOW | 0.76 |  |
| 2026-04-15T21:04:55.052696780Z | 1236 | EUR_GBP | SHORT | 0.65 | -11.6 | manual_close | LOW | 0.83 |  |
| 2026-04-15T21:37:17.465167825Z | 1232 | USD_JPY | LONG | 0.67 | -17.9 | sl_hit | LOW | 0.83 |  |
| 2026-04-16T02:35:27.612125000Z | 1249 | USD_CAD | LONG | 0.65 | -12.7 | sl_hit | LOW | 0.83 |  |
| 2026-04-16T03:00:14.749581093Z | 1255 | USD_CHF | LONG | 0.67 | -8.0 | sl_hit | LOW | 0.80 |  |
| 2026-04-16T14:05:58.109694358Z | 1261 | USD_CAD | LONG | 0.64 | -7.4 | sl_hit | LOW | 0.83 |  |
| 2026-04-16T14:22:10.529471256Z | 1243 | EUR_GBP | SHORT | 0.68 | -11.8 | sl_hit | LOW | 0.83 |  |

<!-- BUDDY_OUTCOMES_END -->

### Auto-extracted 2026-05-13
- **[2026-05-13]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $381.50 → *Lower max_model_disagreement threshold*
- **[2026-05-13]** `pair_behavior` | EUR_GBP SHORT lost: 11.5p (conf=69%) → *Track EUR_GBP directional accuracy*
- **[2026-05-13]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $290.00 → *Lower max_model_disagreement threshold*
- **[2026-05-13]** `pair_behavior` | NZD_USD LONG lost: 11.6p (conf=69%) → *Track NZD_USD directional accuracy*
- **[2026-05-13]** `pair_behavior` | AUD_JPY LONG lost: 17.7p (conf=63%) → *Track AUD_JPY directional accuracy*
- **[2026-05-13]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $534.57 → *Lower max_model_disagreement threshold*
- **[2026-05-13]** `pair_behavior` | USD_CHF LONG lost: 16.8p (conf=65%) → *Track USD_CHF directional accuracy*
- **[2026-05-13]** `pair_behavior` | EUR_AUD SHORT lost: 27.6p (conf=59%) → *Track EUR_AUD directional accuracy*
- **[2026-05-13]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $114.34 → *Lower max_model_disagreement threshold*
- **[2026-05-13]** `pair_behavior` | EUR_JPY SHORT lost: 20.8p (conf=55%) → *Track EUR_JPY directional accuracy*
- **[2026-05-13]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $107.17 → *Lower max_model_disagreement threshold*
- **[2026-05-13]** `pair_behavior` | AUD_JPY LONG lost: 15.5p (conf=68%) → *Track AUD_JPY directional accuracy*
- **[2026-05-13]** `agent_accuracy` | uncertainty_was_warning for USD_CHF: score=0.80, lost $365.90 → *Lower max_uncertainty_score threshold*
- **[2026-05-13]** `agent_accuracy` | disagreement_predicted_loss: disagreement=0.50, lost $365.90 → *Lower max_model_disagreement threshold*
- **[2026-05-13]** `pair_behavior` | USD_CHF LONG lost: 11.3p (conf=70%) → *Track USD_CHF directional accuracy*
- **[2026-05-13]** `pair_behavior` | EUR_JPY SHORT lost: 16.5p (conf=64%) → *Track EUR_JPY directional accuracy*
