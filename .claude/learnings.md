# Buddy Trading Learnings

Date-stamped insights extracted from trade outcomes, scan analysis, and system behavior. Patterns that repeat 3+ times get promoted to `.claude/rules/`. Archive: `.claude/learnings-archive.md`.

---

## Promotion Log
- [2026-03-23] **US-225**: Promoted 6 cross-cutting patterns to `.claude/rules/improvement.md`:
  - JSON Safety Gates (31 observations across Phases 4-34)
  - Retry & Robustness Gates (27 observations across Phases 4-34)
  - State Persistence Gates (8 observations across Phases 17-34)
  - Test Coverage Gates (8 observations across Phases 28-34)
  - Config Validation Gates (6 observations across Phases 12-34)
  - Silent Exception Prevention (4 observations across Phases 29-34)

## Active Learnings

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

### Phase 64 AdaptiveDrawdownCeiling — 2026-03-27

- [2026-03-27] **PATTERN/direction_inversion_requires_guard_inversion**: AdaptiveDrawdownCeiling raises max_drawdown_pct (INCREASE) vs AdaptiveMomentumFloor which lowers min_momentum (DECREASE). This flip means every guard must be inverted: floor guard (`candidate >= floor`) becomes ceiling guard (`candidate <= ceiling`), total_reduction becomes total_increase, and the state field name changes accordingly. **Rule**: When mirroring an adaptive floor for an inverted domain (ceiling vs floor), make a checklist of ALL direction-dependent components: (1) `candidate = current ± step`, (2) guard direction `<= ceiling` vs `>= floor`, (3) state variable name `total_increase` vs `total_reduction`, (4) log message text `raised` vs `reduced`, (5) test assertion `result > input` vs `result < input`. Missing any one produces silent incorrect direction.

- [2026-03-27] **PATTERN/conservative_risk_guards_reflect_risk_asymmetry**: Phase 64 uses step=0.001 (vs 0.02 for momentum = 20x smaller), cooldown=30 (vs 20 = 50% longer), max_adaptations=3 (vs 5 = 40% fewer). The conservatism encodes the fact that raising max_drawdown_pct directly affects how much portfolio loss the system will accept before blocking a trade. **Rule**: Adaptive gates that touch risk-of-ruin parameters (drawdown thresholds, max position size, max leverage) should have guards 2-5x more conservative than gates that touch signal quality parameters (confidence, momentum). Encode this asymmetry explicitly in the class docstring and default constructor values.

- [2026-03-27] **PATTERN/three_gate_adaptive_loop_milestone_complete**: Phase 64 completes the full observe→adapt loop for all three gates: Confidence gate (Phase 58: RawConfidenceRecorder → ThresholdGapAnalyzer → AdaptiveConfidenceFloor), Momentum gate (Phase 61-62: MomentumScoreRecorder → MomentumGapAnalyzer → AdaptiveMomentumFloor), Risk gate (Phase 63-64: RiskRejectRecorder → RiskGateAnalyzer → AdaptiveDrawdownCeiling). Each gate now has: (a) a rolling recorder capturing the raw model output, (b) a gap analyzer classifying rejects by proximity to the threshold, (c) an adaptive floor/ceiling that auto-adjusts the threshold when structurally near-threshold. This closes the primary diagnostic loop for the system stall (0 trades in 24 scan cycles).

### Phase 63 Risk Gate Diagnostics — 2026-03-27

- [2026-03-27] **PATTERN/signal_field_names_differ_between_inference_and_engine**: The `ModularSignal` dataclass in `modular_inference.py` stores `rf_drawdown_pips` (pips value) while the PRD referenced `rf_drawdown_pct`. The local variable `rf_drawdown_pct` exists inside `modular_inference.py` but never propagates to the Signal. engine.py already performed the conversion inline at `details["drawdown"] = float(signal.rf_drawdown_pips) / 10000.0`. **Rule**: When wiring a new recorder at a signal site in engine.py, always grep the actual Signal dataclass definition (results.py / modular_inference.py Signal class) before writing record() calls — PRD field names may reflect internal local variables, not the exported Signal fields. The conversion from pips to pct must mirror the existing inline conversion.

- [2026-03-27] **PATTERN/inverted_gap_direction_requires_inverted_near_miss_logic**: For confidence/momentum gates, gap = threshold - score (positive = approaching threshold from below = near miss). For the risk gate, gap = score - threshold (positive = over threshold = fail). This inversion affects both the classification thresholds AND the near_miss_count calculation: a near-miss for the risk gate is a record where `(rf_drawdown_pct - max_drawdown_pct) <= 0.005` (just barely over). **Rule**: For every new gate diagnostic, explicitly document gap direction in the module docstring with sign conventions. The sign choice propagates through gap_at_p50, classification boundary comparisons, near_miss count logic, and recommendation text. Inverting one without the others produces silent misclassification.

- [2026-03-27] **PATTERN/three_gate_diagnostic_coverage_milestone_confidence_momentum_risk**: Phase 63 completes the three-gate diagnostic coverage pattern: Phase 58 (confidence gate: RawConfidenceRecorder + ThresholdGapAnalyzer + AdaptiveConfidenceFloor), Phase 61-62 (momentum gate: MomentumScoreRecorder + MomentumGapAnalyzer + AdaptiveMomentumFloor), Phase 63 (risk gate: RiskRejectRecorder + RiskGateAnalyzer). Each phase follows the same observe→analyze→classify→recommend pattern. The risk gate only has OBSERVE+ANALYZE (no adaptive floor yet) because the max_drawdown_pct threshold has different safety implications than min_confidence or min_momentum — lowering the drawdown threshold is higher risk. **Rule**: Complete the observe→analyze pair before adding adapt. The adapt phase (equivalent of Phase 58 AdaptiveConfidenceFloor or Phase 62 AdaptiveMomentumFloor) for the risk gate is a future phase and should only fire for NEAR_THRESHOLD with conservative guards.

### Phase 62 AdaptiveMomentumFloor — 2026-03-27

- [2026-03-27] **PATTERN/decrement_before_check_semantics_for_cooldown_gates**: In AdaptiveMomentumFloor.tick(), the cooldown decrement happens BEFORE the guard check. This means `cooldown=3` blocks for 2 ticks (not 3) before the next adaptation fires: tick after adaptation sets cooldown=3, tick2 decrements to 2 (check 2>0 → block), tick3 decrements to 1 (check 1>0 → block), tick4 decrements to 0 (check 0>0 → false → fires). **Rule**: When writing interval-gate tests for decrement-before-check patterns, always trace through the exact tick sequence rather than assuming `cooldown=N` means N blocking ticks. Document the semantics in the class docstring with a concrete example.

- [2026-03-27] **PATTERN/observe_then_adapt_pipeline_is_safer_than_adapt_immediately**: Phase 61 (observe) was built before Phase 62 (adapt). This deliberate sequencing meant by the time AdaptiveMomentumFloor was wired, MomentumGapAnalyzer's classification (NEAR_THRESHOLD / MODERATE / FAR_BELOW) was already statistically verified. If adaptation had been added in Phase 61, it would have fired on INSUFFICIENT_DATA (< 3 fail records), producing spurious config mutations. **Rule**: For any auto-adaptation mechanism: always build and test the diagnostic signal in a separate phase before coupling the actuation logic. The diagnostic phase also reveals the appropriate thresholds for adaptation guards.

- [2026-03-27] **PATTERN/sharing_classification_variable_across_analyze_and_adapt_in_same_periodic_block**: The `_mga_cls` variable computed from `MomentumGapAnalyzer.analyze()` is reused immediately by `adaptive_momentum_floor.tick(_mga_cls, ...)` in the same `if self._scan_cycle_count % 50 == 0:` block. This avoids double-reading the recorder and ensures the adaptation decision is based on the exact same classification that was logged. **Rule**: When two modules in the same periodic block need the same derived signal, cache it in a local variable and share it. Never call the expensive source twice, and never let the two calls diverge on a race condition.

### Phase 61 Momentum Gate Diagnostics — 2026-03-27

- [2026-03-27] **PATTERN/domain_scale_determines_threshold_constants**: MomentumGapAnalyzer uses near-miss thresholds of 0.05/0.15 while ThresholdGapAnalyzer (confidence domain) uses 3.0/8.0. Both classify the same conceptual gap but the constants differ by ~60x. The momentum domain is XGBoost probability (0-1) while confidence is a scaled score (0-100). **Rule**: When mirroring a diagnostic module across domains, identify the scale factor first (XGBoost outputs 0-1, confidence is 0-100 here), then rescale ALL threshold constants proportionally. Never copy constants across domain boundaries without explicit scale analysis.

- [2026-03-27] **PATTERN/periodic_diagnostic_cadence_should_match_signal_velocity**: MomentumGapAnalyzer fires every 50 scans (slow) vs ScanHealthSynthesizer every 10 scans (fast). Momentum scores change with XGBoost model drift — which is structural and slow-moving. Confidence scores change with every penalty application — fast-moving. **Rule**: Match diagnostic cadence to signal velocity. Fast-changing signals (confidence, penalties) need frequent synthesis (10 scans). Slow-changing signals (model weights, XGBoost quality) can tolerate infrequent analysis (50 scans) to avoid log noise.

- [2026-03-27] **PATTERN/inline_import_avoids_circular_dependency_in_periodic_path**: The MomentumGapAnalyzer periodic call in engine.py uses `from src.scanner.momentum_gap_analyzer import analyze as _mga_analyze` inside the `if` branch rather than a top-level import. This avoids any potential circular import chain (engine imports analyzer which might import scanner types). **Rule**: For periodic diagnostic calls in large orchestrators, prefer inline imports inside the periodic gate rather than top-level imports. This keeps the hot path free of import overhead on every call and avoids circular dependency chains.

- [2026-03-27] **PATTERN/wiring_verification_grep_in_integration_tests_catches_missing_call_sites**: Phase 61 integration tests assert `"_momentum_recorder.record(" in engine_src` rather than relying on unit-test mocks. This directly caught when the `save_state()` call was not yet added — the wiring grep failed before any behavioral test ran. **Rule**: Per Live Wiring Verification Gates: always add literal grep assertions for critical call sites in integration tests. Unit test mocks pass even when the production call is missing; only source-file grep catches the gap.

### Bug-Squash Session — 2026-03-28

- [2026-03-28] **PATTERN/variable_shadowing_in_parameter_reuse_path**: In `position_sizing._calculate_base_position_size()`, the fix for C-4 (account_equity rounding for small accounts) introduced a new bug: the function parameter `account_equity: float` was re-assigned via `getattr(self.config, 'account_equity', 100000)` (which always returned 100000 since the config has no such field). This pattern — reassigning a function parameter with a fallback that always fires — completely silenced the fix. **Rule**: When adding a branch that depends on a function parameter, verify the parameter is still in scope and not already shadowed. `flake8 --select=F841` catches assignment-to-unused but not parameter shadowing; static analysis (pyright, mypy) is needed to catch the shadowing class.

- [2026-03-28] **PATTERN/guard_condition_sign_assumptions_for_loss_pnl**: The Kelly calculation guard `if avg_loss <= 0` was intended to handle zero/invalid loss values but fired on EVERY call because loss PnL values are negative by definition (e.g., -50, -100). The original author assumed `avg_loss` would be a positive magnitude, but the accumulation stores signed PnL. This class of sign-assumption bug is subtle because the code compiles and runs without error — it just silently disables the feature. **Rule**: When computing a ratio or guard from financial PnL values, explicitly document and verify the sign convention. Add an assertion or log the raw value at DEBUG level to catch sign inversions early.

- [2026-03-28] **PATTERN/recursive_intelligence_persistence_lag**: Three persistence modules (`weight_learner.py`, `learner.py`, `persistence.py`) in `src/recursive_intelligence/` all used raw `write_text()` for JSON persistence while the scanner automation modules already had `safe_json_write()` available. The lag occurred because the recursive intelligence modules were extracted/abstracted from Buddy's modules but did not carry forward the atomic-write infrastructure. **Rule**: When extracting a module into a shared library, audit ALL persistence paths and ensure they use the same safe I/O infrastructure as the source system. A new abstraction layer is not an excuse for regressing persistence safety.

- [2026-03-28] **PATTERN/interval_gating_initialization_without_counter**: `_drift_auto_remediation_dispatch()` computed `_check_interval` from config but never used it — it ran `check_and_remediate()` every dispatch cycle. A `_remediation_cycle_counter` attribute was initialized in `__init__` (signaling intent) but was never incremented. This gap between intent (the counter attribute) and implementation (no gate) is a class of bug that does not produce errors — only unnecessary overhead. **Rule**: When you initialize a counter for interval-gating, immediately write the gating logic (`if counter % interval == 0:`) in the same commit to prevent the counter from becoming permanently orphaned.

- [2026-03-28] **PATTERN/stale_bug_report_entries_for_wired_modules**: H-2 (EnsembleDivergenceMonitor) and H-4 (SignalFunnelTracker) were marked as unresolved in the bug report but were already fixed in the codebase. The automated scan that generated the report ran at 01:46; the wiring was committed in a session that ran between the scan and the squash. **Rule**: The bug-squash task must verify each open bug against the CURRENT source code (grep for call sites) before attempting a fix. Never apply a patch to a call site that may already exist.

### Phase 60 Unified Scan Health Synthesizer — 2026-03-27
- [2026-03-27] **PATTERN/synthesis_layer_closes_observability_to_action_gap**: Building multiple independent observability modules (Phase 56-59) without a synthesis layer leaves the operator with N log streams to mentally combine. A read-only ScanHealthSynthesizer that reads from all monitoring modules and returns a ranked bottleneck list + single top_action converts fragmented telemetry into a single decision point. The operator should never have to cross-reference 6 modules — the synthesis layer does it for them.
- [2026-03-27] **PATTERN/multiplicative_impact_scores_beat_categorical_priority**: Assigning impact scores as floats (0.0-1.0) rather than integer priority tiers allows fine-grained discrimination when multiple bottlenecks of the same type coexist. CONFIDENCE_TOO_HIGH = 0.75 * min(gap_at_p75/20.0, 1.0) means a 20pt gap scores 0.75 while a 10pt gap scores 0.375 — they both get surfaced but in correct proportion. Categorical priorities would flatten this distinction.
- [2026-03-27] **PATTERN/stateless_synthesizer_avoids_stale_data**: The ScanHealthSynthesizer has no state of its own — it reads live from each source module every time synthesize() is called. This prevents the synthesizer from lagging behind the modules it reads, which would happen if it cached data. The cost is a small read overhead every 10 scans; the benefit is always-fresh synthesis with no state management complexity.
- [2026-03-27] **PATTERN/critical_vs_degraded_threshold_captures_fixability**: The CRITICAL vs DEGRADED distinction captures whether a problem is fixable autonomously. DEGRADED = near-miss cluster > 50% means "lower threshold 1pt and it probably resolves." CRITICAL = stall + ensemble divergence means "needs retraining — no config change will fix this." This distinction makes log severity semantically meaningful rather than just volumetric.

### Phase 59 Signal Funnel Analytics & Gate Proximity Reporting — 2026-03-27
- [2026-03-27] **PATTERN/funnel_view_separates_confidence_from_gate_attrition**: Phase 58's RawConfidenceRecorder tells you whether signals exist before penalties. Phase 59's SignalFunnelTracker tells you what gate kills them AFTER confidence is cleared. Both are required: if Phase 57/58 fixes unblock confidence but momentum_gate kills everything, you need a different lever (momentum tuning, not threshold reduction). The dominant_kill_stage field in 2 lines diagnoses which gate deserves investigation next.
- [2026-03-27] **PATTERN/near_miss_rate_as_threshold_actionability_signal**: A confidence gap of 1-3pt between the pair's score and min_confidence is an "actionable near miss" — a small threshold change unblocks it. A gap of 10pt is structural (the pair's model output is genuinely weak). near_miss_rate_pct > 50% means the current threshold is slightly too high relative to the distribution, not that signals are weak. This two-signal cross-reference (ThresholdGapAnalyzer NEAR_THRESHOLD + GateProximityReporter near_miss_rate > 50%) provides confident evidence for a 1-2pt threshold relaxation.
- [2026-03-27] **PATTERN/ensemble_divergence_as_pre_penalty_structural_signal**: When TCN confidence (tcn_prob → 0-100 scale) and Ridge confidence consistently diverge by >20pp, the ensemble_disagreement multiplier fires before any other penalty. Tracking abs(tcn_conf - ridge_conf) per pair per scan reveals whether divergence is structural (models consistently disagree = feature drift, retrain warranted) or transient (resolved after Phase 57 fixes). A high_divergence_rate > 30% across 500 snapshots is the definitive retrain signal that no threshold or penalty adjustment can fix.
- [2026-03-27] **PATTERN/wiring_at_computation_site_not_gate_site**: EnsembleDivergenceMonitor.record() must be called where tcn_conf and ridge_conf are COMPUTED (inside the per-pair ensemble block), not where gate decisions are made. The gate might short-circuit early, but divergence tracking needs data on every pair regardless of whether it passes any gate. Placing monitoring at the computation site ensures complete coverage and enables per-pair divergence profiles.
- [2026-03-27] **PATTERN/integration_tests_must_grep_call_sites**: Unit tests for wiring phases pass even when production call sites are missing (because unit tests mock the module boundaries). Integration tests that grep source files for method names (`assert "_signal_funnel.record_scan(" in cont_src`) enforce that the call site exists in the production code. This pattern, now in improvement.md as Live Wiring Verification Gates, was the lesson from Phase 56's live wiring audit.

### Phase 58 Signal Unblock Verification & Adaptive Confidence Floor — 2026-03-28
- [2026-03-28] **PATTERN/pre_penalty_recording_required_for_threshold_diagnosis**: Without capturing raw confidence scores BEFORE any penalty modification, you cannot distinguish "signals exist but penalties destroy them" from "signals are intrinsically too weak". A RawConfidenceRecorder inserted before the first penalty block is the observability primitive that makes all downstream analysis meaningful. The JSONL rolling buffer (maxsize=500) keeps overhead constant regardless of scan count.
- [2026-03-28] **PATTERN/distribution_gap_analysis_for_threshold_calibration**: The `gap_at_p75 = threshold - p75_of_raw_confidence` metric is the correct threshold misalignment signal. If even the top-quartile signals fall >5 pts below threshold, no penalty fix will unblock trades — the threshold is structurally above the distribution. Using p75 rather than mean guards against right-tailed outliers inflating the appearance of good signals.
- [2026-03-28] **PATTERN/signal_existence_gate_before_adaptive_floor**: AdaptiveConfidenceFloor should NOT fire when GatePassPredictor reports signal_exists=True (penalty-free pass rate >= 5%). If signals exist penalty-free, Phase 57's penalty remediation is the correct lever — not threshold reduction. Adaptive threshold reduction is only warranted when even a penalty-free simulation produces 0% pass rate (raw signals genuinely below threshold).
- [2026-03-28] **PATTERN/conservative_adaptive_threshold_reduction**: Step=1.0pt, cooldown=20 scans, max=5 adaptations, floor=45.0 — each constraint has a distinct role. Step prevents oscillation. Cooldown prevents rapid descent. Max cap requires human review after 5 attempts (operational signal that something else is wrong). Floor (45.0) sits 5pts above the penalty ceiling floor (40.0) to ensure the two mechanisms never interact.
- [2026-03-28] **PATTERN/binary_search_over_distribution_for_threshold_suggestion**: Finding the threshold that yields a target pass rate (e.g. 10%) is naturally a binary search over the recorded score distribution. `find_threshold_for_target_pass_rate()` with 30 iterations gives 0.1pt precision. The result is immediately actionable: "lower threshold to X to achieve 10% gate pass rate with current ensemble output."

### Phase 57 Penalty Stack Remediation & Stall Recovery — 2026-03-28
- [2026-03-28] **PATTERN/calibration_guard_validity_check**: `ModelCalibrationTracker.get_calibration_report()` computes `overconfidence_ratio` from bins, but requires `MIN_SAMPLES_PER_BIN=5` per bin. With 6 total trades across 10 bins, `valid_bins=0` and yet the ratio can still return non-None from a partial calculation. The engine's check `if _oc_ratio is not None and _oc_ratio > 0.15` fires anyway. Always guard ML-derived penalty signals with a minimum data requirement (n_predictions + n_valid_bins) before applying penalties.
- [2026-03-28] **PATTERN/penalty_ceiling_coordinates_independent_systems**: When multiple penalty systems (overconfidence, drift, disagreement) operate independently without shared state, adding a coordinating ceiling module is the minimal-invasive fix. The ceiling does not modify any of the three systems — it intercepts each penalty at the point of application and trims it if confidence is already near the floor. This avoids changing three separate modules' logic while solving the stacking problem.
- [2026-03-28] **PATTERN/stall_recovery_state_machine_breaks_feedback**: A simple two-state machine (NORMAL → RECOVERY → NORMAL) with a `penalties_suspended()` flag is sufficient to implement the stall recovery intervention. The key insight: don't permanently disable penalties — suspend them for a window (5 scans) and observe whether tradeables appear. If yes, the penalties were the cause. If no, the problem is elsewhere (threshold too high, models degraded). The state machine is the right abstraction.
- [2026-03-28] **PATTERN/drift_proxy_circular_reference**: Concept drift detectors that use `confidence` as a proxy input create circular suppression: penalty reduces confidence → drift proxy sees "disagreement" → drift fires → further reduces confidence → next scan has worse proxies. Always capture the raw/pre-penalty signal for diagnostic and feedback inputs. The fix is not to remove drift detection but to feed it the pre-penalty signal rather than the post-penalty output.
- [2026-03-28] **PATTERN/audit_recommendation_engine_for_stalls**: When a system enters a stall state (0 tradeable pairs for N scans), generating an actionable recommendation (CEILING_NEEDED / CALIBRATION_GUARD_NEEDED / THRESHOLD_RELAXATION_NEEDED) is more valuable than just logging that it's stalled. The recommendation discriminates between causes: stacked penalties vs. invalid calibration data vs. threshold miscalibration. Each requires a different fix.

### Phase 56 Signal Quality Restoration & Execution Feedback Loop — 2026-03-27
- [2026-03-27] **PATTERN/stacked_penalty_self_suppression**: Three simultaneous confidence penalty systems (overconfidence_detector -3pt, concept_drift multiplier, ensemble_disagreement) stacked to push ALL 39 pair signals below the execution threshold across 63+ scan cycles. No single system was at fault — the interaction between systems caused total suppression. Lesson: when a system produces 0 tradeable pairs across multiple scan cycles, the first diagnostic should be to audit ALL active penalty/multiplier systems for compound interaction effects.
- [2026-03-27] **PATTERN/write_only_dead_module_detection**: ExecutionQualityTracker.track_fill() was initialized in engine.py with a config flag, but never called anywhere in the codebase. Unit tests passed because they test the method itself, not whether it's called in production. Always grep for method_name OUTSIDE the class definition to confirm at least one live call site exists. Test-passing != production-wired.
- [2026-03-27] **PATTERN/jsonl_over_json_for_high_frequency_writes**: Virtual trade logger uses append-only JSONL instead of full JSON rewrites. Each scan cycle that detects a promising-but-rejected setup appends one line. With JSON, the full file would be parsed, modified in memory, then rewritten atomically — expensive and data-loss-prone under concurrent writes. JSONL is O(1) append with no read-modify-write cycle.
- [2026-03-27] **PATTERN/scan_diagnostics_stalled_flag**: A `stalled` flag that fires after 50+ scans with no tradeable pair provides an early-warning signal before the Ralph PRD generator's 30-day analysis window closes. The flag is consumed by the continuous loop to emit a WARNING-level log that propagates to alerting. Threshold should be tunable (not hardcoded) to allow adjustment if false positives occur.
- [2026-03-27] **PATTERN/cross_module_consistency_tests**: Integration tests that compare two independent modules' outputs on the same input data (e.g. gate_pass_rate_pct from ScanDiagnosticsReporter vs rejection counts from GateRejectionTracker) catch semantic drift when modules evolve separately. Unit tests can't catch this — only integration tests that run both modules on identical synthetic data.

### Phase 54 Adaptive Gate Tuning, Feature Drift Detection & Trade Clustering — 2026-03-25
- [2026-03-25] **PATTERN/gate_threshold_optimization**: Gate attribution reports reveal which gates reject trades that would have won. When good_rejection_rate < 0.30, the gate is killing good trades — relax by +10%. When > 0.70, the gate is protecting profit — tighten by -5%. The asymmetric adjustment (larger relaxation, smaller tightening) prevents over-tightening. Always require 50+ rejection evidence to avoid small-sample bias. Cap at ±15% per cycle.
- [2026-03-25] **PATTERN/page_hinkley_change_detection**: Standard Page-Hinkley for detecting mean decrease: track cumulative sum of (value - mean + lambda/n), record max_sum, signal when max_sum - current_sum > threshold. The lambda parameter acts as tolerance — too large masks real signals, too small causes false alarms. lambda=500 with 50-trade window and threshold=3.0 balances sensitivity for FX win-rate monitoring.
- [2026-03-25] **PATTERN/dbscan_cluster_sensitivity**: DBSCAN's eps parameter critically affects cluster formation. Synthetic test data with tight Gaussian clusters needs larger eps (2.0) while real 522-trade production data across diverse conditions works with default 0.5. Always validate cluster count > 0 after fitting and provide a simple_cluster fallback when sklearn unavailable.
- [2026-03-25] **PATTERN/regime_probability_heuristic**: Without a dedicated regime classifier model, regime probabilities can be estimated from existing indicators: ADX → trending probability (0.6×adx_norm + 0.4×trend_strength), trend_strength → bull/bear split, ATR percentile → uncertainty (reduces max probability by 0-20%). Produces reasonable 3-class distributions for sizing decisions.
- [2026-03-25] **PATTERN/cluster_centroid_matching**: Nearest-centroid matching for cluster classification at execution time is O(n_clusters) — negligible latency. Euclidean distance on standardized 6-feature space. Capping adjustment to ±0.10 confidence and bounding result to [0,1] prevents cluster artifacts from producing extreme position sizes.
- [2026-03-25] **PATTERN/transition_detection_hysteresis**: Regime transition detection via max_prob drop > 15% with 2-cycle holdoff prevents whipsaw sizing during regime shifts. The holdoff (transition_cycles=2) acts as hysteresis — once triggered, sizing stays reduced even if indicators temporarily recover. This is analogous to drawdown hysteresis from Phase 52.

### Phase 53 Gate Diagnostics, Pair-Specific Tuning & Model Stability — 2026-03-25
- [2026-03-25] **PATTERN/confidence_decile_attribution**: Gate rejection quality can be measured by bucketing rejections into confidence deciles, then matching against trade journal outcomes in the same decile. If decile win_rate < 0.50, the rejection was "good" (would have lost). 10 deciles × ~52 trades per decile provides statistically meaningful buckets from 522 trades.
- [2026-03-25] **PATTERN/pair_regime_weight_blending**: Pure pair-specific weights overfit with < 50 trades per pair. Blending 0.6×regime + 0.4×pair gives pair-level adaptation while keeping regime intelligence as anchor. Warm-start threshold (5 trades) prevents noise from tiny samples. Thompson Sampling naturally handles exploration via posterior width.
- [2026-03-25] **PATTERN/signal_freshness_gate_ordering**: Signal freshness guard placed AFTER drawdown (cheapest check) but BEFORE spread/calendar (more expensive API calls). The freshness check only requires numpy computation on small price arrays — negligible cost. Gate ordering principle: compute cost ascending, rejection breadth descending.
- [2026-03-25] **PATTERN/double_transform_coherence_loss**: Applying isotonic calibration THEN confidence decomposition (with recomposition) THEN reapplying produces incoherent signals — each transform optimizes for raw input, not transformed input. Config flag (`use_decomposed_confidence`) forces single path selection, eliminating the chain.
- [2026-03-25] **PATTERN/wfe_stability_state_machine**: Simple threshold triggers (WFE < X → pause) cause oscillation when WFE fluctuates near boundary. Three-state machine (STABLE→UNSTABLE→RECOVERING→STABLE) with different transition criteria for each direction prevents flip-flopping. Recovery requires sustained improvement (2+ positive cycles), not just a single good reading.
- [2026-03-25] **PATTERN/frozen_weight_fallback**: When retraining is paused (UNSTABLE state), the system needs pre-paused weights to use. Freezing weights at every successful retrain cycle ensures there's always a "last known good" snapshot. Without this, pausing retraining means running with whatever weights happened to be loaded — potentially stale from a much earlier cycle.

### Phase 52 Pipeline Integration & Adaptive Intelligence — 2026-03-24
- [2026-03-24] **PATTERN/config_override_restore**: When temporarily overriding dataclass config fields (e.g., atr_tp_multiplier) in a single-threaded execution path, always store the old value in ctx and restore it after the calculation completes. This prevents config drift across subsequent calls.
- [2026-03-24] **PATTERN/drawdown_hysteresis_prevents_whipsaw**: Hard threshold drawdown gates cause rapid on/off toggling as NAV oscillates near the threshold. Recovery margin (2% below entry) creates a hysteresis band that prevents whipsaw. Always use hysteresis for state transitions based on continuous metrics.
- [2026-03-24] **PATTERN/earliest_gate_placement**: Drawdown adapter must run BEFORE all other gates because it can halt all trades. If placed after spread/calendar/fitness gates, those gates waste compute on trades that will be blocked anyway. Gate ordering: cheapest/broadest blockers first, expensive/narrow filters last.
- [2026-03-24] **PATTERN/confidence_decomposition_addresses_inverse**: Raw confidence 0.71-0.80 had 16.1% win rate (inverse relationship). Decomposing into directional_strength × timing_quality × magnitude_estimate with recomposition naturally penalizes high-raw-confidence with weak directional agreement. Symptom-level fixes (capping) are fragile; structural decomposition fixes the root cause.
- [2026-03-24] **PATTERN/regime_rr_targeting**: Static 1.5x atr_tp_multiplier ignores regime context. HIGH/EXTREME regimes need wider TP (2.2-2.5x) to capture larger moves, while LOW regimes can tighten (1.5x). Confidence and pair form further modulate. Floor of 1.2:1 R:R is non-negotiable per trading rules.

### Phase 51 Win-Rate Engine — 2026-03-24
- [2026-03-24] **PATTERN/isotonic_over_platt**: Platt scaling failed (cal_error=0.36, near-identity a=0.85) because confidence→win relationship isn't sigmoid-shaped. Isotonic regression (PAVA) learns any monotonic mapping without parametric assumptions. When Platt fails, always try isotonic next — not more complex methods.
- [2026-03-24] **PATTERN/pava_binning_prevents_overfit**: PAVA on raw 522 trades would overfit to individual trade noise. Binning first (15 bins × ~35 trades) then running PAVA on bin averages produces stable calibration curves. Always bin before isotonic when N < 1000.
- [2026-03-24] **PATTERN/breakeven_sl_after_partial**: Moving SL to breakeven after first partial profit tranche eliminates downside on remaining position. Direction matters: LONG SL moves up to entry, SHORT SL moves down to entry. Always compute SL adjustment relative to direction.
- [2026-03-24] **PATTERN/walkforward_wfe_threshold**: Walk-Forward Efficiency (OOS_win_rate / IS_win_rate) < 0.50 indicates overfitting. Combined with weight stability check (max 30% drift between epochs), prevents deploying overfit agent weights. Two-check gate catches different failure modes.
- [2026-03-24] **PATTERN/composite_quality_scores**: Single-dimension thresholds miss multi-dimensional weakness. A trade with moderate disagreement AND moderate uncertainty AND poor recent form is much worse than any single dimension alone. Weighted composite score catches these cumulative weakness patterns.

### Phase 49 Pipeline Wiring — 2026-03-24
- [2026-03-24] **PATTERN/verdict_tuple_ordering**: TradeAttributionEngine.attribute_trade() expects agent_verdicts as `{name: (verdict, confidence, weight)}` — NOT `(verdict, weight, confidence)`. Tuple field ordering matters critically when consuming third-party dataclasses. Always check source code for field order before wiring.
- [2026-03-24] **PATTERN/verdict_direction_format**: Attribution alignment logic maps BUY→LONG, SELL→SHORT internally. Agent verdicts passed to attribute_trade() must use LONG/SHORT/NEUTRAL/SKIP (not BUY/SELL). Direction format mismatch silently breaks alignment scoring.
- [2026-03-24] **PATTERN/compound_multiplicative_gates**: Pre-execution gates (spread × calendar × fitness) compound multiplicatively. 0.8 × 0.5 × 0.5 = 0.2 — a 5× reduction. Monitor in production for over-reduction risk. May need a floor multiplier (e.g., 0.1) to prevent near-zero sizing.
- [2026-03-24] **PATTERN/spread_api_uses_passed_not_action**: SpreadCheckResult uses `passed` boolean + `size_multiplier` property, NOT an `action` string like CalendarCheckResult/FitnessCheckResult. Each Phase 48 module has its own result API — never assume consistency across modules.
- [2026-03-24] **PATTERN/attribution_field_agents_not_contributions**: TradeAttribution result uses `.agents` (List[AgentContribution]), not `.contributions`. Field naming across Phase 48 modules is inconsistent — always verify actual field names from dataclass definitions.

### Phantom Wiring Audit — 2026-03-24

### Bug Exterminator Run — 2026-03-24
- [2026-03-24] **CRITICAL/gate_bypass_three_paths**: THREE separate code paths bypassed the three-gate requirement (ThresholdOptimizer, EXTREME regime, agent promotion). All three fixed in same run. Pattern: any re-qualification or promotion path must explicitly check `momentum_passed AND confidence_passed AND risk_passed` — partial checks are never safe. Promoted candidate if this appears again.
- [2026-03-24] **CRITICAL/hardcoded_sl_vs_atr_tp**: SL was hardcoded to `max_sl_pips=15.0` while TP correctly used ATR scaling. The `atr_sl_multiplier` config field existed but was never consumed. Always audit both SL AND TP in any position sizing function for consistency.
- [2026-03-24] **CRITICAL/rr_gate_fail_open**: `if sl_pips > 0 and tp_pips > 0: rr_ratio = ...` is a fail-open pattern — if either is zero, the gate is silently skipped. Financial safety checks must be fail-closed: reject if preconditions not met.
- [2026-03-24] **SYSTEMIC/partial_zero_guards**: `if len(x) < 3` guards don't prevent zero-division if code changes; always add explicit `if total == 0` guards directly before division operations in financial paths.
- [2026-03-24] **SYSTEMIC/legacy_type_mismatch**: journal `outcome` field can be legacy string `"win"` or modern dict. Always use `isinstance(e.get("outcome"), dict)` before calling `.get()` on it. Audit all places that read outcome.
- [2026-03-24] **SYSTEMIC/non_atomic_writes_persist**: Despite JSON Safety Gates rule (2026-03-23), 6+ non-atomic `write_text()` calls persisted in state_engine, adaptive_scaler, fx_guardrails, execution.py. Rules alone don't enforce — code review should specifically grep for `write_text(json.dumps` patterns.
- [2026-03-24] **INFO/c5_secondary_bug**: `sum(len(v) for v in dict.items())` iterates tuples `(key, value)` making `len()` always 2 per item, not the length of the value list. Must use `.values()` when summing lengths of dict values.
- [2026-03-24] **INFO/fuse_git_locks**: On FUSE-mounted macOS APFS volumes, stale git lock files (.git/index.lock, HEAD.lock) cannot be deleted via `rm`, `os.unlink()`, or Python even as owner. Workaround: copy `.git` to `/tmp`, delete locks there, commit with `GIT_DIR=/tmp/copy`, sync objects + HEAD ref back.




### Phase 28: Event-Driven PRD Agent Chain
- [2026-03-23] **feature/event_bus**: US-169. EventBus pub/sub system — thread-safe, ordered handlers, JSONL event log persistence, singleton pattern.
- [2026-03-23] **feature/prd_watcher**: US-170. PRDWatcher — inotify/watchdog + polling fallback, MD5 dedup, state persistence across restarts.
- [2026-03-23] **feature/gap_wirer**: US-171. GapWirer agent — AST-based analysis: module_registration, config_flag, integration, state_persistence, feedback_loop gap detection across 82 modules.
- [2026-03-23] **feature/code_reviewer**: US-172. CodeReviewer agent — 7 rule categories from .claude/rules/: JSON safety, error handling, import hygiene, state management, trading domain, integration patterns.
- [2026-03-23] **feature/prd_agent_chain**: US-173. PRDAgentChain — chains watcher→wirer→reviewer via EventBus, persists reports to .claude/ralph/reports/, updates state.json, CLI + daemon mode.
- [2026-03-23] **arch/phase28_complete**: Phase 28 (US-169 through US-173) 5 stories complete. 83 total automation modules (+5: event_bus, prd_watcher, gap_wirer, code_reviewer, prd_agent_chain). Theme: event-driven PRD agent chain — automated detection of PRD completion triggers gap analysis and code review chain.

### Phase 29: Gap Resolution & Automation
- [2026-03-23] **critical/enable_execution_fixed**: US-174. CRITICAL FIX — `enable_execution: bool = False` was master switch blocking ALL trade execution. Smart profile never overrode it. Added enable_execution=True + enabled session_timing, support_resistance, execution_quality_optimizer agents.
- [2026-03-23] **feature/graceful_shutdown_expanded**: US-175. Graceful shutdown expanded from 5 to 16 modules. Added health_registry, agent_accuracy_matrix, pair_regime_agent_matrix, signal_timing, model_calibration, dynamic_risk_allocator, execution_quality_optimizer, drift_monitor, module_dispatcher, replay_validator, module_activation.
- [2026-03-23] **feature/dispatcher_full_coverage**: US-176. ModuleDispatcher extended with 20 new module frequencies and register_all_modules() mappings. All Phase 17-28 modules now cycle-scheduled.
- [2026-03-23] **feature/observation_consumer_orchestrator**: US-177. ObservationConsumer wired into orchestrator run_cycle as Step 8d. Runs every 5 cycles: consume_observations → detect_patterns → generate recommendations. Populates observation_patterns and observation_recommendations in OrchestrationResult. Spike alerts logged at WARNING.
- [2026-03-23] **feature/execution_diagnostic_logging**: US-178. Replaced 14 silent `except Exception: pass` blocks in execution.py with contextual `logger.debug()` messages. Added pre-flight validation to execute_trades: NAV check, max_concurrent_trades gate (default 10), trade parameter validation (price, ATR, SL, TP, R:R ratio). Added configurable confidence blend weights (0.55/0.30/0.15 + boost 0.08/0.06).
- [2026-03-23] **feature/global_feature_activation**: US-179. Enabled 3 features globally (all profiles): enable_confidence_calibration, enable_dynamic_sl_tp, enable_concept_drift_detection. Previously only active in smart profile.
- [2026-03-23] **arch/phase29_complete**: Phase 29 (US-174 through US-179) all 6 stories complete. 86 total automation modules (+3: preflight_validation, configurable_confidence_blend, max_concurrent_trades_gate). Theme: gap resolution — activated dormant features, expanded persistence/dispatch coverage, hardened execution path, replaced silent failures with observability.

### Phase 30: System Optimization
- [2026-03-23] **perf/batch_oanda_pricing**: US-180. _batch_fetch_prices() eliminates N+1 OANDA API calls in drawdown guardian and trailing SL. Single multi-instrument request with per-trade fallback.
- [2026-03-23] **reliability/oanda_retry_wrappers**: US-181. 7 direct OANDA API calls wrapped in _retry_oanda with exponential backoff — monitor_open_trades, drawdown SL modify, close_trade, modify_sl, trailing SL modify, sync closed trades.
- [2026-03-23] **ops/observation_log_rotation**: US-182. _maybe_rotate() in ObservationLog checks file size before each write. Rotates to dated .jsonl at 100KB, prunes to 7 max rotated files.
- [2026-03-23] **quality/observation_dedup**: US-183. Replaced 19 inline `from ... import ObservationLog; ObservationLog()` blocks in execution.py with shared self._observer lazy singleton via _get_observer() helper.
- [2026-03-23] **feature/calibration_ece_consumption**: US-184. Engine.py now reads ModelCalibrationTracker reports for TCN/Ridge after confidence blend. Applies -3pt confidence penalty when overconfidence_ratio > 0.15. Resets when all models < 0.10.
- [2026-03-23] **feature/aggressive_profile_parity**: US-185. Aggressive profile brought to feature parity: use_rl_sizer, use_rl_gates, soft_uncertainty_blocking, enable_agent_trade_promotion all True. Also added enable_execution=True.
- [2026-03-23] **arch/phase30_complete**: Phase 30 (US-180 through US-185) all 6 stories complete. Theme: system optimization — performance (batch pricing), reliability (retry wrappers), code quality (dedup, log rotation), intelligence (ECE consumption), profile parity.

### Phase 31: Code Review Fixes & Hardening
- [2026-03-23] **bugfix/cycle_count_nameerror**: US-186. RUNTIME BUG — `cycle_count` used at 2 locations in `_run_smart_loop` but never defined. Replaced with `self._scan_count`. Agent degradation and observation consumer paths were silently crashing.
- [2026-03-23] **bugfix/agent_team_config_type**: US-187. ScannerAgentTeam(self.config) passed ContinuousConfig instead of ScannerConfig in session snapshot save. Fixed to self.scanner.config.
- [2026-03-23] **bugfix/json_parse_guard**: US-188. `_update_pair_performance` read journal JSON without try/except. Corrupted file would crash RL sync. Wrapped with JSONDecodeError/OSError guard.
- [2026-03-23] **quality/silent_except_replaced**: US-189. 6 silent `except Exception: pass` blocks in continuous.py replaced with contextual `logger.debug()` messages — pair accuracy, degradation obs, alert logging, A/B weights (2x), shadow rotation.
- [2026-03-23] **bugfix/path_import_retrain**: US-190. `_spawn_background_retrain` used `Path()` without local import. Added explicit `from pathlib import Path`.
- [2026-03-23] **perf/cached_nav_execute_trade**: US-191. `execute_trade` called `fetch_live_nav()` 4 times per invocation. Cached as `_cached_nav` — single API call, reused at 3 position sizing locations.
- [2026-03-23] **arch/phase31_complete**: Phase 31 (US-186 through US-191) all 6 stories complete. Theme: code review fixes — 3 runtime bugs (NameError, wrong type, unguarded JSON), 6 silent except blocks, 1 missing import, 1 performance optimization (4x→1x NAV fetch).

## Aura Learnings — MOVED to P-90
> Aura (Eve) is a separate system. Her learnings now live in `P-90/.claude/learnings.md`.
> Phase 1 and Phase 2 findings migrated on 2026-03-23.

### Phase 32: Code Review Suggestions (2026-03-23)
- [2026-03-23] **robustness/apply_profile_validation**: US-192. apply_profile now validates keys against __dataclass_fields__ before setattr — typos no longer silently create rogue attributes. Unknown keys logged as warnings and skipped.
- [2026-03-23] **robustness/journal_file_locking**: US-193. _append_journal_entry fallback path now uses fcntl.LOCK_EX when safe_json_write unavailable — prevents journal corruption from concurrent processes. Graceful fallback if fcntl unavailable.
- [2026-03-23] **robustness/smart_loop_error_visibility**: US-194. Outer _run_smart_loop exception escalated from logger.debug to logger.warning with exc_info=True — trading loop errors no longer silently swallowed.
- [2026-03-23] **perf/learning_loop_optimization**: US-195. _run_learning_loop journal read now guarded with try/except and only processes last trades_synced entries instead of full journal — prevents crashes on corrupted files and reduces memory usage.
- [2026-03-23] **non-issue/console_guard**: US-196. Investigation confirmed all console.print calls in _run_smart_loop already inside `if console:` guards — no change needed.
- [2026-03-23] **meta/review_fixes_resolved**: US-197. All 7 stories in prd_review_fixes.json resolved — mapped to fixes delivered in Phases 30-32.
- [2026-03-23] **arch/phase32_complete**: Phase 32 (US-192 through US-197) all 6 stories complete. 5 implemented, 1 non-issue. Code review suggestions from Phase 28 CodeReviewer fully addressed.

### Phase 33: Gap Closure & Profile Parity (2026-03-23)
- [2026-03-23] **config/aggressive_parity**: US-212. Aggressive profile elevated from 3 to 56+ enable flags — full feature parity with smart profile while keeping aggressive's own gate thresholds (lower confidence, wider uncertainty). Also added ATR/HRP/execution strategy settings.
- [2026-03-23] **robustness/shutdown_expansion**: US-213. Graceful shutdown expanded from 16 to 28 modules. All modules with save_state() are now persisted on clean exit: affinity_portfolio, attention_feedback, causal_filter, exec_quality_tracker, execution_router, model_bandit, model_router, pair_transfer, position_manager, regime_broadcaster, temporal_attention, training_augmenter.
- [2026-03-23] **config/news_session_agents**: US-214. enable_news_risk_agent and enable_session_filter were False in ALL profiles despite being fully implemented. Now enabled in smart and aggressive.
- [2026-03-23] **config/dispatcher_gap**: US-215. Module dispatcher was missing 3 dispatchable modules: macro_stress(freq=20), microstructure_regime(freq=5), regime_tracker(freq=3).
- [2026-03-23] **config/conservative_modernization**: US-216. Conservative profile modernized with 11 infrastructure flags — enables monitoring, calibration, and drift detection without enabling execution or trade-boosting features.
- [2026-03-23] **testing/config_coverage**: US-217. 21 unit tests for ScannerConfig: profile application, unknown key validation, feature parity assertions, volatility clamping, profile switching. All pass in 0.1s.
- [2026-03-23] **arch/phase33_complete**: Phase 33 (US-212 through US-217) all 6 stories complete. Config flag gap fully closed — aggressive has smart parity, conservative has infrastructure, all profiles have news_risk and session_filter.

### Finding: Gap resolution PRDs become stale quickly
- Ralph's gap-resolution-phase11 PRD (30 stories, 0 passed) was generated from Phase 10 analysis
- By Phase 33, most gaps had already been fixed in Phases 19-32
- Fresh analysis found only real remaining gaps: profile parity, shutdown coverage, dispatcher holes
- **Lesson**: Run fresh gap analysis rather than working through stale PRDs

### Phase 34: Critical Path Testing & Hardening (2026-03-23)
- [2026-03-23] **testing/execution_gates**: US-218. 9 unit tests for can_trade() daily limit enforcement, _check_portfolio_risk_limit(), _check_correlation_exposure(). Mock-based to avoid OANDA calls.
- [2026-03-23] **testing/atr_sl_tp**: US-219. 7 unit tests for _calculate_base_tp_pips() — normal ATR calc, min/max clamping, high-probability bonus, zero ATR and zero pip_value fallbacks.
- [2026-03-23] **testing/exit_classification**: US-220. 9 unit tests for _determine_exit_reason() — tp_hit, sl_hit, breakeven_stop, trailing_stop (long/short), timeout, manual_close, missing prices. Critical for RL feedback accuracy.
- [2026-03-23] **testing/journal_integrity**: US-221. 4 unit tests for journal file handling — required RL fields, corrupt file recovery, empty file recovery, append preservation.
- [2026-03-23] **robustness/sizing_zero_guard**: US-222. Added explicit zero-division guards in calculate_position_size() — sl_pips and tp_pips clamped to max(min_value, 1.0) after risk manager adjustment, with warning log on trigger.
- [2026-03-23] **testing/position_sizing**: US-223. 5 unit tests for position sizing — fallback lots range, zero ATR handling, zero equity, regime-aware sizing, PIP_VALUES positivity assertion.
- [2026-03-23] **arch/phase34_complete**: Phase 34 (US-218 through US-223) all 6 stories complete. 34 new tests for execution.py (was 0). Combined with Phase 33's 21 config tests = 55 new tests total.

### Finding: Mock-based testing strategy for OANDA-dependent code
- ExecutionManager.__init__ requires OANDA credentials — mock it with __new__ + manual attribute setup
- Pure calculation methods (_calculate_base_tp_pips, _determine_exit_reason) can be tested directly
- Gate methods (can_trade, portfolio_risk) need MagicMock on fetch methods
- Position sizing has multiple fallback paths — test each independently
- **Lesson**: Separate pure calculation logic from API-dependent code for testability

### Phase 35: Knowledge Consolidation & Maintenance (2026-03-23)
- [2026-03-23] **maintenance/learnings_archive**: US-224. Archived 173 learnings entries (Phases 4-27) to learnings-archive.md. Active count reduced from 228 to 55, well within 30-entry threshold after filtering.
- [2026-03-23] **maintenance/rule_promotion**: US-225. Promoted 6 cross-cutting patterns to rules/improvement.md — json_safety(31 obs), retry_robustness(27), state_persistence(8), test_coverage(8), config_validation(6), silent_exception(4). Each rule section has 4-5 actionable directives.
- [2026-03-23] **maintenance/state_archive**: US-226. Archived 10 phase detail blocks (Phases 18-27 + Aura Phase 1) to state-archive.json. state.json reduced from 282 to ~160 lines.
- [2026-03-23] **maintenance/module_list_update**: US-227. Added 8 missing automation modules from Phases 29-32. Total now 90.
- [2026-03-23] **verification/config_flags**: US-228. All 93 profile keys validated against ScannerConfig dataclass. Zero orphan keys. 19 enable_ flags use defaults (core agents).
- [2026-03-23] **verification/test_regression**: US-229. Full test suite: 951 passed, 55/55 Phase 33-34 tests pass. 34 pre-existing failures documented (not regressions).
- [2026-03-23] **arch/phase35_complete**: Phase 35 (US-224 through US-229) all 6 stories complete. Theme: knowledge consolidation — archived stale data, promoted recurring patterns to actionable rules, verified system integrity. 90 automation modules, 951 tests passing.

### Phase 36: Test Coverage Expansion for Core Modules (2026-03-23)
- [2026-03-23] **test/agent_team_core**: US-230. 22 unit tests for ScannerAgentTeam: _clip01, _safe_float, AgentVerdict, weight load/save/validation/decay/regime/migration/corruption recovery. Mock-based with temp dirs + patched _WEIGHTS_FILE.
- [2026-03-23] **test/continuous_scanner**: US-231. 17 unit tests for ContinuousScanner: ContinuousConfig defaults, init/lifecycle, signal handler, scan cycle logging (chdir trick for local Path), pair rotation, correlation filter, direction extraction (predicted_direction not direction).
- [2026-03-23] **test/module_dispatcher**: US-232. 16 unit tests for ModuleDispatcher: DEFAULT_FREQUENCIES validation, module registration with frequency clamping, cycle scheduling (modulo-based), execute_cycle success/failure/tracking, status reporting, state persistence.
- [2026-03-23] **test/orchestrator_core**: US-233. 13 unit tests for Orchestrator: OrchestrationResult dataclass defaults/serialization/independent-errors, Orchestrator init flags, get_system_status structure (requires orch.scanner=None pre-set).
- [2026-03-23] **test/position_sizing_core**: US-234. 23 unit tests for DynamicPositionSizer: config defaults/bands, PositionSize dataclass, confidence band classification, JPY pip calculation, zero SL handling, FixedPositionSizer, factory functions.
- [2026-03-23] **test/scan_pipeline_smoke**: US-235. 10 integration smoke tests: PairAnalysis structure integrity, ScanResult collection, AgentTeam.evaluate() pipeline with mock DataFrames (LONG/error/HOLD paths).
- [2026-03-23] **arch/phase36_complete**: Phase 36 (US-230 through US-235) all 6 stories complete. 101 new unit tests across 6 test files. Total test count: 1052+. Theme: test coverage expansion for 5 largest untested source files. Key pattern: mock-based testing with temp dirs, patched class-level paths, MagicMock configs.
- **Lesson**: Always test with `predicted_direction` not `direction` for ContinuousScanner — the attribute name matters. Use `os.chdir(tmpdir)` when code uses local `Path()` calls that can't be easily patched. Pre-set `orch.scanner = None` before calling get_system_status() on default-constructed Orchestrator.

### Phase 37: Critical Path Hardening (2026-03-23)
- [2026-03-23] **test/trade_journal_core**: US-236. 28 unit tests for TradeJournal: TradeEntry dataclass roundtrip, JSON serialization, _load/_save (empty, create, roundtrip, corrupted recovery), log_trade (slippage calc EUR/JPY, persistence), get_statistics (empty, open, mixed, all-wins, exit reasons, by-instrument), get_recent_trades, clear_journal, format_journal_stats, LiveTradeInfo/AccountSummary/ClosedTradeInfo defaults.
- [2026-03-23] **test/trade_journal_analytics**: US-237. 25 unit tests for SlippageAnalyzer: _std_dev (empty, single, identical, known), _get_lot_bucket (micro through 10+), analyze_slippage (empty journal, single/multiple trades, by_lot_size, by_instrument, cost_impact, recommendations), _generate_recommendations (high/moderate/good slippage, split orders, best hour), get_kelly_adjusted_stats (empty, with trades, slippage reduces edge, impact message).
- [2026-03-23] **test/gate_evaluator_core**: US-238+239. 41 unit tests for GateEvaluator: init (default, non-joint, joint path, vol regime clamping), load_models (missing dir, no models, require_tcn), _prepare_features_input (valid, single row, None/empty, target features), _extract_momentum_score (2d multiclass/single, 1d, clamping, None/empty), evaluate_momentum (no model neutral, CatBoost mock), evaluate_confidence (ADX fallback), evaluate_risk (default, mock RF), evaluate_volatility_regime (no model, insufficient data), _compute_adx (valid, insufficient, missing cols, period 1), _predict_with_named_input_if_needed, gate constants.
- [2026-03-23] **test/scanner_config_validation**: US-240. 44 unit tests for ScannerConfig: __post_init__ clamping (15 threshold params), path resolution (string→Path, relative→absolute, absolute preserved), profile normalization (unknown→balanced, lowercase, stripped), get_pip_value (major, JPY, unknown, coverage), get_trading_session_status (weekday, Saturday, filter disabled), module constants (pair counts, PROJECT_ROOT), config defaults (profile, execution, top_n, meta_labeler), profile safety invariants (conservative no-execution, blocked_pairs, aggressive R:R, no negative confidence).
- [2026-03-23] **test/scan_to_gate_regression**: US-241. 12 regression tests for Scanner→GateEvaluator pipeline: Scanner init, _run_inference paths (technical RSI fallback LONG/SHORT, gate evaluator pass/fail), _scan_pair E2E (blocked pair, no data, full pipeline with mock gates, score range validation), PairAnalysis data integrity.
- [2026-03-23] **arch/phase37_complete**: Phase 37 (US-236 through US-241) all 6 stories complete. 150 new unit tests across 5 test files. Modules hardened: trade_journal.py (1329 LOC, was 0 tests), gates.py (1631 LOC, was minimal), config.py (1081 LOC, expanded), engine.py (pipeline regression). Total test count now ~1200.
- **Lesson**: ScannerConfig has no `dry_run` field — use `enable_execution=False` instead. Modular ensemble lazy-inits via `_init_modular_ensemble()` — must patch the init method, not just set `_modular_ensemble=None`. Profile-applied `blocked_pairs` override config-level empty lists — explicitly set `scanner.config.blocked_pairs=[]` in tests.

### Phase 39: Training & Feature Infrastructure Hardening
- [2026-03-23] **FeatureEngineering**: 150+ features generated. RSI uses Wilder EWM smoothing with clip [0.01, 99.99]. create_features needs apply_candle_smoothing=False for isolated testing. text_features/candle_smoothing must be mocked.
- [2026-03-23] **WalkForwardValidator**: split() yields (train, val, test) triples. Expanding mode grows train set; rolling mode slides fixed window. purged_kfold_split removes near-boundary samples to prevent look-ahead bias.
- [2026-03-23] **calculate_trading_metrics**: Pure numpy. Sharpe = mean/std * sqrt(252). max_drawdown always <= 0. Works with binary 0/1 predictions.
- [2026-03-23] **ReplayBuffer**: Uses reservoir sampling for capacity management. get_replay_samples returns None when empty. save/load uses .npz + .json metadata.
- [2026-03-23] **DriftDetector**: Needs >=3 history entries for declining trend detection. check_drift returns (bool, str) with recommendation. full_drift_check returns dict with action field.
- [2026-03-23] **TrainingLineage**: @dataclass with to_dict/from_dict serialization. generate_checkpoint_id uses timestamp + random hex. should_retrain checks time, drift, and trend.
- [2026-03-23] **Phase 39 complete**: 160 new tests across 3 files. Cumulative: Phase 36 (101) + 37 (150) + 38 (91) + 39 (160) = **502 tests added in 4 phases**.

### Phase 40: Scaling, Data Pipeline & Meta-Labeling Hardening
- [2026-03-23] **KellyCalculator**: Formula is (p*b - q) / b. Phase multipliers: VALIDATION=0.25, OPTIMIZATION=0.50, SCALING=0.75, ACCELERATION=1.0. Clamped to [min_kelly, max_kelly].
- [2026-03-23] **AggressiveRiskManager**: Tracks losing/winning streaks. Blocks trading after max_trades_per_day or daily_drawdown > threshold. Recovery requires recovery_win_streak consecutive wins.
- [2026-03-23] **augment_time_series**: Pure numpy jittering/scaling. Labels unchanged. augment_time_series_batch adds config wrapper.
- [2026-03-23] **ScalerState**: transform/inverse_transform round-trip. Zero scale produces NaN (RuntimeWarning).
- [2026-03-23] **MetaLabeler.generate_meta_labels**: Adds primary_prediction column to features. Binary labels based on whether primary was correct. Weights adjusted by confidence distance from 0.5.
- [2026-03-23] **Phase 40 complete**: 190 new tests. Cumulative: 36(101)+37(150)+38(91)+39(160)+40(190) = **692 tests in 5 phases**.

### Phase 41: Learning Engine, RL Gating & Retrainer Hardening
- [2026-03-23] **LearningEngine.analyze_trade**: Returns list of LearningEntry. Always produces pair_behavior entry. Exit accountability maps agent YES/NO verdicts to trade outcomes.
- [2026-03-23] **LearningEngine file ops**: append_to_learnings creates .claude/ dir if missing. check_promotions uses regex to count category:pattern_key occurrences, promotes at 3+.
- [2026-03-23] **GateRLConfig**: Pure dataclass, no gym needed. GateThresholdEnv requires gymnasium for reset/step but math methods (_scale_action, _check_gates, _simulate_trade, _calculate_reward) are pure numpy.
- [2026-03-23] **MetaLabelerRetrainer**: simulate_trades_from_model generates synthetic TradeSamples. prepare_training_data extracts 4 features (tcn_prob, ridge_conf, xgb_mom, rf_dd) + optional primary_proba. Ridge and RF train without xgboost.
- [2026-03-23] **Phase 41 complete**: 103 new tests (81 passing, 22 gym-dependent skipped). Cumulative: 36-41 = **795 tests in 6 phases**.

### Phase 42: Dashboard, Enterprise Training & Metal Optimizer Hardening
- [2026-03-23] **Dashboard _apply_overrides_to_config**: Safety-critical type coercion for trading configs. Coerces bool/int/float/str/list/dict. Skips `_`-prefixed keys. Catches ValueError/TypeError silently. Used by both _make_scanner_config and _make_execution_config.
- [2026-03-23] **Dashboard _make_execution_config**: Maps `daily_trade_limit` → `max_trades_per_day` (field name mismatch between ScannerConfig and ExecutionConfig). Type coercion uses destination field's type as constructor. Silent error handling (no logging on coercion failure).
- [2026-03-23] **Enterprise WalkForwardValidator.split()**: Uses np.linspace for test_starts, ensures min_train_samples per fold, supports expanding vs sliding window. Skips folds where training data is insufficient. Critical for preventing look-ahead bias.
- [2026-03-23] **StatisticalValidator.sharpe_ratio_significance**: Uses Lo (2002) standard error formula: `se = sqrt((1 + 0.5*sharpe²) / n)`. Z-test against benchmark_sharpe. Annualized with sqrt(periods_per_year).
- [2026-03-23] **M1 Metal LR schedules**: WarmupCosineDecay uses linear warmup then cosine decay. CosineDecayRestarts supports equal (t_mul=1) and geometric (t_mul>1) cycle lengths. Both use tf.where for phase switching. Pure math is testable without TF.
- [2026-03-23] **SWA online mean formula**: `swa_weights[i] = (swa_weights[i] * (count-1) + current[i]) / count`. Equivalent to numpy.mean over all snapshots. Applied on_train_end via model.set_weights().
- [2026-03-23] **Pure-Python test extraction pattern**: When source modules have hard TF imports, write pure-Python math tests in the same file (no skipif marker) to validate formulas without TF. 26/87 tests pass without TF this way.
- [2026-03-23] **Phase 42 complete**: 154 passing + 61 skipped (TF). Cumulative: 36-42 = **949 passing tests in 7 phases**.

### Phase 43: Research-Backed Capability Upgrades (Regime Detection, Adaptive Exits, Confidence Calibration)
- [2026-03-23] **Web research methodology**: Parallel scattered agents (6 axes) produced ~250KB of structured findings. Key: always launch 5+ research agents simultaneously — reduces total wait from serial to wall-clock of slowest agent (~2 min vs ~10 min).
- [2026-03-23] **BOCPD (Adams & MacKay 2007)**: Bayesian Online Changepoint Detection tracks run-length distribution P(r_t | data). Constant hazard h=0.01 means expected regime duration of 100 bars. Changepoint when P(r=0) > 0.3. O(H) per observation where H=max_run_length=200. Perfectly causal — no look-ahead bias.
- [2026-03-23] **Hurst exponent via R/S analysis**: Divide price series into chunks of varying size N, compute rescaled range R/S per chunk, regress log(R/S) vs log(N) — slope is H. H>0.55 = trending, H<0.45 = mean-reverting, 0.45-0.55 = uncertain. Window of 100 bars works well for 4H FX data.
- [2026-03-23] **Chandelier Exit formula**: CE_Long = max(high[-22:]) - ATR * 3.0. Hangs trailing stop from highest high. Widens in high vol, tightens in low vol. Multiplier 3.0 is default; 2.5 for tight, 4.0 for high-vol pairs (GBP/JPY). Drop-in improvement over static SL.
- [2026-03-23] **Partial profit scale-out pattern**: Close 50% at 1R, trail 25% with Chandelier, hold 25% for 2R or time exit. Reduces max profit slightly but significantly lowers return volatility (30-40% drawdown reduction per research). More complex position tracking needed.
- [2026-03-23] **Ensemble disagreement as uncertainty signal**: std(agent_scores) across 12 agents. Classification: <0.15=LOW, <0.30=MODERATE, <0.45=HIGH, >=0.45=CRITICAL. CRITICAL disagreement → 0.6x confidence penalty. FREE signal — already have 12 agents, no extra compute.
- [2026-03-23] **Regime-aware Platt scaling**: Separate sigmoid calibration per regime (LOW/NORMAL/HIGH/EXTREME). P(win|score) = 1/(1+exp(-(coef*score+intercept))). Need 15+ trades per regime before activating regime-specific curve; fall back to global calibration otherwise.
- [2026-03-23] **Confidence time-decay rates**: Regime-dependent exponential decay: LOW=0.99/bar, NORMAL=0.97, HIGH=0.95, EXTREME=0.92. After 20 bars in EXTREME regime, confidence decays to 0.92^20 = 0.19. Older positions carry more uncertainty — this quantifies it.
- [2026-03-23] **Priority-based exit merging**: confidence_decay > time_exit > chandelier > vol_trail > partial_profit. Critical-urgency exits override everything. This prevents conflicting signals — always take the most protective action.
- [2026-03-23] **Phase 43 complete**: 3 NEW PRODUCTION MODULES (~91KB source), 175 new tests (all passing). First phase with actual capability upgrades (not just test coverage). Cumulative: 36-43 = **1,124 passing tests in 8 phases**. Buddy now has: multi-factor regime detection, 5-strategy adaptive exits, and 5-layer confidence calibration.

## Phase 44 — Adaptive Position Sizing, Bayesian Agent Weights & Pipeline Wiring (2026-03-23)

### Sizing Insights
1. **Fractional Kelly is essential for FX**: Full Kelly is too aggressive; f=0.33 (third-Kelly) balances growth vs drawdown. Rolling 50-trade window prevents stale estimates from early losing streaks.
2. **Confidence sigmoid mapping prevents extreme positions**: σ(x,k=4) maps [0,1]→[~0.02,~0.98] — confidence 0.3 gives 0.27x multiplier, 0.7 gives 0.73x. Steepness k=4 gives good discrimination without cliff effects.
3. **Drawdown-recovery is critical for survival**: During 10% drawdown (of 15% max), position size drops to 63% of normal. Floor at 30% ensures we don't freeze out during recovery.
4. **Regime multipliers stack multiplicatively with other factors**: LOW=1.3× allows bigger positions in calm markets, EXTREME=0.4× reduces to 40% in crisis. Combined with drawdown factor, crisis sizing can be as low as 12% of normal.

### Bayesian Agent Weights Insights
5. **Thompson Sampling naturally balances exploration/exploitation**: No need for complex UCB calculations. Beta(α,β) with online updates and ε-greedy provides robust exploration.
6. **Regime-conditional Beta distributions are key**: Agent "trend" may be α=10,β=5 in NORMAL regime but α=3,β=8 in LOW regime — TS naturally adapts weights per regime without manual tuning.
7. **Weight decay (0.995 every 20 updates) prevents stale beliefs**: Without decay, agents that performed well 200 trades ago dominate despite market regime shifts. Decay forgets at ~50% per 139 updates.
8. **Partial credit via score-proportional updates outperforms binary**: An agent scoring 0.8 on a winning trade gets α+=0.8, not α+=1. This captures calibration quality, not just direction.

### Pipeline Wiring Insights
9. **Lazy imports with try/except are the right pattern for optional modules**: All three wiring points (engine.py, execution.py, _team.py) use lazy init — system works identically without Phase 43/44 modules.
10. **Gate_details dict is the right vehicle for regime metadata**: Adding `regime_detector` sub-dict doesn't break any downstream consumers; they just check `if "regime_detector" in gate_details`.
11. **Calibration overlay pattern preserves backward compatibility**: Compute raw weighted_vote_score first, then optionally override with calibrated score. If calibrator fails, raw score flows through unchanged.
12. **Per-trade exception handling in exit evaluation prevents cascading failures**: One bad trade context (e.g., zero entry price) doesn't block exit evaluation for other trades.

### Cumulative Stats
- Phase 36-44: **1,508 passing tests** across 9 phases
- Phase 43-44: 5 new production modules (~3,000 lines source) + 3 pipeline integrations
- Key upgrade: 38% win rate should improve as regime detection + calibrated confidence + adaptive exits filter bad entries and improve exit timing

---

## Phase 45 Learnings (2026-03-23) — Production Wiring, EWMA Correlation & Streak Sizing

### Production Wiring Insights
1. **Unwired modules are dead code**: Phase 44 created `adaptive_position_sizing.py` (627 lines) and `bayesian_agent_weights.py` (554 lines) but they were only called by tests. Production pipeline still used `DynamicPositionSizer` and flat-file weights. Gap analysis by dedicated research agents caught this immediately.
2. **Fallback-first wiring pattern is essential**: Every new module wired as "try adaptive → fall back to legacy" — this means zero risk of production regression. The `calculate_position_size()` method tries `_calculate_adaptive_position_size()` first, falls through to `DynamicPositionSizer` on any failure.
3. **Blend ratios smooth transitions**: Thompson Sampling weights blended 70/30 with flat weights (not 100% Bayesian). This prevents sudden weight shifts while the Beta distributions warm up. Same principle applies to position sizing — adaptive sizer is preferred but fallback is seamless.

### EWMA Correlation Insights
4. **λ=0.94 is the RiskMetrics standard for FX daily correlation**: Higher λ (0.97) is too slow for FX regime changes; lower λ (0.90) overreacts to noise. 0.94 gives ~17-day effective window.
5. **Diversification multiplier D = 1/√(1 + avg_corr) is elegant**: When avg correlation to portfolio is 0, D=1.0 (full size). When avg corr is 0.7, D≈0.77. When corr is 1.0, D≈0.71. Natural concave decay.
6. **RISK_OFF correlation regime (all pairs moving together) is the most dangerous**: Average off-diagonal correlation > 0.6 signals contagion risk — reducing portfolio risk limit from 15% to ~10% is a critical safety valve.
7. **Union-Find stays as hard block, EWMA as soft adjustment**: The static correlation filter prevents double exposure. EWMA adjusts position size for partially correlated pairs. Two layers, different purposes.

### Streak-Aware Sizing Insights
8. **Anti-martingale with caps prevents Kelly's ruin**: Raw Kelly with winning streaks could suggest 3-5x normal size. Capping at 2x and flooring at 0.3x keeps sizing within safe bounds while still exploiting momentum.
9. **Regime-change resets are essential**: A 5-win streak in LOW volatility doesn't mean you should bet big entering HIGH volatility. The streak tracker resets on regime transitions, starting fresh in the new environment.
10. **Exponential streak multipliers are aggressive — use sparingly**: 1.3^3 = 2.197 (hits cap). 1/1.8^2 = 0.31 (near floor). Just 2-3 consecutive results produce significant sizing changes. This is by design — streak detection should react quickly.

### Architecture Insights
11. **Parallel agent pattern works brilliantly for independent modules**: US-282, US-283, US-284 were developed simultaneously by 3 parallel agents. No conflicts because each touched different files/methods. Integration agent (US-287) verified they compose correctly.
12. **State persistence at sync_closed_trades_rl is the right place**: All module state (adaptive sizer, Bayesian weights, EWMA correlation) is persisted atomically after RL updates. Single sync point prevents partial state on crash.

### Cumulative Stats
- Phase 36-45: **~1,960 passing tests** across 10 phases
- Phase 45: 1 new module (EWMA correlation, 604 lines) + 5 production wiring edits + streak overlay (~180 lines)
- Phase 45 test breakdown: 392 Phase 45 tests + 60 Phase 44 regression tests = 452 total, 0 failures
- Production pipeline now fully adaptive: regime→sizing→correlation→execution→exit→bayesian learning

---

## Phase 46 Learnings — Entry Signal Filtering & MTF Confluence (2026-03-24)

### Architecture Insights
1. **Regime-conditional gates are composable**: Each regime profile is a self-contained dataclass that can override any combination of thresholds. The `apply_regime_gates()` function merges regime adjustments onto base thresholds, making it trivial to add new regimes or tweak existing ones without touching gate evaluation logic.

2. **Elder Triple Screen works well as a filter, not a signal generator**: The MTF confluence module scores existing signals rather than generating new ones. This design means it can be layered on top of any agent's output without changing the agent itself — just multiply confluence score into confidence.

3. **Session detection must handle midnight-crossing**: Tokyo session (23:00-08:00 UTC) wraps past midnight. The SessionDetector handles this by checking `hour >= start OR hour < end` for wrapping sessions vs `start <= hour < end` for normal ones. This edge case bit us in initial testing.

4. **Graduated penalty curves beat binary thresholds**: The ensemble conflict resolver uses linear interpolation between 5 penalty points instead of hard cutoffs. This gives the system smooth degradation — a 0.35 disagreement gets -12% vs the old flat -15%, while 0.50 correctly blocks entirely.

5. **Expectancy reframes optimization correctly**: Win rate alone is misleading. An agent with 30% win rate but 3:1 avg win/loss has positive expectancy ($0.30×3 - $0.70×1 = $0.20). The ExpectancyTracker captures this per agent per regime, enabling data-driven weight adjustments that optimize for returns, not just accuracy.

### Testing Patterns
6. **Synthetic OHLCV data with controlled indicators**: For MTF confluence tests, we generate DataFrames where SMA slopes, RSI values, and ADX strength are predictable from the input prices. This makes tests deterministic without needing real market data.

7. **Integration tests verify module composition, not individual logic**: Phase 46 integration tests verify that regime gates → MTF confluence → session filter → ensemble conflict compose correctly. Each module's unit tests already verify internal logic — integration tests verify the contracts between modules.

### Cumulative Statistics
- Phase 36-46: ~2,270 passing tests across 11 phases
- Phase 46: 5 new modules, 310 new tests, 6/6 stories complete
- New capabilities: regime-adaptive gates, Elder confluence, session awareness, graduated ensemble penalties, expectancy tracking

## Phase 47 Learnings — Production Wiring of Phase 46 Modules (2026-03-24)

### Wiring Patterns
1. **Lazy-init with fallback is the canonical wiring pattern**: Every Phase 46 module was wired using: try/import/init in `__init__()`, wrapped call in the hot path, except→fallback to legacy. This pattern (established in Phase 45) proved reliable again — 0 production crashes from integration failures.

2. **Legacy hard blocks interact with new graduated penalties**: The existing `_evaluate_uncertainty()` has a hard block at `model_disagreement > 0.30` computed from heuristic indicators. The new EnsembleConflictResolver operates on TCN/Ridge/RF model scores independently. Both must pass — the legacy check is a safety floor that the graduated penalty can't override.

3. **Position sizing chain order matters**: The final chain is: base → adaptive (Kelly+confidence+drawdown+regime) → regime multiplier → session multiplier → EWMA diversification → streak overlay. Each layer compounds multiplicatively, so a 0.65x regime × 0.75x session × 0.8x EWMA = 0.39x total. The floor protections in each layer prevent sizing from going below minimum lot.

4. **Test data must match heuristic expectations**: Random OHLCV data creates unpredictable heuristic values (close vs SMA, RSI direction, return sign). For deterministic tests of uncertainty/disagreement, use synthetic data with controlled monotonic trends so heuristics agree with the direction under test.

### Cumulative Statistics
- Phase 36-47: ~2,324 passing tests across 12 phases
- Phase 47: 0 new modules, 5 pipeline integrations, 54 new tests, 6/6 stories
- Full pipeline now active: regime gates → MTF confluence → session → ensemble conflict → expectancy → adaptive sizing chain

---

## Phase 48 — Trade Intelligence & Adaptive Filtering (2026-03-24)

### Learnings

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
## 2026-03-27 — Tier 5 Certification

**System reached full autonomous self-improvement architecture.**

Key closures this session:
- Episodic memory suppression gate wired into `execute_trade()` before Gate 1 — the last gap between Tier 4.9 and Tier 5. Buddy now blocks repeated losing patterns without human input.
- Aura brain stripped out of `src/aura/` — only bridge protocol remains (`signals.py`, `rules_engine.py`). ~210 lines of dead dispatch code removed from orchestrator + continuous.
- All 6 architectural fixes from Phase 48 audit verified clean: dispatch table, config flag regression test, attribution loop closed, FilterChain wired, Ralph reading PRD stories, Aura shutdown handling.

**Episodic memory full lifecycle confirmed:**
1. `execute_trade()` → `get_suppression_signal()` (earliest gate, fail-open)
2. `_append_journal_entry()` → `record_setup()` → episode_id stored in journal
3. `sync_closed_trades_rl()` → `record_outcome()` → memory learns win/loss

**Ralph is live but waiting:** `PerformancePRDGenerator` needs 30 days of trade outcomes for gap analysis. Not a failure — a dependency on production data. First analysis fires automatically once window is full.

**Architecture is now self-sustaining:** trade → outcome → RL weights + episodic memory + drift detection → config adjustment → Ralph PRD generation → code improvement → repeat. No human in the loop required.

## 2026-03-27 — Phase 55 Complete

**US-343 (DisagreementTracker):** Coefficient of variation (std/mean) is the correct disagreement metric because it normalizes spread relative to magnitude — allowing comparison across different confidence scales. Toxic pair detection via above/below-mean camp split is more stable than raw pair-score comparison. The 70% loss-rate threshold with 5-observation minimum prevents premature flagging.

**US-344 (FeatureHealthMonitor):** PSI bins must use the combined (recent + baseline) range to ensure consistent bucketing. Flooring bin counts at 1 (not 0) prevents log(0) in the PSI formula. The `feature_health_score = mean(1 - min(psi/0.50, 1.0))` formula maps directly to [0,1] and feeds WFEStabilityMonitor without any custom alert pathway.

**US-345 wiring lesson:** The "live wiring" test pattern (grep for `self._tracker.method()` in production files) is more reliable than unit tests that mock boundaries. Both should exist.

### Phase 65 ScanHealthSynthesizer v2 — 2026-03-29

- [2026-03-29] **PATTERN/always_read_actual_method_signature_before_writing_call**: `MomentumScoreRecorder.record()` takes `momentum_score=` not `xgb_momentum=`, and has no `threshold=` parameter — the PRD described the field conceptually, not the actual Python signature. **Rule**: Before writing any `.record()` call in a test or wiring site, grep the actual `def record(` in the target class. PRD names describe intent; actual signatures may differ. This applies equally to test helpers and production wiring.

- [2026-03-29] **PATTERN/fourth_gate_agent_consensus_has_no_adaptive_mechanism**: After Phases 58/62/64 gave confidence/momentum/risk adaptive floors, the agent consensus gate (`weighted_vote_threshold=0.55`) remains completely static. Virtual trade analysis shows `agent_passed=False` as the dominant kill stage in ALL 5 most recent rejects. The three-gate adaptive loop is complete but the fourth gate (agent vote) is still hardcoded. **Rule**: When building adaptive floors for a multi-gate system, enumerate ALL gates and confirm each one has an adaptive mechanism. A single static gate becomes the new bottleneck once the others adapt.

- [2026-03-29] **PATTERN/stall_diagnosis_requires_virtual_trades_not_gate_attribution**: `gate_attribution.json` had `decisions: []` despite an active stall (Phase 56 GateRejectionTracker not accumulating data in this run). `virtual_trades.jsonl` provided the actual gate failure breakdown (`agent_passed=False` in 5/5 recent entries). **Rule**: When diagnosing a 0-trade stall, read `virtual_trades.jsonl` directly — it contains `gate_failures` per pair/scan and gives definitive kill-stage evidence. `gate_attribution.json` is a derived aggregate that may be stale or empty after process restarts.
