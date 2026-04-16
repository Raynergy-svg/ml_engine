# Self-Heal Cycle 3 — DEEP Analysis
# Timestamp: 2026-04-16T11:50:47Z
# Trigger: CRITICAL health degradation — 14 consecutive losses, 28.5d stale modular_ensemble

## Root Cause Analysis (grounded in journal data)

### PRIMARY: Modular Ensemble Staleness (28.5 days)
- modular_ensemble.meta.json trained_at=2026-03-18 (28.5d ago)
- 14/14 closed trades have MFE=0.0 pips — directional model is 100% wrong
- transformer_direction_best.keras retrained today (0d) but modular_ensemble meta unchanged
- buddy_meta.model.json=28.4d, buddy_tf.meta.json=28.4d stale
- rl_position_sizer.meta.json=23.7d, journal_patterns.json=22.6d, pair_rankings.json=27.7d
- The autonomous trainer retrained checkpoints but did NOT update the ensemble meta or per-pair models

### SECONDARY: Config Adjuster Double Bug
Bug 1 (FIXED in code, awaiting restart): _load_state() now loads pending (line 186)
Bug 2 (ACTIVE): 5 of 8 pending config keys don't match ScannerConfig field names:
- min_confidence_threshold -> NO MATCH (nearest: min_confidence, 0-100 scale)
- min_adx_for_trend_trade -> NO MATCH (no such field exists)
- atr_sl_multiplier_low_regime -> NO MATCH (nearest: atr_sl_multiplier, not regime-aware)
- optimized_confidence_threshold -> NO MATCH
- optimized_momentum_threshold -> NO MATCH
MATCHED keys (will apply on restart): max_model_disagreement, max_uncertainty_score, weighted_vote_threshold

### TERTIARY: Model Disagreement Gate Too Permissive
- 9/14 losses had model_disagreement=0.50 (coin flip). Default gate=0.65 passes them.
- 5/14 with disagreement=0.0 also lost — models agree on wrong direction = stale ensemble

## New Learnings (Cycle 3)

- [2026-04-16] PATTERN/config_adjuster_key_mismatch_5_of_8_orphaned: 4th observation of config adjuster pipeline failure. Promotes to rule candidate.
- [2026-04-16] PATTERN/autonomous_trainer_partial_retrain_3rd_observation: 3rd time autonomous_trainer ran without updating ensemble meta. Promotes to rule.
- [2026-04-16] PATTERN/open_trades_on_stale_predictions: 2 real open trades on 28d-stale predictions. Manual close recommended.

## Recommended Actions (Priority Order)

1. IMMEDIATE: Retrain ALL models — python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY,AUD_USD,NZD_USD,USD_CAD,USD_CHF
2. IMMEDIATE: Close open trades 1243, 1261 — stale model predictions
3. FIX: Add key-mapping layer to ConfigAdjuster.apply_adjustments() or fix source key names
4. FIX: Autonomous trainer must verify ensemble meta updated (3rd obs, promotes to rule)
5. VERIFY: Restart scanner to activate _load_state() fix — pending adjustments will load
