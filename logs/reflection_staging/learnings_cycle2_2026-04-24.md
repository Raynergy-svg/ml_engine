# Emergency Reflection — Cycle #2 — 2026-04-24T22:21Z

**Trigger:** losing_streak (3 consecutive losses)
**Evidence trades:** 1243 (EUR_GBP SHORT), 1255 (USD_CHF LONG), 1261 (USD_CAD LONG)
**Model freshness:** AGING — agent_weights=8d, modular_ensemble=3d, joint_gates=3d, per_pair=unknown
**Note:** These are the SAME trades from 04-15/04-16. Scanner has produced zero new closed trades since 04-16. Prior cycle 2/3 proposals (04-21) were never applied. Self-heal is firing on stale evidence.

## Shared Loss Fingerprint (all 3 trades)

| Field | 1243 | 1255 | 1261 |
|---|---|---|---|
| regime_at_entry | LOW | LOW | LOW |
| regime_at_exit | NORMAL | NORMAL | NORMAL |
| sl_mult | 0.8 | 0.8 | 0.8 |
| sl_pips | 11.8 | 8.0 | 7.3 |
| atr_pips | ~8-11 | 7.6 | 9.6 |
| SL / ATR | ~1.0x | 1.06x | 0.76x |
| mfe_pips | 0.0 | 0.0 | 0.0 |
| exit_reason | sl_hit | sl_hit | sl_hit |
| model_disagreement | ? | 0.5 | 0.5 |
| uncertainty | ? | 0.2 | 0.2 |
| used_learned_weights | — | false | false |
| agent_total | 3 | 3 | 3 |

## Root Causes (ranked)

### 1. PRIMARY — LOW-regime sl_mult=0.8 still live despite 04-15 rule
Rule `regime_sl` (trading.md, promoted 2026-04-15) mandates LOW regime `sl_mult >= 1.2`. Code still uses 0.8. Every loss had SL within 1x ATR — indistinguishable from noise in a chop market. MFE=0 across all three: price never moved in favor once, meaning SL was triggered by entry-bar noise, not adverse move. **This is not a new finding — it's an un-applied rule.**

### 2. SECONDARY — model_disagreement=0.5 did not hard-block
rules/trading.md line: "Model disagreement > 0.30 is a loss predictor." Trades 1255/1261 had disagreement=0.5 and still executed (risk_passed=true). Mean-reversion-veto rule (2026-04-17) should have fired here — but `enable_mean_reversion_veto` is not set to true in active profile.

### 3. TERTIARY — LOW-regime classifier is miscalibrated at entry
All 3 entered as LOW regime and exited as NORMAL. Volatility classification was wrong at entry bar OR volatility jumped immediately. Either way, SL sized for LOW is guaranteed to blow on NORMAL-regime ATR. Classifier needs a stickiness/hysteresis correction OR SL must size to `max(LOW_atr, NORMAL_atr)` when within 20% of regime boundary.

### 4. QUATERNARY — `used_learned_weights=false` on gate ensemble
Core gate scoring fell back to static weights (direction=0.35, conf=0.3, momentum=0.2, risk=0.15). The learned-weights path is not loading — likely a file-path or schema issue. Unrelated to agent_weights, but silently degrades gate quality.

### 5. QUINARY — Agent team reduced to 3 of 15
`agent_total=3` (trend, mean_reversion, risk_sentinel) across all 3 trades. The Extended 3 (order_flow, trader_readiness, devil_advocate) and the other 9 core agents did NOT vote. This is massive agent-signal degradation. Profile likely has extended agents disabled OR their enable flags haven't reached the dataclass (see `rules/improvement.md` Config Adjustment Consumer Verification).

### 6. Model staleness — NOT primary
oldest_age_days=8.0 (< 14d hard-block threshold). Staleness is a contributing factor but not the trigger. However agent_weights last trained 04-16 = right when this losing streak ended. RL sync never updated weights to reflect the 14-loss streak learnings because retrain did not execute.

## Regime Analysis

Buddy is losing in LOW-classified-but-actually-NORMAL volatility. Training data regime distribution is unknown but all recent losses classified as LOW at entry → confidence in the LOW classifier itself is low. Recommend: add regime-confidence requirement `min_regime_classifier_confidence=0.7` to admit a trade as LOW, else treat as NORMAL.

## Exit Reason Analysis

**100% SL hits with MFE=0.** Not a TP-fast problem, not a time-stop problem, not a trailing issue. Entry bar itself was the wrong direction or SL was inside natural ATR noise. Root cause is SL sizing, not TP or exit logic.

## Actions

1. **Write config proposals** to `logs/reflection_staging/config_adjustments_cycle2_2026-04-24.json` (below) — keys verified against `ScannerConfig` dataclass per orphan-key rule (2026-04-16).
2. **Flag for operator:** cycle 2/3 proposals from 04-21 never reached ConfigAdjuster pending queue. Check ConfigAdjuster.collect_adjustment() and the persistence path. This is the same closed-loop gap documented in learnings.md as a $3,527 bug.
3. **Retrain trigger:** schedule retrain of agent_weights + modular_ensemble + per_pair within 24h. Per_pair age is unknown — that alone warrants a forced retrain.
4. **Do NOT propose weight reductions** this cycle: only 3 of 15 agents even voted. Reducing weights on a degenerate vote set would amplify the degradation. Instead, fix agent enablement first (see action #5).
5. **Operator check:** verify `enable_order_flow_agent`, `enable_trader_readiness_agent`, `enable_devil_advocate_agent` are True in the active profile and reaching ScannerConfig. If any are False or missing from apply_profile(), the trade was approved by 3 unweighted agents.
