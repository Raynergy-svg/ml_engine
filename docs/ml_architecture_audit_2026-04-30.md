# Buddy ML Architecture Audit — April 30, 2026

**Audience:** operator + ML engineering
**Scope:** training pipeline, model architectures, validation, online learning, RL feedback, calibration
**Verdict:** the stack is *mostly current* for FX intraday/swing on a modest data budget, but it leaves three real performance levers on the table (probabilistic forecasting, self-supervised pretraining, online drift adaptation), and the "RL" layer is mislabeled — it's contextual EWMA, not RL. None of the gaps are catastrophic.

---

## 1. Executive Summary

- **The "TCN+Ridge+RF" framing in `CLAUDE.md` is out of date.** The actual ensemble is **Transformer (direction) + LightGBM (momentum) + LightGBM (risk) + LightGBM-as-Ridge (confidence) + TCN (volatility regime, 4-class) + HistGB (direction baseline) + TransformerRegime (3-class trend/chop/MR) + XGBoost (meta-labeler)**. RF survives only as a fallback on the risk gate. Ridge is now LightGBM under the hood.
- **What's strong:** López de Prado-style purged k-fold + embargo (`walkforward_validation.py:130,365`), triple-barrier meta-labeling with XGBoost (`meta_labeling.py:75-76`), Platt + Isotonic calibration (`src/risk/confidence_calibration.py:118-129`), EWC + EMA shadow weights for continual learning in the Transformer, joint multi-pair training with per-instrument fine-tune, proper online retrainer with cooldowns, regime-conditioned ensemble weighter (small NumPy MLP), drift detector wired to retraining trigger.
- **What's weak:** (1) **No probabilistic/distributional forecasting anywhere** — every head outputs a point estimate, then calibrators sit on top. Quantile/conformal would directly improve position sizing. (2) **No self-supervised pretraining** on unlabeled OHLCV. The Transformer trains 17 epochs from scratch on ~28k samples and gets 52.9% val accuracy — masked-bar pretraining would likely move the floor 1-3 points. (3) **"RL agent weight updates" are EWMA-damped multiplicative bandits** (`_team.py:920-994`); not actually RL despite the name. PPO position sizer (`rl/position_sizer.py`) is the only real RL component.
- **Top 3 upgrades (ranked by ROI/effort):** (a) Replace Transformer direction head with a **conformal-prediction wrapper around the existing LightGBM/HistGB direction baseline** for distributional outputs (low effort, immediate position-sizing benefit). (b) Add **masked-bar self-supervised pretraining** for the Transformer encoder (medium effort, free alpha on unlabeled OANDA history). (c) Pilot a **time-series foundation model (Chronos-Bolt / TimesFM v2 / Moirai-2)** zero-shot as a model-bandit member; if it survives, switch the direction head's role to "forecast → meta-labeler decides".
- **Biggest risk if you change nothing:** the direction head is at the noise floor (~53% val), gates do most of the alpha lifting, and the meta-labeler is over-fit-prone (`max_overfitting_gap=0.10`). If meta-labeler quality drifts, edge collapses fast — and there's no ensemble-of-meta-labelers to catch it.

---

## 2. Current Architecture Inventory

| Component | File:line | Status (Apr 2026) | SOTA gap |
|---|---|---|---|
| **Direction** (primary signal) | `src/training/trainers/transformer_trainer.py:289-382` | Tiny Transformer: `d_model=16, num_heads=2, num_layers=1`, sinusoidal PE, BCE loss, EMA + EWC + replay + drift detector + LLRD optimizer | Architecture is fine for the data scale. Missing: SSL pretraining, distributional output. PatchTST/iTransformer would be marginal at this seq_len. |
| **Direction baseline** | `src/training/trainers/histgb_trainer.py:29-65` | sklearn `HistGradientBoostingClassifier`, optional PCA, native NaN handling, used as sanity check + hybrid voter | Solid. Could swap to CatBoost (ordered boosting, time-series-friendly) — already a dep at `requirements.txt:26`. |
| **Volatility regime (4-class)** | `src/training/trainers/tcn_trainer.py:88-169` | Dilated causal Conv1D, `filters=32, kernel=3, layers=2, dropout=0.4`, BN + L2 + spatial dropout + Gaussian noise, gradual unfreeze on warm-start | TCN is dated for sequence work but **fine for 4-class regime classification** at this seq_len. Don't change. |
| **Trend/chop/MR regime (3-class)** | `src/training/trainers/transformer_regime_trainer.py:40-80` | Same Transformer skeleton, 3-class softmax | Overlaps with TCN volatility. Consider merging into one multi-task head. |
| **Momentum** | `src/training/trainers/lightgbm_trainers.py:428-510` | LightGBM regressor (momentum_score) + classifier (acceleration), warm-start via `init_model` | Strong. CatBoost or NGBoost variant would add probabilistic output. |
| **Risk** (drawdown + streak) | `src/training/trainers/lightgbm_trainers.py:655-680` | LightGBM regressors (replaces RF). RF still loaded as fallback (`modular_inference.py:1797-1803`) | Strong. Quantile regression would beat point estimates for SL/TP sizing. |
| **Confidence (gate scoring)** | `src/training/trainers/ridge_trainer.py:48-122,150-234` | LightGBM regressor (was ElasticNetCV). GPU→CPU fallback. R² 0.997 on val (suspiciously high — verify no leakage) | The R² 0.997 in `joint_training_meta.json` smells like target leakage. Verify confidence target is not derived from future prices. |
| **Meta-labeler** | `src/training/meta_labeling.py:51-80` | XGBoost (200→100 estimators, max_depth=2, reg_alpha=1, reg_lambda=5, dropout=0.35, early stopping). Triple-barrier labels. | Solid López de Prado implementation. Consider ensemble-of-meta-labelers for robustness. |
| **Validation** | `src/training/walkforward_validation.py:130,248,365` | Walk-forward (expanding + rolling) + purged k-fold with embargo gap. `gap=10`, `embargo_gap=12` defaults. | Correct. Don't change. |
| **Calibration** | `src/risk/confidence_calibration.py:96-129,222-229` | Platt + Isotonic + 'both' (mean of two). Recalibrates from journal periodically. | Good. Add temperature scaling on Transformer logits as a third option (cheaper, often as accurate). |
| **Online retraining** | `online_retrainer.py:33-70` | Cooldown-protected retrain of XGB/RF/Ridge on replay buffer. Transformer skipped (too expensive). 60min cooldown, max 3/day. | OK. But replay buffer is unweighted; recent samples should dominate (EWMA weighting or windowed importance sampling). |
| **Agent weight RL** | `src/scanner/agents/_team.py:901-994` | **Mislabeled.** EMA-damped multiplicative bandit: `w_new = α · clip(w + δ) + (1-α) · w_old`, with α=0.7 default, 0.6 for high-variance agents. Per-regime + global. | Not RL. It's a sane online weight updater. Real contextual bandit (LinUCB / Thompson) on regime+agent_id features would be a strict improvement. |
| **Position sizing RL** | `src/training/rl/position_sizer.py:51-100` | PPO via stable-baselines3, 50k timesteps, sequence_length=60, sharpe-shaped reward + drawdown penalty. Real RL. | Modern. Could add CQL/IQL on the offline trade journal as an alternative. |
| **Ensemble weighting** | `src/recursive_intelligence/ensemble_weighting.py:152-212` | NumPy MLP (n_context=6 → hidden → n_models=4), context-conditioned weights, online retrain | Correct shape. Could switch to gradient-boosted model selector. |
| **Joint multi-pair training** | `src/training/trainers/joint_trainer.py:43-125` | Combines 15 pairs with per-pair normalization + instrument one-hot. Pair-specific fine-tune via LightGBM `init_model`. | Good. But one-hot is weaker than learned pair embeddings (8-16 dim) — easy upgrade. |
| **Feature stack** | `src/data/feature_engineering.py:32-280` | ~186 hand-crafted features: SMA/EMA, MACD, RSI (Wilder), Bollinger, Stochastic, ATR + ATR-pct (4 windows), CCI, OBV, MFI, Williams %R, ROC, ADX/DI, regime flags, MA alignment, BB squeeze, divergences, interactions | Heavy hand engineering. A modern sequence model would learn most of this implicitly, but LightGBM **needs** these and the meta-labeler benefits. Don't strip; do consider adding learned embeddings as auxiliary features. |
| **Labels** | `src/core/modular_data_loaders.py:1684-1716`, `multitask_labels.py:42-100` | Direction: binary up/down over `lookahead` bars with threshold filter (clear/unclear weight). Risk: forward max drawdown over horizon. Volatility: 4-class quantile bins. Meta: triple-barrier. | Mixed. Direction's threshold-filter is a poor man's triple-barrier — could unify with the meta-labeler's triple-barrier. |
| **Drift detection** | `src/scanner/automation/concept_drift.py`, `transformer_trainer.py:280` | 3-stream distributional shift monitor, threshold 0.03, performance + feature drift triggers retrain | Good. KS-test or MMD would be more principled but the heuristic works. |

---

## 3. Component-by-Component Evaluation

### 3.1 Sequence Model (Transformer Direction) — defensible, not lapped
The Transformer at `transformer_trainer.py:307-380` is intentionally *tiny* (`d_model=16, num_heads=2, num_layers=1`, ~few thousand params). For 28k training samples this is correct sizing — bigger models overfit. Joint training across 15 pairs gets sample count up.

**Is TCN+Ridge+RF lapped by 2026 SOTA?** For FX bars at this data scale: **no**. PatchTST, iTransformer, TimesNet matter when `seq_len > 96` and you have millions of samples. Buddy uses short sequences and walk-forward folds with thousands of samples each. State-space models (S4, Mamba-2) shine on very long sequences — irrelevant here.

**Where Buddy actually loses ground:**
- **No SSL pretraining.** A masked-bar BERT-style pretrainer on years of unlabeled OANDA tick/bar data would warm-start the encoder and likely move val accuracy from 0.529 → 0.54-0.55. This is the single best architectural win available.
- **No distributional output.** All heads output point estimates; calibration is post-hoc Platt/Isotonic on a scalar. Quantile regression (pinball loss) or conformal prediction would directly improve position sizing because `DynamicPositionSizer` (`src/risk/position_sizing.py`) is currently sizing off a single confidence number.

**Foundation models (Chronos-Bolt, TimesFM v2, Moirai-2, Lag-Llama, TabPFN-TS):** plausibly useful as **zero-shot ensemble members**, gated by the model bandit (`trained_data/model_bandit.json` exists). Inference latency is the constraint — Chronos-Bolt is fast (<50ms/forecast on CPU for short horizons), Moirai-2 is heavier. These do **not** survive the "no LLM in hot path" rule strictly speaking, but they aren't generative LLMs and the rule's intent (auto-regressive token generation per scan) doesn't apply. Frame as "specialized forecaster", not LLM.

### 3.2 Tabular/Ensemble Layer — already modernized
LightGBM is in for momentum, risk, and confidence. XGBoost runs the meta-labeler. CatBoost is in `requirements.txt:26` but I see no consumer in production paths — verify whether it's actually used or dead dependency.

NGBoost (probabilistic gradient boosting) would drop in cleanly for the risk head — output a Normal/Laplace distribution over expected drawdown instead of a point estimate. ROI is real because position sizing reads drawdown.

TabPFN v2 is **not** a fit here — it caps at ~10k training rows, doesn't do time series natively, and overlaps with what HistGB already does.

### 3.3 Probabilistic Forecasting — biggest gap
Every head outputs a point estimate. No quantile loss, no conformal intervals, no NGBoost, no MC dropout sampling at inference, no Bayesian heads. The closest thing is calibration of binary direction probability (Platt/Isotonic).

For risk-aware position sizing this is the clearest miss. Options ranked by effort:
1. **Conformal prediction wrapper** (`mapie` library, ~1 day) over the existing direction model — gives valid prediction intervals with no retraining. Cheapest possible distributional output.
2. **Quantile LightGBM** for the risk head (`objective='quantile'`, train 3 models for q=0.1, 0.5, 0.9 — 1 week including position-sizer rewiring).
3. **NGBoost** for momentum + risk (probabilistic GB, replaces LightGBM regressors). 1-2 weeks.

### 3.4 Self-Supervised Pretraining — leaving free alpha on the table
No masked pretraining anywhere. The Transformer trains supervised from scratch on labeled bars. OANDA gives you years of unlabeled history for free. A masked-bar reconstruction objective (predict masked patches of OHLCV given context, MAE-style) on 5+ years of bars across 15 pairs would pretrain the encoder, then fine-tune on labeled direction. Expected lift: 1-3% val accuracy on direction head, more on small pairs. Effort: 2-3 weeks. ROI: high.

Contrastive SSL (TS2Vec, TF-C) is an alternative — pretrain by maximizing similarity between augmented views of the same bar window. Slightly more involved than masked reconstruction. Same target lift.

### 3.5 Online Learning / Drift — adequate, room to grow
`online_retrainer.py:33-70` does cooldown-protected retraining of sklearn models from a replay buffer, with drift-detection trigger. Fine. River (online gradient boosting) could replace the replay-buffer-then-batch-retrain pattern with true online updates — but the current scheme is safer for live trading because every retrain is checkpointed.

The replay buffer is unweighted (FIFO). Recent samples should dominate; consider EWMA-weighted sampling at fit time.

### 3.6 RL Layer — partially mislabeled
- `agents/_team.py:901-994` calls itself "RL feedback" but is **EMA-damped contextual EWMA bandit**: per-agent multiplicative weight updates by trade win/loss, regime-conditioned + global. There's no value function, no policy, no Bellman update. This is fine — bandit-style updates are standard for ensemble weighting — but the naming is misleading.
- `rl/position_sizer.py:51-100` is **real PPO** via stable-baselines3, 50k timesteps, on a custom `TradingEnv`. This is modern.

Upgrade options:
- **Replace EMA-bandit with LinUCB or Thompson sampling** for agent weights: same online property, but with explicit exploration-exploitation tradeoff and uncertainty estimates per arm. Modest ROI, ~1 week effort.
- **Offline RL (CQL/IQL)** on `trade_journal_rl.json` for position sizing — could replace PPO. Higher sample efficiency on Buddy's modest trade volume. 2-4 weeks effort, real risk of regression.

### 3.7 Calibration — well-done
Platt + Isotonic + ensemble of both, recalibrated periodically from the trade journal (`confidence_calibration.py:491+`). This is industry standard. Adding **temperature scaling** as a third option is cheap and often as accurate as Platt with one fewer parameter to fit.

### 3.8 Validation — well-done
Walk-forward (expanding + rolling) + purged k-fold + embargo (`walkforward_validation.py:130, 365-398`). López de Prado-grade. The `gap=10`, `embargo_gap=12` defaults are reasonable for H1 bars. Don't change.

### 3.9 Latency
No explicit per-scan timing instrumentation found in `engine.py`. Models are pre-loaded at startup (`engine.py:213-237`, "Phase 69 warm-up"). Per-prediction latency for the current stack:
- TCN: ~5-15ms (small Conv1D)
- Transformer: ~5-15ms (tiny)
- 3× LightGBM: ~1-3ms each
- HistGB: ~1-3ms
- Meta-labeler XGB: ~1-3ms

Total per-scan inference budget is comfortably under 100ms on CPU, leaving headroom for additions. Foundation model would consume most of that headroom.

---

## 4. Ranked Upgrade Roadmap

| Rank | Upgrade | Effort | ROI | Risk | Library / Reference |
|---|---|---|---|---|---|
| **1** | **Conformal prediction intervals on direction model** | 3-5 days | High (immediate position-sizing benefit) | Low | `mapie` (Taquet et al., 2022, still SOTA in 2026 for tabular CP); `crepes` for online conformal |
| **2** | **Masked-bar self-supervised pretraining for Transformer** | 2-3 weeks | High (~1-3% val accuracy floor lift) | Low-medium | TS-MAE (Du et al., 2024), or simpler: BERT-style masked reconstruction. Pretrain on 5+ years OANDA history across all pairs, fine-tune on labels. |
| **3** | **Quantile LightGBM for risk head + position-sizer rewire** | 1 week | High (proper distributional drawdown forecast) | Low | LightGBM `objective='quantile'`. Train q=0.1/0.5/0.9 ensemble. |
| **4** | **Pair embeddings (replace one-hot)** | 3-5 days | Medium (better cross-pair transfer) | Low | Standard learned embedding layer (8-16 dim) in joint trainer. |
| **5** | **Foundation model as bandit ensemble member** | 1-2 weeks | Medium-high (zero-shot baseline catches regime shifts) | Medium (latency + dependency size) | Chronos-Bolt (Amazon, 2024-2025) — fastest. Moirai-2 (Salesforce, 2025) — strongest. TimesFM v2 (Google, 2025). Use via Hugging Face. Gate via `model_bandit.json`. |
| **6** | **Temperature scaling on Transformer logits** | 1 day | Low-medium (parameter-efficient calibration) | None | Guo et al. 2017, still standard. Add as third option in `confidence_calibration.py`. |
| **7** | **LinUCB / Thompson sampling for agent weights** | 1 week | Medium (better exploration; explicit uncertainty) | Low (parallel-deploy, A/B vs current EMA) | `vowpal_wabbit` contextual bandit, or roll-your-own (~150 LOC). |
| **8** | **EWMA-weighted replay buffer sampling** | 2 days | Low-medium (fresher retrains) | None | Modify `online_retrainer.py:42-46` to weight by recency at fit time. |
| **9** | **Merge TCN volatility + Transformer regime into multi-task head** | 1-2 weeks | Low (cleanup; minor compute saving) | Medium (regression risk on existing trained models) | One encoder, two softmax heads. Defer until you have other reasons to retrain. |
| **10** | **Offline RL (CQL/IQL) for position sizing** | 2-4 weeks | Medium (sample efficiency) | High (PPO works; don't break it) | `d3rlpy` library. Run as shadow mode against PPO before promoting. |

---

## 5. What NOT to Change

- **Walk-forward + purged k-fold + embargo** (`walkforward_validation.py`). This is the single most important thing standing between "works in backtest" and "works live." Don't touch.
- **Triple-barrier meta-labeling with XGBoost** (`meta_labeling.py:51-80`). López de Prado canon. Already conservatively regularized (`reg_lambda=5, max_depth=2, dropout=0.35`). Solid.
- **TCN for 4-class volatility regime** (`tcn_trainer.py`). TCN is "dated" for sequence modeling generally, but for short-sequence multi-class regime classification it's right-sized and fast. Replacing with a Transformer here would gain you nothing.
- **EMA + EWC + replay buffer in Transformer** (`transformer_trainer.py`). This is sophisticated continual-learning machinery already in place. EWC for cross-pair learning without catastrophic forgetting is exactly right for joint training.
- **Platt + Isotonic calibration ensemble** (`confidence_calibration.py:118-129`). Both methods together is more robust than either alone. Recalibration from trade journal closes the loop.
- **PPO position sizer** (`rl/position_sizer.py`). Real RL with proper environment, sharpe-shaped reward, drawdown penalty. Mature. Don't replace until offline RL proves itself in shadow.
- **The 186-feature engineered stack.** LightGBM + meta-labeler genuinely benefit from these. A modern sequence model could learn many implicitly, but you'd lose the gradient-boosting models. Keep.
- **Cooldown-protected online retrainer** (`online_retrainer.py:38-43`). Conservative; financial-safe; correctly excludes Transformer from in-process retraining.

---

## 6. Open Questions for the Operator

1. **Confidence model R² = 0.997** in `joint_training_meta.json`. That's almost surely target leakage — what is the confidence target derived from? If it's a function of future returns or future volatility, this is a real bug and the gate is meaningless. Top priority to verify.
2. **CatBoost** is in `requirements.txt:26` ("Primary momentum gate model") but I find no live consumer — momentum gate uses `LightGBMMomentumTrainer`. Is CatBoost stale dep, or is there a deployment path I missed?
3. **Direction val accuracy 52.9%** — is the strategy designed to be edge-from-gates-not-direction (i.e., direction is a coin-flip filter and the meta-labeler + agent consensus produce alpha)? If yes, lifting direction is lower priority than I ranked it. If no, item #2 (SSL pretraining) becomes #1.
4. **Foundation models (Chronos/TimesFM/Moirai)** — do these violate the "no LLM in hot path" rule (CLAUDE.md)? My read: rule is about generative LLMs (token-by-token, Claude-class). These are dedicated time-series transformers with deterministic forward passes, ~50ms inference. Want operator's interpretation.
5. **Trade journal volume** — how many trades/month is Buddy producing? Determines whether offline RL (CQL/IQL) is feasible (need ≥1k trades for stable training) or whether we should stick with PPO + simulator.
6. **Compute budget for retraining** — current retrain cycle is M1 Metal (`m1_metal_optimizer.py`). Is GPU/cloud retraining viable for monthly SSL pretraining runs, or are we constrained to M1?
7. **Do you want item #1 (conformal) and item #3 (quantile risk) wired into `DynamicPositionSizer`, or kept as advisory signals?** Wiring them means reworking `src/risk/position_sizing.py` to consume distributions, not scalars — small but real change.

---

## Appendix: Inventory at a Glance

**Live model artifacts** (`trained_data/models/joint/`):
- `transformer_direction.keras` — primary direction model (Transformer, d_model=16)
- `tcn_volatility_regime.keras` — TCN 4-class
- `lgbm_momentum.pkl` — LightGBM momentum + acceleration
- `lgbm_risk.pkl` — LightGBM drawdown + streak
- `ridge_confidence.pkl` — LightGBM confidence (legacy name)
- Per-pair fine-tunes in `trained_data/models/{INSTRUMENT}/`
- EMA shadow + EWC penalties stored alongside

**Active training entry points:**
- `src/training/trainers/joint_trainer.py` — orchestrator
- `src/training/trainers/train_all.py` — multi-trainer launcher
- `src/training/walkforward_orchestrator.py` — walk-forward loop
- `src/training/meta_labeler_retrainer.py` — periodic meta-labeler refresh
- `online_retrainer.py` — drift-triggered in-process retraining
