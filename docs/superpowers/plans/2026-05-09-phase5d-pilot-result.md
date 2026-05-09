# Phase 5.D Pilot — USD_JPY W&B Sweep Result

> **Status**: 10-trial Bayesian + Hyperband sweep complete. Best config
> beats baseline by **+2.45pp** at training-time val_balanced. Below the
> +3pp auto-expand threshold; shipping best config without expanding to
> other masters.
>
> Pending: confirm the gain transfers to TRUE OOS (Path 4 holdout
> rerun with tuned config).

---

## 1. Sweep configuration

- **Sweep ID**: `buddy-master-tuning/s4wcqfle`
- **Method**: Bayesian + Hyperband (min_iter=5, eta=3)
- **Pair**: USD_JPY (highest baseline; pilot for sweep methodology)
- **Granularity / candles / lookahead**: H1 / 25000 / 24
- **Metric**: val_balanced_accuracy (maximize)
- **Trials**: 10
- **Wall clock**: ~2h

Search space:
- Architecture: d_model {8,16,32,64}, num_heads {1,2,4}, num_layers {1,2,3}, dff {16,32,64,128}
- Regularization: dropout uniform(0.1, 0.5), ewc_lambda log-uniform(0.01, 10)
- Optimization: lr log-uniform(1e-5, 5e-3), batch_size {32,64,128,256}, top_k_features {30,50,80}
- Training: patience {10,15,20,30}, epochs {50,100,200}

## 2. Trial leaderboard

| Rank | Trial | val_balanced | d | h | l | dff | drop | lr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | **56.17%** | 8 | 4 | 2 | 64 | 0.47 | 9.5e-4 |
| 2 | 8 | 56.09% | 32 | 4 | 1 | 128 | 0.35 | 3.2e-3 |
| 3 | 6 | 54.83% | 32 | 4 | 1 | 128 | 0.44 | 3.0e-3 |
| 4 | 3 | 54.38% | 16 | 1 | 2 | 128 | 0.48 | 4.4e-4 |
| 5 | 7 | 53.89% | 16 | 4 | 1 | 32 | 0.44 | 7.6e-4 |
| 6 | 9 | 53.53% | 64 | 4 | 1 | 128 | 0.17 | 1.2e-3 |
| 7 | 4 | 53.33% | 8 | 4 | 3 | 64 | 0.39 | 6.5e-4 |
| 8 | 2 | 51.21% | 64 | 1 | 2 | 32 | 0.37 | 5.0e-3 |
| 9 | 10 | 50.64% | 16 | 4 | 2 | 64 | 0.49 | 3.0e-3 |
| 10 | 5 | 48.28% | 8 | 1 | 3 | 16 | 0.46 | 1.5e-5 |

Baseline (Path 4 default config): val_balanced = 53.72%.

**Best lift: +2.45pp.**

## 3. Patterns in the data

**What works**:
- **Higher dropout** (0.35-0.50). The default 0.2 underregularized given the
  ~12K filtered training samples. Top 4 trials have dropout ≥ 0.35.
- **Moderate-to-high LR** (~1e-3 to 3e-3). Default LR was probably too low.
- **Either narrow-deep (d=8, l=2) or wide-shallow (d=32, l=1, dff=128)**.
  Both hit ~56%. The architecture itself matters less than capacity-
  regularization balance.
- **More attention heads** (4 over 1). Three of the top-4 trials use heads=4.

**What hurts**:
- **Very low LR** (1e-5, trial 5): underfit, val_bal=48.28%.
- **Very high d_model** without matching dropout (trial 9, d=64 drop=0.17):
  overfit, train 68.8% / val 53.5%.
- **Very high LR** with big model (trial 2, d=64 lr=5e-3): didn't converge
  to anything useful, val_bal=51.21%.

## 4. The 56% ceiling

Trial 1 (small narrow deep) and trial 8 (wide shallow) reached the same
~56.1% from different architectures. This is consistent with **the data's
information content being the binding constraint**, not the model's
expressive capacity. Adding more capacity beyond what trials 1/8 use
doesn't help — additional epochs / parameters just memorize noise.

Implication: this 56% is approximately the H1 lookahead=24 ceiling for
USD_JPY at current training-data scale. To break past it, the levers are:
- **More training data** (longer history + more pairs joint-trained)
- **Different label scheme** (shorter lookahead, or threshold-filtered to
  high-conviction moves only)
- **Different architecture class** (state-space / Mamba; foundation-model
  fine-tune)

## 5. Decision: ship best config without expansion

Per the operator's stated decision tree (53.72-56.71% range):
- Lift is real (Bayesian search consistently found ~56% from multiple paths)
- Lift is modest (+2.45pp; below the +3pp expansion threshold)
- Pattern doesn't suggest other masters will hit higher (USD_JPY is the
  highest-baseline master; if it ceilings at 56%, others are likely lower)

**Action: apply best config to USD_JPY production model + run TRUE OOS
holdout.** If OOS improves over the 60.2% baseline, ship it. If OOS is
flat or worse, the val_balanced gain didn't transfer (overfitting); revert
to default config.

## 6. Best config (locked)

```python
TrainerConfig(
    transformer_d_model=8,
    transformer_num_heads=4,
    transformer_num_layers=2,
    transformer_dff=64,
    transformer_dropout=0.4723,
    learning_rate=0.0009483,
    batch_size=64,
    top_k_features=80,
    ewc_lambda=0.0341,
    patience=20,
    epochs=200,
)
```

## 7. What this run produced

- W&B sweep `buddy-master-tuning/s4wcqfle` with 10 trials, viewable at
  https://wandb.ai/tencylinder8310-smartdebtflow-com/buddy-master-tuning
- Best-config validation (in flight; will land at
  `trained_data/models/USD_JPY_tuned/`)
- This decision document

Halt remains True throughout. USD_JPY H1 demo path still uses the
non-tuned model (60.2% OOS) until tuned-OOS confirms the gain transfers.
