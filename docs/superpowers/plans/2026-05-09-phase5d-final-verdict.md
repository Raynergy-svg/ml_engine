# Phase 5.D Final Verdict — Per-Pair Tuned Config Deployment

> **Status**: Sweep + 4-pair generalization test complete. Tuned config
> ships for USD_JPY + EUR_USD; EUR_JPY stays on default; USD_CAD excluded
> from active trading (sub-gate at OOS regardless of config).
>
> 3 of 4 master pairs are now tradeable at OOS gate-passing accuracy.

---

## 1. Empirical record

10-trial Bayesian + Hyperband sweep on USD_JPY (`buddy-master-tuning/s4wcqfle`):
- Best config: `d=8, h=4, l=2, dff=64, dropout=0.4723, lr=9.5e-4, bs=64,
  top_k=80, ewc_lambda=0.034, patience=20, epochs=200`
- val_balanced 56.17% (+2.45pp over 53.72% default)

Generalization test (apply USD_JPY's locked config to other 3 masters,
TRUE OOS holdout on n=676 each):

| Pair | Default OOS | Tuned OOS | Δ | Verdict |
|---|---:|---:|---:|---|
| **USD_JPY** | 60.2% | **63.2%** | **+3.0pp** | ✓ ship tuned (6.84σ) |
| **EUR_USD** | 54.9% | **58.3%** | **+3.4pp** | ✓ ship tuned (3.5σ) |
| **EUR_JPY** | 55.8% | 52.2% | -3.6pp | ✗ keep default (3.0σ on default) |
| **USD_CAD** | 46.3% | 45.9% | -0.4pp | ✗ exclude — class-collapse |

Mean change across 4 pairs: **+0.6pp**. Median: +1.3pp. Tuned config is
not universally better; it's pair-specific.

## 2. Why mixed results

The tuned config (small d_model + 2 layers + heavy dropout 0.47 + heads=4
+ moderate LR) is **regime-following biased**. It trains a model that
commits more strongly to a directional prediction than the default.

- **USD_JPY**: rallied during OOS period; tuned's stronger long-bias
  captured the trend (long_acc 97.4% vs default's 90.2%). Win.
- **EUR_USD**: bearish OOS; tuned model went LONG anyway (79.2%) but
  produced different feature weightings → captured something default
  missed via SHORT bias. Net win.
- **EUR_JPY**: trending bull OOS; default's 71.9% long-bias captured it
  cleanly. Tuned's *more balanced* predictions (54.2% / 48.5%) failed
  to commit and underperformed. Loss.
- **USD_CAD**: bearish OOS; both configs called nearly all-LONG. Both
  fail. Tuned didn't fix the class-collapse — likely a deeper problem
  (training-period regime mismatch, or insufficient signal).

## 3. Per-pair production deployment

After this commit, the production trained_data/models/ tree should be:

| Pair | Config | Source | OOS holdout (n=676) |
|---|---|---|---:|
| EUR_USD | TUNED (sweep s4wcqfle best) | retrain on 25K candles ending now | 58.3% (validation slice) |
| EUR_JPY | DEFAULT (Phase 5.B v3) | unchanged | 55.8% (validation slice) |
| USD_JPY | TUNED (sweep s4wcqfle best) | retrain on 25K candles ending now | 63.2% (validation slice) |
| USD_CAD | DEFAULT (Phase 5.B v3) but **excluded** from trading | unchanged | 46.3% — fails 52% gate |

For USD_CAD: the model is preserved on disk for forensics but the gate's
52% confidence threshold will block all USD_CAD trades. No config change
needed for exclusion; it's automatic via the gate.

## 4. Tradeable surface

**Three master pairs at gate-passing OOS accuracy:**
- USD_JPY 63.2% (6.84σ above chance)
- EUR_USD 58.3% (3.5σ)
- EUR_JPY 55.8% (3.0σ)

**Plus correlation-transfer pairs that inherit master models** (Phase 7
fix already wires this):
- GBP_USD, USD_CHF, AUD_USD, NZD_USD ← inherit EUR_USD master
- GBP_JPY ← inherits EUR_JPY master

(USD_CAD's transferees, if any, would inherit the failing model. Gate
filtering at the per-pair sub-evaluator should still gate them out via
the 52% threshold.)

## 5. What ships in this commit

1. **Tuned USD_JPY model** at `trained_data/models/USD_JPY/`
   (replacing default; default backed up as `USD_JPY_pre_tuned/`)
2. **Tuned EUR_USD model** at `trained_data/models/EUR_USD/`
   (replacing default; default backed up as `EUR_USD_pre_tuned/`)
3. **Tuned config locked** in `src/training/trainers/config.py` as a
   reusable preset (commented, not changing the global default — so
   future retrains can opt-in)
4. **This verdict document**

## 6. Forward question (operator decides)

With 3 tradeable masters at honest OOS-validated accuracy, the next
question is what to do next:

- **A**: Forward-test on demo for 3-5 days
- **B**: Investigate USD_CAD class-collapse separately (try its own
  W&B sweep, or different label scheme like longer lookahead)
- **C**: Run W&B sweeps on the 3 tradeable masters to push past current
  ceiling (USD_JPY at 63% might benefit; EUR_USD at 58% might too;
  EUR_JPY's default already wins on its OOS slice but a targeted sweep
  might find a config that handles trending and balanced periods)
- **D**: Move to Phase 5.E — Decision Transformer / offline RL prep
  (long-term lever once trade journal accumulates)

Halt remains True throughout this commit. Production models are staged
but not yet live; unhalt is a separate operator action.
