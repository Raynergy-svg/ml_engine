# Training Run Timeline — Chronological Audit of val_accuracy Claims

**Date:** 2026-05-18
**Question:** does val_accuracy >= 0.60 actually appear in any logged Buddy training run, and under what config?
**Confidence on the answer:** HIGH (data on disk, every claim cited).

## TL;DR

- **Across 129 wandb runs + 23 meta.pkl artifacts (11 active + 12 quarantined) + 7 trainer logs: ZERO transformer `[canonical]` runs ship a val_accuracy >= 0.60.** Top transformer val ever: **0.5963** (`trained_data/models/USD_JPY/transformer_direction.meta.pkl`, 10 epochs, train=0.8285, gap=23pp — overfit signature, not a clean 60% generalizer).
- The **60%-class numbers the operator likely remembers come from the HistGB hybrid voter, where `val_accuracy` is biased by class imbalance**. `tier1_AUD_USD.log:11:06:56` shows `HistGB trained: val_accuracy=0.6295, balanced=0.5196` — accuracy is 63% only because labels are skewed; balanced is essentially chance. `m15_pair_expansion.log:23:00:30` shows the extreme: `val_accuracy=0.9521, balanced=0.5017`. Reading raw `val_accuracy` from HistGB as "edge" is the misread.
- **Wandb sweeps `s4wcqfle` / `vbq2krwq` ran on H1 with lookahead=24** — NOT M15 with lookahead=12 (which is `direction_training_config.json:7,20-22`). The control plane has not been used in any logged training. Sweep tops: `i3f8wy7o` val=0.5849 / `jety1gsj` val=0.5816, both EUR_JPY H1.
- **All 12 active `trained_data/models/*/transformer_direction.meta.pkl` files lack `lookahead`, `threshold`, `granularity`, `candles` in the `config` sub-dict.** Only `lineage.granularity` survives (`USD_JPY` = `H1`). The control-plane config-bump auditable contract is partially broken.
- **`AUD_JPY`, `EUR_AUD`, `NZD_USD` show identical `val=0.5827 / bal=0.5190 / train=0.5210`** — these are clones from a single broadcast warm-start, not three independent trainings.
- **Disk-full pressure is live**: `df -h /Users/buddy` = 92% used, 18Gi free. joblib already throws "No space left on device" when reading pkls. Relevant to Bug B (autonomous-retrain shipping `val_acc=0.4839` on disk-full degraded I/O).

## Top 10 by val_accuracy — transformer canonical only

Source: wandb summaries (`wandb/run-*/files/wandb-summary.json`) + active `transformer_direction.meta.pkl` files. HistGB excluded (see §"60% verification").

| Rank | val | balanced | train | gap | pair | gran | lookahead | n_train | epochs | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5963 | 0.5368 | 0.8285 | 23.2pp | USD_JPY | H1 | n/a in cfg | 3,731 | 10 | `trained_data/models/USD_JPY/transformer_direction.meta.pkl` (active) |
| 2 | 0.5860 | 0.5703 | n/a | n/a | (unknown) | M15 | 24 | n/a | n/a | `trained_data/logs/m15_pair_expansion.log:06:13:43` |
| 3 | 0.5849 | 0.5451 | 0.7604 | 17.6pp | EUR_JPY | H1 | 24 | 19,153 | 11 | `wandb/run-20260509_172053-i3f8wy7o/` |
| 4 | 0.5827 | 0.5190 | 0.5210 | -0.2pp | AUD_JPY/EUR_AUD/NZD_USD (clones) | n/a | n/a | n/a | n/a | `trained_data/models/{AUD_JPY,EUR_AUD,NZD_USD}/transformer_direction.meta.pkl` |
| 5 | 0.5816 | 0.5547 | 0.5570 | -2.5pp | EUR_JPY | H1 | 24 | 17,412 | 30 | `wandb/run-20260509_134740-jety1gsj/` |
| 6 | 0.5812 | 0.5442 | 0.5679 | -1.3pp | EUR_JPY | H1 | 24 | 19,153 | 22 | `wandb/run-20260509_142400-h91cz9fj/` |
| 7 | 0.5806 | 0.5592 | 0.6648 | 8.4pp | EUR_JPY | H1 | 24 | 19,153 | 22 | `wandb/run-20260509_140534-3uztkvoa/` |
| 8 | 0.5757 | 0.5455 | n/a | n/a | USD_JPY | M15 | 24 | n/a | 6 | `trained_data/logs/m15_usd_jpy_smoke_postfix.log:05:53:55` (post w_train/w_val fix, commit 1a05e75) |
| 9 | 0.5752 | 0.5440 | 0.7485 | 17.3pp | USD_JPY | n/a | n/a | n/a | n/a | `trained_data/models/USD_JPY/_quarantine/transformer_direction-20260513T231542Z/transformer_direction.meta.pkl` |
| 10 | 0.5737 | 0.4935 | 0.6389 | 6.5pp | EUR_JPY | H1 | 24 | 17,412 | 36 | `wandb/run-20260509_130550-y5eqaogo/` |

## Top 10 by val_balanced_accuracy (the metric that actually means something)

| Rank | balanced | val | train | pair | source |
|---|---|---|---|---|---|
| 1 | 0.7755 | 0.5000 | 0.6074 | GBP_CHF | `trained_data/models/GBP_CHF/transformer_direction.meta.pkl` (active — val=50% but bal=77% is class-collapse / metric-mismatch artifact; flag) |
| 2 | 0.5919 | 0.4577 | 0.7261 | AUD_NZD | `trained_data/models/AUD_NZD/_quarantine/transformer_direction-20260514T003324Z/transformer_direction.meta.pkl` |
| 3 | 0.5703 | 0.5860 | n/a | (unknown M15) | `trained_data/logs/m15_pair_expansion.log:06:13:43` |
| 4 | 0.5625 | 0.2808 | n/a | USD_CAD | `trained_data/logs/tier1_USD_CAD.log:11:44:55` (M15) |
| 5 | 0.5617 | 0.5556 | 0.6144 | USD_JPY | `wandb/run-20260509_020850-bu48swr6/` (H1) |
| 6 | 0.5609 | 0.5715 | 0.6645 | USD_JPY | `wandb/run-20260509_032503-hwhg2jbl/` (H1) |
| 7 | 0.5592 | 0.5806 | 0.6648 | EUR_JPY | `wandb/run-20260509_140534-3uztkvoa/` (H1) |
| 8 | 0.5547 | 0.5816 | 0.5570 | EUR_JPY | `wandb/run-20260509_134740-jety1gsj/` (H1) |
| 9 | 0.5542 | 0.5652 | 0.6835 | EUR_JPY | `wandb/run-20260509_174115-zw6kav63/` (H1) |
| 10 | 0.5522 | 0.5723 | 0.6282 | EUR_JPY | `wandb/run-20260509_131330-cm4c4o0d/` (H1) |

Best honest signal: **balanced ~0.55-0.57 on H1 EUR_JPY/USD_JPY lookahead=24**. Nothing on M15 reaches balanced 0.57 except the one `m15_pair_expansion.log:06:13:43` entry, whose context (val=0.5860, balanced=0.5703) is the only M15 transformer that genuinely beat chance by >5pp.

## Chronological timeline (run-bearing dates only)

| Week | Date(s) | Activity | Notable |
|---|---|---|---|
| W18 | 2026-05-04 | `USD_JPY` H1 transformer warm-start, 10 epochs, val=0.5963 (overfit gap 23pp) | Currently the highest-val active model. Trained one week before the 2026-05-08 pipeline-contract fix. Quarantine-eligible by today's contract standards. |
| W18 | 2026-05-05 | 9 wandb runs `ib5lew7q...kqk7q5nt` — all are `retrain/*` orchestration metadata, NO training data, NO val_accuracy. The retrain manifest writes summary but no model trained. | Audit gap. |
| W19 | 2026-05-07 | ~70 wandb runs — same pattern, retrain orchestration only. | No transformer training in wandb. |
| W19 | 2026-05-09 | **20 wandb runs WITH val_accuracy** — sweep `s4wcqfle` (USD_JPY H1) + sweep `vbq2krwq` (EUR_JPY H1). Lookahead=24, candles=25000, granularity=H1. NOT the control-plane config. Top val: 0.5849 (i3f8wy7o). | All on H1, not M15. Sweep tunes dropout/LR/batch over a HPO range; best val plateaus at ~0.58. |
| W19 | 2026-05-13 | M15 pair expansion (`m15_pair_expansion.log`) + quarantine of 8 models (`*/transformer_direction-20260513T*Z/`). M15 transformer val mostly catastrophic (0.13-0.52, balanced 0.49-0.53). One bright spot: val=0.5860 / bal=0.5703 around 06:13. Bug B retrain ships val=0.4839 to `trained_data/models/*/` and quarantines previous artifacts. | The 2026-05-11 SHORT-bias incident + Bug A/B context. |
| W19 | 2026-05-13 09:47 | `m15_usd_jpy_smoke_postfix.log` AFTER w_train/w_val fix (commit `1a05e75`) — val=0.5757 / bal=0.5455, 6 epochs, USD_JPY M15 65k candles | First post-fix smoke. Result is in range of pre-fix H1 sweep performance, NOT a step-change. |
| W20 | 2026-05-14 | More M15 quarantines (USD_CAD, AUD_NZD, EUR_GBP). All val < 0.54, balanced near 0.50-0.59. | Quarantine churn. |
| W20 | 2026-05-18 | `m15_usd_cad_regularization_2026_05_19.log` — USD_CAD M15 val=0.4664 / bal=0.4663 (chance) | Why USD_CAD just got re-quarantined. |

## The "60% is routine" verification — answer

**No transformer canonical run in any of the 129 wandb summaries, 11 active meta.pkl files, 12 quarantined meta.pkl files, or 7 trainer logs has val_accuracy >= 0.60.** (HIGH confidence; cited per-source.)

The closest call: **USD_JPY active meta at val=0.5963 / train=0.8285** — gap of 23pp marks it overfit, not a clean 60% generalizer. Per `_quarantine` policy plus the train-vs-val gap >0.12 threshold in `config.max_acceptable_gap=0.12`, this artifact would be quarantined today; it survives only because it was trained 2026-05-04 (pre the 2026-05-08 contract fix) and the auto-promote logic was different then.

**Most likely source of the "60% routine" claim — HistGB val_accuracy.** Examples:
- `tier1_AUD_USD.log:11:06:56`: HistGB val=0.6295, balanced=0.5196
- `tier1_GBP_JPY.log:10:19:28`: HistGB val=0.6380, balanced=0.5156
- `tier1_EUR_GBP.log:10:46:47`: HistGB val=0.7540, balanced=0.5094
- `m15_pair_expansion.log:20:20:35`: HistGB val=0.8227, balanced=0.5006
- `m15_pair_expansion.log:23:00:30`: HistGB val=0.9521, balanced=0.5017

These are **majority-class predictions on class-imbalanced labels** (in label-skewed M15 windows, predicting "no significant move" → 80% val_accuracy with balanced_acc=0.50). Reading them as 60%+ "edge" is the metric-misread; the model has no edge.

## Configs that produced the genuinely-best results

| Setting | Best-by-balanced run (`bu48swr6`) | Operator's control-plane default | Notes |
|---|---|---|---|
| Granularity | H1 | M15 | Mismatch. Sweeps were H1; control plane is M15. |
| Lookahead bars | 24 | 12 | Mismatch. 24-bar H1 ≈ 24 hours; control plane is 12-bar M15 ≈ 3 hours. |
| Candles | 25,000 | (no explicit; cache shows 64,801 for M15 65k) | Mismatch. |
| Learning rate | 9.48e-4 | 3e-4 | Mismatch (3× higher). |
| Dropout | 0.4723 | 0.4 | Close. |
| Epochs (config) | 200 (early-stopped at 21) | 50 | Sweep gave more headroom; control plane caps at 50. |
| Batch size | 64 | (not in control-plane) | — |
| transformer_d_model | 8 | (16 hardcoded per docstring in `direction_training_config.json:5`) | Sweep used 8, doc says 16. |
| transformer_num_layers | 2 | 1 (docstring) | Mismatch. |

**The control plane has not been used in any of the 129 logged wandb training runs.** All recorded transformer trainings ran sweep configs or autonomous-retrain configs. Confidence: HIGH on "wandb runs never used control plane"; MEDIUM on "no run anywhere used control plane" because trainer-log entries don't always cite their config provenance.

## Open follow-ups (for operator)

1. **Reconcile control plane vs sweeps.** Either the operator's M15+lookahead=12 control plane is stale (sweeps prove lookahead=24 H1 was where best val emerged) or the sweeps were exploratory and shouldn't be retrained against. State the decision and re-run with the chosen config.
2. **Stop reading HistGB `val_accuracy` as "edge"**. Promote `balanced_acc` to the headline metric in autonomous-retrain gates so a `val_acc=0.95 / balanced=0.50` artifact never ships again.
3. **Quarantine USD_JPY active model.** val=0.5963 / train=0.8285 / gap=23pp violates current `max_acceptable_gap=0.12`. Most likely promoted before the gate. Either re-promote with documented exception, or quarantine + retrain on the post-2026-05-08 contract.
4. **Investigate the AUD_JPY/EUR_AUD/NZD_USD triplet.** All three have identical metrics (val=0.5827, bal=0.5190, train=0.5210). Either a broadcast bug or per-pair training never actually happened for two of them.
5. **GBP_CHF anomaly.** val=0.5000 (chance) but balanced=0.7755 — investigate whether labels are reversed, classes are heavily imbalanced, or the metric calculation is wrong.
6. **Disk space.** 92% used; joblib already failing. Will block any retrain attempt soon.

## Sources cited

- 129 wandb run directories: `wandb/run-*` (20 with transformer val_accuracy, 109 retrain-orchestration only)
- 2 wandb sweeps: `wandb/sweep-s4wcqfle/` (10 USD_JPY runs), `wandb/sweep-vbq2krwq/` (10 EUR_JPY runs)
- 11 active model meta files: `trained_data/models/*/transformer_direction.meta.pkl`
- 12 quarantined model meta files: `trained_data/models/*/_quarantine/transformer_direction-*/transformer_direction.meta.pkl`
- 7 training logs: `trained_data/logs/*.log`
- Control plane: `src/training/training_defaults/direction_training_config.json`
- Disk: `df -h /Users/buddy/Documents/ml_engine` = 92% used, 18Gi free
