# Experiments Log

> Pending / dated experiment plans relocated out of `.claude/rules/improvement.md` (2026-06-09)
> so the always-loaded rules file holds only promoted *rules*, not one-off experiment runbooks.

## 2026-05-18 — HistGB capacity-shrink (PENDING: run after disk freed)

Context: 2026-05-13 post commit `983892a` (HistGB train_acc honest-reporting fix), USD_JPY
M15/25k smoke produced `train=96.11% / val=48.82% / gap=47.29%`. The gap is real (not a
metric bug), driven by HistGB's default capacity being too high for forex M15 features.
Capacity-shrink change shipped to `src/training/trainers/histgb_trainer.py` on `main`
(defaults only; `TrainerConfig` getattr overrides still supported for future sweeps).

Change summary (old → new): `max_iter` 200→100 · `max_depth` 8→4 · `learning_rate` 0.05→0.03
· `l2_regularization` 0.1→1.0 · `min_samples_leaf` 20→50 (explicit) · `max_leaf_nodes` 31→15
(explicit) · `n_iter_no_change` 20→10 · `validation_fraction` 0.15→0.2.

Validation plan (do NOT run under ENOSPC — model-save would risk corrupting a per-pair
`histgb_direction.pkl` that gate routing reads):

```
python scripts/train_single_model_m1.py --pair USD_JPY --model histgb_direction --granularity M15 --candles 25000
```

Expected RESULT shape: `train_acc=0.58-0.65, val_acc=0.50-0.55, gap<0.10, quarantined=false`.

Decision rules:
- `gap < 0.10` AND `val_acc > 0.51`: SHIP; confirm on EUR_USD + GBP_USD, then remaining majors.
- `gap < 0.10` AND `val_acc ≈ 0.50`: capacity-shrink fixed the gap but features have no signal at this scale (coinflip). HistGB ships only as an ensemble baseline; news/macro P1 is the next lever.
- `gap > 0.10` AND `< 0.20`: single-knob nudge — drop `max_iter` to 60 OR `max_depth` to 3. Re-run.
- `gap > 0.20`: capacity still too high — halve `max_iter` to 50, `max_depth=3`, `max_leaf_nodes=8`. If still >0.20, suspect a leakage issue in `compute_normalized_features` (rolling stats including the target bar).

Reverse plan: every changed knob is a single literal swap in the `histgb_trainer.py` dataclass-default block; revert with one `git revert` on the experiment commit.

Confidence: HIGH the gap drops below 20% (capacity cut on 5 axes); MEDIUM it drops below 10%
first try; LOW-MEDIUM `val_acc` materially improves (HistGB on price-only forex M15 has a
~50% ceiling — capacity-shrink lowers train_acc, not necessarily raises val_acc); UNKNOWN
whether the same hyperparams generalize across all majors (need EUR_USD + GBP_USD smokes first).
