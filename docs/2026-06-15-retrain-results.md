# Retrain results — 2026-06-15 (price-only transformer_direction, H1)

Operator request: "train the models with proper existing training." Ran the canonical
pipeline (`scripts/train_single_model_m1.py --model transformer`, H1) on locally-cached
candle data (no OANDA fetch — no credentials in this environment). 4 of 7 per-pair models
trainable from local data; GBP_JPY / EUR_GBP / AUD_NZD have no local H1 cache and were skipped.

Metric of record is **calibrated balanced accuracy** (chance = 0.50), not raw val_accuracy.
The 10% hard ship-gate (`HARD_MAX_GAP`) was active and enforced.

| Pair | train_acc | val_acc | balanced_acc | gap | ship gate |
|------|----------:|--------:|-------------:|----:|-----------|
| EUR_USD | 0.701 | 0.535 | **0.562** | 16.6% | **QUARANTINED** (>10% gap) |
| USD_JPY | 0.554 | 0.562 | **0.512** | 0.8% | passed (but ≈ chance) |
| USD_CAD | 0.518 | 0.545 | **0.526** | 2.7% | passed (barely above chance) |
| AUD_USD | 0.557 | 0.599 | **0.627** | 4.2% | passed — see caveat |

## Read (calibrated)

- **3 of 4 reproduce the documented ceiling.** EUR_USD/USD_JPY/USD_CAD land at balanced
  accuracy 0.51–0.56, consistent with the ~52% price-only intraday ceiling. EUR_USD overfit
  to a 16.6% gap and was correctly auto-quarantined; its prior (stale, at-chance) live model
  was moved to `_quarantine/` and **not** replaced — EUR_USD now has no shippable model. This
  is the truthful outcome, and the gap-gate worked exactly as designed.
- **AUD_USD (balanced 0.627) is an outlier and is NOT trusted (confidence: LOW).** It is a
  single temporal train/val split, not walk-forward OOS. `val_acc (0.599) > train_acc (0.557)`
  is a yellow flag — typically a favorable/easy validation window, not generalization. Best
  epoch was 3 (very early). The previously-shipped AUD_USD model scored balanced 0.528; a jump
  to 0.627 from the same price-only pipeline should raise suspicion, not confidence (cf. the
  prior leakage-artifact incident where one pair looked good and wasn't). **Do not act on this
  number until it is confirmed by purged walk-forward k-fold** (`walkforward_validation.py`).

## Not done / blocked

- **OANDA trade-data pull for agent training: not possible here.** No OANDA credentials in
  this environment, and conceptually OANDA stores fills/P&L only — not the per-agent verdicts
  the neural-agent trainer needs. The only real history is ~16 losing demo trades, far below
  the ≥50-samples-per-agent floor. Neural agents cannot be trained from OANDA history; the
  path is to run the (retrained) system in demo long enough to generate verdict-tagged outcomes.

## Next step if pursued

Run `walkforward_validation.py` on AUD_USD specifically. If balanced accuracy holds ≥ ~0.54
across folds with a CI excluding 0.50, that is the first price-only result worth a second look.
If it collapses to ~0.52 (expected), the four-approach no-edge conclusion stands.
