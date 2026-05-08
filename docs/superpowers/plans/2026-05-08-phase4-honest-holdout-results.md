# Phase 4 — First Honest M15 Holdout Results (2026-05-08)

> **Status.** Pipeline reconciliation is empirically successful. Inference path correctly applies the saved scaler + regime quantiles to fresh OANDA data. Holdout numbers are now honest. **They're also much weaker than the broken pipeline claimed.**
>
> **Bot stays halted.** No pair has tradeable signal under the 52% gate.

---

## 1. Numbers (9 pairs evaluated, 1500 windows each, lookahead=24, M15)

| Pair | Holdout accuracy | LONG calls | SHORT calls | Verdict |
|------|---:|---:|---:|---|
| **GBP_JPY** | **52.7%** | high | low | ✓ passes 52% gate (transferred from EUR_JPY master) |
| EUR_JPY | 51.1% | — | — | ✗ |
| USD_CAD | 50.7% | — | — | ✗ |
| USD_CHF | 49.9% | — | — | ✗ |
| EUR_USD | 44.9% | — | — | ✗ master, weakest predictor |
| NZD_USD | 44.7% | — | — | ✗ |
| GBP_USD | 44.6% | — | — | ✗ |
| USD_JPY | 43.7% | — | — | ✗ |
| AUD_USD | 41.6% | — | — | ✗ |

**Mean: 47.1%. 1/9 passes.** Report at `trained_data/m15_per_pair_eval.json`.

## 2. What the numbers mean

**Pipeline is verified end-to-end.** Pre-Phase-2 holdouts reported "70%+" because every prediction was the all-SHORT fallback predictor on a SHORT-heavy test slice. Post-Phase-2, the trained transformer actually runs — and what it predicts is barely above chance for most pairs. The 52% gate that "passed" pre-Phase-2 was rubber-stamping the fallback, not the model.

**These are the first honest measurements.** The match between EUR_USD's `val_balanced_accuracy=0.4897` and its holdout (44.9%) is within ~5pp — pipeline works correctly; the model just doesn't have signal at this combination of (M15, lookahead=24, 65k candles, current architecture).

**Most pairs *underperform random.*** Mean 47.1% vs 50% chance means the trained model is mildly anti-predictive on out-of-sample data. The likely cause is a combination of:
- Small, noisy training data (~12k labeled samples after threshold filtering)
- 60-bar context window — only 15 hours of M15 history
- 24-bar lookahead — 6 hours forward at M15 — long horizon, weak signal
- Class imbalance and prediction bias (the 91% LONG / 9% SHORT split on GBP_JPY is suspicious)
- 22-month corpus may be regime-mixed (training distribution ≠ recent live distribution)

## 3. CLAUDE.md modernization claim was based on broken numbers

CLAUDE.md says: _"The price-only direction-prediction holdout has plateaued at ~70.0% on M15 EUR_USD/GBP_USD across architectures"_ and _"News/macro fusion is the remaining lever to break 70%."_

The 70% number is invalid. With honest measurement:
- EUR_USD M15 lookahead=24: 44.9% (was claimed 70%)
- GBP_USD M15 lookahead=24: 44.6% (was claimed 70%)

The price-only ceiling on M15 with current architecture is around **45-52%**, not 70%. News/macro fusion is still likely the right lever, but the gap to climb is ~10pp larger than CLAUDE.md represented.

**Action item**: update CLAUDE.md modernization-stance block with the corrected ceiling.

## 4. Phase 5 decision

Per CLAUDE.md decision tree:
- ≥58% per pair → tradeable. **No pair qualifies.**
- 52-57% → pipeline correct, price-only ceiling. **Only GBP_JPY (52.7%) — n=1 is not statistically significant; likely random variance.**
- <52% → deeper architecture issue or pivot to news fusion. **8/9 pairs.**

**Verdict: do not unhalt.** Current models too weak.

**Recommended next steps (operator decides)**:

| Option | Cost | Reward |
|---|---|---|
| **5.A — Lookahead/threshold sweep** at M15. Try `lookahead ∈ {6, 12}`, `threshold ∈ {0.0005, 0.001, 0.0015}`. Cheaper signal capture; smaller forward horizon may be more learnable. | ~30 min × 6 combinations × 2 master pairs = 6h | Possibly +5-10pp if shorter horizon has more signal |
| **5.B — Switch trading TF back to H1**. Built-in H1 holdout from tonight's retrain hit 53.8% (vs M15 48.3%). H1 was abandoned earlier on broken-validation grounds; may actually be the right TF. | ~10 min config change, 2h retrain at H1 | Tradeable signal possible; bot can resume H1 strategy |
| **5.C — News/macro fusion (CLAUDE.md P1)** | Multi-week implementation; design doc exists at `docs/superpowers/plans/2026-05-08-news-macro-signal-design.md` | The actual remaining lever; may add 5-15pp if data alignment is solid |
| **5.D — Decision Transformer / offline RL** on trade journal | Needs >5K trades first per CLAUDE.md; deferred | Long-term lever; not now |

**Recommendation order (load-bearing-question first)**:
1. **5.A first** — cheapest experiment that distinguishes "M15 has signal at shorter horizon" from "M15 has no signal at any horizon". Single-day work.
2. **5.B in parallel** — cheap config change; the H1 number from tonight (53.8%) is the most encouraging signal we have.
3. If 5.A and 5.B both confirm "price-only is near-chance regardless of horizon/TF" → move to 5.C with the corrected ceiling claim.

## 5. Phase 7 follow-ups (queued)

1. **Fix `fine_tune_for_pair`** — currently only fine-tunes LGBM heads (Momentum/Risk/Confidence). Transferred pairs end up with stale transformer artifacts. Phase 3 needed a manual `cp` of master's transformer to each transferred pair dir. The fix is to save the master's `transformer_direction.{keras,meta.pkl,weights.h5,arch.json}` to `pair_save_dir` inside `fine_tune_for_pair`. (`src/training/trainers/joint_trainer.py:736`)
2. **Retrain 6 missing pairs** — AUD_JPY, EUR_GBP, EUR_AUD, GBP_AUD, EUR_CHF, GBP_CHF. Correlation < 0.7 to existing masters → orchestrator skipped them. They need separate forced-master retrain. Or: invoke with `--pairs <pair>` per skipped pair.
3. **Train-fold-only regime quantiles** — Phase 2.A captures quantiles over the full corpus (mild test/val leak). Refactor to use train_idx-only.
4. **Update CLAUDE.md** — replace `70.0% M15 EUR_USD/GBP_USD` ceiling claim with the corrected `45-52%` figure once Phase 5 settles which architecture is the new baseline.

## 6. What this proves

| Question | Answer |
|---|---|
| Does the Phase 2 pipeline reconciliation actually work? | **Yes, empirically.** Saved scaler has real per-column stats. Regime one-hots reproduce at inference from saved quantiles. Pipeline-version assertion fires correctly. |
| Are the holdout numbers now honest? | **Yes.** Match val_balanced_accuracy ± ~5pp. Pre-Phase-2 numbers were the fallback predictor. |
| Can the bot trade with current artifacts? | **No.** 1/9 pairs above the 52% gate. |
| Is the issue model architecture or data? | **Both, plus weak signal.** Validates the CLAUDE.md modernization roadmap (news fusion as P1) but resets the ceiling assumption. |
| Should we pursue news fusion? | **Likely yes, after cheaper 5.A/5.B experiments.** |
