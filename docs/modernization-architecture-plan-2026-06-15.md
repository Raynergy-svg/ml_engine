# ML Engine — Modernization Architecture Plan (checked vs SOTA 2026)

**Date:** 2026-06-15 · **Status:** Plan for review
**Operator claim under test:** "~65%+ is reachable; 52% was bad model + inference logic."
**Framing:** This plan gives that claim its best real shot — structured so the cheapest
disconfirming test runs first, against the same leak guards that produced the 52% number,
so we cannot re-discover the OBV-anchor trap (70%→52% collapse, commit dad8624).

---

## 0. Verdict up front (honest, calibrated)

- **65% on raw M15 direction from price alone: LOW probability** (MEDIUM confidence). The 52%
  is a *validation* metric measured in-training ([transformer_trainer.py:2153](../src/training/trainers/transformer_trainer.py), `monitor="val_accuracy"`),
  before any inference code runs. Inference bugs make *live* worse than val; they do not cap
  *val* at 52%. Model scale raises train-acc and the train/val gap, which the 10% hard-gap gate
  quarantines. Scaling the backbone is therefore not a lever on val.
- **60%+ on a re-framed target (triple-barrier meta-label) over a traded subset, with non-price
  features: PLAUSIBLE and SOTA-supported.** This is the legitimate reading of "65%" — it is a
  *trade-quality* number, not a raw-direction number, and it is enough for positive expectancy if
  it clears costs. The plan targets this.

The levers that actually move the headline number, cheapest-to-disconfirm first: **(A) target →
(B) features → (C) backbone → (D) regime selectivity.** Backbone is third on purpose: a modern
architecture *multiplies* signal that exists; it does not create signal that doesn't.

---

## 1. SOTA 2026 reference (what we're checking against)

| SOTA family | Examples (2026) | What it optimizes | Relevance here |
|---|---|---|---|
| Time-series foundation models | TimesFM-2.5 (200M, 16k ctx), Chronos-2, Moirai-2, PatchTST-FM | Forecast MAE/RMSE, zero-shot | Repo already has `src/sota_core/timesfm_adapter.py`. Use as **frozen feature extractor**, not as a direction oracle — these win on MAE, direction stays ~50–53%. |
| Patch / inverted transformers | PatchTST, iTransformer | Long-horizon multivariate forecast | The right backbone *if* Phase A/B show signal: patching + variate-attention beats the bespoke d_model=16 stem. |
| Meta-labeling | López de Prado, *Advances in Financial ML* | Trade / no-trade precision on triple-barrier labels | **The actual SOTA for weak-signal → trade conversion.** Already in stack: `meta_labeling.py`, `meta_labeler_retrainer.py`. Underexploited. |
| Probabilistic / uncertainty | Lag-Llama, conformal prediction | Calibrated intervals | Composes with existing abstention + confidence calibration. |

**Honest read of the 2025 directional-FX literature:** credible papers emphasize leakage
prevention (expanding-window, strict holdout, purged CV) because the field's high accuracy claims
are usually leakage. Reported *regression* wins (e.g. −24% MAE on EUR/USD) do not translate to
65% direction. No credible source reports sustained 65% OOS direction on liquid majors from price.

---

## 2. The plan — four phases, falsification-first

Each phase: pre-register the metric + baseline, run the cheapest test on **one pair**, gate on a
**real OOS number** beating the price-only baseline under walk-forward, then decide go/kill BEFORE
building further. No phase starts before the prior phase's number exists.

### Phase A — Re-frame the target (cheapest; highest SOTA support; ~days)
**Hypothesis:** the predictable thing is not raw next-bar direction; it is *whether a triple-barrier
trade (TP before SL within horizon) resolves favorably*. Meta-labeling raises the headline because
it measures an easier, more autocorrelated target.
- Reuse `meta_labeling.py` triple-barrier labels + the XGBoost meta-labeler.
- Base model stays the tiny transformer (or even HistGB) as the *primary* signal; the meta-labeler
  decides trade/skip.
- **Gate:** meta-label OOS precision ≥ 58% on the traded subset (purged k-fold + embargo), with
  the subset large enough to trade (≥20% of bars). FAIL = direction reframing doesn't help here.
- **Why first:** zero new data, zero new architecture, reuses existing code. If 65% is "real," the
  cheapest place it shows up is here.

### Phase B — Expand beyond price (real signal candidates; ~1–2 weeks)
**Hypothesis:** the missing signal is in non-price data, not in model capacity.
- Add feature blocks, each ablated independently for OOS lift over the price-only baseline:
  1. **Cross-sectional** relative strength / rank across the 7 USD pairs (not price-only by
     construction; this is where factor premia live).
  2. **Order-flow / positioning** — OANDA order+position book (`order_flow` agent already exposes
     `pb_*` graded features) and CFTC COT if free.
  3. **Realized-vol & vol-of-vol** term structure (vol is genuinely autocorrelated → predictable).
  4. **Carry/trend factor scores** as features (shared with the factor portfolio data layer).
- **Explicitly excluded:** synthetic/diffusion data (cannot add signal a generator was fit without),
  premium tick data as a v1 dependency (defer to its own kill-ruled experiment).
- **Gate:** at least one block adds ≥3pp OOS lift over price-only, ablation-confirmed. FAIL on all
  blocks = signal is not in the data we can access; stop here and the factor pivot is the answer.

### Phase C — Modern backbone (ONLY if A or B shows signal; ~2–3 weeks)
**Hypothesis:** given real signal from A/B, a modern encoder extracts more of it than d_model=16.
- Replace the bespoke CNN-stem transformer with a **patch-based encoder** (PatchTST/iTransformer
  style): patch embedding, RoPE, SwiGLU FFN, pre-LayerNorm, variate-attention for the multivariate
  feature blocks from Phase B.
- **And/or** inject **TimesFM-2.5 embeddings** (`timesfm_adapter.py`) as frozen features — foundation
  model as representation, not oracle.
- M1 Metal: mixed precision (`mixed_float16`) + gradient checkpointing; keep model small enough that
  the **10% gap gate** stays satisfiable (capacity sized to the signal, not maxed).
- **Gate:** beats the Phase-A/B baseline OOS by a pre-registered margin with gap ≤ 10%. A bigger
  model that only widens the gap is an automatic quarantine — that is the gate working, not a bug.

### Phase D — Regime-conditioned selectivity (~1 week)
**Hypothesis:** accuracy is regime-dependent; trade only where conditional OOS accuracy clears a bar.
- Use the existing TCN volatility regime head to condition; emit a signal only in regimes whose
  conditional OOS accuracy ≥ threshold; abstain elsewhere (composes with `direction=None` fail-closed).
- **Gate:** traded-subset expectancy positive net of costs in walk-forward.

---

## 3. Guardrails — non-negotiable (so we don't relive the OBV trap)

- **Pre-registration:** metric + baseline + traded-subset size written down *before* each phase runs.
- **Same leak discipline that produced the 52%:** window-invariance canary
  (`tests/test_feature_window_invariance.py` style), purged k-fold + embargo, next-bar fills,
  `FEATURE_PIPELINE_VERSION` bump on any feature-set change.
- **10% hard-gap ship gate stays.** No exceptions without operator-signed commit.
- **No-mock tests, atomic artifacts, no LLM in the decision path** — unchanged project rules.
- **Real number before the next phase.** No "should work." Each phase ends in a `RESULT:{...}` line
  with OOS metric, baseline, lift, gap, traded-subset size.

---

## 4. Honest expectation & relationship to the factor pivot

- The single most likely *good* outcome is **Phase A or B producing a 58–62% meta-label precision on
  a traded subset** — a legitimate, costs-clearing edge, and the honest realization of "65%."
- Raw-direction 65% price-only remains LOW probability; if Phase A/B both FAIL, that is itself a
  decisive, cheap answer and the daily **factor portfolio** ([prd-fx-factor-portfolio.md](../tasks/prd-fx-factor-portfolio.md))
  remains the evidence-backed direction.
- **These compose, not compete:** meta-labeling (Phase A) can wrap the *daily factor trades* too, and
  Phase B's cross-sectional + carry/trend feature blocks share the factor data layer. The cheapest
  test (Phase A) advances both bets at once.

---

## 5. Load-bearing first step (recommended)

Run **Phase A on one pair (e.g. EUR_USD), one week of effort**: triple-barrier + existing meta-labeler,
walk-forward, produce the OOS precision number. It is the cheapest experiment that could reveal a real
edge, reuses code that already exists, and either produces an encouraging number that justifies Phases
B–C, or cheaply refutes the 65% hope before any architecture is built. That is the right next action —
a real number, not a bigger model.

---

### Sources (SOTA 2026 check)
- [The 2026 Time Series Toolkit: 5 Foundation Models](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- [TimesFM — foundation models in time series forecasting](https://towardsdatascience.com/timesfm-the-boom-of-foundation-models-in-time-series-forecasting-29701e0b20b5/)
- [Directional forecasting for eight USD forex pairs (Springer, 2025)](https://link.springer.com/article/10.1007/s44163-025-00424-4)
- [Dual-input deep learning EUR/USD (MDPI, 2025)](https://www.mdpi.com/2227-7390/13/9/1472)
- [Forecasting intraday volatility with deep learning (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1062976925001176)
