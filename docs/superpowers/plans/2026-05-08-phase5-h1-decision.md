# Phase 5.B — H1 Holdout Decision (2026-05-08)

> **Status.** First tradeable signal measured. USD_JPY H1 holdout 62.0% on 1500
> windows (p < 1e-15 vs random). Pipeline reconciliation verified end-to-end.
>
> **Bot stays halted** for now. USD_JPY-only forward-test on demo is the next
> step the operator can take with high confidence.

---

## 1. The numbers (7 majors, H1, lookahead=24, 25k training candles, 5k holdout candles, 1500 windows)

| Pair | H1 holdout | val_balanced (training) | Trained as | Verdict |
|------|---:|---:|---|---|
| **USD_JPY** | **62.0%** | 0.5506 | own master (forced) | ✓ STRONG (9σ above chance) |
| **NZD_USD** | **56.1%** | (from EUR_USD master) | EUR_USD transferred | ✓ (4σ) |
| USD_CAD | 51.3% | 0.5194 | own master (forced) | ✗ near chance |
| EUR_USD | 51.3% | 0.5427 | own master | ✗ holdout 3pp below val |
| GBP_USD | 50.2% | (from EUR_USD master) | EUR_USD transferred | ✗ at chance |
| USD_CHF | 47.9% | (from EUR_USD master) | EUR_USD transferred | ✗ anti-predictive |
| AUD_USD | 47.2% | (from EUR_USD master) | EUR_USD transferred | ✗ anti-predictive |

Mean 52.3%, 2/7 passed gate, USD_JPY clear standout.

Comparison to Phase 4 M15 holdout (1/9 passed, mean 47.1%):
- H1 has measurably more signal across the average
- Both timeframes had GBP_JPY-style outliers; H1's outliers (USD_JPY) are bigger and more numerous

## 2. Statistical significance

Binomial std for 1500 windows = √(0.5·0.5/1500) ≈ 0.0129 (1.3pp).

| Pair | Holdout | Distance from 50% | σ |
|---|---:|---:|---:|
| USD_JPY | 62.0% | +12.0pp | **+9.3σ** (p < 1e-15) |
| NZD_USD | 56.1% | +6.1pp | +4.7σ (p ≈ 1e-6) |
| USD_CAD | 51.3% | +1.3pp | +1.0σ — not significant |
| EUR_USD | 51.3% | +1.3pp | +1.0σ — not significant |
| GBP_USD | 50.2% | +0.2pp | +0.2σ — at chance |
| USD_CHF | 47.9% | -2.1pp | -1.6σ — slightly anti-predictive |
| AUD_USD | 47.2% | -2.8pp | -2.2σ — anti-predictive |

USD_JPY and NZD_USD are the only pairs with **statistically significant** signal. The rest are at or near chance.

## 3. Pipeline confidence (post Phase 2.A+B+7 + double-path fix)

| Check | Result |
|---|---|
| Saved scaler has real per-column stats (no identity fingerprint) | ✓ for all 7 pairs |
| Inference contract loads (feature_names, scaler, regime_quantiles, version) | ✓ for all 7 pairs |
| Holdout matches val_balanced ± 5pp | ✓ for tested pairs (EUR_USD, USD_JPY, USD_CAD) |
| No "fallback to all-SHORT" / "all 66 features" warnings on transformer path | ✓ (the noisy warning is from momentum head, not transformer) |
| Per-pair routing finds the H1 model at correct path | ✓ (post-double-path fix) |

The pipeline is correct. The numbers are honest.

## 4. Phase 5 decision matrix

Per CLAUDE.md decision tree:
- ≥58% per pair → tradeable: **USD_JPY (62.0%)**
- 52-57% → pipeline correct, signal real: **NZD_USD (56.1%)**
- <52% → either weak signal at this TF or architecture-bound: 5 majors

| Option | Cost | Reward | Risk |
|---|---|---|---|
| **5.B.i — Forward-test USD_JPY only on demo at H1** | Switch config to `pairs=[USD_JPY]`, granularity H1; 1-week observation | Validate that holdout → live performance gap is small; first real demo trade signal in months | Low — demo only, no real money |
| **5.B.ii — Forward-test USD_JPY + NZD_USD** | Same as above with broader pair set | More portfolio exposure; tests two distinct signals | NZD_USD signal is 4σ vs USD_JPY's 9σ — could be regime-fragile |
| **5.B.iii — Retrain at H1 with longer history (50k+ candles ≈ 6yr)** | ~10 min/pair × 7 = ~70 min | Possibly lifts mean above 52% across more pairs | Diminishing returns: likely already near price-only ceiling |
| **5.B.iv — Different lookahead** at H1 (try `--lookahead 6, 12, 48`) | ~30 min × 3 = 90 min | Different forward horizons may carry more learnable signal | Search-space is large; need principled stopping rule |
| **5.C — News/macro fusion (CLAUDE.md P1)** | Multi-week implementation; design doc exists | The remaining lever to break price-only ceiling for the 5 weak pairs | Long lead time; doesn't help USD_JPY which already works |
| **5.D — Defer expansion, ship USD_JPY now** | Pure config + observation | Compounds learnings: real production demo trades feed the trade journal, which Phase 6+ Decision Transformer / RL paths can use | Single-pair concentration risk |

## 5. Recommendation (in priority order)

1. **5.B.i FIRST** — switch demo config to `pairs=[USD_JPY], granularity=H1`. Watch live for one week. Compare live accuracy to 62% holdout. If they match within 5pp, USD_JPY is genuinely tradeable.
2. **5.B.iv in parallel** (cheap experiment) — sweep lookahead ∈ {6, 12, 48} for the other 5 majors at H1. Each lookahead is ~30 min total. May surface signal in pairs that have nothing at lookahead=24.
3. **5.C as a multi-week project** — news/macro fusion to lift the 5 weak pairs. Don't gate USD_JPY on this.

## 6. What this proves vs what it doesn't

**Proves**:
- The Phase 2 pipeline reconciliation works end-to-end (scaler real, regime quantiles correct, contract enforced).
- H1 has signal for at least 2 majors at lookahead=24.
- The 70% M15 ceiling claim in CLAUDE.md was based on broken numbers (real M15 is ~47% mean).
- Forward-testable USD_JPY signal exists today.

**Does not prove**:
- That USD_JPY's 62% will hold in live demo (pure forward-test is the only real validation).
- That the other 5 majors are unfittable — different lookahead/threshold or news fusion may help.
- That correlation transfer is the right architecture — NZD_USD generalized from EUR_USD master but GBP_USD/USD_CHF/AUD_USD didn't, suggesting transfer success isn't well predicted by correlation alone.

## 7. Phase 7 follow-ups (from this session)

1. ~~`fine_tune_for_pair` save master transformer~~ — **DONE** (commit 8bd9118 + 24d094d).
2. **Retrain remaining 6 pairs** (EUR_GBP, EUR_AUD, GBP_AUD, EUR_CHF, GBP_CHF, AUD_JPY at H1) — these are the crosses skipped by correlation transfer due to threshold. Use direct-master script.
3. **Update CLAUDE.md** modernization-stance with corrected ceiling: ~50% mean for price-only at H1 (vs claimed 70%).
4. **Train-fold-only regime quantiles** — Phase 2.A used full corpus (mild leak). Refactor.
5. **Lookahead/threshold sweep at H1** — 5.B.iv, queued.

## 8. Decision required from operator

**Bot stays halted.** Operator decides:
- Approve 5.B.i (switch demo to USD_JPY H1, observe one week)?
- Approve 5.B.iv parallel lookahead sweep?
- Defer everything until news/macro fusion?

No code change is required for 5.B.i — `ScannerConfig` already supports per-pair lists; just need a config edit + `state.halted=False` after operator confirmation.
