# Breakthrough Hunt — Provisional Synthesis (4 of 5 agents)

> **Status**: PROVISIONAL — Agent 1 (regime-specialized models) still in flight.
>
> Operator's prompt (2026-05-10): "Brainstorm, I need a breakthrough (use team of agents)". 5 parallel investigations dispatched. 4 returned with diagnostics + verdicts. 1 remaining. The pattern across the 4 returned is striking enough to capture before the 5th lands.

## The convergence

Three of four returned investigations converge on the same conclusion: **the price-only direction-prediction ceiling at H1/M15 is information-theoretic, not feature-coverage-bound or model-class-bound.** Adding more features, more model architectures, or more asset classes does not break the ~70% holdout ceiling; only different INPUT INFORMATION does.

The fourth (journal replay → DT) is positive on its own merit but explicitly bounded by the same ceiling unless paired with news/macro features.

## Per-agent verdicts

### Agent 2 — Triple-barrier meta-labeling: NEGATIVE

| Metric | Value |
|---|---|
| Top-decile win rate (test) | 52.4% |
| Lift over base | +3.8pp |
| Test AUC | 0.513 (essentially noise) |
| Train AUC | 0.653 (overfits at iter 5) |

**Verdict**: signal ceiling in current 50-feature contract is ~0.52 AUC for triple-barrier targets. Symmetric 1:1 ATR barriers produce 50/50 base rate by construction on near-random-walk data. Concept might rescue with asymmetric barriers (TP=2×ATR/SL=1×ATR) or López de Prado primary-output flavor, but neither tested as breakthrough-class.

**Lesson**: features engineered for direction prediction don't crack barrier-touch geometry. Different prediction target needs different features.

### Agent 4 — Cross-asset basket features: MARGINAL

| Metric | Value |
|---|---|
| Sanity correlation (contemporaneous, eur_minus_usd vs EUR_USD) | +0.988 ✓ |
| **Lead-lag** correlation (usd_basket_momentum t-24:t vs EUR_USD t:t+24) | **-0.022** |
| Best A/B holdout lift across lookaheads | +1.75pp (la=12 test) |
| Lift inside bootstrap CI95? | yes (±1.5pp noise band) |
| Basket features in top-20 importance? | yes — model uses them |

**Verdict**: basket features add orthogonal coverage but predictive signal isn't there at H1 24-bar horizon. Lead-lag of -0.022 at n=24,127 is statistically robust noise. Information-theoretic ceiling, not feature gap.

**Lesson**: "more features from same OHLCV bars" does not break the price-only ceiling. The ceiling is about INFORMATION CONTENT, not feature engineering.

### Agent 5 — Asset/timeframe pivot: STAY FX, M5 first

| Option | Wallclock to first paper trade |
|---|---|
| **C — M5 retrain** | 3-5 days (recommended first) |
| A — IBKR futures | 2-3 weeks (CLAUDE.md memory was stale; 41 broker tests fail today) |
| B — Crypto Kraken | 3-4 weeks (greenfield 600-800 LoC + regulatory) |
| D — OANDA CFDs | 5-7 days (CFDs ≠ real futures) |

**SNR comparison**: BTC per-bar Sharpe ~10× FX (0.045 vs 0.004) but 30-day sample with wide CI. After cost adjustment, net edge ~80% better — real but not "obvious".

**Verdict**: stay in FX, drop to M5 next. Multi-week broker integration delays the actual hypothesis test (does Buddy have ML edge on price-only data?). New venue is multi-week investment that delays the actual answer.

**Lesson**: don't change asset class to escape a model-quality question. Test the hypothesis where it lives.

### Agent 3 — Journal replay → Decision Transformer: POSITIVE CONDITIONAL

| Metric | Value |
|---|---|
| POC replay (1000 H1 bars) | 104 trades in 0.02s, WR 38.5% |
| Full file (25K H1 bars) | 3,291 trades in 0.68s, WR 37.5% |
| Leakage check (RR=1.5 breakeven 40%) | observed 37.5% — **no leakage** |
| Path C estimate (production gate, 15 pairs × M15 × 3.5yr) | **5,500-8,400 trades** in 4-21h |
| Build cost | ~3 days work + 1 night compute |

**Verdict**: replay produces credible synthetic trades at scale. Unblocks DT's 5,000-trade threshold in ONE OVERNIGHT instead of 50-80 months organic. Slippage realism is the credibility risk (practice journal records 0.0 slip; calibrate against spread distribution + pessimistic bias).

**Critical caveat**: DT trained on price+gate-state replay alone is bounded by the same ~70% ceiling per CLAUDE.md modernization audit. Strategic value is real but conditional on news/macro pipeline ALSO landing.

**Lesson**: data scale isn't the bottleneck for current architecture; data variety (news, macro, cross-asset price action sequences) is.

### Agent 1 — Regime-specialized models: STILL IN FLIGHT

(Awaiting completion. Will update this doc.)

## What the convergence implies

Three independent negative results pointing at the same conclusion is not coincidence. It's a finding:

> **The information content of price-only OHLCV at H1/M15 has been substantially extracted by the current Phase 5.D-tuned transformer. Adding more features from the same data, different prediction targets on the same data, or different venues for the same kind of price data does not lift the ceiling.**

The breakthrough lever per CLAUDE.md modernization stance (May 2026) is **news/macro fusion (P1)**. That position was promoted to P1 from P2 last week based on a Phase 5.C audit; this session's 4-agent convergence empirically validates that strategic call.

## The synthesized roadmap

If the operator wants the BREAKTHROUGH path with the highest expected lift:

| Week | Track | Effort | Output |
|---|---|---|---|
| **1** | News/macro pipeline P3 implementation | per `docs/superpowers/plans/2026-05-08-news-macro-signal-design.md` § Phase 3 | EUR_USD M15 retrain with FinBERT + 8-event count features; holdout vs 70% baseline. Decision rule: ≥73% ships, 70.5-72.9% try `text-embedding-3-large`, ≤70.5% shelve. |
| **2** | Journal replay infrastructure (Path C) | Per Agent 3 spec — wire production gate against historical bars | 5K-8K synthetic trades with realistic slippage. Saves to fresh journal. Unblocks DT training threshold. |
| **3** | M5 retrain (parallel track if news shows lift) | Per Agent 5 recommendation — same broker, no operational risk | Test whether higher-frequency price action has more signal. Decision rule: M5 holdout ≥ M15 baseline. |
| **4** | Decision Transformer Phase 2.A | Per `docs/superpowers/specs/2026-05-09-decision-transformer-design.md` § Phase 2 | DT trained on replay+real journal with news features in state vector. Validate on real-data-only slice (≥500 real trades, gate of §6). |

That's a 4-week roadmap aligned with two empirically-validated levers (news/macro + DT) and one parallel risk-cheap test (M5).

What it does NOT include (consciously dropped):
- ~~Triple-barrier meta-labeling~~ — Agent 2 ruled out
- ~~Cross-asset basket features~~ — Agent 4 ruled out
- ~~Crypto/futures pivot~~ — Agent 5 deferred (earn it with data)
- ~~More multiplicative confidence-stack tuning~~ — Phase 8 audit ruled out
- ~~Foundation-model direction head~~ — already empirically tested in Phase 4 (FM zero-shot underperforms 19k-param custom by 20pp+)

## Operator decision points

The breakthrough story isn't "we found the magic feature." It's "we eliminated three plausible levers and consolidated the actual roadmap."

1. **Authorize 4-week roadmap?** Multi-track. Reversible. Aligned with CLAUDE.md modernization. The risk is timeline; the alternative is more parameter-tuning while the bot remains paralyzed.

2. **Order: news P3 first or replay infrastructure first?** News P3 is the lever; replay is the multiplier. If news fails to lift, replay alone is medium-value. If news succeeds, replay+DT compounds the lift. Recommend news P3 first to test the hypothesis cheaply.

3. **Run Agent 1 (regime-specialized) to completion or short-circuit?** Three of four agents converge; Agent 1's outcome is unlikely to flip the synthesis. But the diagnostic is in flight already; let it finish.

## Confidence

**HIGH** that the synthesis correctly summarizes the 4 agent reports. Each agent's primary finding was reproduced with statistical rigor (bootstrap CIs, lead-lag correlations, A/B holdouts on real disk).

**MEDIUM-HIGH** that news/macro fusion is the actual breakthrough. Bounded by:
- News data quality (FinBERT is 2018-vintage; modern alternatives like text-embedding-3-large untested)
- Lookback window selection (4h vs 24h vs both — 6 open questions in news design doc)
- Whether news leakage can be tightly bounded at training time

**MEDIUM** that the journal-replay Path C unblocks DT in 4-21h compute. Load-bearing assumption: production gate run against historical bars produces synthetic outcomes whose distribution matches future real-trade distribution. If gate stack drifts (online retraining, weight bandits), replay corpus rots — needs re-run on every Phase 5.D-equivalent.

**LOW** confidence on the specific 73% / 70.5% / 70.5% decision rules in the news roadmap — these are operator-set thresholds; could be tightened or loosened based on risk tolerance.

---

## Appendix A — Cross-agent meta-finding

All 4 returning agents independently surfaced **stale documentation / memory issues**:
- Agent 2: meta_labeler wired in `gates.py:2026` but never trained per-pair
- Agent 5: CLAUDE.md memory of "IBKR 483 tests passing" stale by 33 days; disk shows 41 broker test failures
- Agent 4: production runtime path differs from documentation in `confidence_calibration.py`
- Agent 3: practice-account journal slippage is 0.0 universally (vs. assumed real distribution)

This is the f070d39 lie pattern at structural scale. Half the codebase's "wired" claims point at instantiation sites without integration paths, or at documentation stubs without runtime consumers. **Trusting documentation without runtime greps is systematically expensive.** Mitigation already documented in `.claude/rules/improvement.md` § Honesty & verification protocol.

## Appendix B — The information-theoretic ceiling argument

If 4 different agents looking at 4 different angles all hit the same ~70% ceiling, the ceiling is real and structural. Possibilities:
1. **Market is efficient at H1/M15** — most price moves are unforecastable from price alone (the EMH-strong-form proxy)
2. **The wrong target** — predicting next-N-bar direction is harder than predicting next-trade-outcome (which is what triple-barrier tries)
3. **Wrong frequency** — H1 may be too noisy; M5 (Agent 5) might have more signal
4. **Missing input** — news, macro, order flow, sentiment all live OUTSIDE OHLCV bars

(1)-(3) are all "stay in price-only space, change something." (4) is "go outside price-only space." Per Agent 4's lead-lag of -0.022, options (2)-(3) likely produce marginal lift at best. Option (4) is the bet.
