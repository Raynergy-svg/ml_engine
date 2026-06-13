# Daily FX Factor Portfolio — first honest number (2026-06-13)

Implements `tasks/prd-fx-factor-portfolio.md` (US-001…US-007). This is the
operator-approved pivot away from the no-edge intraday stack. Code lives in
`src/factor/`; run with `python scripts/run_factor_backtest.py`.

## What was built

| US | Module | Status |
|----|--------|--------|
| US-001 | `src/factor/data_loader.py` | ✅ real OANDA daily, 7 pairs, 3,229 aligned days (2014-01→2026-06) |
| US-002 | `src/factor/rates.py` + `signals.carry_signal` | ✅ FRED 3m interbank rates → cross-sectional carry rank |
| US-003 | `signals.tsmom_signal` | ✅ causal 3/6/12m TSMOM composite |
| US-004 | `signals.value_signal` | ⏳ pure function ready; BIS REER data layer not yet fetched |
| US-005 | `src/factor/portfolio.py` | ✅ inverse-vol + 10% vol target + hard gross/per-pair guards + weekly no-trade band |
| US-006 | `src/factor/backtest.py` | ✅ next-bar lag, spread + financing + **carry accrual**, walk-forward |
| US-007 | `src/factor/ship_gate.py` | ✅ mechanical PASS/FAIL gate |

All signals are causal (window-invariance tested); all artifacts versioned
(`FACTOR_PIPELINE_VERSION`) and written atomically; 22 no-mock tests green;
`flake8 src/factor` clean.

## The number (net of costs, 13 years, 7 USD majors)

| Book | net Sharpe | gross Sharpe | net CAGR | max DD | +years | turnover |
|------|-----------:|-------------:|---------:|-------:|-------:|---------:|
| carry-only  | −0.09 | **+0.10** | −1.9% | 43% | 6/13 | 5×/yr |
| trend-only  | −0.34 | −0.10 | −4.6% | 51% | 4/13 | 48×/yr |
| **carry+trend** | **−0.22** | **+0.00** | −3.2% | 44% | 4/13 | 38×/yr |

**Ship gate verdict: FAIL** on net Sharpe (need ≥0.40), positive years (need
≥6/10), and max drawdown (need ≤25%).

## What this means (calibrated)

- **HIGH confidence:** at this scale — 7 USD-quote majors, daily, 2014–2026 —
  none of carry, trend, or their combination clears a sane deployable bar. This
  is consistent with, and strengthens, the project's existing "no shippable edge"
  thesis: it now holds for the *strongest-evidence* strategy class, properly costed.
- **HIGH confidence:** trend is genuinely dead on this window (gross −0.10), in
  line with the widely documented "death of FX trend" of the last decade.
- **MEDIUM confidence:** carry has a *faint* positive gross premium (+0.10 Sharpe)
  that the cost stack (2.7%/yr drag) erases. The honest read is "real but too
  small here," not "doesn't exist."
- A backtest bug was found and fixed mid-build: the first version credited price
  moves and charged financing but never credited the **carry actually earned**,
  which made carry look like it lost (−0.18). With carry accrual modeled (PRD
  FR-3), carry's gross flips to +0.10. The negative *net* verdict survives.

## The real levers (evidence-based, not more price-ML)

1. **Cross-section breadth (highest expected value).** All 7 pairs share USD as a
   leg, so the cross-sectional rank is mostly a USD-cycle bet. Real carry/value
   portfolios trade the full G10 cross (AUD_JPY, NZD_CHF, EUR_GBP, …) for breadth.
   This is the single change most likely to move carry's Sharpe up. (PRD OQ-3.)
2. **Lower the bar honestly.** The PRD already states the deliverable at $1k is a
   *verified track record* (~$40–70/yr, not income). A 0.4 Sharpe gate is correct
   for deployment; it is not a referendum on whether to keep researching breadth.
3. **Value factor (US-004).** Diversifies carry crashes; needs the BIS REER layer.
   Won't rescue a ~0.0 gross book alone, but belongs in the full test.

## FP-1 update (2026-06-13) — G10 cross-section breadth tested

Added 12 liquid G10 crosses (no USD leg) so carry/value rank over 19 instruments
instead of 7 USD-only majors. Realistic per-cross spreads modeled (1.6–4.0 pips).

| Universe | net Sharpe | gross Sharpe | max DD | +years | cost drag | verdict |
|----------|-----------:|-------------:|-------:|-------:|----------:|--------|
| 7 majors        | −0.22 | +0.00 | 44% | 4/13 | 2.7%/yr | FAIL |
| 19 (majors+crosses) | −0.25 | +0.01 | 55% | **6/13** | 3.8%/yr | FAIL |

**Read (HIGH confidence):** breadth helped *consistency* — positive years rose
4→6/13 and that gate criterion now passes — but it did NOT create alpha. Net Sharpe
is still negative, drawdown is worse, and the extra cost drag (wider cross spreads +
more turnover) ate the marginal diversification. Decisively: **gross Sharpe is ≈ 0.00
in both universes** — there is essentially no edge *before costs*. That is the
load-bearing finding: the ceiling here isn't costs or breadth, it's that daily
carry+trend on G10 has no gross premium worth harvesting at this scale/window.

Implication for the remaining backlog: FP-3 (turnover/cadence) only helps when gross
is positive — it cannot manufacture alpha from a ~0 gross signal. FP-2 (value) may
add a little diversification but will not lift a 0.00 gross book to the 0.40 bar.
The accumulating evidence (majors FAIL, +crosses FAIL, gross≈0) points the same way
as the project's intraday verdict: **no deployable systematic edge at this scale.**
The honest deliverable is that closed question plus the reusable, correct machinery.

## What did NOT change

Runtime stays halted and fail-closed. No trades. No unhalt. The intraday stack is
untouched (US-009 freeze is a separate operator-signed step). No LLM in any
decision path.
