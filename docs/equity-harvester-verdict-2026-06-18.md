# Equity-Beta Harvester — Verdict (2026-06-18)

**The deployable answer to "get out of the 50% coin flip."** It makes no directional prediction;
it harvests the equity risk premium and manages the tail with a causal vol-management + drawdown
overlay (Moreira-Muir 2017). `scripts/build_equity_harvester.py`. Research/shadow only — NOT wired
to live.

## Result — clears the ship gate full-cycle (no mandate change needed)

Core = equal-weight 9 SPDR sectors; overlay = scale exposure inversely to trailing vol + de-gross in
drawdowns. Full cycle 1999-2026 (dot-com tail, GFC, COVID, 2022), cost-aware (2bps):

| book | net Sharpe | maxDD | CAGR | gate (Sh≥0.40 & DD≤25%) |
|---|---|---|---|---|
| SPY buy & hold | 0.531 | 55.2% | +8.7% | fails DD |
| EW-sectors buy & hold | 0.584 | 52.5% | +9.4% | fails DD |
| **Managed vol10% dd15/25** | **0.596** | **22.6%** | **+5.4%** | **PASS** |

The overlay halved drawdown (55%→23%) AND lifted Sharpe above buy-hold (0.60 vs 0.53). Unlike the
FX carry harvester (Sharpe 0.63 but 36% DD — needed a relaxed mandate), this **fits the existing 25%
gate as-is.**

## Honest caveats (do not skip)
- It is **BETA, not alpha** — the return is the equity risk premium, risk-managed. Not ML stock-
  picking. Transparent and simple; the ML sophistication adds nothing here.
- CAGR 5.4% < buy-hold 8.7% — you trade nominal return for a far smoother ride. The 5.4% is
  conservative (0% cash yield assumed; real T-bills add ~1-2%/yr and raise Sharpe).
- Long-only; in prolonged high-vol bears it sits in cash (earns ~0, real: T-bills) — capital-
  preserving, not return-generating, in those regimes.
- Backtest, not live: deployment needs an equity broker (the partially-wired IBKR client, not OANDA
  FX), shadow validation, and the usual fail-closed discipline. Overlay reaction lag = real slippage.

## Why this is the real escape
Prediction is dead in every market tested (FX 52%; FX + equity factor long-shorts both fail). The
equity risk premium is the strongest harvestable premium (FX spot has none — zero-sum). This engine
clears the gate through every crash since 1999 without forecasting anything. The FX bot's machinery
(vol-target + DD overlay, sizing, guards) transfers directly.

## Remaining upside (the one untested ALPHA shot)
Whether genuine cross-sectional ALPHA exists — value/quality and especially PEAD (post-earnings
drift) on single stocks — is untested. Price-only factor alpha is dead (equity long-short = -0.15
Sharpe). Fundamental/earnings alpha has a far stronger prior and needs the financialdatasets.ai free
API key (`FINANCIAL_DATASETS_API_KEY` in `.env.local`). That is the only path left to an ML edge
beyond risk-managed beta.

## Status / next
- BUILT + validated full-cycle: `build_equity_harvester.py`, artifact `equity_harvester_*.json`.
- Deployable as risk-managed equity beta (fits 25% gate). Next: IBKR broker wiring + shadow.
- Fundamental-alpha test: blocked on the free API key (operator action).
