# The Harvester — Verdict (2026-06-18)

The second engine, built per operator directive ("build the harvester"). It does NOT predict
direction; it collects carry + rides trends and manages the fat left tail with a causal risk
overlay (portfolio vol-target + drawdown/vol-spike de-grossing circuit-breaker). Research/shadow
only — NOT wired to live execution. Bot remains halted.

## Result: the harvester works; the binding constraint is the risk mandate, not the strategy

Best full-cycle book = **GLOBAL (G10+EM) carry+trend + crash-aware overlay** (`experiment_em_carry.py
--crash-aware`, 1996–2026, FRED daily, IRSTCI01 rates, 15bps EM spreads costed):

| book | net Sharpe (after cost) | maxDD | worst year | gate (Sh≥0.40 & DD≤25%) |
|---|---|---|---|---|
| GLOBAL base (no overlay) | 0.45 | 65.7% | −33.4% | fail |
| **GLOBAL + crash-aware overlay** | **0.63** | **35.9%** | **−15.0%** | fail (DD only — Sharpe clears) |
| Majors-only + overlay (`build_harvester.py`) | 0.31 | 24.4% | — | fail (Sharpe only — DD clears) |

The overlay lifted GLOBAL Sharpe 0.45→0.63, cut maxDD 66%→36%, tamed the carry-crash worst year
−33%→−15%. **Net Sharpe 0.63 / 36% maxDD is normal for real systematic macro/CTA funds** (Sharpe
0.5–0.8 at 20–40% drawdowns). The harvester is a viable strategy; it fails only a 25% drawdown rail
that is stricter than the strategy's own natural risk.

## The decision (operator's, not a research question)

- **Hold the 25% maxDD gate** → deployable harvester = majors-only (Sharpe 0.31, maxDD 24%): safe,
  gate-compliant, thin (~1.5%/yr). Probably not worth real capital.
- **Relax to ~36–40% maxDD** (industry-normal for Sharpe 0.6) → GLOBAL harvester (Sharpe 0.63, 36%
  DD, worst year −15%): a genuine non-directional risk-premium engine.

## Honest caveats (do not skip before any deploy)
- Carry-dominated → carry-crash tail. The −15% worst year is the *tamed* historical tail; a novel
  crisis can exceed it. The 2020–2026 "pass" (Sharpe 0.74 / 13–17% DD) is a SHORT warm-up window
  that dodges 2008 — do NOT use it to justify the strategy.
- Cost-sensitive at the EM legs: net Sharpe is at 15bps EM spread; at 30bps it drops. Verify real
  achievable EM spreads on the live broker.
- **G4 access (blocking for GLOBAL):** requires trading EM FX (USD_MXN, USD_ZAR, USD_BRL, USD_INR,
  USD_KRW). OANDA coverage of these is partial/region-dependent — confirm tradeability before shadow.
- Deploying as a runtime engine touches engine/execution + the halt logic, and requires an explicit
  risk-mandate change AND a typed live confirmation. Shadow/paper first, always.

## Status
- Engine BUILT + validated (research): `scripts/build_harvester.py` (majors frontier),
  `scripts/experiment_em_carry.py --crash-aware` (GLOBAL high-premium book). Artifacts in
  `trained_data/backtests/harvester_v1_*.json`, `em_carry_*.json`.
- NOT wired to live. Next gates: operator mandate decision → G4 broker access → shadow deploy.
