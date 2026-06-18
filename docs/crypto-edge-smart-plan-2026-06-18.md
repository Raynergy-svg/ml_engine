# Crypto Directional Edge — SMART Research Plan (2026-06-18)

Governs the one positive lead from the FX edge search: directional ML adds timing alpha in
crypto (ETH beat buy-and-hold 6/7 years) where it does nothing in FX majors or equities.
See `docs/fx-edge-search-final-verdict-2026-06-18.md` §3b for the originating evidence.

## SPECIFIC (the 5 Ws)
- **What:** Determine whether the ETH/crypto daily directional *timing* alpha is a real,
  deployable edge — or a 7-year, single-config, single-coin mirage.
- **Why:** FX majors are a proven dead end (~52%, every lever closed); equities have a premium
  but timing loses to buy-and-hold. Crypto is the ONLY evidence-backed positive lead. The core
  goal is profitable trades, not "an FX bot" — so a real crypto edge IS getting out of the coin flip.
- **Who:** Claude + research subagents; operator owns the deploy/pivot decision.
- **Where:** This repo; free data only (FRED BTC/ETH today; CoinGecko/Binance public APIs for a
  broader universe); existing HistGBM + walk-forward harness (`experiment_cross_asset_direction.py`).
- **Which resources/limits:** Free data only (no paid feeds, no tick data). Most alts have <5–7y
  history. Crypto deployment is a separate infra/regulatory decision (24/7, custody, broker coverage).

## MEASURABLE — four hard gates (a number decides, not a narrative)
- **G1 Robustness (kills multiplicity risk):** ETH long-only-timing beats buy-and-hold in **≥5/7
  years** across **≥3 distinct model configs** (shallow GBM, deep GBM, logistic) AND ≥2 feature
  sets. If the 6/7 result is one lucky config, this fails.
- **G2 Generalization:** across a universe of **≥8 liquid coins**, mean timing alpha (model long-only
  Sharpe − buy-and-hold Sharpe) **≥ +0.20** and **positive on a majority** of coins, after costs.
- **G3 Deployability (the existing ship gate):** a combined crypto book passes the project's
  mechanical gate — **net Sharpe ≥ 0.40 AND maxDD ≤ 25%** — on a walk-forward holdout, net of
  realistic crypto cost (~10–20 bps round trip).
- **G4 Access:** the live broker stack can actually trade **≥3** of the tested coins (else academic).

## ACHIEVABLE
Yes for G1 (data in hand, existing harness). G2 needs a new free-data integration (CoinGecko/
Binance) — moderate. G3/G4 only attempted if G1+G2 pass. Honest ceiling: free daily data + ~7y
history caps statistical power; this can establish a *lead worth piloting*, not a high-confidence edge.

## RELEVANT
Directly serves "use ML to find profitable trades." It is the single positive signal after FX
direction, meta-labeling, news, factors (carry/trend/EM/pre-2014), order flow, options-IV, and
equity timing were all closed with evidence. If it fails, halted-FX stands as the evidenced end.

## TIME-BOUND (milestone-based; attach calendar dates on request)
- **G1 — this work session** (now). Hard kill: if G1 fails, STOP — do not build on a fluke.
- **G2 — next work block** (after a CoinGecko/Binance loader lands).
- **G3 + G4 — only after G1 & G2 pass.**
- **Decision gate:** G1∧G2∧G3∧G4 pass → greenlight a *shadow/paper* crypto pilot (never live first).
  Any of G1/G2 fail → return to halted; FX verdict stands; close the lead honestly.
