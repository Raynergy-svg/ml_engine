# Buddy Trading Learnings

Date-stamped insights extracted from trade outcomes, scan analysis, and system behavior. Patterns that repeat 3+ times get promoted to `.claude/rules/`.

---

## 2026-03-17 — Session 1 (Initial Deployment)

- [2026-03-17] **keras_compat**: Keras 3.x rejects `seed` parameter in Dense/Conv/MHA layers. Fix: strip seed in keras_model_loader.py. Keras 2 models load cleanly after this.
- [2026-03-17] **uncertainty_blocking**: Hard circuit-breaker (uncertainty agent blocks ALL trades when confidence <60%) killed every setup. Soft penalty (proportional confidence reduction) allows good trades through while still discounting uncertain ones.
- [2026-03-17] **sl_tp_method**: Hardcoded 15/30 pip SL/TP was wrong for every pair. ATR-based dynamic sizing (SL=1.2x ATR, TP=2.0x ATR) adapts to actual volatility. EUR_USD ATR=13.6p, USD_JPY ATR=27.3p — one size never fits all.
- [2026-03-17] **position_sizing**: 0.025 lots on $100K account = meaningless. Risk-per-trade 5% base with 2.5x medium-confidence multiplier produces 2.5 lot positions that actually move the needle.
- [2026-03-17] **correlation_filter**: Without correlation filter, system would open EUR_USD LONG + GBP_USD LONG + AUD_USD SHORT — all effectively the same USD bet. Correlation groups prevent this.

## 2026-03-17 — Session 1 (Trade Outcomes)

- [2026-03-17] **pair_behavior/EUR_USD**: EUR_USD LONG signaled twice (67% conf), lost both times. Trade #905 lost -0.5p, trade #919 hit full SL at -25.1p (-$627.50). Model direction was wrong — EUR_USD was actually bearish despite LONG signal.
- [2026-03-17] **pair_behavior/NZD_USD**: NZD_USD SHORT signaled twice, won both times. Trade #899 won +2.9p, trade #923 hit TP at +9.7p (+$242.50). Strong consistent signal.
- [2026-03-17] **sl_tp/EUR_USD**: EUR_USD #919 hit exact SL price (1.1473) — 25.1 pips from entry. Move was decisive, no bounce. When a trade goes against you hard in the first hour, SL does its job. Guardian couldn't help because the move was continuous.
- [2026-03-17] **agent_accuracy**: Trades with higher weighted_vote_score (0.68 for NZD_USD) performed better than lower (0.65 for EUR_USD). Higher consensus correlates with better outcomes.
- [2026-03-17] **sizing**: Net session P/L with 2.5 lot trades: -$385 (NZD +$242.50, EUR -$627.50). One SL hit on 2.5 lots costs $627. Position sizing is correct but need better directional accuracy to be profitable.
