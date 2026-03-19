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

## 2026-03-18 — Autonomous Improvement Session

- [2026-03-18] **model_accuracy/EUR_USD**: EUR_USD blocked from trading. Root cause: model trained on GBP_USD tier2 labels shows 53.6% val accuracy (barely above random). EUR_USD had 100% loss rate (2/2 trades: -0.5p, -25.1p). Action: blocked in all profiles until per-pair retrained model available. sub_inference_min_confidence raised to 60%+ (conservative: 62%, balanced/smart: 60%).
- [2026-03-18] **robustness/learning_engine**: Fixed 6 critical/warning issues: (1) agents dict type validation prevents TypeError crash, (2) _parse_ts() returns None instead of 0.0 to prevent premature archiving, (3) check_promotions() now logs unmatched learning lines for diagnosis, (4) duplicate rule prevention uses normalized comparison (strips dates/counts), (5) pair name regex validation in update_pair_sl_tp(), (6) improved ISO datetime error handling.
- [2026-03-18] **robustness/config_tuner**: Fixed 4 critical issues: (1) file locking (fcntl) prevents race conditions in _log_adjustments(), (2) comprehensive bounds added for all adjustable fields, (3) persistent MD5 rule hashes replace Python hash() for restart stability, (4) case-insensitive regex for rule matching.
- [2026-03-18] **robustness/state_engine**: Fixed 4 issues: (1) JSON structure validation on load (required keys check + merge with defaults), (2) OANDA response validation (check for "account" key, handle 401/429), (3) env var validation before API calls, (4) corrupted trade journal handling (try/except with graceful fallback).
- [2026-03-18] **robustness/improvement_tracker**: Fixed 3 issues: (1) .splitlines() replaces .split("\n") for cross-platform JSONL parsing, (2) empty file returns empty list (no JSONDecodeError), (3) division-by-zero prevention in get_trend() with empty/single-entry guard.
- [2026-03-18] **pattern/code_review_before_live**: Code review of learning loop before first live run revealed 12 critical bugs and 11 warnings. Pattern: always run code review specialist on new subsystems BEFORE first production use, not after failures.
