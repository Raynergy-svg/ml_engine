# PRE-REGISTRATION — Sleeve combinations: RETURN-alpha vs drawdown-control (2026-06-29)

Written BEFORE building/running (anti-p-hacking, L-018). Combining sleeves is HIGH
multiple-testing risk, so: a FIXED short list of combinations, ONE combiner, ONE rule
per sleeve (all already validated/pre-registered individually), portfolio-level judgment,
chronological OOS holdout, significance corrected for the number of combinations.

THE BAR IS RETURN EDGE, NOT another drawdown-reducer. We already have the drawdown-control
finding (multi-asset trend). This batch asks: does ANY combination add real *return* /
risk-adjusted-return over buy-and-hold and over the best constituent — or does it only
lower drawdown again (Sharpe up purely from diversifying vol, with no return gain)?

## Sleeves (each its own already-validated/pre-registered stream)
- **TREND** = the validated multi-asset trend book (`multi_asset_trend.combined_portfolio`,
  21 ETF/crypto proxies, 200d SMA long-or-flat, HRP+vol-target). Gate-clearing drawdown-reducer.
- **CARRY** = G10+EM FX carry (FRED daily, `signals.carry_signal` + the em_carry pipeline,
  crash-aware vol-target+DD-breaker overlay). Gate-rejected alone (fat left tail).
- **HARVESTER** = equity-beta harvester (`variant_eval.book_return_series` on the PIT
  universe). Ship-gate-passing single-stock equity book.

## Combinations (FIXED list — exactly these, no others without a new pre-reg)
1. **CARRY + TREND** — the classic managed-futures pairing (trend hedges carry crashes;
   negatively correlated in stress). HRP across {carry stream, trend stream}.
2. **TREND + HARVESTER** — weakly-correlated orthogonal sleeves (the real "multiple frames").
   HRP across {trend stream, harvester stream}.
3. (ONLY if 1-2 are clean) **XSMOM** — cross-sectional momentum across the multi-asset
   universe as a distinct third sleeve, combined with trend.

## Combiner (FIXED) + metric
- ONE combiner: `sleeve_combiner.combine_sleeves` (HRP across the return matrix, 252d/21d),
  then a 10% vol-target overlay. No per-sleeve tuning.
- **Decomposition (the whole point):** for each book report ann_return (mean×252),
  ann_vol (std×√252), net_sharpe, max_dd. Separate the two ways Sharpe can rise:
  - **RETURN edge** = combo ann_return ≥ max(constituent ann_returns) AND combo Sharpe >
    best constituent Sharpe by a real margin — i.e. risk-adjusted *return* improved.
  - **DRAWDOWN/RISK-only** = combo Sharpe up but ann_return ≤ a constituent's (Sharpe rose
    only from lower vol/DD) — the same drawdown-control finding, NOT new return alpha.
- Baselines: each constituent ALONE, plus EW buy-and-hold of the multi-asset panel.

## Win-bar + anti-p-hacking
- **WIN (real return edge)** = combined book clears the ship gate (Sharpe≥0.40, DD≤0.25,
  pos_yrs≥6, total_yrs≥10) full AND OOS, AND beats BOTH constituents and buy-hold on
  net Sharpe OOS by a real margin, AND the decomposition shows the gain is RETURN-side
  (not purely lower drawdown). Anything less = honest negative.
- 2-3 combinations tried → correct significance for the count; OOS is the chronological
  last 35% (the search never tunes on it); separate verifier independently re-derives +
  flags multiple-testing + whether each "win" is return-edge or drawdown-only.
- A dredged positive (tuned, cherry-picked combo) = the L-018 lie. Refuse it.
- Do NOT auto-promote/trade any winner — report it. Directional transformer closed; practice-only.

## Pre-committed honest conclusion structure
After this batch, state plainly: did we find genuine RETURN alpha in ANY combination, or
is the honest finding that this free-daily-bar / liquid-asset input space is efficient
(risk-control: yes; return-alpha: no)? If the latter, say so — it's a valid, important
result, and it means the path forward is NEW INPUTS (alt-data / higher-frequency /
fundamentals-at-scale), not more backtests on the same data.

---

## RESULT (post-hoc, 2026-06-29) — NO RETURN ALPHA in any combination; risk-control only

`trained_data/backtests/sleeve_combinations_result.json`. ann_ret = RETURN side; vol/maxDD = RISK side.

### Test 1 — CARRY + TREND: NEGATIVE (carry dilutes trend)
carry: ret 0.048 vol 0.073 Sharpe 0.649 DD 0.333 · trend: ret 0.081 vol 0.103 **Sharpe 0.785** DD 0.189 ·
**combined: ret 0.059 vol 0.094 Sharpe 0.624 DD 0.268**. Combined Sharpe (0.624 full / 0.491 OOS) is WORSE
than trend-alone (0.785 / 1.034). Trend-alone dominates on Sharpe, return AND drawdown; the "trend hedges
carry crashes" pairing cut carry's DD (0.333→0.268) but added nothing over holding trend. **Carry is a drag.**

### Test 2 — TREND + HARVESTER: Sharpe up, but DRAWDOWN/RISK-side, NOT return
harvester: ret 0.130 vol 0.123 Sharpe 1.060 DD 0.231 · trend: ret 0.083 vol 0.107 Sharpe 0.777 DD 0.167 ·
**combined: ret 0.126 vol 0.113 Sharpe 1.112 DD 0.218**. Combined Sharpe (1.112) beats BOTH constituents —
a real diversification benefit. BUT the decomposition is unambiguous: combined ann_return (0.126) ≈
harvester (0.130), NOT higher; the Sharpe gain is purely lower vol (0.113<0.123) + lower maxDD. It beats
buy-hold on Sharpe only via lower vol (combined return 0.126 < buy-hold 0.21). **Risk-control, not alpha.**

### Test 3 — DECLINED (anti-dredge discipline)
Pre-registered as conditional ("only if 1-2 clean"). 1-2 ARE clean and unanimous. Running a 3rd combination
(contrived cross-asset XSMOM) after two clean return-edge negatives, hoping for a positive, IS the
multiple-testing dredge this discipline forbids (L-018). Declined; gets its own pre-registration if wanted.

### VERDICT — straight, no spin
**No genuine RETURN alpha in any combination.** carry+trend underperforms trend-alone; trend+harvester
improves Sharpe only via diversification (lower vol/DD at the same return), not excess return; neither beats
buy-and-hold on RETURN. Same as the whole campaign: **risk-control YES, return-alpha NO.** The free-daily-bar
/ liquid-asset input space is efficient for directional/return prediction at this scale. The honest path
forward is NEW INPUTS (alt-data / higher-frequency / fundamentals-at-scale), not more backtests on the same
data. Reported, NOT promoted/traded. Practice-only.
