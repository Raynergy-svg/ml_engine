# Pre-registration — Kronos-small vs Axiom trend lane, trade-level shootout

**Experiment ID:** `kronos-vs-axiom-trend-shootout-2026-07-24`
**Registered:** 2026-07-24, BEFORE any out-of-sample result was computed or read.
**Mode:** confirmatory (primary window), exploratory (secondary window).
**Lane:** research-only. No execution path. No promotion path. The outcome of this
experiment changes NOTHING about live/practice behavior regardless of who wins.
`oanda_environment="practice"` and all halt state untouched by construction — the
experiment script imports no broker, execution, or state module.

## 1. Question

Operator question, verbatim intent: "let's see who wins more trades, Axiom vs Kronos."

Formalized: on the same 19-pair daily FX universe, same decision dates, same horizon,
same cost model — does a zero-shot Kronos-small forecast arm produce more winning
trades (and better net expectancy) than the Axiom trend-lane construction actually
running on the practice account?

## 2. Arms (frozen)

### Arm A — AXIOM (incumbent, exactly the live construction)

- Signal: `src/equity/trend_sleeve.py::trend_sleeve_weights` (reused, not forked),
  per-pair, `sma_window=100`, `step=1`: **long** iff `close[t-1] > SMA100(close ≤ t-1)`,
  else **flat**. Long-or-flat is the lane's real construction — shorts are not
  penalized against it; abstention is part of the strategy.
- This is the identical rule driving `run_oanda_trend_cycle` on the live practice lane.

### Arm B — KRONOS (challenger, zero-shot)

- Model: `NeoQuasar/Kronos-small` (24.7M) + `NeoQuasar/Kronos-Tokenizer-base`,
  loaded via the pinned Kronos repo @ commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.
  Exact HF revision hashes recorded in the result JSON at runtime.
- Input: last **400** completed daily OHLCV bars through decision bar `t` (panel
  columns open/high/low/close/volume; `amount` synthesized by the Kronos predictor
  itself per its own API).
- Inference: `pred_len=5`, `T=1.0`, `top_k=0`, `top_p=0.9` (repo defaults),
  `sample_count=3` (internally averaged), `torch.manual_seed(42 + week_index)` set
  immediately before each batch call. CPU.
- Predicted 5-day return: `pred_close[+5] / close[t] − 1`.
- Direction: **long** if predicted return > deadband, **short** if < −deadband,
  else **abstain**. Deadband = primary round-trip cost (2.5 bps). Kronos may short;
  that is its construction freedom, disclosed.

## 3. Trade contract (identical for both arms)

- Decision dates: the **first trading day of each ISO week** present in the panel
  (per pair; a pair missing that bar skips that week).
- Decision uses only bars ≤ `t` (bar `t` = the decision-day bar, known at its close).
- Entry: open of bar `t+1`. Exit: open of bar `t+6` (5 trading days held).
- Weekly gross return: `direction × (open[t+6]/open[t+1] − 1)`.
- **Trade segmentation:** consecutive weekly decisions in the same direction on the
  same pair merge into ONE trade (entry at first segment's entry, exit at last
  segment's exit, one round-trip cost). A direction change or abstention closes the
  trade. This avoids over-charging the buy-and-hold-style trend arm and over-counting
  its "trades".
- Net trade P&L: gross compounded over the segment − round-trip cost.
- **Win:** net trade P&L > 0.

## 4. Costs

- Primary: flat **2.5 bps round-trip** per trade (conservative for FX majors' spread).
- Scenarios (sensitivity, all reported): **1.5 / 2.5 / 4.0 bps**.
- Weekly hit-rate readout (§6 metric 2) is computed gross — it measures sign skill,
  not cost survival; all expectancy metrics are net.

## 5. Windows

- **Primary (confirmatory):** decisions 2025-09-01 → 2026-06-11 (panel end).
  Kronos-small weights were published 2025-08 (recorded at runtime from HF metadata);
  therefore no bar in the primary window can be inside Kronos' training data.
- **Secondary (exploratory, flagged):** decisions 2024-01-01 → 2025-08-31. Kronos'
  pretraining corpus may overlap this window (bias in KRONOS' favor). Reported, never
  used for the confirmatory verdict.
- OOS discipline: neither window's outcomes are computed until this document is
  committed. Panel: `market_data/factor/*_D.csv`, 19 pairs, 2014 → 2026-06-11.

## 6. Metrics (all reported per arm, per window, per pair and pooled)

1. **Winning-trade count** and trade win rate (the operator's headline question),
   plus trade count and participation rate.
2. Weekly directional hit rate on taken weeks (matched-sample comparison,
   two-proportion z-test on weeks where BOTH arms took a position).
3. Net expectancy per trade and total net return at equal unit risk per trade.
4. Weekly net return series → Sharpe (annualized ×√52), Deflated Sharpe Ratio
   (`gated_harness.significance.deflated_sharpe_ratio`, `n_trials=1` per arm), and
   circular block bootstrap p-value on the ARM DIFFERENCE (Kronos − Axiom weekly
   net return, block length 4, 2000 resamples, seed 7).
5. Max drawdown of the weekly net cumulative curve.

## 7. Verdict rule (frozen)

- **Headline answer** = which arm has more winning trades in the PRIMARY window
  (count, as asked).
- **Statistical verdict:** the difference is DECISIVE only if the same arm also has
  (a) higher pooled net expectancy per trade and (b) bootstrap p < 0.05 on the weekly
  net-return difference in the primary window. Otherwise the headline count stands
  but the verdict is INCONCLUSIVE_STATISTICALLY.
- A Kronos win here does NOT promote Kronos to anything. It would justify exactly
  one follow-up: a governed evidence-slice evaluation. An Axiom win closes the
  question until Kronos ships a materially different model.

## 8. Trial accounting

- `trial_budget=1` — one frozen construction per arm, zero hyperparameter search,
  no correction needed. Any post-hoc variation (different SMA window, deadband,
  horizon, temperature) is a NEW experiment requiring a new pre-registration.

## 9. Honesty constraints

- Point-in-time: every input to a decision is from bars ≤ t; entries at t+1 open.
- Survivorship: fixed 19-pair FX universe; no instrument added/removed mid-window.
- Negative or embarrassing results are committed unchanged.
- The script records: git commit of this repo, Kronos repo commit, HF revision
  hashes, torch version, seeds, and the digest of this document.
