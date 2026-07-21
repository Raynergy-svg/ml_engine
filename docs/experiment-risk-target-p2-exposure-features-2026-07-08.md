# Risk-target P2 — exposure/cost/tick features (GATE marker, not yet runnable)

Status: **BLOCKED on data accumulation.** Not a pre-registration yet — the frozen
construction gets written only when the gate below opens. This file exists so the P2
round is not forgotten and so nobody runs it prematurely on too-thin history (which would
be the anchored-window / thin-walk-forward failure the P1 pre-reg explicitly avoided).

## Lineage

- **P1 (2026-07-08, ran):** `docs/prereg-risk-target-vol-drawdown-2026-07-08.md` +
  `scripts/experiment_risk_target_vol_drawdown.py`. Forward-vol head **learnable** (OOS QLIKE
  0.070 vs naive 1.54, R² 0.77); forward drawdown-state **failed** its bar (AUC 0.625 ≥ 0.55
  but Brier 0.232 lost to base-rate 0.143). P1 deliberately **excluded** exposure/hedge/cost/
  tick features because no historical time series existed to join against — the hedge
  ledgers held 3 demo rows and `portfolio_exposure` is point-in-time.
- **The bridge (2026-07-08, this branch):** `src/hedge/exposure_history.py` +
  `scripts/run_exposure_history_capture.py` now snapshot the live FX book into per-cycle
  currency/correlation bucket nets (notional AND R-basis) → `trained_data/hedge/
  exposure_history.jsonl`. `src/data/tick_capture.py` (started 2026-07-07) accumulates
  intraday microstructure. Both are the P2 feature sources — but they only have value once
  they have **months** of forward-only history (no backfill exists; fabricating one is the
  exact leak class L-001 warns against).

## The gate (all must hold before P2 is pre-registered + run)

1. **Exposure history ≥ ~60 trading days** of non-duplicate rows in
   `trained_data/hedge/exposure_history.jsonl` (the capture loop must have been running).
   Check: `wc -l`, and confirm distinct `source_mtime_utc` spanning ≥ ~2 calendar months.
2. **Tick history ≥ ~60 trading days** per pair under `trained_data/ticks/{PAIR}/`.
3. **A joinable key exists** — exposure rows carry `source_mtime_utc` and `captured_at_utc`;
   tick features must be aggregatable to the same cadence (e.g. daily) without look-ahead.
4. The P1 feature pipeline's window-invariance canary discipline is carried forward: every
   new exposure/tick feature is a trailing statistic, tested for window-invariance, on its
   OWN `RISK_TARGET_FEATURE_PIPELINE_VERSION` bump.

**Earliest realistic gate date: ~2026-09-08** (2 months of capture) to re-evaluate; **~2026-10-08**
(3 months) for an honestly-powered walk-forward split on the added features. Do NOT run before
the row counts in (1)+(2) actually clear — a calendar date is a reminder, not the gate.

## What P2 would test (sketch — freeze only when the gate opens)

Add exposure-derived + tick-derived features to the P1 forward-vol / drawdown-state family
and ask whether they beat the P1 feature set on the SAME frozen OOS bars (QLIKE/AUC), same
naive-persistence baseline, same no-sweep discipline. Candidate features: net currency-bucket
concentration, exposure drift (Δ bucket net between consecutive snapshots), realized
spread/liquidity from ticks, margin utilization. Honest-negative is a valid outcome (as P1's
drawdown-state head already was).

## Hard constraints (carried from P1 §3)

Offline/research only. No execution, unhalt, arm, or broker path. The trained model outputs a
risk ESTIMATE for sizing/gating consumption, never a directional signal, never in the trade
hot-path. Gated write path (no OOS-metric regression). Nothing armed/live/added to the
hot-path config surface.
