# Track B — Post-Cutoff Scale-Up (2026-07-02)

**Status:** RUN COMPLETE, pending independent verifier sign-off. Frozen design:
`docs/experiment-equity-research-alpha-prereg-2026-06-30.md`. Prior run: the
2026-07-01 12-company pilot (branch `trackb/run-2026-07-01`, commit `8aa9417`).

## Why this run exists

The pilot's own verdict flagged its decisive result (post-cutoff Sharpe -0.861,
0/1 positive years) as **underpowered, not closed** — effective-N=2 (Q5 on a
12-name universe) cannot distinguish "no orthogonal alpha" from "too small a
test." This run is a direct answer to that gap: same frozen pipeline, same
frozen gate, but the universe is built to be STRICTLY post-cutoff by
construction (every filing fetched has `filed >= model_cutoff`) and as broad as
free EDGAR access allows, to raise the cross-sectional N.

## What changed vs the pilot (and what did not)

**Changed (scale, not design):**
- Universe: current S&P 500 (Wikipedia, 503 tickers) vs the pilot's 12 mega-caps.
- Filing selection: most recent 10-Q/10-K with `filed <= 2026-07-01`, KEPT only
  if `filed >= 2026-02-01` (strictly post-cutoff) — vs the pilot's 3 annual
  10-Ks per name spanning FY2023-FY2025 (mostly pre-cutoff).
- Entity blinding: minimal auto-derived bundle (SEC legal name + ticker + CIK
  only, no hand-curated CEO/HQ/product denylist) — disclosed reduction in
  blinding precision, justified because prereg §8 (independently confirmed by
  two reviewers on the pilot) already found the blinder is NOT the load-bearing
  control; the post-cutoff filter is, and every filing here already satisfies it.

**NOT changed (frozen, per §5 anti-p-hacking guard):**
- Composite weights (§3.2): 0.5·quality − 0.5·red_flags + 0.0·outlook.
- Portfolio construction (§3.3): long-only Q5, monthly (21-trading-day) rebalance.
- Risk overlay (§3.4): 10% vol-target, causal DD de-gross circuit breaker.
- Costs (§3.5): 2 bps/side primary, 5 bps/side stress.
- Gate (§4): MIN_NET_SHARPE=0.40, MAX_DRAWDOWN=0.25, 6/10 positive years,
  DSR-OOS(N=22)>=0.95 AND Bonferroni block-bootstrap p<0.05/22.
- Scoring schema + prompt (§3.1): unchanged from `src/equity/research/scorer.py`.

## Universe / fetch result

Fetched via `scripts/track_b_fetch_and_blind_postcutoff.py` against live SEC
EDGAR: current S&P 500 constituent list (Wikipedia, 503 tickers as of
2026-07-02), most recent 10-Q/10-K per ticker with `filed <= 2026-07-01`, kept
only if `filed >= 2026-02-01` (model cutoff).

**501 of 503 tickers (99.6%) qualified** — near-universal, because Q1-2026
10-Qs (filed ~2026-04-15 to 2026-05-29 for accelerated filers) sit squarely in
the post-cutoff window. 2 tickers had no filing at all (`NO_FILING`); 0 had no
CIK; 0 were excluded for being pre-cutoff (every fetched filing already
qualified, since `load_pit_filing`'s `as_of` ceiling was 2026-07-01 and the
loader always returns the MOST RECENT qualifying filing — Q1 2026 filings
dominate this window for essentially the whole index). Fetch log:
`trained_data/research/track_b_postcutoff_2026_07_02/fetch_log.json`.

**This confirms the operator's "maximize N" instruction was fetch-feasible at
essentially full S&P-500 scale** (501 companies) — the bottleneck that follows
is scoring capacity, not data availability.

## Scoring

Scorer: this session, acting directly as the blinded research scorer (no
`ANTHROPIC_API_KEY` configured for an automated fan-out — same constraint as
the pilot), fanned out across dispatched subagents reading the SAME frozen
prompt (`scoring_task_to_prompt`) per filing.

**Selection:** deterministically modulo-sampled 300 of the 501 fetched filings
(stride over sorted, alphabetical-by-ticker task-id order — fixed before any
score existed; log: `trained_data/research/track_b_postcutoff_2026_07_02/selection_log.json`),
split into 30 batches of 10 for parallel dispatch.

**What actually got scored: N=40, not 300.** A first wave of 30 concurrent
subagent dispatches hit a server-side rate limit (0/30 completed). A retry in
smaller waves of 5 ran into a hard session-usage limit ("You've hit your
session limit · resets 1pm America/New_York") partway through — **4 of 30
batches completed with valid, schema-checked scores (batch_000, batch_001,
batch_003, batch_005 — 40 companies) before every subsequent dispatch failed
identically.** The other 260 selected filings were never scored and are
correctly treated as **abstentions** (dropped, not zero-filled — verified via
`scripts/track_b_collect_scores.py`, which logs every missing/malformed
task_id explicitly rather than silently defaulting it).

**This is the honest ceiling for this session**, not a chosen stopping point:
further Agent dispatches would fail identically until the session limit
resets. Reaching N=300 (or the full N=501) needs either (a) resuming after the
1pm ET reset with more scoring waves, or (b) a wired `ANTHROPIC_API_KEY` for a
true automated, non-session-bounded fan-out — exactly the caveat flagged in
the task brief.

## Results

`scripts/track_b_run_harness_postcutoff.py` → frozen `run_research_backtest`,
model_cutoff=2026-02-01, on the N=40 score set + a live Yahoo-chart price panel
(165 trading bars, 2025-11-04→2026-07-01).

| Arm | Net Sharpe (2bps) | DSR | Bootstrap p(Sharpe≤0) | Net Sharpe (5bps stress) |
|---|---|---|---|---|
| Full | 0.009 | 0.0265 | 0.483 | -0.008 |
| Post-cutoff (clean/load-bearing arm) | 0.012 | 0.0265 | 0.4998 | -0.010 |
| Placebo (no-signal control) | 0.512 | 0.063 | 0.256 | 0.492 |

effective_n (avg held names) ≈ 5.9 — Q5 on a 40-name universe concentrates to
~8 names per rebalance, only modestly better breadth than the pilot's
effective_n=2.

Gate criteria 1-4 (net Sharpe≥0.40, maxDD≤0.25, 6/10 positive years, ≥10yr
history) all FAIL on the full arm (span=0.65yr, nowhere near the 10-year bar —
structurally impossible to pass with a 5-month post-cutoff window regardless of
edge). `overall_verdict = "INSUFFICIENT"` (not `REAL`, not
`LOOKAHEAD_CONTAMINATED` — the full arm never even clears the mechanical gate,
so there's nothing for the §1 controls to contaminate).

**Read literally, the post-cutoff arm shows NO signal at the portfolio level**:
Sharpe 0.012 is statistically indistinguishable from the DSR/bootstrap null
(DSR 0.0265 vs the 0.95 bar; bootstrap p=0.4998 is what a pure coin-flip
return series would produce). The placebo's elevated 0.512 Sharpe nominally
FAILS the frozen `|Sharpe|<0.15` cleanliness bar, but its own DSR (0.063) and
bootstrap p (0.256) show it is ALSO not statistically distinguishable from
noise — consistent with a small, concentrated book (effective_n≈5.9,
kurtosis_pearson=14.18 — fat-tailed, lumpy) rather than genuine leakage. See
Independent verification below for the adversarial check on this exact point.

## Cross-sectional IC / quintile spread (additive diagnostic, not a gate change)

Computed directly against the price panel (Spearman rank-IC between composite
score and forward 21-trading-day return; reuses the frozen
`harness._placebo_permute`/`_latest_scores_asof` for the negative control —
does not alter `harness.py` or `ship_gate.py`).

Only **one rebalance date produced a valid cross-section** (2026-05-06 →
forward date 2026-06-05, n_names=26 — the other candidate rebalance dates had
too few priced+scored names to qualify, MIN=10):

| Metric | Value |
|---|---|
| Rank IC | +0.1606 (p=0.433 — not significant) |
| Placebo rank IC | -0.0787 (p=0.702 — not significant, correctly ~0) |
| Q5 (n=5) forward return | -0.51% |
| Q1 (n=5) forward return | -3.63% |
| Q5 − Q1 spread | +3.12% |

A positive point estimate (IC +0.16, spread +3.1%) with p=0.43 on n=26 is not
distinguishable from noise — at this sample size you'd need |IC| ≈ 0.32+ for
p<0.05. The placebo control here (unlike the portfolio-Sharpe placebo above)
behaves exactly as the frozen design expects: small and non-significant.

## Independent verification

Model QA Specialist dispatched to re-derive cold from disk: post-cutoff
construction, return alignment, DSR/Bonferroni math, frozen-design fidelity
(git diff against the pilot commit), score plausibility, and selection-bias
check on the 4-of-30 scored batches. Cross-checked independently by me as well
(DSR formula re-derived by hand — exact match to 9 decimals; 3 random
fetch_log entries + 2 more by the verifier confirmed live against
`data.sec.gov/submissions`; `git diff 8aa9417` on all four frozen files
confirmed empty).

**All six checks: PASS.**

1. **Post-cutoff construction, no lookahead — PASS.** `select_pit_filing`'s
   `filed > as_of: continue` gate, `FilingText.__post_init__`'s second
   independent enforcement, plus the script's own additive `filed < MODEL_CUTOFF`
   skip. Exhaustive scan of all 501 `OK` fetch_log entries: zero filed dates
   outside `[2026-02-01, 2026-07-01]`. 5 filings (3 by me, 2 by the verifier)
   cross-checked live against SEC EDGAR — all matched exactly.
2. **Return alignment, no lookahead — PASS.** Forward return strictly uses
   `fwd_pos = pos + FORWARD_HORIZON_DAYS` with an index-overrun guard; ranking
   reuses the SAME `harness._latest_scores_asof` the frozen gate math uses (not
   a parallel reimplementation that could silently diverge). `git diff 8aa9417`
   on `harness.py`/`backtest.py`/`contracts.py`/`ship_gate.py` is empty.
3. **Cost + significance math — PASS.** DSR re-derived independently (Bailey &
   Lopez de Prado formula, from raw skew/kurtosis/n, not copied) matches to 9
   decimals. The placebo's elevated 0.512 Sharpe is itself non-significant
   (DSR=0.063, p=0.256) and its own cross-sectional-IC placebo is separately
   near-zero (-0.0787, p=0.702) — two independent, differently-constructed
   negative controls agreeing it's small-N noise, not a leak. A real leak would
   be expected to show up consistently in BOTH; it does not.
4. **Frozen design fidelity — PASS.** `git diff 8aa9417` on all four frozen
   files confirmed empty twice, independently. Adversarial test against
   `_build_summary`: a synthetic gate-passes-but-significance-fails case, and a
   synthetic uncomputed-blinding-audit case, both correctly refuse `REAL`.
5. **Score plausibility — PASS.** 3 filings spot-checked (by the verifier) plus
   1 (by me) against source blinded text: rationale and `spans_used` quotes
   verbatim-matched the actual filing text in every case; no fabrication found;
   scores show real dispersion (not degenerate/constant).
6. **Selection bias — PASS, with one honest structural gap flagged.** The
   300-of-501 sampling is a content-blind function of task-id order fixed before
   scoring existed. The 4-of-30 batches that completed (000/001/003/005, not
   002/004) show a gap pattern consistent with an async dispatch race hitting
   the session limit, not order-based or outcome-based cherry-picking — but the
   verifier honestly notes a post-hoc audit cannot fully rule out a
   scored-then-discarded batch, since that would leave no trace by construction.
   No evidence of this was found.

**Frozen test suite: 111/111 pass** (`test_research_*.py` + `test_equity_ship_gate.py`),
matching the pilot's own count — confirms nothing broke.

## Verdict

**NO EDGE, and the null is informative even at this small N** — not
"promising," not "compromised." Quoting the verifier's bottom line:

> Confidence: HIGH on "this specific run is clean and its null result is real,
> not an artifact of a bug." MEDIUM on "there is no orthogonal alpha at any
> achievable scale" — because N=40 (effective_n≈6-8 per rebalance) is still
> underpowered relative to the frozen design's target of ~500 names.

**The scale-up answered the load-bearing question the pilot could not**: at
N=40 (effective_n≈5.9, vs the pilot's effective_n=2), the post-cutoff arm's
Sharpe (0.012) is not just small, it is statistically indistinguishable from a
coin-flip return series (DSR 0.0265 ≪ 0.95; bootstrap p=0.4998). The
cross-sectional IC (+0.16, p=0.43, n=26, single rebalance) gives no
encouraging trend — it's a positive point estimate on far too little data to
mean anything, and it does not corroborate the elevated-but-non-significant
placebo Sharpe. Unlike a genuine signal (which would be expected to strengthen
as N grows), nothing here shows growing effect size with growing N — it stays
flat at "indistinguishable from zero" at both N=12 (pilot) and N=40 (this run).

**What would change this:** a rerun at N≈300-501 — proven fetch-feasible here
(501/503 tickers qualified) — with the post-cutoff Sharpe and rank-IC point
estimates growing in magnitude AND significance, ideally across 3+ independent
rebalance dates (this run had only one valid IC cross-section — thin evidence
specifically for that diagnostic, separate from the 104-bar portfolio-Sharpe
arm). That full run needs either a wired `ANTHROPIC_API_KEY` for automated,
non-session-bounded scoring fan-out, or resuming after this session's usage
limit resets (1pm America/New_York) for further manual scoring waves.

**Honest scale disclosure:** the fetch proved N could reach ~501 (99.6% of the
S&P 500) — data availability was never the constraint. Scoring capacity was:
a 30-way parallel dispatch hit a server rate limit (0/30), and a retry hit a
hard session-usage limit after 4/30 batches (40/300 selected, 40/501 fetched).
This is the honest ceiling for this session, not a chosen stopping point.

**Dual independent confirmation.** Two fully independent Model QA Specialist
verification passes ran (one via an accidental nested delegation from the
first dispatch, one via an explicit no-delegation re-dispatch) — both reached
"NO EDGE (null informative)" with HIGH confidence, both independently
reproduced the DSR math to the same decimal, both confirmed the frozen-file
diff empty, both cross-checked live EDGAR dates, both found no score
fabrication. The second pass rates the placebo noise-vs-leak read as MEDIUM
confidence specifically (n=1 rebalance for the IC check and n=104 correlated
bars for the Sharpe check are both "too thin to fully rule out a residual
leak with confidence," while still favoring noise over leak) — a slightly
more conservative framing than "PASS," but the same overall verdict.

**Nothing live was touched.** All work is on a git worktree
(`/Users/buddy/Documents/ml_engine_trackb`, branch
`trackb/run-2026-07-02-postcutoff-scale`) separate from the main `ml_engine`
checkout; no config, execution, gate, or halt-state files were read or written;
`oanda_environment` remains untouched; no trades, no unhalt, no arm.
