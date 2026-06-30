# Equity Research-Alpha — Pre-Registration (the agentic-research portfolio, ONE frozen test)

**Date written:** 2026-06-30 (BEFORE any result is computed)
**Branch:** claude/agentic-research-portfolio-qetsfr (harvester foundation merged in at da83b16)
**Status:** PRE-REGISTERED — §0–§6 are FROZEN and committed BEFORE any result (separate commit).
Research / shadow only — **no live execution, no real money, no paid data.** Bot remains halted,
fail-closed. Immutables intact.

This is the operator-chosen upside shot of the "Both, sequenced" decision (2026-06-30): harden the
deployable equity-beta harvester (Track A) AND, in parallel, pre-register the ONE genuinely-untested
alpha lever (this doc, Track B).

---

## 0. The hypothesis, and why it is the only equity alpha door still open

The harvester campaign closed every **mechanical** equity edge on true point-in-time data:

| Tested (PIT, SEC-EDGAR) | Verdict | Source |
|---|---|---|
| Value / Quality / Momentum / LowVol / Reversal **L/S** | dead (net Sharpe −0.32 … −1.12) | `fundamental_factor_*.json`, `equity_xs_factor_*.json` |
| Quality **L/S** vs EW baseline | quality does NOT beat baseline | `pit_quality_bakeoff.json` |
| Value+Quality **long-only Q5** | gate-passes (Sharpe 1.04) but ~ties SPY → **smart-beta, not alpha** | `fundamental_factor_*.json` |
| PEAD (post-earnings drift) | real signal (monotonic Q5>Q1, IR≈0.16 @42d) but **untradeable** (net Sharpe 0.009) | `pead_sec_*.json` |

**The closed finding (HIGH confidence):** the *mechanical, ratio-based* cross-section carries no
market-neutral alpha at this scale/window — only a long-only quality/value smart-beta tilt that the
risk-managed harvester already captures.

**The ONE thing not tested:** whether a **qualitative** research layer — an LLM research team reading
the actual *text* of filings / transcripts / news per name, rather than XBRL ratios — extracts
**idiosyncratic** signal the ratio-factors structurally cannot see (management tone, litigation and
guidance nuance, segment-mix shifts, accounting-quality red flags, supply-chain disclosures).

**Pre-stated prior (calibrated MEDIUM-LOW, honest):** the mechanical-factor death *tempers* this — if
fundamentals-as-numbers carry no L/S alpha, fundamentals-as-text must add something *orthogonal* to
numbers to clear the bar. That is possible (text ≠ ratios) but not the way to bet. We run this once,
to a hard gate, precisely because the prior is weak and the failure mode (a fake-alpha p-hacked /
lookahead-contaminated result) is the most dangerous outcome in the whole project.

---

## 1. THE binding validity threat — lookahead via pretraining (read before anything else)

This test has a failure mode that **no mechanical factor in the campaign had**, and it is fatal if
unhandled: **the research model's pretraining already contains the future of every well-known company.**
An LLM asked to "research" a 2015 10-K already knows the ticker rose 4× by 2026, already knows SVB
failed in 2023, already knows which names were acquired. Restricting the *input* text to point-in-time
does **NOT** remove pretraining memory. A naive backtest ("LLM reads the period-T filing, ranks names")
would manufacture spectacular fake alpha that is pure hindsight — the L-018 lie in its purest, most
seductive form.

**The validity of this entire experiment rests on the four-layer lookahead control below.** It is
FROZEN here. If the control cannot be implemented as specified, the test is NOT run.

1. **Entity-blinding (primary control).** The research agent receives ONLY de-identified text: company
   name, ticker, CIK, exchange, executive/board names, brand & product names, headquarters/geographic
   identifiers, and any date at or after the as-of timestamp are STRIPPED or replaced with neutral
   placeholders (`COMPANY_A`, `PRODUCT_1`, `[REDACTED_DATE]`). The agent scores fundamentals it cannot
   map to a remembered outcome. A separate verifier audits a random 30-sample for residual
   re-identifiability (can a blind reader name the company? → leak).
2. **Post-cutoff OOS arm (decisive, power-limited).** The only window where pretraining *cannot* contain
   outcomes is **after the research model's training cutoff** (the model used is recorded in the result;
   its cutoff is treated as the OOS_start floor — e.g. a model with a Jan-2026 cutoff yields a clean OOS
   from 2026-02 forward). This window is short (low statistical power) but **uncontaminated**. It is
   reported SEPARATELY and is the arm that actually decides "real vs hindsight." A signal that is strong
   in-sample (pre-cutoff) and vanishes post-cutoff IS lookahead, full stop.
3. **Forward live-shadow arm (gold standard, accrues over time).** A live shadow lane that scores names
   as of each go-forward rebalance and logs immutably (atomic append, hash-chained). No backtest
   contamination is possible. It produces no verdict today; it is pre-registered NOW so the clean track
   record starts accruing immediately, and it is the evidence that ultimately governs any deploy.
4. **Blinded-scramble placebo (negative control).** The identical pipeline is run on entity-blinded text
   whose financial *content* is permuted across names (label shuffle). It MUST produce ≈0 alpha
   (|net Sharpe| < 0.15). If the placebo shows alpha, the pipeline is leaking and the primary result is
   void regardless of how good it looks.

**Decision rule on the controls (frozen):** the result is reportable as a real edge ONLY IF
(a) entity-blinding audit passes, AND (b) the post-cutoff OOS arm is directionally consistent with the
full-sample arm, AND (c) the placebo is ≈0. Any one failing → the result is classified
LOOKAHEAD-CONTAMINATED and the lever is closed, not iterated.

---

## 2. FROZEN universe

- **Membership:** point-in-time S&P 500 via `src/equity/universe.py` `UniverseSnapshot` (survivorship-aware
  reconstruction; `universe_hash` recorded). Names enter/exit on their PIT membership dates — delisted /
  removed names are KEPT for the periods they were members (no survivorship inflation).
- **Per-rebalance tradeable set:** members on the rebalance date with (i) a price series, (ii) at least one
  EDGAR filing on/before the as-of date. Names failing either are excluded that period (logged).
- **Text inputs (PIT, free, no paid data):** SEC EDGAR primary documents reachable via the existing
  `src/equity/edgar_fundamentals.py` PIT layer (10-K / 10-Q / 8-K, filing-date-aligned; later restatements
  are NOT used — same discipline already enforced in that module). News, if included, is restricted to
  headlines/bodies timestamped strictly before the as-of date from a free PIT-safe source; if no PIT-safe
  news source is available, the news channel is DROPPED rather than risk lookahead (frozen: filings-only is
  the default arm; news is an explicitly-labelled secondary arm only if PIT-clean).
- **Universe size / window:** all PIT members over the EDGAR clean window (~2012-01 → as-of), consistent
  with `pit_quality_bakeoff.json`'s stated clean floor. The window is mechanically determined, not chosen.

---

## 3. FROZEN construction (ONE pipeline — no variants)

1. **Research fan-out.** For each name in the per-rebalance set, one research pass over the entity-blinded
   PIT text emits a STRUCTURED score: `fundamental_quality ∈ [-1,1]`, `accounting_red_flags ∈ [0,1]`,
   `forward_outlook ∈ [-1,1]`, `conviction ∈ [0,1]`, plus a short rationale and the list of text spans used.
   Schema-validated; a pass that cannot produce a schema-valid score abstains (name dropped that period —
   never zero-filled, mirroring the inference-contract rule).
2. **Cross-sectional score.** Composite = pre-registered fixed weights (0.5·quality − 0.5·red_flags +
   0.0·outlook for the PRIMARY arm; outlook is logged but zero-weighted in the primary to avoid the
   most lookahead-prone field driving the book — outlook-weighted is a labelled secondary arm). Ranked
   cross-sectionally per rebalance.
3. **Portfolio (PRIMARY = long-only Q5, matching the only gate-passing equity template).** Equal-weight the
   top quintile by composite; monthly rebalance (step=21 trading days), `shift(1)` causal, whole-name.
   Secondary reported book: Q5−Q1 L/S (we expect it dead, per mechanical prior — reported for completeness,
   not the ship candidate).
4. **Risk overlay (identical to the deployable harvester).** Causal 10% vol-target + drawdown/vol-spike
   de-gross circuit-breaker (`src/equity/` overlay used by the single-stock harvester). `target_vol=0.10`
   passed explicitly. `max_lev` per the harvester default.
5. **Costs:** 2 bps/side + ADV-aware slippage (5 bps per %ADV), matching `equity_harvester_singlestock_*`.
   Stress re-run at 5 bps/side (robustness, not a new trial).

**Frozen knobs:** score weights (§3.2), quintile cut (Q5), cadence (21d), vol-target (0.10), costs (2 bps,
5 bps stress). NONE are tuned after seeing results. The research prompt + blinding ruleset are committed in
this doc's companion module before the run and hash-recorded in the result.

---

## 4. FROZEN gate (canonical harvester gate + campaign significance extension)

Clears iff ALL hold on the PRIMARY long-only Q5 book, net of costs:

1. OOS net Sharpe ≥ **0.40** (`MIN_NET_SHARPE`, `src/factor/ship_gate.py`)
2. full-sample maxDD ≤ **0.25** (`MAX_DRAWDOWN`)
3. positive years ≥ **6** of ≥ **10** (`MIN_POSITIVE_YEARS` / `MIN_TOTAL_YEARS`)
4. OOS-confirmed — no in-sample→OOS sign flip
5. **DSR-OOS(N=22) ≥ 0.95 AND Bonferroni block-bootstrap p-OOS < α**, where the multiple-testing count
   increments to **N_TRIALS = 22** (campaign was 21 through edge-round-4) → **Bonferroni α = 0.05/22 = 0.00227**
6. effective-N reported
7. **all three lookahead controls (§1) pass** — entity-blinding audit clean, post-cutoff OOS arm consistent,
   placebo ≈0. This is an ADDITIONAL hard criterion unique to this test and OVERRIDES 1–6: a book that clears
   1–6 but fails any §1 control is LOOKAHEAD-CONTAMINATED, not a pass.

Plus the L-021 return-vs-risk decomposition (β-to-SPY, vs EW buy-hold, vs 60/40) to classify any clear as
genuine idiosyncratic ALPHA vs disguised beta/smart-beta.

---

## 5. Anti-p-hacking guard (the binding constraint — this is the trap)

- §2–§4 are FROZEN and committed BEFORE the run, in a separate commit from any result.
- **Exactly ONE run** of the PRIMARY arm (plus the pre-named secondary L/S arm, the 5 bps stress, and the
  three §1 control arms — these are pre-registered, not new trials). No universe variants, no score-weight
  search, no quintile/cadence tuning, no "try another prompt." The research prompt is frozen and hashed.
- If it clears the FULL gate (incl. §4.7 controls) AND a separate verifier re-derives it leakage-free →
  real result, report and advance to the forward live-shadow arm. If it does NOT clear → **definitive close
  of the equity research-alpha lever at this data scale.** Either outcome is accepted as final.
- The temptation will be to re-weight the score / re-prompt / extend news until the gate crosses. That is
  dredging the now-binding metric — the exact L-018 lie. It is NOT done.

---

## 6. Pre-stated honest framing (frozen, before results)

- **Even a full clear is "small idiosyncratic alpha at a verifiable track-record scale," not income.** The
  deployable floor remains the risk-managed beta harvester (Track A). This lever determines only whether a
  qualitative-research overlay adds *orthogonal* alpha on top — and the post-cutoff/forward arms, not the
  contaminated full-sample backtest, are what will ever justify real capital.
- **Most likely outcome (pre-stated): the full-sample arm looks good and the post-cutoff/placebo arms kill
  it** — i.e. the apparent edge is pretraining hindsight. Pre-registering that expectation is the point: it
  stops a contaminated backtest from being mistaken for a discovery.
- This is the natural-language analogue of the directional-ML wall: the research *motions* are easy and
  impressive; whether the output is information the market hasn't already priced is the entire question, and
  the controls in §1 are the only thing standing between an honest answer and a flattering lie.

---

## 7. Results (appended after the fact — SEPARATE commit from §0–§6)

_To be filled by the run. Will record: research model + cutoff, universe_hash, N members, OOS_start
(= max(EDGAR floor, model cutoff) for the clean arm), per-arm {full / pre-cutoff / post-cutoff / placebo}
net Sharpe, maxDD, positive years, DSR-OOS(N=22), p-OOS, effective-N, blinding-audit verdict, L-021
decomposition, and the frozen prompt/blinding hashes. Verifier verdict attached on any clear._
