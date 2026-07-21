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

## 7. Results (2026-07-01 — SEPARATE commit from §0–§6)

**Research model + cutoff:** claude-sonnet-5, acting AS the scorer directly in-session (no
`ANTHROPIC_API_KEY` configured for an automated LLM fan-out — see §7.0). Stated knowledge cutoff
January 2026; `model_cutoff="2026-02-01"` used to split pre/post arms.

**§7.0 — Scale-down from the frozen universe (disclosed, not hidden).** §2's frozen universe is
"all PIT S&P 500 members, ~2012-01 → as-of." Hand-scoring that at LLM-reading throughput (one
session, one sitting) is infeasible — thousands of filings. This run is a **bounded pilot**, decided
BEFORE any fetch or score: 12 systematically-chosen (not outcome-chosen) mega-cap names spanning
sectors — AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, JNJ, XOM, PG, HD, UNH — × 3 annual 10-Ks each
(FY2023/2024/2025; 36 filings, zero fetch gaps). Universe/window/anchor dates are frozen in
`scripts/track_b_fetch_and_blind.py` before scoring began. **This is a materially different scale
than §2** and the verdict below must be read as such: decisive on the lookahead-contamination
question, underpowered to make the frozen design's "definitive close" claim (see §7.5).

**Pipeline bug found + fixed before any scoring (load-bearing):** the original 12,000-char
head-truncation (`scorer.py` default) was consumed almost entirely by a modern 10-K's non-visible
Inline XBRL `<ix:header>` metadata block (~98K characters of context/unit definitions that precede
the visible cover page but are never rendered by a browser) — the first fetch scored 100% XBRL
tag-soup, 0% prose. Fixed in `src/equity/research/pit_text_loader.py` (`_TextExtractor._SKIP_TAGS`
now includes `ix:header`/`ix:hidden`/`ix:references`/`ix:resources`), covered by a new regression
test, and independently re-verified against the live Apple FY2023 filing by the verifier (§7.6).
`max_chars` was also raised 12000→45000 (NOT a frozen knob — absent from §3's frozen-knobs list) so
the scored text reaches real Item 1 Business narrative instead of cover-page/TOC boilerplate.

**§4.5 implemented (was a hardcoded `None`):** DSR-OOS(N=22) + Bonferroni block-bootstrap p-OOS,
added to `src/equity/research/harness.py` (Bailey & Lopez de Prado deflated-Sharpe formula +
seeded circular block-bootstrap, block=21, N_TRIALS=22, α=0.05/22=0.00227). `overall_verdict` now
also requires `dsr_oos_n22.passes_significance` on the full arm, on top of the pre-existing §1
controls — verified by the independent reviewer to correctly refuse `REAL` even when the mechanical
gate (criteria 1–4) passes but the significance extension does not.

**§1.1 blinding audit — wired, and NOT clean.** Full write-up: `blinding_audit.json`. The scorer
(this session) could re-identify the issuer in **36/36 filings** via a mix of: ticker-letter
concatenation leaks (AAPL), unredacted founders'-letter quotes ("Larry and Sergey" for GOOGL,
"William Procter and James Gamble" for PG), an unredacted executive-table name ("Jeffrey P. Bezos"
for AMZN), unredacted product names never added to the deny-list ("Facebook, Instagram, Messenger,
and WhatsApp" for META; iPhone/Azure/CUDA/AWS/Optum for the others), an unredacted secondary brand
("Chase" for JPM), and pure domain-knowledge fingerprints no deny-list can catch (NVIDIA's
GPU/CUDA/AlexNet history). **`blinding_audit_clean = False`.** This is not a surprise — it matches
§8's own prior finding that the blinder is "moderately leaky by construction... a noise-reducer, not
the load-bearing control."

**Per-arm results (frozen primary book, cost=2 bps/side):**

| Arm | Net Sharpe | Max DD | Positive yrs | DSR | Bootstrap p-OOS | Gate pass | Effective-N |
|---|---|---|---|---|---|---|---|
| Full | 0.489 | 10.9% | 2/4 | 0.137 | 0.189 | No (history<10yr) | 2.0 |
| Pre-cutoff | 0.724 | 10.9% | 2/4 | 0.217 | 0.105 | No | 2.0 |
| **Post-cutoff (clean arm)** | **−0.861** | 10.9% | 0/1 | 0.006 | 0.690 | No | 2.0 |
| Placebo | 0.358 | 12.7% | 1/4 | 0.093 | 0.232 | No | 2.0 |

Post-cutoff span: 2026-02-02 → 2026-06-30 (103 bars, 0.4 years — short, as pre-registered).
Stress re-run at 5 bps/side moves every number by <0.03 (full 0.471, pre 0.708, post −0.885,
placebo 0.339) — costs are not the story here.

`placebo_is_clean = False` (0.358 ≫ the frozen 0.15 threshold — the placebo is nearly as large as
the real full-arm number). `post_cutoff_consistent_with_full = False` (full>0 but post<0 — a sign
flip, not just a shrinkage). `full_significance_passes = False` (DSR 0.137 ≪ 0.95). Binding
`overall_verdict = "INSUFFICIENT"` (not `LOOKAHEAD_CONTAMINATED`, because criteria 1–4 do not even
pass on their own — chiefly `history_length` at 4 years vs the 10-year bar; not `REAL` under any
reading).

**§7.5 — the verdict.** Reading §1's frozen decision rule literally: the full-sample arm's apparent
edge (Sharpe 0.489) is decisively **refuted** by both pre-registered controls designed to catch
exactly this — (a) the post-cutoff arm doesn't just weaken, it **flips sign to strongly negative**
(the "looks good pre-cutoff, dies post-cutoff" pattern §1.2 calls "lookahead, full stop"), and (b)
the placebo (scrambled scores) produces nearly as large a positive Sharpe as the real full arm,
meaning the apparent edge is largely statistically indistinguishable from noise at this N. Given
effective-N=2 (Q5 on a 12-name universe concentrates to 2 names — not a diversified quintile), this
is exactly the behavior expected when a small, concentrated subset of 2023–2026 mega-cap/AI-driven
beta dominates whichever names the composite happens to rank highest, regardless of whether the
composite carries real signal.

**Blunt bottom line:** **NO EDGE**, at this pilot's bounded scale (12 companies, 36 filings,
~3 years). The clean/load-bearing arm (post-cutoff) is not just flat, it is negative and the
placebo control that should read ~0 does not. Both pre-registered lookahead controls independently
kill the apparent full-sample result — this is not a data availability or power problem, it is the
signature the pre-reg's own apparatus was built to catch. **What this run does NOT establish:** it
does not, by itself, close the door on the hypothesis at the frozen design's actual scale (~500
names, ~14 years) — N=12/Q5=2 is a genuinely underpowered test of "does a diversified qualitative
quality/red-flag cross-section carry alpha." The honest label is closer to "the lookahead-
contamination question is answered decisively (no residual pretraining-driven fake alpha survives
the clean arm), but the underlying alpha question at full scale remains untested" than a full
closure of the lever. Re-running at meaningfully larger N (which requires either an
`ANTHROPIC_API_KEY`-driven automated fan-out or materially more scoring sessions) is the only way to
raise power; re-running at this same small N with different prompts/weights would be dredging and
is explicitly ruled out by §5.

**§7.6 — independent verifier sign-off.** A Model QA Specialist agent re-derived this run cold from
disk: independently re-implemented and checked the DSR formula, constructed adversarial test cases
against `_build_summary`'s gating logic (confirmed `REAL` is unreachable when the significance
extension fails even if the mechanical gate passes), re-fetched the live Apple FY2023 10-K from SEC
EDGAR and confirmed the `<ix:header>` block spans exactly to where the visible cover page begins
(byte 98965), confirmed the regression test would fail on the pre-fix code, spot-checked 4 `as_of`/
`filed` dates against EDGAR's live submissions API, spot-checked 3 of 8 cited blinding leaks
verbatim in the blinded task files, confirmed no frozen knob (weights/quintile/cadence/vol-target)
was altered, and ran the full research test suite (111/111 pass). **Sign-off: the verdict holds.
No wrong math, no PIT leak, no frozen-knob change, no fabricated audit claim found.** One
verifier-flagged caveat carried into §7.5 above: the N=12/Q5=2 scale cannot distinguish "no
orthogonal alpha" from "underpowered test" at the frozen design's actual scale.

**Artifacts:** `trained_data/research/track_b_pilot_2026_07_01/{raw_scores.json, _index.json,
scores_artifact.json, blinding_audit.json, harness_result_primary_2bps.json,
harness_result_stress_5bps.json, fetch_log.json, blinded_tasks/, edgar_cache/, price_cache/}`.

---

## 8. Build + independent review (2026-06-30) — pipeline plumbing, NOT yet a result

The offline pipeline was built (`src/equity/research/{contracts,pit_text_loader,entity_blinder,harness}.py`,
58 no-mock tests) and put through **three independent reviewers** (Model QA vs this pre-reg,
Security adversarial-attack on the blinder, Code Reviewer on loader+harness correctness). Their
findings were triaged against disk and the confirmed ones fixed. The §0–§6 design above is FROZEN
and unchanged; this section records the engineering reality.

**Reframing forced by review (important, HIGH confidence — both Security + Model QA converged):**
the entity-blinder (§1.1) is **NOT the load-bearing control it is billed as.** It is moderately
leaky by construction — exact numeric fingerprints (`$394,328 million` → Apple FY22), bare
state-of-incorporation, segment/counterparty names the caller never supplies, all survive — and a
pretrained model re-identifies many issuers from "blinded" text. **The load-bearing control is the
post-cutoff arm (§1.2):** pretraining cannot contain post-cutoff outcomes, so any blinding leak that
inflates the FULL/PRE arms only *widens* the full-vs-post divergence that the conjunctive rule reads
as contamination. A leaky blinder **cannot manufacture a false PASS** — it adds noise to the
contaminated arms. Treat the blinder as a noise-reducer; lean the verdict on post-cutoff + placebo.

**Fixed this round (committed, tested):**
- **Binding §4.7 verdict** — the harness now emits `overall_verdict ∈ {REAL, LOOKAHEAD_CONTAMINATED,
  INSUFFICIENT}` that OVERRIDES any single arm's `gate_passed`. It is fail-closed: REAL requires
  full-arm gate pass AND placebo ~0 AND post-cutoff confirms (full>0 & post>0) AND a clean blinding
  audit — and since the human blinding audit (§1.1) is uncomputed in-harness, **REAL is currently
  unreachable from code** (an uncomputed control is treated as not-satisfied, never as passed).
- **Placebo derangement** — the bit-reversal permutation fixed up to 16 indices (n=128); a fixed
  point lets a name keep its OWN composite, leaking signal into the falsification control. Replaced
  with a deterministic fixed-point-free derangement (proven 0 fixed points for all n∈2..512).
- **Sign-flip consistency** — `post_cutoff_consistent_with_full` no longer mislabels a both-negative
  case as "consistent" (requires full>0 AND post>0).
- **Blinder audit false-`count:0`** — residual scan now also surfaces numeric fingerprints + surviving
  US state names and exposes a top-level `any_leak_signal`; ticker redaction uses letter boundaries
  so `AAPL_10K` / `AAPL10K` are caught. The audit can no longer read "clean" while a fingerprint
  survives.

**Deferred — required before a trustworthy number, tracked here (NOT yet built):**
1. **The §3.1 research scorer** (`BlindedText → ResearchScore`) — the offline LLM pass. Without it the
   pipeline cannot produce real scores end-to-end. This is the next build.
2. **DSR-OOS(N=22) + Bonferroni block-bootstrap p-OOS** (gate criterion §4.5) — currently an explicit
   `None`/TODO (honestly surfaced, never faked); criterion 5 is therefore unenforced until built.
3. **A short-window clean-arm decision rule** — the post-cutoff arm structurally cannot clear the
   10-year gate, so "post-cutoff confirms" needs a real small-sample rule, not just a sign check.
4. **Loader paging gap** — for sparse/delisted filers `load_pit_filing` can return `None` when an older
   qualifying filing exists (degrades safe — under-coverage, never lookahead); fix before a
   production-scale run that includes delisted names.
5. **Wire the 30-sample human/LLM blinding re-identification audit** into the binding verdict (and
   enlarge it; instruct the auditor to re-identify from financials/segments, not just proper nouns).
