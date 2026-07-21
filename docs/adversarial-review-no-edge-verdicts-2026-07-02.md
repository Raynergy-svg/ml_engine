# Adversarial Review of the "No Edge" Verdicts (2026-07-02)

**Reviewer stance:** independent adversary. Mandate = attack the NEGATIVES, not re-confirm the
house view. Hunt for a real edge killed by a bug, failed by an over-strict gate, mis-costed, or
dismissed while merely underpowered. Read-only; the system stayed **halted (all lanes) throughout**
— verified before and after (`.claude/state.json halted=true, halted_lanes all true`). The only
artifacts I wrote are offline scratch scripts outside the repo; no committed script, backtest JSON,
config, or state file was modified (confirmed via `git status` — all pending changes predate this
session and belong to the concurrent Ralph loop).

**One re-run of my own:** the equity multi-factor BLEND (the operator's explicitly-flagged "blends
never tested" shot), executed offline against the cached PIT panels. Result below. Every other case
was settled by reading the committed evidence + power arithmetic; where a defensible alt-spec was the
right test, I checked whether it had *already* been run before claiming it hadn't (it had, for crypto).

---

## Ranking — most likely we were WRONG → least likely

| # | Case | Verdict | Why it ranks here |
|---|------|---------|-------------------|
| 1 | **Track B** (filing-text factor) | **QUESTIONABLE — underpowered, not a measured zero** | N=40 scored vs **N≈405 needed**; scale-up truncated by a rate limit. Self-labeled INSUFFICIENT but at risk of downstream "NO EDGE" hardening. |
| 2 | **LLM-as-signal** | **QUESTIONABLE — the test is under-evidenced/degenerate** | The shadow artifact I found ran in **heuristic fallback, "LLM unavailable"** for all 4 events → 0/4 is not a real LLM test. Practical "shelve it" still holds via the FX prior. |
| 3 | **Crypto H2 XS momentum** | **CORRECT as un-shippable, but the *significance* sub-failure is genuine underpower** | +0.75 OOS, market-neutral, price-driven, real. Vol-target rescue **already run** (fixed DD, kept Sharpe) — only significance fails, and that's the structurally-untunable wall (effective-N≈3.9, ~12y OOS needed vs 2.4 have). |
| 4 | **FX H1 direction** | **CORRECT — minor framing caveat** | The load-bearing evidence is the clean 22-yr walk-forward coin flip, NOT the collapsed per-pair H1 transformers. Don't cite the broken artifact as the proof; the null holds via 4+ independent confirmations. |
| 5 | **PIT fundamentals** (value/quality/accruals) | **CORRECT — I ran the never-tested blend; it fails harder** | Value construction verified clean (no suppression bug). Value's +0.21 OOS is a 2022 value-regime artifact, full-sample **negative**, insignificant. Blend loses to plain EW beta in both windows. |
| 6 | **Equity harvester** ("0.92 artifact / ~0.35 OOS") | **CORRECT — and if anything generous** | Broad-universe 0.355 uses *cheaper* costs than the 0.908 curated run, so cost-parity pushes it **down**. Beta, not alpha. Least likely we were wrong. |

**Bottom line:** no case is a buried gate-clearing alpha. But two verdicts (Track B, LLM-signal)
overstate "no edge" when the honest label is **inconclusive / not-yet-tested-properly**, and two more
(crypto H2, the FX H1 artifact) rest partly on evidence that should be re-characterized. The negatives
that are *decisive* (PIT fundamentals, equity-harvester-as-alpha, crypto H1/H3) hold up to attack.

---

## Case 1 — Track B (agentic filing-text factor): **NEGATIVE IS QUESTIONABLE (underpowered)**

**What was run.** Pilot (12 companies, effective-N≈2) then a post-cutoff scale-up (`trackb/
run-2026-07-02-postcutoff-scale`, d43a585): strictly post-cutoff filings, N=40 scored (of 300 selected,
of 501 fetched), Sharpe 0.012, DSR 0.0265, bootstrap p≈0.50, cross-sectional rank-IC **+0.16 (p=0.43,
n=26)**. The commit **self-labels `overall_verdict=INSUFFICIENT`** — the full arm never even reaches the
gate's history-length criterion.

**The attack that lands.** N=40 was not a chosen stopping point — the scale-up was **truncated by a
session rate/usage limit** after 4/30 scoring batches. This is the *same* underpowered wall the pilot
flagged, not an independent confirmation of the null. Power arithmetic settles it:

> To detect a **true** cross-sectional rank-IC of 0.16 at 80% power under the Bonferroni bar
> (α=0.05/3, two-sided), Fisher-z requires **N ≈ 405 scored filings** in the cross-section.
> They scored **26–40**. That is ~10× short. A +0.16 point estimate that cannot be distinguished
> from 0 at N=26 is **not** a measured zero.

**Where the negative is right anyway:** nothing here is *positive* either — 0.16 with p=0.43 is
consistent with pure noise, and there is no economic prior forcing it to be real. So Track B is
**genuinely inconclusive**, and the self-label INSUFFICIENT is the honest one.

**The real risk = FRAMING.** The pilot commit title says "NO EDGE"; `.remember` logs say "no edge."
That hardens an *underpowered* result into a *measured* one. Recommend the standing record read
**"INSUFFICIENT / inconclusive — needs ~400+ scored filings across multiple rebalances,"** not "no edge."

**What settles it:** a non-rate-limited scoring run of ~300–500 filings across ≥3–4 quarterly
rebalances (needs an API-key path, not in-session subagent dispatch). Until then it is an open question,
not a closed one. *(Note the honest cost/benefit: the prior from every other lever failing is
unfavorable, so this is low-EV to pursue — but that is a resourcing call, not a scientific verdict of
"no edge.")*

---

## Case 2 — LLM-as-signal shadow OOS: **NEGATIVE IS QUESTIONABLE (test is degenerate/underpowered)**

**What I found.** `trained_data/reports/llm_macro_shadow_validation.md` (2026-07-02): 4 macro events,
0/4 threshold passes. **But every event's note reads `"LLM unavailable; heuristic fallback used"`,
`key_drivers: degraded_mode_heuristic`, confidence pinned 0.500.** The LLM never actually ran — this is
the degraded-mode heuristic, not the LLM. A 0/4 from a fallback stub is **uninformative about
LLM-as-signal**, and N=4 events / 2–3 pairs is underpowered regardless.

**Verdict:** the specific artifact does **not** support a "no edge" conclusion about LLM signals — it
supports "the LLM wasn't wired/available and the fallback heuristic doesn't flag regimes." The negative
is under-evidenced.

**Why the practical conclusion ("shelve it") still holds:** any LLM *directional* edge on FX majors
must beat the same efficient-market wall that killed price-only, news-fusion, and meta-labeling
(50.3% clean walk-forward; L-016/L-022). The prior is strongly unfavorable. So: don't deploy, but
don't cite this artifact as *proof* of no edge. If pursued, it needs a **real** test — LLM actually
running, pre-registered, adequately powered, with a gross-vs-net-of-cost decomposition (the one genuinely
interesting sub-question — "was a positive gross signal killed by turnover cost?" — is unanswerable from
this stub).

---

## Case 3 — Crypto campaign (H1 carry / H2 XS momentum / H3 order-flow): **H2 = CORRECT un-shippable, significance-failure is genuine underpower; H1/H3 = CORRECT decisively**

**H2 is the strongest lead found in the whole program** and deserves precise language: OOS net Sharpe
**+0.75**, market-neutral (BTC-β −0.07), return lives in price (real XS momentum), survivorship barely
moves it (+0.70 survivor-only). Independent verifier reproduced all 13 figures exactly. It fails the
gate on (a) full-sample maxDD −0.49, (b) significance (DSR 0.62, p 0.10), (c) cost-fragility (+0.09 at
2× cost).

**The alt-spec rescue was ALREADY RUN — I checked before claiming it wasn't.** Round-2 H4
(vol-target 10% + weekly rebalance) took **maxDD −0.49 → −0.196 (PASS ≤0.25)** while **holding OOS
Sharpe +0.85, β −0.03** → **3 of 4 ex-history gate criteria pass.** Only the **significance** criterion
fails for every config (best DSR ≈ 0.68). Round-4 multi-asset trend even **cleared significance
(DSR-OOS 0.99)** and missed the *full* gate by **maxDD 0.275 vs 0.25 — 0.025.** So the team was not
lazy: they vol-targeted, they lengthened, they expanded breadth.

**The honest decomposition:**
- The **DD failure is fixable** (vol-targeting fixed it) → cosmetic.
- The **significance failure is genuine UNDERPOWER, not a measured zero.** At SR_ann 0.75 you need
  **~12 years of OOS** to push the deflated-Sharpe t past ~2.6; crypto has ~2.4y OOS / ~6.5y total.
  Effective-N≈3.9 (the cross-section is ~one factor) makes it worse. This is **structural and
  untunable** — exactly what the codified lesson **L-020** says.
- The **cost-fragility (+0.09 at 2×)** is a real disqualifier independent of sample size.

**Verdict:** the "qualified negative" label is **correct and honest** — H2 is a *suggestive,
fat-tailed, cost-fragile risk premium,* not a confirmable alpha at this scale. It is NOT overclaimed
as a measured zero (they explicitly call it "the lead worth refining"). The one framing fix: the
significance FAIL should always be stated as "underpowered by history/effective-N," never as "signal
is absent." **H1 (funding carry, adverse-selection + cost, sign-flipped IS→OOS) and H3 (order-flow,
~50%/yr turnover cost, negative price leg) are decisive, correct negatives** with clean verifier
reproduction — no attack lands.

*(Risk-premium footnote, per L-021: crypto XS-momentum + TS-trend are reproducible drawdown-controlled
risk premia. If the operator ever wants a *non-alpha, risk-control* sleeve, this is the least-dead thing
in the program — but it is a bearing-crash-risk premium, not mispricing, and can never clear an
alpha-calibrated significance gate at 6.5y of history.)*

---

## Case 4 — FX H1 direction models (USD_JPY/USD_CAD/AUD_USD): **NEGATIVE IS CORRECT (framing caveat)**

The task frames this as "balanced-acc ~0.500, constant-direction collapse." Those are two *opposite*
signatures and the distinction is the whole ballgame:

- A **constant-direction collapse** (model predicts one class) shows **raw ≠ balanced** accuracy and is
  a **trainer pathology** (class imbalance, the documented C1 double-fit/identity-scaler skew, sigmoid
  killing the MAML gradient) — it would be evidence of a *broken model*, not "no edge."
- The **load-bearing verdict** does NOT rest on the collapsed per-pair transformers. It rests on the
  **clean 22-yr pooled walk-forward** (`daily_direction_oos_*.json`): mean **balanced acc 0.5029**,
  **raw ≈ balanced**, up-day rate ~0.50 every year — "no imbalance to inflate → **true coin flip.**"
  That is a *healthy* null from a properly-trained, class-balanced model, corroborated **four ways**
  (news-fusion no lift, meta-labeling dry, pre-2014 factor decayed, 24-source literature sweep).

**Verdict:** the FX-direction negative is **robustly correct** — but via the clean walk-forward, not the
collapsed artifacts. **Caveat:** if any verdict cites the H1 per-pair "constant-direction collapse" as
the *proof* of no edge, that conflates a broken quarantined artifact (correctly caught by the 10%-gap
gate) with a measured null. Fix the citation, not the conclusion. Re-training those 3 pairs would
near-certainly reproduce the coin flip and is not worth the effort (L-016).

---

## Case 5 — PIT fundamentals (value / quality / accruals): **NEGATIVE IS CORRECT — I ran the missing blend**

**Bug hunt (value factor):** construction verified clean end-to-end in `src/equity/value_data.py` —
cross-sectional z on the correct axis (per-date), correct sign (higher = cheaper = better), causal
filing-date join, market-cap `>0` guard, mean-of-available-z. The one real bug (unbounded
shares-outstanding forward-fill) was **already caught and fixed**, and critically its direction was to
**inflate** book_to_market ~1500× for ~12 stale-filer mega-caps — i.e. removing it makes value look
**worse, not better.** No suppression bug remains that would bias the Sharpe *down*.

**The positive point estimate, dissected:** value OOS net Sharpe 0.575 vs EW 0.361 (**margin +0.214,
beats baseline OOS**) — the operator's flagged number. But:
- Full-sample value margin is **−0.196** (value *underperforms* EW over 2012–2026).
- The OOS window (2021-07→2026) is exactly the **2022 value-factor resurgence** — a regime, not a
  persistent edge. The sign literally flips full→OOS.
- **DSR 0.66, bootstrap p 0.10** → not significant even before the Bonferroni bar; maxDD 0.366 > 0.25.
- The `earnings_yield_only` sensitivity is OOS +0.165 but **full-sample −0.047** — non-robust.

**The never-tested BLEND — my re-run** (`scratchpad/blend_edgar_factors.py`, pre-registered in-file,
same cached PIT panels / universe(629) / window / gate / costs, offline, zero repo writes):

| Blend | full net Sharpe (vs EW 0.737) | full maxDD | OOS net Sharpe (vs EW 0.361) | OOS maxDD | DSR-OOS | gate |
|---|---|---|---|---|---|---|
| value + accruals + margin | **0.376 (−0.361)** | 0.543 | **0.249 (−0.112)** | 0.335 | 0.38 | FAIL |
| value + margin | 0.326 (−0.411) | 0.583 | 0.205 (−0.156) | 0.336 | 0.35 | FAIL |

The blend **loses to plain equal-weight beta in both windows** — worse than value alone, because
blending value's lone regime-driven OOS win with the two robustly-negative factors (accruals −0.44 OOS,
quality −0.32 OOS) dilutes it into a net loss. **The tradeable object here is the equity risk premium
itself** (EW: full 0.737 / OOS 0.361, gate-PASS full-sample); every fundamental tilt and their blend
fail to add significant value over just holding the market.

**Framing check:** "3-for-3 FAIL" does NOT conflate "no alpha" with "no edge" here — the honest read is
explicit and correct: *the beta is the edge; the factor tilts don't add.* Negative confirmed, now with
the blend receipt the prior sessions were missing.

---

## Case 6 — Equity harvester ("0.92 concentration artifact / ~0.35 OOS"): **NEGATIVE IS CORRECT (generous)**

The 2026-07-01 audit already did the hard work: reran the 0.908 live twice (0.907/0.921 — real,
lookahead-checked), then showed the same EW+vol/DD overlay on the broad survivorship-corrected universe
gives full 0.740 / **OOS 0.355 (gate FAIL)**. Attacks considered:

- **Is 0.355 an over-strict-gate/regime artifact?** It's positive but below the 0.40 alpha gate, and the
  2022–26 sub-window is the fastest hiking cycle in 40y. **But** the broad-universe number uses
  **cheaper costs** than the curated run (flat 2bps, no ADV slippage vs 2bps+5bps/%ADV) — so
  cost-parity pushes 0.355 **down**, and the broad universe also holds illiquid since-removed names
  whose true costs are understated. **The negative is if anything generous, not too strict.**
- **Is failing an alpha gate even the right test?** The strategy self-describes as *risk-premium
  harvesting, not alpha* — a defensible point that a beta book shouldn't be judged by an alpha-Sharpe
  gate. But the audit's actual recommendation is right: **don't treat 0.908 as "the number"; surface
  0.740/0.355.** The 0.908 is a legible concentration effect (top-20-by-ADV ≈ the mega-caps that won
  2010–2026), not a bug, not leakage — and not an alpha.

**Verdict:** negative correct. Not a killed edge — it's beta, honestly labeled. The only open item is
the benign parameter-provenance gap (overlay params don't trace to a grid on the single-stock book),
which is a traceability chore, not an edge question.

---

## What would change my mind (per case, concrete)

- **Track B:** ~400+ LLM-scored filings across ≥3 rebalances with rank-IC holding ≥0.10 at p<0.017.
- **LLM-signal:** a real (non-fallback) LLM run, pre-registered, ≥100 events, gross-vs-net decomposition.
- **Crypto H2:** impossible to settle on significance at current history — it is permanently
  underpowered; only deployable (if ever) as a *labeled risk-control sleeve*, never as confirmed alpha.
- **FX H1 / PIT / harvester:** nothing short of a materially new input (not more of the same data) —
  these negatives are robust.

## Honesty ledger
Confidence: value-blend result **HIGH** (I ran it, reproduced the harness math). Track B / crypto power
numbers **HIGH** (arithmetic + committed figures). LLM-signal degeneracy **HIGH** (read the artifact).
FX H1 collapse-vs-clean distinction **MEDIUM-HIGH** (clean walk-forward figures read directly; the exact
provenance of the "H1 collapse" phrasing in the task I could not fully source to a single doc — but the
clean null is independently dispositive). Nothing live was touched; halt intact on all lanes.
