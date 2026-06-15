# SOTA 2026 Modernization — Review & Corrected Plan

**Date:** 2026-06-15 · **Status:** Decision record (rejects the proposed plan; redirects to the approved one)
**Reviewer verdict:** The proposed "Full-Stack Modernization" plan does not serve the end goal
(profitable FX trades). It is rejected as the wrong horse. The active, operator-approved plan is
[tasks/prd-fx-factor-portfolio.md](../tasks/prd-fx-factor-portfolio.md). This document records *why*
so the autonomous loop and future sessions stop re-proposing it.

---

## 1. Why the proposed plan fails (grounded, not stylistic)

| Claim in the proposal | Reality on disk | Confidence |
|---|---|---|
| "Current model is 128-dim/3-layer ~200K params, two orders of magnitude too small" | Production direction head is `transformer_d_model=16`, ~2 layers ([transformer_trainer.py:272](../src/training/trainers/transformer_trainer.py)). Small capacity is the **overfitting control**, not an oversight. The premise argues against a model that doesn't exist. | HIGH (read disk) |
| "Scaling to 512-dim/8-layer fixes underfitting of regime transitions" | The binding constraint is **signal, not capacity**. Documented verdict (commit dad8624): price-only M15 = ~52% val, >10% gap, all three majors; the prior ~70% was the OBV-anchor leak. A bigger model fits train to ~90% and still hits ~52% val → quarantined by the 10% hard-gap ship gate (`HARD_MAX_GAP=0.10`, [.claude/rules/improvement.md]). Net result: weeks of M1 compute reproducing the current fail-closed state. | HIGH |
| "Keras latent diffusion generates realistic microstructure for pre-training" | A DDPM that passes KS + autocorrelation tests has, by construction, reproduced a distribution with the **same absent predictability**. You cannot sample signal into existence from a generator fit on signal-free data. Synthetic volume raises overfitting risk, not edge. | HIGH |
| "MoE routing / NOTEARS causal / LLM judge modernize agent deliberation" | These optimize the 15-agent voting + meta layers that the approved PRD (US-009) **explicitly freezes** as "built for a signal that doesn't exist." Also: LLM-in-decision-path violates the hard project rule (FR-6, CLAUDE.md "never use an LLM in the runtime hot path"). | HIGH |
| "Premium tick/L2 data unlocks price discovery" | The **one** fragment with a coherent thesis (order-flow may carry signal raw OHLCV lacks). But the documented "only remaining lever" (news/macro) was tested **dry** 2026-06-12, and tick edge rarely survives retail spreads at $1k capital. Defensible as a *separate, kill-ruled experiment* — never as a v1 dependency. | MEDIUM |
| "Market impact / predicted slippage model" | At $1k capital you are a price-taker with negligible market impact; modeling it optimizes a non-binding constraint. | MEDIUM |

**Root cause of the proposal's error:** it treats the gap as an *engineering* deficit (model too small,
context too short, data too shallow) when the project's own evidence shows it is a *signal* deficit —
a property of the data-generating process, not the model. More parameters lower training error, not the
irreducible error floor. This is precisely the "research tour" CLAUDE.md's strategy guardrails forbid
("Don't optimize the custom Transformer beyond bug fixes — wrong horse. Don't research-tour.").

---

## 2. Disposition of each proposed workstream

| # | Proposed workstream | Disposition | Do instead |
|---|---|---|---|
| 1 | Scale sequence model 512-dim/8-layer | **DROP** | Nothing. Tiny model is correct; gap gate would quarantine the big one. |
| 2 | Keras latent diffusion engine | **DROP** | Already exists as `DiffusionMarketModel`, scoped to backtest stress-test only ([backtest.py:177](../src/scanner/analysis/backtest.py)). Leave dormant; do not expand. |
| 3 | Premium data adapters (TrueFX/Polygon/Dukascopy) | **DEFER → separate experiment** | Only if the factor portfolio ships and an order-flow signal is hypothesized. Must carry a pre-registered kill rule like the news experiment. Not a factor-v1 dependency (PRD §7). |
| 4 | Curriculum + distillation training | **DROP** | No new model training (PRD Non-Goal §6). |
| 5 | MoE agent routing | **DROP** | The agent layer is frozen (US-009). |
| 6 | NOTEARS causal discovery | **DROP** | `CausalDiscovery`/`CausalFeatureSelector` already wired (see §4); freeze them. |
| 7 | Market impact prediction | **DROP at this capital** | Revisit only if capital scales 2+ orders of magnitude after a track record exists. |

Net: of seven workstreams, six are dropped and one is deferred behind the factor gate. That is the
honest answer — the modernization does not advance the end goal at this data scale and capital.

---

## 3. The corrected plan = execute the approved factor PRD

The proper plan for ml_engine already exists and is operator-approved. No rewrite of *strategy* is
needed — only disciplined execution in dependency order. Summarized from
[tasks/prd-fx-factor-portfolio.md](../tasks/prd-fx-factor-portfolio.md):

1. **US-001** Daily data layer — 7 USD pairs, ≥10y OANDA `D` candles, gap-validated (blocking; OQ-1).
2. **US-002** Carry signal — cross-sectional rank of policy-rate differential, causal.
3. **US-003** Trend signal — TSMOM 3/6/12m composite, causal + window-invariant.
4. **US-006** Cost-aware walk-forward backtest — **the load-bearing number** (next-bar fills,
   per-pair spread, OANDA financing markup ±1%/yr). Run this *before* value (US-004) per the PRD's
   staging. This is the cheapest experiment that answers the one unknown: does a 3-factor portfolio
   clear retail costs?
5. **US-007** Mechanical ship gate — net Sharpe ≥ 0.4, ≥6/10 positive years, max DD ≤ 25%,
   walk-forward true. Code decides, not judgment. FAIL = "no deployable edge at this bar," stated plainly.
6. **US-004 / US-005** Value factor + vol-targeted portfolio construction (value staged after US-006).
7. **US-008** Practice deployment — *blocked, fail-closed, until US-007 = PASS* (FR-7).
8. **US-009** Freeze the intraday stack — see §4; this review **expands** its scope.

**Honest expectation, unchanged (PRD §5):** at $1k / 10% vol / Sharpe 0.4–0.7 → ≈ $40–$70/yr with
$100–$250 drawdowns. $1k→$5k/yr is a binding NON-GOAL (ruin math). Deliverable = a verified track
record, not income.

**The load-bearing question** is US-006's net-of-cost number. Everything upstream (US-001–005) is
plumbing to produce it; everything downstream (US-007–008) is gated on it. Bias every session toward
getting that number, honestly.

---

## 4. Freeze gap surfaced by this review (action required in US-009)

The audit of commit 286509d ("wire 11 SOTA modules into execution path") found the freeze is
**incomplete**. US-009 must explicitly neutralize these, not just disable scanning:

| Module | Live reach today | Gap vs. freeze | Required US-009 action |
|---|---|---|---|
| HybridInference + TimesFMAdapter | flag `use_hybrid_inference` (default False) | dormant but present | Assert `False` in the `factor` profile; add a startup guard. |
| CQLRebalancer | flag `use_cql_rebalancer` (default False) | dormant but present | Assert `False` in `factor` profile. |
| MetaStrategyAgent | **no flag** — inits "if available", filters agents | **no off-switch** | Add `enable_meta_strategy_agent` flag, default False; off in `factor`. |
| SACExecutionAgent | **no flag** — shadow order slicing | **no off-switch** | Add flag, default False; off in `factor`. |
| CausalFeatureSelector | **no flag** — prunes features pre-decision | **no off-switch** | Add flag, default False; off in `factor`. |
| CausalDiscovery | **no flag** — builds causal graph | **no off-switch** | Add flag, default False; off in `factor`. |

**Why this matters:** four of these have no config gate at all — they initialize whenever their import
succeeds. That is the dead-write / silent-wiring failure mode this project has been burned by before.
Even though none flips a live trade *today*, "no off-switch" is incompatible with a strategy freeze, and
the autonomous Ralph loop is actively adding more such modules (commits 1b7bd17, 286509d) against the
approved direction. **(MEDIUM confidence — module flag defaults came from a subagent audit; verify each
default with a direct grep before implementing US-009.)**

**Recommended next concrete step:** before touching factor signals, land the US-009 freeze with these
six flags so the autonomous loop can't keep widening the intraday surface while the factor work proceeds.

---

## 5. One-line summary for CLAUDE.md / the loop

> SOTA-scaling the intraday transformer is rejected (signal-bound, not capacity-bound; verdict dad8624).
> Active strategy = daily factor portfolio (tasks/prd-fx-factor-portfolio.md). Intraday stack FROZEN.
> Do not re-propose model scaling, synthetic-data pretraining, MoE/causal agent routing, or LLM judges
> without materially new evidence.
