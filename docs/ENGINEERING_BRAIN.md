# ENGINEERING BRAIN — the single, sourced, actionable blueprint

> **Purpose.** One coherent picture that fuses the *full internal reality of this bot* with *rigorous
> external evidence*, so any future session (and eventually the bot's own brain loop) reasons from the
> same foundation instead of re-deriving it. This is a **research/writing** document — it changes no
> config, arms nothing, unhalts nothing, trades nothing.
>
> **Author/verify:** built 2026-07-06 from disk-verified internal state + adversarially fact-checked
> web research (every external anchor independently re-verified before inclusion; confidence tagged
> per claim). Read `CLAUDE.md` → `.claude/INTENT.md` → `.claude/NOTES.md` → `.claude/LESSONS.md`
> alongside this; this doc points at them, it does not replace them.
>
> **The one-sentence truth of this system:** *it is an honest, fail-closed research-and-risk-harvesting
> machine that has proven — not assumed — that no free-data return-alpha is accessible to a small
> operator, and whose real assets are its gated verification harness and its safety spine, not a
> predictive edge.*

---

## Part I — FAITHFUL CURRENT STATE OF THE BOT

### I.1 What this system actually is (2026-07-06)

An autonomous ML trading bot on a **practice** OANDA account (~$102k paper NAV), one solo operator
(Buddy) + LLM agents (Claude) used **only** for planning/research/post-mortems — **never in the
per-scan/per-trade hot path**, which is deterministic and Claude-free by construction.

Two things are true at once and both must be held honestly:

1. **The predictive-edge search is exhausted and came up empty on free data.** After fixing every
   real defect (feature leakage, a broken ship-gate metric, a double-fit scaler), price-only intraday
   FX direction lands at **~52% balanced accuracy — a coin flip** — and stays there under every lever
   tried. This is not a bug; it is the efficient-market wall. (LESSONS L-001/L-016/L-020/L-022;
   `docs/training-architecture-audit-2026-07-03.md` §1.3; adversarial re-review
   `docs/adversarial-review-no-edge-verdicts-2026-07-02.md`.)
2. **The engineering around that null is genuinely strong.** A gated research harness
   (pre-registration + untouched OOS + Deflated Sharpe + Bonferroni bootstrap + separate verifier)
   repeatedly caught in-sample optimism that would have shipped false positives, and a fail-closed
   safety spine makes "trade real money by accident" structurally hard. These are the durable assets.

**Live posture right now** (from `.claude/state.json`, read this session): global `halted: false`,
but `halted_lanes.oanda_fx: true` — i.e. the FX trend lane is **halted at the lane level** while the
other lanes (equity, brain, crypto_momentum, track_b) are lane-unhalted but running **shadow/dormant**.
`mode: live`, `oanda_environment: practice` (immutable). NAV $101,955; 3 open OANDA-practice trades
(trend lane), unrealized −$25. `last_actor: operator-directed-scoped-unhalt-2026-07-06T1032Z`.

> **Note on `mode: live`.** This is *not* real money. Real-money risk is the **environment/endpoint**,
> never the mode flag: the broker is hard-pinned to `api-fxpractice.oanda.com`. A bot in `mode=live`
> on a practice environment trades paper money. (LESSONS L-014.)

### I.2 Architecture in two layers

```
Scanner (engine.py) → 15-agent consensus (_team.py) → Gates → Execution (execution.py) → OANDA practice
     ↑                                                                    ↓
     └────── Config Tuner ← Rules ← Learnings ← RL feedback ←──── Trade outcomes
```

- **Tier 6** — meta-learning ensemble (MetaLearner + Bayesian adapter + ensemble weighter),
  shadow-only. WIRED-LIVE via Orchestrator dispatch but welded to the (currently dormant) TUI scan loop.
- **Tier 7** — autonomous control loop: incident → propose → gate → soak → promote → close. Deterministic.
- **AXIOM agent runtime** (`src/agent_runtime/`) — the newest layer: a policy-gated tool registry +
  reasoning loop that lets an LLM operator *observe and propose* safely. **Nothing armed; escalation is
  structurally proposal-only** (see Part II).

**The ML stack is real and modern** (not "TCN/Ridge/RF"): Tiny Transformer (d_model=16) + EMA + EWC +
replay for direction; HistGradientBoosting hybrid voter; TCN dual-head volatility regime; LightGBM
momentum/risk/confidence; XGBoost meta-labeler on triple-barrier labels; PPO position sizer;
EMA-damped multiplicative-bandit agent weights; walk-forward + purged k-fold + embargo validation;
Platt+isotonic calibration. Feature pipeline is **window-invariant v2/v3** (`FEATURE_PIPELINE_VERSION
= "2026-06-12-v3"`) after the L-001 anchored-OBV artifact; v1 artifacts refuse to load.

### I.3 The edge reality — what was tested, what was found (the campaign ledger)

This is the load-bearing honesty of the whole project. Every door below was tested with the gated
harness (pre-registration, untouched OOS, Deflated Sharpe ≥ 0.95, Bonferroni block-bootstrap p, maxDD
≤ 0.25, ≥10y history), and independently verified.

| Door tested | Result | Verdict |
|---|---|---|
| FX daily/intraday **direction** (price-only, M15/H1/H4/D) | ~52% balanced (own 22-yr walk-forward **50.3%**) | **No edge** — efficient-market coin flip. FX **retired** (L-016). |
| **News/macro fusion** (FinBERT + FF news) EUR_USD M15 | +0.96pp val, gap *worse* (quarantined) | **No lift.** Shelved. |
| **Meta-labeling** (predict *when* the signal is reliable) | meta-AUC ≤ 0.53, 3 majors | **Dry.** Same wall, other door. |
| **Daily cross-sectional factors** (carry/value/quality/momentum/low-vol) | gross Sharpe ≈ 0; net negative after turnover | **No edge.** Beta is the only tradeable object. |
| **True-PIT SEC-EDGAR fundamentals** (value/quality/accruals + blend) | 3-for-3 gate FAIL; blend loses to plain EW beta | **No edge** — the beta is the edge; tilts don't add. |
| **EM + global carry** (FRED daily rates) | full-cycle net Sharpe ~0.5 but maxDD ~40%, fat left tail | Risk premium with a **carry-crash tail**, not a prediction edge. |
| **OANDA order-book sentiment** (contrarian fade) | pooled IC ≈ 0; IS Sharpe collapses OOS | **No edge.** |
| **Crypto funding carry (H1)** | OOS net Sharpe **−0.25** | Decisive negative (price adverse-selection + cost). |
| **Crypto XS momentum (H2)** | OOS net Sharpe **+0.75**, market-neutral, real — but maxDD −49%, DSR 0.62, cost-fragile | **Un-shippable / underpowered** (structural eff-N ≈ 3.9, ~6.5y history). The strongest lead found. |
| **Crypto TS-trend (H5) / multi-asset trend (R4)** | Beats buy-hold on Sharpe *and* drawdown; clears 4/5 gate criteria | **Risk premium, not alpha** — fails significance (thin breadth) or drawdown at century history (L-021). |
| **Crypto order-flow (H3)** | OOS net Sharpe **−4.45** | Decisive negative. |
| **Track B — agentic filing-text factor** | rank-IC +0.16 (N=40) **fades to +0.09** as N→291; self-labeled INSUFFICIENT | **Underpowered / inconclusive**, NOT a measured zero (needs ~405 scored filings across ≥3 rebalances). |
| **LLM-as-signal** (macro shadow) | 0/4 — but ran in *heuristic fallback*, LLM never fired | **Degenerate test**, uninformative; prior unfavorable. |

**Standing conclusion (L-022, verifier-confirmed, literature-consistent):** *no free-data,
small-operator-accessible, cost-surviving, OOS-confirmed, significance-clearing return-ALPHA exists.*
The one reproducible survivor across FX / multi-asset / crypto is **trend / time-series momentum — a
RISK PREMIUM (compensation for crash/tail risk), not mispricing alpha** (L-021), and at this operator's
scale it fails the significance gate on effective-N and history length. The honest end-state: **real
edge, if it exists at all here, needs a materially new input — paid PIT-fundamentals / options-implied /
microstructure data, or cross-exchange/basis infra — none of which is free.** Six free-data doors are
closed. Staying fail-closed on direction prediction is the literature-consistent call, not a failure.

> This matches the published record exactly (independently re-verified, Part III): FX cannot beat a
> random walk OOS (**Meese–Rogoff 1983**); most anomalies decay ~58% post-publication
> (**McLean–Pontiff 2016**); the surviving anomaly premium is eaten by stock-borrow fees — *"the borrow
> fee IS the anomaly"* (**Muravyev–Pearson–Pollet 2025, Journal of Finance**); trend is a risk premium
> (**Moskowitz–Ooi–Pedersen 2012**). Full citations in Part III.

### I.4 The shadow lanes (live posture, honest running-state)

Per L-017, "shipped" ≠ "running." Current lane reality:

- **`oanda_fx` (trend lane)** — the ONLY thing placing real (practice) fills. Strategy is
  **non-directional managed-futures trend** (price vs MA, long-or-flat, `shift(1)` causal) — *not* the
  retired FX transformer. `scripts/run_oanda_trend.py`, practice-pinned. A ~0-Sharpe **drawdown-reducer**
  (risk premium), never claimed as alpha. Currently **lane-halted**.
- **Equity harvester** — `scripts/run_equity_harvester.py --broker shadow`. Equity-**beta** risk-premium
  sleeve. Headline `SHIP_GATE.json` = net Sharpe **0.908 / maxDD 0.229 on a curated 20-name mega-cap
  universe** — but the **same construction on the survivorship-corrected wide PIT universe scores 0.740
  full / 0.355 OOS, gate FAIL** (2026-07-01 independent audit). **Always report both; the defensible
  number fails.** It is beta, honestly labeled — not alpha. IBKR-paper only; nothing armed.
- **Crypto momentum** — shadow/research lane. Best OOS Sharpe +0.75 but underpowered/un-shippable.
- **Track B (filing-text)** — shadow. rank-IC faded +0.16→+0.09 as N grew; INSUFFICIENT, not "no edge."
- **Brain loop** — `scripts/run_brain_loop.py`, deterministic hypothesis→gate→**shadow-promote** only;
  structurally cannot flatten/arm/unhalt (verified in `tests/test_brain_loop_capability_absence.py`).

**Nothing is armed. Nothing promotes to live without an operator-typed `"LIVE"` token** (Part II).

---

## Part II — THE SAFETY MODEL (fail-closed spine, disk-verified 2026-07-06 @ HEAD 0cc5c49)

The safety spine is the system's single greatest engineering asset. It is **structural**, not advisory
(L-005): every rail is backed by disk-reading code or a hook, not a sentence in a doc.

### II.1 The four Hard NOs (immutable; the `/evolve` loop can NEVER relax these)

1. **`oanda_environment` stays `"practice"`** — default at `src/scanner/config.py:742`; passed through
   `src/brokers/factory.py:24,68`; base URL defaults to `api-fxpractice.oanda.com`
   (`src/scanner/execution.py:801`). (L-003.)
2. **Respect halt** — halted means halted at every layer, including a halt set *mid-cycle* (L-004).
3. **Nothing promotes to champion without passing the ship gate** (`HARD_MAX_GAP = 0.10`; L-002).
4. **No real money, ever.** Practice account only.

### II.2 Per-lane halt system — fail-closed by construction

`src/scanner/automation/state_engine.py`:
- **`KNOWN_LANES`** (`:31`) = `("oanda_fx","equity","brain","crypto_momentum","track_b")`; typo'd lanes
  raise `ValueError` (fail loud).
- **`_atomic_write`** (`:110–122`) — `mkstemp` + `os.replace`, corruption-safe; every mutation uses it.
- **`get_halted_strict(lane)`** (`:269–328`) — the **last-line-of-defense** guard: file missing →
  `True`; unreadable JSON → `True`; payload not a dict → `True`; global `halted=True` → `True`; lane
  entry missing/corrupt → `True`. **Every failure mode returns halted.** Raises `ValueError` if no lane
  is bound (there is deliberately no "global strict" escape).
- **`set_halted(value, lane=None)`** (`:336–374`) — `lane=None` sets global AND cascades to all 5 lanes
  (master switch); a lane arg updates only that lane. Global `halted=True` always wins via OR logic.

`src/equity/decision_gate.py` — `decide_cycle` (`:187–240`) evaluates rails **most-severe-first**:
halt (REFUSE) → drawdown breach (HALT) → ship-gate missing/failing (NO_ACT) → data stale (ABSTAIN) →
CONTINUE. `_lane_halted` (`:98–125`) fails closed: missing file / unreadable / lane-not-configured all
return halted.

**Execution mid-cycle re-check** (`src/scanner/execution.py:2081–2114`): the last guard before an OANDA
order fires calls `StateEngine(lane="oanda_fx").get_halted_strict()`; on *any* exception it sets
`_halted = True` (fail-closed, operator decision 2026-07-02). This closed the Trade-1306 gap where a
mid-cycle halt was bypassed (L-004).

### II.3 AXIOM agent-runtime policy engine — the resident-operator boundary

`src/agent_runtime/policy.py` — a **4-tier** action model, every tier re-deriving the practice pin at
runtime (`_preflight_practice_pin`, `:131–138` → raises `PolicyDenied` if env ≠ practice):

| Tier | Examples | Autonomy |
|---|---|---|
| `OPERATIONAL` (`:48`) | restart daemon, refresh data, diagnostics | autonomous *if* the autonomy flag is on |
| `DEESCALATION` (`:49`) | halt lane, reduce gross leverage | autonomous, but **structurally risk-decreasing only** (halt passes only `True`; leverage refuses any non-strict-decrease, `:244–261`) |
| `SELF_IMPROVE` (`:50`) | edit data/doctrine files | autonomy-gated **+** independent structural allow/deny + test-verify-commit-or-revert |
| `ESCALATION` (`:51`) | unhalt, arm, size up, promote model, enable new exposure, change code | **NEVER autonomous — proposal-only** |

**Escalation is proposal-only by *construction*, not by policy text:** `ActionSpec.__post_init__`
(`:112–122`) *raises* if an ESCALATION spec is built carrying an `execute` callable; and
`PolicyEngine.submit` (`:175–200`) has a redundant runtime check — even a spec tampered post-construction
via `object.__setattr__` is caught (`denied_tampered_spec`) and never executed. Escalation returns a
`Proposal` to the operator, full stop.

**Reasoning loop** (`src/agent_runtime/loop.py`): autonomy is **fail-closed default-False** —
`is_autonomy_enabled` (`:73–88`) requires a literal JSON `true` (not truthy coercion); missing/corrupt →
`False`. With autonomy off, the loop OBSERVEs (read-only tools) and PROPOSEs but logs a "shadow" audit
instead of executing.

**Self-improvement tier** (`src/agent_runtime/self_improve.py`) — a bounded door for the agent to edit
*itself*: a `STRUCTURAL_DENYLIST` (`:61–83`) unconditionally refuses edits to execution.py, state_engine,
gates, config (practice pin), decision/live gates, brokers, risk_monitor.sh, state.json, and the agent's
own autonomy flag; an `ALLOWLIST` (`:88–92`) permits only `agent_weights.json`, `LESSONS.md`, `NOTES.md`.
Every edit runs apply → test → `verify_gate.py` → `risk_monitor.sh` → scoped `git commit` → **revert on
any failure** (`_run_gated_edit`, `:184–249`). The agent cannot edit its way to more permission.

**Read-only tool registry** (`src/agent_runtime/tools/`): 9 tools (health_check, gate_health,
tier7_status, lane_halt_status, oanda_account_state, trade_feedback, agent_weights, learnings,
shadow_ledger), all `OPERATIONAL`, all read-only — "none writes trading state, none places an order,
none arms or unhalts anything" (enforced structurally + tested).

### II.4 Live gate — shadow→live is operator-token-gated

`src/equity/live_gate.py`: `armed = False` by default (`:293`); `arm()` (`:445–568`) requires a typed
confirmation equal to `LIVE_CONFIRMATION_TOKEN = "LIVE"` (`:84`) **and** a passing ship-gate **and** a
constructable kill-switch + drawdown guardian, else `LiveGateError`. There is no autonomous arm path.

### II.5 Enforcement layer (the self-improver's own rails)

- **Deterministic gate** `.claude/loop/verify_gate.py` re-derives the Hard NOs from disk (can't be
  narrated around) + a separate **Code Reviewer agent** for semantics; both must PASS for STOP-DONE.
- **Stop hook** `.claude/tools/stop_gate.sh` runs `risk_monitor.sh` at every turn-end, fail-closed,
  blocks on ALARM.
- **Objective stopping conditions** `.claude/loop/loop_gate.py` (HALT-SAFETY / STOP-BLOCKED / STOP-DONE
  / STOP-CHURN / CONTINUE) — signals derived from observable artifacts, not self-report (L-007/L-010).
- **Gate integrity** — every enforcement script hash-pinned in a committed manifest; checkers
  cross-verify and fail closed on drift (L-008); managed-settings anchor puts the Stop hook above worker
  write access (L-012/L-013).

**The honest floor (documented, not hidden):** these gates make lazy-skip and silent-tamper
structurally hard; they cannot *prove* an LLM agent was dispatched or judged honestly (L-011). The
irreducible lie-dimension rests on human review — stated plainly, per the lie-policy (L-018).

### II.6 The doctrine memory (why the rules exist)

`.claude/LESSONS.md` (L-001…L-024) is an append-only, recall-indexed ledger of hard-won failure modes,
scanned at the ORIENT step of every plan. The load-bearing ones for this doc: **L-016** (FX retired),
**L-020** (infra tuning moves cosmetic gate-failures, never significance), **L-021** (trend = risk
premium not alpha), **L-022** (no free-data alpha; don't re-run the exhausted hunt without a materially
new input). These are not opinions — they are the compressed output of the campaign in Part I.3.

---

## Part III — EXTERNAL BEST-PRACTICE + EDGE/DATA MAP (rigorously sourced)

Every external claim below was researched multi-source and the load-bearing ones independently
re-verified by the author before inclusion. Confidence tags: **HIGH** = primary source(s), cross-checked;
**MEDIUM** = credible but secondary/practitioner or contested; **LOW** = single-source/likely optimistic.
Marketing is flagged as marketing. **No edge is invented here.**

### III.1 Architecture & process — where this design already meets or beats the literature

The system's design is a faithful (and in places stronger-than-standard) instance of published
best-practice for agentic, self-improving quant systems:

- **Keep the LLM out of the money path — this is a *validity* safeguard, not just latency.** Anthropic's
  *Building Effective Agents* (2024) prescribes workflow-first orchestration with programmatic **gates**
  on transitions and sandboxed testing + human checkpoints before production — exactly this system's
  propose→gate→soak→promote shape. More sharply, *Profit Mirage: Revisiting Information Leakage in
  LLM-based Financial Agents* (arXiv 2510.07920, 2025) shows LLM trading agents produce **artificially
  high backtests via lookahead/information leakage that cannot be replicated live** — a finance-specific
  reason to keep Claude out of the hot path. **HIGH.**
  (https://www.anthropic.com/research/building-effective-agents · https://arxiv.org/pdf/2510.07920)
- **The 3/4-tier policy engine *exceeds* OWASP "Least Agency".** OWASP's Top-10 for Agentic Applications
  (2026) says autonomy "should be earned, not a default" and mandates human approval for high-impact
  actions. This system goes further: ESCALATION actions are **type-unrepresentable as executable** (the
  `ActionSpec` cannot carry an `execute` callable) rather than merely policy-checked — a genuinely rare,
  strong design. **HIGH.** (https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html)
- **The gated harness is the correct answer to multiple-testing / self-fooling.** Harvey-Liu-Zhu (*RFS*
  2016) argue a new factor needs **t > ~3.0**, not 2.0, after correcting for the hundreds of strategies
  tried; a single p<0.05 is uninformative. Bailey & López de Prado's **Deflated Sharpe Ratio** (2014,
  SSRN 2460551) corrects a Sharpe for selection bias, sample length, skew, and kurtosis — the exact
  inflations a self-improving search generates. **Pre-registration** (committing hypothesis + threshold
  before OOS) is the direct countermeasure to "second-order overfitting" (tuning the *validation setup*).
  This system already does DSR + Bonferroni + pre-registration + independent verifier — the single
  strongest thing in the design relative to the literature. **HIGH.**
  (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 · Harvey-Liu-Zhu 2016)
- **Champion-challenger / shadow lanes + "nothing promotes without the gate"** is textbook progressive-
  delivery MLOps (MLflow, DataRobot). The **versioned feature-pipeline contract that refuses cross-version
  inference** is a strong hand-rolled implementation of Google *Rules of ML* #32/#37 (reuse train/serve
  code; measure train/serve skew) — directly addressing this project's own C1 scaler-skew incident.
  **EWC + EMA + replay** in the transformer trainer is exactly the regularization+replay recipe the
  continual-learning canon (catastrophic-forgetting literature, arXiv 2509.18133) endorses. **HIGH/MEDIUM-HIGH.**
  (https://developers.google.com/machine-learning/guides/rules-of-ml)

**Improvable gaps (external best-practice this system does NOT yet fully implement), prioritized:**

1. **Add CPCV + a Probability-of-Backtest-Overfitting (PBO) number to every promotion.** An untouched
   single hold-out is exactly what Bailey et al. (SSRN 2326253) call "unreliable" for investment
   backtests; **Combinatorial Purged CV** yields lower PBO / higher DSR than walk-forward or plain purged
   k-fold (ScienceDirect S0950705124011110, 2024). **HIGH priority.**
2. **Move from Bonferroni to a correlation-aware multiple-testing adjustment**, and track a *cumulative*
   trial count across the whole self-improvement history, not per-session. Bonferroni over-penalizes
   correlated trials (can hide a real edge) while still admitting a lucky one. **HIGH.** (Harvey-Liu-Zhu 2016)
3. **Treat gate/threshold parameters as pre-registered and frozen; changing one becomes an ESCALATION
   action.** The verifier thresholds are now part of the search space an agent can tune — the subtlest,
   most system-specific self-fooling risk. **MEDIUM-HIGH.**
4. **Add live train/serve *distributional* skew monitoring** (distance between training baseline and live
   feature distributions), per Google Rule #37 — the version-pin catches *schema* skew but not *drift*. **MEDIUM.**
5. **Explicit model-registry lineage triple** (version → exact code commit + dataset hash + env) so every
   quarantined/promoted artifact is reproducible from the registry alone. **MEDIUM.** (MLflow)

### III.2 Where accessible edge actually lives for a small operator (honest map)

The prior is not a treasure map. Almost everything that reads like retail "edge" is one of: (a) a **risk
premium** (paid for bearing a tail/crash/illiquidity risk — real return, not mispricing); (b) an
**arbitraged-away anomaly** (decays ~58% post-publication, McLean-Pontiff 2016; capital-market anomaly
profits attenuated sharply post-2000, Hou-Xue-Zhang NBER w23394); or (c) a **capacity-constrained niche**
that survives *because* it is too small/illiquid/operationally annoying for institutions — which demands
labor and specific data, not a passive screen. Against that prior, ranked by evidence strength:

| # | Lever | Evidence | Honest nature | The catch |
|---|---|---|---|---|
| 1 | **Crypto funding / cash-and-carry** (delta-neutral spot-long + perp-short) | **MEDIUM-HIGH** (BIS WP 1087 frames the perp basis as segmentation/friction *premium*, not riskless arb) | Carry/liquidity premium, retail-sized, ~8–18%/yr normal regimes | Double funding (no cross-margin), forced-liquidation stress on the perp leg, **exchange-solvency tail**, self-crowding when funding is high. A job with a tail, not free money. |
| 2 | **Defined-risk volatility-risk-premium** (sell index-option insurance, small hedged sleeve) | **HIGH** premium is real/pervasive (Sharpe ~0.6 eq / ~1.0 global composite) | The crash-insurance premium in disguise | Catastrophic negative skew — **XIV −96% in a day (Volmageddon 2018); naive short-put ~−65% the month after COVID**. Sharpe *flatters* it. Never naked; size for the tail. |
| 3 | **Cost-disciplined PEAD on liquid mid-caps** | **MEDIUM-HIGH** (contested-but-defensible 2025; UCLA Anderson) | Residual post-earnings drift | Headline ~20%/yr is a **microcap + gross-cost illusion**; survives only with strict limit-order / liquid-name gates. Cost-fragile. |
| 4 | **Microcap/nanocap fundamental + insider-cluster confirmation** | **MEDIUM** (real institutional-exclusion advantage; OSAM) | Operational-labor edge | Highest transaction/impact costs *exactly here*; the size premium largely vanishes dropping sub-$5M names; the insider signal **goes negative once dollar-capped**. Forensic labor + fraud screening required. |
| 5 | **Spinoffs / catalyst CEF-discount value tilts** (long-only, patient) | **MEDIUM** | Slow catalyst value premium | Low frequency, multi-year holds, participates in post-2000 anomaly decay; the market-neutral CEF-short leg is usually **uncapturable on borrow**. |
| 6 | **Prediction-market longshot-fade / passive liquidity provision** (Kalshi/Polymarket) | **MEDIUM** | Maker-side judgment edge | Edge is on the *maker* side only, capped at tiny dollars; **takers lose ~32% avg**; the favorite-longshot bias is measurably weakening in 2025. |

**Dead / gated — do not re-pitch:** trend/TS-momentum & daily factors (risk premia *already* failing this
operator's effective-N/history gates — L-020/L-021); index reconstitution (arbitraged, engineered down,
now semi-annual); merger-arb at concentrated retail scale (cost drag halves gross, deal-break tail);
crypto market-making rebates & MEV searching (capital/latency-gated — **retail is the prey**); any options
"implied-information" *directional* alpha (priced away — see III.3). **The bottom line matches the internal
prior exactly: there is no accessible mispricing alpha left on this map — only risk premia you get paid to
bear and capacity-constrained niches that pay for operational labor.**

### III.3 Paid / alt-data — cost-vs-value at $100k (grounds the "edge needs data we don't have" claim)

The honest finding is sharper than "edge needs paid data": **paid data buys a cleaner null, not an edge**,
at this scale. Specifics (2025-26 pricing, independently checked):

| Rank | Data category | Approx. retail cost/yr | Plausibly unlocks | The catch (honest) |
|---|---|---|---|---|
| 1 | **Cheap options data** (ThetaData $40–80/mo, Polygon $29–199/mo) | $350–$1,000 | Real IV/skew/VRP signal → *vol-premium harvesting* | The directional skew/vol-spread edge is a **stock-borrow-fee proxy → not exploitable net of costs** (Muravyev-Pearson-Pollet, *JFE* 172, 2025; *J. Finance* 2025). You'd buy data to rediscover a fee you must then pay. Value is risk-premium *selling*, not prediction. **HIGH.** |
| 2 | **Aggregator starter stack** (Alpaca/IBKR/Tiingo) | $120–$1,200 | Clean execution + research-grade prices | Pure hygiene — zero alpha by itself. **HIGH.** |
| 3 | **PIT fundamentals** (Sharadar/EODHD/FMP; Compustat-PIT is institutional-only) | $150–$1,000 | Survivorship-bias-free factor research | The factors it "cleans" are **half-decayed** (McLean-Pontiff 2016: −26% OOS / −58% post-publication); this operator *already* found value/quality/accruals fail on a free true-PIT EDGAR pipeline. Cleaner null, not new edge. **HIGH.** |
| 4 | **Tick / microstructure** (Databento $199/mo, Polygon, Kibot) | $200–$2,400 | Intraday backtest fidelity; crypto funding | Execution edge is **latency-gated** — unusable without colocation. Research-only. **HIGH.** |
| 5 | **Institutional alt-data** (satellite/card panels/RavenPack) | **$25k–$1M+** | Genuine near-real-time consumer/spend lead | **Priced for funds, crowds → fast alpha decay.** A single feed ≈ 25–75% of the operator's entire capital. Categorically irrational at $100k. ("Alt-data +3%/yr" is **vendor-adjacent marketing**, not independently verified.) **HIGH.** |

**The ruin/capacity math:** a serious retail research stack (~$1,500–$3,600/yr) is a **~3% fixed annual
drag** at $100k, *before a single trade*. Any edge paid data plausibly unlocks is tiny, half-arbitraged,
or (options) definitionally unexploitable without a cheap short-borrow book. **A 3% cost hurdle on a
decayed sub-1% edge is negative expected value.** This *confirms* the operator's prior null; it does not
overturn it. If the operator buys anything, buy **cheap options data (~$40–80/mo) reframed as a
vol-risk-premium tool**, plus the sub-$1k/yr aggregator floor for hygiene — everything above is negative-EV.

---

## Part IV — FORWARD ROADMAP (prioritized, honest) + DURABLE FACTS

### IV.1 The realistic ceiling — state it plainly, up front

**This is a risk-premium harvesting book and a verification/safety engine — NOT a path to 10x on free
data.** Every edge search closed at "no accessible mispricing alpha." The literature says the same. So the
forward plan is *not* "find the alpha we missed"; it is **(a) harden the machine that keeps us honest and
safe, (b) redirect ML skill from the dead 52% direction target to the risk targets where skill demonstrably
helps, and (c) if the operator chooses to spend, spend on the one lever class with a defensible prior —
risk-premium harvesting (crypto funding-carry, defined-risk VRP) — sized for its tail, not on alpha-chasing
paid data.** Realistic expectation for a well-run risk-premium book at this scale: modest positive Sharpe
(≈0.4–1.0) *conditional on surviving a fat left tail*, not a compounding-to-riches curve. Anyone who
promises more is selling something.

### IV.2 Roadmap (fuses the internal training-architecture audit P0–P3 with the external gaps)

Ordering principle: **honesty/wiring fixes first (cheap, no operator decision, close active lies), then
consolidation, then operator-gated levers.** Nothing below touches the hot path without an explicit
escalation; everything ships with no-mock tests + separate-verifier PASS per the Definition of Done.
Source for internal items: `docs/training-architecture-audit-2026-07-03.md` (Phase 3).

**P0 — Truth & safety patches (days; no operator decision):**
- **Heartbeat honesty** — `scanner_alive` true only when the engine loop is actually alive; supervisor gets
  its own key; extend `running_status.py` to all lanes. (Fixes the lying beacon; L-017.)
- **Eval gate on `online_retrainer`** — holdout + refuse-to-write on degradation (quarantine-shaped). It
  currently rewrites sklearn gate models in place with *zero* validation.
- **Alert routing + ack ledger** — route the standing 7-consecutive-loss WARNING class to a surfaced action.
- **Trend-lane journal enrichment** — write regime/ATR/spread/slippage even without 15-agent context, so
  forward records become trainable.
- **Per-lane coverage in `risk_monitor.sh`** — the tripwire currently checks only the global halt flag.

**P1 — Consolidation (1–2 weeks):**
- **Extract a `gated_harness` library** from the copy-pasted experiment scripts; **add CPCV + a PBO number
  + regime-stratified OOS** (external gap #1), and **cumulative-trial-count, correlation-aware multiple
  testing** (external gap #2). Port one existing experiment as a byte-identical regression proof.
- **Single post-trade feedback module** — collapse the three divergent paths (`sync_closed_trades_rl`,
  `post_trade_loop`, `trend_journal_sync`); delete-by-merge the duplicate AlertManager/calibrator twins.
- **Headless Learning Supervisor** — a bounded daemon (same shape as the Tier-7 supervisor) that runs the
  canonical feedback path and *consumes* approved config adjustments, closing the producer-alive/
  consumer-dead asymmetry. Read-only w.r.t. trading; never unhalts; Claude stays outside the loop.
- **Second-order-overfitting guard** — pre-register and freeze gate/threshold params; a threshold change
  becomes an ESCALATION action (external gap #3).

**P2 — Operator-decision levers (surface, don't start):**
- **Activate `tick_capture.py`** against the practice account — *free*, and it builds the "materially new
  input" precondition L-022 requires before any future intraday research is even admissible. Zero hot-path
  contact.
- **Fill-quality / execution-cost model** from the 188+ accumulating ORDER_FILL book-ladders — legitimate
  ML that pays regardless of alpha, and the honest way to keep the harness cost model calibrated.
- **Add distributional train/serve-skew monitoring** (external gap #4) and formalize the **registry lineage
  triple** (external gap #5).
- **Paid data — only if the operator chooses**, and only the defensible slice: cheap options data reframed
  as a **VRP tool**, not directional alpha. Grounded null: PIT fundamentals buy a cleaner null (III.3).

**P3 — Strategy direction (operator's call; each is an ESCALATION, none is autonomous):**
- **Redirect ML to risk targets, not direction.** Volatility forecasting, drawdown-state estimation, regime
  classification feeding sizing/overlays — skill demonstrably exists here and the efficient-market wall does
  *not* apply. This improves the risk-premium lanes (trend, harvester) without needing to beat 52%.
- **The candidate risk-premium sleeves, shadow-first, ranked by prior** (III.2): (1) **crypto funding /
  cash-and-carry** — the strongest live retail-sized lever, run it as a *new Lane-Contract shadow lane*
  with the harness gating it before any arm; (2) **defined-risk VRP** — only if cheap options data is
  bought, only hedged/small-sleeve, sized so a −65% month is survivable. Both are risk premia with tails —
  label them as such (L-021), never as "alpha", and they arm only through the operator-typed `"LIVE"` token.
- **Lane Contract unification** — migrate the legacy FX machinery onto the pattern the new lanes already
  prove (fail-closed halt, hash-chained ledger, pre-registered params, ship gate, LiveGate, honest liveness,
  per-lane decay monitor). Finish the joint-fallback deprecation.

**What the roadmap deliberately does NOT contain:** FX direction retrains or "one more" free-data alpha
hunt (L-016/L-022); transformer scaling (signal-bound, not capacity-bound — already rejected); any
relaxation of HARD_MAX_GAP / halt / practice-pin / ship-gate-before-champion; any autonomous promotion to
live.

### IV.3 The top priorities in one breath

1. **P0 heartbeat honesty + online-retrainer eval gate** — stop the beacon lying and close the one ungated
   model-write path (safety/truth, days).
2. **P1 `gated_harness` library with CPCV + PBO + correlation-aware multiple testing** — turn the crown-jewel
   verification logic from copy-discipline into structure, and adopt the two external upgrades the literature
   demands.
3. **P1 Headless Learning Supervisor** — make learning survive the TUI being closed; close the producer/
   consumer dead-write asymmetry.
4. **P2 activate free `tick_capture.py` + fill-quality model** — build the only "materially new input" that
   is free, and the execution-cost model that pays regardless of alpha.
5. **P3 (operator's call): stand up crypto funding-carry as a shadow Lane-Contract sleeve, harness-gated,
   labeled risk-premium** — the single most defensible forward lever, arming only via the `"LIVE"` token.

---

## DURABLE FACTS FOR MEMORY (fold into the persistent brain)

- **The bot is a risk-premium harvesting + verification/safety engine, NOT a free-data alpha machine.** Six
  free-data doors are closed; the realistic ceiling is a modest-Sharpe risk-premium book with a fat tail —
  never 10x on free data. State this ceiling in any strategy discussion.
- **No free-data, small-operator return-ALPHA exists** (L-022) and the *published record agrees*: FX can't
  beat a random walk OOS (Meese-Rogoff 1983); anomalies decay ~58% post-publication (McLean-Pontiff 2016);
  the surviving anomaly premium is eaten by stock-borrow fees — *"the borrow fee IS the anomaly"*
  (Muravyev-Pearson-Pollet, *J. Finance*/*JFE* 2025); trend is a risk premium (Moskowitz-Ooi-Pedersen 2012).
- **Trend / TS-momentum is a RISK PREMIUM, not alpha** (L-021) — crisis-convex, but ran <½ its historical
  return in the QE era. Report it on two axes (risk-control vs alpha-gate); never call it "edge."
- **Paid data buys a cleaner null, not an edge, at $100k.** A ~$3k/yr stack ≈ 3% annual drag → negative-EV on
  a decayed sub-1% edge. Options data is the only cheap *real* signal, but it's a borrow-fee proxy →
  reframe as a **vol-risk-premium tool**, not directional alpha. Institutional alt-data (satellite/cards) is
  categorically irrational at this scale.
- **The strongest genuinely-live retail lever is crypto funding / cash-and-carry** — a carry/segmentation
  premium (BIS WP 1087), ~8–18%/yr in normal regimes, but with a real exchange-solvency/liquidation tail and
  self-crowding. It is a job with a tail, not free money.
- **The safety spine is structural, not advisory.** Practice pin re-derived every policy call
  (`config.py:742`, `execution.py:801`); fail-closed per-lane halt (`state_engine.get_halted_strict`,
  `:269-328`); mid-cycle halt re-check fail-closed (`execution.py:2081-2114`); ESCALATION actions
  *type-unrepresentable as executable* (`policy.py:112-122`); LiveGate needs a typed `"LIVE"` token
  (`live_gate.py:84,445-568`); agent self-improve denylist/allowlist (`self_improve.py:61-92`).
- **The gated research harness is the crown jewel** (pre-registration + untouched OOS + Deflated Sharpe +
  Bonferroni bootstrap + independent verifier) — it repeatedly caught in-sample optimism that would have
  shipped false positives. Upgrade path: CPCV + PBO + correlation-aware multiple testing; guard the gate
  params themselves against second-order overfitting.
- **This design already meets or beats published best-practice**: LLM out of the hot path (neutralizes the
  "Profit Mirage" backtest-leakage hazard), policy tiers exceed OWASP "Least Agency," feature-version
  contract implements Google Rules-of-ML #32/#37 against train/serve skew.
- **The real modernization target is runtime learning *wiring*, not the model.** Learning loops are welded to
  the dormant TUI; self-heal produces adjustments only the dead TUI can consume; the online retrainer is
  ungated; three divergent post-trade paths coexist. Fix the plumbing (Headless Learning Supervisor, single
  feedback path), don't chase accuracy.
- **Redirect ML from the dead 52% direction target to risk targets** (volatility, drawdown-state, regime) —
  skill exists there and the efficient-market wall doesn't apply; it improves the risk-premium lanes.
- **Report both harvester numbers, always:** 0.908 curated-20-name vs **0.740 full / 0.355 OOS gate-FAIL** on
  the survivorship-corrected wide universe. It is beta, honestly labeled — not alpha.
- **Track B is INSUFFICIENT / underpowered, not "no edge"** — rank-IC faded +0.16→+0.09 as N grew; needs
  ~405 scored filings across ≥3 rebalances to settle. Don't harden an underpowered result into a measured zero.
- **`mode: live` ≠ real money** — the environment/endpoint is the real-money guard (L-014); broker is
  hard-pinned to `api-fxpractice.oanda.com`. A practice bot in `mode=live` trades paper.

