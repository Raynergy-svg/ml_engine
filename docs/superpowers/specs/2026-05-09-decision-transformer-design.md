# Decision Transformer / Offline RL — Design Spec

**Status:** SCAFFOLDING ONLY (Phase 1 — design + stubs, no model, no training)
**Date:** 2026-05-09
**Author:** Claude (AI engineering agent, in collaboration with operator)
**CLAUDE.md mapping:** Modernization stance §6 — "Decision Transformer / offline RL on trade journal — DEFER, needs >5K trades first"
**Predecessor work:** PPO position sizer (`src/training/rl/position_sizer.py`), trade journal accumulator (`trained_data/trade_journal_rl.json`), Phase 5.D pair tuning (USD_JPY/EUR_USD)

---

## 1. Goals + Non-Goals

### 1.1 Goals
- Replace the *decision* part of the stack (current: per-pair Transformer direction head + ridge confidence + DynamicPositionSizer + 15-agent voting) with a single sequence-modeling policy that maps `(state, return-to-go, action)` trajectories to next-action distributions.
- Learn from the **actual trade journal** (real fills, real slippage, real outcomes) rather than supervised labels on price-only history. Trade outcomes encode information that price alone does not — gate rejections, agent disagreement at decision time, regime, drawdown, fill quality.
- Stay strictly **offline**. No live policy iteration. The policy is trained once per journal-snapshot, validated on a holdout, and either promoted via the meta-pipeline (shadow → canary → live) or shelved.
- Preserve the current decoupling: prediction (direction + confidence) stays as today; the DT augments or replaces the *gate-and-size* layer. The DT does not retrain the volatility regime head or the news/macro pipeline.

### 1.2 Non-Goals (explicit)
- **Not online RL.** No live exploration. The bot's runtime stays Claude-free and deterministic; the DT policy is a frozen `nn.Module` loaded at scan time (or, in shadow mode, a parallel evaluator that doesn't influence execution).
- **Not a replacement for the agent team.** Agent verdicts feed the DT *state*; they don't get retrained by it. The 15-agent ensemble is part of the input, not the output.
- **Not a position sizer replacement (yet).** The existing PPO position sizer stays; the DT outputs discrete action {LONG, SHORT, NO_TRADE}. Continuous sizing is a Phase-3 extension if and only if Phase-2 (discrete) shows holdout lift.
- **Not a continuous-time / market-making policy.** Discrete decisions at scan boundaries only.

## 2. Survey + Recommendation (one-paragraph synthesis)

The four candidates are **Decision Transformer (DT)**, **Implicit Q-Learning (IQL)**, **Conservative Q-Learning (CQL)**, and **Trajectory Transformer (TT)**. CQL and IQL are value-based: they learn a Q-function from the journal then act greedily, with regularizers (CQL's pessimism penalty, IQL's expectile regression on V) to avoid querying out-of-distribution actions. Both are stronger than DT *when the dataset has good action coverage and meaningful reward density* — neither holds for our journal: action coverage is biased (the existing gates only let "high-confidence" trades through, so we have almost no NO_TRADE labels with full state context, and almost no SHORT labels in regimes that historically went LONG), and reward is sparse + noisy (one realized P/L per trade, ~0–10 trades per pair per day). DT and TT are sequence-modeling: they treat offline RL as supervised learning on `(R̂_t, s_t, a_t)` triplets with a return-to-go conditioning at inference time, sidestepping the OOD-action problem entirely. DT is the simpler of the two (single autoregressive Transformer over flat token streams; TT adds beam-search planning at inference time which we don't need). **Recommendation: Decision Transformer.** Justification: (1) sequence-modeling fits the bot's natural time-series structure (the trade journal *is* a sequence ordered by close_time); (2) DT does not require Q-learning over discrete actions, which removes the OOD-action failure mode that bites IQL/CQL on our biased dataset; (3) existing PPO sizer code in `src/training/rl/position_sizer.py` proves the team can ship `nn.Module`-based RL artifacts with the same training-control-plane plumbing (W&B configs, walk-forward validation, calibration); the DT is a drop-in replacement for the policy head, not a new infrastructure burden; (4) inference is one forward pass with a fixed return-to-go target (e.g., +1.5R conditional) — no planning loop, fits the runtime hot-path latency budget. Cost-of-being-wrong: low — if DT underperforms the existing gate stack on the holdout, we keep the existing stack; the journal data we accumulate to train DT also feeds CQL/IQL as a Phase-4 fallback at zero marginal cost.

## 3. Architecture

```
                              ┌───────────────────────────────────────┐
                              │   trade_journal_rl.json (closed only) │
                              │   virtual_trades.jsonl (gate-rejected)│
                              └───────────────┬───────────────────────┘
                                              │
                                              v
        ┌─────────────────────────────────────────────────────────────┐
        │ TrajectoryLoader  (src/training/rl/trajectory_loader.py)    │
        │   - per-pair grouping (USD_JPY trades form one trajectory)  │
        │   - chronological sort by entry_time                        │
        │   - state vector: [agent_verdicts(15) | confidence | regime │
        │                    one-hot(4) | drawdown | atr_pips |       │
        │                    model_disagreement | uncertainty | ...]  │
        │   - action: discrete {0=NO_TRADE, 1=LONG, 2=SHORT}          │
        │   - reward: realized_pl / sl_risk_dollars (R-multiples)     │
        │   - return-to-go: cumulative future R-multiples             │
        │   - context window K (default 32 trades) → tensor batches   │
        └────────────────────────┬────────────────────────────────────┘
                                 │  Tensor batches: (B, 3K, d_model)
                                 v
        ┌─────────────────────────────────────────────────────────────┐
        │ DecisionTransformer  (src/training/rl/decision_transformer) │
        │   - causal Transformer (d_model=64, layers=2, heads=4)      │
        │   - input embedding: state_proj | action_embed | rtg_proj   │
        │   - output: action logits at every "action token" position  │
        │   - loss: cross-entropy on actions only (DT-original recipe)│
        │   - inference: condition on rtg_target = +1.5R, return      │
        │                argmax(action_logits) at the next slot       │
        └────────────────────────┬────────────────────────────────────┘
                                 │
                                 v
        ┌─────────────────────────────────────────────────────────────┐
        │ OfflineRLTrainer  (src/training/rl/offline_rl_trainer.py)   │
        │   - walk-forward CV (purged k-fold, embargo)                │
        │   - W&B logging (action-accuracy, expected-R, sharpe-on-holdout) │
        │   - artifact: trained_data/models/decision_transformer.pt   │
        │   - meta-pipeline handoff: ChangePackage with shadow gates  │
        └─────────────────────────────────────────────────────────────┘
```

## 4. Trajectory Schema Decisions

| Slot | Choice | Rationale |
|---|---|---|
| **State `s_t`** | 32-dim (compact). 15 agent scores + confidence + regime one-hot(4) + drawdown_pct + atr_pips + model_disagreement + uncertainty + group_momentum + spread_pips + 5 reserved slots | Already-extracted journal features; no new feature engineering. Compact = trains on small data. Price-feature embedding is intentionally NOT in `s_t` — direction head is upstream, and re-embedding price would compete with the (better-trained) Transformer direction head |
| **Action `a_t`** | Discrete 3-way: {0 = NO_TRADE, 1 = LONG, 2 = SHORT} | Continuous size deferred to Phase 3 (after discrete shows lift). Discrete = stable on small data; matches the gate-decision the DT is replacing |
| **Reward `r_t`** | R-multiples: `realized_pl / sl_risk_dollars`. Clipped to `[-1.0, +5.0]` (lose at most 1R, cap windfalls). NO_TRADE gets `r=0`. | Pip-PnL is unit-inconsistent across pairs; R-multiples make USD_JPY and EUR_USD trajectories comparable. Cap prevents one outlier trade from dominating gradient |
| **Return-to-go `R̂_t`** | `Σ_{i≥t} r_i` over a fixed horizon (default last-trade-of-pair OR 50 trades ahead, whichever first) | Bounded horizon prevents R̂ from blowing up on long histories; matches DT-original (Chen 2021). Inference uses target `R̂_target = +1.5` (1.5R per trade compounded) |
| **Trajectory** | **Per-pair**, chronologically sorted by entry_time. Each pair = one trajectory. NO_TRADE slots inserted at scan timestamps where gates rejected (from `virtual_trades.jsonl`) | Per-pair preserves pair-specific dynamics (USD_JPY ≠ EUR_USD). Inserting rejected scans gives the model a NO_TRADE supervision signal it would otherwise lack |
| **Context length K** | 32 (~1 trading day at H1 with 4 pairs) | Long enough to capture intra-day regime, short enough that 5K trades = 156 sequences per pair, enough to fit a tiny DT |
| **Pair conditioning** | Pair-id embedded as a learnable 4-dim vector concatenated to `s_t` | Lets one shared DT generalize across pairs while still being pair-aware. Avoids per-pair model proliferation |

## 5. Training Data Requirements

| Threshold | Trades needed | Current count | Action |
|---|---|---|---|
| **Smoke** (does the loader run end-to-end?) | 100 | 18 | Replay against historical with current gates → fast forward to ~500 in days |
| **Phase-2 train** (discrete DT, single-pair) | 1,000 per pair | 18 total | Blocked. Need forward-test acceleration OR replay |
| **Phase-2 promote** (holdout lift vs current stack) | 5,000 total | 18 total | Blocked. CLAUDE.md explicit threshold |
| **Phase-3 extension** (continuous sizing) | 10,000 total | 18 | Blocked + sequenced after Phase-2 |
| **Phase-4 fallback** (CQL/IQL on same journal) | 5,000 total | 18 | Blocked. Same data; reuses TrajectoryLoader |

**Forward-test rate (current Phase 5.D state):** at H1 with 4 pairs (USD_JPY, EUR_USD, EUR_JPY, GBP_USD) at ~3–5 trades/day → 60–100 trades/month → **5,000 trades = 50–80 months**. Unviable as-is.

**Acceleration paths (sequenced from cheapest):**
1. **Dry-run replay** against historical 2024-2026 candles using current gates + Phase 5.D models. Each pair-month produces ~100 simulated trades; 24 months × 6 pairs = ~14,000 simulated trades. Caveat: replay does not reproduce realized slippage/fill quality, so reward-distribution will be optimistic. Mitigate by injecting empirical slippage from the 18 real trades' `slippage_pips` distribution.
2. **Lower granularity** to M15 across all pairs (CLAUDE.md modernization §3 already plans this for direction-head retraining). M15 ≈ 4× the trade rate of H1. Combined with replay: 5K trades reachable in **weeks**, not years.
3. **Multi-pair pooling** with pair-id embedding (already planned in §4) — doesn't add trades but reduces per-pair data requirements by ~3× via shared backbone.

## 6. Validation Strategy

- **Walk-forward, purged k-fold, embargo** — same plumbing as `src/training/walkforward_validation.py`. Five folds, 5-trade embargo between train/test boundaries to prevent overlap leakage.
- **Primary metric:** expected R-multiple per decision on holdout, conditional on `R̂_target = +1.5`. Compared to the current gate-stack's expected R on the same holdout slice.
- **Secondary metrics:** action-accuracy (DT prediction vs operator-graded "correct" homework label, when available), Sharpe on the holdout's reconstructed equity curve, max-drawdown, win-rate.
- **Decision rule:** Phase-2 ships only if holdout-R is **≥ 0.10R better** than the current stack with **p < 0.05** under a paired bootstrap test on per-trade R. Anything less is shelved; the journal data still feeds CQL/IQL as Phase-4 fallback at no marginal cost.
- **Catastrophic guard:** if the DT recommends a trade that violates **any existing trading rule** (`.claude/rules/trading.md` — R:R < 1.2:1, trend-veto, MR-veto, staleness-block), the DT recommendation is **discarded** and the existing gate stack's verdict applies. The DT is allowed to *refuse* (NO_TRADE) but not allowed to *override* hard rules. This is non-negotiable.

## 7. Phasing

| Phase | Trigger | Deliverable | This commit? |
|---|---|---|---|
| **P1** | Today | Spec + stubs + tests (this commit) | YES |
| **P2-prep** | Replay pipeline yields ≥1K trades on EUR_USD | Implement `TrajectoryLoader` + smoke test (loader → tensor shapes match contract) | NO |
| **P2-train** | ≥5K trades total (real + replayed) | Implement `DecisionTransformer.forward` + `OfflineRLTrainer.train_one_epoch`. Single-pair holdout vs Phase 5.D EUR_USD baseline. Decision rule §6 | NO |
| **P3** | P2-train ships ≥0.10R lift | Continuous-action extension: replace softmax(3) with Gaussian head over position-size in [0, 0.05] NAV-fraction. Re-validate vs PPO sizer | NO |
| **P4** | P2-train fails to ship | CQL/IQL fallback on same journal. Reuses TrajectoryLoader. Tests if the failure was DT-specific or data-specific | NO |

## 8. Risks + Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Journal too small even with replay | Medium | High | Replay at M15, pool pairs via pair-id embedding, defer P2 until threshold met. Decision criterion §5 is explicit |
| Replay reward distribution diverges from live (slippage, fills) | High | Medium | Inject empirical slippage from real-trade journal. Gate P2-train on a real-data-only validation slice (≥500 real trades) before promotion |
| DT learns to mimic existing gate stack (no lift) | Medium | Low (just a non-promotion) | Decision rule §6 is explicit: ship only on ≥0.10R lift. Otherwise journal feeds Phase-4 |
| DT recommendation violates a trading rule | Low | Catastrophic | §6 hard guard: trading rules veto the DT, not vice-versa. Non-negotiable |
| Operator overrides safety guard | Low | Catastrophic | Guard lives in code (`offline_rl_trainer.py:_apply_runtime_safety_filter`), not in config. Cannot be flipped via `config_adjustments.json` |
| Foundation-model direction head ships first and obsoletes the action space | Low (CLAUDE.md DEMOTED foundation models 2026-05-08) | Medium | DT consumes the direction head's output as `s_t` regardless of architecture; backbone-agnostic by design |
| News/macro signal (P1 promoted, see CLAUDE.md) lifts price-only baseline above DT's holdout target | Medium | Low (positive outcome — we ship the better thing) | DT's `s_t` schema reserves 5 slots; news features map there in P3 if news ships first |

## 9. Dependencies

- **Forward-test volume**: critical-path. Without acceleration (M15 + replay), 50+ months. With acceleration, weeks.
- **Phase 5.D models stay deployed**: the gates that produce journal entries depend on the deployed direction heads. Reverting Phase 5.D would invalidate replay.
- **Replay pipeline**: not yet built. One-off script (`scripts/replay_journal.py`) needed to run gates against historical candles with empirical slippage injection. Estimated 1 day of work, reusable for any future offline-RL approach (DT, CQL, IQL).
- **Trading-rule veto integration**: `OfflineRLTrainer._apply_runtime_safety_filter` must call into `src/scanner/gates.py:GateEvaluator` for the rule-veto check. Coupling is one-directional (RL → gates), no circularity.

## 10. Earliest Viable Training Date

Given (a) current journal = 18 trades, (b) replay pipeline = 1 day to build, (c) M15 retraining of 4 pairs = 1 day (per CLAUDE.md modernization §3 estimate), (d) replay sweep across 24 months × 6 pairs ≈ 1 day to run, (e) ≥500 real trades arrive at M15 forward-test rate (~400/month) in ~5 weeks: **earliest viable Phase-2-train start = 2026-06-15**, contingent on operator approving the M15 expansion and replay-pipeline build. Without those, the threshold is reached organically in **late 2028**.

---

## References

- Chen et al. 2021. "Decision Transformer: Reinforcement Learning via Sequence Modeling." NeurIPS.
- Kumar et al. 2020. "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS.
- Kostrikov et al. 2021. "Offline Reinforcement Learning with Implicit Q-Learning." ICLR.
- Janner et al. 2021. "Offline Reinforcement Learning as One Big Sequence Modeling Problem." NeurIPS (Trajectory Transformer).
- López de Prado 2018. *Advances in Financial Machine Learning* — purged k-fold + embargo.
- CLAUDE.md "Modernization stance" §6, "Investment priority" item 6 (DEFER triage).
- `src/training/rl/position_sizer.py` — PPO precedent for shipping RL artifacts.
- `.claude/rules/trading.md` — non-negotiable runtime rule veto.
- `.claude/rules/improvement.md` — No-Mock Rule (this scaffolding's tests use real disk via `tmp_path`).
