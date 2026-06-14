# Tier 7 → SOTA 2026: Executive Summary (1-Page)

**Date:** 2026-06-14 | **Author:** Dex | **Status:** Draft — pending operator review

---

## Where You Are (Tier 7 Today)

A production-grade hybrid quant stack with 15 specialist agents, a custom CNN-Transformer signal core, neural agent policies with online learning, meta-learning (MAML), confidence calibration (Platt scaling), and a policy engine with enforcement modes. Correctly diagnosed as **edgeless for intraday directional trading** (~52% val, leakage artifact). Factor portfolio pivot is the right strategic response.

**Strengths that transfer:** Risk plumbing, holdout harness, promotion gates, agent weight learning, safe JSON persistence, fail-closed posture.

## Where SOTA Is (2025–2026 Frontier)

The frontier has moved from "train a custom model on your data" to **"adapt a pre-trained foundation model with LoRA, then reason over a knowledge graph with an LLM agent."** The 12 dimensions:

1. **Foundation models** for time series (TimesFM, MOIRAI, Chronos) — zero-shot transfer from billions of tokens
2. **LLM agents with tool-use** — runtime macro reasoning, not just sentiment scoring
3. **Diffusion models** — synthetic market simulation for stress testing
4. **Offline RL** (CQL/IQL) — learn from historical journals without live exploration
5. **Causal ML** — SCM + do-calculus, not just Granger causality
6. **Multi-modal** — central bank audio tone, option flow imagery
7. **Neural operators** — continuous-time modeling (Neural SDEs)
8. **Knowledge graphs + RAG** — structured macro reasoning with retrieval
9. **Deep RL execution** — SAC/TD3 for optimal order placement
10. **Federated learning** — train across asset classes without centralizing
11. **Neural Architecture Search** — AutoML for strategy discovery
12. **Quantum-classical hybrids** — portfolio optimization on quantum annealers

## The 5-Phase Upgrade Path

| Phase | What | Timeline | Revenue Impact |
|-------|------|----------|----------------|
| **0** | Ship factor portfolio (carry + trend + value, weekly rebalance) | 4 weeks | **Immediate** — replaces edgeless intraday stack |
| **1** | Wire LLM agents + knowledge graph + macro reasoning | 6 weeks | High — explains *why* factors work |
| **2** | Integrate time-series foundation model (TimesFM/Chronos) | 8 weeks | High — zero-shot transfer learning |
| **3** | Offline RL for rebalancing + execution + RLHF risk alignment | 12 weeks | Medium-High — reduces turnover, improves fills |
| **4** | Causal inference (SCM, do-calculus) on factor drivers | 12 weeks | Medium — predicts regime breakdowns |
| **5** | Autonomy: AutoML + code-generating agents + federated learning | Ongoing | Medium — compounding R&D |

## The Two Paths This Week

**Path A — Disciplined Mechanics:** Implement US-001 (daily data layer) from the factor portfolio PRD. Pure engineering, no model risk. Produces a track record in weeks.

**Path B — Intelligence Layer:** Shadow-deploy 3 of your 37 `.claude/agents/` prompts as runtime LLM agents with FRED tool-use. Biggest SOTA gap closure. Needs a decision on which personas get a vote.

## Highest-Impact Single Action

> **Download TimesFM-base (open weights) and run zero-shot forecasting on your daily EUR_USD data.** Compare MAE vs. `RawSequenceModel`. This 2-hour experiment validates whether Phase 2 is worth the investment.

---

*Full analysis: `docs/TIER7_VS_SOTA_2026_UPGRADE_PATH.md` (441 lines)*
