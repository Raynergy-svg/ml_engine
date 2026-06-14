# Tier 7 Agentic Trading Bot vs. SOTA Quantitative Trading (2025–2026)
## Comprehensive Gap Analysis & Upgrade Path

**Author:** Dex (systematic codebase audit + SOTA research)  
**Date:** 2026-06-14  
**Scope:** Full stack — signal generation, agent consensus, execution, risk, meta-cognition, infrastructure  
**Sources of truth:** `src/scanner/agents/_team.py`, `src/sota_core/`, `src/scanner/agents/neural/`, `tasks/prd-fx-factor-portfolio.md`, `docs/CODE_REVIEW_SOTA_NEURAL.md`, `src/recursive_intelligence/`, `src/strategy_invention/`

---

## 1. Executive Summary: Where Tier 7 Stands Today

Your Tier 7 system is **already in the top decile** of retail/proprietary hybrid quant stacks. You have:

- **15 specialist agents** with regime-aware weighted voting and RL-driven weight adaptation (`_team.py`)
- **A custom CNN-Transformer SOTA core** (`src/sota_core/`) with variable selection networks, two-phase self-supervised pre-training, and adaptive confidence thresholds
- **Neural agent policies** (`src/scanner/agents/neural/`) with prioritized experience replay, online updates, and binary-crossentropy targets (CRIT-4 fixes applied today)
- **Meta-learning**: MAML Ridge (`maml_ridge.py`) and differentiable ensemble weighting (`ensemble_weighting.py`)
- **Policy engine** with action types, rule evaluation, and logging-only enforcement modes
- **Confidence calibration** (5-layer Platt scaling, ensemble disagreement, time decay)
- **Factor portfolio pivot** in draft (`tasks/prd-fx-factor-portfolio.md`) — carry + trend + value across G10
- **Episodic memory**, **concept drift detection**, **causal filtering**, **smart execution (TWAP)**, **per-pair routing**, **holdout harness**, **promotion gates**

**The hard truth:** The intraday direction stack was correctly diagnosed as edgeless (~52% val, leakage artifact). The factor portfolio pivot is the right strategic move. But even within that pivot, there is a **generational gap** between what you have built and what the 2025–2026 frontier is doing.

---

## 2. SOTA Quantitative Trading Circa 2025–2026: The 12 Frontier Dimensions

Based on the trajectory from NeurIPS 2024, ICML 2024, the JPMorgan / Goldman / Two Sigma research pipelines, and the open-source quant-AI ecosystem, here is what "state of the art" looks like right now:

### 2.1 Foundation Models for Time Series (The biggest architectural shift)

**What SOTA looks like:**
Pre-trained models trained on **millions** of time series (retail sales, electricity, weather, sensor data, *and* price) that generalize across domains without hand-engineered features. Key architectures:
- **MOIRAI** (Salesforce, 2024): Probabilistic forecasting with learned marginal distributions. Zero-shot transfer to new instruments.
- **TimesFM** (Google, 2024): Decoder-only Transformer pre-trained on 100B time-series tokens. Fine-tuned with LoRA per asset class.
- **Chronos** (Amazon, 2024): Tokenizes continuous values into a vocabulary and trains like a language model. Handles arbitrary horizons.
- **TinyTimeMixers** (IBM, 2024): Lightweight MLP-Mixer variants that run on edge devices but match Transformer performance.

**Your gap:**
Your `src/sota_core/raw_sequence_model.py` is a **custom CNN-Transformer** trained from scratch on your own OHLCV data. This was SOTA in 2022–2023. In 2025–2026, the frontier has moved to **massive pre-trained foundation models** that transfer knowledge from non-financial time series and are fine-tuned with parameter-efficient adapters (LoRA, DoRA). Your model discovers trend/momentum in latent space — but so does TimesFM, and it learned it from 100B tokens instead of 5,000 candles.

**Severity:** 🔴 **High** — This is the single biggest architectural debt in your signal layer.

### 2.2 Generative AI / LLM Agents as Runtime Decision Makers

**What SOTA looks like:**
LLMs are no longer just "sentiment analysis on headlines." They are:
- **Tool-using agents** that query APIs (FRED, BIS, central bank calendars, corporate filings), synthesize macro reasoning, and emit structured trade recommendations.
- **Portfolio managers** that maintain a "memory" of economic narratives and adjust factor exposures based on narrative shifts (e.g., "the disinflation narrative is breaking down").
- **Code-generating strategists** that write, backtest, and promote new strategy variants autonomously (AlphaTensor-like discovery but for trading logic).

Production examples circa 2026:
- **BloombergGPT-3** derivatives that generate structured factor commentary
- **Open-source FinGPT agents** with ReAct-style tool use over financial APIs
- **Academic experiments** (University of Chicago, 2024–2025) showing GPT-4-level models can generate alpha from earnings call transcripts when paired with retrieval

**Your gap:**
You have **37 LLM personality prompts** in `.claude/agents/` — but the AGENTS.md explicitly states these are "reference material for engineering, testing, and strategy roles." They are **NOT runtime trading agents.** Your `news_risk_agent` does keyword scanning (NFP, CPI, FOMC). There is no LLM agent that reads a central bank statement, queries a knowledge graph, and votes in the agent ensemble.

**Severity:** 🔴 **High** — You have the prompts but no runtime wiring. The gap is engineering, not research.

### 2.3 Diffusion Models for Market Simulation & Stress Testing

**What SOTA looks like:**
Diffusion models (score-based generative models) trained on historical returns to generate **realistic alternative market paths**.
- **Scenario generation:** "What if the 2022 vol spike happened in a high-rate environment?" Generate 10,000 plausible paths.
- **Data augmentation:** Train models on synthetic + real data to improve tail robustness.
- **Counterfactual simulation:** "What would my portfolio have done if the BoJ had hiked in March 2024?"

Key papers: 2024 NeurIPS workshops on generative models for finance; SDE-GAN successors.

**Your gap:**
You have `src/strategy_invention/` (genetic algorithms) and `self_refine.py`. No diffusion-based simulation. Your backtest harness uses historical data only. You cannot generate synthetic stress scenarios that never happened but are statistically plausible.

**Severity:** 🟡 **Medium-High** — Critical for the factor portfolio pivot because carry crashes (e.g., 2007–2008) are rare in historical data.

### 2.4 Offline RL & World Models (Decision-Time RL)

**What SOTA looks like:**
Your current RL is **online PPO** (`rl_position_sizing.py`) which requires environment interaction. The 2025–2026 frontier uses:
- **Offline RL (CQL, IQL, Decision Transformer):** Learn optimal policies from historical trade journals without online exploration. Critical because exploration in live markets is expensive.
- **Model-Based RL / World Models (DreamerV3-style):** Learn a latent dynamics model of the market, then plan inside the imagination. The world model predicts returns, volatility, and correlations in a compact latent space.
- **RLHF for Risk Preferences:** Align the reward function with human portfolio-manager preferences (e.g., "I hate drawdowns more than I love upside") using preference pairs.

**Your gap:**
You have PPO for position sizing and a `GateThresholdEnv` in `plans/rl_integration_strategy.md` that was never implemented. No offline RL, no world model, no RLHF. Your RL modules are 2022-era.

**Severity:** 🔴 **High** — The factor portfolio rebalancing problem is a sequential decision problem perfectly suited to offline RL.

### 2.5 Causal Machine Learning & Structural Causal Models (SCMs)

**What SOTA looks like:**
Moving from "Granger causality with 30% accuracy" (your `COUNTERFACTUAL_REASONING.md` target) to actual causal inference:
- **Causal discovery algorithms** (PC, GES, NOTEARS) on factor returns to identify true drivers
- **Do-calculus** for counterfactual queries: "Would my carry trade have crashed if the SNB had not removed the peg?"
- **Causal representation learning:** Disentangle latent factors into causal mechanisms (e.g., "inflation shock" vs "growth shock")

**Your gap:**
You have `enable_causal_filtering: True` in config and Granger causality at 30% accuracy. Your `COUNTERFACTUAL_REASONING.md` correctly identifies the path to Tier 6 do-calculus but it is **not implemented.** The causal layer is aspirational.

**Severity:** 🟡 **Medium** — For a factor portfolio, understanding causal drivers (e.g., does USD strength cause EM weakness or correlate with it?) is alpha-generating.

### 2.6 Multi-Modal Models (Beyond Text)

**What SOTA looks like:**
- **Earnings call audio:** Whisper-style models extracting tone, hesitation, and sentiment from executive voices
- **Satellite imagery:** Counting cars in Walmart parking lots, monitoring oil tanker movements
- **Option flow visualization:** CNNs trained on volatility surface images
- **Order book "images":** Treating L2/L3 order book as spatial-temporal images and using Vision Transformers (ViT)

**Your gap:**
You have FinBERT for news sentiment. Nothing for audio, imagery, or order-book visualization. Your order_flow agent uses OANDA position-book ratios (`pb_*` features) — this is 2015-era.

**Severity:** 🟡 **Medium** — Relevant mostly if you expand to equities/commodities. For FX, central bank audio and option flow matter more.

### 2.7 Neural Operators & Continuous-Time Modeling

**What SOTA looks like:**
- **Neural SDEs:** Modeling price as a continuous-time stochastic process with neural drift and diffusion terms. Handles irregular sampling and missing data naturally.
- **Fourier Neural Operators (FNO):** Learn operators that map initial conditions to future paths in function space. Good for multi-horizon forecasting.
- **DeepONet:** Operator learning for parametric PDEs; applied to implied volatility surfaces.

**Your gap:**
Your models operate on discrete bars (H1, D1). No continuous-time representation. For the factor portfolio (weekly rebalance), this is less critical, but for execution timing it matters.

**Severity:** 🟢 **Low-Medium**

### 2.8 Knowledge Graphs + RAG for Macro Reasoning

**What SOTA looks like:**
- **Structured knowledge graphs** of macro relationships: "Fed policy rate → USD strength → EM carry trade unwind → VIX spike"
- **RAG (Retrieval-Augmented Generation):** LLM retrieves relevant macro documents, central bank research, and historical analogies before reasoning
- **Graph neural networks (GNNs):** Propagate shocks through the macro knowledge graph to estimate portfolio contagion

**Your gap:**
Your `news_risk_agent` does keyword scanning (NFP, CPI, FOMC). There is no knowledge graph, no RAG, no structured macro reasoning. The `market_intelligence` module (`fetch_forex_news`) is shallow.

**Severity:** 🔴 **High** — A factor portfolio lives and dies by macro regime shifts. Carry crashes when the macro narrative changes abruptly.

### 2.9 Advanced Execution with Deep RL

**What SOTA looks like:**
- **RL for optimal execution:** SAC or TD3 agents that learn to place limit/market orders, split parent orders across time, and hide in the order book to minimize market impact and adverse selection.
- **Adversarial execution modeling:** Train execution agents against simulated adversarial market makers to learn robust strategies.
- **Learning from fill data:** Use actual historical fills (not just TWAP benchmarks) to train execution policies.

**Your gap:**
You have `execution_strategy: "TWAP"` and `enable_smart_execution: True`. TWAP is 1990s. You have `enable_execution_quality_optimizer` and `enable_execution_routing`, but the underlying algorithms are heuristic, not learned.

**Severity:** 🟡 **Medium** — At retail scale ($1k–$50k), execution alpha is smaller than signal alpha. Scales with AUM.

### 2.10 Federated Learning & Privacy-Preserving Intelligence

**What SOTA looks like:**
- **Federated learning across asset classes:** Train a shared model on FX, futures, and crypto data without centralizing them
- **Differential privacy:** Ensure no single client's data can be extracted from the model
- **Secure multi-party computation (SMPC):** Collaborative alpha discovery between funds without revealing positions

**Your gap:**
Not present. Your models are trained on your own data only.

**Severity:** 🟢 **Low** — Relevant for institutional multi-strategy platforms, not single-operator retail.

### 2.11 Neural Architecture Search (NAS) for Strategy Discovery

**What SOTA looks like:**
- **AutoML for trading:** Automatically discover network architectures (number of layers, attention heads, kernel sizes) optimized for Sharpe ratio instead of accuracy
- **Differentiable NAS (DARTS):** Search over candidate operations in a continuous space
- **Program synthesis:** LLMs generate strategy code, which is backtested and promoted automatically

**Your gap:**
You have `src/strategy_invention/` using genetic algorithms over JSON DSL templates. This is 2010s-era strategy discovery. The networks are hand-designed (your CNN-Transformer has fixed filters, layers, heads). No AutoML or NAS.

**Severity:** 🟡 **Medium** — Your genetic algorithm layer is functional but will be out-evolved by differentiable search.

### 2.12 Quantum-Classical Hybrid Optimization

**What SOTA looks like:**
- **Quantum annealing** for portfolio optimization (D-Wave): Find global minima of complex objective functions with cardinality constraints
- **Quantum-inspired tensor networks** for correlation structure learning
- **Variational quantum circuits** as expressive feature maps

**Your gap:**
Not present. Your portfolio optimization uses classical mean-variance or risk-parity heuristics.

**Severity:** 🟢 **Low** — Still largely experimental in 2026, but JPMorgan and Goldman have internal prototypes.

---

## 3. The Upgrade Path: 5 Phases

Given your current architecture and the factor portfolio pivot, here is the recommended upgrade path. Each phase builds on the previous, is independently shippable, and respects your fail-closed safety posture.

### Phase 0: Strategic Consolidation (Now – 4 weeks)
**Goal:** Ship the factor portfolio as the primary strategy. Freeze intraday R&D into "research mode."

**Deliverables:**
1. **Complete the FX Factor Portfolio PRD** (`tasks/prd-fx-factor-portfolio.md`)
   - US-001 through US-007: daily data layer, carry/trend/value signals, portfolio construction, vol targeting, cost-aware backtest, ship gate
   - Hard gate: net Sharpe ≥ 0.4, positive in ≥6 of 10 years, max DD ≤ 25%
2. **Demote intraday stack to shadow mode**
   - Set `use_sota_inference=False`, `use_neural_agents=False` for production
   - Keep them running in paper/simulation mode collecting outcomes
   - Document the "no edge" diagnosis in `docs/STRATEGIC_PIVOT_2026.md`
3. **Consolidate agent memory**
   - The 15 specialist agents were designed for intraday directional edge. They are the wrong abstraction for a weekly-rebalanced factor portfolio.
   - Preserve the `agent_weights.json` learning framework but re-purpose it for **factor timing** (e.g., "should we overweight carry this week?")

**★ Insight ─────────────────────────────────────**
- The factor portfolio pivot is not a retreat — it is a strategic upgrade to a strategy class with decades of academic and institutional validation.
- The intraday stack is not wasted: the SOTA core, neural agents, and confidence calibration become R&D assets for Phase 3+ when you revisit higher-frequency signals.
- Your existing risk plumbing (drawdown guardians, position timeout, policy engine) transfers directly to the factor portfolio.
`─────────────────────────────────────────────────`

### Phase 1: Intelligence Layer — LLM Agents + Knowledge Graphs (4–10 weeks)
**Goal:** Replace keyword-based news scanning with structured macro reasoning. This is the highest-ROI upgrade because you already have the LLM prompts and the infrastructure.

**Deliverables:**
1. **Runtime LLM Macro Agent**
   - File: `src/scanner/agents/llm_macro_agent.py`
   - Uses the 37 personality prompts from `.claude/agents/` but wires them into the scanner at runtime
   - Tool use: queries FRED API, BIS REER database, central bank calendar
   - Output: structured `AgentVerdict` with regime shift probability (0–1)
   - Caching: responses cached per UTC day to avoid API costs
   - Safety: runs in **shadow mode** for 30 days before getting a vote

2. **Knowledge Graph + RAG Module**
   - File: `src/intelligence/macro_knowledge_graph.py`
   - Static graph: entities (central banks, currencies, commodities, economies) + relations ("hikes rates → appreciates against")
   - Dynamic ingest: parse FOMC statements, ECB press conferences, BoJ minutes into graph updates
   - RAG: retrieve relevant historical analogies ("2024 BoJ hike most similar to 2006 Fed pause")
   - Integration: feeds into LLM Macro Agent context window

3. **Multi-Modal Central Bank Audio Processing**
   - File: `src/intelligence/cb_audio_processor.py`
   - Ingest: Powell / Ueda / Bailey press conference audio
   - Model: Whisper-large-v3 → sentiment + tone + hesitation metrics
   - Output: delta in "dovish/hawkish" score per central bank
   - Trigger: only process when `news_risk_score` detects a scheduled event

**★ Insight ─────────────────────────────────────**
- Your `news_risk_agent` scans for "NFP" and "CPI" keywords. A 2026 SOTA system asks: "Given the FOMC statement, the dot plot, and the last 3 press conferences, what is the probability that the Fed surprises hawkish at the next meeting, and which currency pairs are most exposed?"
- The `.claude/agents/` prompts are a latent asset — they represent 37 reasoning personas. Runtime-wiring them with tool-use is a pure engineering lift, not a research risk.
`─────────────────────────────────────────────────`

### Phase 2: Model Architecture — Time Series Foundation Models (10–18 weeks)
**Goal:** Replace the custom CNN-Transformer with a pre-trained foundation model fine-tuned via LoRA. This addresses the single biggest architectural gap.

**Deliverables:**
1. **Foundation Model Integration**
   - File: `src/sota_core/foundation_model_adapter.py`
   - Backends (pick one, design for swappable):
     - **Primary:** TimesFM (Google) — best zero-shot transfer, Apache 2.0
     - **Alternative:** Chronos (Amazon) — tokenizer-based, easy quantization
     - **Edge:** TinyTimeMixer (IBM) — if you need on-device inference
   - Adapter: LoRA or DoRA fine-tuning on your FX daily returns
   - Input: raw OHLCV (same as current SOTA core)
   - Output: probabilistic forecasts (mean + variance) for 1-week, 2-week, 4-week horizons
   - Contract: drop-in replacement for `SOTAInference.predict()`

2. **Hybrid Architecture (Fallback Safety)**
   - File: `src/sota_core/hybrid_inference.py`
   - Ensemble: foundation model + custom CNN-Transformer + XGBoost
   - Gating: if foundation model is >2σ out-of-distribution (detected via input reconstruction error), fall back to custom model
   - This preserves your fail-closed posture while gaining transfer learning benefits

3. **Diffusion-Based Market Simulation**
   - File: `src/simulation/diffusion_market_model.py`
   - Architecture: latent diffusion model (LDM) trained on daily factor returns
   - Use cases:
     - Stress test the factor portfolio under synthetic "2022-style" vol spikes
     - Generate training data for the foundation model adapter
     - Evaluate tail risk (e.g., "probability of 3-sigma drawdown in next quarter")
   - Validation: Kolmogorov-Smirnov tests against real return distributions; autocorrelation structure matching

**★ Insight ─────────────────────────────────────**
- Your `RawSequenceModel` has ~200K parameters and was trained on ~5,000 candles. TimesFM has ~200M parameters and was trained on 100B time-series tokens. The transfer learning advantage is enormous — especially for the factor portfolio where you have only 2,600 daily observations per pair.
- The diffusion model is not just for pretty charts. It lets you ask: "What is the 1% CVaR of my carry portfolio if a geopolitical shock occurs during a high-vol regime?" — a question impossible to answer with historical backtests alone.
`─────────────────────────────────────────────────`

### Phase 3: Decision & Execution — Offline RL + Advanced Execution (18–30 weeks)
**Goal:** Replace heuristic rebalancing and TWAP execution with learned policies.

**Deliverables:**
1. **Offline RL Portfolio Rebalancer**
   - File: `src/rl/offline_portfolio_rl.py`
   - Algorithm: Conservative Q-Learning (CQL) or Implicit Q-Learning (IQL)
   - State: factor exposures, current weights, recent returns, regime context, macro KG embeddings
   - Action: target weight deltas for each pair (continuous, constrained by gross leverage ≤ 4:1)
   - Reward: risk-adjusted return (Sharpe) minus transaction cost penalty minus drawdown penalty
   - Data source: historical factor portfolio backtest + simulated paths from diffusion model
   - Safety: action clipping + hard leverage guardrail (unlearnable)

2. **RLHF for Risk Preferences**
   - File: `src/rl/risk_preference_alignment.py`
   - Collect preference pairs: "Portfolio A had +8% return with -15% max DD" vs "Portfolio B had +12% return with -25% max DD"
   - Train a reward model that reflects operator risk aversion
   - Align the offline RL policy via PPO or DPO (Direct Preference Optimization)
   - Output: personalized risk-adjusted reward function

3. **Deep RL Execution Agent**
   - File: `src/execution/rl_execution_agent.py`
   - Algorithm: Soft Actor-Critic (SAC) or Twin Delayed Deep Deterministic Policy Gradient (TD3)
   - State: order book depth, spread, recent volume, time-of-day, volatility
   - Action: limit order price (relative to mid), order size fraction, cancellation timing
   - Reward: implementation shortfall vs. arrival price
   - Training: use historical OANDA fill data (you already have `execution_quality_tracking`)

**★ Insight ─────────────────────────────────────**
- Your current `GateThresholdEnv` from `plans/rl_integration_strategy.md` was designed for intraday thresholds. The factor portfolio needs a **continuous control** problem (weight deltas), not a discrete threshold problem.
- Offline RL is crucial because you cannot "explore" in live markets. CQL learns conservatively from historical data, avoiding overestimation of out-of-distribution actions.
`─────────────────────────────────────────────────`

### Phase 4: Causal & Structural Upgrades (30–42 weeks)
**Goal:** Move from correlation-based regime detection to causal understanding of factor drivers.

**Deliverables:**
1. **Causal Discovery on Factor Returns**
   - File: `src/causal/factor_causal_discovery.py`
   - Algorithms: NOTEARS (continuous optimization), PC (constraint-based), GES (score-based)
   - Input: daily factor returns (carry, trend, value) + macro proxies (rates, inflation, growth, liquidity)
   - Output: directed acyclic graph (DAG) of causal relationships
   - Update frequency: monthly (causal structure is slower than prices)

2. **Structural Causal Model (SCM) for Regime Identification**
   - File: `src/causal/scm_regime_model.py`
   - Model: parametric SCM where regime is a latent confounder
   - Use do-calculus to answer: "What is the expected return of carry if the Fed hikes by 25bps?"
   - Integration: feeds into LLM Macro Agent reasoning and offline RL state space

3. **Counterfactual Portfolio Simulator**
   - File: `src/causal/counterfactual_simulator.py`
   - Input: SCM + current portfolio weights + hypothetical intervention
   - Output: estimated P&L distribution under the intervention
   - Use case: pre-trade "what-if" analysis before rebalancing

**★ Insight ─────────────────────────────────────**
- Your `COUNTERFACTUAL_REASONING.md` correctly diagnoses the path from Granger causality (30%) to do-calculus (60%+). The missing piece is **not more data** — it is a causal model of the macro economy that disentangles correlation from mechanism.
- For a carry trade, correlation tells you that high-yield currencies appreciate in low-vol regimes. Causality tells you *why* (risk appetite → vol compression → carry returns), which lets you predict when the mechanism breaks.
`─────────────────────────────────────────────────`

### Phase 5: Meta-Cognitive Autonomy (42+ weeks)
**Goal:** The system discovers, implements, tests, and promotes its own upgrades.

**Deliverables:**
1. **AutoML for Strategy Discovery**
   - File: `src/meta_automl/strategy_nas.py`
   - Search space: neural architectures, factor combinations, rebalancing rules
   - Use DARTS or ENAS to search for architectures that maximize out-of-sample Sharpe
   - Constraints: max latency, max parameter count, interpretability score

2. **Self-Improving Agent via Code Generation**
   - File: `src/meta_automl/code_generating_agent.py`
   - LLM agent reads journal outcomes, identifies underperforming factors, and generates new factor candidates
   - Automated backtest on holdout data
   - Promotion gate: same as your existing ship gate (Sharpe ≥ 0.4, etc.)
   - Integration with Ralph: the generated code becomes a Ralph PRD story

3. **Federated Learning Across Asset Classes**
   - File: `src/meta_automl/federated_factor_learning.py`
   - Train factor models on FX + futures + crypto data without centralizing
   - Differential privacy guarantees
   - Use case: learn universal "trend" and "carry" representations that transfer across markets

**★ Insight ─────────────────────────────────────**
- Your `scripts/ralph.sh` spawns AI instances to implement PRD stories. Phase 5 closes the loop: the AI not only implements stories but **writes them** based on journal analytics. Ralph graduates from executor to inventor.
- The 37 `.claude/agents/` prompts become the seed population for an evolutionary system that breeds new reasoning strategies.
`─────────────────────────────────────────────────`

---

## 4. Dependency Graph & Risk Mitigation

```
Phase 0 (Factor Portfolio)
    │
    ├── MUST complete before Phase 1: It is the revenue-bearing strategy
    │
Phase 1 (LLM + KG)
    │
    ├── Needs: daily data layer (US-001)
    ├── Risk: LLM API costs, latency, hallucination
    ├── Mitigation: shadow mode 30 days, structured output schemas (Instructor/Pydantic), caching
    │
Phase 2 (Foundation Models)
    │
    ├── Needs: Phase 1 macro context (regime labels from KG)
    ├── Risk: foundation model weights are large (GBs), inference latency
    ├── Mitigation: quantized models (INT8), ONNX Runtime, edge deployment for inference
    │
Phase 3 (Offline RL)
    │
    ├── Needs: Phase 2 probabilistic forecasts (mean + variance)
    ├── Risk: policy overfitting to historical data
    ├── Mitigation: diffusion model augmentation, conservative CQL regularization, paper trading
    │
Phase 4 (Causal)
    │
    ├── Needs: Phase 0 factor returns as time series
    ├── Risk: causal discovery is computationally expensive and sensitive to assumptions
    ├── Mitigation: ensemble of algorithms (NOTEARS + PC + GES), monthly updates only
    │
Phase 5 (Autonomy)
    │
    ├── Needs: all prior phases stable
    ├── Risk: unbounded code generation, prompt injection, runaway compute
    ├── Mitigation: sandboxed execution, human-in-the-loop promotion gates, cost budgets
```

---

## 5. Recommended Immediate Next Steps (This Week)

1. **Operator decision required:** Approve the Phase 0 factor portfolio as the primary trading strategy. The intraday stack has been correctly diagnosed as edgeless.
2. **File `tasks/prd-fx-factor-portfolio.md` US-001:** Implement the daily data layer. This is a pure engineering task with no model risk.
3. **Shadow-deploy Phase 1 LLM Macro Agent:** Pick 3 of the 37 `.claude/agents/` prompts, wire them with tool-use (FRED API), and run them in shadow mode against historical events (e.g., "What would the agent have said before the July 2024 BoJ hike?").
4. **Foundation model evaluation:** Download TimesFM-base (open weights) and run zero-shot forecasting on your daily EUR_USD data. Measure MAE vs. your current `RawSequenceModel`. This is a 2-hour experiment that validates Phase 2 feasibility.

---

## 6. Appendix: Capability Matrix

| Capability | Current Tier 7 | SOTA 2025–2026 | Gap Size | Phase |
|---|---|---|---|---|
| Signal generation | Custom CNN-Transformer | TimesFM / MOIRAI foundation models | 🔴 High | 2 |
| Agent reasoning | Rule-based + small MLPs | LLM agents with tool-use + RAG | 🔴 High | 1 |
| News / macro | Keyword scanning | KG + multi-modal audio + RAG | 🔴 High | 1 |
| Market simulation | Historical backtest only | Diffusion models for scenario gen | 🟡 Med-High | 2 |
| RL decision-making | Online PPO (position sizing) | Offline CQL/IQL + world models | 🔴 High | 3 |
| Execution | TWAP + heuristics | SAC/TD3 optimal execution | 🟡 Medium | 3 |
| Causal inference | Granger causality (30%) | SCM + do-calculus | 🟡 Medium | 4 |
| Meta-learning | MAML Ridge (shadow) | Full AutoML + NAS | 🟡 Medium | 5 |
| Risk alignment | Fixed drawdown guardrails | RLHF preference alignment | 🟡 Medium | 3 |
| Autonomy | Ralph executes PRDs | AI writes PRDs + backtests + promotes | 🟢 Low | 5 |
| Portfolio construction | Heuristic ensemble | Differentiable optimization | 🟡 Medium | 3 |

---

*End of document.*
