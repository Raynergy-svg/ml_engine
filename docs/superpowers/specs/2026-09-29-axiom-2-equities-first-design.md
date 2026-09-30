# Axiom 2.0 — Equities-First Architecture

**Status:** Revised after Codex-vs-Axiom three-pass adversarial review; proposed for user review  
**Date:** 2026-09-29

## 1. Objective

Axiom 2.0 changes the system from an FX-first directional classifier into a lean equities-first evidence, promotion, portfolio-risk, and execution kernel. General research reasoning and software engineering are delegated to a replaceable research agent (Codex initially); Axiom remains the deterministic authority that decides what evidence is valid, what may be promoted, what portfolio risk is permitted, and what orders may reach the broker.

The first objective is not maximum feature count or maximum autonomy. It is to establish a reproducible, economically meaningful equity edge that survives out-of-sample validation, transaction costs, market-regime changes, and shadow execution.

The existing FX/OANDA system is retained as reproducible legacy research. It is not the primary Axiom 2.0 runtime.

## 2. Design principles

1. **Edge before complexity.** New model families, options, crypto, and advanced autonomy are excluded until the equity baseline proves value.
2. **Axiom is a kernel, not another general agent platform.** Axiom owns four authorities only: evidence integrity, promotion, deterministic portfolio/risk, and execution/reconciliation. Research orchestration, hypothesis generation, code authoring, diagnostics, and narrative analysis belong to a replaceable research agent.
3. **Broker interface is not strategy logic.** Robinhood supplies supported live market/broker capabilities and broker truth. Axiom owns the evidence vault, promoted artifacts, portfolio decisions, risk, and execution authorization.
4. **Immutable research inputs.** Training and validation use timestamped snapshots rather than mutable live MCP responses.
5. **No LLM in the trading hot path.** LLMs may research, diagnose, propose experiments, and review evidence. Deterministic code controls inference, risk, and execution.
6. **Fail closed.** Missing data, stale quotes, unresolved order state, evidence failure, or risk-state uncertainty blocks new exposure.
7. **No weakened promotion gates.** Changing markets is not evidence of edge.
8. **Every live decision is reproducible.** Dataset version, feature schema, model artifact, calibration, decision, risk state, order review, broker response, and outcome are linked.
9. **Research OOS is a consumable resource.** Repeated exposure to the same holdout converts it into training information. Final promotion therefore requires a sealed holdout that ordinary experimentation cannot inspect.
10. **Temporal truth over convenient data.** A value is trainable only when Axiom can prove when it became knowable. Current broker fields are never assumed to be historically point-in-time.
11. **Ablation before accumulation.** New feature families and model complexity must demonstrate incremental OOS economic value over the simpler promoted stack.

## 3. Initial scope

### Included
- US listed liquid equities
- Highly liquid broad/sector ETFs
- Long-only initial execution
- Regular market hours as the initial execution session
- Cross-sectional ranking
- Meta-label trade filtering
- Portfolio construction and exposure controls
- Robinhood adapter for supported market/broker operations
- Shadow portfolio and broker reconciliation

### Excluded from Phase 1
- FX as a primary strategy
- Options
- Crypto
- penny stocks / illiquid microcaps
- short selling
- premarket/after-hours execution
- high-frequency or latency-arbitrage strategies
- LLM-directed order placement
- automatic model promotion directly to live capital

## 4. System boundary and authority model

```text
                 REPLACEABLE RESEARCH AGENT
                      (Codex initially)
                              |
                  structured research proposal
                              v
                    +------------------+
                    |    AXIOM 2.0     |
                    |------------------|
historical sources -> Evidence Authority|
                    |        v         |
                    | Promotion        |
                    | Authority        |
                    |        v         |
live market data -->| Promoted Model   |
                    |        v         |
                    | Portfolio/Risk   |
                    | Authority        |
                    |        v         |
                    | Execution +      |
                    | Reconciliation   |
                    +--------+---------+
                             |
                      approved order only
                             v
                      Robinhood MCP
                             |
                             v
                       Agentic Account
```

There is exactly one production order path: Axiom execution authority -> Robinhood. The research agent has no alternative production order route. A direct research-agent-to-Robinhood connection, if used during isolated investigation, must be read-only/minimum-data and must not possess production order tools.

Codex is not a permanent architectural dependency. Research agents communicate through a versioned Research Proposal Contract. Axiom must be able to accept the same contract from Codex, another approved research agent, deterministic research code, or a human.

## 5. Research Proposal Contract

A research agent may propose experiments but cannot self-promote them. Every proposal must include at minimum:
- experiment ID
- hypothesis
- universe definition
- feature families
- label and holding horizon
- benchmark
- transaction-cost assumptions
- model/baseline definition
- primary metrics and promotion rule
- code commit/artifact identity
- expected data dependencies

Axiom validates the proposal against the experiment registry, data-provenance rules, and holdout policy before any result can enter promotion evidence. Free-form agent conclusions are never promotion evidence by themselves.

## 6. Canonical domain model

Axiom must stop leaking FX concepts into generic interfaces.

### Instrument
The existing EQUITY asset class is retained and expanded with point-in-time metadata required by the research and execution layers. Broker-specific identifiers remain adapter concerns.

### Market data
Canonical records distinguish:
- bars
- quotes
- market session state
- corporate actions
- benchmark/index observations
- fundamentals/financial statements
- earnings/events
- news/filing observations
- order-book observations when available

Every research record carries a temporal-availability contract:
- event_time: when the underlying event occurred
- published_time: when the information was publicly released, when applicable
- available_to_axiom_time: earliest proven time the strategy could have consumed it
- ingested_time: when Axiom stored it
- revision/version: source revision identity when available
- source and schema version

If `available_to_axiom_time` cannot be established conservatively, the field is excluded from historical training. Ingestion time is not a substitute for historical availability.

### Orders
Replace the FX-shaped order API requiring SL/TP on every placement with a generic order intent:
- instrument
- side
- quantity/notional
- order type
- limit/stop fields when relevant
- time in force
- strategy/decision ID
- idempotency key

Risk exits are policy decisions, not mandatory broker-interface parameters.

## 7. Universe

Phase 1 uses a controlled liquid universe: S&P 500 constituents plus a curated set of highly liquid broad-market and sector ETFs, subject to point-in-time membership and liquidity eligibility.

Eligibility filters include:
- sufficient price/history coverage
- minimum liquidity
- no unresolved corporate-action state
- valid quote/session state
- no stale or missing critical features

The research pipeline must avoid survivorship bias by storing point-in-time universe membership rather than training against today's membership retroactively. Robinhood availability does not prove historical point-in-time membership or delisted-security coverage. Phase D must either prove those properties from a source or exclude unsupported periods/securities. Current S&P 500 membership must never be projected backward.

## 8. Feature architecture

Features are grouped into independently versioned families.

### Price/return
- lagged returns over multiple horizons
- momentum/reversal
- realized volatility
- range/gap behavior
- drawdown and trend state

### Cross-sectional
- return rank
- volatility-adjusted rank
- relative strength versus market, sector, and peers
- residual return after benchmark/sector effects

### Liquidity/volume
- dollar volume
- relative volume
- spread/quote quality where available
- turnover/liquidity regime
- abnormal volume

### Market/sector regime
- benchmark trend/volatility
- breadth
- sector-relative state
- correlation/concentration regime

### Phase-1 admissible feature families
The first research campaign is deliberately limited to price/return, volume/liquidity, cross-sectional relative strength, and market/sector regime features whose point-in-time semantics can be proven. Existing technical-indicator code is not automatically inherited; each feature/family must survive ablation.

### Quarantined fundamental/event
- point-in-time fundamental ratios
- financial-statement changes
- earnings proximity/surprise-derived fields where valid
- filing/event state

### Quarantined microstructure
L2/order-book features are excluded from the first research campaign. Broker-provided L2 is treated as a venue/source-specific observation, not a complete market order book. It can enter later only after historical availability, timestamp semantics, coverage limitations, and incremental OOS value are demonstrated.

No feature is admitted without a provenance definition and leakage test.

## 9. Labels and learning objective

The primary model is no longer next-candle binary direction.

### Primary target
Cross-sectional forward excess-return ranking over a defined holding horizon, benchmark/sector adjusted as appropriate.

### Secondary targets
- triple-barrier outcome
- realized volatility/regime
- meta-label: whether a ranked candidate should actually receive capital

Labels are computed strictly after the feature cutoff and incorporate realistic trading calendars. Corporate actions and delistings must not silently disappear.

## 10. Model stack

Start simple and force complexity to earn promotion.

1. **Naive baselines:** market/sector and simple momentum/factor rules.
2. **Primary ML baseline:** LightGBM cross-sectional ranker.
3. **Trade filter:** XGBoost/LightGBM meta-labeler.
4. **Volatility/regime model:** only if it improves portfolio outcomes beyond deterministic regime features.
5. **Temporal neural model:** quarantined research candidate until it beats the simpler stack under identical evidence.

Calibration applies to probability-producing components. Ranking quality is evaluated with ranking and portfolio metrics, not classification accuracy alone.

## 11. Validation and experiment-selection control

A model is not promoted because validation accuracy is higher than the old FX model.

Before a research campaign begins, Axiom preregisters the primary hypothesis, universe, label horizon, benchmark, cost assumptions, primary metrics, and promotion rule. Every attempted experiment receives an immutable experiment ID and remains in the registry, including failures.

Validation has three distinct layers:
1. development train/validation used for fitting and ordinary tuning;
2. rolling purged walk-forward OOS used for research comparison;
3. a sealed final holdout that normal experiment code and researchers do not inspect until a promotion candidate is frozen.

Once a sealed holdout has been opened for a candidate family, it is recorded as consumed. Subsequent tuning cannot continue to call that period untouched OOS. A new promotion campaign requires a genuinely untouched future period or other preregistered evidence.

Where many hypotheses are tested, evidence must report experiment count and apply an explicit multiple-testing/selection-bias control appropriate to the metric. Axiom never reports the best run without the attempted-search context.

Required evidence includes:
- time-ordered walk-forward validation
- purging/embargo where labels overlap
- point-in-time universe membership
- transaction-cost/slippage assumptions
- comparison against naive and simple-factor baselines
- comparison against the current promoted ML baseline
- regime slices
- turnover and capacity diagnostics
- drawdown and concentration diagnostics
- calibration for probability outputs
- shadow portfolio results before capital eligibility
- sealed-holdout result for a frozen candidate
- experiment-count / selection-bias accounting
- feature-family and model-family ablation evidence

Primary success metrics center on economically relevant OOS performance: benchmark-relative return, risk-adjusted return, drawdown, turnover/cost sensitivity, ranking quality, and stability. Accuracy may be reported where meaningful but is not the promotion objective.

## 12. Portfolio construction

The ranker produces opportunities, not orders.

Portfolio construction converts eligible candidates into target weights subject to:
- maximum single-name exposure
- sector concentration
- portfolio gross exposure
- correlation/concentration controls
- liquidity constraints
- turnover budget
- minimum expected edge after costs
- cash reserve
- risk-off/halt state

Phase 1 remains long-only.

## 13. Risk and execution

A deterministic risk governor is authoritative. The research agent cannot call broker order-placement tools in production, cannot override risk decisions, and cannot mutate the promoted model/artifact pointer during a trading session.

Order flow:
1. model decision
2. eligibility/meta-label gate
3. portfolio target
4. risk validation
5. broker order review
6. shadow or approved execution
7. placement
8. reconciliation
9. outcome journal

An ambiguous placement response never triggers an immediate duplicate order. Reconciliation checks broker truth using the decision/idempotency identity before retrying.

External-agent trade approvals should remain enabled during initial integration testing, and Axiom's adapter should additionally disable real submission until the shadow gates pass.

## 14. Robinhood adapter and security boundary

Add a Robinhood implementation behind Axiom's broker/data interfaces rather than allowing MCP-specific structures to propagate through the codebase.

The adapter owns:
- authentication/connection state
- symbol translation
- market-data translation
- quote freshness
- account/position translation
- order preview/review translation
- order placement/cancel translation
- error normalization
- reconciliation

Capabilities are explicitly advertised and version/capability checked at startup. Unsupported or changed operations fail closed rather than being emulated silently.

Robinhood is the primary live market/execution interface, not the authority for Axiom's historical evidence vault. The adapter must minimize retrieved data and prevent unnecessary account information from entering model inputs, logs, prompts, or evidence artifacts.

Security requirements:
- explicit allowlist for the dedicated Agentic trading account;
- redact account numbers and other unnecessary account identifiers from logs/evidence;
- minimum-data retrieval by default;
- separate research and execution sessions/credentials where the provider permits it;
- development, healing, research, and LLM agents receive no production order-placement authority;
- the production MCP/tool policy exposes order side effects only to Axiom's execution gateway; where tool-level allowlists/approval controls are available they are mandatory;
- no second Codex-to-Robinhood production connection may bypass Axiom;
- local deterministic kill state blocks all new orders independently of model state;
- broker disconnect/revocation is documented as the ultimate external kill mechanism;
- startup refuses execution if expected tool capabilities or semantics have changed.

## 15. Legacy FX isolation

OANDA/FX code and artifacts remain reproducible but are removed from default Axiom 2.0 execution and training paths.

Migration should prefer deprecation and isolation over destructive deletion. Existing imports remain compatible until their consumers are migrated and tests prove removal is safe.

The end state separates:
- `axiom2/` or equivalent primary equity-domain modules
- broker-neutral shared governance/evidence infrastructure
- `legacy/fx_oanda/` research/runtime compatibility lane

Exact physical moves happen only when import-safe; logical isolation comes first.

## 16. Evidence spine

Preserve the strongest part of the current system.

Every promoted claim links:
- immutable dataset manifest
- point-in-time universe manifest
- feature schema/version
- label definition/version
- training job
- code commit
- model artifact hash
- validation results
- cost assumptions
- promotion decision
- shadow/live outcomes

Research failures remain evidence. They are not overwritten by subsequent experiments.

## 17. Migration sequence

### Phase A — Freeze and baseline
Freeze the FX promoted state, inventory dependencies, record current evidence, and ensure no migration rewrites historical artifacts.

### Phase B — Lean-kernel and research-agent contracts
Define the four Axiom authorities and the versioned Research Proposal Contract first. Explicitly identify existing autonomous/agent components that become delegated, deprecated, or non-authoritative. Do not rebuild capabilities Codex already supplies unless an ablation or operational requirement proves Axiom must own them.

### Phase C — Domain contracts
Generalize instrument, market-data, order, session, portfolio, and broker capability contracts. Add equity-native tests before changing runtime defaults.

### Phase D — Robinhood read-only adapter
Connect minimum required data/account capabilities only. Prove timestamps, history, quote freshness, capability/version behavior, account allowlisting/redaction, universe mapping, corporate-action semantics where available, and reconciliation semantics without order submission. Do not infer historical research suitability from successful live retrieval.

### Phase E — Equity research dataset and temporal audit
Build the immutable point-in-time data vault and leakage tests for the controlled universe. Every candidate source/field must pass the temporal-availability contract. Prove or source point-in-time universe membership, delisted-security treatment, split/dividend handling, and adjusted-price semantics. Unsupported historical fields are excluded rather than approximated.

### Phase F — Minimal preregistered research campaign
Test one primary question: can point-in-time price, volume, market/sector-relative, and regime features rank liquid US equities such that a top-ranked long-only portfolio produces persistent positive excess return after conservative costs across walk-forward periods? Register naive factor baselines and the LightGBM ranker before opening the sealed holdout. No news, fundamentals, L2, LLM sentiment, transformer, or multi-agent consensus enters this campaign.

### Phase G — Sealed-holdout decision
Freeze the candidate and code commit before evaluating the sealed holdout. Record the result and mark the holdout consumed. If the edge fails, return to research without weakening the gate.

### Phase H — Meta-label + portfolio
Add candidate filtering, cost-aware portfolio construction, and deterministic equity risk controls.

### Phase I — Shadow execution
Run live-market decisions with zero real order submission. Reconcile hypothetical fills/cost assumptions against available broker observations.

### Phase J — Broker execution proof
Enable order review and tightly controlled test execution only after the shadow/evidence gates pass and with human approval.

### Phase K — Controlled feature expansion
Only after demonstrated edge consider broader universe or additional feature families. Fundamentals, earnings, microstructure/L2, news, alternative data, temporal neural models, crypto, and options enter one family at a time and must demonstrate incremental OOS value through preregistered ablation before promotion.

## 18. Failure handling

Axiom blocks new exposure when:
- source data is stale/incomplete
- point-in-time membership is unknown
- feature schema mismatches model schema
- model/evidence signature fails
- quote/session state is ambiguous
- broker connection/auth is unhealthy
- account state cannot be reconciled
- order state is ambiguous
- portfolio/risk state is inconsistent
- promotion evidence is missing

Existing positions enter deterministic reconciliation/risk-management behavior rather than being abandoned.

## 19. Testing strategy

Required test layers:
- Research Proposal Contract validation and agent-substitution tests
- negative tests proving research/development agents cannot invoke production order side effects
- contract tests for generic broker/data interfaces
- Robinhood adapter translation tests
- timestamp/timezone/session tests
- corporate-action tests
- point-in-time universe/survivorship tests
- temporal-availability/provenance tests
- feature leakage tests
- sealed-holdout access-control/consumption tests
- experiment-registry completeness tests
- survivorship/delisting/universe-membership tests
- label cutoff tests
- walk-forward/purge tests
- transaction-cost sensitivity tests
- portfolio constraint property tests
- idempotent order/reconciliation tests
- stale/missing-data fail-closed tests
- shadow end-to-end test from ingestion through journal

No live-capital test is used to discover basic integration bugs.

## 20. Definition of Axiom 2.0 success

Axiom 2.0 is not successful when it merely connects to Robinhood, reports a higher validation score, or recreates a large autonomous-agent framework. The kernel is successful only if its four authorities are independently testable and the research-agent layer can be replaced without changing evidence, promotion, risk, or execution semantics.

The first release milestone is reached when:
1. the equity dataset is reproducible and leakage-tested;
2. the simple baselines are reproducible;
3. a frozen minimal ranker beats preregistered baselines OOS after realistic costs across required walk-forward/regime slices and then survives a sealed final holdout;
4. experiment-selection and multiple-testing evidence is attached, and the sealed holdout is marked consumed;
5. portfolio risk constraints pass;
6. shadow decisions and broker-state reconciliation operate reliably;
7. the full evidence package can reproduce why each candidate was or was not eligible.

Only then does Axiom 2.0 become eligible for controlled real-capital execution.
