# Axiom 2.0 — Equities-First Architecture

**Status:** Proposed architecture for user review  
**Date:** 2026-09-29

## 1. Objective

Axiom 2.0 changes the system from an FX-first directional classifier into an equities-first autonomous portfolio intelligence system.

The first objective is not maximum feature count or maximum autonomy. It is to establish a reproducible, economically meaningful equity edge that survives out-of-sample validation, transaction costs, market-regime changes, and shadow execution.

The existing FX/OANDA system is retained as reproducible legacy research. It is not the primary Axiom 2.0 runtime.

## 2. Design principles

1. **Edge before complexity.** New model families, options, crypto, and advanced autonomy are excluded until the equity baseline proves value.
2. **Broker interface is not strategy logic.** Robinhood supplies supported market/broker capabilities. Axiom owns datasets, features, labels, models, portfolio decisions, risk, evidence, and governance.
3. **Immutable research inputs.** Training and validation use timestamped snapshots rather than mutable live MCP responses.
4. **No LLM in the trading hot path.** LLMs may research, diagnose, propose experiments, and review evidence. Deterministic code controls inference, risk, and execution.
5. **Fail closed.** Missing data, stale quotes, unresolved order state, evidence failure, or risk-state uncertainty blocks new exposure.
6. **No weakened promotion gates.** Changing markets is not evidence of edge.
7. **Every live decision is reproducible.** Dataset version, feature schema, model artifact, calibration, decision, risk state, order review, broker response, and outcome are linked.

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

## 4. System boundary

```text
Robinhood market/broker interface
        |
        v
Canonical ingestion + immutable snapshots
        |
        v
Point-in-time equity feature store
        |
        +--> labels / benchmark-relative outcomes
        |
        v
Research + model training
        |
        v
Evidence / validation / calibration
        |
        v
Cross-sectional opportunity ranking
        |
        v
Meta-label + eligibility gates
        |
        v
Portfolio construction
        |
        v
Deterministic risk governor
        |
        v
Shadow / approval / execution boundary
        |
        v
Robinhood order review + placement
        |
        v
Broker reconciliation + journal
        |
        v
Outcome evidence / learning
```

## 5. Canonical domain model

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

Every record carries source, source timestamp, observation timestamp, ingestion timestamp, and schema version where applicable.

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

## 6. Universe

Phase 1 uses a controlled liquid universe: S&P 500 constituents plus a curated set of highly liquid broad-market and sector ETFs, subject to point-in-time membership and liquidity eligibility.

Eligibility filters include:
- sufficient price/history coverage
- minimum liquidity
- no unresolved corporate-action state
- valid quote/session state
- no stale or missing critical features

The research pipeline must avoid survivorship bias by storing point-in-time universe membership rather than training against today's membership retroactively.

## 7. Feature architecture

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

### Fundamental/event
- point-in-time fundamental ratios
- financial-statement changes
- earnings proximity/surprise-derived fields where valid
- filing/event state

### Microstructure
L2/order-book features are optional feature families. The core model must remain operable without them until their historical availability and timestamp semantics are proven.

No feature is admitted without a provenance definition and leakage test.

## 8. Labels and learning objective

The primary model is no longer next-candle binary direction.

### Primary target
Cross-sectional forward excess-return ranking over a defined holding horizon, benchmark/sector adjusted as appropriate.

### Secondary targets
- triple-barrier outcome
- realized volatility/regime
- meta-label: whether a ranked candidate should actually receive capital

Labels are computed strictly after the feature cutoff and incorporate realistic trading calendars. Corporate actions and delistings must not silently disappear.

## 9. Model stack

Start simple and force complexity to earn promotion.

1. **Naive baselines:** market/sector and simple momentum/factor rules.
2. **Primary ML baseline:** LightGBM cross-sectional ranker.
3. **Trade filter:** XGBoost/LightGBM meta-labeler.
4. **Volatility/regime model:** only if it improves portfolio outcomes beyond deterministic regime features.
5. **Temporal neural model:** quarantined research candidate until it beats the simpler stack under identical evidence.

Calibration applies to probability-producing components. Ranking quality is evaluated with ranking and portfolio metrics, not classification accuracy alone.

## 10. Validation

A model is not promoted because validation accuracy is higher than the old FX model.

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

Primary success metrics center on economically relevant OOS performance: benchmark-relative return, risk-adjusted return, drawdown, turnover/cost sensitivity, ranking quality, and stability. Accuracy may be reported where meaningful but is not the promotion objective.

## 11. Portfolio construction

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

## 12. Risk and execution

A deterministic risk governor is authoritative.

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

## 13. Robinhood adapter

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

Capabilities are explicitly advertised. Unsupported operations fail closed rather than being emulated silently.

## 14. Legacy FX isolation

OANDA/FX code and artifacts remain reproducible but are removed from default Axiom 2.0 execution and training paths.

Migration should prefer deprecation and isolation over destructive deletion. Existing imports remain compatible until their consumers are migrated and tests prove removal is safe.

The end state separates:
- `axiom2/` or equivalent primary equity-domain modules
- broker-neutral shared governance/evidence infrastructure
- `legacy/fx_oanda/` research/runtime compatibility lane

Exact physical moves happen only when import-safe; logical isolation comes first.

## 15. Evidence spine

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

## 16. Migration sequence

### Phase A — Freeze and baseline
Freeze the FX promoted state, inventory dependencies, record current evidence, and ensure no migration rewrites historical artifacts.

### Phase B — Domain contracts
Generalize instrument, market-data, order, session, portfolio, and broker capability contracts. Add equity-native tests before changing runtime defaults.

### Phase C — Robinhood read-only adapter
Connect data/account capabilities only. Prove timestamps, history, quote freshness, universe mapping, corporate-action handling, and reconciliation semantics without order submission.

### Phase D — Equity research dataset
Build immutable point-in-time datasets and leakage tests for the controlled universe.

### Phase E — Baselines
Run naive factors and LightGBM ranking baseline. Establish the economic benchmark Axiom 2.0 must beat.

### Phase F — Meta-label + portfolio
Add candidate filtering, cost-aware portfolio construction, and deterministic equity risk controls.

### Phase G — Shadow execution
Run live-market decisions with zero real order submission. Reconcile hypothetical fills/cost assumptions against available broker observations.

### Phase H — Broker execution proof
Enable order review and tightly controlled test execution only after the shadow/evidence gates pass and with human approval.

### Phase I — Expansion
Only after demonstrated edge consider broader universe, additional feature families, crypto, options, or more sophisticated models.

## 17. Failure handling

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

## 18. Testing strategy

Required test layers:
- contract tests for generic broker/data interfaces
- Robinhood adapter translation tests
- timestamp/timezone/session tests
- corporate-action tests
- point-in-time universe/survivorship tests
- feature leakage tests
- label cutoff tests
- walk-forward/purge tests
- transaction-cost sensitivity tests
- portfolio constraint property tests
- idempotent order/reconciliation tests
- stale/missing-data fail-closed tests
- shadow end-to-end test from ingestion through journal

No live-capital test is used to discover basic integration bugs.

## 19. Definition of Axiom 2.0 success

Axiom 2.0 is not successful when it merely connects to Robinhood or reports a higher validation score.

The first release milestone is reached when:
1. the equity dataset is reproducible and leakage-tested;
2. the simple baselines are reproducible;
3. the promoted ranker/meta-label stack beats those baselines OOS after realistic costs across required walk-forward/regime slices;
4. portfolio risk constraints pass;
5. shadow decisions and broker-state reconciliation operate reliably;
6. the full evidence package can reproduce why each candidate was or was not eligible.

Only then does Axiom 2.0 become eligible for controlled real-capital execution.
