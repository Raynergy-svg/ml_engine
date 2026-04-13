# ML Engine

Buddy is the trading runtime inside ML Engine: a multi-pair scanner, execution engine, monitoring surface, and self-improving learning loop for FX, with newer multi-broker support for futures workflows.

## Disclaimer

This repository contains experimental trading software.

- It is not financial advice.
- Live trading can lose real money.
- Use practice/demo accounts first.
- Review every config and execution path before enabling live orders.

## What Buddy Is Today

The old README described a smaller gated ensemble. The current project is broader:

- A scanner that evaluates many instruments in one pass.
- A weighted agent-consensus layer that can block or promote trades.
- An execution pipeline with ATR-based exits, spread checks, freshness checks, and broker routing.
- A status and journal surface for runtime visibility.
- A learning loop that extracts patterns from outcomes, promotes recurring lessons into rules, and updates persisted agent weights.

At a high level:

```text
Scan market data
  -> run model + feature analysis
  -> collect agent verdicts
  -> apply gates and policy checks
  -> execute through broker if allowed
  -> journal outcomes
  -> learn from results
  -> tune weights, rules, and operating thresholds
```

## Core Architecture

```text
main.py / bin/Buddy
  -> cli/
  -> src/scanner/engine.py
      -> model inference
      -> agent team voting
      -> gate evaluation
      -> execution manager
      -> automation / watch mode / supervision
      -> learning + diagnostics + state persistence
```

Primary runtime areas:

- `main.py`: thin CLI entrypoint and command dispatch.
- `bin/Buddy`: convenience launcher that picks config and Python environment by machine architecture.
- `src/scanner/`: scanner, agents, execution, automation, filters, diagnostics, confidence pipeline, and adaptive logic.
- `src/brokers/`: broker abstraction plus OANDA and IBKR clients.
- `cli/`: user-facing command handlers for scan, train, journal, learn, analyze, status, and monitoring flows.
- `trained_data/`: models, journals, adaptive state, diagnostics, and scanner outputs.
- `.claude/`: Buddy's rules, learnings, Ralph PRD loop state, and operator memory files used by the repo's autonomous workflows.

## Agent Consensus Model

Buddy's scanner still centers on the specialist-agent voting system described in `AGENTS.md`.

### Core scanner agents

| Agent | Base Weight | Role |
| --- | ---: | --- |
| `trend` | 1.15 | Trend direction and strength |
| `mean_reversion` | 0.90 | Pullback / stretch detection |
| `volatility` | 1.00 | ATR and regime context |
| `risk_sentinel` | 1.25 | Portfolio and drawdown protection |
| `uncertainty` | 1.10 | Model disagreement / variance control |
| `execution_quality` | 1.05 | Spread, slippage, liquidity |
| `momentum` | 1.05 | MACD / ROC-style follow-through |
| `news_risk` | 0.95 | Event and headline risk |
| `multi_timeframe` | 1.10 | H1/H4/D1 alignment |
| `pair_performance` | 0.85 | Pair-specific historical behavior |
| `session_timing` | 0.80 | Session quality / overlap timing |
| `support_resistance` | 1.00 | Pivot and nearby structure |

The current codebase also includes newer adjunct evaluators:

- `trader_readiness`: optional human-side readiness signal bridged from Aura.
- `devil_advocate`: adversarial final bear-case evaluator that can warn or veto late in the pipeline.

### Weight learning

Agent weights persist in `trained_data/models/agent_weights.json` and adapt from outcomes. The runtime now supports:

- bounded weight learning
- time-based decay back toward baseline
- regime-aware weight storage
- optional Bayesian weight blending
- confidence calibration and weighted-vote adjustment

That means the scanner is not using static weights only; it carries forward learned behavior between runs.

## Execution Model

Buddy does not rely on fixed 15/30 pip exits anymore as the main story. Current execution is built around:

- ATR-based stop loss and take profit sizing
- regime-adaptive ATR multipliers
- minimum risk/reward enforcement
- stale-signal protection via max data age
- spread checks before order placement
- trailing-stop and live position management hooks
- RL-assisted sizing / gates / exits when enabled by profile

The default config reflects lessons recorded in `.claude/learnings.md`, especially around:

- keeping `max_data_age_seconds` high enough for real scan latency
- enforcing minimum R:R floors
- tightening disagreement handling after repeated losses
- preventing stale or overwritten state from blocking the pipeline silently

## Brokers And Instruments

The project is no longer OANDA-only.

- Default broker: OANDA
- Supported broker abstraction: OANDA and IBKR
- Default instrument set: major and cross FX pairs
- Futures-oriented profiles and instrument registry entries also exist for `ES`, `NQ`, `CL`, and `GC`

Environment variables Buddy expects for OANDA workflows:

```bash
OANDA_API_TOKEN=...
OANDA_ACCOUNT_ID=...
```

`OANDA_API_KEY` is accepted as a fallback alias in several paths. Local runs automatically try `.env.local` first, then `.env`.

## Quick Start

### 1. Create an environment

The repo includes both `environment_tf_metal.yml` and `environment_intel_mac.yml`, plus `requirements.txt`.

Example:

```bash
conda env create -f environment_tf_metal.yml
conda activate tf-metal
pip install -r requirements.txt
```

Intel Macs can use `environment_intel_mac.yml` instead.

### 2. Set credentials

Create `.env.local` in the repo root:

```bash
OANDA_API_TOKEN=your_token
OANDA_ACCOUNT_ID=your_account_id
```

### 3. Check status

```bash
./bin/Buddy status
```

### 4. Run a dry scan

```bash
./bin/Buddy scan --profile balanced
```

### 5. Run single-pair inference

```bash
./bin/Buddy EUR_USD --dry-run
```

### 6. Execute only when you are ready

```bash
./bin/Buddy EUR_USD --execute
./bin/Buddy scan --auto-execute
```

## Main CLI Commands

Buddy can be run through `./bin/Buddy` or `python main.py`.

### Runtime commands

```bash
./bin/Buddy status
./bin/Buddy scan --profile balanced
./bin/Buddy scan --watch --interval 5
./bin/Buddy scan --supervision --enforce-policy 1
./bin/Buddy EUR_USD --dry-run
./bin/Buddy EUR_USD --execute
./bin/Buddy buddy-chat -I EUR_USD
./bin/Buddy journal --days 30
./bin/Buddy learn
./bin/Buddy analyze --top 10
./bin/Buddy monitor --alerts
./bin/Buddy model-status --decisions
```

### Training and model maintenance

Important: `train-buddy` is single-pair training. Scanner behavior depends on the broader scanner stack and persisted runtime state, not just one isolated model artifact.

```bash
./bin/Buddy train -I EUR_USD --oanda-live
python main.py retrain-gates
python main.py retrain-all
python main.py train-rl-sizer
python main.py train-rl-gates
python main.py train-rl-exits
python main.py promote-model
```

### Useful scan flags

- `--profile`: `conservative`, `balanced`, `aggressive`, `smart`, `futures_paper`, `futures_live`
- `--watch`: continuous scan loop
- `--interval`: watch loop cadence in minutes
- `--auto-execute`: place trades automatically when allowed
- `--diversified`: add diversification filtering
- `--force`: bypass some session/filter restrictions for investigation
- `--futures`: shorthand for IBKR futures paper profile
- `--instruments`: override profile instrument list

## Profiles And Modes

Scan behavior is heavily profile-driven.

- `balanced`: full-featured default and the best general starting point
- `conservative`: fewer trades, stricter thresholds
- `aggressive`: looser thresholds, still risk bounded
- `smart`: profile intended to run agent-confirmation enhancements
- `futures_paper` / `futures_live`: broker/profile presets for IBKR-style workflows

Watch mode and supervision mode are both first-class runtime paths now:

- watch mode repeats scans on a timer
- supervision mode shifts toward a supervisor-first lifecycle with policy enforcement levels

## Learning Loop

Buddy's current identity is tightly tied to its learning pipeline.

### Sources of learning

- `trained_data/trade_journal_rl.json`: scanner trade journal and outcomes
- `.claude/learnings.md`: extracted date-stamped lessons
- `.claude/rules/`: promoted recurring patterns that become active rules
- `.claude/config_adjustments.json`: adaptive tuning outputs
- `trained_data/models/agent_weights.json`: persisted learned agent weights

### Manual learning command

```bash
./bin/Buddy learn
./bin/Buddy learn --status
./bin/Buddy learn --analyze
./bin/Buddy learn --promote
./bin/Buddy learn --consolidate
./bin/Buddy learn --report
```

The current learnings file shows the practical direction of the system:

- root-cause analysis of execution stalls
- threshold calibration for disagreement and confidence systems
- promotion of repeated trade-outcome patterns into enforceable rules
- repeated emphasis on safe state persistence and end-to-end audits

## Ralph And Autonomous Improvement

This repo also contains Ralph, the autonomous dev loop used for PRD-driven code improvement:

- `scripts/ralph.sh`
- `.claude/ralph/prd.json`
- `.claude/agents/`

Ralph is reference and automation infrastructure around the codebase, not the scanner agent team itself. The scanner agents and the `.claude/agents/` prompt library are separate concepts.

## Repo Map

```text
bin/                    launcher scripts
cli/                    command handlers and CLI support
config/                 runtime and profile configuration
docs/                   implementation notes
scripts/                maintenance, validation, Ralph loop, status helpers
src/brokers/            broker abstraction, OANDA, IBKR, instrument registry
src/scanner/            scanner, agents, execution, automation, filters, analytics
src/training/           training pipelines and trainer infrastructure
tests/                  automated coverage
trained_data/           models, journals, diagnostics, adaptive state
.claude/                learnings, rules, Ralph state, operator memory
```

## Recommended First Commands For New Contributors

```bash
./bin/Buddy status
./bin/Buddy scan --profile balanced --no-train
./bin/Buddy journal --days 14
./bin/Buddy learn --status
python main.py --help
```

Those four commands give a quick read on:

- whether credentials and runtime state are loaded
- what the scanner thinks is tradeable
- what recent outcomes look like
- what Buddy has learned recently

## Project Notes

- The repo contains legacy and experimental material. Prefer `src/scanner/`, `src/brokers/`, `cli/`, `config/`, and `scripts/` for the current Buddy path.
- macOS-specific memory and OpenMP safeguards are set early in `main.py`; they are intentional and part of the production story for this codebase.
- Many runtime surfaces are fail-open by design so missing optional subsystems do not crash scans. That makes status, diagnostics, and learnings especially important when troubleshooting.

## Where To Look Next

- Start with `AGENTS.md` for the scanner-agent mental model.
- Read `.claude/learnings.md` for the most recent system lessons and tuning direction.
- Use `python main.py --help` for the full command surface.
