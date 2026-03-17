# ML Engine (Buddy) - FX Trading Bot

Autonomous ML-powered forex trading system. Scans markets, evaluates setups through multi-agent consensus, executes on OANDA, and learns from outcomes.

## Architecture
```
Scanner (engine.py) → Agents (agents.py) → Gates → Execution (execution.py) → OANDA
     ↑                                                        ↓
     └── Config Tuner ← Rules ← Learnings ← RL Feedback ←── Trade Outcomes
```

## Core Loop
1. **Scan**: Multi-pair analysis with TCN/Ridge/RF ensemble models
2. **Agents**: Trend, volatility, uncertainty, multi-timeframe, pair performance
3. **Gates**: Confidence, momentum, risk — all must pass
4. **Execute**: ATR-based SL/TP, regime-aware position sizing
5. **Monitor**: Drawdown guardian, trailing SL, real-time P/L
6. **Learn**: RL weight updates, trade journal, pattern extraction

## Key Decisions
- Soft uncertainty blocking (confidence penalty) over hard circuit breaker
- ATR-based dynamic SL/TP over hardcoded pip values
- Correlation filter prevents double exposure on correlated pairs
- Minimum R:R ratio 1.2:1 gate before execution
- Position sizing scales to account size (5% base risk on practice)

## Self-Improvement
- Learnings: `.claude/learnings.md` — date-stamped insights from trade outcomes
- Rules: `.claude/rules/` — promoted patterns that actively gate behavior
- State: `.claude/state.json` — session continuity across context windows
- Config: `.claude/config_adjustments.json` — adaptive parameter tuning

## Key Files
- `buddy_scanner.py` — CLI entry point (scan/watch/trade/learn)
- `src/scanner/engine.py` — Core scanner with model ensemble
- `src/scanner/agents.py` — Sub-inference agent team
- `src/scanner/execution.py` — OANDA trade execution + RL sync
- `src/scanner/automation/continuous.py` — Watch mode loop
- `src/risk/position_sizing.py` — Regime-aware position sizer
- `trained_data/trade_journal_rl.json` — Trade outcomes for RL
- `trained_data/models/agent_weights.json` — Learned agent weights
