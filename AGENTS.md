# ML Engine — Scanner Agent Team

## 12 Specialist Agents

Weighted voting system where each agent evaluates one aspect of a trade setup. Agents emit verdicts that combine into a weighted vote score. If the score falls below a regime-aware threshold, the trade is blocked.

| Agent | Base Weight | Default | Purpose |
|-------|-------------|---------|---------|
| trend | 1.15 | ON | SMA crossover + ADX trend strength |
| mean_reversion | 0.90 | ON | RSI-based pullback/extension detection |
| volatility | 1.00 | ON | ATR + regime scoring (LOW/NORMAL/HIGH/EXTREME) |
| risk_sentinel | 1.25 | ON | Drawdown ratio + portfolio risk check |
| uncertainty | 1.10 | ON | Confidence variance + model disagreement |
| execution_quality | 1.05 | ON | Spread, slippage, liquidity assessment |
| momentum | 1.05 | ON | MACD histogram + rate-of-change alignment |
| news_risk | 0.95 | OFF | Headline keyword scanning (NFP, CPI, FOMC) |
| multi_timeframe | 1.10 | OFF | H1/H4/D1 confluence from aggregated candles |
| pair_performance | 0.85 | OFF | Historical win rate per pair |
| session_timing | 0.80 | OFF | Forex session overlap awareness |
| support_resistance | 1.00 | OFF | Swing pivot S/R proximity scoring |

## RL Weight Learning

Weights adapt from trade outcomes via `update_weights_from_outcome()`:
- Agent voted FOR + trade won: weight += 0.10
- Agent voted FOR + trade lost: weight -= 0.15
- Agent voted AGAINST + trade won: weight -= 0.05
- Agent voted AGAINST + trade lost: weight += 0.075

Weights bounded [0.1, 2.0]. Decay toward baseline each scan cycle.

Learned weights persist in `trained_data/models/agent_weights.json`.

## .claude/agents/ Directory

Contains 37 LLM personality prompts (from agency-agents repo) for Claude Code sessions. These are NOT the scanner agents above — they're reference material for engineering, testing, and strategy roles.

## Ralph (Autonomous Dev Loop)

`scripts/ralph.sh` — Spawns fresh AI instances to implement PRD stories iteratively. PRD tracked in `.claude/ralph/prd.json`.


<claude-mem-context>
# Memory Context

# [ml_engine] recent context, 2026-04-16 9:36pm EDT

No previous sessions found.
</claude-mem-context>