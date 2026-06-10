# ML Engine — Scanner Agent Team

## 15 Specialist Agents (core 12 + extended 3)

Weighted voting system where each agent evaluates one aspect of a trade setup. Agents emit verdicts that combine into a weighted vote score. If the score falls below a regime-aware threshold, the trade is blocked. **Truth source:** `_BASE_WEIGHTS` in `src/scanner/agents/_team.py`; per-agent toggles are `enable_*_agent` fields in `src/scanner/config.py` (profile dicts may override). Defaults below are the dataclass field defaults.

### Core 12

| Agent | Base Weight | Default | Purpose |
|-------|-------------|---------|---------|
| trend | 1.15 | ON | SMA crossover + ADX trend strength. `passed=False` is a HARD veto on directional trades |
| mean_reversion | 0.90 | ON | RSI-based pullback/extension detection |
| volatility | 1.00 | ON | ATR + regime scoring (LOW/NORMAL/HIGH/EXTREME) |
| risk_sentinel | 1.25 | ON | Drawdown ratio + portfolio risk check |
| uncertainty | 1.10 | ON | Confidence variance + model disagreement |
| execution_quality | 1.05 | ON | Spread, slippage, liquidity assessment |
| momentum | 1.05 | ON | MACD histogram + rate-of-change alignment |
| news_risk | 0.95 | ON | Headline keyword scanning (NFP, CPI, FOMC) |
| multi_timeframe | 1.10 | ON | H1/H4/D1 confluence from aggregated candles |
| pair_performance | 0.85 | ON | Historical win rate per pair |
| session_timing | 0.80 | ON | Forex session overlap awareness |
| support_resistance | 1.00 | ON | Swing pivot S/R proximity scoring |

### Extended 3

| Agent | Base Weight | Default | Purpose |
|-------|-------------|---------|---------|
| order_flow | 0.95 | ON | OANDA order/position-book contrarian signal (graded `pb_*` features) |
| trader_readiness | 0.50 | OFF (dormant) | Aura human-side readiness signal; abstains gracefully until the Aura writer ships |
| devil_advocate | 1.30 | ON | Adversarial bear-case evaluator — **runs LAST**, can veto an otherwise-passing setup |

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
