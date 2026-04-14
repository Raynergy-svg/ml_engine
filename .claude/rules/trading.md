# Trading Rules

Imperative rules that actively gate Buddy's trading behavior. Promoted from repeated learnings.

## Execution Gates
- NEVER execute a trade with R:R ratio below 1.2:1 (TP_pips / SL_pips >= 1.2)
- ALWAYS run correlation filter before execution to prevent double exposure
- ALWAYS log every trade to trade_journal_rl.json with full gate/agent context
- NEVER skip RL sync after a trade closes — outcomes must feed back to agent weights

## Risk Management
- Drawdown guardian runs every scan cycle — non-negotiable
- Maximum portfolio risk: 15% of NAV across all open positions
- Position sizing uses ATR-based SL (not hardcoded pips)
- SL = ATR * atr_sl_multiplier, TP = ATR * atr_tp_multiplier

## Agent Consensus
- Higher weighted_vote_score (>0.65) correlates with better outcomes — prefer these
- Uncertainty score > 0.45 is a warning signal — trade with caution
- Model disagreement > 0.30 is a loss predictor — reduce confidence or skip

## Session Discipline
- Update .claude/state.json before session ends
- Extract learnings from every trade outcome (win or loss)
- Never re-enter a position at the same SL/TP if the entry price has changed — recalculate

## Promoted Rules
- [2026-04-02] agent_accuracy: Lower max_uncertainty_score by 0.02 (uncertainty predicted 6 losses)
- [2026-04-02] agent_accuracy: Lower max_model_disagreement by 0.02 (disagreement predicted 24 losses)
- [2026-04-02] sl_tp: Increase atr_tp_multiplier by 0.1 (TP hit too fast 6 times)
