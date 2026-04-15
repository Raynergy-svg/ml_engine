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
- LOW volatility regime MUST use sl_mult >= 1.2 (NOT 0.8). Source: 2026-04-15 Trade 1220 — LOW regime's 0.8 multiplier produced SL/ATR=1.17x in a ranging market where price noise exceeds directional movement. Ranging markets need WIDER relative SL to survive chop, not tighter. LOW regime sl_mult=0.8 is the inverse of correct behavior.

## Agent Consensus
- Higher weighted_vote_score (>0.65) correlates with better outcomes — prefer these
- Uncertainty score > 0.45 is a warning signal — trade with caution
- Model disagreement > 0.30 is a loss predictor — reduce confidence or skip
- NEVER allow directional trade execution when trend agent returns passed=False — promotion of a "soft" penalty to a hard block. Source: 2026-04-15 Trade 1220 EUR_AUD entered with ADX=1 (no trend) because risk_sentinel + execution_quality compensated WVS to 0.76. Trend-following directional trades require a trend; agent veto is absolute.
- When max_component_age_days > 7 for any ensemble component, HARD-BLOCK execution on uncertainty_score > 0.35 (not 0.45). Source: 2026-04-15 10-loss streak — 28d-old models produced confidence ~0.69 that soft penalty only reduced to ~0.56 (above 0.50 threshold). Soft penalties are insufficient during model staleness; tighten the block.

## Session Discipline
- Update .claude/state.json before session ends
- Extract learnings from every trade outcome (win or loss)
- Never re-enter a position at the same SL/TP if the entry price has changed — recalculate

## Promoted Rules
- [2026-04-02] agent_accuracy: Lower max_uncertainty_score by 0.02 (uncertainty predicted 6 losses)
- [2026-04-02] agent_accuracy: Lower max_model_disagreement by 0.02 (disagreement predicted 24 losses)
- [2026-04-02] sl_tp: Increase atr_tp_multiplier by 0.1 (TP hit too fast 6 times)
- [2026-04-15] agent_veto: Trend agent passed=False is a hard block for directional trades (not WVS reduction). Source: 1 observation (Trade 1220 EUR_AUD, ADX=1, executed on WVS=0.76 compensation, lost).
- [2026-04-15] staleness_block: Uncertainty hard-block threshold drops from 0.45 to 0.35 when max_component_age_days > 7. Source: 1 catastrophic observation (10-loss streak, $2,637 loss, soft penalty proved insufficient).
- [2026-04-15] regime_sl: LOW regime sl_mult >= 1.2 (inversion of current 0.8). Source: 1 observation (Trade 1220, LOW regime tightened SL in ranging market).

Note: three 2026-04-15 rules promoted on single-observation catastrophic evidence. Standard promotion criterion is 3+ observations; exception made because the observation was a 10-trade consecutive loss streak with documented root cause. Re-validate after 30 days of live data.
