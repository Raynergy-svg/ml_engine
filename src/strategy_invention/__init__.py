"""Strategy Invention Engine — Tier 6 Dimension B.

Discovers novel trading strategy combinations via genetic algorithms.
Strategy templates are JSON-serializable DSL objects composed of:
- Entry signal blocks (TrendFollowing, MeanReversion, Momentum, Breakout)
- Exit signal blocks (TrailingStop, ProfitTarget, StopLoss, TimeBased, Correlation)
- Position sizing blocks (Fixed, ConfidenceScaling, VolatilityScaling)
- Pre-trade filter blocks (Risk, Liquidity, Session)

Strategies are evolved offline, shadow-tested live, then promoted to active.
"""
