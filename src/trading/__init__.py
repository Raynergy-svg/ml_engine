"""Trading module for FX execution.

This module provides:
- Trade execution via OANDA API
- Paper trade planning and simulation
- Order building and risk management utilities

Example usage:
    from src.trading import TradeExecutor, OrderRequest, ExecutionResult
    
    executor = TradeExecutor.from_env()
    order = OrderRequest(
        instrument="EUR_USD",
        direction="long",
        units=10000,
        stop_loss_pips=15.0,
        take_profit_pips=30.0,
    )
    result = executor.execute(order)
"""

from src.trading.execution import (
    ExecutionResult,
    OrderRequest,
    TradeExecutor,
    PositionInfo,
    QuoteInfo,
)

from src.trading.paper_trade import (
    PaperTradePlan,
    PaperTradeContext,
    build_paper_trade_plan,
    execute_paper_trade_plan,
    setup_paper_trade_context,
)

__all__ = [
    # Execution
    "ExecutionResult",
    "OrderRequest",
    "TradeExecutor",
    "PositionInfo",
    "QuoteInfo",
    # Paper trading
    "PaperTradePlan",
    "PaperTradeContext",
    "build_paper_trade_plan",
    "execute_paper_trade_plan",
    "setup_paper_trade_context",
]
