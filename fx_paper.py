"""Shim module - re-exports from src.utils.fx_paper for backward compatibility."""
from src.utils.fx_paper import *
from src.utils.fx_paper import (
    candles_to_ohlcv_df,
    pip_size,
    simulate_tp_sl_outcome,
    conservative_slippage_pips,
    spread_pips_from_df,
    atr,
    setup_signal,
    RiskRules,
    position_size_units,
)
