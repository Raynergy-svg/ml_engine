# Portfolio Optimizer: Dynamic Pair Rotation System

## Overview

The Portfolio Optimizer is a dynamic pair rotation system for the ML Engine trading bot that automatically ranks FX pairs by rolling Sharpe ratio and adjusts which pairs are actively traded vs. observed-only (scanned but not traded).

**Key benefit:** Focus capital on highest-performing pairs while keeping underperformers under observation for learning.

## Quick Start (30 Seconds)

The system is automatically integrated into ContinuousScanner. Run scanner in watch mode:

```bash
python3 buddy_scanner.py trade --watch --interval=5 --force

# Every ~50 minutes (10 × 5-min scan cycles):
# 📊 PAIR ROTATION (Cycle 10)
# 🟢 Active (7): EUR_USD, GBP_USD, USD_JPY, AUD_USD, NZD_USD, USD_CAD, EUR_GBP
# 🟡 Observe-only (8): USD_CHF, EUR_JPY, GBP_JPY, ...
```

## How It Works

1. **Ranks** all pairs by rolling Sharpe ratio (last 50 trades per pair)
2. **Selects** top 7 pairs for active trading
3. **Checks** USD exposure (prevents concentration)
4. **Demotes** underperformers to observe-only mode
5. **Scans** observe pairs for learning (no execution)
6. **Logs** decisions to audit trail (ObservationLog)

## Key Features

- Rolling Sharpe ratio with edge case handling
- Configurable active pair limit (default: 7)
- USD exposure clustering detection (max: 40%)
- Observe-only pairs still scanned (for learning)
- File locking (fcntl) prevents corruption
- Graceful degradation on errors
- Full audit trail (JSON + observation log)

## Configuration

Default settings:
```
rolling_window: 50           # Last 50 trades per pair
max_active_pairs: 7          # Top 7 by Sharpe
max_usd_exposure: 0.40       # Max 40% USD pairs
rotation_interval: 10        # Every 10 scan cycles (~50 min)
```

## Documentation

- **Quick Start**: docs/PORTFOLIO_OPTIMIZER_QUICKSTART.md
- **Full Guide**: docs/PORTFOLIO_OPTIMIZER_GUIDE.md
- **Integration**: docs/PORTFOLIO_OPTIMIZER_INTEGRATION_DETAILS.md
- **Summary**: docs/PORTFOLIO_OPTIMIZER_IMPLEMENTATION_SUMMARY.md

## Testing

```bash
# Run unit tests
python3 -m pytest tests/test_portfolio_optimizer.py -v

# Verify installation
python3 -c "from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer; print('✓')"
```

## Support

1. Check the Quick Start Guide (30-second overview)
2. See the Full Guide (complete reference)
3. Run unit tests (verify setup)
4. Check logs (debug output)

---

**Status**: Production Ready
**Tested**: Yes (20+ unit tests)
**Documented**: Yes (4 guides)
**Last Updated**: 2026-03-19
