# Buddy Scanner Research Report

**Date:** January 27, 2026  
**Subject:** Scanner Errors, WebSocket Capabilities, and Dependency Analysis

---

## 1. THE ACTUAL ERROR CAUSING `buddy scan` TO FAIL

### Root Cause: Incorrect Config Path
```
FileNotFoundError: Configuration file not found: ./config_improved_H1.yaml
```

**Location:** [main.py#L567](main.py#L567)
```python
DEFAULT_CONFIG_PATH = "./config_improved_H1.yaml"  # ← WRONG: relative path
```

**What happens:**
1. When `buddy scan` is invoked via CLI, it calls `main.py buddy_scan()`
2. `buddy_scan()` uses `DEFAULT_CONFIG_PATH` which is `./config_improved_H1.yaml`
3. This expects config in CWD, but actual path is `config/config_improved_H1.yaml`

**Why `bin/Buddy` works:** The shell script correctly sets:
```bash
CONFIG_FILE="${ROOT_DIR}/config/config_improved_H1.yaml"
```
And passes it via `--config "${CONFIG_FILE}"` to main.py.

### Fix Required
Change `DEFAULT_CONFIG_PATH` in main.py from:
```python
DEFAULT_CONFIG_PATH = "./config_improved_H1.yaml"
```
To:
```python
DEFAULT_CONFIG_PATH = "config/config_improved_H1.yaml"
```

Or use a smarter resolver that checks both locations.

---

## 2. OANDA WEBSOCKET/STREAMING CAPABILITIES

### ✅ OANDA DOES Support Real-Time Streaming

**Streaming API URLs:**
| Environment | URL |
|-------------|-----|
| Practice | `https://stream-fxpractice.oanda.com/` |
| Live | `https://stream-fxtrade.oanda.com/` |

**Endpoint:** `GET /v3/accounts/{accountID}/pricing/stream`

**Key Details:**
- Up to **4 prices per second** (every 250ms) per instrument
- Up to **20 concurrent streams** per IP
- Uses HTTP streaming (not WebSocket) - server sends newline-delimited JSON
- During rapid price movement, only end-of-window price is sent

### Current Implementation Status

**[src/utils/oanda_practice.py](src/utils/oanda_practice.py):**
- ❌ **NO streaming support implemented**
- Only REST API methods:
  - `get_candles()` - Historical candlestick data
  - `get_pricing()` - Snapshot pricing (single request)
  - `get_price_quote()` - Best bid/ask snapshot
  - `create_market_order()` - Order execution
  - `close_trade()` - Trade management

### Streaming Implementation Recommendation

```python
# Example streaming price client (to be added to oanda_practice.py)
PRACTICE_STREAM_URL = "https://stream-fxpractice.oanda.com/v3"

def stream_prices(self, instruments: List[str], callback: Callable):
    """Stream real-time prices for instruments."""
    url = f"{PRACTICE_STREAM_URL}/accounts/{self._config.account_id}/pricing/stream"
    params = {"instruments": ",".join(instruments)}
    
    with self._session.get(url, params=params, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("type") == "PRICE":
                    callback(data)
```

---

## 3. IMPORT DEPENDENCY GRAPH

### Scanner Module Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRY POINTS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  main.py                   bin/Buddy (shell)                                │
│    │                          │                                             │
│    │ buddy_scan()             │ cmd_scan()                                  │
│    │    │                     │    │                                        │
│    └────┴──────────────┬──────┴────┘                                        │
│                        │                                                    │
│                        ▼                                                    │
│              buddy_scanner.py  (ROOT)                                       │
│              ┌────────────────────────────────────────────────┐             │
│              │ BuddyScanner class                             │             │
│              │ - Main scanner implementation                  │             │
│              │ - 2233 lines                                   │             │
│              │ - Lazy imports (all in methods, not module)    │             │
│              └─────────────────┬──────────────────────────────┘             │
│                                │                                            │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────────┐
                    │            │                │
                    ▼            ▼                ▼
    ┌───────────────────┐  ┌──────────────┐  ┌────────────────────────┐
    │ pair_scanner.py   │  │ multi_pair_  │  │ src/core/              │
    │ (ROOT - wrapper)  │  │ inference.py │  │ modular_inference.py   │
    │                   │  │              │  │                        │
    │ Re-exports from   │  │ Pair-specific│  │ ModularEnsembleInference│
    │ src/utils/        │  │ model loader │  │ (4-gate ensemble)      │
    │ pair_scanner.py   │  │              │  │                        │
    └────────┬──────────┘  └──────────────┘  └────────────────────────┘
             │
             ▼
    ┌───────────────────────────┐
    │ src/utils/pair_scanner.py │
    │                           │
    │ - PairAnalysis dataclass  │
    │ - ScanResult dataclass    │
    │ - PairScanner class       │
    │ - scan_pairs_quick()      │
    │ - Constants: ALL_PAIRS,   │
    │   MAJOR_PAIRS, PIP_VALUES │
    └───────────────────────────┘
```

### Detailed Lazy Import Chain (buddy_scanner.py)

The scanner uses **lazy imports inside methods**, not at module level:

| Method | Imports | Purpose |
|--------|---------|---------|
| `_load_config()` | `src.utils.load_config` | YAML config loading |
| `_init_oanda_client()` | `src.utils.oanda_practice.OandaPracticeClient` | API client |
| `_init_feature_engineer()` | `src.data.feature_engineering.FeatureEngineering` | Feature computation |
| `_init_modular_ensemble()` | `src.core.modular_inference.ModularEnsembleInference` | Ensemble model |
| `_init_multi_pair_inference()` | `multi_pair_inference.MultiPairInference` | Pair-specific models |
| `_init_position_sizer()` | `src.risk.position_sizing.*` | Position sizing |
| `_init_risk_manager()` | `src.risk.risk_management.*` | Risk management |
| `_init_memory_client()` | `memory_client.MLEngineMemory` | Historical accuracy |
| `_fetch_pair_data()` | `src.utils.fx_paper.candles_to_ohlcv_df` | Data parsing |

### No Circular Dependencies Found

The import structure is **clean** because:
1. `buddy_scanner.py` uses **lazy imports** (imports inside methods, not at module top)
2. `pair_scanner.py` (root) is a **re-export wrapper** only
3. `multi_pair_inference.py` has **no src.* imports** (self-contained)

**However**, there is **duplicate code**:
- `buddy_scanner.py` defines its own `MAJOR_PAIRS`, `ALL_PAIRS`, `PIP_VALUES`
- `src/utils/pair_scanner.py` defines the same constants
- These are **not shared** - potential sync issues

---

## 4. MINIMUM REQUIRED DEPENDENCIES FOR A CLEAN SCANNER

### Current Dependencies (Full BuddyScanner)

```
buddy_scanner.py
├── Standard Library (concurrent.futures, dataclasses, datetime, json, logging, pathlib)
├── numpy, pandas
├── rich (optional - pretty output)
├── src.utils.load_config
├── src.utils.oanda_practice.OandaPracticeClient
├── src.utils.fx_paper.candles_to_ohlcv_df
├── src.data.feature_engineering.FeatureEngineering
├── src.core.modular_inference.ModularEnsembleInference
├── src.risk.position_sizing.*
├── src.risk.risk_management.*
├── multi_pair_inference.MultiPairInference
└── memory_client.MLEngineMemory
```

### Minimal Scanner (Technical-Only, No ML)

For a **lightweight scanner** that works without trained models:

```python
# Minimum required:
import numpy as np
import pandas as pd
from src.utils.oanda_practice import OandaPracticeClient  # OANDA data
# That's it for basic technical scanning
```

Features available without ML:
- Fetch OHLCV data for all pairs
- Calculate ATR, RSI, MACD, trend strength
- Rank pairs by volatility/trend
- Basic position sizing (fixed risk %)

### What Each Module Provides

| Module | Purpose | Required? |
|--------|---------|-----------|
| `oanda_practice.py` | Data fetching + order execution | **YES** |
| `fx_paper.py` | DataFrame parsing helpers | **YES** |
| `feature_engineering.py` | Advanced features (200+) | Optional |
| `modular_inference.py` | 4-gate ML ensemble | Optional |
| `multi_pair_inference.py` | Pair-specific models | Optional |
| `position_sizing.py` | Kelly/RL position sizing | Optional |
| `risk_management.py` | Risk rules | Optional |
| `memory_client.py` | Historical accuracy tracking | Optional |

---

## 5. RECOMMENDATIONS

### Immediate Fixes

1. **Fix DEFAULT_CONFIG_PATH in main.py:**
   ```python
   DEFAULT_CONFIG_PATH = "config/config_improved_H1.yaml"
   ```

2. **Add config path resolver:**
   ```python
   def _resolve_config_path(config_path: str) -> str:
       """Resolve config path, checking multiple locations."""
       candidates = [
           config_path,
           f"config/{config_path}",
           Path(__file__).parent / config_path,
           Path(__file__).parent / "config" / config_path,
       ]
       for p in candidates:
           if Path(p).exists():
               return str(p)
       raise FileNotFoundError(f"Config not found: {config_path}")
   ```

### Future Enhancements

1. **Add OANDA Streaming Support:**
   - Implement `stream_prices()` in `oanda_practice.py`
   - Add streaming-based scanner mode for real-time monitoring
   - Use `asyncio` for non-blocking price updates

2. **Consolidate Constants:**
   - Move `MAJOR_PAIRS`, `ALL_PAIRS`, `PIP_VALUES` to a single source
   - Have `buddy_scanner.py` import from `src/utils/pair_scanner.py`

3. **Create Lightweight Scanner Module:**
   - Extract minimal scanner into `src/scanner/quick_scan.py`
   - No ML dependencies
   - Fast pair ranking based on technicals only

---

## 6. SUMMARY

| Item | Status |
|------|--------|
| **Primary Error** | `FileNotFoundError` - wrong config path in main.py |
| **OANDA Streaming** | ✅ Supported but ❌ Not implemented in codebase |
| **Circular Imports** | ✅ None found (lazy imports) |
| **Duplicate Code** | ⚠️ Constants defined in multiple places |
| **Min Dependencies** | `numpy`, `pandas`, `oanda_practice.py`, `fx_paper.py` |

**Action Required:** Fix `DEFAULT_CONFIG_PATH` in main.py line 567.
