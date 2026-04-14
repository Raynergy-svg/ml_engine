# US-006 Implementation Summary: Update Scanner Data Fetching to Use BrokerClient

## Overview
Successfully integrated BrokerClient abstraction into Scanner's data fetching pipeline, enabling flexible broker implementations while maintaining backward compatibility with legacy OANDA client.

## Changes Made

### 1. Scanner.__init__ (src/scanner/engine.py:47-62)
- **Added**: `broker: Optional[BrokerClient] = None` parameter
- **Maintains**: `oanda_client` parameter for backward compatibility
- **Stores**: Broker instance as `self._broker` for lazy initialization

### 2. New Method: _init_broker_client() (src/scanner/engine.py:1185-1229)
- Lazy-initializes BrokerClient if not provided
- Falls back to OandaBroker.from_env() when broker=None
- Returns True if broker is ready, False otherwise
- Handles ImportError, OSError gracefully with logging

### 3. Updated: _fetch_pair_data() (src/scanner/engine.py:1571-1680)
- **New primary path**: BrokerClient.fetch_candles()
  - Creates Instrument from pair string
  - Sets pip_value for FX instruments: 0.0001 * 100000 = $10/pip
  - Converts CandleData list to DataFrame with identical format
- **Fallback 1**: Legacy OANDA client (unchanged logic)
- **Fallback 2**: Local CSV files (unchanged)
- **Format preserved**: DataFrame with columns [open, high, low, close, volume], time as index

## Key Design Decisions

1. **Broker-first approach**: Tries BrokerClient before OANDA for extensibility
2. **Lazy initialization**: Only initializes when first needed, reduces startup cost
3. **Zero format change**: Output DataFrame format is identical to original OANDA fetch
4. **Graceful degradation**: Falls back to legacy OANDA, then CSV if broker unavailable
5. **FX instrument definition**: Proper pip_value setup (0.0001 * 100k) for validation

## Testing

Created comprehensive test suite: `tests/test_scanner_broker.py`
- 15 test cases, all passing
- Coverage includes:
  - Broker parameter initialization
  - Lazy initialization from environment
  - BrokerClient data fetching
  - Exception handling and fallbacks
  - DataFrame format preservation
  - Feature engineering pipeline compatibility
  - Backward compatibility with legacy oanda_client

## Acceptance Criteria Met

✓ Scanner.__init__ accepts broker: Optional[BrokerClient] = None parameter
✓ _fetch_pair_data() calls broker.fetch_candles(instrument, granularity, count)
✓ Return format is identical: DataFrame with OHLCV columns, time as index
✓ Local CSV fallback still works when broker fetch fails
✓ Feature engineering (_compute_features) receives identical DataFrame format
✓ Backward compatible: Creates OandaBroker from env if no broker passed
✓ Syntax validation passes (py_compile)
✓ All 15 tests pass

## Files Modified

1. **src/scanner/engine.py**
   - Scanner.__init__: +broker parameter, +self._broker field
   - _init_broker_client(): +new method for lazy initialization
   - _fetch_pair_data(): +BrokerClient path, updated fallback messages

2. **tests/test_scanner_broker.py** (NEW)
   - 15 comprehensive tests
   - MockBrokerClient implementation
   - Coverage of all acceptance criteria

## Implementation Notes

### Instrument Creation
BrokerClient expects Instrument objects with required FX fields:
```python
instrument = Instrument(
    symbol=pair,
    broker_symbol=pair.replace("_", "/"),
    asset_class="FX",
    price_precision=5,
    margin_requirement=0.02,
    exchange="OANDA",
    currency="USD",
    pip_value=0.0001 * 100000,  # Critical for FX validation
)
```

### CandleData Conversion
BrokerClient returns List[CandleData] dataclass, converted to DataFrame:
```python
data = []
for candle in candle_list:
    data.append({
        "time": candle.time,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    })
df = pd.DataFrame(data)
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")
```

### Error Handling
Three-tier fallback ensures robustness:
1. BrokerClient fetch_candles() → None on API error
2. Legacy OANDA client get_candles() → None on API error
3. CSV file fallback → None if no CSV found

## Backward Compatibility

- Legacy code passing only oanda_client continues to work
- New code can pass broker parameter
- If neither passed, lazy-init creates OandaBroker from env
- No changes to Scanner's public API beyond optional broker parameter

## Production Readiness

✓ Syntax validated
✓ 15 unit tests passing
✓ All acceptance criteria met
✓ Comprehensive error handling
✓ Lazy initialization (no startup cost)
✓ Zero behavioral change to existing code paths
✓ Ready for integration with BrokerClient implementations
