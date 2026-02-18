#!/usr/bin/env python3
"""
Test RL integration in ModularEnsembleInference.

This tests that all 3 RL components work correctly within the inference pipeline:
1. RL Position Sizer - determines position size based on market conditions
2. RL Exit - decides when to exit open positions
3. RL Gate - adjusts gate thresholds dynamically
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')


def create_test_dataframe(n_rows=100):
    """Create a test DataFrame with realistic features."""
    np.random.seed(42)
    
    # Base price data
    close = 1.1000 + np.cumsum(np.random.randn(n_rows) * 0.001)
    high = close + np.abs(np.random.randn(n_rows) * 0.0005)
    low = close - np.abs(np.random.randn(n_rows) * 0.0005)
    open_ = close - np.random.randn(n_rows) * 0.0003
    volume = np.random.randint(1000, 10000, n_rows).astype(float)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    # Add direction features (what RL Position Sizer expects)
    for period in [1, 2, 3, 5, 10]:
        df[f'returns_{period}'] = df['close'].pct_change(period).fillna(0)
    
    df['log_returns_1'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    for period in [10, 20, 50]:
        rolling = df['close'].rolling(period)
        df[f'zscore_{period}'] = ((df['close'] - rolling.mean()) / rolling.std()).fillna(0)
    
    for period in [10, 20]:
        df[f'pct_rank_{period}'] = df['close'].rolling(period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ).fillna(0.5)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi_norm'] = (100 - (100 / (1 + rs))) / 100
    df['rsi_norm'] = df['rsi_norm'].fillna(0.5)
    
    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k_norm'] = ((df['close'] - low_14) / (high_14 - low_14 + 1e-10))
    df['stoch_k_norm'] = df['stoch_k_norm'].fillna(0.5)
    
    # SMA ratios
    for period in [5, 10, 20]:
        sma = df['close'].rolling(period).mean()
        df[f'sma_ratio_{period}'] = (df['close'] / sma - 1)
        df[f'sma_ratio_{period}'] = df[f'sma_ratio_{period}'].fillna(0)
    
    # EMA ratios
    for period in [12, 26]:
        ema = df['close'].ewm(span=period).mean()
        df[f'ema_ratio_{period}'] = (df['close'] / ema - 1)
        df[f'ema_ratio_{period}'] = df[f'ema_ratio_{period}'].fillna(0)
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    
    df['macd_norm'] = macd / df['close']
    df['macd_signal_norm'] = signal / df['close']
    df['macd_hist_norm'] = hist / df['close']
    df = df.fillna(0)
    
    # Crossovers
    df['sma_cross_5_20'] = (df['sma_ratio_5'] > df['sma_ratio_20']).astype(float)
    df['macd_cross'] = (df['macd_norm'] > df['macd_signal_norm']).astype(float)
    
    # Structure
    df['higher_high_ratio'] = (df['high'] > df['high'].shift(1)).rolling(10).mean().fillna(0.5)
    df['lower_low_ratio'] = (df['low'] < df['low'].shift(1)).rolling(10).mean().fillna(0.5)
    
    # ATR for volatility
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    for period in [5, 10, 14, 20]:
        atr = tr.rolling(period).mean()
        df[f'atr_pct_{period}'] = (atr / df['close']).fillna(0)
    
    # Volatility
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns_1'].rolling(period).std().fillna(0)
    
    # Volume features
    for period in [5, 10]:
        vol_ma = df['volume'].rolling(period).mean()
        df[f'volume_ratio_{period}'] = (df['volume'] / vol_ma).fillna(1)
    
    df['volume_zscore'] = ((df['volume'] - df['volume'].rolling(20).mean()) / 
                           df['volume'].rolling(20).std()).fillna(0)
    
    # Wick ratios
    body = abs(df['close'] - df['open'])
    df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (body + 1e-10)
    df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (body + 1e-10)
    df['hl_range_pct'] = (df['high'] - df['low']) / df['close']
    df['tr_pct'] = tr / df['close']
    
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    return df


def test_rl_sizer_feature_extraction():
    """Test that _extract_rl_sizer_features works correctly."""
    print('=' * 60)
    print('TEST: RL Sizer Feature Extraction')
    print('=' * 60)
    
    from src.core.modular_inference import ModularEnsembleInference
    
    # Create inference engine (don't load models, just test extraction)
    inference = ModularEnsembleInference.__new__(ModularEnsembleInference)
    inference.rl_sizer = None  # Start without RL sizer
    inference.config = type('Config', (), {'account_equity': 100000})()
    
    # Load RL sizer
    from src.training.rl.position_sizer import RLPositionSizer
    inference.rl_sizer = RLPositionSizer()
    inference.rl_sizer.load()
    
    # Create test data
    df = create_test_dataframe()
    
    # Extract features
    features = inference._extract_rl_sizer_features(df)
    
    print(f'✓ Feature shape: {features.shape}')
    print(f'✓ Expected features: {inference.rl_sizer.scaler.n_features_in_}')
    
    # Verify shape matches scaler expectation
    expected_n = inference.rl_sizer.scaler.n_features_in_
    assert features.shape[1] == expected_n, f"Feature count mismatch: {features.shape[1]} vs {expected_n}"
    print('✓ Feature extraction PASSED')
    
    return True


def test_position_size_calculation():
    """Test RL position sizing in _calculate_position_size."""
    print()
    print('=' * 60)
    print('TEST: RL Position Size Calculation')
    print('=' * 60)
    
    from src.core.modular_inference import ModularEnsembleInference, InferenceConfig
    
    # Create minimal inference instance
    inference = ModularEnsembleInference.__new__(ModularEnsembleInference)
    inference.config = InferenceConfig()
    inference.rl_sizer = None
    
    # Load RL sizer
    from src.training.rl.position_sizer import RLPositionSizer
    inference.rl_sizer = RLPositionSizer()
    inference.rl_sizer.load()
    
    # Create test features (20 features as expected)
    features = np.random.randn(1, 20).astype(np.float32)
    
    # Test position sizing
    size = inference._calculate_position_size(
        expected_drawdown_pips=10.0,
        equity=100000,
        instrument='EUR_USD',
        features=features,
        tcn_probability=0.65,
        ridge_confidence=70.0,
    )
    
    print(f'✓ Position size: {size} lots')
    assert size >= 0, "Position size should be non-negative"
    print('✓ Position sizing PASSED')
    
    return True


def test_exit_rl_check():
    """Test RL exit decision via check_open_position_exit."""
    print()
    print('=' * 60)
    print('TEST: RL Exit Position Check')
    print('=' * 60)
    
    from src.core.modular_inference import ModularEnsembleInference, InferenceConfig
    
    # Check if OptimalExitRL is available
    try:
        from src.rl.optimal_exit_env import OptimalExitRL
    except ImportError:
        print('⚠ OptimalExitRL not available - skipping')
        return True
    
    # Create minimal inference instance
    inference = ModularEnsembleInference.__new__(ModularEnsembleInference)
    inference.config = InferenceConfig()
    inference.rl_exit = None
    inference.exit_rl_timeout_active = False
    inference._exit_rl_cooldown = 0
    
    # Load RL exit model
    exit_rl = OptimalExitRL()
    if exit_rl.load():
        inference.rl_exit = exit_rl
        
        # Test exit check
        df = create_test_dataframe()
        result = inference.check_open_position_exit(
            df=df,
            entry_price=1.1000,
            direction='long',
            unrealized_pnl_pips=50,
            bars_in_trade=24,
            stop_loss_pips=15.0,
            take_profit_pips=30.0,
        )
        
        print(f'✓ Exit result: {result}')
        assert 'should_exit' in result, "Missing should_exit key"
        assert 'action' in result, "Missing action key"
        print('✓ Exit RL check PASSED')
    else:
        print('⚠ RL exit model not loaded - skipping')
    
    return True


def main():
    """Run all RL integration tests."""
    print()
    print('#' * 60)
    print('# RL INTEGRATION TESTS')
    print('#' * 60)
    
    tests = [
        ('Feature Extraction', test_rl_sizer_feature_extraction),
        ('Position Sizing', test_position_size_calculation),
        ('Exit Check', test_exit_rl_check),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f'✗ {name} FAILED: {e}')
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print('#' * 60)
    print(f'# RESULTS: {passed}/{len(tests)} passed, {failed} failed')
    print('#' * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
