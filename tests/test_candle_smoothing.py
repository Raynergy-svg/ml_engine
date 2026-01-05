import pandas as pd

from candle_smoothing import resample_5min_ohlcv_and_ema_close


def test_resample_5min_and_ema14_close():
    # 1-minute candles for 10 minutes -> 2x 5-minute bars
    times = pd.date_range("2025-01-01T00:00:00Z", periods=10, freq="1min")

    df = pd.DataFrame(
        {
            "time": times.astype(str),
            "open": [1.0 + i * 0.01 for i in range(10)],
            "high": [1.1 + i * 0.01 for i in range(10)],
            "low": [0.9 + i * 0.01 for i in range(10)],
            "close": [1.05 + i * 0.01 for i in range(10)],
            "volume": [100] * 10,
        }
    )

    out = resample_5min_ohlcv_and_ema_close(df, ema_span=14)

    assert list(out.columns) == ["time", "open", "high", "low", "close", "volume", "close_ema_14"]
    assert len(out) == 2

    # OHLC aggregation correctness
    # First bin: minutes 00:00-00:04
    assert out["open"].iloc[0] == df["open"].iloc[0]
    assert out["high"].iloc[0] == df["high"].iloc[0:5].max()
    assert out["low"].iloc[0] == df["low"].iloc[0:5].min()
    first_close = float(df["close"].iloc[4])

    # Second bin: minutes 00:05-00:09
    second_close = float(df["close"].iloc[9])

    # EMA(14) on resampled closes
    alpha = 2.0 / (14 + 1)
    ema0 = first_close
    ema1 = alpha * second_close + (1 - alpha) * ema0

    assert abs(float(out["close"].iloc[0]) - first_close) < 1e-12
    assert abs(float(out["close"].iloc[1]) - second_close) < 1e-12
    assert abs(float(out["close_ema_14"].iloc[0]) - ema0) < 1e-12
    assert abs(float(out["close_ema_14"].iloc[1]) - ema1) < 1e-12
# — Raynergy-svg —
