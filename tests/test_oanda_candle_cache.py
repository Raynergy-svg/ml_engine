import pytest

pytest.importorskip("oanda_candle_cache")

import pandas as pd

from oanda_candle_cache import merge_dedupe_by_time


def test_merge_dedupe_by_time_sorts_and_dedupes_keep_last():
    existing = pd.DataFrame(
        {
            "time": [
                "2025-01-01T00:00:00.000000Z",
                "2025-01-01T00:05:00.000000Z",
            ],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [10.0, 20.0],
        }
    )

    incoming = pd.DataFrame(
        {
            "time": [
                # duplicate time with updated values
                "2025-01-01T00:05:00.000000Z",
                # new time out of order
                "2025-01-01T00:10:00.000000Z",
            ],
            "open": [20.0, 3.0],
            "high": [20.1, 3.1],
            "low": [19.9, 2.9],
            "close": [20.05, 3.05],
            "volume": [200.0, 30.0],
        }
    )

    out = merge_dedupe_by_time(existing, incoming)
    assert list(out["time"].astype(str)) == [
        "2025-01-01 00:00:00+00:00",
        "2025-01-01 00:05:00+00:00",
        "2025-01-01 00:10:00+00:00",
    ]
    # Ensure duplicate kept last (incoming)
    row = out.iloc[1]
    assert float(row["open"]) == 20.0
    assert float(row["close"]) == 20.05
# — Raynergy-svg —
