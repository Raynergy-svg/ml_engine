from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np

from market_intelligence import OnlineLearner, TradeOutcome
from memory_client import MLEngineMemory


def _make_trade(trade_id: str = "trade-1") -> TradeOutcome:
    entry_time = datetime(2026, 2, 27, 12, 0, 0)
    exit_time = entry_time + timedelta(hours=1)
    return TradeOutcome(
        trade_id=trade_id,
        instrument="EUR_USD",
        direction=1,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=1.0825,
        exit_price=1.084,
        pnl_pips=15.0,
        pnl_percent=0.35,
        features=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        prediction=0.72,
        confidence=0.81,
        actual_outcome=1,
    )


def test_online_learner_rebuilds_buffer_from_shared_memory(tmp_path):
    memory = MLEngineMemory(storage_path=tmp_path / ".memory")
    learner = OnlineLearner(
        buffer_size=10,
        retrain_threshold=3,
        storage_dir=str(tmp_path / "legacy"),
        memory_client=memory,
    )

    learner.record_trade(_make_trade())

    reloaded = OnlineLearner(
        buffer_size=10,
        retrain_threshold=3,
        storage_dir=str(tmp_path / "legacy"),
        memory_client=memory,
    )

    assert len(memory.get_recent_trades()) == 1
    assert len(reloaded.trade_buffer) == 1
    assert reloaded.trade_buffer[0].trade_id == "trade-1"
    assert np.allclose(reloaded.trade_buffer[0].features, np.array([0.1, 0.2, 0.3, 0.4]))
    assert reloaded.trades_since_retrain == 1
    assert memory.get_runtime_state("online_learning")["trades_since_retrain"] == 1


def test_online_learner_mark_retrained_persists_runtime_state(tmp_path):
    memory = MLEngineMemory(storage_path=tmp_path / ".memory")
    learner = OnlineLearner(
        buffer_size=10,
        retrain_threshold=2,
        storage_dir=str(tmp_path / "legacy"),
        memory_client=memory,
    )

    learner.record_trade(_make_trade())
    learner.mark_retrained()

    reloaded = OnlineLearner(
        buffer_size=10,
        retrain_threshold=2,
        storage_dir=str(tmp_path / "legacy"),
        memory_client=memory,
    )

    assert reloaded.trades_since_retrain == 0
    assert len(reloaded.trade_buffer) == 1
    assert reloaded.trade_buffer[0].trade_id == "trade-1"


def test_online_learner_migrates_legacy_trade_buffer_once(tmp_path):
    memory = MLEngineMemory(storage_path=tmp_path / ".memory")
    storage_dir = tmp_path / "legacy"
    storage_dir.mkdir()
    trade = _make_trade("legacy-1")

    legacy_payload = {
        "trades_since_retrain": 4,
        "trades": [
            {
                "trade_id": trade.trade_id,
                "instrument": trade.instrument,
                "direction": trade.direction,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl_pips": trade.pnl_pips,
                "pnl_percent": trade.pnl_percent,
                "features": trade.features.tolist(),
                "prediction": trade.prediction,
                "confidence": trade.confidence,
                "actual_outcome": trade.actual_outcome,
            }
        ],
    }
    (storage_dir / "trade_buffer.json").write_text(json.dumps(legacy_payload))

    learner = OnlineLearner(
        buffer_size=10,
        retrain_threshold=5,
        storage_dir=str(storage_dir),
        memory_client=memory,
    )

    assert len(learner.trade_buffer) == 1
    assert learner.trade_buffer[0].trade_id == "legacy-1"
    assert learner.trades_since_retrain == 4
    assert len(memory.get_recent_trades()) == 1

    reloaded = OnlineLearner(
        buffer_size=10,
        retrain_threshold=5,
        storage_dir=str(storage_dir),
        memory_client=memory,
    )

    assert len(reloaded.trade_buffer) == 1
    assert reloaded.trade_buffer[0].trade_id == "legacy-1"
    assert reloaded.trades_since_retrain == 4
    assert memory.get_runtime_state("online_learning")["legacy_trade_buffer_migrated"] is True


def test_shared_trade_metadata_is_pruned(tmp_path):
    memory = MLEngineMemory(storage_path=tmp_path / ".memory")

    memory.log_trade({
        "timestamp": "2026-02-28T12:00:00+00:00",
        "trade_id": "meta-1",
        "instrument": "EUR_USD",
        "direction": "long",
        "confidence": 0.71,
        "model": "market_intelligence",
        "metadata": {
            "headline_blob": "A" * 400,
            "events": [{"name": f"event-{idx}", "payload": "B" * 120} for idx in range(20)],
            "nested": {"a": {"b": {"c": {"deep": "value"}}}},
        },
    })

    stored = memory.get_trade("meta-1")
    metadata = stored["metadata"]

    assert metadata["headline_blob"].endswith("...")
    assert len(metadata["headline_blob"]) <= 180
    assert len(metadata["events"]) == 9
    assert metadata["events"][-1]["_truncated_items"] == 12
    assert isinstance(metadata["nested"]["a"]["b"], str)
