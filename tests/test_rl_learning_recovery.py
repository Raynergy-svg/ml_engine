import json
import logging
from pathlib import Path

from src.scanner.expectancy_tracker import ExpectancyConfig, ExpectancyTracker
from src.scanner.walkforward_retrainer import WalkForwardRetrainer


def test_expectancy_stale_warning_threshold_respects_config(tmp_path, caplog):
    path = tmp_path / "expectancy_tracker.json"
    path.write_text(json.dumps({
        "version": "1.0.0",
        "timestamp": 0.0,
        "windows": {},
    }))

    tracker = ExpectancyTracker(ExpectancyConfig(
        persistence_path=str(path),
        stale_warning_hours=999999.0,
    ))

    with caplog.at_level(logging.WARNING):
        tracker.load_state(str(path))

    assert "stale" not in caplog.text.lower()


def test_walkforward_logs_pending_outcome_context(tmp_path, caplog):
    journal_path = tmp_path / "trade_journal_rl.json"
    journal_path.write_text(json.dumps([
        {"trade_id": "1", "outcome": None},
        {"trade_id": "2", "outcome": None},
    ]))

    retrainer = WalkForwardRetrainer()
    with caplog.at_level(logging.INFO):
        epochs = retrainer.run(str(journal_path))

    assert epochs == []
    assert "pending_outcomes=2" in caplog.text
