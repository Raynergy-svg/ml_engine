import json
from pathlib import Path

from src.scanner.automation.accuracy_gate import AccuracyGate


def test_record_outcome_skips_duplicate_trade_id(tmp_path):
    gate = AccuracyGate(data_path=str(tmp_path / "pair_accuracy.json"))

    gate.record_outcome(
        pair="EUR_USD",
        predicted_direction="LONG",
        actual_outcome=True,
        confidence=0.7,
        trade_id="T100",
    )
    gate.record_outcome(
        pair="EUR_USD",
        predicted_direction="LONG",
        actual_outcome=True,
        confidence=0.7,
        trade_id="T100",
    )

    data = json.loads((tmp_path / "pair_accuracy.json").read_text())
    assert data["EUR_USD"]["total"] == 1
    assert len(data["EUR_USD"]["trades"]) == 1


def test_rebuild_from_journal_replaces_bloated_accuracy_state(tmp_path):
    acc_path = tmp_path / "pair_accuracy.json"
    acc_path.write_text(
        json.dumps({
            "EUR_USD": {
                "trades": [{"trade_id": "dup"} for _ in range(9)],
                "accuracy": 0.0,
                "total": 9,
                "wins": 0,
                "rolling_total": 9,
                "rolling_wins": 0,
            }
        }),
        encoding="utf-8",
    )
    journal_path = tmp_path / "trade_journal_rl.json"
    journal_path.write_text(
        json.dumps([
            {
                "trade_id": "A1",
                "pair": "EUR_USD",
                "direction": "SHORT",
                "confidence": 0.57,
                "outcome": {"trade_won": False, "close_time": "2026-04-02T01:00:00Z"},
            },
            {
                "trade_id": "A2",
                "pair": "EUR_USD",
                "direction": "SHORT",
                "confidence": 0.53,
                "outcome": {"trade_won": False, "close_time": "2026-04-02T02:00:00Z"},
            },
        ]),
        encoding="utf-8",
    )

    gate = AccuracyGate(data_path=str(acc_path))
    rebuilt = gate.rebuild_from_journal(str(journal_path))

    data = json.loads(acc_path.read_text())
    assert rebuilt == 2
    assert data["EUR_USD"]["total"] == 2
    assert data["EUR_USD"]["rolling_total"] == 2
    assert len(data["EUR_USD"]["trades"]) == 2
