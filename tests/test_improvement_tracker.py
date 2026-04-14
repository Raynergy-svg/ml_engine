from pathlib import Path

from src.scanner.automation.improvement_tracker import ImprovementTracker


def test_generate_report_falls_back_to_closed_journal(tmp_path, monkeypatch):
    journal_path = tmp_path / "trained_data" / "trade_journal_rl.json"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(
        """
[
  {
    "pair": "EUR_USD",
    "outcome": {"trade_won": true, "realized_pl": 12.5, "pnl_pips": 8.0}
  },
  {
    "pair": "GBP_USD",
    "outcome": {"trade_won": false, "realized_pl": -5.0, "pnl_pips": -4.0}
  }
]
""".strip()
    )
    monkeypatch.setattr(
        "src.scanner.automation.improvement_tracker.JOURNAL_PATH",
        journal_path,
    )

    tracker = ImprovementTracker(log_path=tmp_path / "trained_data" / "improvement_log.jsonl")
    report = tracker.generate_report()

    assert "Source: trade_journal_rl.json (session log empty)" in report
    assert "Closed trades in journal: 2 (1W / 1L)" in report
    assert "Total P/L: $7.50" in report


def test_get_trend_returns_dashboard_fields_from_records(tmp_path):
    log_path = tmp_path / "trained_data" / "improvement_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                '{"win_rate": 0.4, "net_pnl": -10.0, "learnings_added": 1}',
                '{"win_rate": 0.6, "net_pnl": 15.0, "learnings_added": 3}',
            ]
        )
    )

    tracker = ImprovementTracker(log_path=log_path)
    trend = tracker.get_trend()

    assert trend["sessions_analyzed"] == 2
    assert trend["recent_win_rate"] == 0.5
    assert trend["recent_pnl"] == 2.5
    assert trend["pnl_trend"] in {"improving", "declining", "stable"}


def test_get_trend_falls_back_to_journal_when_no_records(tmp_path, monkeypatch):
    journal_path = tmp_path / "trained_data" / "trade_journal_rl.json"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(
        """
[
  {
    "pair": "USD_JPY",
    "outcome": {"trade_won": true, "realized_pl": 22.0, "pnl_pips": 11.0}
  }
]
""".strip()
    )
    monkeypatch.setattr(
        "src.scanner.automation.improvement_tracker.JOURNAL_PATH",
        journal_path,
    )

    tracker = ImprovementTracker(log_path=tmp_path / "trained_data" / "improvement_log.jsonl")
    trend = tracker.get_trend()

    assert trend["sessions_analyzed"] == 1
    assert trend["recent_win_rate"] == 1.0
    assert trend["recent_pnl"] == 22.0
