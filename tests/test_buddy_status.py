from pathlib import Path

from scripts import buddy_status


def test_show_risk_scaler_prefers_adaptive_sizer_state(tmp_path, capsys, monkeypatch):
    root = Path(tmp_path)
    (root / "trained_data").mkdir()
    (root / ".claude").mkdir()

    (root / "trained_data" / "adaptive_sizer_state.json").write_text(
        """
        {
          "config": {
            "drawdown_floor": 0.3,
            "max_acceptable_drawdown": 0.15
          },
          "trade_history": [
            [100.0, true],
            [-50.0, false],
            [-25.0, false]
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (root / ".claude" / "state.json").write_text(
        '{"risk_scaler": {"level": "reduced", "factor": 0.7}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(buddy_status, "_ROOT", root)

    buddy_status.show_risk_scaler()
    out = capsys.readouterr().out

    assert "adaptive position sizer" in out
    assert "0.31x position sizing" in out
    assert "1/3 wins" in out
    assert "no scaler state recorded" not in out


def test_show_risk_scaler_uses_legacy_state_when_adaptive_missing(tmp_path, capsys, monkeypatch):
    root = Path(tmp_path)
    (root / "trained_data").mkdir()
    (root / ".claude").mkdir()

    (root / "trained_data" / "risk_scaler_state.json").write_text(
        """
        {
          "scale_factor": 0.7,
          "consecutive_wins": 0,
          "consecutive_losses": 3,
          "total_trades": 8
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(buddy_status, "_ROOT", root)

    buddy_status.show_risk_scaler()
    out = capsys.readouterr().out

    assert "legacy adaptive risk scaler" in out
    assert "0.70x position sizing" in out
    assert "0W/3L" in out


def test_show_recent_trades_derives_rr_from_prices_for_legacy_rows(capsys):
    journal = [
        {
            "pair": "EUR_JPY",
            "direction": "SHORT",
            "entry_price": 183.906,
            "sl_price": 184.206,
            "tp_price": 183.306,
            "outcome": {
                "realized_pl": 74.91,
                "pnl_pips": 20.0,
                "trade_won": True,
                "close_time": "2026-04-02T06:02:22.469573602Z",
            },
        }
    ]

    buddy_status.show_recent_trades(journal)
    out = capsys.readouterr().out

    assert "2.0:1" in out
    assert "N/A" not in out


def test_show_risk_scaler_prefers_journal_history_over_stale_state(tmp_path, capsys, monkeypatch):
    root = Path(tmp_path)
    (root / "trained_data").mkdir()
    (root / ".claude").mkdir()

    (root / "trained_data" / "adaptive_sizer_state.json").write_text(
        """
        {
          "config": {
            "kelly_window": 50,
            "drawdown_floor": 0.3,
            "max_acceptable_drawdown": 0.15
          },
          "trade_history": [
            [100.0, true],
            [-50.0, false]
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (root / "trained_data" / "trade_journal_rl.json").write_text(
        """
        [
          {"pair": "EUR_USD", "timestamp": "2026-04-01T00:00:00Z", "outcome": {"realized_pl": 10.0, "trade_won": true, "close_time": "2026-04-01T01:00:00Z"}},
          {"pair": "GBP_USD", "timestamp": "2026-04-01T02:00:00Z", "outcome": {"realized_pl": -5.0, "trade_won": false, "close_time": "2026-04-01T03:00:00Z"}},
          {"pair": "USD_CHF", "timestamp": "2026-04-01T04:00:00Z", "outcome": {"realized_pl": 7.5, "trade_won": true, "close_time": "2026-04-01T05:00:00Z"}}
        ]
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(buddy_status, "_ROOT", root)

    buddy_status.show_risk_scaler()
    out = capsys.readouterr().out

    assert "rehydrated from journal" in out
    assert "2/3 wins" in out
    assert "1.30x position sizing" in out


def test_build_status_diagnostics_reads_scan_and_funnel_state(tmp_path, monkeypatch):
    root = Path(tmp_path)
    (root / "trained_data").mkdir()
    (root / ".claude").mkdir()

    (root / "trained_data" / "scan_diagnostics_report.json").write_text(
        '{"diagnostic_summary":"OK","gate_pass_rate_pct":33.3,"pairs_with_direction_pct":66.7,"avg_confidence":0.54,"stalled":false}',
        encoding="utf-8",
    )
    (root / "trained_data" / "scan_diagnostics_state.json").write_text(
        '{"scans_since_last_trade":7,"total_scans":137}',
        encoding="utf-8",
    )
    (root / "trained_data" / "signal_funnel_state.jsonl").write_text(
        '\n'.join([
            '{"passed_confidence_pct":40.0,"passed_momentum_pct":50.0,"passed_risk_pct":60.0,"tradeable_pct":20.0,"dominant_kill_stage":"confidence"}',
            '{"passed_confidence_pct":20.0,"passed_momentum_pct":30.0,"passed_risk_pct":40.0,"tradeable_pct":10.0,"dominant_kill_stage":"momentum"}',
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(buddy_status, "_ROOT", root)

    diag = buddy_status.build_status_diagnostics()

    assert diag["scan"]["gate_pass_rate_pct"] == 33.3
    assert diag["scan"]["scans_since_last_trade"] == 7
    assert diag["funnel"]["window"] == 2
    assert diag["funnel"]["dominant_kill_stage"] in {"confidence", "momentum"}


def test_pair_accuracy_snapshot_prefers_journal_when_accuracy_file_is_bloated(tmp_path, monkeypatch):
    root = Path(tmp_path)
    (root / "trained_data").mkdir()
    (root / ".claude").mkdir()

    (root / "trained_data" / "pair_accuracy.json").write_text(
        """
        {
          "AUD_USD": {"accuracy": 0.0, "total": 18, "wins": 0, "rolling_total": 18, "rolling_wins": 0}
        }
        """.strip(),
        encoding="utf-8",
    )
    (root / "trained_data" / "trade_journal_rl.json").write_text(
        """
        [
          {"pair": "AUD_USD", "direction": "LONG", "confidence": 0.52, "outcome": {"trade_won": false, "close_time": "2026-04-02T01:00:00Z"}},
          {"pair": "AUD_USD", "direction": "LONG", "confidence": 0.48, "outcome": {"trade_won": false, "close_time": "2026-04-02T02:00:00Z"}}
        ]
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(buddy_status, "_ROOT", root)

    snap = buddy_status._build_pair_accuracy_snapshot()

    assert snap["source"] == "journal"
    assert snap["worst"][0]["pair"] == "AUD_USD"
    assert snap["worst"][0]["rolling_total"] == 2
