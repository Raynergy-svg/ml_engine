import json

from src.scanner.automation.online_rl import OnlineWeightUpdater


def test_online_rl_reads_structured_outcomes_and_nested_agent_reasons(tmp_path):
    journal = tmp_path / "trade_journal_rl.json"
    weights = tmp_path / "agent_weights.json"
    log = tmp_path / "online_rl_updates.jsonl"

    journal.write_text(json.dumps([
        {
            "pair": "EUR_USD",
            "direction": "LONG",
            "timestamp": "2026-05-12T00:00:00+00:00",
            "outcome": {"trade_won": False, "realized_pl": -10},
            "agents": {"agent_reasons": [
                {"name": "volatility", "passed": True},
                {"name": "risk_sentinel", "passed": False},
            ]},
        },
        {
            "pair": "EUR_USD",
            "direction": "LONG",
            "timestamp": "2026-05-12T01:00:00+00:00",
            "outcome": {"trade_won": False, "realized_pl": -5},
            "agents": {"agent_reasons": [
                {"name": "volatility", "passed": True},
                {"name": "risk_sentinel", "passed": False},
            ]},
        },
        {
            "pair": "EUR_USD",
            "direction": "LONG",
            "timestamp": "2026-05-12T02:00:00+00:00",
            "outcome": {"trade_won": True, "realized_pl": 8},
            "agents": {"agent_reasons": [
                {"name": "volatility", "passed": False},
                {"name": "risk_sentinel", "passed": True},
            ]},
        },
    ]))
    weights.write_text(json.dumps({
        "_global": {"volatility": 1.0, "risk_sentinel": 1.0}
    }))

    updater = OnlineWeightUpdater(
        update_interval=1,
        lookback_trades=10,
        learning_rate=0.02,
        journal_path=journal,
        weights_path=weights,
        log_path=log,
    )

    adjustments = updater.maybe_update(scan_cycle=1)

    saved = json.loads(weights.read_text())
    assert adjustments
    assert saved["_global"]["volatility"] < 1.0
    assert saved["_global"]["risk_sentinel"] > 1.0
