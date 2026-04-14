import json
from pathlib import Path

import requests

from scripts.validate_deployment import DeploymentValidator


def test_normalize_trade_entry_derives_sl_tp_pips():
    validator = DeploymentValidator()

    trade = {
        "pair": "EUR_USD",
        "entry_price": 1.1000,
        "sl_price": 1.0985,
        "tp_price": 1.1030,
    }

    normalized = validator._normalize_trade_entry(trade)

    assert normalized["sl_pips"] == 15.0
    assert normalized["tp_pips"] == 30.0


def test_blocked_pairs_snapshot_treats_dynamic_gate_as_blocked(monkeypatch):
    validator = DeploymentValidator()

    class FakeGate:
        min_trades = 5

        def __init__(self):
            self._data = {
                "EUR_USD": {
                    "total": 5,
                    "wins": 0,
                    "rolling_total": 5,
                    "rolling_wins": 0,
                    "accuracy": 0.0,
                }
            }

        def get_blocked_pairs(self):
            return ["EUR_USD"]

        def get_pair_stats(self, pair):
            return {
                "pair": pair,
                "rolling_total": 5,
                "alltime_accuracy": 0.0,
            }

        def check_pair(self, pair):
            return (False, 0.0, "rolling accuracy 0.0% < 55.0%")

    import src.scanner.automation.accuracy_gate as accuracy_gate_module

    monkeypatch.setattr(accuracy_gate_module, "AccuracyGate", FakeGate)

    configured, dynamic, unresolved = validator._blocked_pairs_snapshot([])

    assert configured == []
    assert dynamic == ["EUR_USD"]
    assert unresolved == []


def test_check_trade_journal_accepts_derivable_pip_fields(tmp_path: Path):
    validator = DeploymentValidator()
    validator.project_root = tmp_path
    validator.results = []

    trained_data = tmp_path / "trained_data"
    trained_data.mkdir(parents=True)
    (trained_data / "trade_journal_rl.json").write_text(
        """
[
  {
    "pair": "EUR_USD",
    "direction": "LONG",
    "timestamp": "2026-04-02T00:00:00Z",
    "entry_price": 1.1000,
    "sl_price": 1.0985,
    "tp_price": 1.1030
  }
]
""".strip()
    )

    validator._check_trade_journal()

    required_fields_result = next(
        result for result in validator.results if result.name == "Trade journal: Required fields"
    )
    assert required_fields_result.status == "PASS"


def test_auto_fix_trade_journal_archives_closed_low_rr(tmp_path: Path):
    validator = DeploymentValidator()
    validator.project_root = tmp_path

    trained_data = tmp_path / "trained_data"
    trained_data.mkdir(parents=True)
    journal_path = trained_data / "trade_journal_rl.json"
    journal_path.write_text(
        json.dumps(
            [
                {
                    "trade_id": "closed-low",
                    "pair": "EUR_USD",
                    "entry_price": 1.1000,
                    "sl_pips": 20.0,
                    "tp_pips": 18.0,
                    "outcome": {"trade_won": False},
                },
                {
                    "trade_id": "open-low",
                    "pair": "EUR_USD",
                    "entry_price": 1.1000,
                    "sl_pips": 20.0,
                    "tp_pips": 18.0,
                },
                {
                    "trade_id": "closed-good",
                    "pair": "EUR_USD",
                    "entry_price": 1.1000,
                    "sl_pips": 20.0,
                    "tp_pips": 24.0,
                    "outcome": {"trade_won": True},
                },
            ],
            indent=2,
        )
    )

    fixes = validator._auto_fix_trade_journal()

    assert fixes == 1
    repaired = json.loads(journal_path.read_text())
    repaired_ids = {row["trade_id"] for row in repaired}
    assert "closed-low" not in repaired_ids
    assert "open-low" in repaired_ids
    assert "closed-good" in repaired_ids

    archive_path = trained_data / "trade_journal_low_rr_archive.json"
    archived = json.loads(archive_path.read_text())
    archived_ids = {row["trade_id"] for row in archived}
    assert "closed-low" in archived_ids


def test_check_oanda_connectivity_uses_sandbox_rest_host(monkeypatch):
    validator = DeploymentValidator()
    validator.results = []

    monkeypatch.setenv("OANDA_API_TOKEN", "token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "acct-123")
    monkeypatch.setenv("OANDA_ENVIRONMENT", "sandbox")

    captured = {}

    class _Resp:
        status_code = 200
        text = '{"account":{"balance":"1000.00"}}'

        @staticmethod
        def json():
            return {"account": {"balance": "1000.00"}}

    def _fake_get(url, headers=None, timeout=0):
        captured["url"] = url
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(requests, "get", _fake_get)

    validator._check_oanda_connectivity()

    assert captured["url"] == "https://api-sandbox.oanda.com/v3/accounts/acct-123/summary"
    result = next(r for r in validator.results if r.name == "OANDA: Connectivity")
    assert result.status == "PASS"


def test_check_oanda_connectivity_classifies_dns_failure(monkeypatch):
    validator = DeploymentValidator()
    validator.results = []

    monkeypatch.setenv("OANDA_API_TOKEN", "token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "acct-123")
    monkeypatch.setenv("OANDA_ENVIRONMENT", "practice")

    def _raise_dns_error(*args, **kwargs):
        raise requests.ConnectionError("Name or service not known")

    monkeypatch.setattr(requests, "get", _raise_dns_error)

    validator._check_oanda_connectivity()

    result = next(r for r in validator.results if r.name == "OANDA: Connectivity")
    assert result.status == "WARN"
    assert "DNS resolution failed" in result.message


def test_check_oanda_connectivity_classifies_tls_failure(monkeypatch):
    validator = DeploymentValidator()
    validator.results = []

    monkeypatch.setenv("OANDA_API_TOKEN", "token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "acct-123")
    monkeypatch.setenv("OANDA_ENVIRONMENT", "practice")

    def _raise_ssl_error(*args, **kwargs):
        raise requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(requests, "get", _raise_ssl_error)

    validator._check_oanda_connectivity()

    result = next(r for r in validator.results if r.name == "OANDA: Connectivity")
    assert result.status == "WARN"
    assert "TLS/SSL handshake failed" in result.message


def test_check_oanda_connectivity_404_account_hint(monkeypatch):
    validator = DeploymentValidator()
    validator.results = []

    monkeypatch.setenv("OANDA_API_TOKEN", "token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "acct-404")
    monkeypatch.setenv("OANDA_ENVIRONMENT", "practice")

    class _Resp:
        status_code = 404
        text = "not found"
        headers = {}

        @staticmethod
        def json():
            return {}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Resp())

    validator._check_oanda_connectivity()

    result = next(r for r in validator.results if r.name == "OANDA: Connectivity")
    assert result.status == "FAIL"
    assert "Not found (404)" in result.message
