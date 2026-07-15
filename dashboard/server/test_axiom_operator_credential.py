"""No-mock tests for the off-server operator credential tool."""

from __future__ import annotations

import json

from dashboard.server.axiom_evidence_control import (
    NonceLedger,
    OperatorAuthorizer,
    OperatorCredential,
    load_operator_trust,
)
from scripts.axiom_operator_credential import main
from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import sha256_bytes


def test_operator_tool_initializes_trust_and_issues_verifiable_run_credential(tmp_path, capsys):
    private = tmp_path / "operator.pem"
    trust = tmp_path / "trust.json"
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"partitions": {"EUR_USD": "EUR_USD.csv"}}), encoding="utf-8")

    assert main(["--private-key", str(private), "--trust-anchor", str(trust), "init"]) == 0
    capsys.readouterr()
    assert private.stat().st_mode & 0o777 == 0o600
    assert main([
        "--private-key", str(private),
        "--trust-anchor", str(trust),
        "issue-run", "--dataset", str(dataset), "--ttl", "60",
    ]) == 0
    credential = OperatorCredential.from_mapping(json.loads(capsys.readouterr().out))
    request = {
        "dataset_sha256": sha256_bytes(dataset.read_bytes()),
        "lane": "risk_target",
        "program": "risk_target_baseline",
    }
    subject = sha256_bytes(canonical_bytes(request))

    authorizer = OperatorAuthorizer(load_operator_trust(trust), NonceLedger(tmp_path / "nonces.jsonl"))
    authorizer.authorize(credential, action="run", subject_digest=subject)
    assert (tmp_path / "nonces.jsonl").is_file()


def test_operator_tool_refuses_key_that_does_not_match_trust_anchor(tmp_path, capsys):
    first_private = tmp_path / "first.pem"
    first_trust = tmp_path / "first-trust.json"
    second_private = tmp_path / "second.pem"
    second_trust = tmp_path / "second-trust.json"
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"partitions": {"EUR_USD": "EUR_USD.csv"}}', encoding="utf-8")
    assert main(["--private-key", str(first_private), "--trust-anchor", str(first_trust), "init"]) == 0
    capsys.readouterr()
    assert main(["--private-key", str(second_private), "--trust-anchor", str(second_trust), "init"]) == 0
    capsys.readouterr()

    assert main([
        "--private-key", str(second_private),
        "--trust-anchor", str(first_trust),
        "issue-run", "--dataset", str(dataset),
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does not match" in captured.err


def test_operator_tool_refuses_insecure_private_key_permissions(tmp_path, capsys):
    private = tmp_path / "operator.pem"
    trust = tmp_path / "trust.json"
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"partitions": {"EUR_USD": "EUR_USD.csv"}}', encoding="utf-8")
    assert main(["--private-key", str(private), "--trust-anchor", str(trust), "init"]) == 0
    capsys.readouterr()
    private.chmod(0o644)

    assert main([
        "--private-key", str(private),
        "--trust-anchor", str(trust),
        "issue-run", "--dataset", str(dataset),
    ]) == 2
    assert "must not be accessible" in capsys.readouterr().err
