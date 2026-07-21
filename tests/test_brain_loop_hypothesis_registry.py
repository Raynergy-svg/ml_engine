"""No-mock tests for src.brain_loop.hypothesis_registry.

Real disk (tmp_path), real hash-chain verification — mirrors the no-mock
convention used by src/equity/cycle_ledger.py's own test suite.
"""
from __future__ import annotations

import json
import threading

import pytest

from src.brain_loop.hypothesis_registry import (
    FrozenParamsViolation,
    HypothesisNotRegistered,
    attach_result,
    read_ledger,
    record_gate_verdict,
    register_hypothesis,
    verify_chain,
)


def test_register_hypothesis_writes_a_record(tmp_path):
    ledger_path = tmp_path / "hypotheses.jsonl"
    rec = register_hypothesis(
        ledger_path,
        hypothesis_id="h-001",
        name="crypto-basis-carry",
        mechanism="cross-exchange futures basis, not tested before per L-022",
        novelty_justification="uses cross-exchange basis, not tried in the closed campaign",
        params={"lookback": 30, "threshold": 0.02},
    )
    assert rec["kind"] == "register"
    assert rec["hypothesis_id"] == "h-001"
    assert rec["status"] == "REGISTERED"

    records = read_ledger(ledger_path)
    assert len(records) == 1
    assert records[0]["hypothesis_id"] == "h-001"


def test_frozen_before_results_same_params_succeeds(tmp_path):
    ledger_path = tmp_path / "hypotheses.jsonl"
    register_hypothesis(
        ledger_path,
        hypothesis_id="h-001",
        name="x",
        mechanism="m",
        novelty_justification="n",
        params={"a": 1, "b": 2},
    )
    rec = attach_result(
        ledger_path,
        "h-001",
        params_used={"a": 1, "b": 2},
        result_summary={"sharpe": 0.4, "max_dd": 0.15},
    )
    assert rec["kind"] == "result"
    assert rec["hypothesis_id"] == "h-001"


def test_frozen_before_results_refuses_mutated_params(tmp_path):
    """The load-bearing guarantee: a result cannot attach to a hypothesis whose
    params were changed after registration (no post-hoc reshaping to fit the
    result)."""
    ledger_path = tmp_path / "hypotheses.jsonl"
    register_hypothesis(
        ledger_path,
        hypothesis_id="h-001",
        name="x",
        mechanism="m",
        novelty_justification="n",
        params={"a": 1, "b": 2},
    )
    with pytest.raises(FrozenParamsViolation):
        attach_result(
            ledger_path,
            "h-001",
            params_used={"a": 1, "b": 999},  # mutated after the fact
            result_summary={"sharpe": 5.0},
        )


def test_attach_result_refuses_unregistered_hypothesis(tmp_path):
    ledger_path = tmp_path / "hypotheses.jsonl"
    with pytest.raises(HypothesisNotRegistered):
        attach_result(
            ledger_path,
            "never-registered",
            params_used={"a": 1},
            result_summary={"sharpe": 1.0},
        )


def test_record_gate_verdict_refuses_unregistered_hypothesis(tmp_path):
    ledger_path = tmp_path / "hypotheses.jsonl"
    with pytest.raises(HypothesisNotRegistered):
        record_gate_verdict(ledger_path, "never-registered", decision="fail", reasons=("x",))


def test_hash_chain_is_tamper_evident(tmp_path):
    ledger_path = tmp_path / "hypotheses.jsonl"
    register_hypothesis(
        ledger_path, hypothesis_id="h-001", name="x", mechanism="m",
        novelty_justification="n", params={"a": 1},
    )
    attach_result(
        ledger_path, "h-001", params_used={"a": 1}, result_summary={"sharpe": 0.3},
    )
    record_gate_verdict(ledger_path, "h-001", decision="fail", reasons=("dsr_below_threshold",))

    ok, broken = verify_chain(ledger_path)
    assert ok is True
    assert broken is None

    # Tamper with the middle record's payload without recomputing the chain.
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["result_summary"]["sharpe"] = 999.0
    lines[1] = json.dumps(tampered, sort_keys=True)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, broken = verify_chain(ledger_path)
    assert ok is False
    assert broken == 1


def test_verify_chain_on_missing_file_is_vacuously_ok(tmp_path):
    ok, broken = verify_chain(tmp_path / "does_not_exist.jsonl")
    assert ok is True
    assert broken is None


def test_read_ledger_missing_file_returns_empty_list(tmp_path):
    assert read_ledger(tmp_path / "nope.jsonl") == []


def test_concurrent_appends_do_not_lose_records_or_break_chain(tmp_path):
    """Real threads, real disk, real flock — proves the lost-update race (two
    writers both reading the same tail seq/prev_hash, second write clobbering
    the first) is closed, not just that a single-writer path still works."""
    ledger_path = tmp_path / "hypotheses.jsonl"
    n_writers = 12
    errors = []

    def _write(i: int) -> None:
        try:
            register_hypothesis(
                ledger_path, hypothesis_id=f"h-{i}", name=f"concurrent-{i}",
                mechanism="m", novelty_justification="n", params={"i": i},
            )
        except Exception as exc:  # noqa: BLE001 - collected below, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected exceptions from concurrent writers: {errors}"
    records = read_ledger(ledger_path)
    assert len(records) == n_writers, "a concurrent write was silently dropped"
    assert {r["hypothesis_id"] for r in records} == {f"h-{i}" for i in range(n_writers)}
    assert len({r["seq"] for r in records}) == n_writers, "two records share a seq (lost update)"

    ok, broken = verify_chain(ledger_path)
    assert ok is True
    assert broken is None
