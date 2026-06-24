"""Pillar 2 — tamper-evident per-cycle ledger + recall-triggers.

The harvester records what it learns each cycle (the gate decision + re-derived
facts) to an append-only, hash-chained JSONL ledger so the record is objective
and tamper-evident (any edit/insert/delete breaks the chain), and `recall_triggers`
aggregates recurring conditions (drawdown halts, stale abstains, ship-gate blocks,
executions) so the loop sharpens over cycles.

No mocks: real disk via `tmp_path`, real hashing.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.equity.cycle_ledger import (
    GENESIS_HASH,
    append_cycle,
    read_ledger,
    recall_triggers,
    target_weight_hash,
    verify_chain,
)


def _append(ledger: Path, **kw):
    base = dict(
        ledger_path=ledger,
        asof="2026-06-24T00:00:00Z",
        decision="continue",
        reasons=(),
        actionable=True,
        universe_hash="deadbeef" * 8,
        n_orders=4,
        target_weights={"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25},
    )
    base.update(kw)
    return append_cycle(**base)


def test_append_and_read_roundtrip(tmp_path: Path):
    ledger = tmp_path / "cycle_ledger.jsonl"
    r0 = _append(ledger)
    r1 = _append(ledger, decision="abstain", actionable=False, n_orders=0, reasons=("stale_data:30d>7d",))
    recs = read_ledger(ledger)
    assert len(recs) == 2
    assert recs[0].seq == 0 and recs[1].seq == 1
    assert recs[0].prev_hash == GENESIS_HASH
    assert recs[1].prev_hash == r0.record_hash  # chained
    assert recs[1].decision == "abstain"
    assert r1.record_hash == recs[1].record_hash


def test_verify_chain_ok_on_clean_ledger(tmp_path: Path):
    ledger = tmp_path / "cycle_ledger.jsonl"
    for _ in range(3):
        _append(ledger)
    ok, broken = verify_chain(ledger)
    assert ok is True
    assert broken is None


def test_hash_chain_detects_tampering(tmp_path: Path):
    ledger = tmp_path / "cycle_ledger.jsonl"
    _append(ledger)
    _append(ledger, decision="continue")
    _append(ledger)
    # Tamper: rewrite line 1's decision in place, leaving its stored hash intact.
    lines = ledger.read_text().splitlines()
    obj = json.loads(lines[1])
    obj["decision"] = "halt"  # forge the record without recomputing the hash
    lines[1] = json.dumps(obj)
    ledger.write_text("\n".join(lines) + "\n")
    ok, broken = verify_chain(ledger)
    assert ok is False
    assert broken == 1


def test_target_weight_hash_is_deterministic_and_order_invariant(tmp_path: Path):
    a = target_weight_hash({"AAA": 0.5, "BBB": 0.5})
    b = target_weight_hash({"BBB": 0.5, "AAA": 0.5})  # different insertion order
    c = target_weight_hash({"AAA": 0.6, "BBB": 0.4})
    assert a == b  # order-invariant
    assert a != c  # value-sensitive
    assert len(a) == 64  # sha256 hex


def test_recall_triggers_aggregates_recurring_conditions(tmp_path: Path):
    ledger = tmp_path / "cycle_ledger.jsonl"
    _append(ledger, decision="continue", actionable=True)
    _append(ledger, decision="continue", actionable=True)
    _append(ledger, decision="abstain", actionable=False, reasons=("stale_data:30d>7d",))
    _append(ledger, decision="halt", actionable=False, reasons=("drawdown_breach:0.25>=dd_hard:0.20",))
    _append(ledger, decision="no_act", actionable=False, reasons=("ship_gate_fail",))
    rollup = recall_triggers(ledger)
    assert rollup["total_cycles"] == 5
    assert rollup["triggers"]["executed"] == 2
    assert rollup["triggers"]["stale_abstain"] == 1
    assert rollup["triggers"]["drawdown_halt"] == 1
    assert rollup["triggers"]["ship_gate_block"] == 1


def test_read_ledger_missing_file_is_empty(tmp_path: Path):
    assert read_ledger(tmp_path / "nope.jsonl") == []
    ok, broken = verify_chain(tmp_path / "nope.jsonl")
    assert ok is True and broken is None
