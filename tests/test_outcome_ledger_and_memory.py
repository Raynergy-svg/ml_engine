"""Outcome ledger + performance memory — durability, idempotency, honest gaps.

No mocks. Real dataclasses, real disk via tmp_path, real concurrent processes
for the locking test (per the No-Mock Rule in .claude/rules/improvement.md).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.learning.outcome_attribution import DecisionRecord, OutcomeRecord, attribute
from src.learning.outcome_ledger import (
    append_rows,
    build_row,
    existing_keys,
    load_ledger,
    outcome_key,
)
from src.learning.performance_memory import MIN_N_FOR_ACTIONABLE, build_memory

REPO_ROOT = Path(__file__).resolve().parents[1]


def _outcome(trade_id="t1", pl=10.0, **kw) -> OutcomeRecord:
    base = dict(
        trade_id=trade_id, instrument="EUR_USD",
        entry_time="2026-07-01T00:00:00Z", exit_time="2026-07-01T01:00:00Z",
        entry_price=1.1000, exit_price=1.1100, units=1000.0,
        realized_pl_home=pl, conversion_factor=1.0, lane="oanda_fx",
    )
    base.update(kw)
    return OutcomeRecord(**base)


def _row(outcome: OutcomeRecord, decision=None):
    return build_row(outcome, attribute(outcome, decision))


# ---------------------------------------------------------------------------
# Ledger: idempotency and durability
# ---------------------------------------------------------------------------


def test_append_is_idempotent_across_repeated_runs(tmp_path):
    """Re-running attribution over the whole journal must not duplicate rows."""
    path = tmp_path / "ledger.jsonl"
    rows = [_row(_outcome("t1")), _row(_outcome("t2", pl=-5.0))]

    first = append_rows(rows, path)
    second = append_rows(rows, path)

    assert first == {"appended": 2, "skipped_duplicate": 0, "skipped_keyless": 0}
    assert second == {"appended": 0, "skipped_duplicate": 2, "skipped_keyless": 0}
    assert len(load_ledger(path)) == 2


def test_material_change_is_a_different_row_but_cost_restatement_is_not(tmp_path):
    """Dedupe keys on materially-identifying fields only."""
    path = tmp_path / "ledger.jsonl"
    append_rows([_row(_outcome("t1"))], path)

    # Same event, broker restated a cost figure -> NOT a new close.
    append_rows([_row(_outcome("t1", financing_home=-0.25))], path)
    assert len(load_ledger(path)) == 1

    # Genuinely different exit -> IS a new close.
    append_rows([_row(_outcome("t1", exit_time="2026-07-01T02:00:00Z"))], path)
    assert len(load_ledger(path)) == 2


def test_outcome_key_is_stable_across_equal_records():
    assert outcome_key(_outcome("t1")) == outcome_key(_outcome("t1"))
    assert outcome_key(_outcome("t1")) != outcome_key(_outcome("t2"))


def test_row_without_key_is_refused_not_silently_written(tmp_path):
    path = tmp_path / "ledger.jsonl"
    bad = _row(_outcome("t1"))
    del bad["outcome_key"]

    stats = append_rows([bad], path)

    assert stats["skipped_keyless"] == 1
    assert stats["appended"] == 0


def test_malformed_line_does_not_abort_the_read(tmp_path):
    """A torn append must not make the whole ledger unreadable."""
    path = tmp_path / "ledger.jsonl"
    append_rows([_row(_outcome("t1"))], path)
    with open(path, "a") as fh:
        fh.write("{not json\n")
    append_rows([_row(_outcome("t2"))], path)

    rows = load_ledger(path)

    assert len(rows) == 2
    assert {r["outcome"]["trade_id"] for r in rows} == {"t1", "t2"}


def test_existing_keys_reflects_disk(tmp_path):
    path = tmp_path / "ledger.jsonl"
    o = _outcome("t1")
    append_rows([_row(o)], path)
    assert existing_keys(path) == {outcome_key(o)}


def test_missing_ledger_reads_as_empty_not_error(tmp_path):
    assert load_ledger(tmp_path / "nope.jsonl") == []


def test_concurrent_writers_do_not_tear_or_duplicate(tmp_path):
    """Two real processes appending the same rows at once.

    The dedupe check happens inside the lock, so exactly one writer wins each
    key and no line is torn.
    """
    path = tmp_path / "ledger.jsonl"
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from src.learning.outcome_attribution import OutcomeRecord, attribute
        from src.learning.outcome_ledger import append_rows, build_row
        rows = []
        for i in range(40):
            o = OutcomeRecord(
                trade_id=f"t{{i}}", instrument="EUR_USD",
                entry_time="2026-07-01T00:00:00Z", exit_time="2026-07-01T01:00:00Z",
                entry_price=1.10, exit_price=1.11, units=1000.0,
                realized_pl_home=10.0, conversion_factor=1.0,
            )
            rows.append(build_row(o, attribute(o)))
        append_rows(rows, {str(path)!r})
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script]) for _ in range(4)]
    for p in procs:
        assert p.wait(timeout=120) == 0

    raw = path.read_text().splitlines()
    rows = load_ledger(path)
    assert len(rows) == 40, f"expected 40 unique rows, got {len(rows)}"
    assert len(raw) == 40, "torn or duplicated lines on disk"
    for line in raw:
        json.loads(line)  # every line individually parseable


# ---------------------------------------------------------------------------
# Performance memory: the alarm and the honest gaps
# ---------------------------------------------------------------------------


def test_residual_alarm_fires_when_attribution_disagrees_with_broker():
    """A wrong model must not quietly feed confident statistics downstream."""
    good = _row(_outcome("t1"))
    bad = _row(_outcome("t2", realized_pl_home=999.0))  # arithmetic won't close

    memory = build_memory([good, bad])

    assert memory["residual_health"]["status"] == "ALARM"
    assert memory["residual_health"]["unreconciled_count"] == 1
    assert "t2" in memory["residual_health"]["unreconciled_trade_ids"]
    assert "MODEL is wrong" in memory["residual_health"]["interpretation"]


def test_residual_health_ok_on_clean_rows():
    # A losing row must move the EXIT PRICE, not just the reported P&L --
    # otherwise the decomposition correctly refuses to reconcile it.
    loser = _outcome("t2", exit_price=1.0900, pl=-10.0)
    memory = build_memory([_row(_outcome("t1")), _row(loser)])
    assert memory["residual_health"]["status"] == "OK"


def test_unavailable_components_are_counted_not_averaged_as_zero():
    """The core honesty rule: absent != zero."""
    memory = build_memory([_row(_outcome("t1")), _row(_outcome("t2"))])
    cov = memory["attribution_coverage"]

    assert cov["rows_without_decision_record"] == 2
    assert cov["unavailable_component_counts"]["sizing_pl_home"] == 2
    assert "not zero" in cov["note"]


def test_decision_record_rows_report_full_coverage():
    decision = DecisionRecord(
        decision_id="d1", asof="2026-07-01T00:00:00Z", lane="oanda_fx",
        instrument="EUR_USD", intended_units=1000.0, signal_source="s",
        pre_degross_units=2000.0, degross_factor=0.5,
    )
    memory = build_memory([_row(_outcome("t1", decision_id="d1"), decision)])

    assert memory["attribution_coverage"]["rows_with_decision_record"] == 1
    assert memory["attribution_coverage"]["unavailable_component_counts"] == {}


def test_small_buckets_are_reported_but_marked_not_actionable():
    memory = build_memory([_row(_outcome("t1", instrument="NZD_USD", pl=2667.0))])
    bucket = memory["by_instrument"]["NZD_USD"]

    assert bucket["n"] == 1
    assert bucket["actionable"] is False


def test_bucket_becomes_actionable_at_the_promotion_threshold():
    rows = [_row(_outcome(f"t{i}")) for i in range(MIN_N_FOR_ACTIONABLE)]
    assert build_memory(rows)["by_instrument"]["EUR_USD"]["actionable"] is True


def test_undefined_ratios_are_none_not_zero():
    """No losses must not report avg_loss = 0.0 -- that reads as 'losses cost nothing'."""
    memory = build_memory([_row(_outcome("t1", pl=10.0))])
    overall = memory["overall"]

    assert overall["n_losses"] == 0
    assert overall["avg_loss_home"] is None
    assert overall["profit_factor"] is None


def test_expectancy_and_win_rate_are_arithmetically_correct():
    rows = [_row(_outcome("t1", pl=10.0)),
            _row(_outcome("t2", exit_price=1.0900, pl=-10.0)),
            _row(_outcome("t3", pl=10.0))]
    overall = build_memory(rows)["overall"]

    assert overall["n"] == 3
    assert overall["win_rate"] == pytest.approx(2 / 3)
    assert overall["total_pl_home"] == pytest.approx(10.0)
    assert overall["expectancy_home"] == pytest.approx(10.0 / 3)


def test_empty_ledger_yields_no_data_not_a_crash():
    memory = build_memory([])
    assert memory["residual_health"]["status"] == "NO_DATA"
    assert memory["overall"]["n"] == 0


def test_malformed_ledger_rows_are_skipped():
    memory = build_memory([_row(_outcome("t1")), {"garbage": True}, "not a dict"])
    assert memory["overall"]["n"] == 1


# ---------------------------------------------------------------------------
# Runner: exit codes are the contract a scheduler depends on
# ---------------------------------------------------------------------------


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_outcome_attribution.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )


# The broker journal lives under trained_data/, which is gitignored. On a clean
# checkout it is absent and the runner correctly exits 2 -- these tests exercise
# the happy path and must skip rather than fail. Caught by running the suite on
# a fresh worktree; the dirty dev tree always has the journal, so it passed
# there and would have shipped broken.
requires_journal_for_runner = pytest.mark.skipif(
    not (REPO_ROOT / "trained_data" / "oanda" / "transactions.jsonl").exists(),
    reason="broker journal (gitignored) not present in this checkout",
)


@requires_journal_for_runner
def test_runner_dry_run_writes_nothing_and_exits_zero(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    memory = tmp_path / "memory.json"

    proc = _run("--dry-run", "--quiet", "--ledger", str(ledger), "--memory", str(memory))

    assert proc.returncode == 0, proc.stderr
    assert not ledger.exists()
    assert not memory.exists()
    assert json.loads(proc.stdout)["residual_health"] == "OK"


@requires_journal_for_runner
def test_runner_is_idempotent_end_to_end(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    memory = tmp_path / "memory.json"

    first = json.loads(_run("--quiet", "--ledger", str(ledger), "--memory", str(memory)).stdout)
    second = json.loads(_run("--quiet", "--ledger", str(ledger), "--memory", str(memory)).stdout)

    assert first["ledger"]["appended"] > 0
    assert second["ledger"]["appended"] == 0
    assert second["ledger"]["skipped_duplicate"] == first["ledger"]["appended"]
    assert first["ledger_total_rows"] == second["ledger_total_rows"]
    assert memory.exists()


def test_runner_exits_2_when_journal_is_missing(tmp_path):
    proc = _run("--quiet", "--transactions", str(tmp_path / "nope.jsonl"))
    assert proc.returncode == 2


@requires_journal_for_runner
def test_runner_leaves_no_tmp_siblings(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    memory = tmp_path / "memory.json"
    _run("--quiet", "--ledger", str(ledger), "--memory", str(memory))
    assert list(tmp_path.glob("*.tmp")) == []
