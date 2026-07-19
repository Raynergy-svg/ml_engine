"""2026-07-19 re-review items 1+2: ONE canonical evidence view.

Covers: legacy pre-engine row exclusion (item 1), exact-duplicate collapse
vs fail-closed conflicts vs explicit supersession/voiding (item 2) in BOTH
canonical resolvers (forward_ledger.active_observations and the hedge
lane's dedupe_ledger_rows), and the audited supersession repair script.

No mocks: real dict rows, real tmp_path ledgers, the real committed hedge
ledger read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from repair_hedge_ledger_supersession import repair  # noqa: E402
from src.evidence.forward_ledger import (  # noqa: E402
    EvidenceConflictError,
    active_observations,
    cadence_counts,
    legacy_rows,
    read_rows,
)
from src.hedge.hedged_shadow_lane import (  # noqa: E402
    RAW_VS_HEDGED_LEDGER_PATH,
    _material_evidence,
    dedupe_ledger_rows,
)


def _fwd(asof, ret, period, snap, **extra):
    row = {"kind": "forward", "asof_date": asof, "today_net_return": ret,
           "evidence_period_id": period, "snapshot_version_id": snap}
    row.update(extra)
    return row


# ── item 1: legacy pre-engine rows are NOT forward evidence ────────────────

def test_legacy_rows_excluded_from_canonical_view_and_reported():
    rows = [
        {"asof_date": "2026-06-17", "today_net_return": 0.010242},  # legacy
        {"kind": "activation", "asof_date": "2026-06-24",
         "today_net_return": None},
        _fwd("2026-07-01", 0.005, "s:a->b", "v1"),
    ]
    assert [r["asof_date"] for r in legacy_rows(rows)] == ["2026-06-17"]
    canon = active_observations(rows)
    assert [r["asof_date"] for r in canon] == ["2026-07-01"]
    cad = cadence_counts(rows)
    assert cad["n_return_bars"] == 1
    assert cad["n_legacy_rows_excluded"] == 1


# ── item 2: forward-ledger canonical resolution ────────────────────────────

def test_exact_duplicate_forward_rows_collapse_to_one():
    dup = _fwd("2026-07-01", 0.005, "s:a->b", "v1")
    canon = active_observations([dup, dict(dup)])
    assert len(canon) == 1 and canon[0]["today_net_return"] == 0.005


def test_conflicting_forward_rows_fail_closed():
    with pytest.raises(EvidenceConflictError):
        active_observations([
            _fwd("2026-07-01", 0.005, "s:a->b", "v1"),
            _fwd("2026-07-01", 0.007, "s:a->b", "v2"),   # no supersedes
        ])
    # same snapshot id but a DIFFERENT return is still a conflict
    with pytest.raises(EvidenceConflictError):
        active_observations([
            _fwd("2026-07-01", 0.005, "s:a->b", "v1"),
            _fwd("2026-07-01", 0.007, "s:a->b", "v1"),
        ])


def test_explicit_supersession_replaces_never_counts_beside():
    canon = active_observations([
        _fwd("2026-07-01", 0.005, "s:a->b", "v1"),
        _fwd("2026-07-01", 0.002, "s:a->b", "v2", supersedes="v1"),
    ])
    assert len(canon) == 1 and canon[0]["today_net_return"] == 0.002


# ── item 2: hedge twin-lane canonical resolution ───────────────────────────

def _hrow(strategy, asof, net, **extra):
    row = {"strategy": strategy, "asof_date": asof,
           "cycle_ts": extra.pop("cycle_ts", "t0"),
           "raw": {"net_return": net, "gross_return": net, "cost": 0.0},
           "hedged": {"net_return": net, "gross_return": net, "cost": None},
           "hedge": {"decision": "SHADOW_ONLY"},
           "exposure": {"position_count": 1},
           "paper_only": True, "runtime_allowed": False,
           "human_review_required": True}
    row.update(extra)
    return row


def test_hedge_exact_duplicate_collapses_even_across_provenance():
    a = _hrow("eq", "2026-07-01", 0.004, cycle_ts="t0",
              weights={"AAA": 1.0}, notional=100.0,
              meta={"source_path": "/machine/one"})
    b = _hrow("eq", "2026-07-01", 0.004, cycle_ts="t9",
              weights={"AAA": 1.0}, notional=100.0,
              meta={"source_path": "/machine/two"})
    # provenance (cycle_ts, meta) differs; identity + material evidence match
    assert _material_evidence(a) == _material_evidence(b)
    out = dedupe_ledger_rows([a, b])
    assert len(out) == 1


def test_hedge_conflicting_rows_fail_closed_without_supersession():
    a = _hrow("eq", "2026-07-01", 0.004, weights={"AAA": 1.0}, notional=100.0)
    b = _hrow("eq", "2026-07-01", 0.009, weights={"AAA": 1.0}, notional=100.0)
    with pytest.raises(EvidenceConflictError):
        dedupe_ledger_rows([a, b])
    # explicit supersession resolves it: LAST revision is active
    b2 = dict(b)
    b2["supersedes"] = "some-prior-identity"
    out = dedupe_ledger_rows([a, b2])
    assert len(out) == 1 and out[0]["raw"]["net_return"] == 0.009


def test_hedge_voided_rows_are_excluded_never_conflict():
    a = _hrow("tb", "2026-06-24", 0.00126)
    degraded = _hrow("tb", "2026-06-24", None, cycle_ts="t9",
                     voided=True, voided_reason="unresolved re-capture")
    out = dedupe_ledger_rows([a, degraded])
    assert len(out) == 1 and out[0]["raw"]["net_return"] == 0.00126


def test_hedge_book_change_same_period_needs_supersession_not_two_rows():
    # a book-hash change on the same (strategy, return-period) must never
    # count as an additional observation
    a = _hrow("eq", "2026-07-01", 0.004, weights={"AAA": 1.0}, notional=100.0)
    b = _hrow("eq", "2026-07-01", 0.004, weights={"BBB": 1.0}, notional=100.0)
    with pytest.raises(EvidenceConflictError):
        dedupe_ledger_rows([a, b])


# ── the audited repair script ──────────────────────────────────────────────

def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")


def test_repair_stamps_supersession_on_identical_material(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    a = _hrow("eq", "2026-07-01", 0.004, cycle_ts="t0")
    b = _hrow("eq", "2026-07-01", 0.004, cycle_ts="t9",
              weights={"AAA": 1.0}, notional=100.0)
    _write(ledger, [a, b])
    summary = repair(ledger)
    assert summary["n_stamped"] == 1
    assert summary["stamped"][0]["action"] == "supersedes"
    rows = read_rows(ledger)
    assert rows[1]["supersedes"]                    # never falsy
    assert dedupe_ledger_rows(rows)[0]["cycle_ts"] == "t9"
    # idempotent: second run is a no-op
    assert repair(ledger)["n_stamped"] == 0


def test_repair_voids_degraded_recapture(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    resolved = _hrow("tb", "2026-06-24", 0.00126, cycle_ts="t0")
    degraded = _hrow("tb", "2026-06-24", None, cycle_ts="t9")
    degraded["raw"]["gross_return"] = None
    degraded["raw"]["cost"] = None
    degraded["hedged"] = {"net_return": None, "gross_return": None,
                          "cost": None}
    _write(ledger, [resolved, degraded])
    summary = repair(ledger)
    assert summary["stamped"][0]["action"] == "voided"
    rows = read_rows(ledger)
    active = dedupe_ledger_rows(rows)
    # the RESOLVED prior stays active; the degraded row never overwrites it
    assert len(active) == 1 and active[0]["raw"]["net_return"] == 0.00126


def test_repair_aborts_on_genuine_material_conflict(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _write(ledger, [_hrow("eq", "2026-07-01", 0.004),
                    _hrow("eq", "2026-07-01", -0.009, cycle_ts="t9")])
    before = ledger.read_bytes()
    with pytest.raises(SystemExit) as exc:
        repair(ledger)
    assert exc.value.code == 2
    assert ledger.read_bytes() == before            # nothing rewritten


# ── the real committed ledger resolves cleanly after the repair ────────────

def test_committed_hedge_ledger_resolves_one_row_per_period():
    rows = read_rows(RAW_VS_HEDGED_LEDGER_PATH)
    active = dedupe_ledger_rows(rows)               # must not raise
    periods = [(r["strategy"], r["asof_date"]) for r in active]
    assert len(periods) == len(set(periods))
    assert len(active) < len(rows)                  # history retained in file
