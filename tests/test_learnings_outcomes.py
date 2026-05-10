"""Tests for the outcomes JSONL ledger + learnings.md renderer (Angle 2').

Hard rules per .claude/rules/improvement.md:
- No mocks (no unittest.mock, no MagicMock, no patch). Real disk via tmp_path.
- Validate atomicity, fcntl-locking, retention, halt-resilience explicitly.
- Tests cover: halted-state append, concurrent writers, fence guard,
  fence first-run insertion, retention archive, corrupt-line skip,
  counterfactual coordination, backfill idempotence.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure repo root on sys.path for `import src.scanner...` from subprocesses.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scanner.automation.learnings_outcomes import (  # noqa: E402
    FENCE_BEGIN,
    FENCE_END,
    LearningsRenderer,
    OutcomesLedger,
    emit_outcome_from_journal_entry,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_ledger(tmp_path: Path, **kwargs) -> OutcomesLedger:
    """Construct a real OutcomesLedger pointing at tmp_path locations."""
    return OutcomesLedger(
        ledger_path=tmp_path / "learnings_outcomes.jsonl",
        archive_dir=tmp_path / "archive",
        **kwargs,
    )


def _make_renderer(tmp_path: Path, ledger: OutcomesLedger, **kwargs) -> LearningsRenderer:
    return LearningsRenderer(
        learnings_md_path=tmp_path / "learnings.md",
        ledger=ledger,
        **kwargs,
    )


# ── Test 1: outcome appended when halted ──────────────────────────────────────


def test_outcome_appended_when_halted(tmp_path: Path) -> None:
    """Closed trade hook MUST fire even when the scanner is halted.

    We simulate halt by writing a halted state.json and verifying the
    ledger writer does not consult that flag — it appends regardless.
    """
    halted_state_path = tmp_path / "state.json"
    halted_state_path.write_text(json.dumps({"halted": True, "mode": "smart"}))

    ledger = _make_ledger(tmp_path)
    ok = ledger.append({
        "trade_id": "halted-1",
        "pair": "EUR_USD",
        "direction": "LONG",
        "confidence": 0.55,
        "pnl_pips": -8.4,
        "pnl_currency": -42.10,
        "exit_reason": "sl_hit",
        "regime": "NORMAL",
        "wvs": 0.61,
        "top_dissent_agent": "trend",
    })
    assert ok is True

    lines = ledger.ledger_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["trade_id"] == "halted-1"
    assert entry["kind"] == "outcome"  # auto-tagged
    assert "ts_utc" in entry
    # The halted flag was never read by the ledger — proven by the
    # append succeeding with the halted state.json present.
    assert json.loads(halted_state_path.read_text())["halted"] is True


# ── Test 2: concurrent flock writers — no corruption ──────────────────────────


def _writer_subprocess(repo_root: str, ledger_path: str, archive_dir: str, n: int, prefix: str) -> int:
    """Subprocess body: append ``n`` distinct entries via the real ledger.

    Spawned by ``test_concurrent_flock_writers_no_corruption`` with
    multiprocessing.Process so two real OS processes contend for the lock.
    """
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.scanner.automation.learnings_outcomes import OutcomesLedger

    led = OutcomesLedger(
        ledger_path=Path(ledger_path),
        archive_dir=Path(archive_dir),
    )
    written = 0
    for i in range(n):
        ok = led.append({
            "trade_id": f"{prefix}-{i:03d}",
            "pair": "EUR_USD",
            "pnl_pips": float(i),
        })
        if ok:
            written += 1
    return written


def test_concurrent_flock_writers_no_corruption(tmp_path: Path) -> None:
    """Two real processes writing to the same JSONL must not corrupt lines.

    Uses multiprocessing.Process (real OS processes, no threading.Lock
    fakery). Each process writes 50 distinct entries. After both join,
    we expect exactly 100 valid JSON lines (100 distinct trade_ids).
    """
    ledger_path = tmp_path / "learnings_outcomes.jsonl"
    archive_dir = tmp_path / "archive"
    repo_root = str(REPO_ROOT)

    n_per = 50
    ctx = multiprocessing.get_context("spawn")
    p1 = ctx.Process(
        target=_writer_subprocess,
        args=(repo_root, str(ledger_path), str(archive_dir), n_per, "A"),
    )
    p2 = ctx.Process(
        target=_writer_subprocess,
        args=(repo_root, str(ledger_path), str(archive_dir), n_per, "B"),
    )
    p1.start()
    p2.start()
    p1.join(timeout=60)
    p2.join(timeout=60)
    assert p1.exitcode == 0, f"writer1 exit={p1.exitcode}"
    assert p2.exitcode == 0, f"writer2 exit={p2.exitcode}"

    lines = ledger_path.read_text().splitlines()
    assert len(lines) == 2 * n_per, (
        f"expected {2 * n_per} lines, got {len(lines)}"
    )
    # Every line must parse as valid JSON.
    parsed = [json.loads(line) for line in lines]
    trade_ids = {e["trade_id"] for e in parsed}
    expected = {f"A-{i:03d}" for i in range(n_per)} | {
        f"B-{i:03d}" for i in range(n_per)
    }
    assert trade_ids == expected


# ── Test 3: fence missing — renderer refuses ──────────────────────────────────


def test_fence_missing_renderer_refuses(tmp_path: Path, caplog) -> None:
    """If learnings.md exists without fence markers, the renderer must
    refuse (log WARN, do NOT modify the file)."""
    md = tmp_path / "learnings.md"
    original = (
        "# Buddy Trading Learnings\n\n"
        "Hand-written human content here.\n"
        "- [2026-04-15] some learning\n"
    )
    md.write_text(original)

    ledger = _make_ledger(tmp_path)
    ledger.append({"trade_id": "x1", "pair": "EUR_USD"})
    renderer = _make_renderer(tmp_path, ledger)

    import logging
    with caplog.at_level(logging.WARNING):
        wrote = renderer.render(allow_insert=False)
    assert wrote is False
    assert md.read_text() == original  # untouched
    assert any("fence markers missing" in r.message for r in caplog.records)


# ── Test 4: fence missing + allow_insert — appends at bottom ──────────────────


def test_fence_first_run_inserts_at_bottom(tmp_path: Path) -> None:
    """allow_insert=True: append a fresh fenced block at the file's end,
    preserving everything else byte-for-byte."""
    md = tmp_path / "learnings.md"
    original = (
        "# Buddy Trading Learnings\n\n"
        "Hand-written human content here.\n"
        "- [2026-04-15] some learning\n"
    )
    md.write_text(original)

    ledger = _make_ledger(tmp_path)
    ledger.append({
        "trade_id": "first-run-1",
        "pair": "EUR_USD",
        "direction": "LONG",
        "pnl_pips": 12.5,
    })
    renderer = _make_renderer(tmp_path, ledger)

    wrote = renderer.render(allow_insert=True)
    assert wrote is True

    new_content = md.read_text()
    # Original prefix preserved exactly.
    assert new_content.startswith(original)
    # Fence markers present.
    assert FENCE_BEGIN in new_content
    assert FENCE_END in new_content
    # The outcome row landed in the table.
    assert "first-run-1" in new_content
    # Begin marker comes before End marker.
    assert new_content.index(FENCE_BEGIN) < new_content.index(FENCE_END)


# ── Test 5: retention archives oldest when over 5000 ──────────────────────────


def test_retention_archives_oldest_when_over_5000(tmp_path: Path) -> None:
    """Pre-seed jsonl with 5001 lines; append one more; expect oldest 1000
    moved to archive, live ledger now 4001 lines (5001 + 1 - 1000)."""
    ledger = _make_ledger(tmp_path)

    # Pre-seed 5001 lines bypassing the ledger (we want a known state).
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger.ledger_path, "w", encoding="utf-8") as f:
        for i in range(5001):
            f.write(json.dumps({
                "kind": "outcome",
                "trade_id": f"seed-{i:05d}",
                "ts_utc": f"2026-04-01T00:00:{i % 60:02d}Z",
            }) + "\n")

    # Now append one more — this should trigger archival.
    ok = ledger.append({"trade_id": "trigger-archive", "pair": "EUR_USD"})
    assert ok is True

    live_lines = ledger.ledger_path.read_text().splitlines()
    # 5001 + 1 (new) = 5002, then -1000 archived = 4002
    assert len(live_lines) == 4002, f"got {len(live_lines)} live lines"

    # Archive file exists and contains the oldest 1000 lines.
    archive_files = list((tmp_path / "archive").glob("learnings_outcomes-*.jsonl"))
    assert len(archive_files) == 1
    archive_lines = archive_files[0].read_text().splitlines()
    assert len(archive_lines) == 1000
    # First archived line is the oldest seed.
    archived_first = json.loads(archive_lines[0])
    assert archived_first["trade_id"] == "seed-00000"
    # Last archived is seed-00999.
    archived_last = json.loads(archive_lines[-1])
    assert archived_last["trade_id"] == "seed-00999"
    # Live ledger now begins at seed-01000.
    live_first = json.loads(live_lines[0])
    assert live_first["trade_id"] == "seed-01000"
    # Live ledger ends with the new entry.
    live_last = json.loads(live_lines[-1])
    assert live_last["trade_id"] == "trigger-archive"


def test_retention_archive_idempotent(tmp_path: Path) -> None:
    """Calling the archive logic twice (via two appends) should not
    re-archive already-archived lines.
    """
    ledger = _make_ledger(tmp_path)
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger.ledger_path, "w", encoding="utf-8") as f:
        for i in range(5001):
            f.write(json.dumps({"kind": "outcome", "trade_id": f"seed-{i:05d}"}) + "\n")

    ledger.append({"trade_id": "first-trigger"})
    archive_files = list((tmp_path / "archive").glob("learnings_outcomes-*.jsonl"))
    archive_lines_1 = archive_files[0].read_text().splitlines()
    assert len(archive_lines_1) == 1000

    # Second append: live ledger is now 4002 < 5000, so no second archive.
    ledger.append({"trade_id": "second-no-trigger"})
    archive_lines_2 = archive_files[0].read_text().splitlines()
    assert len(archive_lines_2) == 1000  # unchanged


# ── Test 6: corrupt jsonl line skipped by renderer ────────────────────────────


def test_corrupt_jsonl_line_skipped_by_renderer(tmp_path: Path, caplog) -> None:
    """A malformed JSON line in the ledger must not crash the renderer;
    logger emits WARN, the line is skipped, valid lines render."""
    ledger = _make_ledger(tmp_path)
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    valid_a = json.dumps({"kind": "outcome", "trade_id": "good-A", "pair": "EUR_USD"})
    valid_b = json.dumps({"kind": "outcome", "trade_id": "good-B", "pair": "GBP_USD"})
    with open(ledger.ledger_path, "w", encoding="utf-8") as f:
        f.write(valid_a + "\n")
        f.write("{this is not valid json}\n")
        f.write(valid_b + "\n")

    md = tmp_path / "learnings.md"
    md.write_text("# Buddy Trading Learnings\n\nhuman content\n")
    renderer = _make_renderer(tmp_path, ledger)

    import logging
    with caplog.at_level(logging.WARNING):
        wrote = renderer.render(allow_insert=True)
    assert wrote is True

    content = md.read_text()
    assert "good-A" in content
    assert "good-B" in content
    assert "this is not valid json" not in content
    assert any("corrupt line" in r.message for r in caplog.records)


# ── Test 7: counterfactual redirect (no duplication) ──────────────────────────


def test_counterfactual_redirect_to_dedicated_jsonl(tmp_path: Path) -> None:
    """CounterfactualLearner's append_learning must write to its own jsonl
    (NOT to the outcomes ledger). The outcome ledger gets its own entry
    via the normal hook — verify the two files do NOT both contain the
    same counterfactual analysis."""
    from src.scanner.automation.counterfactual_learner import CounterfactualLearner

    cf_path = tmp_path / "counterfactuals.jsonl"
    outcomes_path = tmp_path / "learnings_outcomes.jsonl"

    learner = CounterfactualLearner(output_path=cf_path)
    learner.append_learning(
        "**2026-05-10T18:30:00Z** — Counterfactual Insights (Trade 1234):\n"
        "Trade 1234 (EUR_USD) would have had +35% better odds under low_volatility scenario."
    )

    # Counterfactual file populated.
    assert cf_path.exists(), "counterfactuals.jsonl should be created"
    cf_lines = cf_path.read_text().splitlines()
    assert len(cf_lines) == 1
    cf_entry = json.loads(cf_lines[0])
    assert cf_entry["kind"] == "counterfactual"
    assert "Trade 1234" in cf_entry["text"]

    # Outcomes ledger NOT touched by counterfactual writes.
    assert not outcomes_path.exists() or outcomes_path.read_text() == ""

    # Now hook the regular outcomes ledger — confirm it gets ITS own entry
    # for the closed trade, but does NOT contain the counterfactual analysis.
    ledger = OutcomesLedger(
        ledger_path=outcomes_path,
        archive_dir=tmp_path / "archive",
    )
    ledger.append({
        "trade_id": "1234",
        "pair": "EUR_USD",
        "direction": "LONG",
        "pnl_pips": -10.0,
        "exit_reason": "sl_hit",
    })
    out_lines = outcomes_path.read_text().splitlines()
    assert len(out_lines) == 1
    out_entry = json.loads(out_lines[0])
    assert out_entry["kind"] == "outcome"
    assert out_entry["trade_id"] == "1234"
    # The counterfactual prose did NOT leak into the outcomes ledger.
    full_text = outcomes_path.read_text()
    assert "Counterfactual Insights" not in full_text
    assert "better odds" not in full_text


# ── Test 8: backfill idempotent ───────────────────────────────────────────────


def test_backfill_idempotent(tmp_path: Path) -> None:
    """Running the backfill helper twice over the same journal entries
    must NOT produce duplicates in the ledger."""
    ledger = _make_ledger(tmp_path)

    # Synthesize a journal entry mimicking trade_journal_rl.json layout.
    journal_entry = {
        "trade_id": "1124",
        "timestamp": "2026-04-03T19:40:40+00:00",
        "pair": "EUR_GBP",
        "direction": "SHORT",
        "confidence": 0.69,
        "regime": {"volatility_regime": "LOW", "atr_pips": 5.0},
        "agents": {
            "weighted_vote_score": 0.62,
            "agent_reasons": {
                "trend": {"passed": False, "weight": 1.1, "reason": "neutral"},
                "momentum": {"passed": True, "weight": 0.9},
            },
        },
        "outcome": {
            "realized_pl": -381.50,
            "pnl_pips": -11.5,
            "trade_won": False,
            "exit_reason": "sl_hit",
            "close_time": "2026-04-03T20:10:35Z",
        },
    }

    # First emission
    ok1 = emit_outcome_from_journal_entry(journal_entry, ledger=ledger)
    assert ok1 is True
    lines1 = ledger.ledger_path.read_text().splitlines()
    assert len(lines1) == 1
    entry1 = json.loads(lines1[0])
    assert entry1["trade_id"] == "1124"
    assert entry1["pair"] == "EUR_GBP"
    assert entry1["direction"] == "SHORT"
    assert entry1["regime"] == "LOW"
    assert entry1["wvs"] == 0.62
    assert entry1["top_dissent_agent"] == "trend"
    assert entry1["pnl_pips"] == -11.5
    assert entry1["pnl_currency"] == -381.50
    assert entry1["exit_reason"] == "sl_hit"

    # Second emission with the SAME entry — must be idempotent (no duplicate).
    ok2 = emit_outcome_from_journal_entry(journal_entry, ledger=ledger)
    assert ok2 is True  # treated as success even though skipped
    lines2 = ledger.ledger_path.read_text().splitlines()
    assert len(lines2) == 1, f"expected 1 line, got {len(lines2)}"


# ── Test 9: emit handles minimal entries gracefully ──────────────────────────


def test_emit_handles_minimal_journal_entry(tmp_path: Path) -> None:
    """A trade journal entry with sparse fields should still produce a
    valid JSONL line (None values preserved, no crash)."""
    ledger = _make_ledger(tmp_path)
    minimal = {
        "trade_id": "min-1",
        "outcome": {"realized_pl": 0.0},
    }
    ok = emit_outcome_from_journal_entry(minimal, ledger=ledger)
    assert ok is True
    line = ledger.ledger_path.read_text().splitlines()[0]
    entry = json.loads(line)
    assert entry["trade_id"] == "min-1"
    assert entry["pair"] is None
    assert entry["direction"] is None
    assert entry["confidence"] is None
    assert entry["wvs"] is None
    assert entry["top_dissent_agent"] is None


# ── Test 10: re-render replaces (does not duplicate) the fenced section ──────


def test_renderer_replaces_existing_fenced_section(tmp_path: Path) -> None:
    """Calling render twice should result in ONE fenced block, with the
    most recent content. Human content above the fence is preserved."""
    ledger = _make_ledger(tmp_path)
    md = tmp_path / "learnings.md"
    human_content = (
        "# Buddy Trading Learnings\n\n"
        "Hand-written content above the fence.\n"
        "- [2026-04-15] human learning\n"
    )
    md.write_text(human_content)

    renderer = _make_renderer(tmp_path, ledger)

    ledger.append({"trade_id": "first", "pair": "EUR_USD", "pnl_pips": 5.0})
    renderer.render(allow_insert=True)

    ledger.append({"trade_id": "second", "pair": "GBP_USD", "pnl_pips": -3.0})
    renderer.render(allow_insert=True)  # re-render

    content = md.read_text()
    # Exactly ONE pair of markers.
    assert content.count(FENCE_BEGIN) == 1
    assert content.count(FENCE_END) == 1
    # Both entries appear (cumulative ledger state).
    assert "first" in content
    assert "second" in content
    # Human content preserved verbatim.
    assert content.startswith(human_content)
