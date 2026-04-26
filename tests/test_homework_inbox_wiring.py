"""Regression for InboxScreen ↔ HomeworkStore wiring.

Following the US-604 pattern (src/tui/embedded_scanner.py wiring gap that
went silent for days): static + behavioral checks. If anyone ever removes
the homework integration from inbox_screen.py, this fails loudly.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


class TestStaticWiring:
    def test_inbox_screen_imports_homework(self) -> None:
        text = Path("src/tui/screens/inbox_screen.py").read_text()
        assert "HomeworkStore" in text, (
            "InboxScreen must import HomeworkStore. "
            "See spec §5 and Task 5 of the implementation plan."
        )
        assert "homework" in text.lower()

    def test_filter_pill_ids_present(self) -> None:
        text = Path("src/tui/screens/inbox_screen.py").read_text()
        assert "filter-all" in text or "filter_all" in text
        assert "filter-homework" in text or "filter_homework" in text
        assert "filter-adjustments" in text or "filter_adjustments" in text

    def test_compose_uses_horizontal_two_pane(self) -> None:
        from src.tui.screens.inbox_screen import InboxScreen
        src = inspect.getsource(InboxScreen.compose)
        assert "Horizontal" in src

    def test_action_handlers_present(self) -> None:
        from src.tui.screens.inbox_screen import InboxScreen
        for action in ("approve", "reject", "snooze"):
            assert hasattr(InboxScreen, f"action_{action}"), (
                f"InboxScreen missing action_{action}"
            )

    def test_homework_reviewer_imported_or_referenced(self) -> None:
        text = Path("src/tui/screens/inbox_screen.py").read_text()
        assert "HomeworkReviewer" in text


class TestBehavioralWiring:
    def test_load_unified_rows_returns_both_types(self, tmp_path: Path, monkeypatch) -> None:
        """Build a stub InboxScreen that reads from tmp homework store + a stub
        adjustment list, verify both sources end up in the queue."""
        # Implementation depends on the final InboxScreen class structure.
        # Minimum guarantee: a function `_load_unified_rows` exists and returns
        # a list of UnifiedInboxRow objects with mixed entry_type values.
        import src.tui.screens.inbox_screen as mod
        assert hasattr(mod, "UnifiedInboxRow"), (
            "module must export UnifiedInboxRow adapter dataclass"
        )
        assert hasattr(mod, "_homework_to_row")
        assert hasattr(mod, "_adjustment_to_row")


# ── Behavioral wiring tests (US-604 fingerprint defense) ─────────────
#
# These guard the exact failure mode that bit Phase 96 v1: a feature flag
# / record-side existed, BUT the live consumer never reached the call site.
# Static-shape tests pass while the production path is dead. The tests
# below simulate full end-to-end paths and assert observable side-effects
# (file mutations) rather than imports/attributes.


def _seed_homework_entry(pending_path: Path, trade_id: str = "1220") -> str:
    """Write a single HomeworkEntry to pending. Returns its homework_id."""
    import json as _json
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    homework_id = f"hw-{trade_id}"
    entry = {
        "homework_id": homework_id, "trade_id": trade_id,
        "generated_at": "2026-04-25T22:47:35Z",
        "pair": "EUR_AUD", "direction": "SHORT",
        "entry_price": 1.6543, "sl_price": 1.6580, "tp_price": 1.6470,
        "rr_ratio": 1.97, "confidence": 0.68, "weighted_vote_score": 0.76,
        "regime": "LOW", "agent_verdicts": [],
        "close_time": "2026-04-15T02:46:03Z", "close_price": 1.6580,
        "realized_pl": -354.56, "close_reason": "SL",
        "duration_minutes": 32, "mfe_pips": 4.0, "mae_pips": 39.0,
        "analysis_markdown": "# Analysis", "proposed_lesson": "trend hard-veto",
        "confidence_in_analysis": 0.70,
        "agents_to_reinforce": ["trend"],
        "agents_to_penalize": ["mean_reversion"],
    }
    with open(pending_path, "a") as fh:
        fh.write(_json.dumps(entry, sort_keys=True) + "\n")
    return homework_id


def _seed_weights_file(weights_path: Path) -> None:
    """Write a baseline regime-keyed weights file."""
    import json as _json
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(_json.dumps({
        "NORMAL": {"trend": 1.0, "mean_reversion": 0.9},
        "HIGH":   {"trend": 1.15, "mean_reversion": 0.9},
        "LOW":    {"trend": 1.0, "mean_reversion": 0.95},
    }))


class TestApproveLoopAppliesWeights:
    """Phase 96.5 fingerprint: approve() must MOVE agent_weights.json.

    This is the exact gap that survived Phase 96 v1: signals were built
    but never applied. A static "is TrainingSignalApplicator imported in
    reviewer.py?" test would have passed. THIS test would have failed.
    """

    def test_approve_writes_deltas_to_weights_file(self, tmp_path: Path) -> None:
        from src.scanner.automation.homework.applicator import TrainingSignalApplicator
        from src.scanner.automation.homework.reviewer import HomeworkReviewer
        from src.scanner.automation.homework.store import HomeworkStore

        pending = tmp_path / "homework_pending.jsonl"
        history = tmp_path / "homework_history.jsonl"
        weights = tmp_path / "agent_weights.json"

        _seed_weights_file(weights)
        homework_id = _seed_homework_entry(pending, trade_id="1220")

        store = HomeworkStore(pending_path=pending, history_path=history)
        applicator = TrainingSignalApplicator(weights_path=weights)
        reviewer = HomeworkReviewer(
            store=store,
            rejected_log_path=tmp_path / "rejected.jsonl",
            applicator=applicator,
        )

        # BEFORE: weights file has the seeded baseline
        import json as _json
        before = _json.loads(weights.read_text())
        assert before["LOW"]["trend"] == 1.0
        assert before["LOW"]["mean_reversion"] == 0.95

        # ACT: approve the entry
        signal = reviewer.approve(homework_id)
        assert signal is not None, "approve must return a signal for known id"

        # AFTER: weights file MUST have moved (this is what was broken in v1)
        after = _json.loads(weights.read_text())
        assert after["LOW"]["trend"] == pytest.approx(1.02), "trend was reinforced (+0.02)"
        assert after["LOW"]["mean_reversion"] == pytest.approx(0.93), "mean_reversion was penalized (-0.02)"
        # Other regimes untouched
        assert after["NORMAL"]["trend"] == 1.0
        assert after["HIGH"]["trend"] == 1.15
        # Pending now empty (entry moved to history)
        assert store._read_jsonl(pending) == []
        history_entries = store.list_history()
        assert len(history_entries) == 1
        assert history_entries[0].operator_grade == "approved"


class TestRejectLoopDoesNotApplyWeights:
    """reject() MUST NOT move weights — operator's note is the future signal."""

    def test_reject_leaves_weights_untouched_and_logs_heuristic(self, tmp_path: Path) -> None:
        from src.scanner.automation.homework.applicator import TrainingSignalApplicator
        from src.scanner.automation.homework.reviewer import HomeworkReviewer
        from src.scanner.automation.homework.store import HomeworkStore

        pending = tmp_path / "homework_pending.jsonl"
        history = tmp_path / "homework_history.jsonl"
        weights = tmp_path / "agent_weights.json"
        rejected_log = tmp_path / "rejected_heuristics_log.jsonl"

        _seed_weights_file(weights)
        homework_id = _seed_homework_entry(pending, trade_id="1220")

        store = HomeworkStore(pending_path=pending, history_path=history)
        applicator = TrainingSignalApplicator(weights_path=weights)
        reviewer = HomeworkReviewer(
            store=store, rejected_log_path=rejected_log, applicator=applicator,
        )

        import json as _json
        before_text = weights.read_text()

        # ACT: reject with a note
        signal = reviewer.reject(homework_id, note="trend isn't tradable in LOW regime here")
        assert signal is not None
        assert signal.operator_action == "rejected"
        assert signal.agent_weight_deltas == {}, "rejected signals must carry empty deltas"

        # weights file MUST be byte-identical (no apply on reject path)
        after_text = weights.read_text()
        assert before_text == after_text, "reject must not touch agent_weights.json"

        # rejected heuristic log MUST have an entry
        assert rejected_log.exists(), "rejected heuristic must be logged for future tuning"
        log_lines = rejected_log.read_text().strip().splitlines()
        assert len(log_lines) == 1
        log_entry = _json.loads(log_lines[0])
        assert log_entry["homework_id"] == homework_id
        assert "trend isn't tradable" in log_entry["operator_note"]


class TestBulkApproveDispatchesBothTypes:
    """The 23c132b bug: action_approve_all only ran AdjustmentApprover.approve_all,
    homework rows were silently skipped. This test guards the fix."""

    def test_action_approve_all_processes_homework_rows(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import src.scanner.automation.homework.store as store_mod
        import src.scanner.automation.homework.applicator as applicator_mod
        import src.scanner.automation.homework.reviewer as reviewer_mod

        pending = tmp_path / "homework_pending.jsonl"
        history = tmp_path / "homework_history.jsonl"
        weights = tmp_path / "agent_weights.json"
        rejected_log = tmp_path / "rejected.jsonl"

        # Redirect default paths so HomeworkStore() / TrainingSignalApplicator()
        # built inside action_approve_all hit the tmp dir.
        monkeypatch.setattr(store_mod, "DEFAULT_PENDING_PATH", pending)
        monkeypatch.setattr(store_mod, "DEFAULT_HISTORY_PATH", history)
        monkeypatch.setattr(
            store_mod, "DEFAULT_QUARANTINE_PATH", tmp_path / "quarantine.jsonl"
        )
        monkeypatch.setattr(applicator_mod, "DEFAULT_WEIGHTS_PATH", weights)
        monkeypatch.setattr(reviewer_mod, "DEFAULT_REJECTED_LOG_PATH", rejected_log)

        _seed_weights_file(weights)
        _seed_homework_entry(pending, trade_id="42")
        _seed_homework_entry(pending, trade_id="43")

        from src.tui.screens.inbox_screen import InboxScreen, _homework_to_row
        screen = InboxScreen(project_root=str(tmp_path))

        # Hand-build _current_rows with two homework entries (no adjustments)
        from src.scanner.automation.homework.store import HomeworkStore
        store = HomeworkStore(pending_path=pending, history_path=history)
        screen._current_rows = [_homework_to_row(e) for e in store.list_pending()]
        assert len(screen._current_rows) == 2

        # Stub Textual-runtime methods that need an App context
        screen.notify = lambda *a, **kw: None  # type: ignore
        screen._load_proposals = lambda: None  # type: ignore

        # ACT
        screen.action_approve_all()

        # ASSERT: BOTH homework rows were processed (would fail pre-23c132b)
        assert store._read_jsonl(pending) == [], "all homework moved to history"
        history_entries = store.list_history()
        assert len(history_entries) == 2
        assert all(e.operator_grade == "approved" for e in history_entries)

        # AND: weights file moved (proves the full chain ran, not just store ops)
        import json as _json
        after = _json.loads(weights.read_text())
        # 2 approves × +0.02 trend each = +0.04 total
        assert after["LOW"]["trend"] == pytest.approx(1.04)


class TestFilterPillsBehavioral:
    """Filter pills must actually filter what _load_unified_rows returns."""

    def test_homework_filter_hides_adjustments(self, tmp_path: Path, monkeypatch) -> None:
        import src.scanner.automation.homework.store as store_mod
        import json as _json

        pending_hw = tmp_path / "homework_pending.jsonl"
        history_hw = tmp_path / "homework_history.jsonl"
        monkeypatch.setattr(store_mod, "DEFAULT_PENDING_PATH", pending_hw)
        monkeypatch.setattr(store_mod, "DEFAULT_HISTORY_PATH", history_hw)
        monkeypatch.setattr(
            store_mod, "DEFAULT_QUARANTINE_PATH", tmp_path / "q.jsonl"
        )

        _seed_homework_entry(pending_hw, trade_id="100")
        _seed_homework_entry(pending_hw, trade_id="101")

        # Adjustments file
        pending_adj = tmp_path / ".claude" / "pending_adjustments.json"
        pending_adj.parent.mkdir(parents=True, exist_ok=True)
        pending_adj.write_text(_json.dumps({"proposals": [
            {"id": "adj-1", "status": "pending", "key": "min_confidence",
             "current_value": 60.0, "proposed_value": 62.0,
             "timestamp": "2026-04-26T00:00:00Z"}
        ]}))

        from src.tui.screens.inbox_screen import InboxScreen
        screen = InboxScreen(project_root=str(tmp_path))
        screen._proposals = screen._read_proposals()
        assert len(screen._proposals) == 1, "adjustment seeded"

        # All filter sees both types
        screen._current_filter = "all"
        rows_all = screen._load_unified_rows()
        types_all = {r.entry_type for r in rows_all}
        assert types_all == {"homework", "adjustment"}, types_all

        # Homework filter hides adjustments
        screen._current_filter = "homework"
        rows_hw = screen._load_unified_rows()
        assert all(r.entry_type == "homework" for r in rows_hw)
        assert len(rows_hw) == 2

        # Adjustments filter hides homework
        screen._current_filter = "adjustments"
        rows_adj = screen._load_unified_rows()
        assert all(r.entry_type == "adjustment" for r in rows_adj)
        assert len(rows_adj) == 1
