"""US-006 RejectModal tests — non-empty note validation.

NO MOCKS per CLAUDE.md No-Mock Rule. Tests drive the real RejectModal
via Textual Pilot to verify:
  (a) empty input → Submit button disabled
  (b) whitespace-only input → button disabled; validation message 'note required'
      shown when _try_submit() is called (e.g. via Enter key path)
  (c) non-empty note → modal dismisses with the note string;
      HomeworkReviewer.reject() builds a TrainingSignal carrying that note

Async tests use asyncio.run() directly (no pytest-asyncio plugin required).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Iterator

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Label, Static

from src.scanner.automation.homework import HomeworkEntry, HomeworkStore, TrainingSignal
from src.scanner.automation.homework.reviewer import HomeworkReviewer
from src.tui.screens.inbox_screen import RejectModal


# ─────────────────────── Host app ──────────────────────────────────────────


class _Host(App):
    """Minimal host that pushes RejectModal and captures its dismiss value."""

    def __init__(self, proposal_key: str = "test-proposal") -> None:
        super().__init__()
        self._proposal_key = proposal_key
        self.captured: list = []

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(RejectModal(self._proposal_key), self.captured.append)


# ─────────────────────── Helpers ───────────────────────────────────────────


def _minimal_entry() -> HomeworkEntry:
    now = datetime.now(timezone.utc).isoformat()
    return HomeworkEntry(
        homework_id=str(uuid.uuid4()),
        trade_id="T-9999",
        generated_at=now,
        pair="EUR_USD",
        direction="LONG",
        entry_price=1.0823,
        sl_price=1.0810,
        tp_price=1.0850,
        rr_ratio=2.1,
        confidence=0.68,
        weighted_vote_score=0.71,
        regime="NORMAL",
        agent_verdicts=[{"agent": "trend", "passed": True}],
        close_time=now,
        close_price=1.0850,
        realized_pl=27.0,
        close_reason="TP",
        duration_minutes=45,
        mfe_pips=2.7,
        mae_pips=0.8,
        analysis_markdown="Trend aligned. Entry clean.",
        proposed_lesson="Trust high-ADX setups in NORMAL regime.",
        confidence_in_analysis=0.75,
        agents_to_reinforce=["trend"],
        agents_to_penalize=[],
    )


# ─────────────────────── AC1: button disabled state ────────────────────────


class TestSubmitButtonState:

    def test_button_disabled_on_empty_input(self) -> None:
        """AC1(a): Submit button starts disabled when Input is empty."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                btn = screen.query_one("#reject-confirm", Button)
                assert btn.disabled is True

        asyncio.run(_run())
        # Nothing dismissed
        assert host.captured == []

    def test_button_disabled_for_whitespace_only(self) -> None:
        """AC1(b): Button stays disabled when Input contains only whitespace."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                inp = screen.query_one("#reject-input", Input)
                inp.value = "   "
                await pilot.pause()
                btn = screen.query_one("#reject-confirm", Button)
                assert btn.disabled is True

        asyncio.run(_run())
        assert host.captured == []

    def test_button_enabled_for_non_empty_input(self) -> None:
        """Button becomes enabled as soon as non-whitespace text is entered."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                inp = screen.query_one("#reject-input", Input)
                inp.value = "reason"
                await pilot.pause()
                btn = screen.query_one("#reject-confirm", Button)
                assert btn.disabled is False

        asyncio.run(_run())
        assert host.captured == []


# ─────────────────────── AC2: validation label ─────────────────────────────


class TestValidationLabel:

    def test_validation_label_empty_initially(self) -> None:
        """Validation label starts blank (no false error on open)."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                label = screen.query_one("#reject-validation", Label)
                assert str(label.render()) == ""

        asyncio.run(_run())

    def test_whitespace_submit_shows_note_required(self) -> None:
        """AC2: whitespace-only _try_submit shows 'note required', no dismiss."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                inp = screen.query_one("#reject-input", Input)
                inp.value = "   "
                await pilot.pause()
                # Call _try_submit directly — same code path as Enter key
                screen._try_submit()
                await pilot.pause()
                label = screen.query_one("#reject-validation", Label)
                assert str(label.render()) == "note required"

        asyncio.run(_run())
        # Modal was NOT dismissed
        assert host.captured == []

    def test_empty_submit_shows_note_required(self) -> None:
        """Empty _try_submit also shows 'note required' and does not dismiss."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                screen._try_submit()
                await pilot.pause()
                label = screen.query_one("#reject-validation", Label)
                assert str(label.render()) == "note required"

        asyncio.run(_run())
        assert host.captured == []

    def test_validation_label_cleared_when_user_types_valid_text(self) -> None:
        """Validation label clears as soon as the user types a non-empty value."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                inp = screen.query_one("#reject-input", Input)
                # First trigger validation
                screen._try_submit()
                await pilot.pause()
                label = screen.query_one("#reject-validation", Label)
                assert str(label.render()) == "note required"
                # Now type a valid value — label should clear
                inp.value = "valid reason"
                await pilot.pause()
                assert str(label.render()) == ""

        asyncio.run(_run())


# ─────────────────────── AC3: ESC cancels cleanly ──────────────────────────


class TestEscapeCancel:

    def test_esc_dismisses_none(self) -> None:
        """AC3: ESC emits None (no rejection) and does not raise."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

        asyncio.run(_run())
        assert host.captured == [None]

    def test_esc_with_text_in_input_still_dismisses_none(self) -> None:
        """ESC even with a typed note still cancels cleanly."""
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                inp = screen.query_one("#reject-input", Input)
                inp.value = "some reason"
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

        asyncio.run(_run())
        assert host.captured == [None]


# ─────────────────────── AC4(c): non-empty submit ──────────────────────────


class TestNonEmptySubmit:

    def test_valid_note_dismisses_with_stripped_note(self) -> None:
        """AC4(c): non-empty note → modal dismisses with the stripped note string."""
        note = "confidence head over-weights thin-book hours; reduce influence"
        host = _Host()

        async def _run() -> None:
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, RejectModal)
                inp = screen.query_one("#reject-input", Input)
                inp.value = f"  {note}  "  # leading/trailing spaces stripped
                await pilot.pause()
                btn = screen.query_one("#reject-confirm", Button)
                assert btn.disabled is False
                await pilot.click("#reject-confirm")
                await pilot.pause()

        asyncio.run(_run())
        assert host.captured == [note]

    def test_non_empty_submit_produces_training_signal(self, tmp_path) -> None:
        """AC4(c): captured note → HomeworkReviewer.reject() returns TrainingSignal with note."""
        note = "momentum head voted wrong; risk sentinel too permissive"
        store = HomeworkStore(
            pending_path=tmp_path / "pending.jsonl",
            history_path=tmp_path / "history.jsonl",
            quarantine_path=tmp_path / "quarantine.jsonl",
        )
        entry = _minimal_entry()
        store.add(entry)

        reviewer = HomeworkReviewer(store=store)
        signal = reviewer.reject(entry.homework_id, note=note)

        assert signal is not None
        assert isinstance(signal, TrainingSignal)
        assert signal.operator_note == note
        assert signal.operator_action == "rejected"
        assert signal.trade_id == entry.trade_id
