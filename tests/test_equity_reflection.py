"""US-019 — Outcome-grounded reflection loop tests.

Real classes + real disk via ``tmp_path``. Zero ``unittest.mock``, zero
``MagicMock``, zero ``patch``. Every test constructs a real
:class:`ReflectionLedger` against a real ``SHIP_GATE.json`` file written
to ``tmp_path``, and asserts behavior by reading the persisted ledger
back from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pytest

from src.equity.reflection import (
    DEFAULT_MAX_LESSONS,
    LEDGER_VERSION,
    Lesson,
    PendingDecision,
    ReflectionError,
    ReflectionLedger,
    enforce_ship_gate,
)


UNIVERSE_HASH = "abc123" + "0" * 58


def _ship_gate(
    tmp_path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = UNIVERSE_HASH,
    write: bool = True,
) -> Path:
    path = tmp_path / "backtests" / "SHIP_GATE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not write:
        return path
    payload = {
        "gate_pass": gate_pass,
        "universe_hash": universe_hash,
        "asof": "2026-06-21T21:36:00Z",
        "sharpe": 0.92,
        "max_drawdown": 0.229,
        "positive_years_ratio": 0.85,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _decision(
    decision_id: str = "d-1",
    ticker: str = "SPY",
    target_weight: float = 0.6,
    rationale: str = "vol-managed beta long",
    decided_at: str = "2026-06-21T20:00:00Z",
    mature_at: str = "2026-06-28T20:00:00Z",
) -> PendingDecision:
    return PendingDecision(
        decision_id=decision_id,
        ticker=ticker,
        target_weight=target_weight,
        rationale=rationale,
        decided_at=decided_at,
        mature_at=mature_at,
    )


# ---------------------------------------------------------------------------
# Ship-gate guard
# ---------------------------------------------------------------------------


class TestShipGateGuard:
    def test_missing_ship_gate_blocks(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_gate.json"
        with pytest.raises(ReflectionError, match="not found"):
            enforce_ship_gate(missing, UNIVERSE_HASH)

    def test_gate_pass_false_blocks(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path, gate_pass=False)
        with pytest.raises(ReflectionError, match="gate_pass"):
            enforce_ship_gate(gate, UNIVERSE_HASH)

    def test_universe_hash_mismatch_blocks(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        with pytest.raises(ReflectionError, match="universe_hash mismatch"):
            enforce_ship_gate(gate, "different_hash")

    def test_universe_hash_blank_blocks(self, tmp_path: Path) -> None:
        gate = tmp_path / "SHIP_GATE.json"
        gate.write_text(
            json.dumps({"gate_pass": True, "universe_hash": ""})
        )
        with pytest.raises(ReflectionError, match="universe_hash"):
            enforce_ship_gate(gate, UNIVERSE_HASH)

    def test_malformed_json_blocks(self, tmp_path: Path) -> None:
        gate = tmp_path / "SHIP_GATE.json"
        gate.write_text("{not json")
        with pytest.raises(ReflectionError, match="cannot read"):
            enforce_ship_gate(gate, UNIVERSE_HASH)

    def test_non_object_payload_blocks(self, tmp_path: Path) -> None:
        gate = tmp_path / "SHIP_GATE.json"
        gate.write_text("[1, 2, 3]")
        with pytest.raises(ReflectionError, match="not a JSON object"):
            enforce_ship_gate(gate, UNIVERSE_HASH)

    def test_passing_gate_allows(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        enforce_ship_gate(gate, UNIVERSE_HASH)

    def test_ledger_construction_blocked_without_gate(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ReflectionError, match="not found"):
            ReflectionLedger(
                universe_hash=UNIVERSE_HASH,
                ledger_path=tmp_path / "reflection.json",
                ship_gate_path=tmp_path / "missing_gate.json",
            )

    def test_ledger_construction_blocked_when_gate_false(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path, gate_pass=False)
        with pytest.raises(ReflectionError, match="gate_pass"):
            ReflectionLedger(
                universe_hash=UNIVERSE_HASH,
                ledger_path=tmp_path / "reflection.json",
                ship_gate_path=gate,
            )

    def test_ledger_construction_blocked_on_hash_mismatch(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path, universe_hash="something_else")
        with pytest.raises(ReflectionError, match="universe_hash mismatch"):
            ReflectionLedger(
                universe_hash=UNIVERSE_HASH,
                ledger_path=tmp_path / "reflection.json",
                ship_gate_path=gate,
            )


# ---------------------------------------------------------------------------
# Decision logging + resolution
# ---------------------------------------------------------------------------


class TestReflectionLedger:
    def test_construct_writes_initial_ledger(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger_path = tmp_path / "reflection.json"
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=ledger_path,
            ship_gate_path=gate,
        )
        assert ledger_path.exists()
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert isinstance(payload, Mapping)
        assert payload["version"] == LEDGER_VERSION
        assert payload["universe_hash"] == UNIVERSE_HASH
        assert payload["pending"] == []
        assert payload["lessons"] == []
        assert ledger.pending() == []
        assert ledger.lessons() == []

    def test_log_decision_appends_atomically(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger_path = tmp_path / "reflection.json"
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=ledger_path,
            ship_gate_path=gate,
        )
        assert ledger.log_decision(_decision("d-1", "SPY")) is True
        assert ledger.log_decision(_decision("d-2", "QQQ")) is True
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        ids = [p["decision_id"] for p in payload["pending"]]
        assert ids == ["d-1", "d-2"]
        # No stray temp files left behind by the atomic writer.
        temps = [
            p
            for p in ledger_path.parent.iterdir()
            if p.name.startswith(f".{ledger_path.name}.")
        ]
        assert not temps, f"atomic write leaked temp files: {temps}"

    def test_log_decision_idempotent_on_decision_id(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        assert ledger.log_decision(_decision("d-1")) is True
        assert ledger.log_decision(_decision("d-1", ticker="other")) is False
        assert len(ledger.pending()) == 1

    def test_log_decision_rejects_non_dataclass(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        with pytest.raises(ReflectionError, match="PendingDecision"):
            ledger.log_decision({"decision_id": "d-1"})  # type: ignore[arg-type]

    def test_resolve_computes_alpha_and_writes_lesson(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger_path = tmp_path / "reflection.json"
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=ledger_path,
            ship_gate_path=gate,
        )
        ledger.log_decision(_decision("d-1", ticker="SPY"))
        lesson = ledger.resolve(
            decision_id="d-1",
            realized_return=0.012,
            benchmark_return=0.005,
        )
        assert lesson is not None
        assert lesson.alpha == pytest.approx(0.007)
        assert lesson.ticker == "SPY"
        assert "alpha=+0.0070" in lesson.text
        # Pending moved into lessons; both reflected on disk.
        on_disk = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert on_disk["pending"] == []
        assert len(on_disk["lessons"]) == 1
        assert on_disk["lessons"][0]["decision_id"] == "d-1"

    def test_resolve_idempotent_on_decision_id(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        ledger.log_decision(_decision("d-1"))
        first = ledger.resolve("d-1", 0.01, 0.0)
        second = ledger.resolve("d-1", 999.0, -999.0)  # ignored
        assert first is not None and second is not None
        assert first.alpha == pytest.approx(second.alpha)
        assert len(ledger.lessons()) == 1

    def test_resolve_unknown_decision_raises(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        with pytest.raises(ReflectionError, match="no pending decision"):
            ledger.resolve("missing-id", 0.0, 0.0)

    def test_resolve_with_extra_note(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        ledger.log_decision(_decision("d-1"))
        lesson = ledger.resolve(
            "d-1", 0.02, 0.01, extra_note="held through CPI print"
        )
        assert lesson is not None
        assert "held through CPI print" in lesson.text

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger_path = tmp_path / "reflection.json"
        first = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=ledger_path,
            ship_gate_path=gate,
        )
        first.log_decision(_decision("d-1", ticker="SPY"))
        first.log_decision(_decision("d-2", ticker="QQQ"))
        first.resolve("d-1", 0.015, 0.005)

        second = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=ledger_path,
            ship_gate_path=gate,
        )
        assert [p["decision_id"] for p in second.pending()] == ["d-2"]
        lessons = second.lessons()
        assert [le["decision_id"] for le in lessons] == ["d-1"]

    def test_persistence_universe_hash_mismatch_raises(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger_path = tmp_path / "reflection.json"
        ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=ledger_path,
            ship_gate_path=gate,
        )
        # Now swap the gate to a different universe hash; the new
        # ledger constructor should fail closed at load.
        new_hash = "z" * 64
        _ship_gate(tmp_path, universe_hash=new_hash)
        with pytest.raises(ReflectionError, match="universe_hash mismatch"):
            ReflectionLedger(
                universe_hash=new_hash,
                ledger_path=ledger_path,
                ship_gate_path=gate,
            )

    def test_corrupt_ledger_reinitialises(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger_path = tmp_path / "reflection.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text("{not valid json")
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=ledger_path,
            ship_gate_path=gate,
        )
        assert ledger.pending() == []
        assert ledger.lessons() == []
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert payload["version"] == LEDGER_VERSION

    def test_rotation_caps_ledger(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
            max_lessons=3,
        )
        for i in range(5):
            ledger.log_decision(_decision(f"d-{i}", ticker="SPY"))
            ledger.resolve(f"d-{i}", 0.001 * (i + 1), 0.0)
        kept = [le["decision_id"] for le in ledger.lessons()]
        assert kept == ["d-2", "d-3", "d-4"], kept

    def test_invalid_universe_hash_rejected(self, tmp_path: Path) -> None:
        _ship_gate(tmp_path)
        with pytest.raises(ReflectionError, match="universe_hash"):
            ReflectionLedger(
                universe_hash="",
                ledger_path=tmp_path / "reflection.json",
                ship_gate_path=tmp_path / "backtests" / "SHIP_GATE.json",
            )

    def test_invalid_max_lessons_rejected(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        with pytest.raises(ReflectionError, match="max_lessons"):
            ReflectionLedger(
                universe_hash=UNIVERSE_HASH,
                ledger_path=tmp_path / "reflection.json",
                ship_gate_path=gate,
                max_lessons=0,
            )


# ---------------------------------------------------------------------------
# Curated lessons + planning prompt
# ---------------------------------------------------------------------------


class TestCurationAndPrompt:
    def test_curated_lessons_sorted_by_abs_alpha(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        # Three matured decisions with controlled alphas.
        ledger.log_decision(_decision("d-small", ticker="SPY"))
        ledger.log_decision(_decision("d-big-neg", ticker="QQQ"))
        ledger.log_decision(_decision("d-medium", ticker="IWM"))
        ledger.resolve("d-small", 0.001, 0.0)      # |alpha|=0.001
        ledger.resolve("d-big-neg", -0.02, 0.001)  # |alpha|=0.021
        ledger.resolve("d-medium", 0.011, 0.001)   # |alpha|=0.010
        curated = ledger.curated_lessons(top_k=2)
        assert [le.decision_id for le in curated] == [
            "d-big-neg",
            "d-medium",
        ]

    def test_curated_lessons_ticker_filter(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        ledger.log_decision(_decision("d-spy", ticker="SPY"))
        ledger.log_decision(_decision("d-qqq", ticker="QQQ"))
        ledger.resolve("d-spy", 0.01, 0.0)
        ledger.resolve("d-qqq", -0.02, 0.0)
        only_spy = ledger.curated_lessons(top_k=5, ticker="SPY")
        assert [le.ticker for le in only_spy] == ["SPY"]

    def test_curated_lessons_top_k_validation(self, tmp_path: Path) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        with pytest.raises(ReflectionError, match="top_k"):
            ledger.curated_lessons(top_k=0)

    def test_render_planning_prompt_with_lessons(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        ledger.log_decision(_decision("d-1", ticker="SPY"))
        ledger.resolve("d-1", 0.012, 0.005)
        text = ledger.render_planning_prompt(top_k=3)
        assert "Curated trading lessons" in text
        assert "SPY" in text
        assert "alpha=+0.0070" in text

    def test_render_planning_prompt_empty_ledger(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        text = ledger.render_planning_prompt()
        assert "no matured lessons yet" in text


# ---------------------------------------------------------------------------
# Re-enforce: a gate invalidated mid-session fails-closed on next call
# ---------------------------------------------------------------------------


class TestMidSessionGateInvalidation:
    def test_log_decision_fails_when_gate_flipped(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        # Flip the gate to False on disk.
        _ship_gate(tmp_path, gate_pass=False)
        with pytest.raises(ReflectionError, match="gate_pass"):
            ledger.log_decision(_decision("d-1"))

    def test_resolve_fails_when_gate_flipped(
        self, tmp_path: Path
    ) -> None:
        gate = _ship_gate(tmp_path)
        ledger = ReflectionLedger(
            universe_hash=UNIVERSE_HASH,
            ledger_path=tmp_path / "reflection.json",
            ship_gate_path=gate,
        )
        ledger.log_decision(_decision("d-1"))
        _ship_gate(tmp_path, gate_pass=False)
        with pytest.raises(ReflectionError, match="gate_pass"):
            ledger.resolve("d-1", 0.01, 0.0)


# ---------------------------------------------------------------------------
# Module-level sanity: no LLM imports leak via dataclass attributes
# ---------------------------------------------------------------------------


def test_module_defaults_are_sane() -> None:
    assert DEFAULT_MAX_LESSONS > 0
    assert LEDGER_VERSION >= 1


def test_lesson_and_decision_are_frozen() -> None:
    decision = _decision("d-1")
    lesson = Lesson(
        decision_id="d-1",
        ticker="SPY",
        realized_return=0.01,
        benchmark_return=0.005,
        alpha=0.005,
        text="SPY: ...",
    )
    with pytest.raises(Exception):
        decision.ticker = "QQQ"  # type: ignore[misc]
    with pytest.raises(Exception):
        lesson.alpha = 0.0  # type: ignore[misc]
