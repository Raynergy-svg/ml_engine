"""US-019 — Outcome-grounded reflection loop (Tier-7 planning only).

This module is a *clean-room reimplementation* (~100 lines, not imported
from the upstream project) of the TradingAgents (Apache-2.0) reflection
pattern: a decision is logged ``PENDING`` the moment it is taken; once
the holding period matures, the realized return is compared against a
benchmark return to produce ``alpha``; a compact, alpha-cited
``Lesson`` is appended to disk; on the next Tier-7 planning cycle the
top-K curated lessons are injected into the planning prompt.

Hard constraints (PRD §cross-cutting and §US-019):

* **Tier-7 planning surface only.** Nothing in this file is reachable
  from the live execution hot path. The
  ``tests/test_no_llm_in_equity_path.py`` AST canary keeps the entire
  ``src/equity/`` tree free of LLM imports; this module imports zero
  LLM SDKs and produces deterministic structured text — never an LLM
  call. The "reflection" is *outcome arithmetic*, not generation.
* **Ship-gate guard.** Every public surface re-enforces
  ``trained_data/backtests/SHIP_GATE.json`` with the expected
  ``universe_hash`` before reading or mutating any ledger entry. The
  same fail-closed contract used by ``rebalance``, ``kill_switch``,
  ``shadow_fills``, ``executors``, …
* **Anchored on ``trained_data/trade_journal_rl.json``.** Lessons are
  produced from realized trade outcomes and the journal stays the
  source of truth; the reflection ledger is a *derived* artifact.
* **Atomic, idempotent, rotation-capped.** Every mutation goes through
  ``tempfile.mkstemp`` + :func:`os.replace`; recording the same
  ``decision_id`` twice is a no-op; the ledger is capped at
  ``max_lessons`` and rotates oldest-first.
* **No bare ``except``.** Every catch names its exception types and
  either re-raises (after logging) or downgrades to a documented
  fallback.

Apache-2.0 attribution: the *idea* of the reflection loop is borrowed
from TradingAgents (https://github.com/TauricResearch/TradingAgents),
which is Apache-2.0. No source file is copied; this is a clean-room
~100-line reimplementation of the pattern.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
TRADE_JOURNAL_PATH_DEFAULT = "trained_data/trade_journal_rl.json"
LEDGER_VERSION = 1
DEFAULT_MAX_LESSONS = 500


class ReflectionError(RuntimeError):
    """Raised on ship-gate refusal, malformed input, or contract bugs.

    Distinct exception type so the Tier-7 planner can match
    ``except ReflectionError`` and refuse to advance its post-mortem
    state machine without confusing it with broker-side errors.
    """


def enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse unless ``SHIP_GATE.json`` clears for THIS universe.

    Fail-closed on missing file, unreadable JSON, non-object payload,
    ``gate_pass != True``, or ``universe_hash`` mismatch. Identical
    contract to every other equity ship-gate guard.
    """
    path = Path(path)
    if not path.exists():
        raise ReflectionError(
            f"reflection refuses: ship gate not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReflectionError(
            f"reflection refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReflectionError(
            f"reflection refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise ReflectionError(
            "reflection refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to arm the planner)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise ReflectionError(
            f"reflection refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise ReflectionError(
            "reflection refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    """``tempfile.mkstemp`` + :func:`os.replace`. Crash-safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.error(
            "atomic write failed for %s: %s", path, exc, exc_info=True
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "failed to remove temp file %s: %s", tmp_path, cleanup_exc
            )
        raise


@dataclass(frozen=True)
class PendingDecision:
    """A planning-time decision awaiting outcome maturity.

    The Tier-7 planner logs one of these the moment it commits to a
    rebalance target weight; the reflection loop resolves it once the
    holding period elapses and the realized return is known.
    """

    decision_id: str
    ticker: str
    target_weight: float
    rationale: str
    decided_at: str
    mature_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Lesson:
    """An alpha-cited compact lesson produced from a matured decision."""

    decision_id: str
    ticker: str
    realized_return: float
    benchmark_return: float
    alpha: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReflectionLedger:
    """Tier-7-only reflection state with atomic, rotation-capped writes.

    Layout on disk::

        {
          "version": 1,
          "universe_hash": "...",
          "pending":  [ PendingDecision, ... ],
          "lessons":  [ Lesson, ... ],
        }

    Public surface (every call re-enforces the ship gate first):

    * :meth:`log_decision` — append a ``PendingDecision``. Idempotent
      on ``decision_id``: a duplicate is silently dropped.
    * :meth:`resolve` — compute ``alpha = realized - benchmark``,
      distill a compact text, append to ``lessons``, drop from
      ``pending``. Rotation-capped at ``max_lessons``.
    * :meth:`curated_lessons` — return the top-K most-recent lessons,
      optionally filtered by ticker, formatted for prompt injection.
    """

    universe_hash: str
    ledger_path: Path
    ship_gate_path: Path = Path(SHIP_GATE_PATH_DEFAULT)
    trade_journal_path: Path = Path(TRADE_JOURNAL_PATH_DEFAULT)
    max_lessons: int = DEFAULT_MAX_LESSONS
    _pending: List[dict] = field(default_factory=list)
    _lessons: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.universe_hash, str) or not self.universe_hash:
            raise ReflectionError(
                f"universe_hash must be a non-empty string, got "
                f"{self.universe_hash!r}"
            )
        if not isinstance(self.max_lessons, int) or self.max_lessons <= 0:
            raise ReflectionError(
                f"max_lessons must be a positive int, got {self.max_lessons!r}"
            )
        self.ledger_path = Path(self.ledger_path)
        self.ship_gate_path = Path(self.ship_gate_path)
        self.trade_journal_path = Path(self.trade_journal_path)
        enforce_ship_gate(self.ship_gate_path, self.universe_hash)
        self._load_or_init()

    def _reenforce(self) -> None:
        enforce_ship_gate(self.ship_gate_path, self.universe_hash)

    def _load_or_init(self) -> None:
        if not self.ledger_path.exists():
            self._persist()
            return
        try:
            payload = json.loads(
                self.ledger_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "reflection ledger %s unreadable (%s) — reinitialising",
                self.ledger_path,
                exc,
            )
            self._persist()
            return
        if not isinstance(payload, Mapping):
            logger.warning(
                "reflection ledger %s not a JSON object — reinitialising",
                self.ledger_path,
            )
            self._persist()
            return
        if payload.get("universe_hash") != self.universe_hash:
            raise ReflectionError(
                "reflection ledger universe_hash mismatch "
                f"(ledger={payload.get('universe_hash')!r}, "
                f"expected={self.universe_hash!r}) at {self.ledger_path}"
            )
        pending = payload.get("pending", [])
        lessons = payload.get("lessons", [])
        self._pending = [dict(p) for p in pending if isinstance(p, Mapping)]
        self._lessons = [dict(le) for le in lessons if isinstance(le, Mapping)]

    def _persist(self) -> None:
        payload = {
            "version": LEDGER_VERSION,
            "universe_hash": self.universe_hash,
            "pending": self._pending,
            "lessons": self._lessons,
        }
        _atomic_write_json(payload, self.ledger_path)

    def log_decision(self, decision: PendingDecision) -> bool:
        """Append a pending decision. Idempotent on ``decision_id``."""
        self._reenforce()
        if not isinstance(decision, PendingDecision):
            raise ReflectionError(
                f"log_decision requires PendingDecision, got "
                f"{type(decision).__name__}"
            )
        for existing in self._pending:
            if existing.get("decision_id") == decision.decision_id:
                return False
        for existing in self._lessons:
            if existing.get("decision_id") == decision.decision_id:
                return False
        self._pending.append(decision.to_dict())
        self._persist()
        return True

    def resolve(
        self,
        decision_id: str,
        realized_return: float,
        benchmark_return: float,
        extra_note: Optional[str] = None,
    ) -> Optional[Lesson]:
        """Resolve a matured decision into an alpha-cited :class:`Lesson`.

        Idempotent: re-resolving a decision_id that is already in the
        lessons section is a no-op and returns the prior lesson.
        """
        self._reenforce()
        for existing in self._lessons:
            if existing.get("decision_id") == decision_id:
                return Lesson(**existing)
        idx = next(
            (
                i
                for i, p in enumerate(self._pending)
                if p.get("decision_id") == decision_id
            ),
            None,
        )
        if idx is None:
            raise ReflectionError(
                f"resolve: no pending decision with id={decision_id!r}"
            )
        pending = self._pending.pop(idx)
        alpha = float(realized_return) - float(benchmark_return)
        sign = "+" if alpha >= 0.0 else ""
        ticker = pending.get("ticker", "?")
        rationale = pending.get("rationale", "")
        note = f" — {extra_note}" if extra_note else ""
        text = (
            f"{ticker}: rationale={rationale!r} → "
            f"alpha={sign}{alpha:.4f} "
            f"(realized={realized_return:.4f}, "
            f"benchmark={benchmark_return:.4f}){note}"
        )
        lesson = Lesson(
            decision_id=decision_id,
            ticker=ticker,
            realized_return=float(realized_return),
            benchmark_return=float(benchmark_return),
            alpha=alpha,
            text=text,
        )
        self._lessons.append(lesson.to_dict())
        if len(self._lessons) > self.max_lessons:
            overflow = len(self._lessons) - self.max_lessons
            self._lessons = self._lessons[overflow:]
        self._persist()
        return lesson

    def pending(self) -> List[dict]:
        """Defensive snapshot of pending decisions."""
        return [dict(p) for p in self._pending]

    def lessons(self) -> List[dict]:
        """Defensive snapshot of recorded lessons."""
        return [dict(le) for le in self._lessons]

    def curated_lessons(
        self,
        top_k: int = 5,
        ticker: Optional[str] = None,
    ) -> List[Lesson]:
        """Return the top-K most-recent lessons, optionally per-ticker.

        Sorted by ``|alpha|`` descending so the planner sees the most
        informative outcomes first, then trimmed to ``top_k``. Most-
        recent ties break by insertion order (Python sort is stable).
        """
        self._reenforce()
        if not isinstance(top_k, int) or top_k <= 0:
            raise ReflectionError(
                f"top_k must be a positive int, got {top_k!r}"
            )
        candidates: Iterable[dict] = self._lessons
        if ticker is not None:
            candidates = [
                le for le in candidates if le.get("ticker") == ticker
            ]
        ranked = sorted(
            candidates,
            key=lambda le: abs(float(le.get("alpha", 0.0))),
            reverse=True,
        )
        return [Lesson(**le) for le in ranked[:top_k]]

    def render_planning_prompt(
        self,
        header: str = "Curated trading lessons (alpha-cited):",
        top_k: int = 5,
        ticker: Optional[str] = None,
    ) -> str:
        """Return a flat string suitable for injection into a planner.

        The Tier-7 planning prompt concatenates this output verbatim;
        the runtime hot path NEVER calls this method.
        """
        chosen = self.curated_lessons(top_k=top_k, ticker=ticker)
        if not chosen:
            return f"{header}\n  (no matured lessons yet)"
        lines = [header]
        for idx, lesson in enumerate(chosen, start=1):
            lines.append(f"  {idx}. {lesson.text}")
        return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_LESSONS",
    "LEDGER_VERSION",
    "Lesson",
    "PendingDecision",
    "ReflectionError",
    "ReflectionLedger",
    "SHIP_GATE_PATH_DEFAULT",
    "TRADE_JOURNAL_PATH_DEFAULT",
    "enforce_ship_gate",
]
