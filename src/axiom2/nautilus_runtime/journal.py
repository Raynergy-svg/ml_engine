"""Durable SQLite journal for candidate state, material events, and wakeups."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sqlite3
from typing import Any

from .contracts import (
    CandidateObservation,
    CandidateState,
    MaterialEvent,
    ResearchResult,
    ResearchWakeup,
    TERMINAL_STATES,
    canonical_json,
    stable_digest,
    stable_id,
)
from .policy import observation_state_path, transition_allowed


class CandidateJournalError(RuntimeError):
    pass


class ObservationConflictError(CandidateJournalError):
    pass


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    candidate_id: str
    candidate_version: str
    state: CandidateState
    state_sequence: int
    updated_at_ns: int
    freshness_deadline_ns: int
    evidence_digest: str
    latest_observation_id: str


@dataclass(frozen=True, slots=True)
class DurableSnapshot:
    candidates: tuple[CandidateSnapshot, ...]
    pending_wakeups: tuple[ResearchWakeup, ...]
    now_ns: int


@dataclass(frozen=True, slots=True)
class ApplyResult:
    observation_id: str
    duplicate: bool
    state: CandidateState
    material_event: MaterialEvent | None


@dataclass(frozen=True, slots=True)
class WakeupResult:
    wakeup_id: str
    outcome: str
    duplicate: bool
    state: CandidateState | None


class CandidateJournal:
    """SQLite-backed durable state with transactionally coupled wakeups.

    A pending wakeup is persistent state, not an in-memory delivery flag.
    Duplicate filtering is durable and idempotent; this class intentionally
    makes no exactly-once delivery claim.
    """

    def __init__(self, path: str | Path, *, max_storage_bytes: int | None = None) -> None:
        self.path = Path(path)
        if max_storage_bytes is not None and (
            type(max_storage_bytes) is not int or not 262_144 <= max_storage_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError('journal limit must be 256KiB through 64MiB')
        self.max_storage_bytes = max_storage_bytes
        if max_storage_bytes is not None:
            # Reserve three quarters for WAL frames, SHM and transaction overhead.
            # Reject an existing oversized store before SQLite can mutate it.
            if any(p.exists() and p.stat().st_size > max_storage_bytes // 4
                   for p in self._storage_paths()):
                raise CandidateJournalError('existing journal exceeds storage allocation')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        if max_storage_bytes is not None:
            page_size = self.connection.execute('PRAGMA page_size').fetchone()[0]
            page_limit = (max_storage_bytes // 4 - 65_536) // page_size
            pages = self.connection.execute('PRAGMA page_count').fetchone()[0]
            if page_limit < pages or page_limit < 1:
                self.connection.close()
                raise CandidateJournalError('existing journal exceeds page allocation')
            self.connection.execute(f'PRAGMA max_page_count={page_limit}')
            self.connection.execute('PRAGMA cache_spill=OFF')
            self.connection.execute('PRAGMA wal_autocheckpoint=1')
            self.connection.execute('PRAGMA busy_timeout=0')
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._create_schema()
        self.check_storage()

    def _storage_paths(self):
        return tuple(self.path.with_name(self.path.name + suffix) for suffix in ('', '-wal', '-shm'))

    def storage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._storage_paths() if p.exists())

    def check_storage(self) -> None:
        """Checkpoint before writes; never prune the coupled journal history.

        With cache spill disabled, one transaction appends at most the capped
        database's dirty pages. Starting from a truncated WAL leaves ample room
        under the combined cap. A pinned reader denies further writes rather
        than allowing WAL accumulation. The cap is a mechanics resource bound,
        not protection against an owner changing SQLite or filesystem contents.
        """
        if self.max_storage_bytes is None:
            return
        if self.storage_bytes() > self.max_storage_bytes:
            raise CandidateJournalError('journal storage budget exhausted')
        if not self.connection.in_transaction:
            busy, _, _ = self.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            if busy:
                raise CandidateJournalError('journal checkpoint blocked; writes denied')
        if self.storage_bytes() > self.max_storage_bytes:
            raise CandidateJournalError('journal storage budget exhausted')

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS replay_clock (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                now_ns INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO replay_clock VALUES (1, 0);

            CREATE TABLE IF NOT EXISTS raw_observations (
                observation_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                candidate_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                observed_at_ns INTEGER NOT NULL,
                received_at_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_states (
                candidate_id TEXT NOT NULL,
                candidate_version TEXT NOT NULL,
                state TEXT NOT NULL,
                state_sequence INTEGER NOT NULL,
                updated_at_ns INTEGER NOT NULL,
                freshness_deadline_ns INTEGER NOT NULL,
                evidence_digest TEXT NOT NULL,
                latest_observation_id TEXT NOT NULL,
                PRIMARY KEY (candidate_id, candidate_version)
            );

            CREATE TABLE IF NOT EXISTS state_events (
                event_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                candidate_version TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                previous_state TEXT,
                new_state TEXT NOT NULL,
                cause_id TEXT NOT NULL,
                occurred_at_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(candidate_id, candidate_version, sequence)
            );

            CREATE TABLE IF NOT EXISTS material_events (
                event_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                candidate_version TEXT NOT NULL,
                source_observation_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                occurred_at_ns INTEGER NOT NULL,
                evidence_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_wakeups (
                wakeup_id TEXT PRIMARY KEY,
                material_event_id TEXT NOT NULL UNIQUE,
                candidate_id TEXT NOT NULL,
                candidate_version TEXT NOT NULL,
                created_at_ns INTEGER NOT NULL,
                bound_evidence_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                claimed_at_ns INTEGER,
                consumed_at_ns INTEGER,
                result_digest TEXT,
                outcome TEXT
            );
            """
        )

    @contextmanager
    def _transaction(self):
        self.check_storage()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def replay(self) -> DurableSnapshot:
        """Read coupled recovery state and check retained raw/event consistency.

        This detects accidental corruption; it is not a signed authority journal
        or protection against an attacker replacing an entire SQLite database.
        """
        with self._transaction():
            for raw in self.connection.execute("SELECT * FROM raw_observations"):
                if hashlib.sha256(raw["payload_json"].encode()).hexdigest() != raw["payload_digest"]:
                    raise CandidateJournalError("raw observation digest mismatch")
            candidates = tuple(self._snapshot_from_row(row) for row in
                self.connection.execute("SELECT * FROM candidate_states ORDER BY candidate_id, candidate_version"))
            for current in candidates:
                history = self.state_history(current.candidate_id, current.candidate_version)
                previous = None
                for index, event in enumerate(history):
                    new = CandidateState(event['new_state'])
                    if (event['sequence'] != index or event['previous_state'] != (previous.value if previous else None)
                            or not transition_allowed(previous, new)):
                        raise CandidateJournalError("candidate state history mismatch")
                    previous = new
                if not history or previous != current.state or len(history)-1 != current.state_sequence:
                    raise CandidateJournalError("candidate snapshot differs from history")
            pending = self.pending_wakeups()
            for wakeup in pending:
                material = self.connection.execute("SELECT * FROM material_events WHERE event_id=?", (wakeup.material_event_id,)).fetchone()
                if material is None or (material['candidate_id'], material['candidate_version'], material['evidence_digest']) != (wakeup.candidate_id, wakeup.candidate_version, wakeup.bound_evidence_digest):
                    raise CandidateJournalError("wakeup differs from material event")
            return DurableSnapshot(candidates, pending, self.now_ns)

    def advance_time(self, now_ns: int) -> None:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        with self._transaction():
            if now_ns < self.now_ns:
                raise CandidateJournalError("replay clock regressed")
            self.connection.execute("UPDATE replay_clock SET now_ns = ?", (now_ns,))
            rows = self.connection.execute("SELECT * FROM candidate_states").fetchall()
            for row in rows:
                current = self._snapshot_from_row(row)
                if current.state not in TERMINAL_STATES and now_ns >= current.freshness_deadline_ns:
                    self._terminate(current, CandidateState.EXPIRED, f"clock:{now_ns}", now_ns)

    @property
    def now_ns(self) -> int:
        return self.connection.execute("SELECT now_ns FROM replay_clock").fetchone()[0]

    def _terminate(self, current, state, cause, timestamp):
        updated = self._append_state_event(current, candidate_id=current.candidate_id,
            candidate_version=current.candidate_version, new_state=state,
            cause_id=cause, occurred_at_ns=timestamp, payload={"cause": cause})
        self.connection.execute("""UPDATE candidate_states SET state=?, state_sequence=?, updated_at_ns=?
            WHERE candidate_id=? AND candidate_version=?""",
            (state.value, updated.state_sequence, timestamp, current.candidate_id, current.candidate_version))

    def close(self) -> None:
        self.connection.close()

    def _snapshot_from_row(self, row: sqlite3.Row | None) -> CandidateSnapshot | None:
        if row is None:
            return None
        return CandidateSnapshot(
            candidate_id=row["candidate_id"],
            candidate_version=row["candidate_version"],
            state=CandidateState(row["state"]),
            state_sequence=row["state_sequence"],
            updated_at_ns=row["updated_at_ns"],
            freshness_deadline_ns=row["freshness_deadline_ns"],
            evidence_digest=row["evidence_digest"],
            latest_observation_id=row["latest_observation_id"],
        )

    def snapshot(self, candidate_id: str, candidate_version: str) -> CandidateSnapshot | None:
        row = self.connection.execute(
            """
            SELECT * FROM candidate_states
            WHERE candidate_id = ? AND candidate_version = ?
            """,
            (candidate_id, candidate_version),
        ).fetchone()
        return self._snapshot_from_row(row)

    def raw_observation(self, observation_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM raw_observations WHERE observation_id = ?",
            (observation_id,),
        ).fetchone()
        return dict(row) if row else None

    def state_history(
        self, candidate_id: str, candidate_version: str
    ) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM state_events
            WHERE candidate_id = ? AND candidate_version = ?
            ORDER BY sequence
            """,
            (candidate_id, candidate_version),
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def pending_wakeups(self) -> tuple[ResearchWakeup, ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM research_wakeups
            WHERE status IN ('PENDING', 'IN_PROGRESS')
            ORDER BY created_at_ns, wakeup_id
            """
        ).fetchall()
        return tuple(
            ResearchWakeup(
                wakeup_id=row["wakeup_id"],
                material_event_id=row["material_event_id"],
                candidate_id=row["candidate_id"],
                candidate_version=row["candidate_version"],
                created_at_ns=row["created_at_ns"],
                bound_evidence_digest=row["bound_evidence_digest"],
            )
            for row in rows
        )

    def _append_state_event(
        self,
        snapshot: CandidateSnapshot | None,
        *,
        candidate_id: str,
        candidate_version: str,
        new_state: CandidateState,
        cause_id: str,
        occurred_at_ns: int,
        payload: Any,
    ) -> CandidateSnapshot:
        previous_state = snapshot.state if snapshot else None
        if not transition_allowed(previous_state, new_state):
            raise CandidateJournalError(
                f"invalid transition {previous_state} -> {new_state}"
            )
        sequence = snapshot.state_sequence + 1 if snapshot else 0
        event_id = stable_id(
            "candidate-state",
            candidate_id,
            candidate_version,
            str(sequence),
            new_state.value,
            cause_id,
        )
        self.connection.execute(
            """
            INSERT INTO state_events (
                event_id, candidate_id, candidate_version, sequence,
                previous_state, new_state, cause_id, occurred_at_ns, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                candidate_id,
                candidate_version,
                sequence,
                previous_state.value if previous_state else None,
                new_state.value,
                cause_id,
                occurred_at_ns,
                canonical_json(payload),
            ),
        )
        return CandidateSnapshot(
            candidate_id=candidate_id,
            candidate_version=candidate_version,
            state=new_state,
            state_sequence=sequence,
            updated_at_ns=occurred_at_ns,
            freshness_deadline_ns=0,
            evidence_digest="",
            latest_observation_id="",
        )

    def append_observation(
        self,
        observation: CandidateObservation,
        *,
        state_path: tuple[CandidateState, ...],
        material_event: MaterialEvent | None,
        wakeup: ResearchWakeup | None,
    ) -> ApplyResult:
        payload = observation.to_payload()
        payload_json = canonical_json(payload)
        payload_digest = stable_digest(payload)
        with self._transaction():
            existing = self.connection.execute(
                "SELECT payload_digest FROM raw_observations WHERE observation_id = ?",
                (observation.observation_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_digest"] != payload_digest:
                    raise ObservationConflictError(
                        f"observation id conflict: {observation.observation_id}"
                    )
                current = self.snapshot(
                    observation.candidate_id, observation.candidate_version
                )
                event_row = self.connection.execute(
                    """
                    SELECT payload_json FROM material_events
                    WHERE source_observation_id = ?
                    """,
                    (observation.observation_id,),
                ).fetchone()
                event = (
                    self._material_from_payload(event_row["payload_json"])
                    if event_row
                    else None
                )
                if current is None:
                    raise CandidateJournalError("duplicate observation has no state")
                return ApplyResult(
                    observation_id=observation.observation_id,
                    duplicate=True,
                    state=current.state,
                    material_event=event,
                )

            current = self.snapshot(
                observation.candidate_id, observation.candidate_version
            )
            if current is not None and observation.observed_at_ns < current.updated_at_ns:
                raise CandidateJournalError("observation timestamp regressed")
            if current is None:
                older = self.connection.execute("SELECT * FROM candidate_states WHERE candidate_id=?", (observation.candidate_id,)).fetchall()
                if any(row["updated_at_ns"] > observation.observed_at_ns for row in older):
                    raise CandidateJournalError("candidate version timestamp regressed")
                for row in older:
                    old = self._snapshot_from_row(row)
                    if old.state not in TERMINAL_STATES:
                        self._terminate(old, CandidateState.INVALIDATED, "superseded:"+observation.candidate_version, observation.observed_at_ns)
            # Compute policy again under the write lock: another process may have advanced state.
            previous = current.state if current else None
            state_path = observation_state_path(previous, observation)
            if current and previous in {CandidateState.TRIGGERED, CandidateState.REVALIDATING, CandidateState.READY} and current.evidence_digest != observation.evidence_digest:
                state_path = (CandidateState.INVALIDATED,)
            if current and previous not in TERMINAL_STATES and max(self.now_ns, observation.observed_at_ns) >= current.freshness_deadline_ns:
                state_path = (CandidateState.EXPIRED,)
            if previous not in TERMINAL_STATES and self.now_ns >= observation.freshness_deadline_ns:
                state_path = (CandidateState.WATCHING, CandidateState.EXPIRED) if current is None else (CandidateState.EXPIRED,)
            if CandidateState.TRIGGERED not in state_path:
                material_event = wakeup = None
            self.connection.execute(
                """
                INSERT INTO raw_observations (
                    observation_id, candidate_id, candidate_version,
                    payload_json, payload_digest, observed_at_ns, received_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.observation_id,
                    observation.candidate_id,
                    observation.candidate_version,
                    payload_json,
                    payload_digest,
                    observation.observed_at_ns,
                    observation.observed_at_ns,
                ),
            )

            if current is None and not state_path:
                raise CandidateJournalError("new candidate requires a state path")
            for new_state in state_path:
                current = self._append_state_event(
                    current,
                    candidate_id=observation.candidate_id,
                    candidate_version=observation.candidate_version,
                    new_state=new_state,
                    cause_id=observation.observation_id,
                    occurred_at_ns=observation.observed_at_ns,
                    payload=payload,
                )
                current = CandidateSnapshot(
                    candidate_id=current.candidate_id,
                    candidate_version=current.candidate_version,
                    state=current.state,
                    state_sequence=current.state_sequence,
                    updated_at_ns=current.updated_at_ns,
                    freshness_deadline_ns=observation.freshness_deadline_ns,
                    evidence_digest=observation.evidence_digest,
                    latest_observation_id=observation.observation_id,
                )
                self.connection.execute(
                    """
                    INSERT INTO candidate_states (
                        candidate_id, candidate_version, state, state_sequence,
                        updated_at_ns, freshness_deadline_ns, evidence_digest,
                        latest_observation_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(candidate_id, candidate_version) DO UPDATE SET
                        state = excluded.state,
                        state_sequence = excluded.state_sequence,
                        updated_at_ns = excluded.updated_at_ns,
                        freshness_deadline_ns = excluded.freshness_deadline_ns,
                        evidence_digest = excluded.evidence_digest,
                        latest_observation_id = excluded.latest_observation_id
                    """,
                    (
                        current.candidate_id,
                        current.candidate_version,
                        current.state.value,
                        current.state_sequence,
                        current.updated_at_ns,
                        current.freshness_deadline_ns,
                        current.evidence_digest,
                        current.latest_observation_id,
                    ),
                )

            if current is not None and not state_path:
                deadline = observation.freshness_deadline_ns
                if current.state is CandidateState.READY:
                    deadline = min(deadline, current.freshness_deadline_ns)
                self.connection.execute(
                    """
                    UPDATE candidate_states
                    SET updated_at_ns = ?, freshness_deadline_ns = ?,
                        evidence_digest = ?, latest_observation_id = ?
                    WHERE candidate_id = ? AND candidate_version = ?
                    """,
                    (
                        observation.observed_at_ns,
                        deadline,
                        observation.evidence_digest,
                        observation.observation_id,
                        observation.candidate_id,
                        observation.candidate_version,
                    ),
                )

            if material_event is not None:
                if wakeup is None:
                    raise CandidateJournalError("material event requires wakeup")
                self.connection.execute(
                    """
                    INSERT INTO material_events (
                        event_id, candidate_id, candidate_version,
                        source_observation_id, kind, occurred_at_ns,
                        evidence_digest, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        material_event.event_id,
                        material_event.candidate_id,
                        material_event.candidate_version,
                        material_event.source_observation_id,
                        material_event.kind,
                        material_event.occurred_at_ns,
                        material_event.evidence_digest,
                        canonical_json(material_event.to_payload()),
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO research_wakeups (
                        wakeup_id, material_event_id, candidate_id,
                        candidate_version, created_at_ns, bound_evidence_digest,
                        status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING')
                    """,
                    (
                        wakeup.wakeup_id,
                        wakeup.material_event_id,
                        wakeup.candidate_id,
                        wakeup.candidate_version,
                        wakeup.created_at_ns,
                        wakeup.bound_evidence_digest,
                    ),
                )

            if current is None:
                raise CandidateJournalError("state path did not create state")
            return ApplyResult(
                observation_id=observation.observation_id,
                duplicate=False,
                state=current.state,
                material_event=material_event,
            )

    @staticmethod
    def _material_from_payload(payload_json: str) -> MaterialEvent:
        import json

        payload = json.loads(payload_json)
        return MaterialEvent(
            event_id=payload["event_id"],
            candidate_id=payload["candidate_id"],
            candidate_version=payload["candidate_version"],
            source_observation_id=payload["source_observation_id"],
            kind=payload["kind"],
            occurred_at_ns=payload["occurred_at_ns"],
            evidence_digest=payload["evidence_digest"],
        )

    def _wakeup_row(self, wakeup_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM research_wakeups WHERE wakeup_id = ?",
            (wakeup_id,),
        ).fetchone()
        if row is None:
            raise CandidateJournalError(f"unknown research wakeup: {wakeup_id}")
        return row

    def claim_research_wakeup(self, wakeup_id: str) -> WakeupResult:
        with self._transaction():
            row = self._wakeup_row(wakeup_id)
            current = self.snapshot(row["candidate_id"], row["candidate_version"])
            status = row["status"]
            if status != "PENDING":
                return WakeupResult(
                    wakeup_id=wakeup_id,
                    outcome=row["outcome"] or status,
                    duplicate=True,
                    state=current.state if current else None,
                )
            if current is None or current.state is not CandidateState.TRIGGERED:
                self.connection.execute(
                    """
                    UPDATE research_wakeups
                    SET status = 'REJECTED_STALE', consumed_at_ns = ?, outcome = 'STALE'
                    WHERE wakeup_id = ?
                    """,
                    (current.updated_at_ns if current else row["created_at_ns"], wakeup_id),
                )
                return WakeupResult(
                    wakeup_id=wakeup_id,
                    outcome="STALE",
                    duplicate=False,
                    state=current.state if current else None,
                )
            updated = self._append_state_event(
                current,
                candidate_id=current.candidate_id,
                candidate_version=current.candidate_version,
                new_state=CandidateState.REVALIDATING,
                cause_id=wakeup_id,
                occurred_at_ns=max(current.updated_at_ns, row["created_at_ns"]),
                payload={"wakeup_id": wakeup_id, "phase": "claim"},
            )
            updated = CandidateSnapshot(
                candidate_id=updated.candidate_id,
                candidate_version=updated.candidate_version,
                state=updated.state,
                state_sequence=updated.state_sequence,
                updated_at_ns=updated.updated_at_ns,
                freshness_deadline_ns=current.freshness_deadline_ns,
                evidence_digest=current.evidence_digest,
                latest_observation_id=current.latest_observation_id,
            )
            self.connection.execute(
                """
                UPDATE candidate_states
                SET state = ?, state_sequence = ?, updated_at_ns = ?
                WHERE candidate_id = ? AND candidate_version = ?
                """,
                (
                    updated.state.value,
                    updated.state_sequence,
                    updated.updated_at_ns,
                    updated.candidate_id,
                    updated.candidate_version,
                ),
            )
            self.connection.execute(
                """
                UPDATE research_wakeups
                SET status = 'IN_PROGRESS', claimed_at_ns = ?, outcome = 'CLAIMED'
                WHERE wakeup_id = ?
                """,
                (updated.updated_at_ns, wakeup_id),
            )
            return WakeupResult(
                wakeup_id=wakeup_id,
                outcome="CLAIMED",
                duplicate=False,
                state=updated.state,
            )

    def complete_research_wakeup(self, result: ResearchResult) -> WakeupResult:
        with self._transaction():
            row = self._wakeup_row(result.wakeup_id)
            current = self.snapshot(row["candidate_id"], row["candidate_version"])
            if (
                row["candidate_id"] != result.candidate_id
                or row["candidate_version"] != result.candidate_version
            ):
                raise CandidateJournalError("research result candidate version mismatch")
            if row["result_digest"] is not None and row["result_digest"] != stable_digest(result.to_payload()):
                raise ObservationConflictError("research result identity conflict")
            if row["status"] != "IN_PROGRESS":
                return WakeupResult(
                    wakeup_id=result.wakeup_id,
                    outcome=row["outcome"] or row["status"],
                    duplicate=True,
                    state=current.state if current else None,
                )
            if current is None or current.state is not CandidateState.REVALIDATING:
                self.connection.execute(
                    """
                    UPDATE research_wakeups
                    SET status = 'REJECTED_STALE', consumed_at_ns = ?, outcome = 'STALE',
                        result_digest = ?
                    WHERE wakeup_id = ?
                    """,
                    (
                        result.completed_at_ns,
                        stable_digest(result.to_payload()),
                        result.wakeup_id,
                    ),
                )
                return WakeupResult(
                    wakeup_id=result.wakeup_id,
                    outcome="STALE",
                    duplicate=False,
                    state=current.state if current else None,
                )

            if current.evidence_digest != row["bound_evidence_digest"]:
                raise CandidateJournalError("research result evidence changed")
            if current.updated_at_ns > result.completed_at_ns:
                raise CandidateJournalError("research result timestamp regressed")
            if result.invalidated or not result.qualifies:
                final_state = CandidateState.INVALIDATED
            elif (
                max(self.now_ns, result.completed_at_ns) >= result.fresh_until_ns
                or max(self.now_ns, result.completed_at_ns) >= current.freshness_deadline_ns
            ):
                final_state = CandidateState.EXPIRED
            else:
                final_state = CandidateState.READY

            updated = self._append_state_event(
                current,
                candidate_id=current.candidate_id,
                candidate_version=current.candidate_version,
                new_state=final_state,
                cause_id=result.wakeup_id,
                occurred_at_ns=result.completed_at_ns,
                payload=result.to_payload(),
            )
            self.connection.execute(
                """
                UPDATE candidate_states
                SET state = ?, state_sequence = ?, updated_at_ns = ?, freshness_deadline_ns = ?
                WHERE candidate_id = ? AND candidate_version = ?
                """,
                (
                    updated.state.value,
                    updated.state_sequence,
                    updated.updated_at_ns,
                    min(current.freshness_deadline_ns, result.fresh_until_ns),
                    updated.candidate_id,
                    updated.candidate_version,
                ),
            )
            self.connection.execute(
                """
                UPDATE research_wakeups
                SET status = 'CONSUMED', consumed_at_ns = ?, result_digest = ?,
                    outcome = ?
                WHERE wakeup_id = ?
                """,
                (
                    result.completed_at_ns,
                    stable_digest(result.to_payload()),
                    final_state.value,
                    result.wakeup_id,
                ),
            )
            return WakeupResult(
                wakeup_id=result.wakeup_id,
                outcome=final_state.value,
                duplicate=False,
                state=final_state,
            )

    def consume_research_wakeup(self, result: ResearchResult) -> WakeupResult:
        row = self._wakeup_row(result.wakeup_id)
        if row["candidate_id"] != result.candidate_id or row["candidate_version"] != result.candidate_version:
            raise CandidateJournalError("research result candidate version mismatch")
        if row["result_digest"] is not None and row["result_digest"] != stable_digest(result.to_payload()):
            raise ObservationConflictError("research result identity conflict")
        claimed = self.claim_research_wakeup(result.wakeup_id)
        if claimed.outcome != "CLAIMED":
            return claimed
        return self.complete_research_wakeup(result)
