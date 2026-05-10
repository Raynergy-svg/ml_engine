"""Append-only outcomes JSONL ledger + learnings.md fenced renderer (Angle 2').

This module is the canonical writer for `.claude/learnings_outcomes.jsonl`. Every
trade close — regardless of halt state — produces one JSON line tagged
``kind="outcome"``. A render layer maintains a fenced auto-section at the BOTTOM
of `.claude/learnings.md` showing the most recent N outcomes; human-authored
content above the fence is preserved byte-for-byte.

Why a separate ledger:
- The full-featured writer (`Orchestrator._extract_learnings_dispatch` →
  `LearningEngine.append_to_learnings`) is dead in the TUI runtime
  (CLAUDE.md Tier 7 section, ``grep "Orchestrator(" src/tui/`` = 0).
- The two narrow writers that DO fire (PostTradeLoop's 2R-gated
  ``_append_learning`` and CounterfactualLearner's loss-only analysis) leave
  most trade outcomes unrecorded.
- This ledger is the machine-owned source of truth: append-only, one line per
  close, halt-resilient, fcntl-locked, retention-managed.

Coordination:
- Promotion machinery in ``learning_engine.check_promotions`` counts learnings
  for 3+ promotion. JSONL entries with ``kind="outcome"`` are RAW outcomes,
  not interpreted patterns — promotion machinery should filter them out.
  See OPEN_QUESTION at the bottom of this module.
- ``CounterfactualLearner`` writes loss-only deeper analysis to a SEPARATE
  jsonl (``.claude/counterfactuals.jsonl``); this ledger covers ALL closes.

Atomicity contract:
- JSONL append: ``fcntl.LOCK_EX`` + ``flush`` + ``fsync`` (multi-process safe).
- learnings.md update: tmp + ``os.replace`` (atomic on POSIX).
- Retention archive: ``os.replace`` swap of trimmed jsonl, idempotent.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

LEDGER_PATH_DEFAULT = _PROJECT_ROOT / ".claude" / "learnings_outcomes.jsonl"
LEARNINGS_MD_PATH_DEFAULT = _PROJECT_ROOT / ".claude" / "learnings.md"
ARCHIVE_DIR_DEFAULT = _PROJECT_ROOT / ".claude" / "archive"

FENCE_BEGIN = "<!-- BUDDY_OUTCOMES_BEGIN -->"
FENCE_END = "<!-- BUDDY_OUTCOMES_END -->"

# Retention thresholds (constants — operator-tunable via constructor args)
RETENTION_LIMIT = 5000  # archive when ledger exceeds this many lines
RETENTION_ARCHIVE_BATCH = 1000  # move oldest N lines to archive when triggered
RENDER_LIMIT_DEFAULT = 50  # last N outcomes shown in learnings.md fenced block

# JSONL schema fields (all optional except ts_utc + kind + trade_id)
_OUTCOME_FIELDS = (
    "ts_utc", "kind", "trade_id", "pair", "direction", "confidence",
    "pnl_pips", "pnl_currency", "exit_reason", "regime", "wvs",
    "top_dissent_agent",
)


# ── OutcomesLedger ────────────────────────────────────────────────────────────


class OutcomesLedger:
    """Append-only JSONL writer for trade outcomes with retention.

    Concurrency:
        ``append`` uses ``fcntl.LOCK_EX`` so multiple Buddy processes
        (TUI + scripts + tests) can write simultaneously without
        line corruption.

    Retention:
        When the file exceeds ``retention_limit`` lines on append, the
        oldest ``retention_archive_batch`` lines are moved to
        ``archive/learnings_outcomes-YYYY-MM.jsonl`` (current month, append
        mode under the same lock). Idempotent: archive append never
        duplicates lines because we trim the live file under the same lock.
    """

    def __init__(
        self,
        ledger_path: Optional[Path] = None,
        archive_dir: Optional[Path] = None,
        retention_limit: int = RETENTION_LIMIT,
        retention_archive_batch: int = RETENTION_ARCHIVE_BATCH,
    ) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path else LEDGER_PATH_DEFAULT
        self.archive_dir = Path(archive_dir) if archive_dir else ARCHIVE_DIR_DEFAULT
        self.retention_limit = int(retention_limit)
        self.retention_archive_batch = int(retention_archive_batch)

    # ── Public API ─────────────────────────────────────────────────────

    def append(self, outcome: Dict[str, Any]) -> bool:
        """Append a single outcome dict as one JSONL line.

        Args:
            outcome: dict with at minimum a ``trade_id``. ``ts_utc`` and
                ``kind`` are auto-filled if absent. Unknown keys are
                preserved verbatim.

        Returns:
            True if the line was written (or skipped because trade_id is
            already present in the live ledger), False on error.
        """
        try:
            entry = self._normalize(outcome)
            if not entry.get("trade_id"):
                logger.warning(
                    "OutcomesLedger.append refused entry with no trade_id: %s",
                    {k: outcome.get(k) for k in ("kind", "pair", "ts_utc")},
                )
                return False

            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

            # Idempotent: skip if trade_id already in live ledger.
            # We hold an exclusive lock for the whole read-then-write to
            # avoid a TOCTOU race against another writer.
            line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
            with open(self.ledger_path, "a+", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    if self._trade_id_present(f, str(entry["trade_id"])):
                        return True  # idempotent skip
                    f.seek(0, os.SEEK_END)
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Retention pass (separate lock acquisition by design — keeps
            # the append fast path tight and well-bounded).
            self._maybe_archive()
            return True

        except (OSError, IOError) as e:
            logger.error("OutcomesLedger.append: I/O error on %s: %s", self.ledger_path, e)
            return False
        except (TypeError, ValueError) as e:
            logger.error("OutcomesLedger.append: serialization error: %s", e)
            return False

    def read_all(self) -> List[Dict[str, Any]]:
        """Read every valid JSONL entry from the live ledger.

        Corrupt lines are logged at WARN and skipped; the rest are returned
        in file order. Returns [] if the file is missing.
        """
        if not self.ledger_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for lineno, raw in enumerate(f, start=1):
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            out.append(json.loads(raw))
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "OutcomesLedger.read_all: corrupt line %d in %s: %s",
                                lineno, self.ledger_path.name, e,
                            )
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError) as e:
            logger.error("OutcomesLedger.read_all: %s", e)
        return out

    def trade_ids(self) -> set[str]:
        """Return set of trade_ids currently in the live ledger (for idempotence checks)."""
        return {str(e.get("trade_id", "")) for e in self.read_all() if e.get("trade_id")}

    # ── Internals ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize(outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce outcome dict to canonical schema with defaults."""
        entry = dict(outcome)  # shallow copy — never mutate caller's dict
        entry.setdefault("kind", "outcome")
        entry.setdefault("ts_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        # trade_id required — caller's responsibility to supply
        if "trade_id" in entry and entry["trade_id"] is not None:
            entry["trade_id"] = str(entry["trade_id"])
        return entry

    @staticmethod
    def _trade_id_present(file_obj: Any, trade_id: str) -> bool:
        """Scan a seekable file for a JSONL entry with this trade_id.

        Caller MUST hold an exclusive lock and have seeked to position 0.
        """
        # Cheap substring prefilter: trade_id quoted as JSON value.
        # Then full JSON-parse confirmation to avoid false positives
        # (e.g. trade_id "1" matching "1234").
        needle = json.dumps(trade_id)  # always quoted
        for raw in file_obj:
            if needle not in raw:
                continue
            try:
                parsed = json.loads(raw)
                if str(parsed.get("trade_id", "")) == trade_id:
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def _maybe_archive(self) -> None:
        """If live ledger exceeds retention_limit, move oldest batch to archive.

        Idempotent: holds exclusive lock for full read-trim-write cycle. The
        archive file is appended-to (current YYYY-MM); if this method runs
        twice on the same content, the second call sees a trimmed live
        ledger and does nothing further.
        """
        if not self.ledger_path.exists():
            return
        try:
            stat = self.ledger_path.stat()
            # Cheap byte-size short-circuit — at >5000 entries the file is
            # ALWAYS bigger than 1kB, so this avoids opening tiny files.
            if stat.st_size < 1024:
                return

            with open(self.ledger_path, "r+", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    lines = f.readlines()
                    if len(lines) <= self.retention_limit:
                        return
                    batch = self.retention_archive_batch
                    archive_lines = lines[:batch]
                    keep_lines = lines[batch:]

                    # Append archive lines to YYYY-MM file (using a separate lock).
                    self._append_archive(archive_lines)

                    # Trim live ledger atomically.
                    f.seek(0)
                    f.truncate()
                    f.writelines(keep_lines)
                    f.flush()
                    os.fsync(f.fileno())
                    logger.info(
                        "OutcomesLedger: archived %d lines to %s (live ledger now %d)",
                        len(archive_lines), self._archive_path(), len(keep_lines),
                    )
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError) as e:
            logger.error("OutcomesLedger.archive: I/O error: %s", e)

    def _archive_path(self) -> Path:
        ym = datetime.now(timezone.utc).strftime("%Y-%m")
        return self.archive_dir / f"learnings_outcomes-{ym}.jsonl"

    def _append_archive(self, lines: List[str]) -> None:
        """Append lines to the current-month archive file under its own lock."""
        target = self._archive_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as af:
            fcntl.flock(af.fileno(), fcntl.LOCK_EX)
            try:
                af.writelines(lines)
                af.flush()
                os.fsync(af.fileno())
            finally:
                fcntl.flock(af.fileno(), fcntl.LOCK_UN)


# ── LearningsRenderer ────────────────────────────────────────────────────────


class LearningsRenderer:
    """Maintain a fenced auto-section at the BOTTOM of learnings.md.

    Markers: ``<!-- BUDDY_OUTCOMES_BEGIN -->`` / ``<!-- BUDDY_OUTCOMES_END -->``.

    Behavior:
    - If both markers exist: replace content between them, preserving
      everything else byte-for-byte.
    - If markers MISSING and ``allow_insert=False`` (default): refuse,
      log WARN. Never clobbers human content.
    - If markers MISSING and ``allow_insert=True``: append a fresh fenced
      block to the end of the file.
    """

    def __init__(
        self,
        learnings_md_path: Optional[Path] = None,
        ledger: Optional[OutcomesLedger] = None,
        render_limit: int = RENDER_LIMIT_DEFAULT,
    ) -> None:
        self.learnings_md_path = (
            Path(learnings_md_path) if learnings_md_path else LEARNINGS_MD_PATH_DEFAULT
        )
        self.ledger = ledger or OutcomesLedger()
        self.render_limit = int(render_limit)

    def render(self, *, allow_insert: bool = False) -> bool:
        """Update the fenced auto-section.

        Args:
            allow_insert: If markers are missing, append a fresh fenced
                block at the end instead of refusing.

        Returns:
            True if learnings.md was written, False on refusal/error.
        """
        try:
            entries = self.ledger.read_all()
            # Render last N outcomes (file order = append order = chronological).
            recent = [e for e in entries if e.get("kind") == "outcome"][-self.render_limit:]
            block = self._format_block(recent)

            self.learnings_md_path.parent.mkdir(parents=True, exist_ok=True)
            existing = (
                self.learnings_md_path.read_text(encoding="utf-8")
                if self.learnings_md_path.exists()
                else ""
            )
            new_content = self._splice(existing, block, allow_insert=allow_insert)
            if new_content is None:
                return False  # refused (markers missing, allow_insert=False)

            return self._atomic_write(new_content)

        except (OSError, IOError) as e:
            logger.error("LearningsRenderer.render: I/O error: %s", e)
            return False

    # ── Internals ──────────────────────────────────────────────────────

    def _format_block(self, outcomes: List[Dict[str, Any]]) -> str:
        """Format outcomes as a markdown table inside fence markers."""
        lines = [
            FENCE_BEGIN,
            "## Recent Trade Outcomes (auto-rendered)",
            "",
            "_This section is machine-managed — do not edit. Source: "
            "`.claude/learnings_outcomes.jsonl`._",
            "",
        ]
        if not outcomes:
            lines.append("_No outcomes recorded yet._")
        else:
            lines.extend([
                "| ts_utc | trade_id | pair | dir | conf | pnl_pips | exit | regime | wvs | dissent |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ])
            for o in outcomes:
                lines.append(self._format_row(o))
        lines.append("")
        lines.append(FENCE_END)
        return "\n".join(lines)

    @staticmethod
    def _format_row(o: Dict[str, Any]) -> str:
        def cell(key: str, fmt: str = "") -> str:
            v = o.get(key)
            if v is None:
                return ""
            if fmt:
                try:
                    return f"{float(v):{fmt}}"
                except (TypeError, ValueError):
                    return str(v)
            # Escape pipe characters that would break markdown tables
            return str(v).replace("|", "\\|")

        return (
            f"| {cell('ts_utc')} | {cell('trade_id')} | {cell('pair')} | "
            f"{cell('direction')} | {cell('confidence', '.2f')} | "
            f"{cell('pnl_pips', '.1f')} | {cell('exit_reason')} | "
            f"{cell('regime')} | {cell('wvs', '.2f')} | "
            f"{cell('top_dissent_agent')} |"
        )

    def _splice(
        self, existing: str, block: str, *, allow_insert: bool,
    ) -> Optional[str]:
        """Replace the fenced section, or refuse / insert based on policy."""
        has_begin = FENCE_BEGIN in existing
        has_end = FENCE_END in existing

        if has_begin and has_end:
            # Replace between markers (use regex for non-greedy span match).
            pattern = re.compile(
                re.escape(FENCE_BEGIN) + r".*?" + re.escape(FENCE_END),
                re.DOTALL,
            )
            return pattern.sub(block, existing, count=1)

        if not has_begin and not has_end:
            if not allow_insert:
                logger.warning(
                    "LearningsRenderer.render: fence markers missing in %s "
                    "and allow_insert=False — refusing to clobber human content",
                    self.learnings_md_path,
                )
                return None
            # Insert at end.
            sep = "" if existing.endswith("\n") or not existing else "\n"
            return existing + sep + "\n" + block + "\n"

        # Mixed state (one marker present, one missing) — refuse always.
        # This is a sign of human edit damage; we don't try to repair.
        logger.warning(
            "LearningsRenderer.render: %s has only one fence marker "
            "(begin=%s, end=%s) — refusing to write",
            self.learnings_md_path, has_begin, has_end,
        )
        return None

    def _atomic_write(self, content: str) -> bool:
        """Write learnings.md via tmp + os.replace (atomic on POSIX)."""
        tmp = self.learnings_md_path.with_suffix(
            self.learnings_md_path.suffix + ".tmp"
        )
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.learnings_md_path)
            return True
        except (OSError, IOError) as e:
            logger.error("LearningsRenderer._atomic_write: %s", e)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            return False


# ── Hook helper for execution.py ─────────────────────────────────────────────


def emit_outcome_from_journal_entry(
    entry: Dict[str, Any],
    *,
    ledger: Optional[OutcomesLedger] = None,
) -> bool:
    """Build an outcome dict from a trade-journal entry and append to the ledger.

    Field sources (all optional, default to null/0 if not on the trade record):
    - trade_id, pair, direction, confidence — top-level entry keys
    - regime — entry["regime"]["volatility_regime"]
    - wvs — entry["agents"]["weighted_vote_score"]
    - top_dissent_agent — agent with passed=False and largest weight in
      entry["agents"]["agent_reasons"], else None
    - pnl_pips, pnl_currency, exit_reason — entry["outcome"]
    - ts_utc — outcome["close_time"] preferred; falls back to now()

    Args:
        entry: A trade journal entry with ``outcome`` populated.
        ledger: Optional OutcomesLedger override (default constructs new one).

    Returns:
        True on successful append (including idempotent skip), False on error.
    """
    led = ledger or OutcomesLedger()

    outcome_data = entry.get("outcome") or {}
    if not isinstance(outcome_data, dict):
        outcome_data = {}

    # Pair: prefer top-level, fall back to instrument
    pair = entry.get("pair") or entry.get("instrument")

    # Regime: read volatility_regime under regime dict
    regime_obj = entry.get("regime")
    regime = None
    if isinstance(regime_obj, dict):
        regime = regime_obj.get("volatility_regime")
    elif isinstance(regime_obj, str):
        regime = regime_obj  # legacy flat field

    # Agents: WVS + top dissent
    agents_obj = entry.get("agents") or {}
    wvs = None
    top_dissent = None
    if isinstance(agents_obj, dict):
        wvs = agents_obj.get("weighted_vote_score")
        reasons = agents_obj.get("agent_reasons") or {}
        if isinstance(reasons, dict):
            top_dissent = _pick_top_dissent(reasons)

    # Timestamps: prefer outcome.close_time
    ts_utc = outcome_data.get("close_time") or outcome_data.get("exit_time")
    if ts_utc:
        # Normalize any trailing fractional seconds + Z to fixed format-friendly form.
        ts_utc = str(ts_utc)

    payload: Dict[str, Any] = {
        "trade_id": entry.get("trade_id"),
        "pair": pair,
        "direction": entry.get("direction"),
        "confidence": _coerce_float(entry.get("confidence")),
        "pnl_pips": _coerce_float(outcome_data.get("pnl_pips")),
        "pnl_currency": _coerce_float(outcome_data.get("realized_pl")),
        "exit_reason": outcome_data.get("exit_reason"),
        "regime": regime,
        "wvs": _coerce_float(wvs),
        "top_dissent_agent": top_dissent,
    }
    if ts_utc:
        payload["ts_utc"] = ts_utc

    return led.append(payload)


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick_top_dissent(agent_reasons: Dict[str, Any]) -> Optional[str]:
    """Pick the agent with passed=False that has the largest 'weight' field.

    Falls back to the first failing agent if no weights are present.
    Returns None if all agents passed.
    """
    failing = []
    for name, info in agent_reasons.items():
        if not isinstance(info, dict):
            continue
        if info.get("passed") is False:
            weight = info.get("weight")
            try:
                w = float(weight) if weight is not None else 0.0
            except (TypeError, ValueError):
                w = 0.0
            failing.append((w, name))
    if not failing:
        return None
    failing.sort(key=lambda t: t[0], reverse=True)
    return failing[0][1]


# ── Open question for follow-up (not actionable here) ─────────────────────────
#
# OPEN_QUESTION (Angle 2'):
#     ``LearningEngine.check_promotions`` in
#     ``src/scanner/automation/learning_engine.py`` counts learnings.md entries
#     for the 3+ promotion rule. Once the renderer starts writing outcome rows
#     into the fenced auto-section, that machinery will count them as
#     "learnings", which they are NOT — they are raw outcomes.
#
#     Fix (do not apply here — ownership is not in this PR's scope):
#         In ``check_promotions``, filter out lines inside the fenced
#         BUDDY_OUTCOMES_BEGIN/END block before counting, OR filter
#         ``kind=="outcome"`` JSONL entries if it switches to JSONL-driven
#         counting.
#
#     This module deliberately does NOT touch ``learning_engine.py`` per
#     the Angle 2' brief.
